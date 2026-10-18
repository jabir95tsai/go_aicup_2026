"""Frozen full-dataset loss and optional index-aligned score publication."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from go_aicup.data.board_cache import _close_mmap
from go_aicup.tasks.strength.dataset import GlobalBatchSampler, PackedBoardDataset, make_loader
from go_aicup.tasks.strength.losses import distributed, ranking_loss_from_means
from go_aicup.tasks.strength.runtime import autocast_context, barrier

LOSS_DTYPE = np.dtype([("step", "<i8"), ("train_loss", "<f8"), ("valid_loss", "<f8")])


def on_rank_zero(function, rank):
    """Propagate rank-zero publication failures instead of stranding peers."""
    if not distributed():
        return function() if rank == 0 else None
    message = [None]
    result = None
    if rank == 0:
        try:
            result = function()
        except InterruptedError as exc:
            message[0] = ("interrupted", str(exc))
        except Exception as exc:
            message[0] = ("error", f"{type(exc).__name__}: {exc}")
    if distributed():
        dist.broadcast_object_list(message, src=0)
    if message[0]:
        kind, detail = message[0]
        if kind == "interrupted":
            raise InterruptedError(detail)
        raise RuntimeError(detail)
    return result


def write_losses(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(rows, dtype=LOSS_DTYPE), allow_pickle=False)
    os.replace(temporary, path)


@torch.no_grad()
def infer_split(model, config, manifest, prefix, device, rank=0, world=1,
                *, output=None, stop_check=None, sample_indices=None):
    """Each selected sample exactly once; loss from FP64 rank sums.

    sample_indices optionally selects a balanced subset instead of the full split.

    No per-batch rank completeness requirement. Optional scores are written
    to disjoint shared memmap slices and published only after all ranks close.
    """
    directory = Path(config["paths"]["dataset_dir"])
    games, plies = (directory / f"{prefix}_{kind}.npy" for kind in ("game", "ply"))
    if sample_indices is not None:
        if output is not None:
            raise ValueError("sampled inference cannot publish full-split scores")
        games, plies = (np.load(path, mmap_mode="r", allow_pickle=False)[sample_indices]
                        for path in (games, plies))
    dataset = PackedBoardDataset(games, plies,
                                Path(config["paths"]["cache_dir"]),
                                max_open_shards=config["loader"]["max_open_shards"],
                                augment=False, seed=config["seed"])
    sampler = GlobalBatchSampler(len(dataset), config["evaluation"]["global_batch_size"],
                                 rank=rank, world_size=world)
    options = {key: config["loader"][key] for key in
               ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
    loader = make_loader(dataset, sampler, **options)
    score_store = None
    temporary = output.with_name(output.name + ".tmp") if output is not None else None
    if temporary is not None:
        def initialize():
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(dataset),))
            array[:] = np.nan
            array.flush()
            _close_mmap(array)
        on_rank_zero(initialize, rank)
        score_store = np.load(temporary, mmap_mode="r+", allow_pickle=False)
    sums = torch.zeros(10, dtype=torch.float64, device=device)
    counts = torch.zeros(10, dtype=torch.int64, device=device)
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(loader):
            stopping = torch.tensor(int(stop_check is not None and stop_check()), device=device)
            if distributed():
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                raise InterruptedError("full-dataset inference interrupted")
            valid = batch["valid"].bool()
            if valid.any():
                x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                labels = batch["rank_id"][valid].to(device)
                if config["training"]["channels_last"]:
                    x = x.contiguous(memory_format=torch.channels_last)
                with autocast_context(config, device):
                    scores = model(x).float()
                bad = int(not torch.isfinite(scores).all().item())
            else:
                scores = torch.empty(0, device=device)
                labels = torch.empty(0, dtype=torch.int64, device=device)
                bad = 0
            bad_tensor = torch.tensor(bad, device=device)
            if distributed():
                dist.all_reduce(bad_tensor, op=dist.ReduceOp.MAX)
            if bad_tensor.item():
                raise FloatingPointError("nonfinite inference scores")
            # CPU reductions avoid CUDA scatter_add nondeterminism under the
            # production deterministic-algorithms setting.
            values = scores.cpu().numpy()
            ranks = labels.cpu().numpy()
            sums += torch.from_numpy(np.bincount(ranks, weights=values, minlength=10)).to(device)
            counts += torch.from_numpy(np.bincount(ranks, minlength=10)).to(device)
            if score_store is not None:
                score_store[batch["sample_index"][valid].numpy()] = values
            if rank == 0 and (batch_index + 1) % 100 == 0:
                print(f"full inference {prefix}: {batch_index + 1}/{len(sampler)}", flush=True)
        if score_store is not None:
            score_store.flush()
    finally:
        # Destroy workers before closing the parent's cache/index handles.
        del loader
        dataset.close()
        if score_store is not None:
            _close_mmap(score_store)
        model.train(was_training)
    if distributed():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    expected = manifest["sets"][prefix]
    expected_counts = (expected["per_rank_samples"] if sample_indices is None else [len(sample_indices) // 10] * 10)
    if counts.cpu().tolist() != expected_counts:
        raise ValueError(f"{prefix}: incomplete sample/rank coverage")
    if (counts == 0).any():
        raise ValueError(f"{prefix}: full split missing ranks")
    loss = float(ranking_loss_from_means((sums / counts).float()).item())
    if output is not None:
        barrier()
        on_rank_zero(lambda: os.replace(temporary, output), rank)
    return loss
