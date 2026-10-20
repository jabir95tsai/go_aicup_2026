"""Frozen whole-game V2 inference: per-game softmax and log-softmax sums.

Each selected sample is evaluated exactly once across processes (eval mode,
no D4). A game's moves may span batches and processes; sums are accumulated
in FP64 on the CPU (np.add.at is deterministic, unlike CUDA scatter_add) and
all-reduced at the end, so every process returns the identical result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from go_aicup.data.board_cache import _close_mmap
from go_aicup.tasks.strength.shared.dataset import GlobalBatchSampler, PackedBoardDataset, make_loader
from go_aicup.tasks.strength.shared.runtime import autocast_context, distributed
from go_aicup.tasks.strength.v2.aggregate import N_RANKS, build_category


def _reduce(array: np.ndarray, device, op=None) -> np.ndarray:
    if not distributed():
        return array
    tensor = torch.from_numpy(array).to(device)
    dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
    return tensor.cpu().numpy()


@torch.no_grad()
def infer_games(model, config: dict, prefix: str, device, rank: int = 0, world: int = 1,
                *, stop_check=None) -> np.ndarray:
    """Return the CATEGORY_DTYPE result for every game in {prefix}_game.npy."""
    directory = Path(config["paths"]["dataset_dir"])
    games_path, plies_path = (directory / f"{prefix}_{kind}.npy" for kind in ("game", "ply"))
    sample_games = np.load(games_path, mmap_mode="r", allow_pickle=False)
    try:
        game_index, expected_moves = np.unique(np.asarray(sample_games), return_counts=True)
    finally:
        _close_mmap(sample_games)
    dataset = PackedBoardDataset(games_path, plies_path, Path(config["paths"]["cache_dir"]),
                                 max_open_shards=config["loader"]["max_open_shards"],
                                 augment=False, seed=config["seed"])
    sampler = GlobalBatchSampler(len(dataset), config["evaluation"]["global_batch_size"],
                                 rank=rank, world_size=world)
    options = {key: config["loader"][key] for key in
               ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
    loader = make_loader(dataset, sampler, **options)
    confidence = np.zeros((len(game_index), N_RANKS), dtype=np.float64)
    log_probability = np.zeros_like(confidence)
    moves = np.zeros(len(game_index), dtype=np.int64)
    labels = np.full(len(game_index), -1, dtype=np.int64)
    was_training = model.training
    model.eval()
    try:
        for batch_index, batch in enumerate(loader):
            stopping = torch.tensor(int(stop_check is not None and stop_check()), device=device)
            if distributed():
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                raise InterruptedError(f"{prefix} inference interrupted")
            valid = batch["valid"].bool()
            bad = 0
            if valid.any():
                x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                if config["training"]["channels_last"]:
                    x = x.contiguous(memory_format=torch.channels_last)
                with autocast_context(config, device):
                    logits = model(x)
                log_p = torch.log_softmax(logits.float(), dim=-1)
                bad = int(not torch.isfinite(log_p).all().item())
            bad_tensor = torch.tensor(bad, device=device)
            if distributed():
                dist.all_reduce(bad_tensor, op=dist.ReduceOp.MAX)
            if bad_tensor.item():
                raise FloatingPointError(f"{prefix}: nonfinite inference logits")
            if valid.any():
                rows = np.searchsorted(game_index, batch["game_index"][valid].numpy())
                values = log_p.double().cpu().numpy()
                np.add.at(log_probability, rows, values)
                np.add.at(confidence, rows, np.exp(values))
                np.add.at(moves, rows, 1)
                labels[rows] = batch["rank_id"][valid].numpy()
            if rank == 0 and (batch_index + 1) % 100 == 0:
                print(f"game inference {prefix}: {batch_index + 1}/{len(sampler)}", flush=True)
    finally:
        # Destroy workers before closing the parent's cache/index handles.
        del loader
        dataset.close()
        model.train(was_training)
    confidence, log_probability, moves = (_reduce(value, device)
                                          for value in (confidence, log_probability, moves))
    labels = _reduce(labels, device, dist.ReduceOp.MAX if distributed() else None)
    if not np.array_equal(moves, expected_moves):
        raise ValueError(f"{prefix}: incomplete per-game move coverage")
    return build_category(game_index, labels, moves, confidence, log_probability)
