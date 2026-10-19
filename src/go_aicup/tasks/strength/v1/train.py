"""Strength V1 direct-forward training with global rank-mean loss.

Production: torchrun, single-node 1-8 CUDA GPUs, SyncBN and BF16. There is
one forward/backward/update per global batch: no accumulation or activation
recomputation. Dataset preparation is a separate explicit command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from go_aicup.tasks.strength.v1.model import StrengthResNet
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v1.config import DEFAULT_CONFIG, read_config
from go_aicup.tasks.strength.shared.training import (seed_process, configure_torch,
    environment, rng_state, restore_rng, scheduler_for, log_event)
from go_aicup.tasks.strength.shared.dataset import (GlobalBatchSampler, PackedBoardDataset,
                                            create_permutation, make_loader)
from go_aicup.tasks.strength.v1.losses import global_rank_loss
from go_aicup.tasks.strength.v1.infer import infer_split, write_losses
from go_aicup.tasks.strength.shared.runtime import (autocast_context, barrier, distributed, on_rank_zero,
    check_index_hashes, file_hash, load_dataset_manifest, model_state_hash, object_hash,
    setup_distributed, write_json)


def code_fingerprints() -> dict:
    names = ("artifacts.py", "tasks/strength/shared/model.py",
             "tasks/strength/shared/dataset.py", "tasks/strength/shared/runtime.py",
             "tasks/strength/shared/training.py", "tasks/strength/shared/prepare.py",
             "tasks/strength/v1/model.py", "tasks/strength/v1/config.py",
             "tasks/strength/v1/losses.py", "tasks/strength/v1/train.py",
             "tasks/strength/v1/evaluate.py", "tasks/strength/v1/infer.py",
             "tasks/strength/v1/prepare.py", "data/board_cache.py", "data/cv.py")
    package = ROOT / "src" / "go_aicup"
    return {name: file_hash(package / name) for name in names}


def save_checkpoint(directory: Path, model, optimizer, scheduler, scaler,
                    config: dict, manifest: dict, state: dict, device,
                    rank: int, world: int, *, tag: str = "periodic") -> Path:
    local_rng = rng_state(device, model)
    all_rng = [None] * world
    if distributed():
        dist.all_gather_object(all_rng, local_rng)
    else:
        all_rng[0] = local_rng
    name_holder = [None]
    if rank == 0:
        name_holder[0] = (f"update_{state['optimizer_steps']:08d}_epoch_{state['epoch']:04d}"
                          f"_batch_{state['next_batch']:06d}_{time.time_ns()}.pt")
    if distributed():
        dist.broadcast_object_list(name_holder, src=0)
    path = directory / "checkpoints" / name_holder[0]
    def publish():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                   "precision": config["precision"], "config": config,
                   "config_sha256": object_hash(config), "dataset_id": manifest["dataset_id"],
                   "cache_build_id": manifest["cache_build_id"], "world_size": world,
                   "code_sha256": code_fingerprints(), "state": dict(state),
                   "rng_by_rank": all_rng, "tag": tag,
                   "saved_utc": datetime.now(timezone.utc).isoformat()}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        record = {"file": path.name, "sha256": file_hash(path), "tag": tag,
                  "epoch": state["epoch"], "next_batch": state["next_batch"],
                  "optimizer_steps": state["optimizer_steps"]}
        write_json(path.with_suffix(".complete.json"), record)
        write_json(directory / "last.json", record)
        # Publish the replacement completely before removing previous weights.
        prune_checkpoints(directory, path)
    on_rank_zero(publish, rank)
    barrier()
    return path


def prune_checkpoints(directory: Path, latest: Path) -> None:
    """Keep only the latest trainer checkpoint and its completion marker."""
    for old in latest.parent.glob("update_*.pt"):
        if old != latest:
            old.unlink()
            old.with_suffix(".complete.json").unlink(missing_ok=True)
    # Historical best weights are not retained under latest-only retention.
    (directory / "best.json").unlink(missing_ok=True)


def resolve_checkpoint(path: Path) -> Path:
    if path.suffix == ".json":
        record = json.loads(path.read_text(encoding="utf-8"))
        if Path(record["file"]).name != record["file"]:
            raise ValueError("unsafe checkpoint reference")
        target = path.parent / "checkpoints" / record["file"]
        if file_hash(target) != record["sha256"]:
            raise ValueError("checkpoint hash mismatch")
        return target
    marker = path.with_suffix(".complete.json")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if file_hash(path) != record["sha256"]:
        raise ValueError("checkpoint hash mismatch")
    return path


def finalize_run(directory, model, config, manifest, state, device, world, started, stop_check):
    """Rank zero only: verify artifacts and publish compact model before cleanup."""
    from go_aicup.tasks.strength.v1.evaluate import evaluate_saved
    payload = {"artifact_format": "strength_v1_compact_1", "seed": config["seed"],
               "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
               "config": config, "dataset_id": manifest["dataset_id"],
               "cache_build_id": manifest["cache_build_id"], "index_files": manifest["files"],
               "optimizer_steps": state["optimizer_steps"], "epochs": state["epoch"],
               "model_state_sha256": model_state_hash(model), "code_sha256": code_fingerprints(),
               "environment": json.loads((directory / "environment.json").read_text()),
               "gpu_seconds": state["gpu_seconds"],
               "artifact_sha256": {name: file_hash(directory / name) for name in
                                   ("losses.npy", "train_scores.npy", "valid_scores.npy")}}
    reports = evaluate_saved(directory, payload=payload, fit=True, stop_check=stop_check)
    print(json.dumps(reports, allow_nan=False), flush=True)
    # Reports are cheap to recompute from the four artifacts; retain cutpoints.
    path = directory / "model.pt"
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    verifier = StrengthResNet(**config["model"])
    verifier.load_state_dict(loaded["model"])
    if model_state_hash(verifier) != payload["model_state_sha256"]:
        raise ValueError("final weights failed reload verification")
    os.replace(temporary, path)
    cleanup_run(directory)
    return path


def cleanup_run(directory):
    """Remove only trainer-owned transient files after model publication."""
    for name in ("checkpoints", "permutations"):
        target = directory / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("config.json", "environment.json", "last.json", "train.jsonl",
                 "train_scores.npy.tmp", "valid_scores.npy.tmp", "losses.npy.tmp", "model.pt.tmp"):
        (directory / name).unlink(missing_ok=True)

VALIDATION_SAMPLES_PER_RANK = 10000


def validation_sample_indices(config: dict) -> np.ndarray:
    """Fixed, seed-specific validation boards sampled without replacement."""
    directory = Path(config["paths"]["dataset_dir"])
    rank_ids = pd.read_parquet(directory / "strength_v1_games.parquet",
                               columns=["rank_id"])["rank_id"].to_numpy()
    games = np.load(directory / "valid_game.npy", mmap_mode="r", allow_pickle=False)
    labels = rank_ids[games]
    rng = np.random.default_rng(config["seed"])
    selected = []
    for rank_id in range(10):
        candidates = np.flatnonzero(labels == rank_id)
        if len(candidates) < VALIDATION_SAMPLES_PER_RANK:
            raise ValueError(f"validation rank {rank_id} needs at least {VALIDATION_SAMPLES_PER_RANK} boards")
        selected.append(rng.choice(candidates, VALIDATION_SAMPLES_PER_RANK, replace=False))
    return np.sort(np.concatenate(selected))


def train(config: dict, *, resume: Path | None = None,
          max_steps: int | None = None, skip_evaluation: bool = False) -> Path:
    if config["training"]["epochs"] != 1:
        raise ValueError("compact training requires exactly one epoch")
    configure_torch(config)
    manifest = load_dataset_manifest(config)
    device, rank, world = setup_distributed(config)
    directory = Path(config["paths"]["output_dir"])
    stopped = [False]
    previous_handlers = {}
    def request_stop(signum, frame):
        stopped[0] = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    started = time.monotonic()
    checkpoint_time = started
    dataset = None
    try:
        def initialize():
            check_index_hashes(config, manifest)
            if resume is None:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("run directory is nonempty; use --resume or a new output_dir")
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "config.json", config)
                write_json(directory / "environment.json", environment(device, world))
            elif json.loads((directory / "config.json").read_text()) != config:
                raise ValueError("resume output directory config mismatch")
        on_rank_zero(initialize, rank)
        seed_process(config["seed"])
        raw_model = StrengthResNet(**config["model"])
        if world > 1 and config["training"]["sync_batch_norm"]:
            raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        raw_model = raw_model.to(device)
        if config["training"]["channels_last"]:
            raw_model.to(memory_format=torch.channels_last)
        model = (DistributedDataParallel(raw_model, device_ids=[device.index] if device.type == "cuda" else None,
                                        broadcast_buffers=True) if world > 1 else raw_model)
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
        batches_per_epoch = math.ceil(manifest["sets"]["train"]["num_samples"] /
                                      config["training"]["global_batch_size"])
        scheduler = scheduler_for(optimizer, batches_per_epoch, config["training"]["warmup_fraction"],
                                  config["training"]["minimum_lr_ratio"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16")
        state = {"epoch": 0, "next_batch": 0, "optimizer_steps": 0, "skipped_batches": 0,
                 "samples_seen": 0, "permutation": None, "permutation_sha256": None,
                 "gpu_seconds": 0.0, "loss_rows": [], "pending_evaluation": False,
                 "last_batch_loss": None}
        seed_process(config["seed"] + rank)
        source = None
        if resume is not None:
            source = resolve_checkpoint(resume.resolve())
            saved = torch.load(source, map_location="cpu", weights_only=False)
            if (saved.get("format_version") != 1 or saved["config_sha256"] != object_hash(config)
                    or saved["dataset_id"] != manifest["dataset_id"] or saved["world_size"] != world):
                raise ValueError("resume requires the same config, dataset and world_size")
            raw_model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            if scaler.is_enabled():
                scaler.load_state_dict(saved["scaler"])
            state.update(saved["state"])
            if "last_batch_loss" not in saved["state"]:
                # Old checkpoints logged full-split losses and may be waiting
                # for that evaluation. Resume updates without mixing metrics.
                legacy_path = directory.parent / f"{directory.name}.legacy_full_losses.json"
                on_rank_zero(lambda: write_json(legacy_path, {
                    "checkpoint": str(source), "loss_rows": state["loss_rows"],
                    "optimizer_steps": state["optimizer_steps"],
                    "columns": ["step", "full_train_loss", "full_valid_loss"],
                }), rank)
                state["loss_rows"] = []
                state["pending_evaluation"] = False
                log_event(directory, rank, {"event": "resume_loss_logging_migrated",
                                           "optimizer_steps": state["optimizer_steps"],
                                           "legacy_losses": str(legacy_path)})
            if state["permutation"] is not None and file_hash(Path(state["permutation"])) != state["permutation_sha256"]:
                raise ValueError("saved epoch permutation is missing or changed")
            restore_rng(saved["rng_by_rank"][rank], device, raw_model)
        previous_gpu_seconds = state["gpu_seconds"]
        def save(tag):
            state["gpu_seconds"] = previous_gpu_seconds + (time.monotonic() - started) * world * int(device.type == "cuda")
            return save_checkpoint(directory, raw_model, optimizer, scheduler, scaler,
                                   config, manifest, state, device, rank, world, tag=tag)
        def stop_check():
            deadline = config["training"]["max_runtime_seconds"]
            return stopped[0] or (deadline is not None and time.monotonic() - started >=
                                  max(0, deadline - config["training"]["shutdown_margin_seconds"]))
        validation_indices = None
        def record_evaluation():
            nonlocal validation_indices
            # Evaluation must not advance training RNG or alter local BN buffers.
            snapshot = rng_state(device, raw_model)
            try:
                if distributed():
                    for buffer in raw_model.buffers():
                        dist.broadcast(buffer, src=0)
                if validation_indices is None:
                    validation_indices = validation_sample_indices(config)
                valid_loss = infer_split(raw_model, config, manifest, "valid", device, rank, world,
                                         sample_indices=validation_indices, stop_check=stop_check)
                values = [state["last_batch_loss"], valid_loss]
                step = state["optimizer_steps"]
                state["loss_rows"] = [row for row in state["loss_rows"] if row[0] < step]
                state["loss_rows"].append((step, *values))
                on_rank_zero(lambda: write_losses(directory / "losses.npy", state["loss_rows"]), rank)
                state["pending_evaluation"] = False
                log_event(directory, rank, {"event": "batch_validation_loss", "optimizer_steps": step,
                                           "train_loss": values[0], "valid_loss": values[1],
                                           "valid_samples": len(validation_indices),
                                           "valid_samples_per_rank": VALIDATION_SAMPLES_PER_RANK})
            finally:
                restore_rng(snapshot, device, raw_model)
        # Rewrite the log from checkpoint-owned rows; discard abandoned branches.
        on_rank_zero(lambda: write_losses(directory / "losses.npy", state["loss_rows"]), rank)
        data_directory = Path(config["paths"]["dataset_dir"])
        dataset = PackedBoardDataset(data_directory / "train_game.npy", data_directory / "train_ply.npy",
                                    Path(config["paths"]["cache_dir"]),
                                    max_open_shards=config["loader"]["max_open_shards"],
                                    augment=config["loader"]["d4"], seed=config["seed"])
        limit = max_steps if max_steps is not None else config["training"]["max_steps"]
        if limit is not None and limit < 1:
            raise ValueError("max_steps must be positive")
        last_path = source
        try:
            if state["pending_evaluation"] and state["next_batch"] < batches_per_epoch and not skip_evaluation:
                record_evaluation()
                last_path = save("evaluated")
            if state["epoch"] == 0:
                on_rank_zero(lambda: create_permutation(directory / "permutations", len(dataset), config["seed"], 0), rank)
                permutation = directory / "permutations" / "epoch_0000.npy"
                state["permutation"] = str(permutation)
                state["permutation_sha256"] = file_hash(permutation)
                sampler = GlobalBatchSampler(len(dataset), config["training"]["global_batch_size"],
                                             rank=rank, world_size=world, permutation=permutation,
                                             start_batch=state["next_batch"])
                options = {key: config["loader"][key] for key in
                           ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
                loader = make_loader(dataset, sampler, **options)
                model.train()
                for step, batch in enumerate(loader, start=state["next_batch"]):
                    flag = torch.tensor(int(stop_check() or (limit is not None and state["optimizer_steps"] >= limit)), device=device)
                    if distributed():
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    if flag.item():
                        return save("stopped")
                    valid = batch["valid"].bool()
                    labels = batch["rank_id"][valid].to(device, non_blocking=True)
                    counts = torch.bincount(labels, minlength=10).float()
                    if distributed():
                        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    if (counts == 0).any():
                        state["next_batch"] = step + 1
                        state["skipped_batches"] += 1
                        continue
                    x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                    if config["training"]["channels_last"]:
                        x = x.contiguous(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(config, device):
                        scores = model(x)
                    loss, means, loss_counts = global_rank_loss(scores, labels)
                    scaler.scale(loss).backward()
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                    bad = (~torch.isfinite(norm)).to(torch.int32)
                    if distributed():
                        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
                    if bad.item():
                        raise FloatingPointError("nonfinite gradient; no optimizer update")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    state["last_batch_loss"] = float(loss.detach().item())
                    state["optimizer_steps"] += 1
                    state["next_batch"] = step + 1
                    state["samples_seen"] += int(loss_counts.sum().item())
                    interval = config["training"].get("evaluate_every_steps", 100)
                    if not skip_evaluation and state["optimizer_steps"] % interval == 0 and step + 1 < batches_per_epoch:
                        state["pending_evaluation"] = True
                        last_path = save("before_validation_evaluation")
                        record_evaluation()
                        last_path = save("evaluated")
                        checkpoint_time = time.monotonic()
                    due = torch.tensor(int(rank == 0 and time.monotonic() - checkpoint_time >=
                                           config["training"]["checkpoint_seconds"]), device=device)
                    if distributed():
                        dist.broadcast(due, src=0)
                    if due.item():
                        last_path = save("periodic")
                        checkpoint_time = time.monotonic()
                del loader
                state["epoch"] = 1
            state["pending_evaluation"] = not skip_evaluation
            last_path = save("training_complete")
            if skip_evaluation:
                return last_path  # Explicit development run, retain resumable state.
            if state["pending_evaluation"]:
                record_evaluation()
            # Full scores are still required for final calibration/evaluation;
            # their full-split losses are not added to the training loss log.
            for prefix in ("train", "valid"):
                infer_split(raw_model, config, manifest, prefix, device, rank, world,
                            output=directory / f"{prefix}_scores.npy", stop_check=stop_check)
            # Keep a fully resumable checkpoint until compact publication succeeds.
            last_path = save("scores_complete")
            on_rank_zero(lambda: finalize_run(directory, raw_model, config, manifest, state,
                                                        device, world, started, stop_check), rank)
            return directory / "model.pt"
        except InterruptedError:
            return save("stopped_during_evaluation")
    finally:
        if dataset is not None:
            dataset.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if distributed():
            dist.destroy_process_group()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, help="latest checkpoint .pt or run last.json")
    parser.add_argument("--max-steps", type=int, help="absolute optimizer-update limit for a short run")
    parser.add_argument("--skip-evaluation", action="store_true", help="development only: keep resumable checkpoint")
    args = parser.parse_args(argv)
    path = train(read_config(args.config), resume=args.resume, max_steps=args.max_steps,
                 skip_evaluation=args.skip_evaluation)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Artifact: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
