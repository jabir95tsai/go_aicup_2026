"""Strength V2 ten-class training: global CE, seen/unseen whole-game monitors.

Production: torchrun, single-node 1-8 CUDA GPUs, SyncBN and BF16; one
forward/backward/update per global batch with a fixed learning rate. Every
evaluate_every_steps updates and after training, the frozen model scores
monitor_unseen and monitor_seen (whole games, all labelled-player moves, no
D4). After the configured epochs it scores every valid_unseen and valid_seen
game. Dataset preparation is a separate explicit command.

Final run directory: model.pt, losses.npy, validation_unseen_category.npy,
validation_seen_category.npy. Checkpoints, permutations and logs are removed
only after model.pt has been reloaded and verified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from go_aicup.data.board_cache import RANKS
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.shared.checkpoint import (resolve_checkpoint,
    save_checkpoint as shared_save_checkpoint)
from go_aicup.tasks.strength.shared.dataset import (GlobalBatchSampler, PackedBoardDataset,
                                                    create_permutation, make_loader)
from go_aicup.tasks.strength.shared.runtime import (autocast_context, barrier, check_index_hashes,
    distributed, file_hash, load_dataset_manifest, model_state_hash, object_hash, on_rank_zero,
    setup_distributed, write_json)
from go_aicup.tasks.strength.shared.training import (configure_torch, environment, log_event,
    restore_rng, rng_state, scheduler_for, seed_process)
from go_aicup.tasks.strength.v2.aggregate import monitor_metrics
from go_aicup.tasks.strength.v2.config import DEFAULT_CONFIG, read_config
from go_aicup.tasks.strength.v2.infer import infer_games
from go_aicup.tasks.strength.v2.losses import global_cross_entropy
from go_aicup.tasks.strength.v2.model import StrengthResNet

LOSS_DTYPE = np.dtype([("step", "<i8"), ("train_loss", "<f8"),
                       ("unseen_loss", "<f8"), ("unseen_game_score", "<f8"),
                       ("seen_loss", "<f8"), ("seen_game_score", "<f8")])
MONITORS = ("monitor_unseen", "monitor_seen")
CATEGORY_FILES = {"valid_unseen": "validation_unseen_category.npy",
                  "valid_seen": "validation_seen_category.npy"}
ARTIFACTS = ("losses.npy", *CATEGORY_FILES.values())
STEP_LOG_INTERVAL = 10


def code_fingerprints() -> dict:
    names = ("artifacts.py", "metrics.py", "data/board_cache.py", "data/cv.py",
             *(f"tasks/strength/shared/{name}.py" for name in
               ("model", "dataset", "runtime", "training", "prepare", "checkpoint")),
             *(f"tasks/strength/v2/{name}.py" for name in
               ("model", "config", "losses", "aggregate", "infer", "train", "prepare")))
    package = ROOT / "src" / "go_aicup"
    return {name: file_hash(package / name) for name in names}


def save_checkpoint(*args, **kwargs) -> Path:
    return shared_save_checkpoint(*args, code_sha256=code_fingerprints(), **kwargs)


def write_array(path: Path, array) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def write_losses(path: Path, rows) -> None:
    write_array(path, np.asarray([tuple(row) for row in rows], dtype=LOSS_DTYPE))


def finalize_run(directory: Path, model, config: dict, manifest: dict, state: dict) -> Path:
    """Rank zero only: publish a verified compact model, then clean up."""
    payload = {"artifact_format": "strength_v2_1", "class_names": list(RANKS),
               "seed": config["seed"],
               "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
               "config": config, "dataset_id": manifest["dataset_id"],
               "cache_build_id": manifest["cache_build_id"], "index_files": manifest["files"],
               "monitor_sets": {name: manifest["sets"][name] for name in MONITORS},
               "loss_columns": list(LOSS_DTYPE.names),
               "player_overlap": manifest["player_overlap"],
               "optimizer_steps": state["optimizer_steps"], "epochs": state["epoch"],
               "samples_seen": state["samples_seen"], "timings": state["timings"],
               "model_state_sha256": model_state_hash(model), "code_sha256": code_fingerprints(),
               "environment": json.loads((directory / "environment.json").read_text()),
               "gpu_seconds": state["gpu_seconds"],
               "artifact_sha256": {name: file_hash(directory / name) for name in ARTIFACTS}}
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


def save_snapshot(directory: Path, model, epoch: int) -> Path:
    """Rank zero only: state_dict after a completed epoch, kept by cleanup_run."""
    path = directory / "snapshots" / f"epoch_{epoch:04d}.pt"
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, temporary)
    os.replace(temporary, path)
    return path


def cleanup_run(directory: Path) -> None:
    """Remove only trainer-owned transient files after model publication."""
    for name in ("checkpoints", "permutations"):
        target = directory / name
        if target.exists():
            shutil.rmtree(target)
    for name in ("config.json", "environment.json", "last.json", "train.jsonl"):
        (directory / name).unlink(missing_ok=True)
    for temporary in directory.glob("*.tmp"):
        temporary.unlink()


def train(config: dict, *, resume: Path | None = None,
          max_steps: int | None = None, skip_evaluation: bool = False) -> Path:
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
        if "warmup_fraction" in config["training"]:
            # A-024 recipe: linear warmup, then cosine decay over all configured epochs.
            scheduler = scheduler_for(optimizer, batches_per_epoch * config["training"]["epochs"],
                                      config["training"]["warmup_fraction"],
                                      config["training"]["minimum_lr_ratio"])
        else:
            # Retain a checkpointable scheduler object as a no-op for fixed LR.
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16")
        state = {"epoch": 0, "next_batch": 0, "optimizer_steps": 0, "samples_seen": 0,
                 "permutation": None, "permutation_sha256": None, "gpu_seconds": 0.0,
                 "loss_rows": [], "pending_evaluation": False, "last_batch_loss": None,
                 "timings": {"train_seconds": 0.0, "monitor_seconds": [], "final_inference_seconds": {}}}
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
        def frozen_inference(prefix):
            # Evaluation must not advance training RNG or alter local BN buffers.
            snapshot = rng_state(device, raw_model)
            try:
                if distributed():
                    for buffer in raw_model.buffers():
                        dist.broadcast(buffer, src=0)
                return infer_games(raw_model, config, prefix, device, rank, world, stop_check=stop_check)
            finally:
                restore_rng(snapshot, device, raw_model)
        def record_evaluation():
            began = time.monotonic()
            metrics = {name: monitor_metrics(frozen_inference(name)) for name in MONITORS}
            seconds = time.monotonic() - began
            step = state["optimizer_steps"]
            unseen, seen = metrics["monitor_unseen"], metrics["monitor_seen"]
            row = (step, state["last_batch_loss"], unseen["loss"], unseen["game_score"],
                   seen["loss"], seen["game_score"])
            state["loss_rows"] = [old for old in state["loss_rows"] if old[0] < step] + [row]
            state["timings"]["monitor_seconds"] = [old for old in state["timings"]["monitor_seconds"]
                                                   if old[0] < step] + [[step, seconds]]
            on_rank_zero(lambda: write_losses(directory / "losses.npy", state["loss_rows"]), rank)
            state["pending_evaluation"] = False
            log_event(directory, rank, {"event": "monitor", "optimizer_steps": step,
                                        "train_loss": row[1], "monitor_seconds": seconds,
                                        **{f"{name}_{key}": value for name, values in metrics.items()
                                           for key, value in values.items()}})
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
        try:
            if state["pending_evaluation"] and state["next_batch"] < batches_per_epoch and not skip_evaluation:
                record_evaluation()
                save("evaluated")
            while state["epoch"] < config["training"]["epochs"]:
                epoch = state["epoch"]
                if state["permutation"] is None:
                    on_rank_zero(lambda: create_permutation(
                        directory / "permutations", len(dataset), config["seed"], epoch), rank)
                    permutation = directory / "permutations" / f"epoch_{epoch:04d}.npy"
                    state["permutation"] = str(permutation)
                    state["permutation_sha256"] = file_hash(permutation)
                else:
                    permutation = Path(state["permutation"])
                sampler = GlobalBatchSampler(len(dataset), config["training"]["global_batch_size"],
                                             rank=rank, world_size=world, permutation=permutation,
                                             epoch=epoch, start_batch=state["next_batch"])
                options = {key: config["loader"][key] for key in
                           ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
                loader = make_loader(dataset, sampler, **options)
                model.train()
                window = time.monotonic()
                for step, batch in enumerate(loader, start=state["next_batch"]):
                    flag = torch.tensor(int(stop_check() or (limit is not None and state["optimizer_steps"] >= limit)), device=device)
                    if distributed():
                        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    if flag.item():
                        return save("stopped")
                    began = time.monotonic()
                    valid = batch["valid"].bool()
                    labels = batch["rank_id"][valid].to(device, non_blocking=True)
                    x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                    if config["training"]["channels_last"]:
                        x = x.contiguous(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast_context(config, device):
                        logits = model(x)
                    loss, _, count = global_cross_entropy(logits, labels)
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
                    state["samples_seen"] += int(count.item())
                    state["timings"]["train_seconds"] += time.monotonic() - began
                    if state["optimizer_steps"] % STEP_LOG_INTERVAL == 0:
                        now = time.monotonic()
                        log_event(directory, rank, {"event": "train", "optimizer_steps": state["optimizer_steps"],
                                                    "loss": state["last_batch_loss"], "grad_norm": float(norm.item()),
                                                    "lr": optimizer.param_groups[0]["lr"],
                                                    "mean_step_seconds": (now - window) / STEP_LOG_INTERVAL})
                        window = now
                    interval = config["training"]["evaluate_every_steps"]
                    if not skip_evaluation and state["optimizer_steps"] % interval == 0 and step + 1 < batches_per_epoch:
                        state["pending_evaluation"] = True
                        save("before_monitor")
                        record_evaluation()
                        save("evaluated")
                        checkpoint_time = window = time.monotonic()
                    due = torch.tensor(int(rank == 0 and time.monotonic() - checkpoint_time >=
                                           config["training"]["checkpoint_seconds"]), device=device)
                    if distributed():
                        dist.broadcast(due, src=0)
                    if due.item():
                        save("periodic")
                        checkpoint_time = window = time.monotonic()
                del loader
                if config["training"].get("epoch_snapshots"):
                    # Weights only, for post-hoc per-epoch evaluation; never used to resume.
                    on_rank_zero(lambda: save_snapshot(directory, raw_model, epoch + 1), rank)
                state["epoch"] += 1
                state["next_batch"] = 0
                state["permutation"] = None
                state["permutation_sha256"] = None
            state["pending_evaluation"] = not skip_evaluation
            last_path = save("training_complete")
            if skip_evaluation:
                return last_path  # Explicit development run, retain resumable state.
            if state["pending_evaluation"]:
                record_evaluation()
            for prefix, name in CATEGORY_FILES.items():
                began = time.monotonic()
                category = frozen_inference(prefix)
                on_rank_zero(lambda: write_array(directory / name, category), rank)
                state["timings"]["final_inference_seconds"][prefix] = time.monotonic() - began
            # Keep a fully resumable checkpoint until compact publication succeeds.
            save("categories_complete")
            on_rank_zero(lambda: finalize_run(directory, raw_model, config, manifest, state), rank)
            barrier()
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
