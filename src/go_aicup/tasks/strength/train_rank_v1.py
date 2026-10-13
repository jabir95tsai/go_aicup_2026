"""Strength V1 direct-forward training with global rank-mean loss.

Production: torchrun, single-node 1-8 CUDA GPUs, SyncBN and BF16. There is
one forward/backward/update per global batch: no accumulation or activation
recomputation. Dataset preparation is a separate explicit command.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from go_aicup.models.strength_resnet import StrengthResNet
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.dataset import (GlobalBatchSampler, PackedBoardDataset,
                                            create_permutation, make_loader)
from go_aicup.tasks.strength.losses import distributed, global_rank_loss
from go_aicup.tasks.strength.runtime import (DEFAULT_CONFIG, autocast_context, barrier,
    check_index_hashes, file_hash, load_dataset_manifest, object_hash, read_config,
    setup_distributed, write_json)


def seed_process(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch(config: dict) -> None:
    options = config["training"]
    torch.use_deterministic_algorithms(options["deterministic"])
    torch.backends.cudnn.benchmark = options["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = options["deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = options["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = options["allow_tf32"]
    if options["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def code_fingerprints() -> dict:
    names = ("models/strength_resnet.py", "tasks/strength/dataset.py",
             "tasks/strength/losses.py", "tasks/strength/train_rank_v1.py",
             "tasks/strength/evaluate_rank_v1.py", "tasks/strength/runtime.py",
             "data/board_cache.py", "data/cv.py")
    package = ROOT / "src" / "go_aicup"
    return {name: file_hash(package / name) for name in names}


def environment(device, world: int) -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=False)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    versions = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": versions, "git_commit": git("rev-parse", "HEAD"),
              "git_status": git("status", "--short"), "git_diff": git("diff", "--", "src", "configs"),
              "world_size": world, "torch_cuda": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(),
              "slurm": {key: os.environ.get(key) for key in
                        ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_CPUS_PER_TASK")}}
    if device.type == "cuda":
        result["gpu"] = [torch.cuda.get_device_name(i) for i in range(world)]
        result["gpu_memory_bytes"] = [torch.cuda.get_device_properties(i).total_memory for i in range(world)]
        result["nccl"] = list(torch.cuda.nccl.version()) if dist.is_nccl_available() else None
    return result


def rng_state(device, model) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
            # Also preserve local buffers in explicit non-SyncBN development runs.
            "buffers": {name: value.detach().cpu() for name, value in model.named_buffers()}}


def restore_rng(state: dict, device, model) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    for name, value in model.named_buffers():
        value.copy_(state["buffers"][name].to(value.device))


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
    if rank == 0:
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
    barrier()
    return path


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


def scheduler_for(optimizer, total_steps: int, warmup_fraction: float, minimum: float):
    if not 0 <= warmup_fraction < 1 or not 0 <= minimum <= 1:
        raise ValueError("invalid warmup or minimum learning-rate ratio")
    warmup = max(1, math.ceil(total_steps * warmup_fraction)) if warmup_fraction else 0

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def log_event(directory: Path, rank: int, event: dict) -> None:
    if rank == 0:
        event = dict(event, utc=datetime.now(timezone.utc).isoformat())
        message = json.dumps(event, allow_nan=False)
        with (directory / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")
        print(message, flush=True)


def train(config: dict, *, resume: Path | None = None,
          max_steps: int | None = None, skip_evaluation: bool = False) -> Path:
    configure_torch(config)
    manifest = load_dataset_manifest(config)
    device, rank, world = setup_distributed(config)
    directory = Path(config["paths"]["output_dir"])
    stop_requested = [False]
    previous_handlers = {}

    def request_stop(signum, frame):
        stop_requested[0] = True

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    started = time.monotonic()
    checkpoint_time = started
    run_samples = 0
    run_successful_steps = 0
    timings = []
    run_opened = False
    data_directory = Path(config["paths"]["dataset_dir"])
    dataset = None
    try:
        if rank == 0:
            check_index_hashes(config, manifest)
            if resume is None:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError("run directory is nonempty; use --resume or a new output_dir")
                directory.mkdir(parents=True, exist_ok=True)
                write_json(directory / "config.json", config)
                write_json(directory / "environment.json", environment(device, world))
                write_json(directory / "dataset_manifest.json", manifest)
                write_json(directory / "code_sha256.json", code_fingerprints())
                run_opened = True
        barrier()
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
        scheduler = scheduler_for(optimizer, batches_per_epoch * config["training"]["epochs"],
                                  config["training"]["warmup_fraction"], config["training"]["minimum_lr_ratio"])
        scaler = torch.amp.GradScaler("cuda", enabled=config["precision"] == "fp16")
        state = {"epoch": 0, "next_batch": 0, "optimizer_steps": 0,
                 "skipped_batches": 0, "samples_seen": 0, "permutation": None,
                 "permutation_sha256": None, "best_validation_score": None,
                 "gpu_seconds": 0.0}
        seed_process(config["seed"] + rank)
        if resume is not None:
            source = resolve_checkpoint(resume.resolve())
            saved = torch.load(source, map_location="cpu", weights_only=False)
            if (saved.get("format_version") != 1 or saved["config_sha256"] != object_hash(config)
                    or saved["dataset_id"] != manifest["dataset_id"] or saved["world_size"] != world
                    or saved["code_sha256"] != code_fingerprints()):
                raise ValueError("resume requires the same config, dataset, world_size and implementation")
            if json.loads((directory / "config.json").read_text(encoding="utf-8")) != config:
                raise ValueError("resume output directory config mismatch")
            run_opened = True
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

        def evaluation_stop_check():
            deadline = config["training"]["max_runtime_seconds"]
            return stop_requested[0] or (deadline is not None and
                time.monotonic() - started >= max(0, deadline - config["training"]["shutdown_margin_seconds"]))

        dataset = PackedBoardDataset(data_directory / "train_game.npy", data_directory / "train_ply.npy",
                                    Path(config["paths"]["cache_dir"]),
                                    max_open_shards=config["loader"]["max_open_shards"],
                                    augment=config["loader"]["d4"], seed=config["seed"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        log_event(directory, rank, {"event": "start", "resumed": resume is not None,
                    "world_size": world, "parameters": sum(p.numel() for p in model.parameters()),
                    "global_batch_size": config["training"]["global_batch_size"],
                    "training_samples": len(dataset), "batches_per_epoch": batches_per_epoch,
                    "precision": config["precision"]})
        limit = max_steps if max_steps is not None else config["training"]["max_steps"]
        if limit is not None and limit < 1:
            raise ValueError("max_steps must be positive")
        last_path = source if resume is not None else None
        while state["epoch"] < config["training"]["epochs"]:
            epoch = state["epoch"]
            if rank == 0:
                create_permutation(directory / "permutations", len(dataset), config["seed"], epoch)
            barrier()
            permutation = directory / "permutations" / f"epoch_{epoch:04d}.npy"
            state["permutation"] = str(permutation)
            state["permutation_sha256"] = json.loads(permutation.with_suffix(".json").read_text(encoding="utf-8"))["sha256"]
            sampler = GlobalBatchSampler(len(dataset), config["training"]["global_batch_size"],
                                         rank=rank, world_size=world, permutation=permutation,
                                         epoch=epoch, start_batch=state["next_batch"])
            options = {key: config["loader"][key] for key in
                       ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
            loader = make_loader(dataset, sampler, **options)
            model.train()
            previous_step_end = time.monotonic()
            for step, batch in enumerate(loader, start=state["next_batch"]):
                elapsed = time.monotonic() - started
                deadline = config["training"]["max_runtime_seconds"]
                stopping = stop_requested[0] or (limit is not None and state["optimizer_steps"] >= limit)
                if deadline is not None and elapsed >= max(0, deadline - config["training"]["shutdown_margin_seconds"]):
                    stopping = True
                flag = torch.tensor(int(stopping), device=device, dtype=torch.int32)
                if distributed():
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item():
                    last_path = save("stopped")
                    log_event(directory, rank, {"event": "stopped", "epoch": epoch,
                                               "next_batch": state["next_batch"], "optimizer_steps": state["optimizer_steps"]})
                    return last_path
                # Remove tail placeholders before Conv/SyncBN, not just in loss.
                valid = batch["valid"].bool()
                labels = batch["rank_id"][valid].to(device, non_blocking=True)
                counts = torch.bincount(labels, minlength=10).float()
                if distributed():
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                if (counts == 0).any():
                    state["next_batch"] = step + 1
                    state["skipped_batches"] += 1
                    log_event(directory, rank, {"event": "missing_ranks_skip", "epoch": epoch,
                        "batch": step, "n": int(counts.sum().item()),
                        "missing_ranks": torch.where(counts == 0)[0].tolist()})
                    # No BN update, optimizer update, or scheduler update.
                    previous_step_end = time.monotonic()
                    continue
                # At least 10 real examples and at most 8 ranks => fair tail
                # partition guarantees a nonempty real input on every process.
                x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                if config["training"]["channels_last"]:
                    x = x.contiguous(memory_format=torch.channels_last)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                data_done = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(config, device):
                    scores = model(x)
                loss, means, loss_counts = global_rank_loss(scores, labels)
                scaler.scale(loss).backward()
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                bad_gradient = (~torch.isfinite(norm)).to(torch.int32)
                if distributed():
                    dist.all_reduce(bad_gradient, op=dist.ReduceOp.MAX)
                if bad_gradient.item():
                    raise FloatingPointError("nonfinite gradient on at least one process; no optimizer update")
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_done = time.monotonic()
                state["optimizer_steps"] += 1
                state["next_batch"] = step + 1
                state["samples_seen"] += int(loss_counts.sum().item())
                run_samples += int(loss_counts.sum().item())
                run_successful_steps += 1
                # Max process timing, with CUDA work completed before timing.
                measured = torch.tensor([data_done - previous_step_end, step_done - data_done],
                                        device=device, dtype=torch.float64)
                if distributed():
                    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
                data_seconds, compute_seconds = measured.cpu().tolist()
                timings.append(data_seconds + compute_seconds)
                if state["optimizer_steps"] % config["training"]["log_every_steps"] == 0:
                    log_event(directory, rank, {"event": "update", "epoch": epoch, "batch": step,
                        "optimizer_steps": state["optimizer_steps"], "loss": float(loss.item()),
                        "rank_means": means.detach().cpu().tolist(), "rank_counts": loss_counts.cpu().tolist(),
                        "gradient_norm_before_clip": float(norm.item()), "next_lr": optimizer.param_groups[0]["lr"],
                        "data_seconds": data_seconds, "compute_seconds": compute_seconds,
                        "samples_per_second": int(loss_counts.sum().item()) / max(data_seconds + compute_seconds, 1e-9)})
                due = torch.tensor(int(rank == 0 and time.monotonic() - checkpoint_time >=
                                       config["training"]["checkpoint_seconds"]), device=device, dtype=torch.int32)
                if distributed():
                    dist.broadcast(due, src=0)
                if due.item():
                    last_path = save("periodic")
                    checkpoint_time = time.monotonic()
                previous_step_end = time.monotonic()
            # A checkpoint at an epoch boundary retains epoch + end cursor until
            # its fixed-weight evaluation completes. Resume can finish that work.
            last_path = save("epoch")
            should_evaluate = (not skip_evaluation and (epoch + 1) % config["evaluation"]["every_epochs"] == 0)
            if should_evaluate:
                from go_aicup.tasks.strength.evaluate_rank_v1 import EvaluationInterrupted, evaluate
                output = directory / "evaluation" / f"epoch_{epoch:04d}"
                completion = output / "complete.json"
                result = None
                if completion.exists():
                    record = json.loads(completion.read_text(encoding="utf-8"))
                    original = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                    # A completed evaluation must match the saved model weights.
                    if original["dataset_id"] != manifest["dataset_id"] or file_hash(output / "metrics.json") != record["metrics_sha256"]:
                        raise ValueError("completed evaluation integrity mismatch")
                    bound_path = resolve_checkpoint(Path(original["checkpoint"]))
                    if file_hash(bound_path) != original["checkpoint_sha256"]:
                        raise ValueError("evaluation checkpoint checksum mismatch")
                    bound = torch.load(bound_path, map_location="cpu", weights_only=False)
                    if any(not torch.equal(value.detach().cpu(), bound["model"][name])
                           for name, value in raw_model.state_dict().items()):
                        raise ValueError("completed evaluation belongs to different weights")
                    if rank == 0:
                        result = original
                else:
                    # If interrupted mid-evaluation, retain that partial output
                    # and publish a new attempt, without overwriting evidence.
                    attempt = [str(output)]
                    if rank == 0 and output.exists() and any(output.iterdir()):
                        attempt[0] = str(directory / "evaluation" / f"epoch_{epoch:04d}_retry_{time.time_ns()}")
                    if distributed():
                        dist.broadcast_object_list(attempt, src=0)
                    try:
                        result = evaluate(raw_model, config, manifest, last_path, Path(attempt[0]),
                                          device, rank, world, stop_check=evaluation_stop_check)
                    except EvaluationInterrupted:
                        last_path = save("stopped_during_evaluation")
                        log_event(directory, rank, {"event": "evaluation_interrupted", "epoch": epoch,
                                                   "next_batch": state["next_batch"]})
                        return last_path
                best = [state["best_validation_score"]]
                if rank == 0:
                    score = result["questions"]["validation"]["equal_k"]["score"]
                    if best[0] is None or score > best[0]:
                        best[0] = score
                        checkpoint_reference = Path(result["checkpoint"])
                        write_json(directory / "best.json", {"file": checkpoint_reference.name,
                                    "sha256": result["checkpoint_sha256"], "validation_equal_k_score": score,
                                    "metrics": str(Path(attempt[0]) / "metrics.json") if not completion.exists() else str(output / "metrics.json")})
                    log_event(directory, rank, {"event": "validation", "epoch": epoch,
                                               "equal_k": result["questions"]["validation"]["equal_k"],
                                               "by_k": result["questions"]["validation"]["by_k"]})
                if distributed():
                    dist.broadcast_object_list(best, src=0)
                state["best_validation_score"] = best[0]
            state["epoch"] = epoch + 1
            state["next_batch"] = 0
            state["permutation"] = None
            state["permutation_sha256"] = None
            last_path = save("epoch_complete")
            checkpoint_time = time.monotonic()
            del loader
        if rank == 0:
            write_json(directory / "finished.json", {"checkpoint": str(last_path),
                        "optimizer_steps": state["optimizer_steps"], "epochs": state["epoch"],
                        "skipped_batches": state["skipped_batches"], "samples_seen": state["samples_seen"],
                        "best_validation_score": state["best_validation_score"],
                        "evaluation_skipped": skip_evaluation})
        return last_path
    finally:
        if dataset is not None:
            dataset.close()
        if rank == 0 and run_opened:
            duration = time.monotonic() - started
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            write_json(directory / f"session_{time.time_ns()}.json", {
                "seconds": duration, "world_size": world,
                "gpu_hours": duration * world / 3600 if device.type == "cuda" else 0.0,
                "successful_updates": run_successful_steps, "samples_processed": run_samples,
                "measured_step_p50_seconds": float(np.quantile(timings, 0.5)) if timings else None,
                "measured_step_p95_seconds": float(np.quantile(timings, 0.95)) if timings else None,
                "estimated_training_epoch_seconds": float(np.mean(timings)) * batches_per_epoch if timings else None,
                "epoch_estimate_excludes": "shuffle, validation/calibration, checkpoint, initialization and queue time",
                "rank_zero_peak_cuda_allocated_bytes": peak,
                "skip_evaluation": skip_evaluation})
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if distributed():
            dist.destroy_process_group()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, help="completed checkpoint .pt or run last.json")
    parser.add_argument("--max-steps", type=int, help="absolute optimizer-update limit, for an explicit short run")
    parser.add_argument("--skip-evaluation", action="store_true", help="explicit short-run option; no cutpoints/metrics")
    args = parser.parse_args(argv)
    path = train(read_config(args.config), resume=args.resume, max_steps=args.max_steps,
                 skip_evaluation=args.skip_evaluation)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Checkpoint: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
