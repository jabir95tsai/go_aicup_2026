"""Objective-independent RNG, environment, scheduler and logging helpers."""
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist

from go_aicup.paths import ROOT

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

