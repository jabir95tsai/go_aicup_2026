"""Shared dataset validation and runtime setup; no import-time I/O."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

from go_aicup.artifacts import file_hash, object_hash, write_json


def load_dataset_manifest(config: dict) -> dict:
    directory = Path(config["paths"]["dataset_dir"])
    record = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((Path(config["paths"]["cache_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    expected = {"cache_build_id": cache["build_id"], "split_config": config["split"],
                "questions_config": config["questions"], "seed": config["seed"]}
    if "monitoring" in config:
        expected["monitoring_config"] = config["monitoring"]
    if record.get("complete") is not True or record.get("format_version") != 2:
        raise ValueError("dataset preparation has not completed")
    if object_hash({key: value for key, value in record.items() if key != "dataset_id"}) != record.get("dataset_id"):
        raise ValueError("dataset manifest fingerprint mismatch")
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("dataset/cache/config mismatch; prepare a new dataset directory")
    for name, info in record["files"].items():
        path = directory / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"missing/invalid dataset file: {name}")
    for name, info in record["table_files"].items():
        path = directory / name
        if Path(name).name != name or file_hash(path) != info["sha256"]:
            raise ValueError(f"CV file changed: {name}")
    return record


def check_index_hashes(config: dict, manifest: dict) -> None:
    directory = Path(config["paths"]["dataset_dir"])
    for name, info in manifest["files"].items():
        if file_hash(directory / name) != info["sha256"]:
            raise ValueError(f"sample index checksum mismatch: {name}")


def setup_distributed(config: dict):
    """CUDA/NCCL production, CPU/Gloo only for explicit development configs."""
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    if not (1 <= world <= 8 and world == local_world and rank == local_rank and 0 <= rank < world):
        raise ValueError("only single-node 1-8 process jobs are supported")
    if config["device"] == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < world:
            raise RuntimeError("CUDA device/allocation unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("configured CUDA device does not support bf16")
        backend = "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
        if world > 1 and config["training"]["sync_batch_norm"]:
            raise ValueError("SyncBatchNorm requires CUDA; disable it only in a development config")
    if world > 1:
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    for batch in (config["training"]["global_batch_size"], config["evaluation"]["global_batch_size"]):
        if batch < world or batch % world:
            raise ValueError("global batch sizes must be divisible by world_size")
    return device, rank, world


def barrier() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def autocast_context(config: dict, device):
    import torch
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype, enabled=config["precision"] != "fp32")


def model_state_hash(model) -> str:
    """Fingerprint weights and BN buffers independently of checkpoint metadata."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def distributed() -> bool:
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized()


def on_rank_zero(function, rank):
    """Propagate rank-zero publication failures instead of stranding peers."""
    import torch.distributed as dist

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
