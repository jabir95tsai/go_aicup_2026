"""Shared V1 configuration and atomic artifact utilities; no import-time I/O."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path

from go_aicup.paths import ROOT


DEFAULT_CONFIG = ROOT / "configs" / "strength" / "rank_cnn_v1.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["format_version"] != 1 or config["cache_codec"] != "packed":
        raise ValueError("unsupported V1 configuration")
    for key, value in config["paths"].items():
        target = Path(value)
        config["paths"][key] = str((target if target.is_absolute() else ROOT / target).resolve())
    if config["precision"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16 or fp32")
    if config["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if not 1 <= config["training"]["epochs"] <= 10:
        raise ValueError("V1 supports one to ten epochs")
    if config["training"]["global_batch_size"] < 10:
        raise ValueError("global batch must fit all ten ranks")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU development requires explicit fp32 precision")
    return config


def load_dataset_manifest(config: dict) -> dict:
    directory = Path(config["paths"]["dataset_dir"])
    record = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cache = json.loads((Path(config["paths"]["cache_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    expected = {"cache_build_id": cache["build_id"], "split_config": config["split"],
                "questions_config": config["questions"], "seed": config["seed"]}
    if record.get("complete") is not True or record.get("format_version") != 1:
        raise ValueError("dataset preparation has not completed")
    if object_hash({key: value for key, value in record.items() if key != "dataset_id"}) != record.get("dataset_id"):
        raise ValueError("dataset manifest fingerprint mismatch")
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("dataset/cache/config mismatch; prepare a new dataset directory")
    for name, info in record["files"].items():
        path = directory / name
        if Path(name).name != name or not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"missing/invalid dataset file: {name}")
    for name, info in record["cv_files"].items():
        path = Path(config["paths"]["cv_dir"]) / name
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
