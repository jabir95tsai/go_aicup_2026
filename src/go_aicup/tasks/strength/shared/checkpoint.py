"""Objective-independent trainer checkpoints: save, latest-only retention, resolve.

Each checkpoint carries model/optimizer/scheduler/scaler state, every rank's
RNG, the config/dataset fingerprints and the caller's code hashes. Writes are
atomic and published through last.json before older weights are removed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from go_aicup.tasks.strength.shared.runtime import (barrier, distributed, file_hash,
    object_hash, on_rank_zero, write_json)
from go_aicup.tasks.strength.shared.training import rng_state


def save_checkpoint(directory: Path, model, optimizer, scheduler, scaler,
                    config: dict, manifest: dict, state: dict, device,
                    rank: int, world: int, *, code_sha256: dict,
                    tag: str = "periodic") -> Path:
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
                   "code_sha256": code_sha256, "state": dict(state),
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
