"""Trainer checkpoints: atomic save, latest-only or loss-based retention, resolve.

Each checkpoint carries model/optimizer/scheduler/scaler state, every rank's
RNG, the config/dataset fingerprints and the caller's code hashes. Writes are
atomic and published through last.json before older weights are removed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
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
                    tag: str = "periodic", loss: float | None = None) -> Path:
    # Omitting loss preserves V1/V2 latest-only retention; SE keeps its two best losses plus latest.
    if loss is not None and not math.isfinite(loss):
        raise ValueError("checkpoint loss must be finite")
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
        previous = None
        if loss is not None and (directory / "last.json").exists():
            previous = json.loads((directory / "last.json").read_text())
            resolve_checkpoint(directory / "last.json")
            if not math.isfinite(previous["loss"]):
                raise ValueError("previous checkpoint loss must be finite")
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
        if loss is not None:
            record["loss"] = loss
        write_json(path.with_suffix(".complete.json"), record)
        write_json(directory / "last.json", record)
        # Publish the replacement completely before removing previous weights.
        if loss is None:
            prune_checkpoints(directory, path)
        else:
            best_path = directory / "best.json"
            candidates = json.loads(best_path.read_text())["checkpoints"] if best_path.exists() else []
            candidates += ([previous] if previous is not None else []) + [record]
            candidates = list({item["file"]: item for item in candidates}.values())
            for item in candidates:
                if Path(item["file"]).name != item["file"] or not math.isfinite(item["loss"]):
                    raise ValueError("invalid ranked checkpoint record")
            best = sorted(candidates, key=lambda item: (item["loss"], -item["optimizer_steps"], item["file"]))[:2]
            for item in best:
                if file_hash(path.parent / item["file"]) != item["sha256"]:
                    raise ValueError("ranked checkpoint hash mismatch")
            write_json(best_path, {"checkpoints": best})
            keep = {path.name, *(item["file"] for item in best)}
            for old in path.parent.glob("update_*.pt"):
                if old.name not in keep:
                    old.unlink()
                    old.with_suffix(".complete.json").unlink(missing_ok=True)
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
