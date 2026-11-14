"""Train the competition-1 GE2E insurance encoder (torchrun, single node, 1-8 GPUs).

Training players exclude every player of the synthetic pools in
config["exclude_splits"] (cv.pid_train_players), so those pools stay valid
for evaluation. Each GPU runs its own GE2E group (players x games x moves);
DDP averages gradients. last.pt is written every checkpoint_steps and on
SIGTERM/SIGINT; rerunning the same command resumes from it.

After the last step, every gallery and question game of config["eval_splits"]
is embedded with ALL its black moves from ply 20 on, and rank 0 writes
embeddings_split{n}.npz (game_index, vector = mean move embedding) next to
model.pt. Scoring is CPU-only: scripts/identity/identity_ge2e_eval.py.

    torchrun --standalone --nproc-per-node=8 -m go_aicup.tasks.identity.train_ge2e \\
        --config configs/identity/pid_ge2e_v1_short.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data
from go_aicup.tasks.identity.ge2e import MIN_PLY, GE2EBatches, IdentityEncoder, ge2e_loss, own_plies
from go_aicup.tasks.strength.shared.runtime import autocast_context, distributed, setup_distributed
from go_aicup.tasks.strength.shared.training import (configure_torch, environment, log_event,
                                                     scheduler_for, seed_process)


def read_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in config["paths"].items():
        target = Path(value)
        config["paths"][key] = str((target if target.is_absolute() else ROOT / target).resolve())
    batch = config["batch"]
    per_gpu = batch["players"] * batch["games"] * batch["moves"]
    # Keys required by the shared runtime helpers.
    config.setdefault("evaluation", {})["global_batch_size"] = config["evaluation"].get("batch", 8192)
    config["training"]["global_batch_size"] = per_gpu
    config["training"].setdefault("sync_batch_norm", False)
    return config


def training_players(config: dict):
    """(game lists per player, lengths, held-out players, player colour per game or None).

    config["sampling"] = {"colors": "both", "min_ply": n} trains on the player's own moves in
    games of either colour (white moves colour-swapped); absent = black games, ply >= MIN_PLY."""
    meta = data.load_meta()
    gallery, _ = data.pid_splits(meta)
    held = set(gallery.loc[gallery["split"].isin(config["exclude_splits"]), "player_id"])
    both = config.get("sampling", {}).get("colors", "black") == "both"
    keep = ~meta["player_id"].astype(str).isin(held)
    if not both:
        keep &= meta["color"].astype(str) == "B"
    max_handicap = config.get("sampling", {}).get("max_handicap")
    if max_handicap is not None:
        # HD >= 2 records carry the opponent's moves under the player's colour and lack the
        # handicap stones (CLAUDE.md section 1); the official competition-1 files have none.
        table = pd.read_csv(ROOT / "data" / "raw" / "training" / "handicap.csv",
                            usecols=["rank", "source_row", "handicap"])
        handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                      validate="1:1")["handicap"].to_numpy()
        keep &= handicap <= max_handicap
    games = meta[keep]
    groups = games.groupby(games["player_id"].astype(str), observed=True)["game_index"].apply(np.asarray)
    player_games = [g for g in groups if len(g) >= config["batch"]["games"]]
    color = meta["player_color"].to_numpy().astype(np.int8) if both else None
    return player_games, meta["length"].to_numpy(), len(held), color


class EvalChunks(torch.utils.data.Dataset):
    def __init__(self, cache_dir, games: np.ndarray, plies: np.ndarray, batch: int):
        self.cache_dir, self.games, self.plies, self.batch = cache_dir, games, plies, batch
        self._cache = None

    def __len__(self):
        return math.ceil(len(self.games) / self.batch)

    def __getitem__(self, item):
        if self._cache is None:
            self._cache = BoardCache(self.cache_dir)
        span = slice(item * self.batch, (item + 1) * self.batch)
        histories, colors = self._cache.history_batch(self.games[span], self.plies[span])
        return torch.from_numpy(assemble_inputs(histories, colors)), item

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


@torch.no_grad()
def embed_games(model, config, games: np.ndarray, lengths, device, rank, world) -> np.ndarray:
    """[len(games), D] mean move embedding over every black move at ply >= 20."""
    plies = [own_plies(int(lengths[g])) for g in games]
    rows = np.concatenate([np.full(len(p), i) for i, p in enumerate(plies)])
    sample_games, sample_plies = games[rows], np.concatenate(plies)
    mine = np.arange(rank, len(rows), world)       # rank-strided, every sample exactly once
    dataset = EvalChunks(config["paths"]["cache_dir"], sample_games[mine], sample_plies[mine],
                         config["evaluation"]["global_batch_size"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=config["loader"]["num_workers"],
                                         multiprocessing_context="spawn" if config["loader"]["num_workers"] else None)
    dim = config["model"]["embedding"]
    sums = np.zeros((len(games), dim), np.float64)
    model.eval()
    for x, item in loader:
        with autocast_context(config, device):
            out = model(x.to(device, dtype=torch.float32, non_blocking=True))
        span = mine[item * dataset.batch:(item + 1) * dataset.batch]
        np.add.at(sums, rows[span], out.float().cpu().numpy().astype(np.float64))
    if distributed():
        tensor = torch.from_numpy(sums).to(device)
        dist.all_reduce(tensor)
        sums = tensor.cpu().numpy()
    return (sums / np.array([len(p) for p in plies])[:, None]).astype(np.float32)


def export_embeddings(model, config, splits, lengths, device, rank, world) -> None:
    out = Path(config["paths"]["output_dir"])
    gallery_all, questions_all = data.pid_splits(data.load_meta())
    for split in splits:
        started = time.monotonic()
        games = np.unique(np.concatenate([gallery_all.loc[gallery_all["split"] == split, "game"].to_numpy(),
                                          *questions_all.loc[questions_all["split"] == split, "games"]]))
        vectors = embed_games(model, config, games, lengths, device, rank, world)
        if rank == 0:
            np.savez(out / f"embeddings_split{split}.npz", game_index=games, vector=vectors)
        log_event(out, rank, {"event": "embedded", "split": split, "games": len(games),
                              "seconds": time.monotonic() - started})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--embed-only", type=int, nargs="+", metavar="SPLIT",
                        help="skip training; embed these splits with the run's saved model.pt")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    device, rank, world = setup_distributed(config)
    configure_torch(config)
    if args.embed_only:
        out = Path(config["paths"]["output_dir"])
        raw = IdentityEncoder(**config["model"])
        raw.load_state_dict(torch.load(out / "model.pt", map_location="cpu", weights_only=False)["model"])
        raw.to(device)
        export_embeddings(raw, config, args.embed_only, data.load_meta()["length"].to_numpy(), device, rank, world)
        return 0
    seed_process(config["seed"] + rank)
    out = Path(config["paths"]["output_dir"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(config, indent=2))
        (out / "environment.json").write_text(json.dumps(environment(device, world), indent=2))
    player_games, lengths, held, player_color = training_players(config)
    batch, training = config["batch"], config["training"]
    if batch["players"] > len(player_games):
        raise ValueError("not enough training players for one GE2E group")

    raw = IdentityEncoder(**config["model"])
    source = torch.load(config["paths"]["init_model"], map_location="cpu", weights_only=False)
    raw.load_trunk(source["model"])
    raw.to(device)
    model = DistributedDataParallel(raw, device_ids=[device.index]) if world > 1 else raw
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
    scheduler = scheduler_for(optimizer, training["steps"], training["warmup_fraction"],
                              training["minimum_lr_ratio"])
    start = 0
    last = out / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["world"] != world or saved["config"] != config:
            raise ValueError("resume requires the same config and world size")
        raw.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["step"]
    log_event(out, rank, {"event": "start", "world": world, "training_players": len(player_games),
                          "excluded_players": held, "start_step": start, "per_gpu_boards":
                          batch["players"] * batch["games"] * batch["moves"]})

    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))

    def save(step):
        if rank == 0:
            payload = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "step": step, "world": world, "config": config}
            torch.save(payload, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", last)
        if distributed():
            dist.barrier()

    cross_gpu = bool(batch.get("cross_gpu", False)) and world > 1
    dataset = GE2EBatches(config["paths"]["cache_dir"], player_games, lengths, players=batch["players"],
                          games=batch["games"], moves=batch["moves"], steps=training["steps"],
                          seed=config["seed"], rank=rank, start_step=start,
                          max_open_shards=config["loader"]["max_open_shards"], player_color=player_color,
                          min_ply=config.get("sampling", {}).get("min_ply", MIN_PLY),
                          player_d4=config.get("sampling", {}).get("player_d4", False),
                          world=world if cross_gpu else 1)
    workers = config["loader"]["num_workers"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=True,
                                         prefetch_factor=config["loader"]["prefetch_factor"] if workers else None,
                                         persistent_workers=workers > 0,
                                         multiprocessing_context="spawn" if workers else None)
    model.train()
    step, clock, window = start, time.monotonic(), []
    n, m, s = batch["players"], batch["games"], batch["moves"]
    for x in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        with autocast_context(config, device):
            moves = model(x)
        games = moves.float().reshape(n, m, s, -1).mean(dim=2)
        if cross_gpu:
            # one GE2E group of world * n players: the other ranks' game vectors enter as constants,
            # this rank's keep their graph. Every rank evaluates the same loss, so the DDP average of
            # the per-rank gradients is 1/world of the full gradient; the factor restores it.
            pooled = [torch.zeros_like(games) for _ in range(world)]
            dist.all_gather(pooled, games.detach())
            pooled[rank] = games
            loss, accuracy = ge2e_loss(torch.cat(pooled), raw.w, raw.b)
            loss = loss * world
        else:
            loss, accuracy = ge2e_loss(games, raw.w, raw.b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
        optimizer.step()
        scheduler.step()
        step += 1
        now = time.monotonic()
        window.append(now - clock)
        clock = now
        if step % training["log_every"] == 0 or step == training["steps"]:
            stats = torch.stack([loss.detach() / (world if cross_gpu else 1), accuracy.detach()])
            if distributed():
                dist.all_reduce(stats)
                stats /= world
            log_event(out, rank, {"event": "train", "step": step, "loss": float(stats[0]),
                                  "own_centroid_accuracy": float(stats[1]), "w": float(raw.w), "b": float(raw.b),
                                  "lr": optimizer.param_groups[0]["lr"],
                                  "mean_step_seconds": float(np.mean(window[-training["log_every"]:]))})
        flag = torch.tensor(int(stop[0]), device=device)
        if distributed():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if step % training["checkpoint_steps"] == 0 or flag.item():
            save(step)
        if flag.item():
            log_event(out, rank, {"event": "stopped", "step": step})
            return 1
    save(step)
    del loader

    export_embeddings(raw, config, config["eval_splits"], lengths, device, rank, world)
    if rank == 0:
        torch.save({"model": raw.state_dict(), "config": config, "steps": step}, out / "model.pt")
    if distributed():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
