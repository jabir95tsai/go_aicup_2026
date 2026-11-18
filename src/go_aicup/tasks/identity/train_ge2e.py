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


class EncoderHead(torch.nn.Module):
    """Encoder + B aggregation head in one module (2026-11-17, E2E-scratch), so DDP syncs both.
    forward(x, plies) -> game vectors [N, M, D]: encoder per move (autocast), then the B head
    (per-move residual blocks with the ply embedding, FP32) and the mean over the game's moves."""

    def __init__(self, encoder, head, config, device, shape):
        super().__init__()
        self.encoder, self.head = encoder, head
        self._config, self._device, self._shape = config, device, shape

    def forward(self, x: torch.Tensor, plies: torch.Tensor) -> torch.Tensor:
        from go_aicup.tasks.identity.train_e2e import joint_games
        with autocast_context(self._config, self._device):
            moves = self.encoder(x)
        return joint_games(self.head, moves, plies, self._shape)


def wsd_scheduler(optimizer, total_steps: int, warmup_fraction: float, decay_fraction: float):
    """Warmup-stable-decay (2026-11-17): linear warmup, constant LR, then linear decay to 0 over the
    last decay_fraction of the steps. Returns (scheduler, decay_start). The LR before decay_start
    does not depend on total_steps, so a run can be extended from its pre-decay checkpoint."""
    if not 0 <= warmup_fraction < 1 or not 0 < decay_fraction < 1:
        raise ValueError("invalid warmup or decay fraction")
    warmup = max(1, math.ceil(total_steps * warmup_fraction)) if warmup_fraction else 0
    decay_start = total_steps - int(round(total_steps * decay_fraction))
    if decay_start <= warmup:
        raise ValueError("decay starts before warmup ends")

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / (total_steps - decay_start))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier), decay_start


def head_payload(head, step: int, seed: int) -> dict:
    """seqagg.load_head format."""
    return {"model": head.state_dict(), "step": step, "arm": "B", "position": True, "seed": seed}


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


def stratified_subset(players: np.ndarray, ranks: np.ndarray, counts: np.ndarray, fraction: float,
                      seed: int) -> np.ndarray:
    """bool mask: `fraction` of the players of every (rank, game-count tertile within the rank)
    cell. One seed gives nested subsets: a smaller fraction is a subset of a larger one."""
    keep = np.zeros(len(players), dtype=bool)
    rng = np.random.default_rng([seed, 82])
    for rank in sorted(set(ranks.tolist())):
        members = np.flatnonzero(ranks == rank)
        members = members[np.lexsort((players[members], counts[members]))]
        for cell in np.array_split(members, 3):
            order = rng.permutation(cell)
            keep[order[:int(round(fraction * len(cell)))]] = True
    return keep


def training_players(config: dict):
    """(game lists per player, lengths, held-out players, player colour per game or None, info).

    config["sampling"] = {"colors": "both", "min_ply": n} trains on the player's own moves in
    games of either colour (white moves colour-swapped); absent = black games, ply >= MIN_PLY.
    sampling["player_subset"] = {"fraction": f, "seed": s} keeps a stratified share of the
    training players (player-count probe). sampling["gallery_players"] = {"pool": p,
    "gallery_seed": s, "share": x} adds the candidates of bench pool p as training players with
    only their fixed gallery games (cv.pid2_fixed_gallery); round(x * batch players) of every
    group are such players. info = {"players", "gallery_players", "gallery_games"}."""
    meta = data.load_meta()
    gallery, _ = data.pid_splits(meta)
    held = set(gallery.loc[gallery["split"].isin(config["exclude_splits"]), "player_id"])
    sampling = config.get("sampling", {})
    both = sampling.get("colors", "black") == "both"
    keep = ~meta["player_id"].astype(str).isin(held)
    if not both:
        keep &= meta["color"].astype(str) == "B"
    max_handicap = sampling.get("max_handicap")
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
    groups = groups[groups.map(len) >= config["batch"]["games"]]
    subset = sampling.get("player_subset")
    if subset:
        rank_of = games.groupby(games["player_id"].astype(str), observed=True)["rank"].first().astype(str)
        mask = stratified_subset(groups.index.to_numpy().astype(str), rank_of[groups.index].to_numpy(),
                                 groups.map(len).to_numpy(), subset["fraction"], subset["seed"])
        groups = groups[mask]
    player_games = list(groups)
    rank_ids = games.groupby(games["player_id"].astype(str), observed=True)["rank_id"].first()
    info = {"players": [str(p) for p in groups.index], "gallery_players": [], "gallery_games": [],
            "rank_ids": [int(r) for r in rank_ids[groups.index]]}
    extra = sampling.get("gallery_players")
    if extra:
        from go_aicup.data import cv
        bank = pd.read_parquet(ROOT / "data" / "processed" / "identity" / "pid2_bank.parquet")
        candidates, fixed, rest = cv.pid2_fixed_gallery(bank, extra["pool"], extra["gallery_seed"])
        trained = set(np.concatenate(player_games).tolist())
        shown = set(np.concatenate(fixed).tolist())
        if shown & set(np.concatenate(rest).tolist()) or shown & trained:
            raise ValueError("gallery games overlap the question games or the training games")
        if set(candidates.astype(str)) & set(info["players"]):
            raise ValueError("a gallery player is already a training player")
        if min(len(g) for g in fixed) < config["batch"]["games"]:
            raise ValueError("a gallery is smaller than batch.games")
        info["gallery_players"] = [str(p) for p in candidates]
        info["gallery_games"] = [g.tolist() for g in fixed]
    color = meta["player_color"].to_numpy().astype(np.int8) if both else None
    lengths = meta["length"].to_numpy()
    external = sampling.get("external")
    if external:
        # OGS players appended after the Yugen ones (identity/external.py, 2026-11-18); paths.cache_dir must
        # be the merged cache, whose first len(meta) rows are the Yugen cache unchanged.
        if not both:
            raise ValueError("sampling.external needs colors=both")
        cache_dir = Path(config["paths"]["cache_dir"])
        merged = pd.read_parquet(cache_dir / "games.parquet", columns=["game_index", "length", "player_color"])
        n = len(meta)
        if not (np.array_equal(merged["length"].to_numpy()[:n], lengths)
                and np.array_equal(merged["player_color"].to_numpy()[:n], meta["player_color"].to_numpy())):
            raise ValueError("merged cache does not start with the Yugen cache")
        ext = pd.read_parquet(cache_dir / "external_players.parquet")
        if ext["game_index"].min() < n:
            raise ValueError("external rows overlap the Yugen rows")
        ext_groups = ext.groupby("player", sort=True)["game_index"].apply(np.asarray)
        ext_groups = ext_groups[ext_groups.map(len) >= config["batch"]["games"]]
        limit = external.get("max_players")
        if limit is not None and len(ext_groups) > limit:
            pick = np.sort(np.random.default_rng([external.get("seed", 2026), 41]).choice(
                len(ext_groups), size=limit, replace=False))
            ext_groups = ext_groups.iloc[pick]
        info["yugen_players"] = len(player_games)
        info["external_players"] = len(ext_groups)
        player_games = player_games + list(ext_groups)
        info["players"] += [str(p) for p in ext_groups.index]
        info["rank_ids"] += [-1] * len(ext_groups)
        # OGS games run to 980 plies; the E2E head's ply-position table has 512 rows (seqagg.MAX_PLY), and
        # Yugen games stop at 445. Sample OGS plies only from the first max(Yugen length) moves (35 of
        # 800,000 OGS games exceed 511 plies; job 796341 crashed on one). Yugen rows are unchanged.
        lengths = merged["length"].to_numpy().copy()
        lengths[n:] = np.minimum(lengths[n:], int(lengths[:n].max()))
        color = merged["player_color"].to_numpy().astype(np.int8)
    return player_games, lengths, len(held), color, info


def player_probabilities(config: dict, info: dict) -> np.ndarray | None:
    """sampling.external.yugen_share = s: each drawn player is a Yugen player with probability s
    (uniform within each source); absent = uniform over all players."""
    share = config.get("sampling", {}).get("external", {}).get("yugen_share")
    if share is None:
        return None
    n_y, n_e = info["yugen_players"], info["external_players"]
    return np.concatenate([np.full(n_y, share / n_y), np.full(n_e, (1 - share) / n_e)])


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
    player_games, lengths, held, player_color, info = training_players(config)
    extra_games = [np.asarray(g) for g in info["gallery_games"]]
    extra_share = config.get("sampling", {}).get("gallery_players", {}).get("share", 0.0)
    extra_per_group = int(round(extra_share * config["batch"]["players"])) if extra_games else 0
    if rank == 0:       # reproducible list of what this run trains on
        (out / "training_players.json").write_text(json.dumps(
            {**info, "extra_per_group": extra_per_group}, separators=(",", ":")))
    batch, training = config["batch"], config["training"]
    if config["model"].get("pick") and config["eval_splits"]:
        raise ValueError("eval_splits is not supported with a pick encoder; embed with pid2_bench")
    if batch["players"] > len(player_games):
        raise ValueError("not enough training players for one GE2E group")

    raw = IdentityEncoder(**config["model"])
    source = torch.load(config["paths"]["init_model"], map_location="cpu", weights_only=False)
    raw.load_trunk(source["model"])
    raw.to(device)
    head_config = config.get("head")
    head = None
    if head_config:
        # B head in the loop (2026-11-17): same seed on every rank -> identical initial weights;
        # GE2E uses the head's w, b (initialised to the encoder's), so the encoder's are frozen.
        from go_aicup.tasks.identity.seqagg import Head
        if head_config.get("arm", "B") != "B" or raw.pick or batch.get("cross_gpu"):
            raise ValueError("head: only arm B, without pick or cross_gpu")
        torch.manual_seed(int(head_config.get("seed", config["seed"])))
        head = Head(attention=False).to(device)
        with torch.no_grad():
            head.w.copy_(raw.w)
            head.b.copy_(raw.b)
        raw.w.requires_grad_(False)
        raw.b.requires_grad_(False)
        net = EncoderHead(raw, head, config, device, (batch["players"], batch["games"], batch["moves"]))
    else:
        net = raw
    model = DistributedDataParallel(net, device_ids=[device.index]) if world > 1 else net
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **config["optimizer"])
    keep_steps = set(int(k) for k in training.get("keep_steps", []))
    if training.get("schedule", "cosine") == "wsd":
        scheduler, decay_start = wsd_scheduler(optimizer, training["steps"], training["warmup_fraction"],
                                               training["decay_fraction"])
    else:
        scheduler = scheduler_for(optimizer, training["steps"], training["warmup_fraction"],
                                  training["minimum_lr_ratio"])
        decay_start = None
    start = 0
    last = out / "last.pt"
    extend = config["paths"].get("extend_from")
    if extend and not last.exists():
        # extension of a WSD run (2026-11-17): continue from its pre-decay checkpoint with a longer
        # total; weights, head, AdamW moments and the step counter carry over, the LR stays constant
        # until this config's decay_start.
        saved = torch.load(extend, map_location="cpu", weights_only=False)
        if saved["world"] != world or decay_start is None or saved["step"] >= decay_start:
            raise ValueError("extend_from needs the same world size, a WSD schedule and a later decay start")
        raw.load_state_dict(saved["model"])
        if head is not None:
            head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["step"]
    elif last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["world"] != world or saved["config"] != config:
            raise ValueError("resume requires the same config and world size")
        raw.load_state_dict(saved["model"])
        if head is not None:
            head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["step"]
    log_event(out, rank, {"event": "start", "world": world, "training_players": len(player_games),
                          "gallery_players": len(extra_games), "extra_per_group": extra_per_group,
                          "excluded_players": held, "start_step": start, "per_gpu_boards":
                          batch["players"] * batch["games"] * batch["moves"]})

    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))

    def save(step):
        if rank == 0:
            payload = {"model": raw.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "step": step, "world": world, "config": config}
            if head is not None:
                payload["head"] = head.state_dict()
            torch.save(payload, out / "last.pt.tmp")
            os.replace(out / "last.pt.tmp", last)
            if step == decay_start:         # pre-decay state: a later run can extend from here
                torch.save(payload, out / f"stable_step{step}.pt.tmp")
                os.replace(out / f"stable_step{step}.pt.tmp", out / f"stable_step{step}.pt")
            if step in keep_steps:          # candidates for the development checkpoint choice
                kept = out / f"step{step}"
                kept.mkdir(exist_ok=True)
                torch.save({"model": raw.state_dict(), "config": config, "steps": step}, kept / "model.pt")
                if head is not None:
                    torch.save(head_payload(head, step, config["seed"]), kept / "best.pt")
        if distributed():
            dist.barrier()

    cross_gpu = bool(batch.get("cross_gpu", False)) and world > 1
    if head is not None:
        from go_aicup.tasks.identity.train_e2e import PlyBatches
        batches = PlyBatches
    else:
        batches = GE2EBatches
    dataset = batches(config["paths"]["cache_dir"], player_games, lengths, players=batch["players"],
                          games=batch["games"], moves=batch["moves"], steps=training["steps"],
                          seed=config["seed"], rank=rank, start_step=start,
                          max_open_shards=config["loader"]["max_open_shards"], player_color=player_color,
                          min_ply=config.get("sampling", {}).get("min_ply", MIN_PLY),
                          player_d4=config.get("sampling", {}).get("player_d4", False),
                          world=world if cross_gpu else 1, extra_games=extra_games,
                          extra_per_group=extra_per_group, pick=config["model"].get("pick", False),
                          player_rank=np.asarray(info["rank_ids"]),
                          player_p=player_probabilities(config, info),
                          rank_focus=config.get("sampling", {}).get("rank_focus"))
    if rank == 0 and dataset.rank_focus:
        (out / "rank_exposure.json").write_text(json.dumps(dataset.rank_exposure(world)))
    workers = config["loader"]["num_workers"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=True,
                                         prefetch_factor=config["loader"]["prefetch_factor"] if workers else None,
                                         persistent_workers=workers > 0,
                                         multiprocessing_context="spawn" if workers else None)
    model.train()
    step, clock, window = start, time.monotonic(), []
    n, m, s = batch["players"], batch["games"], batch["moves"]
    w_b = (head.w, head.b) if head is not None else (raw.w, raw.b)
    for item in loader:
        if head is not None:
            x, plies = item
            games = model(x.to(device, dtype=torch.float32, non_blocking=True), plies.to(device, non_blocking=True))
        else:
            x = item.to(device, dtype=torch.float32, non_blocking=True)
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
            loss, accuracy = ge2e_loss(torch.cat(pooled), *w_b)
            loss = loss * world
        else:
            loss, accuracy = ge2e_loss(games, *w_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                       training["gradient_clip_norm"])
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
                                  "own_centroid_accuracy": float(stats[1]), "w": float(w_b[0]), "b": float(w_b[1]),
                                  "lr": optimizer.param_groups[0]["lr"],
                                  "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30
                                  if device.type == "cuda" else 0.0,
                                  "mean_step_seconds": float(np.mean(window[-training["log_every"]:]))})
        flag = torch.tensor(int(stop[0]), device=device)
        if distributed():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if step % training["checkpoint_steps"] == 0 or flag.item() or step == decay_start:
            save(step)
        if flag.item():
            log_event(out, rank, {"event": "stopped", "step": step})
            return 1
    save(step)
    del loader

    export_embeddings(raw, config, config["eval_splits"], lengths, device, rank, world)
    if rank == 0:
        torch.save({"model": raw.state_dict(), "config": config, "steps": step}, out / "model.pt")
        if head is not None:
            (out / "head").mkdir(exist_ok=True)
            torch.save(head_payload(head, step, config["seed"]), out / "head" / "best.pt")
    if distributed():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
