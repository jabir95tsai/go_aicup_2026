"""Cross-move aggregation on a frozen identity encoder (2026-11-16): stage 1, sizing probe.

The 9,600-step GE2E encoder maps each of the player's moves to a 256-d vector and a game
vector is their plain mean. The planned probe trains a small head on frozen per-move
vectors (arm B: shared MLP then mean; arm C: 2-layer Transformer with move-order positions
then mean). Before any of that, this stage measures what a per-move vector cache costs:

    probe   1 GPU: (a) per-move vectors of a sample of bench games, averaged per game, must
            reproduce the saved bench game vectors (control A); (b) GPU throughput;
            (c) number of own moves of the training players the encoder was trained on, and
            the projected cache size and extraction time for the whole set and for capped
            games per player. Writes outputs/pid/pid2_seqagg/probe.json; trains nothing.

    extract 1 GPU: per-move vectors (float16) of every training game of the encoder's training
            players and of every bench game, written in chunks of whole games with plain np.save
            (no memmap writes: WekaFS lost memmap writes before) and read back to verify.
            Layout under --cache: <part>/games.npy (game_index), offsets.npy (moves per game,
            cumulative), players.npy (training only: player index per game), chunk_XXX.npy and
            chunks.json. A move's ply is recomputed from (game, colour, min_ply).

    python -m scripts.identity.pid2_seqagg probe --run pid_ge2e_v4_d4_hd_9600
    python -m scripts.identity.pid2_seqagg extract --cache /home/jason1122/pid2_seqagg_cache
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, twocolor
from go_aicup.tasks.identity.ge2e import IdentityEncoder, as_black
from go_aicup.tasks.identity.train_ge2e import training_players
from go_aicup.tasks.strength.shared.runtime import autocast_context

OUT = ROOT / "outputs" / "pid" / "pid2_seqagg"
BENCH = ROOT / "outputs" / "pid" / "pid2_bench"
CAPS = (8, 16, 32, 64, None)


def own_plies(color: np.ndarray, length: np.ndarray, min_ply: int) -> list[np.ndarray]:
    first = min_ply + (color.astype(np.int64) - min_ply) % 2
    return [np.arange(f, n, 2) for f, n in zip(first, length)]


class _Planes(torch.utils.data.Dataset):
    """Item i = assembled input planes of samples [i * batch, (i + 1) * batch), white colour-swapped."""

    def __init__(self, cache_dir, games, plies, colors, batch):
        self.cache_dir, self.games, self.plies, self.colors, self.batch = cache_dir, games, plies, colors, batch
        self._cache = None

    def __len__(self):
        return (len(self.games) + self.batch - 1) // self.batch

    def __getitem__(self, i):
        if self._cache is None:
            self._cache = BoardCache(self.cache_dir)
        span = slice(i * self.batch, (i + 1) * self.batch)
        histories, side = self._cache.history_batch(self.games[span], self.plies[span])
        if not np.array_equal(side, self.colors[span]):
            raise ValueError("sampled ply is not the player's move")
        histories, side = as_black(histories, side)
        return torch.from_numpy(assemble_inputs(histories, side)), i

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


WORKERS = 10


@torch.no_grad()
def move_vectors(model, config, cache, games, colors, lengths, device, min_ply, batch=8192):
    """(float32 [moves, D] per-move vectors in game order then ply order, int64 [games + 1] offsets);
    white moves colour-swapped as in twocolor.embed_games(swap=True)."""
    plies = own_plies(colors, lengths, min_ply)
    sizes = np.array([len(p) for p in plies])
    owner = np.repeat(np.arange(len(games)), sizes)
    sample_game, sample_ply = games[owner], np.concatenate(plies)
    out = np.empty((len(owner), config["model"]["embedding"]), np.float32)
    model.eval()
    dataset = _Planes(cache.directory, sample_game, sample_ply, colors[owner], batch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                                         multiprocessing_context="spawn", pin_memory=True)
    for x, i in loader:
        with autocast_context(config, device):
            vec = model(x.to(device, dtype=torch.float32, non_blocking=True))
        out[i * batch:(i + 1) * batch] = vec.float().cpu().numpy()
    return out, np.concatenate([[0], np.cumsum(sizes)])


def probe(args) -> None:
    started = time.time()
    run_dir = ROOT / "outputs" / "pid" / "ge2e" / args.run
    config = json.loads((run_dir / "config.json").read_text())
    min_ply = config.get("sampling", {}).get("min_ply", 20)
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)["model"])
    device = torch.device("cuda")
    model.to(device)
    meta = data.load_meta()
    colors_all = meta["player_color"].to_numpy().astype(np.int8)
    lengths_all = meta["length"].to_numpy()
    cache = BoardCache(data.CACHE_DIR)
    report = {"run": args.run, "min_ply": min_ply, "precision": config["precision"]}

    # (a) control A: per-move vectors averaged per game reproduce the saved bench vectors
    saved = np.load(BENCH / args.run / "embeddings.npz")
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(saved["game_index"]), size=args.check_games, replace=False))
    games = saved["game_index"][pick]
    vec, off = move_vectors(model, config, cache, games, colors_all[games], lengths_all[games], device, min_ply)
    means = np.add.reduceat(vec.astype(np.float64), off[:-1], axis=0) / np.diff(off)[:, None]
    ref = saved["swap"][pick].astype(np.float64)
    cos = (means * ref).sum(1) / np.linalg.norm(means, axis=1) / np.linalg.norm(ref, axis=1)
    report["control_a"] = {"games": int(len(games)), "moves": int(off[-1]),
                           "max_abs_diff": float(np.abs(means - ref).max()),
                           "max_rel_diff": float((np.linalg.norm(means - ref, axis=1) / np.linalg.norm(ref, axis=1)).max()),
                           "min_cosine": float(cos.min())}
    report["control_a"]["pass"] = bool(report["control_a"]["min_cosine"] > 0.9999)
    print(json.dumps(report["control_a"]), flush=True)

    # (b) throughput on a larger sample of training-player games
    player_games, _, _, _, info = training_players(config)
    flat = np.concatenate(player_games)
    sample = np.sort(rng.choice(flat, size=args.speed_games, replace=False))
    torch.cuda.synchronize()
    t0 = time.time()
    vec, off = move_vectors(model, config, cache, sample, colors_all[sample], lengths_all[sample], device, min_ply)
    torch.cuda.synchronize()
    seconds = time.time() - t0
    report["throughput"] = {"games": int(len(sample)), "moves": int(off[-1]), "seconds": seconds,
                            "moves_per_second": off[-1] / seconds,
                            "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30}
    print(json.dumps(report["throughput"]), flush=True)

    # (c) training-set size, the whole set and capped games per player (random games, fixed seed)
    own = lambda g: ((lengths_all[g] - (min_ply + (colors_all[g].astype(np.int64) - min_ply) % 2)) + 1) // 2
    rate = report["throughput"]["moves_per_second"]
    sizes = {}
    for cap in CAPS:
        moves = 0
        n_games = 0
        for games_p in player_games:
            chosen = games_p if cap is None or len(games_p) <= cap else rng.choice(games_p, size=cap, replace=False)
            moves += int(own(np.asarray(chosen)).sum())
            n_games += len(chosen)
        sizes["all" if cap is None else str(cap)] = {
            "games": n_games, "moves": moves, "cache_gib_fp16": moves * config["model"]["embedding"] * 2 / 2 ** 30,
            "extract_gpu_hours": moves / rate / 3600}
    report["training"] = {"players": len(player_games), "per_cap": sizes}
    bank_moves = int(own(saved["game_index"]).sum())
    report["bench"] = {"games": int(len(saved["game_index"])), "moves": bank_moves,
                       "cache_gib_fp16": bank_moves * config["model"]["embedding"] * 2 / 2 ** 30,
                       "extract_gpu_hours": bank_moves / rate / 3600}
    report["seconds"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def load_encoder(run: str):
    run_dir = ROOT / "outputs" / "pid" / "ge2e" / run
    config = json.loads((run_dir / "config.json").read_text())
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)["model"])
    device = torch.device("cuda")
    return model.to(device), config, device


def write_part(target: Path, model, config, device, cache, games, colors, lengths, min_ply,
               chunk_moves: int, extra: dict, max_chunks: int | None = None) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    done = target / "chunks.json"
    if done.exists():
        return json.loads(done.read_text())
    sizes = np.array([len(p) for p in own_plies(colors[games], lengths[games], min_ply)])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    np.save(target / "games.npy", games)
    np.save(target / "offsets.npy", offsets)
    for name, value in extra.items():
        np.save(target / f"{name}.npy", value)
    chunks, start, t0 = [], 0, time.time()
    while start < len(games):
        stop = int(np.searchsorted(offsets, offsets[start] + chunk_moves, side="right")) - 1
        stop = max(stop, start + 1)
        part = games[start:stop]
        path = target / f"chunk_{len(chunks):03d}.npy"
        moves = int(offsets[stop] - offsets[start])
        try:
            done_before = np.load(path, mmap_mode="r").shape == (moves, config["model"]["embedding"])
        except (OSError, ValueError):            # missing or cut short by a time limit
            done_before = False
        if not done_before and max_chunks is not None and max_chunks <= 0:
            raise SystemExit(f"{target.name}: stopped after the requested number of new chunks")
        if not done_before:                      # chunk boundaries are deterministic, so a rerun resumes
            if max_chunks is not None:
                max_chunks -= 1
            vec, off = move_vectors(model, config, cache, part, colors[part], lengths[part], device, min_ply)
            np.save(path.with_suffix(".tmp.npy"), vec.astype(np.float16))
            path.with_suffix(".tmp.npy").replace(path)
            back = np.load(path, mmap_mode="r")
            if back.shape != vec.shape or not np.array_equal(back[-1], vec[-1].astype(np.float16)):
                raise IOError(f"{path}: read-back mismatch")
        chunks.append({"file": path.name, "first_game": int(start), "games": int(stop - start), "moves": moves})
        print(f"{target.name}: games {stop}/{len(games)}, {time.time() - t0:.0f}s", flush=True)
        start = stop
    record = {"chunks": chunks, "games": int(len(games)), "moves": int(offsets[-1]), "min_ply": min_ply,
              "dtype": "float16"}
    assert sum(c["moves"] for c in chunks) == offsets[-1]
    done.write_text(json.dumps(record, indent=1))
    return record


def extract(args) -> None:
    model, config, device = load_encoder(args.run)
    min_ply = config.get("sampling", {}).get("min_ply", 20)
    meta = data.load_meta()
    colors = meta["player_color"].to_numpy().astype(np.int8)
    lengths = meta["length"].to_numpy()
    cache = BoardCache(data.CACHE_DIR)
    player_games, _, _, _, info = training_players(config)
    games = np.concatenate([np.sort(g) for g in player_games])
    players = np.repeat(np.arange(len(player_games)), [len(g) for g in player_games])
    bench = np.load(BENCH / args.run / "embeddings.npz")["game_index"]
    assert not np.isin(bench, games).any(), "a bench game is a training game"
    out = {"run": args.run}
    if "bench" in args.parts:
        out["bench"] = write_part(args.cache / "bench", model, config, device, cache, bench, colors, lengths,
                                  min_ply, args.chunk_moves, {}, args.max_chunks)
    if "train" in args.parts:
        out["train"] = write_part(args.cache / "train", model, config, device, cache, games, colors, lengths,
                                  min_ply, args.chunk_moves, {"players": players,
                                                              "rank_ids": np.asarray(info["rank_ids"])},
                                  args.max_chunks)
    if set(out) >= {"bench", "train"}:
        (args.cache / "manifest.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if k == "run" else {"games": v["games"], "moves": v["moves"]}) for k, v in out.items()}))




# ---------------------------------------------------------------------------------------------
# stage 2: aggregation heads on the frozen per-move vectors (arms B and C)
# ---------------------------------------------------------------------------------------------
# A: plain mean of the per-move vectors (the current model).
# B: per-move residual feed-forward blocks (pre-LN, 256 -> 512 -> 256, ReLU) x 2, masked mean.
# C: the same blocks with 4-head self-attention over the game's moves before each feed-forward
#    block (nn.TransformerEncoderLayer, norm_first), masked mean. B is C without attention, so
#    C - B isolates the cross-move interaction. Both add a learned embedding of the move's ply.
# Training: GE2E (ge2e.ge2e_loss) on groups of 64 players x 8 games with every valid move of each
# game; the player / game draws depend only on (seed, step), so both arms see the same sequence.

SEED = 2026
HOLDOUT_SHARE = 0.05
MAX_PLY = 512


class Head(torch.nn.Module):
    def __init__(self, attention: bool, dim: int = 256, hidden: int = 512, layers: int = 2, heads: int = 4,
                 position: bool = True):
        super().__init__()
        self.attention = attention
        self.position = torch.nn.Embedding(MAX_PLY, dim) if position else None
        if position:
            torch.nn.init.normal_(self.position.weight, std=0.02)
        if attention:
            self.blocks = torch.nn.ModuleList(torch.nn.TransformerEncoderLayer(
                dim, heads, hidden, dropout=0.0, batch_first=True, norm_first=True) for _ in range(layers))
        else:
            self.blocks = torch.nn.ModuleList(torch.nn.ModuleDict({
                "norm": torch.nn.LayerNorm(dim),
                "ff": torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, dim))})
                for _ in range(layers))
        self.w = torch.nn.Parameter(torch.tensor(10.0))
        self.b = torch.nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor, ply: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [G, L, D] padded per-move vectors, ply [G, L], valid [G, L] bool -> game vectors [G, D]
        (mean over valid moves only)."""
        h = x + self.position(ply) if self.position is not None else x
        for block in self.blocks:
            if self.attention:
                h = block(h, src_key_padding_mask=~valid)
            else:
                h = h + block["ff"](block["norm"](h))
        mask = valid[..., None].to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1)


class MoveStore:
    """All per-move vectors of one cache part in RAM, with per-game offsets and plies."""

    def __init__(self, directory: Path, colors: np.ndarray):
        record = json.loads((directory / "chunks.json").read_text())
        self.games = np.load(directory / "games.npy")
        self.offsets = np.load(directory / "offsets.npy")
        if record["moves"] != self.offsets[-1] or record["games"] != len(self.games):
            raise ValueError(f"{directory}: chunks.json does not match offsets")
        self.vectors = np.empty((int(self.offsets[-1]), 256), np.float16)
        for chunk in record["chunks"]:
            start = self.offsets[chunk["first_game"]]
            part = np.load(directory / chunk["file"])
            if len(part) != chunk["moves"] or start + len(part) != self.offsets[chunk["first_game"] + chunk["games"]]:
                raise ValueError(f"{chunk['file']}: size does not match its games")
            self.vectors[start:start + len(part)] = part
        self.finite = bool(np.isfinite(self.vectors).all())
        self.first_ply = colors[self.games].astype(np.int64) + record["min_ply"]   # min_ply 0 -> colour
        if record["min_ply"] != 0:
            raise ValueError("ply positions assume min_ply 0")
        self.extra = {p.stem: np.load(p) for p in directory.glob("*.npy")
                      if p.stem in ("players", "rank_ids")}

    def batch(self, rows: np.ndarray, device):
        """rows = positions in self.games -> (x [G, L, D] float32, ply [G, L], valid [G, L])."""
        sizes = self.offsets[rows + 1] - self.offsets[rows]
        width = int(sizes.max())
        index = np.concatenate([np.arange(a, b) for a, b in zip(self.offsets[rows], self.offsets[rows + 1])])
        game = np.repeat(np.arange(len(rows)), sizes)
        step = np.arange(len(index)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        x = torch.zeros(len(rows), width, self.vectors.shape[1], device=device)
        x[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = \
            torch.from_numpy(self.vectors[index].astype(np.float32)).to(device)
        ply = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        ply[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = \
            torch.from_numpy(self.first_ply[rows][game] + 2 * step).to(device)
        valid = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        valid[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = True
        return x, ply, valid


def holdout_players(rank_ids: np.ndarray) -> np.ndarray:
    """bool [players]: 5% of the players of every rank (rounded), fixed seed; shared by both arms."""
    rng = np.random.default_rng([SEED, 5])
    out = np.zeros(len(rank_ids), bool)
    for r in np.unique(rank_ids):
        members = np.flatnonzero(rank_ids == r)
        out[rng.choice(members, size=int(round(HOLDOUT_SHARE * len(members))), replace=False)] = True
    return out


def monitor_set(store: MoveStore, held: np.ndarray, draws: int = 3):
    """Fixed monitor questions on the held-out players: per draw every such player with >= 20 games
    gets a 10-game gallery, and 400 questions (k = 1/2/3/5 x 100, player drawn with replacement)
    take unused non-gallery games. -> list of (gallery rows per player, question rows, k, truth)."""
    players = store.extra["players"]
    by_player = {p: np.flatnonzero(players == p) for p in np.flatnonzero(held)}
    eligible = [p for p, rows in by_player.items() if len(rows) >= 20]
    out = []
    for draw in range(draws):
        rng = np.random.default_rng([SEED, 9, draw])
        order = {p: rng.permutation(by_player[p]) for p in eligible}
        used = {p: 10 for p in eligible}
        gallery = [np.sort(order[p][:10]) for p in eligible]
        questions, ks, truth = [], [], []
        for k in rng.permutation(np.repeat([1, 2, 3, 5], 100)):
            while True:
                i = int(rng.integers(len(eligible)))
                if used[eligible[i]] + k <= len(order[eligible[i]]):
                    break
            p = eligible[i]
            questions.append(np.sort(order[p][used[p]:used[p] + k]))
            used[p] += int(k)
            ks.append(int(k))
            truth.append(i)
        rows = np.concatenate(gallery + questions)
        assert len(np.unique(rows)) == len(rows), "a monitor game is used twice"
        out.append((gallery, questions, np.array(ks), np.array(truth)))
    return out, len(eligible)


@torch.no_grad()
def game_vectors(head, store: MoveStore, rows: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    out = np.empty((len(rows), 256), np.float32)
    if head is not None:
        head.eval()
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        x, ply, valid = store.batch(part, device)
        if head is None:
            mask = valid[..., None].float()
            v = (x * mask).sum(1) / mask.sum(1)
        else:
            v = head(x, ply, valid)
        out[s:s + len(part)] = v.float().cpu().numpy()
    if head is not None:
        head.train()
    return out


def monitor_score(head, store, sets, device) -> float:
    """Pure-style official score over the monitor draws (k buckets equally weighted)."""
    from go_aicup.metrics import PID_WEIGHTS
    from go_aicup.tasks.identity import assign
    rows = np.unique(np.concatenate([np.concatenate(g + q) for g, q, _, _ in sets]))
    vec = game_vectors(head, store, rows, device)
    where = {r: i for i, r in enumerate(rows)}
    totals = []
    for gallery, questions, ks, truth in sets:
        cos = twocolor_style([[where[r] for r in g] for g in gallery], [[where[r] for r in q] for q in questions], vec)
        ranked = assign.top5(cos)
        pts = ((ranked == truth[:, None]) * PID_WEIGHTS).sum(axis=1)
        totals.append(np.mean([pts[ks == k].mean() for k in (1, 2, 3, 5)]))
    return float(np.mean(totals))


def twocolor_style(gallery, questions, vec):
    from go_aicup.tasks.identity import twocolor
    return twocolor.style_scores(vec, [np.asarray(g) for g in gallery], [np.asarray(q) for q in questions])


def train(args) -> None:
    from go_aicup.tasks.identity.ge2e import ge2e_loss
    from go_aicup.tasks.strength.shared.training import scheduler_for
    out = OUT / f"head_{args.arm}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    self_check(device)
    meta = data.load_meta()
    started = time.time()
    store = MoveStore(args.cache / "train", meta["player_color"].to_numpy().astype(np.int8))
    if not store.finite:
        raise ValueError("non-finite per-move vectors in the training cache")
    players, rank_ids = store.extra["players"], store.extra["rank_ids"]
    held = holdout_players(rank_ids)
    rows_of = [np.flatnonzero(players == p) for p in range(len(rank_ids))]
    train_players = np.flatnonzero(~held)
    sets, eligible = monitor_set(store, held)
    print(f"loaded in {time.time() - started:.0f}s; {len(train_players)} training players, {int(held.sum())} held out "
          f"({eligible} with >= 20 games in the monitor)", flush=True)

    torch.manual_seed(args.seed)
    head = Head(attention=args.arm == "C", position=not args.no_position).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = scheduler_for(optimizer, args.steps, 0.05, 0.0)
    log = (out / "train.jsonl").open("w")
    monitor_a = monitor_score(None, store, sets, device)
    log.write(json.dumps({"step": 0, "monitor_A": monitor_a}) + "\n")
    print(f"monitor, plain mean (arm A): {monitor_a:.4f}", flush=True)
    best = {"score": -1.0, "step": -1, "monitor_A": monitor_a}
    t0, window = time.time(), []
    for step in range(1, args.steps + 1):
        rng = np.random.default_rng([args.seed, step])
        chosen = rng.choice(train_players, size=64, replace=False)
        rows = np.concatenate([rng.choice(rows_of[p], size=8, replace=False) for p in chosen])
        x, ply, valid = store.batch(rows, device)
        games = head(x, ply, valid).reshape(64, 8, -1)
        loss, accuracy = ge2e_loss(games, head.w, head.b)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == args.timing_steps:
            torch.cuda.synchronize()
            per_step = (time.time() - t0) / step
            projected = per_step * args.steps + (args.steps // 500) * args.monitor_seconds_guess
            print(f"timing: {per_step:.3f}s/step -> projected {projected / 3600:.2f}h for {args.steps} steps", flush=True)
            log.write(json.dumps({"timing_steps": step, "seconds_per_step": per_step, "projected_hours": projected / 3600,
                                  "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30}) + "\n")
            log.flush()
            if projected > args.budget_hours * 3600:
                raise SystemExit(f"projected {projected / 3600:.2f}h exceeds the {args.budget_hours}h cap; stopping")
        if step % 50 == 0:
            log.write(json.dumps({"step": step, "loss": float(loss), "acc": float(accuracy), "w": float(head.w),
                                  "b": float(head.b), "lr": optimizer.param_groups[0]["lr"]}) + "\n")
        if step % 500 == 0:
            score = monitor_score(head, store, sets, device)
            log.write(json.dumps({"step": step, "monitor": score, "elapsed_h": (time.time() - t0) / 3600}) + "\n")
            log.flush()
            print(f"step {step}: loss {float(loss):.4f} monitor {score:.4f}", flush=True)
            if score > best["score"]:          # ties keep the earlier checkpoint
                best.update(score=score, step=step)
                torch.save({"model": head.state_dict(), "step": step, "arm": args.arm, "position": not args.no_position,
                            "seed": args.seed}, out / "best.pt")
    torch.save({"model": head.state_dict(), "step": args.steps, "arm": args.arm, "position": not args.no_position,
                "seed": args.seed}, out / "last.pt")
    (out / "result.json").write_text(json.dumps({"arm": args.arm, "best": best, "steps": args.steps,
                                                 "learned_w": float(head.w), "learned_b": float(head.b),
                                                 "hours": (time.time() - t0) / 3600}, indent=1))
    print(json.dumps(best))


def self_check(device) -> None:
    """Padding must not change a game vector: the same game alone and next to a longer one."""
    torch.manual_seed(0)
    for attention in (False, True):
        head = Head(attention).to(device).eval()
        x = torch.randn(2, 30, 256, device=device)
        ply = torch.arange(30, device=device).repeat(2, 1)
        valid = torch.ones(2, 30, dtype=torch.bool, device=device)
        valid[0, 12:] = False
        x_noise = x.clone()
        x_noise[0, 12:] = 1e3 * torch.randn(18, 256, device=device)
        with torch.no_grad():
            a = head(x, ply, valid)[0]
            b = head(x_noise, ply, valid)[0]
            c = head(x[:1, :12], ply[:1, :12], valid[:1, :12])[0]
        assert torch.allclose(a, b, atol=1e-4) and torch.allclose(a, c, atol=1e-4), f"padding leaks (attention={attention})"


def evaluate(args) -> None:
    """Bench scoring of A (plain mean of the cached per-move vectors), B and C (best monitor
    checkpoints): fused = opening + 16 * w * cosine + category with the saved full-precision w of
    the 9,600-step run; pool 0 screens, pool 1 only if C passes the screen."""
    from go_aicup.data import cv
    from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_transductive import load
    device = torch.device("cuda")
    self_check(device)
    run = "pid_ge2e_v4_d4_hd_9600"
    env = load(run)
    saved = np.load(BENCH / run / "embeddings.npz")
    meta = data.load_meta()
    store = MoveStore(args.cache / "bench", meta["player_color"].to_numpy().astype(np.int8))
    report = {"integrity": {
        "games_match_bench_order": bool(np.array_equal(store.games, saved["game_index"])),
        "finite": store.finite,
        "moves": int(store.offsets[-1]),
        "moves_expected": int(sum(len(p) for p in own_plies(meta["player_color"].to_numpy()[store.games].astype(np.int8),
                                                             meta["length"].to_numpy()[store.games], 0)))}}
    report["integrity"]["moves_match"] = report["integrity"]["moves"] == report["integrity"]["moves_expected"]
    rows = np.arange(len(store.games))
    vectors = {"A": game_vectors(None, store, rows, device)}
    cos = (vectors["A"] * saved["swap"]).sum(1) / np.linalg.norm(vectors["A"], axis=1) / np.linalg.norm(saved["swap"], axis=1)
    report["integrity"]["A_min_cosine_vs_saved"] = float(cos.min())
    for arm in ("B", "C"):
        state = torch.load(OUT / f"head_{arm}" / "best.pt", map_location="cpu")
        head = Head(attention=arm == "C").to(device)
        head.load_state_dict(state["model"])
        vectors[arm] = game_vectors(head, store, rows, device)
        report[f"checkpoint_{arm}"] = int(state["step"])
        report[f"learned_w_{arm}"] = float(head.w)
    w = float(saved["w"])
    report["w"] = w
    failed = [k for k, v in report["integrity"].items() if v is False]
    if failed:
        raise SystemExit(f"integrity failed: {failed}")

    def score_pool(pool: int, n: int) -> pd.DataFrame:
        out = []
        for draw in range(100, 100 + n):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
            query = list(q["games"])
            ks = q["k"].to_numpy()
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], env["summary"]["opening"])
                    + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
            pos = env["position"]
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            for arm, vec in [("saved", saved["swap"])] + list(vectors.items()):
                cosine = twocolor.style_scores(vec, g, qq)
                for what, sc in (("fused", rest + 16.0 * w * cosine), ("style only", cosine)):
                    out.append({"pool": pool, "draw": draw, "arm": arm, "what": what,
                                **bucket_means(points(sc, candidates, q), ks)})
            print(f"pool {pool} draw {draw} done", flush=True)
        return pd.DataFrame(out)

    def summarise(frame, pool, base):
        tables = {}
        for what in ("fused", "style only"):
            part = frame[(frame["pool"] == pool) & (frame["what"] == what)]
            t = paired(part, base)
            wide = part.pivot(index="draw", columns="arm", values=COLS)
            for c in COLS[:-1]:
                t[f"se_{c}"] = [float((wide[c][a] - wide[c][base]).std(ddof=1) / np.sqrt(len(wide))) for a in t["arm"]]
            tables[what] = t
            print(f"\n== pool {pool}, {what}, difference to {base} ==\n{t.round(4).to_string(index=False)}", flush=True)
        return tables

    frame = score_pool(0, 20)
    phase1 = pd.read_csv(ROOT / "outputs/pid/pid2_diag/pid_ge2e_v4_d4_hd_9600/d1_per_draw.csv")

    def reproduce(pool):
        old = phase1[(phase1["pool"] == pool) & (phase1["arm"] == "full")].set_index("draw")[COLS]
        res = {}
        for arm in ("saved", "A"):
            new = frame[(frame["pool"] == pool) & (frame["arm"] == arm) & (frame["what"] == "fused")].set_index("draw")[COLS]
            d = (new - old.loc[new.index]).abs()
            res[arm] = {"max_abs_all": float(d["all"].max()), "max_abs_bucket": float(d.max().max()),
                        "draws_differing": int((d["all"] > 0).sum())}
        return res

    report["reproduce_pool0"] = reproduce(0)
    report["A_gate_pass_pool0"] = bool(report["reproduce_pool0"]["saved"]["max_abs_all"] < 1e-12
                                       and report["reproduce_pool0"]["A"]["max_abs_all"] <= 0.0025)
    t0 = summarise(frame, 0, "A")
    get = lambda t, arm, col: float(t["fused"].set_index("arm").loc[arm, col])
    c_minus_a = get(t0, "C", "d_all")
    b_minus_a = get(t0, "B", "d_all")
    report["pool0"] = {w_: t.to_dict(orient="records") for w_, t in t0.items()}
    report["pool0_screen"] = {"C_minus_A": c_minus_a, "C_minus_B": c_minus_a - b_minus_a,
                              "pass": bool(report["A_gate_pass_pool0"] and c_minus_a >= 0.005 and c_minus_a > b_minus_a)}
    if report["pool0_screen"]["pass"]:
        frame = pd.concat([frame, score_pool(1, 30)], ignore_index=True)
        report["reproduce_pool1"] = reproduce(1)
        t1 = summarise(frame, 1, "A")
        report["pool1"] = {w_: t.to_dict(orient="records") for w_, t in t1.items()}
        f1 = t1["fused"].set_index("arm")
        part = frame[(frame["pool"] == 1) & (frame["what"] == "fused")].pivot(index="draw", columns="arm", values="all")
        cb = part["C"] - part["B"]
        z = f1.loc["C", "d_all"] / f1.loc["C", "se_all"]
        report["pool1_gate"] = {
            "C_minus_A": float(f1.loc["C", "d_all"]), "se": float(f1.loc["C", "se_all"]), "z": float(z),
            "buckets": {k: float(f1.loc["C", f"d_{k}"]) for k in ("1", "2", "3", "5")},
            "C_minus_B": float(cb.mean()), "C_minus_B_se": float(cb.std(ddof=1) / np.sqrt(len(cb)))}
        g = report["pool1_gate"]
        g["pass"] = bool(g["C_minus_A"] >= 0.010 and z >= 2.5 and min(g["buckets"].values()) >= -0.005
                         and g["C_minus_B"] > 0)
    frame.to_csv(OUT / "eval_per_draw.csv", index=False)
    (OUT / "eval.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k not in ("pool0", "pool1")}, indent=1, default=float))


def load_head(path: Path, device):
    state = torch.load(path, map_location="cpu")
    head = Head(attention=state["arm"] == "C", position=state.get("position", True)).to(device)
    head.load_state_dict(state["model"])
    return head, int(state["step"])


def confirm(args) -> None:
    """Sealed draws (first = 500): A and every named head, fused and style-only, paired against A.
    Gate per head named B*: pool 1 delta >= +0.010, delta / se >= 2.5, every k >= -0.005, pool 0 delta > 0."""
    from go_aicup.data import cv
    from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_transductive import load
    device = torch.device("cuda")
    run = "pid_ge2e_v4_d4_hd_9600"
    env = load(run)
    saved = np.load(BENCH / run / "embeddings.npz")
    meta = data.load_meta()
    store = MoveStore(args.cache / "bench", meta["player_color"].to_numpy().astype(np.int8))
    assert np.array_equal(store.games, saved["game_index"]) and store.finite
    rows = np.arange(len(store.games))
    vectors, steps = {"A": game_vectors(None, store, rows, device)}, {}
    for item in args.heads:
        name, directory = item.split("=")
        head, steps[name] = load_head(Path(directory) / "best.pt", device)
        vectors[name] = game_vectors(head, store, rows, device)
    w = float(saved["w"])
    out = []
    for pool, n in ((0, 20), (1, 30)):
        for draw in range(args.first, args.first + n):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
            query = list(q["games"])
            ks = q["k"].to_numpy()
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], env["summary"]["opening"])
                    + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
            pos = env["position"]
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            for name, vec in vectors.items():
                cosine = twocolor.style_scores(vec, g, qq)
                for what, sc in (("fused", rest + 16.0 * w * cosine), ("style only", cosine)):
                    out.append({"pool": pool, "draw": draw, "arm": name, "what": what,
                                **bucket_means(points(sc, candidates, q), ks)})
        print(f"pool {pool} done", flush=True)
    frame = pd.DataFrame(out)
    report = {"first_draw": args.first, "checkpoints": steps, "tables": {}, "gate": {}}
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for pool in (0, 1):
        for what in ("fused", "style only"):
            part = frame[(frame["pool"] == pool) & (frame["what"] == what)]
            t = paired(part, "A")
            wide = part.pivot(index="draw", columns="arm", values=COLS)
            for c in COLS[:-1]:
                t[f"se_{c}"] = [float((wide[c][a] - wide[c]["A"]).std(ddof=1) / np.sqrt(len(wide))) for a in t["arm"]]
            report["tables"][f"pool {pool} | {what}"] = t.to_dict(orient="records")
            print(f"\n== sealed pool {pool}, {what}, difference to A ==\n{t.round(4).to_string(index=False)}")
    f0 = pd.DataFrame(report["tables"]["pool 0 | fused"]).set_index("arm")
    f1 = pd.DataFrame(report["tables"]["pool 1 | fused"]).set_index("arm")
    for name in vectors:
        if name == "A":
            continue
        z = f1.loc[name, "d_all"] / f1.loc[name, "se_all"]
        buckets = {k: float(f1.loc[name, f"d_{k}"]) for k in ("1", "2", "3", "5")}
        report["gate"][name] = {"pool0_delta": float(f0.loc[name, "d_all"]), "pool1_delta": float(f1.loc[name, "d_all"]),
                                "pool1_se": float(f1.loc[name, "se_all"]), "pool1_z": float(z), "pool1_buckets": buckets,
                                "pass": bool(f0.loc[name, "d_all"] > 0 and f1.loc[name, "d_all"] >= 0.010 and z >= 2.5
                                             and min(buckets.values()) >= -0.005)}
    gated = [n for n in report["gate"] if n.startswith("B_") and not n.startswith("B_nopos")]
    report["confirmed"] = bool(gated) and all(report["gate"][n]["pass"] for n in gated)
    b = [report["gate"][n]["pool1_delta"] for n in gated]
    nopos = [report["gate"][n]["pool1_delta"] for n in report["gate"] if n.startswith("B_nopos")]
    report["summary"] = {"B_seeds": gated, "B_mean": float(np.mean(b)), "B_range": [float(min(b)), float(max(b))],
                         "nopos_mean": float(np.mean(nopos)) if nopos else None,
                         "nopos_share_of_B": float(np.mean(nopos) / np.mean(b)) if nopos else None}
    frame.to_csv(OUT / "confirm_per_draw.csv", index=False)
    (OUT / "confirm.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ("checkpoints", "gate", "confirmed", "summary")}, indent=1, default=float))


def export(args) -> None:
    """Official added set: replace the style term of a predict2 output directory by head B.
    A (plain mean of freshly extracted per-move vectors) must first reproduce the saved style
    term and top-1 of that directory; then the style term is rebuilt from the head's game vectors
    with the same w, lambda, mu, opening and category terms."""
    from go_aicup.data.preprocess import load_store
    from go_aicup.tasks.identity import assign
    model, config, device = load_encoder(args.run)
    info = json.loads((args.src / "info.json").read_text())
    summary = info["summary"]
    if summary.get("use_per_k", False) or summary["style"] != "style-swap":
        raise ValueError("expects the fixed-lambda colour-swapped setting")
    head, step = load_head(args.head / "best.pt", device)
    parts = np.load(args.src / "parts.npz")
    vec_a, vec_h = [], []
    for name in (args.pool, args.questions):
        _, _, meta_s = load_store(name, args.store_dir, mmap=False)
        cache_dir = args.src / name / "cache"
        games = pd.read_parquet(cache_dir / "games.parquet")
        if len(games) != len(meta_s):
            raise ValueError(f"{name}: board cache lost games")
        rows = games["store_row"].to_numpy()
        colour = (meta_s["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
        cache = BoardCache(cache_dir)
        index = games["game_index"].to_numpy()
        min_ply = config.get("sampling", {}).get("min_ply", 20)
        if min_ply != 0:
            raise ValueError("ply positions assume min_ply 0")
        vec, off = move_vectors(model, config, cache, index, colour, games["length"].to_numpy(), device, min_ply)
        # per-move rows for the head, in store order
        part = MoveStore.__new__(MoveStore)
        part.vectors = vec.astype(np.float16)
        part.offsets = off
        part.games = index
        part.first_ply = colour.astype(np.int64)
        a = np.empty((len(meta_s), 256), np.float32)
        h = np.empty((len(meta_s), 256), np.float32)
        a[rows] = game_vectors(None, part, np.arange(len(index)), device)
        h[rows] = game_vectors(head, part, np.arange(len(index)), device)
        old = np.load(args.src / f"{name}_{args.run}_swap.npy")
        cos = (a * old).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(old, axis=1)
        print(f"{name}: {len(meta_s)} games, A vs saved min cosine {cos.min():.6f}", flush=True)
        if cos.min() < 0.9999:
            raise ValueError(f"{name}: per-move vectors do not reproduce the saved game vectors")
        vec_a.append(a)
        vec_h.append(h)
    _, _, pool_meta = load_store(args.pool, args.store_dir, mmap=False)
    _, _, q_meta = load_store(args.questions, args.store_dir, mmap=False)
    pool_players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(pool_players)))
    assert np.array_equal(candidates, np.load(args.src / "candidates.npy", allow_pickle=True))
    gallery = [np.flatnonzero(pool_players == c) for c in candidates]
    question_ids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    query = [len(pool_meta) + np.sort(rows_by_q[q]) for q in question_ids]
    ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[question_ids].to_numpy()
    w = summary["w"]
    style_a = w * twocolor.style_scores(np.concatenate(vec_a), gallery, query)
    rest = parts["opening"] + summary["mu"] * parts["category"]
    base = np.load(args.src / "scores.npy")
    old = pd.read_parquet(args.src / "predictions.parquet")
    assert list(old["question_id"]) == question_ids
    a_scores = rest + summary["lambda"] * style_a
    agree = float((np.argmax(a_scores, 1) == np.argmax(base, 1)).mean())
    report = {"source": str(args.src), "head": str(args.head), "checkpoint": step,
              "A_style_max_abs_diff": float(np.abs(style_a - parts["style"]).max()),
              "A_top1_agreement": agree}
    print(json.dumps(report), flush=True)
    if agree < 0.995:
        raise ValueError("A does not reproduce the saved top-1")
    style_h = w * twocolor.style_scores(np.concatenate(vec_h), gallery, query)
    scores = rest + summary["lambda"] * style_h
    ranked = assign.top5(scores)
    table = pd.DataFrame({"question_id": question_ids, "num_games": ks})
    for j in range(5):
        table[f"top{j + 1}"] = candidates[ranked[:, j]]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "predictions.parquet", index=False)
    np.save(args.out / "scores.npy", scores)
    np.save(args.out / "candidates.npy", candidates)
    np.savez(args.out / "parts.npz", opening=parts["opening"], style=style_h, category=parts["category"])
    changed = table["top1"].to_numpy() != old["top1"].to_numpy()
    report.update(questions=len(question_ids), top1_changed=int(changed.sum()),
                  top1_changed_by_k={str(k): int(changed[ks == k].sum()) for k in sorted(set(ks))},
                  distinct_top1=int(table["top1"].nunique()), summary=summary)
    (args.out / "info.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    p.add_argument("--check-games", type=int, default=200)
    p.add_argument("--speed-games", type=int, default=3000)
    p.set_defaults(func=probe)
    e = sub.add_parser("extract")
    e.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--chunk-moves", type=int, default=2_000_000)
    e.add_argument("--parts", nargs="+", default=["bench", "train"])
    e.add_argument("--max-chunks", type=int, default=None, help="stop after this many new chunks (timing)")
    e.set_defaults(func=extract)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=("B", "C"), required=True)
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--steps", type=int, default=20_000)
    t.add_argument("--timing-steps", type=int, default=200)
    t.add_argument("--budget-hours", type=float, default=1.9)
    t.add_argument("--monitor-seconds-guess", type=float, default=30.0)
    t.add_argument("--tag", default="", help="suffix of the output directory")
    t.add_argument("--seed", type=int, default=SEED, help="initialisation and batch draws (holdout / monitor stay fixed)")
    t.add_argument("--no-position", action="store_true")
    t.set_defaults(func=train)
    v = sub.add_parser("eval")
    v.add_argument("--cache", type=Path, required=True)
    v.set_defaults(func=evaluate)
    c = sub.add_parser("confirm")
    c.add_argument("--cache", type=Path, required=True)
    c.add_argument("--heads", nargs="+", required=True, help="name=output directory with best.pt")
    c.add_argument("--first", type=int, default=500)
    c.set_defaults(func=confirm)
    x = sub.add_parser("export")
    x.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    x.add_argument("--head", type=Path, required=True)
    x.add_argument("--src", type=Path, required=True)
    x.add_argument("--out", type=Path, required=True)
    x.add_argument("--store-dir", type=Path, required=True)
    x.add_argument("--pool", required=True)
    x.add_argument("--questions", required=True)
    x.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
