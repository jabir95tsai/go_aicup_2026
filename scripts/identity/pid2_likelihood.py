"""Player-conditioned move likelihood, stage 0 (2026-11-20; plan PID_PLAYER_LIKELIHOOD_PLAN.md,
registered in RESULTS.md before the run).

Base policy: frozen HumanSL with the metadata replaced by the C v2 Yugen rank vectors (own side to
move) for ranks C, A, 2D, 5D; b(m) = log-mean-exp over the four. Every own move of a bench /
population game keeps the top 32 points of b plus an "other" option (the remaining mass).
Each candidate c gets a log-linear residual over those 33 options,
    log p_c(m | s) = b(m) + theta_c . phi(m, s) - logsumexp_options(...)
with phi = rank (3), consensus bucket (4), distance to the opponent's / own last move (4 + 4),
line (4), and consensus / line x (ply >= 100) (8) = 27 dims. theta_c is the MAP under
N(theta_pop, sigma^2 I) on the candidate's 10 reference games of the draw; theta_pop is fitted on
600 population players (outside the bench) x 10 games. Question score LLR(q, c) = sum over the
question's own moves of log p_c - log p_pop; fused as S + lambda * LLR.

  python -m scripts.identity.pid2_likelihood games                       # CPU: game table
  python -m scripts.identity.pid2_likelihood policy --shard S --num-shards 8 [--limit N]   # GPU
  python -m scripts.identity.pid2_likelihood evaluate                    # GPU: selection + sealed
  python -m scripts.identity.pid2_likelihood selfcheck                   # CPU
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT

DATA = ROOT / "data" / "processed" / "identity" / "likelihood"
OUT = ROOT / "outputs" / "pid" / "likelihood"
GAMES = DATA / "games.parquet"
RICH = DATA / "rich"
VECTORS = ROOT / "data" / "processed" / "strength_features" / "humansl" / "cvec_v2" / "learn" / "vectors.pt"
RANK_IDX = (1, 3, 5, 8)            # C, A, 2D, 5D in D..6D = 0..9
TOP = 32
OPTIONS = TOP + 1
BATCH = 1024                       # positions per compiled forward (x 4 conditions)
POP_PLAYERS, POP_GAMES, POP_SEED = 600, 10, 2026
SIGMAS = (0.1, 0.3, 1.0, 3.0)
LAMBDAS = (0.25, 0.5, 1.0, 2.0)
DEV = [(0, d) for d in range(2600, 2620)]
SEALED = [(0, d) for d in range(2700, 2720)] + [(1, d) for d in range(2700, 2730)]
CHECK = (1, 2000)
LATE_PLY = 100
WEIGHTS = {"B": 0.25, "A1": 0.375, "A2": 0.375}
BASE_M = 1.5


# ---------------------------------------------------------------- game table
def handicap_of(meta: pd.DataFrame) -> np.ndarray:
    from scripts.identity.pid2_paired import HANDICAP
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    return meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                              validate="1:1")["handicap"].to_numpy().astype(np.int64)


def games(_args) -> None:
    from go_aicup.tasks.identity import data
    from scripts.identity.pid2_paired import BANK
    meta = data.load_meta()
    hd = handicap_of(meta)
    bank = pd.read_parquet(BANK)
    bench = np.sort(bank["game"].to_numpy())
    assert (hd[bench] <= 1).all(), "bench games are HD <= 1"
    outside = ~meta["player_id"].astype(str).isin(bank["player_id"].astype(str).unique()).to_numpy()
    ok = outside & (hd <= 1)
    counts = meta[ok].groupby("player_id").size()
    eligible = np.sort(counts[counts >= POP_GAMES].index.astype(str).to_numpy())
    rng = np.random.default_rng(POP_SEED)
    chosen = rng.choice(eligible, size=POP_PLAYERS, replace=False)
    pop = []
    pid = meta["player_id"].astype(str).to_numpy()
    for p in np.sort(chosen):
        own = np.flatnonzero(ok & (pid == p))
        pop.append(np.sort(rng.choice(own, size=POP_GAMES, replace=False)))
    pop = np.concatenate(pop)
    assert not np.isin(pop, bench).any()
    rows = np.concatenate([bench, pop])
    table = pd.DataFrame({"game_index": rows, "role": ["bench"] * len(bench) + ["pop"] * len(pop),
                          "player_id": pid[rows], "player_color": meta["player_color"].to_numpy()[rows].astype(np.int8),
                          "hd": hd[rows].astype(np.int8), "length": meta["length"].to_numpy()[rows]})
    DATA.mkdir(parents=True, exist_ok=True)
    table.to_parquet(GAMES)
    print(json.dumps({"bench_games": int(len(bench)), "pop_games": int(len(pop)), "pop_players": POP_PLAYERS,
                      "eligible_pop_players": int(len(eligible)),
                      "hd1_share": float((table["hd"] == 1).mean()),
                      "own_positions_estimate": int((table["length"] // 2).sum())}))


# ---------------------------------------------------------------- GPU stage 1: policy top-32
def extract_positions(groups, workers: int) -> dict:
    """groups = [(hd, items, ids)], items = [(moves, colors, keep)], ids = int id per item. Returns the
    top-32 arrays (game id, ply, top, lp, slot) of every kept ply, plus timing and the path check."""
    import torch

    from go_aicup.features.humansl_rankvec import game_logp, load_frozen
    from go_aicup.features.humansl_torch import _init_worker, encode_game

    device = torch.device("cuda")
    model, _ = load_frozen(device)
    vectors = torch.load(VECTORS, map_location=device)["vectors"]                  # [10, 2, 384]
    meta4 = vectors[list(RANK_IDX), 0].float()                                      # own side to move
    meta_batch = meta4.repeat(BATCH, 1)                                             # row b * 4 + r
    forward = torch.compile(model, dynamic=False)
    parts = {k: [] for k in ("game", "ply", "top", "lp", "slot")}
    check = {"max_abs_diff": 0.0, "positions": 0}
    diffs: list[np.ndarray] = []
    near: list[np.ndarray] = []
    steady = {}

    def verify():
        """own-condition agreement within compiled-vs-eager TF32 noise, and far below the gap to the
        neighbouring rank condition (a wrong rank vector or mask would land there)."""
        check["p99_abs_diff"] = float(np.quantile(np.concatenate(diffs), 0.99))
        check["neighbour_p99_abs_diff"] = float(np.quantile(np.concatenate(near), 0.99))
        print(json.dumps({"path_check": check}), flush=True)
        if (check["max_abs_diff"] > 1e-2 or check["p99_abs_diff"] > 3e-3
                or check["neighbour_p99_abs_diff"] < 10 * check["p99_abs_diff"]):
            raise SystemExit(f"path check failed: {check}")

    def run(bins, globals_, played, game, ply):
        n = len(played)
        if n < BATCH:
            pad = BATCH - n
            bins, globals_, played = (np.concatenate([a, np.repeat(a[-1:], pad, axis=0)])
                                      for a in (bins, globals_, played))
        with torch.no_grad():
            x = torch.as_tensor(bins, device=device).float()
            g = torch.as_tensor(globals_, device=device)
            logits = forward(x.repeat_interleave(4, 0), g.repeat_interleave(4, 0),
                             input_meta=meta_batch)[0][0][:, 0, :].float()
            legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).reshape(BATCH, 361) > 0
            legal = torch.cat([legal, torch.ones_like(legal[:, :1])], dim=1)
            logp = torch.log_softmax(logits.reshape(BATCH, 4, 362).masked_fill(~legal[:, None], float("-inf")), -1)
            base = torch.logsumexp(logp, dim=1) - math.log(4.0)
            top = base.topk(TOP, dim=1).indices                                    # [B, 32]
            lp_top = logp.gather(2, top[:, None, :].expand(BATCH, 4, TOP))
            other = torch.log1p(-torch.exp(torch.logsumexp(lp_top, dim=2)).clamp(max=1 - 1e-6))
            lp = torch.cat([lp_top, other[..., None]], dim=2)                       # [B, 4, 33]
            target = torch.as_tensor(played, device=device)
            hit = top == target[:, None]
            slot = torch.where(hit.any(1), hit.float().argmax(1), torch.full_like(target, TOP))
            if check["positions"] < 2 * BATCH:      # path check against humansl_rankvec.game_logp
                own = np.ones(BATCH, dtype=bool)
                full = game_logp(model, vectors, bins, globals_, played, own, batch=BATCH * 10)
                mine = logp.gather(2, target[:, None, None].expand(BATCH, 4, 1))[:, :, 0]
                diffs.append((full[:, list(RANK_IDX)] - mine)[:n].abs().flatten().cpu().numpy())
                near.append((full[:, [r - 1 for r in RANK_IDX]] - mine)[:n].abs().flatten().cpu().numpy())
                check["max_abs_diff"] = max(check["max_abs_diff"], float(diffs[-1].max()))
                check["positions"] += n
                if check["positions"] >= 2 * BATCH:
                    verify()
        parts["game"].append(game[:n].astype(np.int32))
        parts["ply"].append(ply[:n].astype(np.int16))
        parts["top"].append(top[:n].cpu().numpy().astype(np.int16))
        parts["lp"].append(lp[:n].cpu().numpy().astype(np.float16))
        parts["slot"].append(slot[:n].cpu().numpy().astype(np.int8))

    started = last = time.monotonic()
    positions = 0
    for hd, items, ids in groups:
        if not len(items):
            continue
        komi = None if hd == 0 else 0.5
        pending = []
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker,
                                          initargs=(model.config, komi)) as pool:
            for i, (bins, globals_, played) in enumerate(pool.imap(encode_game, items, chunksize=4)):
                moves, colors, keep = items[i]
                plies = np.flatnonzero(keep)
                if not np.array_equal(played, moves[plies].astype(np.int64)):
                    raise ValueError(f"game {ids[i]}: tensor positions differ from moves")
                pending.append((bins, globals_, played, np.full(len(plies), ids[i]), plies))
                positions += len(plies)
                while sum(len(p[2]) for p in pending) >= BATCH:
                    cat = [np.concatenate(a) for a in zip(*pending)]
                    run(*(a[:BATCH] for a in cat))
                    rest = [a[BATCH:] for a in cat]
                    pending = [tuple(rest)] if len(rest[2]) else []
                now = time.monotonic()
                if not steady and now - started >= 90:      # after compile and pool start-up
                    steady.update(t=now, n=positions)
                if now - last >= 30:
                    print(json.dumps({"hd": hd, "games": i + 1, "of": len(items),
                                      "positions_per_second": round(positions / (now - started), 1)}), flush=True)
                    last = now
        if pending:
            cat = [np.concatenate(a) for a in zip(*pending)]
            for s in range(0, len(cat[2]), BATCH):
                run(*(a[s:s + BATCH] for a in cat))
    seconds = time.monotonic() - started
    result = {name: np.concatenate(value) for name, value in parts.items()}
    meta = {"positions": int(positions), "seconds": round(seconds, 1),
            "positions_per_second": round(positions / seconds, 1), "path_check": check,
            "rank_idx": RANK_IDX, "top": TOP}
    if steady:
        meta["steady_positions_per_second"] = round((positions - steady["n"]) / (time.monotonic() - steady["t"]), 1)
    print(json.dumps(meta), flush=True)
    return result, meta


def policy(args) -> None:
    from go_aicup.features.humansl import CACHE, cache_games

    out = DATA / ("smoke" if args.limit else f"shard_{args.shard:02d}")
    if (out / "meta.json").exists() and not args.limit:
        print(f"{out} done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(GAMES).iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    groups = []
    for hd in (0, 1):
        sub = table[table["hd"] == hd]
        played_games = cache_games(sub["game_index"], CACHE) if len(sub) else []
        items = [(m, c, c == pc) for (m, c), pc in zip(played_games, sub["player_color"])]
        groups.append((hd, items, sub["game_index"].to_numpy()))
    result, meta = extract_positions(groups, args.workers)
    for name, value in result.items():
        np.save(out / f"{name}.npy", value)
    meta["games"] = int(len(table))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- features
CONS_EDGES = (1, 3, 10, 32)        # option index j: 0 | 1-2 | 3-9 | 10-31 | other (32)
DIM = 27


def _bucket_dist(d: np.ndarray) -> np.ndarray:
    """Chebyshev distance -> one-hot over 1 / 2 / 3 / 4-6 (reference: > 6 or undefined)."""
    out = np.zeros(d.shape + (4,), np.float32)
    for i, (lo, hi) in enumerate(((1, 1), (2, 2), (3, 3), (4, 6))):
        out[..., i] = (d >= lo) & (d <= hi)
    return out


DIM_RICH = 50
GROUPS = {"G1": list(range(27, 29)), "G2": list(range(29, 37)), "G3": list(range(37, 47)), "G4": list(range(47, 50))}


def features(top: np.ndarray, lp: np.ndarray, ply: np.ndarray, last_opp: np.ndarray, last_own: np.ndarray,
             rich: np.ndarray | None = None):
    """top int [N, 32], lp float [N, 4, 33] (conditions C, A, 2D, 5D), ply [N], last moves [N]
    (-1 = none) -> (b [N, 33] float32, phi [N, 33, 27] float16). With rich (uint8 [N, 32, 8] from
    likelihood_rich) phi has 50 dims: the 27 above, then G1 log b and log b x late (27-28), G2 absolute
    3x3 zone without the centre (29-36), G3 contact / solid / diagonal / jump / knight and x late (37-46),
    G4 capture / atari / rescue (47-49)."""
    n = len(top)
    lp = lp.astype(np.float32)
    b = np.logaddexp.reduce(lp, axis=1) - math.log(4.0)
    phi = np.zeros((n, OPTIONS, DIM), np.float32)
    phi[:, :, 0:3] = (lp[:, [0, 1, 3], :] - b[:, None, :]).transpose(0, 2, 1)
    j = np.arange(OPTIONS)
    cons = np.zeros((OPTIONS, 4), np.float32)              # reference: the top option
    cons[(j >= 1) & (j < 3), 0] = 1
    cons[(j >= 3) & (j < 10), 1] = 1
    cons[(j >= 10) & (j < 32), 2] = 1
    cons[j == 32, 3] = 1
    phi[:, :, 3:7] = cons[None]
    point = top.astype(np.int64)
    on_board = point < 361
    r, c = point // 19, point % 19
    for col, last in ((7, last_opp), (11, last_own)):
        lr, lc = last // 19, last % 19
        d = np.maximum(np.abs(r - lr[:, None]), np.abs(c - lc[:, None]))
        d = np.where(on_board & (last[:, None] >= 0) & (last[:, None] < 361), d, 99)
        phi[:, :TOP, col:col + 4] = _bucket_dist(d)
    line = np.minimum.reduce([r, c, 18 - r, 18 - c]) + 1
    line = np.where(on_board, line, 99)
    for i in range(4):                                       # lines 1 / 2 / 3 / 4 (reference >= 5)
        phi[:, :TOP, 15 + i] = line == i + 1
    late = (ply >= LATE_PLY).astype(np.float32)[:, None, None]
    phi[:, :, 19:23] = phi[:, :, 3:7] * late
    phi[:, :, 23:27] = phi[:, :, 15:19] * late
    if rich is None:
        return b.astype(np.float32), phi.astype(np.float16)
    assert rich.shape == (n, TOP, 8), rich.shape
    ext = np.zeros((n, OPTIONS, DIM_RICH - DIM), np.float32)
    ext[:, :, 0] = b
    ext[:, :, 1] = b * late[:, :, 0]
    zone = np.where(r <= 5, 0, np.where(r >= 13, 2, 1)) * 3 + np.where(c <= 5, 0, np.where(c >= 13, 2, 1))
    for i, z in enumerate((0, 1, 2, 3, 5, 6, 7, 8)):
        ext[:, :TOP, 2 + i] = on_board & (zone == z)
    shape = rich[:, :, :5].astype(np.float32)
    ext[:, :TOP, 10:15] = shape
    ext[:, :TOP, 15:20] = shape * late
    ext[:, :TOP, 20:23] = rich[:, :, 5:8]
    return b.astype(np.float32), np.concatenate([phi, ext], axis=2).astype(np.float16)


# ---------------------------------------------------------------- MAP fit and scores (torch)
def fit(phi, b, slot, pos, owner, n_owner, prior, sigma, iters=25, chunk=40_000, max_step=1.0, mask=None,
        precision=None):
    """batched Newton MAP of theta [n_owner, D] for positions pos (owner[i] = owner of pos[i]);
    mask [D] zeroes features (their theta stays at the prior). precision [D, D] (float64) replaces
    the isotropic prior I / sigma^2 by N(prior, precision^-1) (PID_EB_COMBINER_PLAN.md)."""
    import torch
    device = phi.device
    d = phi.shape[2]
    theta = prior[None].repeat(n_owner, 1).clone()
    eye = torch.eye(d, device=device, dtype=torch.float64)
    for _ in range(iters):
        grad = torch.zeros(n_owner, d, device=device, dtype=torch.float64)
        hess = torch.zeros(n_owner, d, d, device=device, dtype=torch.float64)
        for s in range(0, len(pos), chunk):
            p, o = pos[s:s + chunk], owner[s:s + chunk]
            f = phi[p].float() if mask is None else phi[p].float() * mask
            z = b[p] + torch.einsum("pjd,pd->pj", f, theta[o].float())
            pi = torch.softmax(z, dim=1)
            mean = torch.einsum("pj,pjd->pd", pi, f)
            g = f[torch.arange(len(p), device=device), slot[p]] - mean
            h = torch.einsum("pj,pjd,pje->pde", pi, f, f) - mean[:, :, None] * mean[:, None, :]
            grad.index_add_(0, o, g.double())
            hess.index_add_(0, o, h.double())
        if precision is None:
            grad -= (theta - prior[None]) / sigma ** 2
            hess += eye / sigma ** 2
        else:
            grad -= (theta - prior[None]) @ precision
            hess += precision[None]
        step = torch.linalg.solve(hess, grad[..., None])[..., 0]
        size = step.abs().max(dim=1, keepdim=True).values
        step = step * torch.clamp(max_step / size.clamp(min=1e-12), max=1.0)   # damped Newton
        theta = theta + step
        if float(step.abs().max()) < 1e-5:
            break
    return theta


def played_logp(phi, b, slot, pos, theta, chunk=2_000, mask=None):
    """[len(pos), n_theta] log p(played option) under every theta row."""
    import torch
    device = phi.device
    out = torch.empty(len(pos), len(theta), device=device, dtype=torch.float64)
    t = theta.float()
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        f = phi[p].float() if mask is None else phi[p].float() * mask
        z = b[p][:, :, None] + torch.einsum("pjd,cd->pjc", f, t)              # [p, 33, C]
        lse = torch.logsumexp(z, dim=1)
        pick = z[torch.arange(len(p), device=device), slot[p]]
        out[s:s + chunk] = (pick - lse).double()
    return out


# ---------------------------------------------------------------- evaluation
def category_prior(codes: np.ndarray, gallery_games, n_codes: int = 4) -> np.ndarray:
    prior = np.bincount(codes[np.concatenate(gallery_games)], minlength=n_codes) + 1.0
    return prior / prior.sum()


def category_scores_fixed(codes, gallery_games, query_games, n_codes: int = 4, prior=None, smoothing: float = 1.0):
    """= twocolor.category_scores, with the pooled prior optionally given (then every column depends
    only on its own candidate's gallery)."""
    prior = category_prior(codes, gallery_games, n_codes) if prior is None else prior
    counts = np.stack([np.bincount(codes[g], minlength=n_codes) for g in gallery_games])
    logp = np.log((counts + smoothing * prior) / (counts.sum(axis=1, keepdims=True) + smoothing))
    return np.stack([logp[:, codes[q]].sum(axis=1) for q in query_games])


def _context(theta_fixed=None, rich: bool = False) -> dict:
    """shared evaluation setup: bench positions and features, theta_pop, base fusion, draw and LLR.
    rich=True builds the 50-dim features (likelihood_rich, PID_LIKELIHOOD_RICH_PLAN.md)."""
    import torch

    from go_aicup.data import cv
    from go_aicup.features.humansl import CACHE, cache_games
    from go_aicup.tasks.identity import data, twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_cheap import (COLS, buckets, cosines, normalise, official, paired,
                                             style, verdict)
    from scripts.identity.pid2_diag_d123a import bucket_means
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
    from scripts.identity.pid2_ogs_a2_eval import ENC, OUT as A2OUT, RUN
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    device = torch.device("cuda")
    # ---- positions
    shards = sorted(p for p in DATA.glob("shard_*") if (p / "meta.json").exists())
    assert len(shards) == 8, shards
    arr = {k: np.concatenate([np.load(s / f"{k}.npy") for s in shards]) for k in ("game", "ply", "top", "lp", "slot")}
    order = np.lexsort((arr["ply"], arr["game"]))
    arr = {k: v[order] for k, v in arr.items()}
    table = pd.read_parquet(GAMES)
    assert np.array_equal(np.unique(arr["game"]), np.sort(table["game_index"].to_numpy())), "missing games"
    played_games = dict(zip(table["game_index"], cache_games(table["game_index"], CACHE)))
    last_opp = np.full(len(order), -1, np.int64)
    last_own = np.full(len(order), -1, np.int64)
    starts = np.flatnonzero(np.r_[True, arr["game"][1:] != arr["game"][:-1]])
    ends = np.r_[starts[1:], len(order)]
    first_pos = np.full(int(arr["game"].max()) + 1, -1, np.int64)
    end_pos = np.full_like(first_pos, -1)
    for s, e in zip(starts, ends):
        g = int(arr["game"][s])
        moves = played_games[g][0].astype(np.int64)
        ply = arr["ply"][s:e].astype(np.int64)
        last_opp[s:e] = np.where(ply >= 1, moves[np.maximum(ply - 1, 0)], -1)
        last_own[s:e] = np.where(ply >= 2, moves[np.maximum(ply - 2, 0)], -1)
        first_pos[g], end_pos[g] = s, e
    rich_arr = None
    if rich:
        rf = np.load(RICH / "bench.npz")
        assert np.array_equal(rf["game"], arr["game"]) and np.array_equal(rf["ply"], arr["ply"]), "rich order"
        rich_arr = rf["rich"]
    b_all, phi_all = [], []
    for s in range(0, len(order), 500_000):
        b, phi = features(arr["top"][s:s + 500_000], arr["lp"][s:s + 500_000], arr["ply"][s:s + 500_000],
                          last_opp[s:s + 500_000], last_own[s:s + 500_000],
                          None if rich_arr is None else rich_arr[s:s + 500_000])
        b_all.append(torch.from_numpy(b))
        phi_all.append(torch.from_numpy(phi))
    B = torch.cat(b_all).to(device)
    PHI = torch.cat(phi_all).to(device)
    SLOT = torch.from_numpy(arr["slot"].astype(np.int64)).to(device)
    report["positions"] = int(len(order))
    report["other_share"] = float((arr["slot"] == TOP).mean())
    print(f"positions {len(order)}, played outside top-32 {report['other_share']:.4f}", flush=True)

    def positions_of(game_lists):
        """game lists -> (position index tensor, owner tensor)."""
        pos, own = [], []
        for i, gl in enumerate(game_lists):
            for g in gl:
                s, e = first_pos[int(g)], end_pos[int(g)]
                assert s >= 0, g
                pos.append(np.arange(s, e))
                own.append(np.full(e - s, i))
        return (torch.from_numpy(np.concatenate(pos)).to(device), torch.from_numpy(np.concatenate(own)).to(device))

    # ---- population theta
    pop = table[table["role"] == "pop"]
    pos_p, own_p = positions_of([pop["game_index"].to_numpy()])
    dims = PHI.shape[2]
    zero = torch.zeros(dims, device=device, dtype=torch.float64)
    base_mask = torch.zeros(dims, device=device)
    base_mask[:DIM] = 1

    def fit_pop(mask):
        return fit(PHI, B, SLOT, pos_p, own_p, 1, zero, 10.0, mask=mask)[0] * (mask > 0)

    theta_pop = fit_pop(base_mask)
    if theta_fixed is not None and len(theta_fixed) < dims:
        theta_fixed = list(theta_fixed) + [0.0] * (dims - len(theta_fixed))
    if theta_fixed is not None:            # later rounds reuse the stage-0 value (as the export does)
        fixed = torch.tensor(theta_fixed, device=device, dtype=torch.float64)
        report["theta_pop_refit_max_diff"] = float((theta_pop - fixed).abs().max())
        theta_pop = fixed
    report["theta_pop"] = theta_pop.cpu().tolist()
    print("theta_pop", np.round(theta_pop.cpu().numpy(), 3).tolist(), flush=True)

    # ---- base fusion (as pid2_cheap / ogsa12w)
    env = load(RUN)
    color, posv = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    unit = {}
    for name, (cache, head_file) in ENC.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, _ = load_head(Path(head_file), device)
        unit[name] = normalise(game_vectors(head, store, np.arange(len(store.games)), device).astype(np.float64))
        del store
    base_sets = [(WEIGHTS[n], unit[n]) for n in WEIGHTS]
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda gm: b1["keys"][row[gm]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    poptab = PopTable(popf["keys"], color[popf["game_index"]])

    def draw(pool, d, with_parts=False):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        g, qq = [posv[x] for x in gallery], [posv[x] for x in query]
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4)
                + s1["lambda1"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L1")[s1["beta"]]
                + s1["lambda2"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L2")[s1["beta"]])
        styl = style(base_sets, g, qq)
        owner = np.searchsorted(candidates, q["player_id"].to_numpy())
        assert (candidates[owner] == q["player_id"].to_numpy()).all()
        out = dict(candidates=candidates, q=q, owner=owner, ks=q["k"].to_numpy(), gallery=gallery,
                   query=query, rest=rest, style=styl, base=rest + scale * BASE_M * styl)
        if with_parts:                     # fusion components in base units (PID_EB_COMBINER_PLAN.md plan 2)
            parts = {"opening": twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                                        env["populations"], env["summary"]["opening"]),
                     "category": twocolor.category_scores(env["codes"], gallery, query, 4),
                     "L1": s1["lambda1"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L1")[s1["beta"]],
                     "L2": s1["lambda2"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L2")[s1["beta"]]}
            for n in WEIGHTS:                  # each encoder's term at style multiplier 1.5
                parts[n] = scale * BASE_M * WEIGHTS[n] * cosines(unit[n], g, qq)
            assert np.allclose(sum(parts[c] for c in ("opening", "category", "L1", "L2")), rest)
            assert np.allclose(sum(parts[n] for n in WEIGHTS), scale * BASE_M * styl)
            out["parts"] = parts
        return out

    def components(gallery, query, cat_prior=None):
        """the 7 non-LLR fusion parts in base units for arbitrary gallery / query game lists
        (pseudo-label expansion, PID_PSEUDO_COLOUR_PLAN.md plan 1); cat_prior fixes the pooled
        category histogram so that columns do not depend on which candidates are passed."""
        g, qq = [posv[x] for x in gallery], [posv[x] for x in query]
        parts = {"opening": twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                                    env["populations"], env["summary"]["opening"]),
                 "category": category_scores_fixed(env["codes"], gallery, query, 4, cat_prior),
                 "L1": s1["lambda1"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L1")[s1["beta"]],
                 "L2": s1["lambda2"] * level_scores(keys1, color, poptab, gallery, query, s1["n"], "L2")[s1["beta"]]}
        for n in WEIGHTS:
            parts[n] = scale * BASE_M * WEIGHTS[n] * cosines(unit[n], g, qq)
        return parts

    x = draw(*CHECK)
    got = bucket_means(points(x["base"], x["candidates"], x["q"]), x["ks"])
    ref = pd.read_csv(A2OUT / "sealed_per_draw.csv")
    ref = ref[(ref["pool"] == CHECK[0]) & (ref["draw"] == CHECK[1]) & (ref["arm"] == "A12w")].iloc[0]
    report["reproduction_max_diff"] = max(abs(got[c] - ref[c]) for c in got)
    print(f"reproduction max diff {report['reproduction_max_diff']:.2e}", flush=True)
    assert report["reproduction_max_diff"] <= 1e-9

    def llr(x, sigma, mask=None, prior=None, precision=None):
        """[Q, C] LLR with candidate thetas fitted on the draw's reference games; mask zeroes the
        features outside the kept dims; prior = that mask's population theta (default theta_pop);
        precision = full prior precision (replaces sigma)."""
        if mask is None and dims > DIM:
            mask = base_mask
        prior = theta_pop if prior is None else prior
        pos_g, own_g = positions_of(x["gallery"])
        th = fit(PHI, B, SLOT, pos_g, own_g, len(x["gallery"]), prior, sigma, mask=mask, precision=precision)
        pos_q, own_q = positions_of(x["query"])
        lp = played_logp(PHI, B, SLOT, pos_q, torch.cat([th, prior[None]]), mask=mask)
        diff = lp[:, :-1] - lp[:, -1:]
        out = torch.zeros(len(x["query"]), len(x["gallery"]), device=device, dtype=torch.float64)
        out.index_add_(0, own_q, diff)
        return out.cpu().numpy(), th

    # engineering: theta = theta_pop gives LLR 0
    pos_q, _ = positions_of(x["query"][:5])
    lp = played_logp(PHI, B, SLOT, pos_q, torch.stack([theta_pop, theta_pop]))
    report["zero_check"] = float((lp[:, 0] - lp[:, 1]).abs().max())
    assert report["zero_check"] < 1e-9

    return dict(report=report, device=device, draw=draw, llr=llr, theta_pop=theta_pop, scale=scale,
                env=env, PHI=PHI, B=B, SLOT=SLOT, positions_of=positions_of, fit_pop=fit_pop,
                base_mask=base_mask, components=components, codes=env["codes"])


def evaluate(args) -> None:
    import torch

    from go_aicup.data import cv
    from go_aicup.features.humansl import CACHE, cache_games
    from go_aicup.tasks.identity import data, twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_cheap import (COLS, buckets, cosines, normalise, official, paired,
                                             style, verdict)
    from scripts.identity.pid2_diag_d123a import bucket_means
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
    from scripts.identity.pid2_ogs_a2_eval import ENC, OUT as A2OUT, RUN
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _context()
    report, device, draw, llr, theta_pop, scale = (ctx[k] for k in ("report", "device", "draw", "llr", "theta_pop", "scale"))
    env, PHI, B, SLOT, positions_of = (ctx[k] for k in ("env", "PHI", "B", "SLOT", "positions_of"))

    # ---- development: sigma x lambda
    dev_rows, alone_rows = [], []
    for pool, d in DEV:
        x = draw(pool, d)
        dev_rows.append({"draw": d, "arm": "base", **buckets(official(x["base"], x["owner"]), x["ks"])})
        for sigma in SIGMAS:
            l, _ = llr(x, sigma)
            alone_rows.append({"draw": d, "arm": f"s{sigma:g}", **buckets(official(l, x["owner"]), x["ks"])})
            for lam in LAMBDAS:
                dev_rows.append({"draw": d, "arm": f"s{sigma:g}_l{lam:g}",
                                 **buckets(official(x["base"] + lam * l, x["owner"]), x["ks"])})
        alone_rows.append({"draw": d, "arm": "style_alone",
                           **buckets(official(scale * BASE_M * x["style"], x["owner"]), x["ks"])})
        print(f"dev draw {d}", flush=True)
    dev = paired(pd.DataFrame(dev_rows), "base").set_index("arm")
    grid = dev.drop(index="base")
    best = grid["d_all"].idxmax()
    sigma = float(best.split("_")[0][1:])
    lam = float(best.split("_")[1][1:])
    alone = pd.DataFrame(alone_rows).groupby("arm")[COLS].mean()
    report["dev"] = {"grid": grid[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index"),
                     "chosen": {"sigma": sigma, "lambda": lam, "dev_delta": float(grid.loc[best, "d_all"])},
                     "grid_edge": bool(sigma in (SIGMAS[0], SIGMAS[-1]) or lam in (LAMBDAS[0], LAMBDAS[-1])),
                     "alone": alone.to_dict(orient="index")}
    print("dev grid (delta to base):\n", grid[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\nalone:\n", alone.round(4).to_string(), f"\nchosen sigma {sigma} lambda {lam}", flush=True)

    # ---- diagnostics on development draws (no decision): ablations and theta stability
    rank_mask = torch.zeros(DIM, device=device, dtype=torch.float16)
    rank_mask[:3] = 1
    abl_rows = []
    for pool, d in DEV[:10]:
        x = draw(pool, d)
        for name, mask in (("rank_only", rank_mask), ("non_rank", 1 - rank_mask)):
            l, _ = llr(x, sigma, mask)
            abl_rows.append({"draw": d, "arm": name, **buckets(official(x["base"] + lam * l, x["owner"]), x["ks"])})
            abl_rows.append({"draw": d, "arm": f"{name}_alone", **buckets(official(l, x["owner"]), x["ks"])})
        abl_rows.append({"draw": d, "arm": "base", **buckets(official(x["base"], x["owner"]), x["ks"])})
    abl = paired(pd.DataFrame(abl_rows), "base")
    report["ablation_dev10"] = abl.to_dict(orient="records")
    print("ablation (dev draws 2600-2609):\n", abl.round(4).to_string(index=False), flush=True)
    # stability: the same player's theta from two disjoint 10-game sets of the bank
    bank = env["bank"]
    sets_a, sets_b = [], []
    for _, grp in bank[bank["pool"] == 0].groupby("player_id"):
        gm = grp.sort_values("order")["game"].to_numpy()
        sets_a.append(gm[:10])
        sets_b.append(gm[10:20])
    pa, oa = positions_of(sets_a)
    pb, ob = positions_of(sets_b)
    ta = (fit(PHI, B, SLOT, pa, oa, len(sets_a), theta_pop, sigma) - theta_pop).cpu().numpy()
    tb = (fit(PHI, B, SLOT, pb, ob, len(sets_b), theta_pop, sigma) - theta_pop).cpu().numpy()
    corr = lambda u, v: float((normalise(u) * normalise(v)).sum(1).mean())
    report["stability"] = {"same_player_cos": corr(ta, tb), "shifted_player_cos": corr(ta, np.roll(tb, 1, axis=0)),
                           "mean_norm": float(np.linalg.norm(ta, axis=1).mean())}
    print("stability:", report["stability"], flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if grid.loc[best, "d_all"] <= 0:
        report["stopped"] = "development best delta <= 0; sealed draws not opened"
        (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return

    # ---- sealed
    rows = []
    for pool, d in SEALED:
        x = draw(pool, d)
        l, _ = llr(x, sigma)
        for name, s in (("base", x["base"]), ("lik", x["base"] + lam * l), ("lik_alone", l),
                        ("style_alone", scale * BASE_M * x["style"])):
            rows.append({"pool": pool, "draw": d, "arm": name, **buckets(official(s, x["owner"]), x["ks"])})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "base"), paired(frame[frame.pool == 0], "base")
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": verdict(t1, t0, "lik")}
    print("sealed pool 1:\n", t1.round(4).to_string(index=False), "\nsealed pool 0:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(report["sealed"]["verdict"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- round 2: wider grid + per-k
R2 = OUT / "round2"
R2_DEV = [(0, d) for d in range(2800, 2820)]
R2_SEALED = [(0, d) for d in range(2900, 2920)] + [(1, d) for d in range(2900, 2930)]
R2_SIGMAS = (3.0, 10.0)
R2_LAMBDAS = (2.0, 3.0, 4.0, 6.0, 8.0)
R2_MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
B0 = (3.0, 2.0)


def evaluate2(_args) -> None:
    """L: (sigma, lambda) on a wider grid; LK: L plus a per-k style multiplier; both against B0
    (sigma 3, lambda 2, multiplier 1.5) on fresh sealed draws (registered in RESULTS.md, 11/20)."""
    from scripts.identity.pid2_cheap import COLS, buckets, official, paired, verdict

    R2.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    assert (stage0["dev"]["chosen"]["sigma"], stage0["dev"]["chosen"]["lambda"]) == B0
    ctx = _context(stage0["theta_pop"])
    report, draw, llr, scale = ctx["report"], ctx["draw"], ctx["llr"], ctx["scale"]
    print("theta_pop refit max diff", report["theta_pop_refit_max_diff"], flush=True)

    def fused(x, l, lam, mk):
        m = np.array([mk[int(k)] for k in x["ks"]])[:, None]
        return x["rest"] + scale * m * x["style"] + lam * l

    flat = {k: BASE_M for k in (1, 2, 3, 5)}
    # ---- development
    cache = {}
    rows = []
    for pool, d in R2_DEV:
        x = draw(pool, d)
        ls = {s: llr(x, s)[0] for s in R2_SIGMAS}
        cache[d] = (x, ls)
        for s in R2_SIGMAS:
            for lam in R2_LAMBDAS:
                rows.append({"draw": d, "arm": f"s{s:g}_l{lam:g}",
                             **buckets(official(fused(x, ls[s], lam, flat), x["owner"]), x["ks"])})
        print(f"dev draw {d}", flush=True)
    dev = pd.DataFrame(rows)
    ref = f"s{B0[0]:g}_l{B0[1]:g}"
    grid = paired(dev, ref).set_index("arm")
    best = grid["d_all"].idxmax()
    sigma, lam = float(best.split("_")[0][1:]), float(best.split("_")[1][1:])
    report["dev_grid"] = grid[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["L"] = {"sigma": sigma, "lambda": lam, "dev_delta_vs_B0": float(grid.loc[best, "d_all"]),
                   "grid_edge": bool(sigma == R2_SIGMAS[-1] or lam == R2_LAMBDAS[-1])}
    print("dev grid vs B0:\n", grid[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          f"\nL: sigma {sigma} lambda {lam}", flush=True)
    mrows = []
    for d, (x, ls) in cache.items():
        for m in R2_MULTS:
            mk = {k: m for k in (1, 2, 3, 5)}
            mrows.append({"draw": d, "m": m, **buckets(official(fused(x, ls[sigma], lam, mk), x["owner"]), x["ks"])})
    mean_m = pd.DataFrame(mrows).groupby("m")[COLS].mean()
    mk = {}
    for k in (1, 2, 3, 5):
        col = mean_m[str(k)]
        ties = [m for m in R2_MULTS if col[m] == col.max()]
        mk[k] = min(ties, key=lambda m: abs(np.log(m / BASE_M)))
    report["LK"] = {"m_k": mk, "dev_table": mean_m.reset_index().to_dict(orient="records"),
                    "grid_edge": [k for k, m in mk.items() if m in (R2_MULTS[0], R2_MULTS[-1])]}
    print("per-k dev means (at L):\n", mean_m.round(4).to_string(), "\nchosen m_k:", mk, flush=True)
    del cache
    (R2 / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- sealed
    same = (sigma, lam) == B0
    out = []
    for pool, d in R2_SEALED:
        x = draw(pool, d)
        l0 = llr(x, B0[0])[0]
        ll = l0 if sigma == B0[0] else llr(x, sigma)[0]
        arms = {"B0": fused(x, l0, B0[1], flat), "LK": fused(x, ll, lam, mk), "ogsa12w": x["base"]}
        if not same:
            arms["L"] = fused(x, ll, lam, flat)
        for name, sc in arms.items():
            out.append({"pool": pool, "draw": d, "arm": name, **buckets(official(sc, x["owner"]), x["ks"])})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(R2 / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "B0"), paired(frame[frame.pool == 0], "B0")
    arms = ["LK"] + ([] if same else ["L"])
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": {a: verdict(t1, t0, a) for a in arms}}
    if not same:
        lk_l = paired(frame[frame.pool == 1], "L").set_index("arm").loc["LK"]
        report["sealed"]["LK_minus_L_pool1"] = {"delta": float(lk_l["d_all"]), "se": float(lk_l["se_all"])}
    v = report["sealed"]["verdict"]
    if v["LK"]["pass"] and (same or report["sealed"]["LK_minus_L_pool1"]["delta"] >= 0):
        final = "LK"
    elif not same and v["L"]["pass"]:
        final = "L"
    else:
        final = "B0"
    report["final"] = final
    report["final_params"] = {"B0": {"sigma": B0[0], "lambda": B0[1], "m_k": flat},
                              "L": {"sigma": sigma, "lambda": lam, "m_k": flat},
                              "LK": {"sigma": sigma, "lambda": lam, "m_k": mk}}[final]
    print("sealed pool 1 vs B0:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs B0:\n",
          t0.round(4).to_string(index=False), "\nverdicts:", json.dumps(v), "\nfinal:", final,
          json.dumps(report["final_params"]), flush=True)
    (R2 / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- rich features (PID_LIKELIHOOD_RICH_PLAN.md)
RICH_OUT = OUT / "rich"
R3_DEV = [(0, d) for d in range(3000, 3020)]
R3_SEALED = [(0, d) for d in range(3100, 3120)] + [(1, d) for d in range(3100, 3130)]
LK = {"sigma": 3.0, "lambda": 3.0, "m_k": {1: 0.75, 2: 1.25, 3: 2.0, 5: 3.0}}
R3_SIGMAS = (3.0, 10.0)
R3_LAMBDAS = (2.0, 3.0, 4.0)


def _official_moves() -> dict:
    """game id (part * 1e6 + row; pool rows first, then question rows) -> int64 moves, as export."""
    from go_aicup.data.preprocess import load_store
    from scripts.identity.pid2_style_export import STORE
    games = {}
    for p_i, part in enumerate(("public", "private")):
        pool_m, _, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
        q_m, _, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
        for base_row, (m, meta) in ((0, (pool_m, pool_meta)), (len(pool_meta), (q_m, q_meta))):
            for r, (off, length) in enumerate(zip(meta["offset"], meta["length"])):
                games[p_i * 1_000_000 + base_row + r] = np.asarray(m[off:off + length], np.int64)
    return games


def _sorted_positions(arr: dict) -> dict:
    order = np.lexsort((arr["ply"], arr["game"]))
    return {k: v[order] for k, v in arr.items()}


def _rich_for(arr: dict, moves_of, budget_games: int, started: float) -> np.ndarray:
    """G3/G4 raw features for sorted positions; aborts if the first 1,000 games extrapolate to > 1 h
    for `budget_games` games in total."""
    from go_aicup.tasks.identity import likelihood_rich as lr
    starts = np.flatnonzero(np.r_[True, arr["game"][1:] != arr["game"][:-1]])
    ends = np.r_[starts[1:], len(arr["game"])]
    out = np.zeros((len(arr["game"]), TOP, lr.N_RICH), np.uint8)
    lr.compute([moves_of(int(arr["game"][0]))], [arr["ply"][starts[0]:ends[0]].astype(np.int64)],
               arr["top"][starts[0]:ends[0]])                     # numba compile outside the timing
    t0 = time.monotonic()
    step = 1000
    for a in range(0, len(starts), step):
        sl = range(a, min(a + step, len(starts)))
        gm = [moves_of(int(arr["game"][starts[i]])) for i in sl]
        gp = [arr["ply"][starts[i]:ends[i]].astype(np.int64) for i in sl]
        lo, hi = starts[sl[0]], ends[sl[-1]]
        out[lo:hi] = lr.compute(gm, gp, arr["top"][lo:hi])
        if a == 0:
            rate = (time.monotonic() - t0) / len(sl)
            estimate = rate * budget_games
            print(json.dumps({"seconds_per_game": round(rate, 5), "estimate_total_seconds": round(estimate)}), flush=True)
            if estimate > 3600:
                raise SystemExit(f"replay estimate {estimate:.0f} s > 1 h: stop and report (plan section 4)")
    print(f"rich features: {len(starts)} games in {time.monotonic() - t0:.0f} s", flush=True)
    return out


def _replay_check(game_moves, game_index, cache) -> int:
    """boards before every ply 1..n-1 equal the cache's after-move boards of ply - 1; ply 0 is empty."""
    from go_aicup.tasks.identity import likelihood_rich as lr
    checked = 0
    for m, g in zip(game_moves, game_index):
        n = len(m)
        boards = lr.replay_boards([m], [np.arange(n)])
        assert not boards[0].any()
        h, _ = cache.history_batch(np.full(n - 1, g), np.arange(n - 1), steps=1)
        ref = (h[:, 0, 0].reshape(-1, 361) + 2 * h[:, 0, 1].reshape(-1, 361)).astype(np.int8)
        assert np.array_equal(boards[1:], ref), f"replay mismatch in cache game {g}"
        checked += n
    return checked


def rich_features_cmd(_args) -> None:
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.data.preprocess import load_store
    from go_aicup.features.humansl import CACHE, cache_games
    from scripts.identity.pid2_qmerge_export import DAY0
    from scripts.identity.pid2_style_export import STORE

    RICH.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    table = pd.read_parquet(GAMES)
    off = np.load(OUT / "official" / "positions.npz")
    budget = len(table) + len(np.unique(off["game"]))
    report = {}
    # replay correctness against the board caches: 200 bench games, 50 official games per part
    bench_games = table["game_index"].to_numpy()
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(bench_games, 200, replace=False))
    report["bench_replay_positions_checked"] = _replay_check([m for m, _ in cache_games(sample, CACHE)], sample,
                                                             BoardCache(CACHE))
    report["official_replay_positions_checked"] = 0
    for part in ("public", "private"):
        for name in (f"pool2_{part}", f"pid2_{part}"):
            m, _, meta = load_store(name, STORE, mmap=False)
            cache_dir = DAY0 / f"{part}2_d4_9600" / name / "cache"
            cg = pd.read_parquet(cache_dir / "games.parquet").iloc[:25]
            moves = [np.asarray(m[int(meta["offset"].iloc[r]):int(meta["offset"].iloc[r] + meta["length"].iloc[r])],
                                np.int64) for r in cg["store_row"]]
            report["official_replay_positions_checked"] += _replay_check(moves, cg["game_index"].to_numpy(),
                                                                         BoardCache(cache_dir))
    print(json.dumps(report), flush=True)
    # bench positions, in the order _context uses
    shards = sorted(p for p in DATA.glob("shard_*") if (p / "meta.json").exists())
    arr = _sorted_positions({k: np.concatenate([np.load(sh / f"{k}.npy") for sh in shards])
                             for k in ("game", "ply", "top")})
    played = dict(zip(table["game_index"], (m for m, _ in cache_games(table["game_index"], CACHE))))
    rich = _rich_for(arr, lambda g: played[g].astype(np.int64), budget, started)
    np.savez(RICH / "bench.npz", game=arr["game"], ply=arr["ply"], rich=rich)
    # official positions, in the order position_tensors uses
    oarr = _sorted_positions({k: off[k] for k in ("game", "ply", "top")})
    omoves = _official_moves()
    orich = _rich_for(oarr, omoves.__getitem__, budget, started)
    np.savez(RICH / "official.npz", game=oarr["game"], ply=oarr["ply"], rich=orich)
    report.update(seconds=round(time.monotonic() - started, 1), bench_positions=int(len(rich)),
                  official_positions=int(len(orich)),
                  bench_rates=rich.reshape(-1, 8)[arr["top"].reshape(-1) < 361].mean(0).round(4).tolist())
    (RICH / "meta.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)


def evaluate3(_args) -> None:
    """single-group screen, union grid and sealed test of the 50-dim features against LK."""
    import torch

    from scripts.identity.pid2_cheap import COLS, buckets, normalise, official, paired, verdict

    RICH_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, scale = ctx["report"], ctx["draw"], ctx["llr"], ctx["scale"]
    PHI, B, SLOT, positions_of, device = ctx["PHI"], ctx["B"], ctx["SLOT"], ctx["positions_of"], ctx["device"]
    base_mask, theta_pop, fit_pop = ctx["base_mask"], ctx["theta_pop"], ctx["fit_pop"]
    assert PHI.shape[2] == DIM_RICH
    mk = LK["m_k"]

    def fused(x, l, lam, m=mk):
        f = np.array([m[int(k)] for k in x["ks"]])[:, None]
        return x["rest"] + scale * f * x["style"] + lam * l

    def mask_of(groups):
        m = base_mask.clone()
        for g in groups:
            m[GROUPS[g]] = 1
        return m

    masks = {g: mask_of([g]) for g in GROUPS}
    priors = {g: fit_pop(masks[g]) for g in GROUPS}
    report["theta_pop_new_dims"] = {g: priors[g][GROUPS[g]].cpu().round(decimals=3).tolist() for g in GROUPS}
    print("theta_pop new dims:", report["theta_pop_new_dims"], flush=True)

    # engineering: the base-27 mask on the 50-dim tensor reproduces the 27-dim LK
    x = draw(*R3_DEV[0])
    l50 = llr(x, LK["sigma"])[0]
    pos_g, own_g = positions_of(x["gallery"])
    phi27 = PHI[:, :, :DIM]
    th = fit(phi27, B, SLOT, pos_g, own_g, len(x["gallery"]), theta_pop[:DIM], LK["sigma"])
    pos_q, own_q = positions_of(x["query"])
    lp = played_logp(phi27, B, SLOT, pos_q, torch.cat([th, theta_pop[None, :DIM]]))
    l27 = torch.zeros(len(x["query"]), len(x["gallery"]), device=device, dtype=torch.float64)
    l27.index_add_(0, own_q, lp[:, :-1] - lp[:, -1:])
    l27 = l27.cpu().numpy()
    a = buckets(official(fused(x, l50, LK["lambda"]), x["owner"]), x["ks"])
    b_ = buckets(official(fused(x, l27, LK["lambda"]), x["owner"]), x["ks"])
    report["base_reproduction"] = {"llr_max_abs_diff": float(np.abs(l50 - l27).max()),
                                   "points_max_diff": max(abs(a[c] - b_[c]) for c in COLS)}
    print("base reproduction:", report["base_reproduction"], flush=True)
    assert report["base_reproduction"]["points_max_diff"] == 0.0 and report["base_reproduction"]["llr_max_abs_diff"] < 1e-3

    # ---- step 1: one group at a time (sigma 3, lambda 3, LK multipliers)
    cache, rows = {}, []
    for pool, d in R3_DEV:
        x = draw(pool, d)
        l_lk = llr(x, LK["sigma"])[0]
        cache[d] = (x, l_lk)
        rows.append({"draw": d, "arm": "LK", **buckets(official(fused(x, l_lk, LK["lambda"]), x["owner"]), x["ks"])})
        for g in GROUPS:
            lg = llr(x, LK["sigma"], masks[g], priors[g])[0]
            rows.append({"draw": d, "arm": g, **buckets(official(fused(x, lg, LK["lambda"]), x["owner"]), x["ks"])})
        print(f"step1 draw {d}", flush=True)
    s1 = paired(pd.DataFrame(rows), "LK").set_index("arm")
    kept = [g for g in GROUPS if s1.loc[g, "d_all"] >= 0.002]
    report["step1"] = s1[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["kept"] = kept
    print("step 1 (vs LK):\n", s1[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\nkept:", kept, flush=True)
    (RICH_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if not kept:
        report["stopped"] = "no group gained >= +0.002 on the development draws; sealed draws not opened"
        (RICH_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return

    # ---- step 2: union, sigma x lambda
    union = mask_of(kept)
    prior_u = fit_pop(union)
    rows = []
    for d, (x, l_lk) in cache.items():
        rows.append({"draw": d, "arm": "LK", **buckets(official(fused(x, l_lk, LK["lambda"]), x["owner"]), x["ks"])})
        for sg in R3_SIGMAS:
            lu = llr(x, sg, union, prior_u)[0]
            for lam in R3_LAMBDAS:
                rows.append({"draw": d, "arm": f"s{sg:g}_l{lam:g}",
                             **buckets(official(fused(x, lu, lam), x["owner"]), x["ks"])})
            if sg == R3_SIGMAS[0]:
                rows.append({"draw": d, "arm": "union_alone", **buckets(official(lu, x["owner"]), x["ks"])})
        print(f"step2 draw {d}", flush=True)
    s2 = paired(pd.DataFrame(rows), "LK").set_index("arm")
    grid = s2.drop(index=["LK", "union_alone"])
    best = grid["d_all"].idxmax()
    sigma, lam = float(best.split("_")[0][1:]), float(best.split("_")[1][1:])
    report["step2"] = s2[["1", "2", "3", "5", "all", "d_all", "se_all"]].to_dict(orient="index")
    report["chosen"] = {"groups": kept, "sigma": sigma, "lambda": lam, "dev_delta": float(grid.loc[best, "d_all"]),
                        "grid_edge": bool(sigma == R3_SIGMAS[-1] or lam == R3_LAMBDAS[-1])}
    print("step 2 (vs LK):\n", s2[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\nchosen:", report["chosen"], flush=True)
    del cache
    # stability (pool 0 bank, games 1-10 vs 11-20), base-27 and union at their sigmas
    bank = ctx["env"]["bank"]
    sets_a, sets_b = [], []
    for _, grp in bank[bank["pool"] == 0].groupby("player_id"):
        gm = grp.sort_values("order")["game"].to_numpy()
        sets_a.append(gm[:10])
        sets_b.append(gm[10:20])
    pa, oa = positions_of(sets_a)
    pb, ob = positions_of(sets_b)
    stab = {}
    for name, m, pr, sg in (("base27", base_mask, theta_pop, LK["sigma"]), ("union", union, prior_u, sigma)):
        ta = (fit(PHI, B, SLOT, pa, oa, len(sets_a), pr, sg, mask=m) - pr).cpu().numpy()
        tb = (fit(PHI, B, SLOT, pb, ob, len(sets_b), pr, sg, mask=m) - pr).cpu().numpy()
        stab[name] = {"same_player_cos": float((normalise(ta) * normalise(tb)).sum(1).mean()),
                      "shifted_player_cos": float((normalise(ta) * normalise(np.roll(tb, 1, 0))).sum(1).mean())}
    report["stability"] = stab
    report["final_params"] = {"groups": kept, "mask": union.cpu().tolist(), "sigma": sigma, "lambda": lam,
                              "m_k": mk, "theta_pop": prior_u.cpu().tolist()}
    print("stability:", stab, flush=True)
    (RICH_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- sealed
    out = []
    for pool, d in R3_SEALED:
        x = draw(pool, d)
        l_lk = llr(x, LK["sigma"])[0]
        lu = llr(x, sigma, union, prior_u)[0]
        for name, sc in (("LK", fused(x, l_lk, LK["lambda"])), ("RICH", fused(x, lu, lam)),
                         ("rich_alone", lu), ("lk_alone", l_lk)):
            out.append({"pool": pool, "draw": d, "arm": name, **buckets(official(sc, x["owner"]), x["ks"])})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(RICH_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "LK"), paired(frame[frame.pool == 0], "LK")
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": verdict(t1, t0, "RICH")}
    print("sealed pool 1 vs LK:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs LK:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(report["sealed"]["verdict"]), flush=True)
    (RICH_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- empirical-Bayes prior (PID_EB_COMBINER_PLAN.md, plan 1)
EB_OUT = OUT / "eb"
R4_DEV = [(0, d) for d in range(3200, 3220)]
R4_SEALED = [(0, d) for d in range(3300, 3320)] + [(1, d) for d in range(3300, 3330)]
EB_CS = (0.5, 1.0, 2.0)
EB_LAMBDAS = (2.0, 3.0, 4.0)
EB_FLOOR = 0.01                    # eigenvalue floor (sd 0.1)
EB_BLOCKS = 4                      # 4 x 10 games per pool-0 player
EB_BLOCK_GAMES = 10
EB_FIT_SIGMA = 10.0                # weak prior for the block fits (as fit_pop)


def cross_cov(thetas: np.ndarray) -> np.ndarray:
    """thetas [blocks, players, d] fitted on disjoint game sets of the same players -> between-player
    covariance: the mean over block pairs of the cross-covariance, symmetrised (noise is independent
    across blocks, so it drops out of the expectation)."""
    t = thetas - thetas.mean(axis=1, keepdims=True)
    n = t.shape[1]
    pairs = [(a, b) for a in range(len(t)) for b in range(a + 1, len(t))]
    c = sum(t[a].T @ t[b] for a, b in pairs) / (len(pairs) * (n - 1))
    return (c + c.T) / 2


def regularise(cov: np.ndarray, floor: float = EB_FLOOR) -> tuple[np.ndarray, dict]:
    w, v = np.linalg.eigh(cov)
    info = {"eig_raw": np.round(w[::-1], 5).tolist(), "clipped": int((w < floor).sum()),
            "negative": int((w < 0).sum())}
    w = np.maximum(w, floor)
    return (v * w) @ v.T, info


def precision_of(cov_kept: np.ndarray, kept: np.ndarray, d: int, c: float) -> np.ndarray:
    """[d, d] prior precision: inv(c * cov) on the kept dims, 1 on the masked dims (theta stays at the prior)."""
    p = np.eye(d)
    p[np.ix_(kept, kept)] = np.linalg.inv(c * cov_kept)
    return p


def evaluate4(_args) -> None:
    """empirical-Bayes prior: Sigma from pool-0 block fits, selection on dev draws, sealed test vs RICH."""
    import torch

    from scripts.identity.pid2_cheap import COLS, buckets, normalise, official, paired, verdict

    EB_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    r3 = json.loads((RICH_OUT / "report.json").read_text())
    fp3 = r3["final_params"]
    assert fp3["groups"] == ["G3"] and fp3["sigma"] == 3.0 and fp3["lambda"] == 3.0, fp3
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, scale, device = ctx["report"], ctx["draw"], ctx["llr"], ctx["scale"], ctx["device"]
    PHI, B, SLOT, positions_of, fit_pop = ctx["PHI"], ctx["B"], ctx["SLOT"], ctx["positions_of"], ctx["fit_pop"]
    assert PHI.shape[2] == DIM_RICH
    mk = {int(k): float(v) for k, v in fp3["m_k"].items()}
    m37 = torch.tensor(fp3["mask"], device=device, dtype=torch.float32)
    prior37 = torch.tensor(fp3["theta_pop"], device=device, dtype=torch.float64)
    report["rich_prior_refit_max_diff"] = float((fit_pop(m37) - prior37).abs().max())
    m50 = torch.ones(DIM_RICH, device=device)
    prior50 = fit_pop(m50)
    arms = {"eb37": (m37, prior37), "eb50": (m50, prior50)}

    def fused(x, l, lam):
        f = np.array([mk[int(k)] for k in x["ks"]])[:, None]
        return x["rest"] + scale * f * x["style"] + lam * l

    def rich_llr(x):
        return llr(x, fp3["sigma"], m37, prior37)[0]

    # ---- engineering: precision = I / sigma^2 reproduces RICH point for point
    x = draw(*R4_DEV[0])
    iso = torch.eye(DIM_RICH, device=device, dtype=torch.float64) / fp3["sigma"] ** 2
    a = buckets(official(fused(x, rich_llr(x), 3.0), x["owner"]), x["ks"])
    b_ = buckets(official(fused(x, llr(x, 1.0, m37, prior37, iso)[0], 3.0), x["owner"]), x["ks"])
    report["iso_reproduction"] = max(abs(a[c] - b_[c]) for c in COLS)
    print("iso reproduction:", report["iso_reproduction"], flush=True)
    assert report["iso_reproduction"] == 0.0

    # ---- Sigma: pool-0 players, 4 disjoint blocks of 10 games, weak-prior fits, cross-covariance
    bank = ctx["env"]["bank"]
    blocks = [[] for _ in range(EB_BLOCKS)]
    for _, grp in bank[bank["pool"] == 0].groupby("player_id"):
        gm = grp.sort_values("order")["game"].to_numpy()
        assert len(gm) >= EB_BLOCKS * EB_BLOCK_GAMES, len(gm)
        for i in range(EB_BLOCKS):
            blocks[i].append(gm[i * EB_BLOCK_GAMES:(i + 1) * EB_BLOCK_GAMES])
    n_players = len(blocks[0])
    covs, prec = {}, {}
    report["sigma"] = {}
    for name, (m, pr) in arms.items():
        kept = np.flatnonzero(m.cpu().numpy() > 0)
        th = []
        for bl in blocks:
            p_, o_ = positions_of(bl)
            th.append(fit(PHI, B, SLOT, p_, o_, n_players, pr, EB_FIT_SIGMA, mask=m).cpu().numpy()[:, kept])
        raw = cross_cov(np.stack(th))
        cov, info = regularise(raw)
        covs[name] = cov
        np.save(EB_OUT / f"sigma_{name}.npy", cov)
        w, v = np.linalg.eigh(cov)
        top = [{"eig": float(w[-i]), "dims": [int(kept[j]) for j in np.argsort(-np.abs(v[:, -i]))[:4]]}
               for i in range(1, 6)]
        noise = float(np.mean([np.diag(np.cov(t.T)).mean() for t in th]) - np.diag(raw).mean())
        report["sigma"][name] = {**info, "players": n_players, "kept_dims": kept.tolist(),
                                 "between_sd_raw": np.round(np.sqrt(np.maximum(np.diag(raw), 0)), 3).tolist(),
                                 "mean_noise_var": noise, "top_eigvecs": top}
        print(name, "between-player sd:", report["sigma"][name]["between_sd_raw"], "clipped",
              info["clipped"], "of", len(kept), "negative", info["negative"], flush=True)
        for c in EB_CS:
            prec[(name, c)] = torch.tensor(precision_of(cov, kept, DIM_RICH, c), device=device, dtype=torch.float64)
    (EB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- selection (dev draws)
    rows = []
    for pool, d in R4_DEV:
        x = draw(pool, d)
        rows.append({"draw": d, "arm": "RICH", **buckets(official(fused(x, rich_llr(x), 3.0), x["owner"]), x["ks"])})
        for name, (m, pr) in arms.items():
            for c in EB_CS:
                le = llr(x, 1.0, m, pr, prec[(name, c)])[0]
                for lam in EB_LAMBDAS:
                    rows.append({"draw": d, "arm": f"{name}_c{c:g}_l{lam:g}",
                                 **buckets(official(fused(x, le, lam), x["owner"]), x["ks"])})
        print(f"dev draw {d}", flush=True)
    s1 = paired(pd.DataFrame(rows), "RICH").set_index("arm")
    grid = s1.drop(index="RICH")
    best = grid["d_all"].idxmax()
    name, c_s, l_s = best.split("_")
    c, lam = float(c_s[1:]), float(l_s[1:])
    report["dev"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].to_dict(orient="index")
    report["chosen"] = {"arm": name, "c": c, "lambda": lam, "dev_delta": float(grid.loc[best, "d_all"]),
                        "grid_edge": bool(c == EB_CS[-1] or lam == EB_LAMBDAS[-1])}
    print("dev (vs RICH):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\nchosen:", report["chosen"], flush=True)
    m, pr = arms[name]
    P = prec[(name, c)]
    report["final_params"] = {"arm": name, "mask": m.cpu().tolist(), "c": c, "lambda": lam, "m_k": mk,
                              "theta_pop": pr.cpu().tolist(), "sigma_file": f"sigma_{name}.npy"}
    np.save(EB_OUT / "precision.npy", P.cpu().numpy())
    (EB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- stability on pool 1 (pool-0 games were used for Sigma)
    sets_a, sets_b = [], []
    for _, grp in bank[bank["pool"] == 1].groupby("player_id"):
        gm = grp.sort_values("order")["game"].to_numpy()
        sets_a.append(gm[:10])
        sets_b.append(gm[10:20])
    pa, oa = positions_of(sets_a)
    pb, ob = positions_of(sets_b)
    stab = {}
    for label, mm, pp, sg, pc in (("rich", m37, prior37, fp3["sigma"], None), ("eb", m, pr, 1.0, P)):
        ta = (fit(PHI, B, SLOT, pa, oa, len(sets_a), pp, sg, mask=mm, precision=pc) - pp).cpu().numpy()
        tb = (fit(PHI, B, SLOT, pb, ob, len(sets_b), pp, sg, mask=mm, precision=pc) - pp).cpu().numpy()
        stab[label] = {"same_player_cos": float((normalise(ta) * normalise(tb)).sum(1).mean()),
                       "shifted_player_cos": float((normalise(ta) * normalise(np.roll(tb, 1, 0))).sum(1).mean())}
    report["stability_pool1"] = stab
    print("stability (pool 1):", stab, flush=True)
    (EB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- sealed
    out = []
    for pool, d in R4_SEALED:
        x = draw(pool, d)
        lr = rich_llr(x)
        le = llr(x, 1.0, m, pr, P)[0]
        for arm, sc in (("RICH", fused(x, lr, 3.0)), ("EB", fused(x, le, lam)), ("eb_alone", le), ("rich_alone", lr)):
            out.append({"pool": pool, "draw": d, "arm": arm, **buckets(official(sc, x["owner"]), x["ks"])})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(EB_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "RICH"), paired(frame[frame.pool == 0], "RICH")
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": verdict(t1, t0, "EB")}
    print("sealed pool 1 vs RICH:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs RICH:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(report["sealed"]["verdict"]), flush=True)
    (EB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


def eb_selfcheck() -> None:
    """synthetic: planted anisotropic Sigma is recovered from 4 disjoint blocks, and the full-Sigma
    MAP beats every isotropic sigma on theta MSE."""
    import torch
    rng = np.random.default_rng(1)
    torch.manual_seed(1)
    n_p, per, d, J = 300, 300, 6, 8
    sd = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
    corr = np.eye(d)
    corr[0, 1] = corr[1, 0] = 0.6
    cov_true = corr * np.outer(sd, sd)
    theta = rng.multivariate_normal(np.zeros(d), cov_true, size=n_p)

    def sample(n_pos):
        n = n_p * n_pos
        phi = torch.from_numpy(rng.normal(size=(n, J, d)).astype(np.float32) * 0.7)
        b = torch.log_softmax(torch.from_numpy(rng.normal(size=(n, J)).astype(np.float32)), dim=1)
        owner = torch.from_numpy(np.repeat(np.arange(n_p), n_pos))
        z = b + torch.einsum("pjd,pd->pj", phi.float(), torch.from_numpy(theta).float()[owner])
        slot = torch.distributions.Categorical(logits=z).sample()
        return phi, b, slot, torch.arange(n), owner

    zero = torch.zeros(d, dtype=torch.float64)
    ths = []
    for _ in range(EB_BLOCKS):
        phi, b, slot, pos, owner = sample(per)
        ths.append(fit(phi, b, slot, pos, owner, n_p, zero, EB_FIT_SIGMA).numpy())
    cov, _ = regularise(cross_cov(np.stack(ths)))
    rel = np.abs(np.diag(cov) - np.diag(cov_true)) / np.diag(cov_true)
    assert rel.max() < 0.25, (np.diag(cov), rel)
    assert abs(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) - 0.6) < 0.15, cov[0, 1]
    phi, b, slot, pos, owner = sample(60)              # a small fresh block: the prior matters
    P = torch.from_numpy(precision_of(cov, np.arange(d), d, 1.0))
    mse_eb = float(((fit(phi, b, slot, pos, owner, n_p, zero, 1.0, precision=P).numpy() - theta) ** 2).mean())
    mse_iso = {s: float(((fit(phi, b, slot, pos, owner, n_p, zero, s).numpy() - theta) ** 2).mean())
               for s in (0.3, 0.5, 1.0, 3.0, 10.0)}
    assert mse_eb < min(mse_iso.values()), (mse_eb, mse_iso)
    print("eb selfcheck ok", {"diag_rel_err": np.round(rel, 3).tolist(), "mse_eb": round(mse_eb, 4),
                              "mse_iso": {k: round(v, 4) for k, v in mse_iso.items()}}, flush=True)


# ---------------------------------------------------------------- per-k learned fusion (PID_EB_COMBINER_PLAN.md, plan 2)
COMB_OUT = OUT / "combine"
R5_DEV = [(0, d) for d in range(3400, 3420)]
R5_SEALED = [(0, d) for d in range(3500, 3520)] + [(1, d) for d in range(3500, 3530)]
PARTS = ("opening", "category", "L1", "L2", "B", "A1", "A2", "LLR")
KS = (1, 2, 3, 5)


def current_weights(mk: dict, lam: float) -> dict:
    """the base fusion as per-k weights on PARTS (encoder terms are in units of multiplier 1.5)."""
    return {k: np.array([1, 1, 1, 1] + [mk[k] / BASE_M] * 3 + [lam], np.float64) for k in KS}


def fit_combiner(sets, w0: np.ndarray, arm: str, device) -> np.ndarray:
    """conditional logit for one k: sets = [(X [Q, C, 8], owner [Q])]; returns the 8 weights.
    FULL: all 8 free; SMALL: one multiplier on the three encoder terms and the LLR weight."""
    import torch
    Xs = [(torch.from_numpy(x).to(device), torch.from_numpy(o).to(device)) for x, o in sets]
    n = sum(len(o) for _, o in sets)
    w0t = torch.tensor(w0, device=device, dtype=torch.float64)
    p = torch.tensor(w0 if arm == "FULL" else w0[[4, 7]], device=device, dtype=torch.float64, requires_grad=True)

    def weights(p):
        if arm == "FULL":
            return p
        return torch.cat([w0t[:4], p[:1].repeat(3), p[1:]])

    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = weights(p)
        loss = 0.0
        for x, o in Xs:
            s = torch.einsum("qcj,j->qc", x, w)
            loss = loss + (torch.logsumexp(s, dim=1) - s[torch.arange(len(o), device=device), o]).sum()
        loss = loss / n
        loss.backward()
        return loss

    opt.step(closure)
    return weights(p).detach().cpu().numpy()


def apply_weights(X: np.ndarray, ks: np.ndarray, W: dict) -> np.ndarray:
    return np.einsum("qcj,qj->qc", X, np.stack([W[int(k)] for k in ks]))


def evaluate5(_args) -> None:
    """per-k learned fusion weights on top of the empirical-Bayes base: cross-fit arm choice on dev
    draws, refit on all dev draws, sealed test vs EB."""
    import torch

    from scripts.identity.pid2_cheap import COLS, buckets, official, paired, verdict

    COMB_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    r4 = json.loads((EB_OUT / "report.json").read_text())
    assert r4["sealed"]["verdict"]["pass"]
    fp = r4["final_params"]
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, scale, device = ctx["report"], ctx["draw"], ctx["llr"], ctx["scale"], ctx["device"]
    mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    prior = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    P = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)
    mk = {int(k): float(v) for k, v in fp["m_k"].items()}
    lam = float(fp["lambda"])
    W0 = current_weights(mk, lam)

    def get(pool, d):
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        X = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le], axis=2)
        f = np.array([mk[int(k)] for k in x["ks"]])[:, None]
        eb = x["rest"] + scale * f * x["style"] + lam * le
        return X, x["ks"], x["owner"], eb

    def score_rows(d, X, ks, owner, arms):
        return [{"draw": d, "arm": a, **buckets(official(apply_weights(X, ks, W), owner), ks)} for a, W in arms.items()]

    # ---- dev draws: components, reproduction of EB at the current weights
    dev = {}
    for pool, d in R5_DEV:
        X, ks, owner, eb = get(pool, d)
        if not dev:
            a = buckets(official(eb, owner), ks)
            b_ = buckets(official(apply_weights(X, ks, W0), owner), ks)
            report["current_reproduction"] = {"score_max_abs_diff": float(np.abs(apply_weights(X, ks, W0) - eb).max()),
                                              "points_max_diff": max(abs(a[c] - b_[c]) for c in COLS)}
            print("current-weights reproduction:", report["current_reproduction"], flush=True)
            assert report["current_reproduction"]["points_max_diff"] == 0.0
        dev[d] = (X, ks, owner)
        print(f"dev draw {d}", flush=True)

    def fit_all(draws, arm):
        return {k: fit_combiner([(dev[d][0][dev[d][1] == k], dev[d][2][dev[d][1] == k]) for d in draws], W0[k], arm,
                                device) for k in KS}

    # ---- cross-fit: fit on one half, score the other
    names = [d for _, d in R5_DEV]
    halves = (names[:10], names[10:])
    rows, cross = [], {}
    for i, (fit_on, held) in enumerate((halves, halves[::-1])):
        arms = {"CURRENT": W0, **{a: fit_all(fit_on, a) for a in ("SMALL", "FULL")}}
        cross[i] = {a: {k: v.tolist() for k, v in W.items()} for a, W in arms.items() if a != "CURRENT"}
        for d in held:
            rows += score_rows(d, *dev[d], arms)
    s1 = paired(pd.DataFrame(rows), "CURRENT").set_index("arm")
    report["crossfit"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["crossfit_weights"] = cross
    print("cross-fit (vs CURRENT):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          flush=True)
    best = s1.drop(index="CURRENT")["d_all"].idxmax()
    if s1.loc[best, "d_all"] <= 0:
        report["stopped"] = "the current weights won the cross-fit; sealed draws not opened"
        (COMB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return

    # ---- refit the chosen arm on all dev draws; weight check
    W = fit_all(names, best)
    flags = []
    for k in KS:
        ratio = W[k] / W0[k]
        for j, c in enumerate(PARTS):
            if W[k][j] < 0 or ratio[j] > 4:
                flags.append({"k": k, "part": c, "weight": float(W[k][j]), "ratio": float(ratio[j])})
    report["chosen"] = {"arm": best, "crossfit_delta": float(s1.loc[best, "d_all"]),
                        "weights": {k: W[k].tolist() for k in KS},
                        "ratio_to_current": {k: np.round(W[k] / W0[k], 3).tolist() for k in KS}, "flags": flags}
    print("chosen:", best, "\nweights / current per k:",
          {k: np.round(W[k] / W0[k], 2).tolist() for k in KS}, "\nflags:", flags, flush=True)
    (COMB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    del dev

    # ---- sealed
    out = []
    for pool, d in R5_SEALED:
        X, ks, owner, _ = get(pool, d)
        for r in score_rows(d, X, ks, owner, {"EB": W0, "COMB": W}):
            out.append({"pool": pool, **r})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(COMB_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "EB"), paired(frame[frame.pool == 0], "EB")
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": verdict(t1, t0, "COMB")}
    print("sealed pool 1 vs EB:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs EB:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(report["sealed"]["verdict"]), flush=True)
    (COMB_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


def combiner_selfcheck() -> None:
    """synthetic: the conditional logit recovers planted weights; SMALL keeps the fixed weights."""
    import torch
    rng = np.random.default_rng(2)
    true = np.array([1.0, 0.5, 2.0, 1.0, 0.8, 0.3, 1.2, 3.0])
    sets = []
    for _ in range(3):
        X = rng.normal(size=(1500, 60, 8)) * 0.5
        s = X @ true
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        owner = np.array([rng.choice(60, p=r) for r in p])
        sets.append((X, owner))
    w = fit_combiner(sets, np.ones(8), "FULL", torch.device("cpu"))
    assert np.abs(w - true).max() < 0.2, w
    ws = fit_combiner(sets, np.ones(8), "SMALL", torch.device("cpu"))
    assert np.array_equal(ws[:4], np.ones(4)) and ws[4] == ws[5] == ws[6], ws
    print("combiner selfcheck ok", {"full": np.round(w, 2).tolist(), "small": np.round(ws, 2).tolist()}, flush=True)


# ---------------------------------------------------------------- EB lambda grid widened (2026-11-21, after plan 1 chose the edge lambda 4)
EB2_OUT = OUT / "eb_lambda"
R6_DEV = [(0, d) for d in range(3600, 3620)]
R6_SEALED = [(0, d) for d in range(3700, 3720)] + [(1, d) for d in range(3700, 3730)]
EB2_LAMBDAS = (4.0, 5.0, 6.0, 8.0)


def evaluate6(_args) -> None:
    """EB50 / c 1 fixed; lambda in {4, 5, 6, 8} on fresh dev draws; sealed test of the chosen lambda vs lambda 4."""
    import torch

    from scripts.identity.pid2_cheap import buckets, official, paired, verdict

    EB2_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    fp = json.loads((EB_OUT / "report.json").read_text())["final_params"]
    assert fp["lambda"] == EB2_LAMBDAS[0]
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, scale, device = ctx["report"], ctx["draw"], ctx["llr"], ctx["scale"], ctx["device"]
    mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    prior = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    P = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)
    mk = {int(k): float(v) for k, v in fp["m_k"].items()}

    def get(pool, d):
        x = draw(pool, d)
        le = llr(x, 1.0, mask, prior, P)[0]
        f = np.array([mk[int(k)] for k in x["ks"]])[:, None]
        return x["rest"] + scale * f * x["style"], le, x

    rows = []
    for pool, d in R6_DEV:
        s0, le, x = get(pool, d)
        for lam in EB2_LAMBDAS:
            rows.append({"draw": d, "arm": f"l{lam:g}", **buckets(official(s0 + lam * le, x["owner"]), x["ks"])})
        print(f"dev draw {d}", flush=True)
    s1 = paired(pd.DataFrame(rows), "l4").set_index("arm")
    best = s1["d_all"].idxmax()
    lam = float(best[1:])
    report["dev"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["chosen"] = {"lambda": lam, "dev_delta": float(s1.loc[best, "d_all"]), "grid_edge": lam == EB2_LAMBDAS[-1]}
    print("dev (vs lambda 4):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\nchosen:", report["chosen"], flush=True)
    (EB2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if lam == EB2_LAMBDAS[0]:
        report["stopped"] = "lambda 4 won on the dev draws; sealed draws not opened"
        (EB2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    out = []
    for pool, d in R6_SEALED:
        s0, le, x = get(pool, d)
        for arm, l_ in (("EB", 4.0), ("EBL", lam)):
            out.append({"pool": pool, "draw": d, "arm": arm, **buckets(official(s0 + l_ * le, x["owner"]), x["ks"])})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(EB2_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "EB"), paired(frame[frame.pool == 0], "EB")
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                        "verdict": verdict(t1, t0, "EBL")}
    print("sealed pool 1 vs EB:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs EB:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(report["sealed"]["verdict"]), flush=True)
    (EB2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- colour-dependent fusion weights (2026-11-21, PID_PSEUDO_COLOUR_PLAN.md plan 2)
COL_OUT = OUT / "combine_colour"
R7_DEV = [(0, d) for d in range(4000, 4040)]
R7_SEALED = [(0, d) for d in range(4100, 4120)] + [(1, d) for d in range(4100, 4130)]
COL_RHOS = (0.0, 1.0, 10.0, 100.0)


def fit_colour(sets, w0: np.ndarray, rho: float, device, refit_only: bool = False) -> np.ndarray:
    """conditional logit for one k with weights w + z v, z = 2 f - 1 per question (f = share of the
    question's games played as black); sets = [(X [Q, C, 8], owner [Q], z [Q])]; L2 penalty rho |v|^2.
    Returns [16] = (w, v). refit_only keeps v = 0 (diagnostic: refit without colour)."""
    import torch
    Xs = [(torch.from_numpy(x).to(device), torch.from_numpy(o).to(device), torch.from_numpy(z).to(device))
          for x, o, z in sets]
    n = sum(len(o) for _, o, _ in sets)
    p = torch.tensor(np.r_[w0, np.zeros(8)], device=device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w, v = p[:8], (p[8:] * 0 if refit_only else p[8:])
        loss = 0.0
        for x, o, z in Xs:
            wq = w[None] + z[:, None] * v[None]
            s = torch.einsum("qcj,qj->qc", x, wq)
            loss = loss + (torch.logsumexp(s, dim=1) - s[torch.arange(len(o), device=device), o]).sum()
        loss = loss / n + (0.0 if refit_only else rho * (v ** 2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    out = p.detach().cpu().numpy().copy()
    if refit_only:
        out[8:] = 0.0
    return out


def apply_colour(X: np.ndarray, ks: np.ndarray, z: np.ndarray, W: dict) -> np.ndarray:
    """W[k] = [16] (w, v) -> scores [Q, C]."""
    wq = np.stack([W[int(k)][:8] + zi * W[int(k)][8:] for k, zi in zip(ks, z)])
    return np.einsum("qcj,qj->qc", X, wq)


def evaluate7(_args) -> None:
    """colour-dependent per-k fusion weights on top of the learned fusion (821010-comb-probe): cross-fit rho on
    dev draws 4000-4039, refit on all 40, sealed test vs the current weights."""
    import torch

    from scripts.identity.pid2_cheap import COLS, buckets, official, paired, verdict

    COL_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    fp = json.loads((EB_OUT / "report.json").read_text())["final_params"]
    r5 = json.loads((COMB_OUT / "report.json").read_text())
    WC = {int(k): np.array(v, np.float64) for k, v in r5["chosen"]["weights"].items()}
    W0 = {k: np.r_[WC[k], np.zeros(8)] for k in KS}
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, device, env = ctx["report"], ctx["draw"], ctx["llr"], ctx["device"], ctx["env"]
    color = env["color"]
    mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    prior = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    P = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)

    def get(pool, d):
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        X = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le], axis=2)
        f = np.array([float((color[np.asarray(g)] == 0).mean()) for g in x["query"]])
        return X, x["ks"], x["owner"], 2 * f - 1

    def score_rows(d, X, ks, owner, z, arms):
        return [{"draw": d, "arm": a, **buckets(official(apply_colour(X, ks, z, W), owner), ks)} for a, W in arms.items()]

    dev, colour_rows = {}, []
    for pool, d in R7_DEV:
        X, ks, owner, z = get(pool, d)
        if not dev:
            diff = float(np.abs(apply_colour(X, ks, z, W0) - apply_weights(X, ks, WC)).max())
            report["v0_reproduction"] = diff
            print(f"v = 0 reproduction max abs diff {diff:.2e}", flush=True)
            assert diff < 1e-9
        dev[d] = (X, ks, owner, z)
        pts = official(apply_weights(X, ks, WC), owner)
        for k in KS:
            for name, sel in (("black", z == 1), ("white", z == -1), ("mixed", np.abs(z) < 1)):
                m = (ks == k) & sel
                if m.any():
                    colour_rows.append({"draw": d, "k": k, "group": name, "n": int(m.sum()), "pts": float(pts[m].sum())})
        print(f"dev draw {d}", flush=True)
    cg = pd.DataFrame(colour_rows).groupby(["k", "group"])[["n", "pts"]].sum()
    cg["mean"] = cg["pts"] / cg["n"]
    report["base_by_colour"] = {f"{k}_{g}": {"n": int(r.n), "mean": float(r["mean"])} for (k, g), r in cg.iterrows()}
    print("base (current weights) by k and colour on dev:\n", cg.round(4).to_string(), flush=True)

    def fit_all(draws, rho, refit_only=False):
        return {k: fit_colour([(dev[d][0][dev[d][1] == k], dev[d][2][dev[d][1] == k], dev[d][3][dev[d][1] == k])
                               for d in draws], WC[k], rho, device, refit_only) for k in KS}

    names = [d for _, d in R7_DEV]
    halves = (names[:20], names[20:])
    rows = []
    for fit_on, held in (halves, halves[::-1]):
        arms = {"COMB": W0, "REFIT": fit_all(fit_on, 0.0, True),
                **{f"COL_r{r:g}": fit_all(fit_on, r) for r in COL_RHOS}}
        for d in held:
            rows += score_rows(d, *dev[d], arms)
    s1 = paired(pd.DataFrame(rows), "COMB").set_index("arm")
    report["crossfit"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    print("cross-fit (vs COMB; REFIT = diagnostic, not selectable):\n",
          s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(), flush=True)
    cand = s1.loc[[f"COL_r{r:g}" for r in COL_RHOS]]
    best = cand["d_all"].idxmax()
    rho = float(best.split("_r")[1])
    if cand.loc[best, "d_all"] <= 0:
        report["stopped"] = "no colour arm beat the current weights in the cross-fit; sealed draws not opened"
        (COL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    W = fit_all(names, rho)
    flags = []
    for k in KS:
        for j, c in enumerate(PARTS):
            for zz in (-1.0, 1.0):
                if np.sign(W[k][j] + zz * W[k][8 + j]) != np.sign(W[k][j]) and abs(W[k][j]) > 0:
                    flags.append({"k": k, "part": c, "z": zz, "w": float(W[k][j]), "v": float(W[k][8 + j])})
    report["chosen"] = {"rho": rho, "crossfit_delta": float(cand.loc[best, "d_all"]), "grid_edge": rho in (COL_RHOS[0], COL_RHOS[-1]),
                        "weights": {k: W[k].tolist() for k in KS},
                        "v_over_w": {k: np.round(W[k][8:] / W[k][:8], 3).tolist() for k in KS}, "flags": flags}
    print("chosen:", report["chosen"]["rho"], "\nv / w per k:", report["chosen"]["v_over_w"], "\nflags:", flags, flush=True)
    (COL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    del dev

    out = []
    for pool, d in R7_SEALED:
        X, ks, owner, z = get(pool, d)
        for r in score_rows(d, X, ks, owner, z, {"COMB": W0, "COL": W}):
            out.append({"pool": pool, **r})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(COL_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "COMB"), paired(frame[frame.pool == 0], "COMB")
    v = verdict(t1, t0, "COL")
    v["probe"] = bool(not v["pass"] and v["pool1_delta"] > 0 and v["pool0_delta"] > 0)
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"), "verdict": v}
    print("sealed pool 1 vs COMB:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs COMB:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(v), flush=True)
    (COL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


def colour_selfcheck() -> None:
    """synthetic: fit_colour recovers planted (w, v); rho -> large shrinks v to ~0; v = 0 equals apply_weights."""
    import torch
    rng = np.random.default_rng(3)
    w_true = np.array([1.0, 0.5, 2.0, 1.0, 0.8, 0.3, 1.2, 3.0])
    v_true = np.array([0.5, 0.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0])
    sets = []
    for _ in range(3):
        X = rng.normal(size=(1500, 60, 8)) * 0.5
        z = rng.choice([-1.0, 0.0, 1.0], size=1500)
        s = np.einsum("qcj,qj->qc", X, w_true[None] + z[:, None] * v_true[None])
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        sets.append((X, np.array([rng.choice(60, p=r) for r in p]), z))
    got = fit_colour(sets, np.ones(8), 0.0, torch.device("cpu"))
    assert np.abs(got[:8] - w_true).max() < 0.25 and np.abs(got[8:] - v_true).max() < 0.25, got
    big = fit_colour(sets, np.ones(8), 1e4, torch.device("cpu"))
    assert np.abs(big[8:]).max() < 0.01, big
    X, o, z = sets[0]
    W = {k: np.r_[w_true, np.zeros(8)] for k in KS}
    ks = np.full(len(o), 1)
    assert np.abs(apply_colour(X, ks, z, W) - apply_weights(X, ks, {1: w_true})).max() < 1e-12
    print("colour selfcheck ok", {"w": np.round(got[:8], 2).tolist(), "v": np.round(got[8:], 2).tolist()}, flush=True)


# ---------------------------------------------------------------- pseudo-label gallery expansion (2026-11-21, PID_PSEUDO_COLOUR_PLAN.md plan 1)
PS_OUT = OUT / "pseudo"
R8_DEV = [(0, d) for d in range(3800, 3820)]
R8_SEALED = [(0, d) for d in range(3900, 3920)] + [(1, d) for d in range(3900, 3930)]
PS_TAUS = (0.5, 0.7, 0.9, 0.97)
PS_ROUNDS = (1, 2)
PS_ORACLE_GATE = 0.010
PS_DEV_GATE = 0.003
ORACLE_GROUPS = {"opening": [0], "category": [1], "L12": [2, 3], "enc": [4, 5, 6], "LLR": [7], "all": list(range(8))}


def pseudo_base() -> dict:
    """the fusion the expansion stacks on: plan-2 colour weights if they passed the switch rule,
    else the 821010-comb-probe FULL weights."""
    r7 = COL_OUT / "report.json"
    if r7.exists():
        rep = json.loads(r7.read_text())
        if rep.get("sealed", {}).get("verdict", {}).get("pass"):
            return {"kind": "colour", "W": {int(k): np.array(v, np.float64) for k, v in rep["chosen"]["weights"].items()}}
    rep = json.loads((COMB_OUT / "report.json").read_text())
    return {"kind": "comb", "W": {int(k): np.array(v, np.float64) for k, v in rep["chosen"]["weights"].items()}}


def fuse(base: dict, X: np.ndarray, ks: np.ndarray, z: np.ndarray) -> np.ndarray:
    return apply_colour(X, ks, z, base["W"]) if base["kind"] == "colour" else apply_weights(X, ks, base["W"])


def posterior_assign(S: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """(assignment [Q] = top-1 candidate where its softmax posterior >= tau else -1, top-1 posterior)."""
    e = np.exp(S - S.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    top = p.argmax(axis=1)
    p1 = p[np.arange(len(p)), top]
    return np.where(p1 >= tau, top, -1), p1


def expand(comp, gallery, query, assign, X0: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, list]:
    """X [Q, C, 8] with every assigned question's games added to its candidate's gallery.
    Leave-one-out: a question's own games never score that question (only cell (q, assign[q])
    can contain them; it is recomputed from the expanded gallery minus q's games). Columns of
    unchanged candidates are copied from X0 (every part depends only on its own candidate once
    the pooled category prior is fixed). comp(gallery_list, query_list, cat_prior) -> [Q', G', 8]."""
    aug = [np.asarray(g) for g in gallery]
    for q, c in enumerate(assign):
        if c >= 0:
            aug[c] = np.concatenate([aug[c], np.asarray(query[q])])
    changed = np.unique(assign[assign >= 0])
    X = X0.copy()
    if not len(changed):
        return X, aug
    prior = category_prior(codes, aug)
    sel = np.flatnonzero(assign >= 0)
    special = [np.setdiff1d(aug[assign[q]], np.asarray(query[q])) for q in sel]
    for q, sp in zip(sel, special):
        assert len(sp) == len(aug[assign[q]]) - len(query[q]), "a question's games overlap a gallery"
    Y = comp([aug[c] for c in changed] + special, list(query), prior)     # [Q, n_changed + P, 8]
    X[:, changed] = Y[:, :len(changed)]
    X[sel, assign[sel]] = Y[sel, len(changed) + np.arange(len(sel))]
    return X, aug


def evaluate8(_args) -> None:
    """pseudo-label expansion of every fusion part: oracle bound (go / no-go), dev choice of tau and
    rounds, sealed test vs the current fusion."""
    import torch

    from scripts.identity.pid2_cheap import COLS, buckets, official, paired, verdict

    PS_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    fp = json.loads((EB_OUT / "report.json").read_text())["final_params"]
    base = pseudo_base()
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, device, env = ctx["report"], ctx["draw"], ctx["llr"], ctx["device"], ctx["env"]
    report["base"] = base["kind"]
    print("base fusion:", base["kind"], flush=True)
    color, codes, components = env["color"], ctx["codes"], ctx["components"]
    mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    prior = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    P = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)

    def comp(gallery, query, cat_prior=None):
        parts = components(gallery, query, cat_prior)
        le = llr({"gallery": gallery, "query": query}, 1.0, mask, prior, P)[0]
        return np.stack([parts[c] for c in PARTS[:-1]] + [le], axis=2)

    def get(pool, d):
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        X = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le], axis=2)
        z = 2 * np.array([float((color[np.asarray(g)] == 0).mean()) for g in x["query"]]) - 1
        return x, X, z

    def arm(x, X0, z, S0, tau, rounds):
        """expanded X after `rounds` rounds starting from the base posterior."""
        S, out, a = S0, None, None
        for _ in range(rounds):              # each round re-selects from the original galleries
            a, _p = posterior_assign(S, tau)
            out, _ = expand(comp, x["gallery"], x["query"], a, X0, codes)
            S = fuse(base, out, x["ks"], z)
        return out, a

    def rows_of(tag, d, pool, X, x, z, a_true_aug):
        pts = official(fuse(base, X, x["ks"], z), x["owner"])
        r = {"pool": pool, "draw": d, "arm": tag, **buckets(pts, x["ks"])}
        if a_true_aug is not None:          # bias check: true owner expanded or not
            r["pts_aug"], r["n_aug"] = float(pts[a_true_aug].sum()), int(a_true_aug.sum())
            r["pts_noaug"], r["n_noaug"] = float(pts[~a_true_aug].sum()), int((~a_true_aug).sum())
        return r, pts

    # ---- dev: engineering checks, oracle, tau x rounds
    rows, quality, first = [], [], True
    for pool, d in R8_DEV:
        x, X0, z = get(pool, d)
        S0 = fuse(base, X0, x["ks"], z)
        if first:
            first = False
            Xr = comp(x["gallery"], x["query"])
            diff = {c: float(np.abs(Xr[:, :, j] - X0[:, :, j]).max()) for j, c in enumerate(PARTS)}
            report["comp_reproduction"] = diff
            print("comp() reproduces the draw parts, max abs diff per part:", diff, flush=True)
            assert max(v for c, v in diff.items() if c != "LLR") <= 1e-9 and diff["LLR"] <= 1e-6
            idx = np.arange(0, len(x["gallery"]), 7)
            Xs = comp([x["gallery"][i] for i in idx], x["query"], category_prior(codes, x["gallery"]))
            col = {c: float(np.abs(Xs[:, :, j] - X0[:, idx, j]).max()) for j, c in enumerate(PARTS)}
            report["column_independence"] = col
            print("column independence (subset of candidates), max abs diff per part:", col, flush=True)
            assert max(v for c, v in col.items() if c != "LLR") <= 1e-9 and col["LLR"] <= 1e-3
            none, _ = expand(comp, x["gallery"], x["query"], np.full(len(x["query"]), -1), X0, codes)
            assert np.array_equal(none, X0)
        r, base_pts = rows_of("BASE", d, pool, X0, x, z, None)
        rows.append(r)
        Xo, _ = expand(comp, x["gallery"], x["query"], x["owner"].copy(), X0, codes)
        for g, cols in ORACLE_GROUPS.items():
            Xm = X0.copy()
            Xm[:, :, cols] = Xo[:, :, cols]
            rows.append(rows_of(f"ORACLE_{g}", d, pool, Xm, x, z, None)[0])
        for tau in PS_TAUS:
            a, p1 = posterior_assign(S0, tau)
            for k in KS:
                m = x["ks"] == k
                quality.append({"draw": d, "tau": tau, "k": k, "n": int(m.sum()), "picked": int((a[m] >= 0).sum()),
                                "correct": int((a[m] == x["owner"][m]).sum())})
            S, rr_done = S0, 0                 # rounds share their prefix: round r+1 starts from round r's scores
            for rr in PS_ROUNDS:
                while rr_done < rr:
                    a1, _p = posterior_assign(S, tau)
                    X1, _ = expand(comp, x["gallery"], x["query"], a1, X0, codes)
                    S, rr_done = fuse(base, X1, x["ks"], z), rr_done + 1
                aug_true = np.isin(x["owner"], np.unique(a1[a1 >= 0]))
                rows.append(rows_of(f"T{tau:g}_R{rr}", d, pool, X1, x, z, aug_true)[0])
        print(f"dev draw {d}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(PS_OUT / "dev_per_draw.csv", index=False)
    s1 = paired(frame, "BASE").set_index("arm")
    q = pd.DataFrame(quality).groupby(["tau", "k"])[["n", "picked", "correct"]].sum()
    q["coverage"], q["precision"] = q["picked"] / q["n"], q["correct"] / q["picked"].clip(lower=1)
    report["dev"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["pseudo_quality"] = {f"{t:g}_{k}": r.to_dict() for (t, k), r in q.iterrows()}
    print("dev (vs BASE):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          "\npseudo-label quality on dev:\n", q.round(3).to_string(), flush=True)
    bias = frame[frame.arm.str.startswith("T")].groupby("arm")[["pts_aug", "n_aug", "pts_noaug", "n_noaug"]].sum()
    b0 = frame[frame.arm == "BASE"]
    report["bias_rows"] = bias.to_dict(orient="index")
    oracle = float(s1.loc["ORACLE_all", "d_all"])
    report["oracle_all"] = oracle
    if oracle < PS_ORACLE_GATE:
        report["stopped"] = f"oracle bound {oracle:.4f} < {PS_ORACLE_GATE}; stage 0 gate failed"
        (PS_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    arms = s1.loc[[f"T{t:g}_R{r}" for t in PS_TAUS for r in PS_ROUNDS]]
    best = arms["d_all"].idxmax()
    tau, rounds = float(best[1:].split("_R")[0]), int(best.split("_R")[1])
    report["chosen"] = {"arm": best, "tau": tau, "rounds": rounds, "dev_delta": float(arms.loc[best, "d_all"]),
                        "grid_edge": tau in (PS_TAUS[0], PS_TAUS[-1])}
    print("chosen:", report["chosen"], flush=True)
    if arms.loc[best, "d_all"] < PS_DEV_GATE:
        report["stopped"] = f"best dev delta {arms.loc[best, 'd_all']:.4f} < {PS_DEV_GATE}; sealed draws not opened"
        (PS_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    (PS_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- sealed
    out, bias_s = [], []
    for pool, d in R8_SEALED:
        x, X0, z = get(pool, d)
        S0 = fuse(base, X0, x["ks"], z)
        X1, a1 = arm(x, X0, z, S0, tau, rounds)
        aug_true = np.isin(x["owner"], np.unique(a1[a1 >= 0]))
        r0, p0 = rows_of("BASE", d, pool, X0, x, z, aug_true)
        r1, p1_ = rows_of("PSEUDO", d, pool, X1, x, z, aug_true)
        out += [r0, r1]
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(PS_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "BASE"), paired(frame[frame.pool == 0], "BASE")
    v = verdict(t1, t0, "PSEUDO")
    v["probe"] = bool(not v["pass"] and v["pool1_delta"] > 0 and v["pool0_delta"] > 0)
    g = frame.groupby(["pool", "arm"])[["pts_aug", "n_aug", "pts_noaug", "n_noaug"]].sum()
    g["mean_aug"], g["mean_noaug"] = g["pts_aug"] / g["n_aug"], g["pts_noaug"] / g["n_noaug"]
    report["sealed_bias"] = {f"{p}_{a}": r.to_dict() for (p, a), r in g.iterrows()}
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"), "verdict": v}
    print("sealed pool 1 vs BASE:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs BASE:\n",
          t0.round(4).to_string(index=False), "\nbias (true owner expanded / not):\n",
          g[["mean_aug", "n_aug", "mean_noaug", "n_noaug"]].round(4).to_string(), "\nverdict:", json.dumps(v), flush=True)
    (PS_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


def pseudo_selfcheck() -> None:
    """synthetic: leave-one-out keeps a question's own games out of its own cell; unassigned
    questions and unchanged candidates keep X0; posterior_assign thresholds correctly."""
    rng = np.random.default_rng(4)
    gallery = [np.arange(10 * c, 10 * c + 10) for c in range(6)]
    query = [np.array([100 + 3 * q, 101 + 3 * q, 102 + 3 * q]) for q in range(8)]

    def comp(gal, qry, cat_prior=None):        # part 0 = overlap count, part 1 = gallery size
        X = np.zeros((len(qry), len(gal), 8))
        for i, q in enumerate(qry):
            for j, g in enumerate(gal):
                X[i, j, 0] = len(np.intersect1d(q, g))
                X[i, j, 1] = len(g)
        return X

    codes = rng.integers(0, 4, 200)
    X0 = comp(gallery, query)
    assign = np.array([2, -1, 2, 5, -1, -1, 0, -1])
    X, aug = expand(comp, gallery, query, assign, X0, codes)
    assert X[:, :, 0].max() == 0, "a question's own games scored it"
    assert X[0, 2, 1] == 13 and X[1, 2, 1] == 16 and X[3, 5, 1] == 10 and X[4, 5, 1] == 13
    assert np.array_equal(X[:, [1, 3, 4]], X0[:, [1, 3, 4]])
    assert len(aug[2]) == 16 and len(aug[5]) == 13 and len(aug[0]) == 13
    S = np.array([[0.0, 5.0, 0.0], [1.0, 1.0, 1.0]])
    a, p1 = posterior_assign(S, 0.9)
    assert a.tolist() == [1, -1] and p1[0] > 0.98
    print("pseudo selfcheck ok", flush=True)


# ---------------------------------------------------------------- official export
def position_tensors(arr: dict, moves_of, device, rich: dict | None = None):
    """sorted positions -> (B, PHI, SLOT tensors, first / end position per game id); rich = the
    likelihood_rich arrays (game, ply, rich) in the same sorted order -> 50-dim features."""
    import torch
    order = np.lexsort((arr["ply"], arr["game"]))
    arr = {k: v[order] for k, v in arr.items()}
    n = len(order)
    if rich is not None:
        assert np.array_equal(rich["game"], arr["game"]) and np.array_equal(rich["ply"], arr["ply"]), "rich order"
    last_opp = np.full(n, -1, np.int64)
    last_own = np.full(n, -1, np.int64)
    starts = np.flatnonzero(np.r_[True, arr["game"][1:] != arr["game"][:-1]])
    ends = np.r_[starts[1:], n]
    span = {}
    for s, e in zip(starts, ends):
        g = int(arr["game"][s])
        moves = moves_of(g).astype(np.int64)
        ply = arr["ply"][s:e].astype(np.int64)
        last_opp[s:e] = np.where(ply >= 1, moves[np.maximum(ply - 1, 0)], -1)
        last_own[s:e] = np.where(ply >= 2, moves[np.maximum(ply - 2, 0)], -1)
        span[g] = (s, e)
    bs, ps = [], []
    for s in range(0, n, 500_000):
        b, phi = features(arr["top"][s:s + 500_000], arr["lp"][s:s + 500_000], arr["ply"][s:s + 500_000],
                          last_opp[s:s + 500_000], last_own[s:s + 500_000],
                          None if rich is None else rich["rich"][s:s + 500_000])
        bs.append(torch.from_numpy(b))
        ps.append(torch.from_numpy(phi))
    return (torch.cat(bs).to(device), torch.cat(ps).to(device),
            torch.from_numpy(arr["slot"].astype(np.int64)).to(device), span)


# ---------------------------------------------------------------- pseudo-label retune + weight refit (2026-11-21, plan A after 824896)
PS2_OUT = OUT / "pseudo2"
R9_DEV = [(0, d) for d in range(4200, 4220)]
R9_SEALED = [(0, d) for d in range(4300, 4320)] + [(1, d) for d in range(4300, 4330)]
PS2_TAUS = (0.2, 0.3, 0.4, 0.5)
PS2_ROUNDS = (2, 3)
PS2_CUR = "T0.5_R2"                      # the current base (824896-comb-pseudo)


def evaluate9(_args) -> None:
    """lower tau / one more round for the pseudo-label expansion, then a cross-fitted refit of the
    per-k fusion weights on the expanded components; sealed test vs the current base (tau 0.5, 2 rounds,
    current weights)."""
    import torch

    from scripts.identity.pid2_cheap import buckets, official, paired, verdict

    PS2_OUT.mkdir(parents=True, exist_ok=True)
    stage0 = json.loads((OUT / "report.json").read_text())
    fp = json.loads((EB_OUT / "report.json").read_text())["final_params"]
    r8 = json.loads((PS_OUT / "report.json").read_text())
    assert r8["base"] == "comb" and r8["chosen"]["arm"] == PS2_CUR and r8["sealed"]["verdict"]["pass"]
    base = pseudo_base()
    assert base["kind"] == "comb"
    W0 = base["W"]
    ctx = _context(stage0["theta_pop"], rich=True)
    report, draw, llr, device, env = ctx["report"], ctx["draw"], ctx["llr"], ctx["device"], ctx["env"]
    color, codes, components = env["color"], ctx["codes"], ctx["components"]
    mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    prior = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    P = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)

    def comp(gallery, query, cat_prior=None):
        parts = components(gallery, query, cat_prior)
        le = llr({"gallery": gallery, "query": query}, 1.0, mask, prior, P)[0]
        return np.stack([parts[c] for c in PARTS[:-1]] + [le], axis=2)

    def get(pool, d):
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        X = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le], axis=2)
        z = 2 * np.array([float((color[np.asarray(g)] == 0).mean()) for g in x["query"]]) - 1
        return x, X, z

    def configs(x, X0, z, wanted):
        """{(tau, rounds): (X, assignment of the last round)} for wanted = {tau: [rounds]}; rounds share prefixes."""
        out = {}
        S0 = fuse(base, X0, x["ks"], z)
        for tau, rs in wanted.items():
            S, done, a, X1 = S0, 0, None, None
            for r in sorted(rs):
                while done < r:
                    a, _p = posterior_assign(S, tau)
                    X1, _ = expand(comp, x["gallery"], x["query"], a, X0, codes)
                    S, done = fuse(base, X1, x["ks"], z), done + 1
                out[(tau, r)] = (X1, a)
        return out

    def row(tag, pool, d, X, x, z, W, a):
        pts = official(apply_weights(X, x["ks"], W), x["owner"])
        aug = np.isin(x["owner"], np.unique(a[a >= 0])) if a is not None else np.zeros(len(pts), bool)
        return {"pool": pool, "draw": d, "arm": tag, **buckets(pts, x["ks"]),
                "pts_aug": float(pts[aug].sum()), "n_aug": int(aug.sum()),
                "pts_noaug": float(pts[~aug].sum()), "n_noaug": int((~aug).sum())}

    name = lambda tau, r: f"T{tau:g}_R{r}"
    grid = {t: list(PS2_ROUNDS) for t in PS2_TAUS}
    dev, rows, quality = {}, [], []
    for pool, d in R9_DEV:
        x, X0, z = get(pool, d)
        out = configs(x, X0, z, grid)
        for (tau, r), (X1, a) in out.items():
            rows.append(row(name(tau, r), pool, d, X1, x, z, W0, a))
            if r == PS2_ROUNDS[0]:
                for k in KS:
                    m = x["ks"] == k
                    quality.append({"tau": tau, "k": k, "n": int(m.sum()), "picked": int((a[m] >= 0).sum()),
                                    "correct": int((a[m] == x["owner"][m]).sum())})
        dev[d] = {"ks": x["ks"], "owner": x["owner"], "X": {key: v[0].astype(np.float32) for key, v in out.items()}}
        print(f"dev draw {d}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(PS2_OUT / "dev_per_draw.csv", index=False)
    s1 = paired(frame, PS2_CUR).set_index("arm")
    q = pd.DataFrame(quality).groupby(["tau", "k"])[["n", "picked", "correct"]].sum()
    q["coverage"], q["precision"] = q["picked"] / q["n"], q["correct"] / q["picked"].clip(lower=1)
    report["dev"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["pseudo_quality_last_round"] = {f"{t:g}_{k}": r.to_dict() for (t, k), r in q.iterrows()}
    bias = frame.groupby("arm")[["pts_aug", "n_aug", "pts_noaug", "n_noaug"]].sum()
    bias["mean_aug"], bias["mean_noaug"] = bias["pts_aug"] / bias["n_aug"], bias["pts_noaug"] / bias["n_noaug"]
    report["dev_bias"] = bias.to_dict(orient="index")
    print("dev (vs current tau 0.5 / 2 rounds):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          f"\npseudo-label quality (round {PS2_ROUNDS[0]}):\n", q.round(3).to_string(),
          "\nbias (true owner expanded / not):\n", bias[["mean_aug", "n_aug", "mean_noaug", "n_noaug"]].round(4).to_string(), flush=True)
    best = s1["d_all"].idxmax()
    key = (float(best[1:].split("_R")[0]), int(best.split("_R")[1]))
    report["chosen_config"] = {"arm": best, "tau": key[0], "rounds": key[1], "dev_delta": float(s1.loc[best, "d_all"]),
                               "grid_edge": key[0] == PS2_TAUS[0] or key[1] == PS2_ROUNDS[-1]}
    print("chosen config:", report["chosen_config"], flush=True)

    # ---- cross-fitted refit of the per-k weights on the chosen expanded components
    names = [d for _, d in R9_DEV]
    halves = (names[:10], names[10:])

    def fit_all(draws):
        return {k: fit_combiner([(dev[d]["X"][key][dev[d]["ks"] == k].astype(np.float64), dev[d]["owner"][dev[d]["ks"] == k])
                                 for d in draws], W0[k], "FULL", device) for k in KS}

    rr = []
    for fit_on, held in (halves, halves[::-1]):
        Wf = fit_all(fit_on)
        for d in held:
            X = dev[d]["X"][key].astype(np.float64)
            for tag, W in (("CURW", W0), ("REFIT", Wf)):
                rr.append({"draw": d, "arm": tag, **buckets(official(apply_weights(X, dev[d]["ks"], W), dev[d]["owner"]), dev[d]["ks"])})
    s2 = paired(pd.DataFrame(rr), "CURW").set_index("arm")
    report["refit_crossfit"] = s2[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    print("refit cross-fit on the chosen config (vs current weights):\n",
          s2[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(), flush=True)
    refit = bool(s2.loc["REFIT", "d_all"] > 0)
    W = fit_all(names) if refit else W0
    report["refit"] = {"used": refit, "weights": {k: W[k].tolist() for k in KS},
                       "ratio_to_current": {k: np.round(W[k] / W0[k], 3).tolist() for k in KS}}
    if refit:
        print("refit weights / current:", report["refit"]["ratio_to_current"], flush=True)
    if best == PS2_CUR and not refit:
        report["stopped"] = "the current setting won the dev draws and the refit did not help; sealed draws not opened"
        (PS2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    (PS2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    del dev

    # ---- sealed: CUR (tau 0.5 / 2 rounds, current weights), TAU (chosen config, current weights), FINAL
    want = {0.5: [2]}
    want.setdefault(key[0], [])
    if key[1] not in want[key[0]]:
        want[key[0]].append(key[1])
    out = []
    for pool, d in R9_SEALED:
        x, X0, z = get(pool, d)
        got = configs(x, X0, z, want)
        Xc, ac = got[(0.5, 2)]
        Xk, ak = got[key]
        out += [row("CUR", pool, d, Xc, x, z, W0, ac), row("TAU", pool, d, Xk, x, z, W0, ak),
                row("FINAL", pool, d, Xk, x, z, W, ak)]
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(PS2_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "CUR"), paired(frame[frame.pool == 0], "CUR")
    v = verdict(t1, t0, "FINAL")
    v["probe"] = bool(not v["pass"] and v["pool1_delta"] > 0 and v["pool0_delta"] > 0)
    g = frame.groupby(["pool", "arm"])[["pts_aug", "n_aug", "pts_noaug", "n_noaug"]].sum()
    g["mean_aug"], g["mean_noaug"] = g["pts_aug"] / g["n_aug"], g["pts_noaug"] / g["n_noaug"]
    report["sealed_bias"] = {f"{p}_{a}": r_.to_dict() for (p, a), r_ in g.iterrows()}
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"), "verdict": v}
    print("sealed pool 1 vs CUR:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs CUR:\n",
          t0.round(4).to_string(index=False), "\nbias:\n", g[["mean_aug", "n_aug", "mean_noaug", "n_noaug"]].round(4).to_string(),
          "\nverdict:", json.dumps(v), flush=True)
    (PS2_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- shared by items 1 / 2 after 829297 (2026-11-21)
PARTS9 = PARTS + ("RLLR",)


def question_thetas(PHI, B, SLOT, positions_of, query, prior, mask, P):
    """MAP theta of every question on its own games, under the same EB prior as the candidates."""
    pq, oq = positions_of(query)
    return fit(PHI, B, SLOT, pq, oq, len(query), prior, 1.0, mask=mask, precision=P)


def reverse_llr(PHI, B, SLOT, positions_of, gallery, theta_q, prior, mask):
    """[Q, C]: sum over candidate c's reference moves of log p(m | theta_q) - log p(m | theta_pop)."""
    import torch
    pg, og = positions_of(gallery)
    lp = played_logp(PHI, B, SLOT, pg, torch.cat([theta_q, prior[None]]), mask=mask)
    out = torch.zeros(len(gallery), len(theta_q), device=PHI.device, dtype=torch.float64)
    out.index_add_(0, og, lp[:, :-1] - lp[:, -1:])
    return out.T.cpu().numpy()


def pair_llr(PHI, B, SLOT, positions_of, query, theta_q, prior, mask):
    """([Q, Q] sum over q's moves of log p(m | theta_q') - log p(m | theta_pop), [Q] own-move counts)."""
    import torch
    pq, oq = positions_of(query)
    lp = played_logp(PHI, B, SLOT, pq, torch.cat([theta_q, prior[None]]), mask=mask)
    out = torch.zeros(len(query), len(theta_q), device=PHI.device, dtype=torch.float64)
    out.index_add_(0, oq, lp[:, :-1] - lp[:, -1:])
    return out.cpu().numpy(), np.bincount(oq.cpu().numpy(), minlength=len(query)).astype(np.float64)


def fit_free(sets, w0: np.ndarray, free: list[int], device) -> np.ndarray:
    """conditional logit for one k with only the weights at `free` fitted (the rest stay at w0)."""
    import torch
    Xs = [(torch.from_numpy(x).to(device), torch.from_numpy(o).to(device)) for x, o in sets]
    n = sum(len(o) for _, o in sets)
    base = torch.tensor(w0, device=device, dtype=torch.float64)
    idx = torch.tensor(free, device=device, dtype=torch.long)
    p = base[idx].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=500, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        w = base.index_put((idx,), p)
        loss = 0.0
        for x, o in Xs:
            s = torch.einsum("qcj,j->qc", x, w)
            loss = loss + (torch.logsumexp(s, dim=1) - s[torch.arange(len(o), device=device), o]).sum()
        loss = loss / n
        loss.backward()
        return loss

    opt.step(closure)
    return base.index_put((idx,), p.detach()).cpu().numpy()


def pseudo_rounds(comp, gallery, query, X0, ks, W, codes, tau=0.5, rounds=2):
    """(X after the last round, last assignment, final scores); each round re-selects from the
    original galleries (as evaluate8 / evaluate9 / export)."""
    S, X1, a = apply_weights(X0, ks, W), X0, np.full(len(query), -1)
    for _ in range(rounds):
        a, _p = posterior_assign(S, tau)
        X1, _ = expand(comp, gallery, query, a, X0, codes)
        S = apply_weights(X1, ks, W)
    return X1, a, S


def _llr_setup():
    """context plus the EB-LLR settings and the comb weights the 824896 base used."""
    import torch
    stage0 = json.loads((OUT / "report.json").read_text())
    fp = json.loads((EB_OUT / "report.json").read_text())["final_params"]
    r8 = json.loads((PS_OUT / "report.json").read_text())
    assert r8["base"] == "comb" and r8["chosen"]["arm"] == "T0.5_R2" and r8["sealed"]["verdict"]["pass"]
    base = pseudo_base()
    assert base["kind"] == "comb"
    ctx = _context(stage0["theta_pop"], rich=True)
    device = ctx["device"]
    ctx["mask"] = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    ctx["prior"] = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
    ctx["P"] = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)
    ctx["W8"] = base["W"]
    return ctx


# ---------------------------------------------------------------- item 1: reverse LLR (symmetric scoring)
RL_OUT = OUT / "reverse"
R10_DEV = [(0, d) for d in range(4400, 4420)]
R10_SEALED = [(0, d) for d in range(4500, 4520)] + [(1, d) for d in range(4500, 4530)]


def evaluate10(_args) -> None:
    """reverse LLR as a 9th fusion part: cross-fitted ADD1 / FULL9 weights on dev draws, sealed test
    of 9 parts + pseudo-labels against the 824896 form."""
    from scripts.identity.pid2_cheap import buckets, official, paired, verdict

    RL_OUT.mkdir(parents=True, exist_ok=True)
    ctx = _llr_setup()
    report, draw, llr, device, env = ctx["report"], ctx["draw"], ctx["llr"], ctx["device"], ctx["env"]
    codes, components = ctx["codes"], ctx["components"]
    PHI, B, SLOT, positions_of = ctx["PHI"], ctx["B"], ctx["SLOT"], ctx["positions_of"]
    mask, prior, P, W8 = ctx["mask"], ctx["prior"], ctx["P"], ctx["W8"]
    W0 = {k: np.r_[W8[k], 0.0] for k in KS}

    def comp_for(thq, with_rl: bool):
        def comp(gallery, query, cat_prior=None):
            parts = components(gallery, query, cat_prior)
            le = llr({"gallery": gallery, "query": query}, 1.0, mask, prior, P)[0]
            cols = [parts[c] for c in PARTS[:-1]] + [le]
            if with_rl:
                assert len(query) == len(thq)
                cols.append(reverse_llr(PHI, B, SLOT, positions_of, gallery, thq, prior, mask))
            return np.stack(cols, axis=2)
        return comp

    def get(pool, d):
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        thq = question_thetas(PHI, B, SLOT, positions_of, x["query"], prior, mask, P)
        rl = reverse_llr(PHI, B, SLOT, positions_of, x["gallery"], thq, prior, mask)
        X = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le, rl], axis=2)
        return x, X, thq

    def corr_rows(X):
        a, b = X[:, :, 7], X[:, :, 8]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        return float(((a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-300)).mean())

    # ---- dev draws (no expansion): engineering checks, cross-fit
    dev, rows, first = {}, [], True
    for pool, d in R10_DEV:
        x, X, thq = get(pool, d)
        if first:
            first = False
            Xr = comp_for(thq, True)(x["gallery"], x["query"])
            diff = {c: float(np.abs(Xr[:, :, j] - X[:, :, j]).max()) for j, c in enumerate(PARTS9)}
            report["comp_reproduction"] = diff
            print("comp() reproduces the draw parts:", diff, flush=True)
            assert max(v for c, v in diff.items() if c not in ("LLR", "RLLR")) <= 1e-9
            assert diff["LLR"] <= 1e-6 and diff["RLLR"] <= 1e-6
            idx = np.arange(0, len(x["gallery"]), 7)
            sub = reverse_llr(PHI, B, SLOT, positions_of, [x["gallery"][i] for i in idx], thq, prior, mask)
            report["rllr_column_independence"] = float(np.abs(sub - X[:, idx, 8]).max())
            print("RLLR column independence:", report["rllr_column_independence"], flush=True)
            assert report["rllr_column_independence"] <= 1e-6
            same = np.array_equal(np.argsort(-apply_weights(X, x["ks"], W0), axis=1)[:, :5],
                                  np.argsort(-apply_weights(X[:, :, :8], x["ks"], W8), axis=1)[:, :5])
            report["zero_weight_top5_identical"] = bool(same)
            assert same, "RLLR weight 0 changes the top-5"
        rows.append({"draw": d, "arm": "LLR_alone", **buckets(official(X[:, :, 7], x["owner"]), x["ks"])})
        rows.append({"draw": d, "arm": "RLLR_alone", **buckets(official(X[:, :, 8], x["owner"]), x["ks"])})
        rows.append({"draw": d, "arm": "CURRENT", **buckets(official(apply_weights(X, x["ks"], W0), x["owner"]), x["ks"])})
        report.setdefault("llr_rllr_corr", []).append(corr_rows(X))
        dev[d] = (X, x["ks"], x["owner"])
        print(f"dev draw {d}", flush=True)
    alone = paired(pd.DataFrame(rows), "CURRENT").set_index("arm")
    report["alone"] = alone[["1", "2", "3", "5", "all"]].to_dict(orient="index")
    report["llr_rllr_corr_mean"] = float(np.mean(report["llr_rllr_corr"]))
    print("alone scores (dev):\n", alone[["1", "2", "3", "5", "all"]].round(4).to_string(),
          "\nmean within-question corr(LLR, RLLR):", round(report["llr_rllr_corr_mean"], 3), flush=True)

    def fit_all(draws, arm):
        free = [8] if arm == "ADD1" else list(range(9))
        return {k: fit_free([(dev[d][0][dev[d][1] == k], dev[d][2][dev[d][1] == k]) for d in draws], W0[k], free, device)
                for k in KS}

    names = [d for _, d in R10_DEV]
    halves = (names[:10], names[10:])
    rows, cross = [], {}
    for i, (fit_on, held) in enumerate((halves, halves[::-1])):
        arms = {"CURRENT": W0, **{a: fit_all(fit_on, a) for a in ("ADD1", "FULL9")}}
        cross[i] = {a: {k: v.tolist() for k, v in W.items()} for a, W in arms.items() if a != "CURRENT"}
        for d in held:
            X, ks, owner = dev[d]
            rows += [{"draw": d, "arm": a, **buckets(official(apply_weights(X, ks, W), owner), ks)} for a, W in arms.items()]
    s1 = paired(pd.DataFrame(rows), "CURRENT").set_index("arm")
    report["crossfit"] = s1[["1", "2", "3", "5", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all", "pos"]].to_dict(orient="index")
    report["crossfit_weights"] = cross
    print("cross-fit (vs CURRENT):\n", s1[["all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(), flush=True)
    best = s1.drop(index="CURRENT")["d_all"].idxmax()
    if s1.loc[best, "d_all"] <= 0:
        report["stopped"] = "neither ADD1 nor FULL9 beat the current weights in the cross-fit; sealed draws not opened"
        (RL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
        print(report["stopped"], flush=True)
        return
    W9 = fit_all(names, best)
    report["chosen"] = {"arm": best, "crossfit_delta": float(s1.loc[best, "d_all"]),
                        "weights": {k: W9[k].tolist() for k in KS},
                        "rllr_weight": {k: float(W9[k][8]) for k in KS},
                        "negative": [{"k": k, "part": c, "w": float(W9[k][j])} for k in KS for j, c in enumerate(PARTS9) if W9[k][j] < 0]}
    print("chosen:", best, "RLLR weight per k:", report["chosen"]["rllr_weight"], flush=True)
    (RL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    del dev

    # ---- sealed: CUR (8 parts + pseudo, = 824896 form) vs NEW (9 parts + pseudo); no-expansion arms for attribution
    out = []
    for pool, d in R10_SEALED:
        x, X, thq = get(pool, d)
        ks, owner = x["ks"], x["owner"]
        Xc, _, Sc = pseudo_rounds(comp_for(thq, False), x["gallery"], x["query"], X[:, :, :8], ks, W8, codes)
        Xn, _, Sn = pseudo_rounds(comp_for(thq, True), x["gallery"], x["query"], X, ks, W9, codes)
        for arm, S in (("CUR", Sc), ("NEW", Sn), ("CUR_noexp", apply_weights(X[:, :, :8], ks, W8)),
                       ("NEW_noexp", apply_weights(X, ks, W9))):
            out.append({"pool": pool, "draw": d, "arm": arm, **buckets(official(S, owner), ks)})
        print(f"sealed pool {pool} draw {d}", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(RL_OUT / "sealed_per_draw.csv", index=False)
    t1, t0 = paired(frame[frame.pool == 1], "CUR"), paired(frame[frame.pool == 0], "CUR")
    v = verdict(t1, t0, "NEW")
    v["probe"] = bool(not v["pass"] and v["pool1_delta"] > 0 and v["pool0_delta"] > 0)
    report["sealed"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"), "verdict": v}
    print("sealed pool 1 vs CUR:\n", t1.round(4).to_string(index=False), "\nsealed pool 0 vs CUR:\n",
          t0.round(4).to_string(index=False), "\nverdict:", json.dumps(v), flush=True)
    (RL_OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- item 2: question merging v2 (LLR in the verifier)
QM2_OUT = OUT / "qmerge2"
R11_FIT = [(0, d) for d in range(4600, 4620)]
R11_TUNE = [(0, d) for d in range(4620, 4640)]
R11_SEAL0 = [(0, d) for d in range(4640, 4660)]
R11_SEAL1 = [(1, d) for d in range(4600, 4630)]
QM2_TAUS = (1.0, 2.0, 4.0, 8.0, 16.0)
QM2_BETAS = (0.125, 0.25, 0.5, 1.0)
QM2_THETAS = (0.0, 0.1, 0.3, 0.5)
QM2_PAIR_PARTS = ("opening", "category", "L1", "L2", "B", "A1", "A2")


def qm2_features(d: dict) -> np.ndarray:
    """[Q, Q, F] symmetric pair features from the saved per-draw arrays (bench and official alike)."""
    from scripts.identity.pid2_qmerge import centred, softmax
    s, k = d["S"], d["k"].astype(np.float64)
    lk = np.log(k)
    a, b = np.meshgrid(lk, lk, indexing="ij")
    sym = lambda x: 0.5 * (x + x.T)
    cos = [sym(d[n].astype(np.float64)) for n in ("B", "A1", "A2")]
    mc = sum(cos) / 3
    out = cos + [mc * (a + b), mc * a * b]
    out += [centred(d[n]) for n in ("opening", "L1", "L2", "category")]
    lp = d["LLRp"].astype(np.float64)
    out += [centred(lp), centred(lp / np.maximum(d["n_moves"], 1)[:, None])]
    for tau in (1.0, 8.0):
        p = softmax(s, tau)
        out.append(np.log(np.maximum(p @ p.T, 1e-300)))
    top = s.argmax(axis=1)
    asg = d["assign"]
    out += [(top[:, None] == top[None, :]).astype(np.float64),
            ((asg[:, None] == asg[None, :]) & (asg[:, None] >= 0)).astype(np.float64), a + b, a * b]
    return np.stack([np.asarray(x, np.float64) for x in out], axis=-1)


def qm2_pair_m(verifier, d: dict) -> np.ndarray:
    f = qm2_features(d)
    q = len(d["k"])
    m = verifier(f.reshape(-1, f.shape[-1])).reshape(q, q)
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return m


def qm2_apply(S: np.ndarray, m: np.ndarray, target: np.ndarray, tau: float, beta: float, theta: float) -> np.ndarray:
    from scripts.identity.pid2_qmerge import merge_term
    out = S.copy()
    if target.any():
        out[target] += beta * merge_term(S, m, tau, theta)[target]
    return out


def qm2_fit(draws):
    """(verifiers, saved state) fitted on the FIT draws' upper-triangle pairs."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    xs, ys = [], []
    for pool, d in draws:
        dd = dict(np.load(QM2_OUT / "draws" / f"pool{pool}_draw{d}.npz"))
        iu = np.triu_indices(len(dd["k"]), 1)
        xs.append(qm2_features(dd)[iu])
        ys.append((dd["owner"][:, None] == dd["owner"][None, :])[iu])
    x, y = np.concatenate(xs), np.concatenate(ys)
    mu, sd = x.mean(0), x.std(0) + 1e-12
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((x - mu) / sd, y)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=2026).fit(x, y)
    print(f"verifier fit: {len(y)} pairs, positive rate {y.mean():.5f}", flush=True)
    fns = {"lr": lambda f: lr.predict_proba((f - mu) / sd)[:, 1], "hgb": lambda f: hgb.predict_proba(f)[:, 1]}
    return fns, {"lr": {"model": lr, "mu": mu, "sd": sd}, "hgb": {"model": hgb}}


def qm2_load_verifier():
    import pickle
    state = pickle.loads((QM2_OUT / "verifier.pkl").read_bytes())
    if state["name"] == "lr":
        lr, mu, sd = state["model"], state["mu"], state["sd"]
        return lambda f: lr.predict_proba((f - mu) / sd)[:, 1]
    return lambda f: state["model"].predict_proba(f)[:, 1]


def evaluate11(_args) -> None:
    """GPU: per-draw 824896-form scores, pseudo assignment and pair arrays; CPU: verifier, tuning, sealed."""
    import pickle

    from sklearn.metrics import average_precision_score, roc_auc_score

    from scripts.identity.pid2_cheap import buckets, official, paired, verdict
    from scripts.identity.pid2_qmerge import separation

    (QM2_OUT / "draws").mkdir(parents=True, exist_ok=True)
    ctx = _llr_setup()
    report, draw, llr, device = ctx["report"], ctx["draw"], ctx["llr"], ctx["device"]
    codes, components = ctx["codes"], ctx["components"]
    PHI, B, SLOT, positions_of = ctx["PHI"], ctx["B"], ctx["SLOT"], ctx["positions_of"]
    mask, prior, P, W8 = ctx["mask"], ctx["prior"], ctx["P"], ctx["W8"]

    def comp(gallery, query, cat_prior=None):
        parts = components(gallery, query, cat_prior)
        le = llr({"gallery": gallery, "query": query}, 1.0, mask, prior, P)[0]
        return np.stack([parts[c] for c in PARTS[:-1]] + [le], axis=2)

    path = lambda pool, d: QM2_OUT / "draws" / f"pool{pool}_draw{d}.npz"
    first = True
    for pool, d in R11_FIT + R11_TUNE + R11_SEAL0 + R11_SEAL1:
        if path(pool, d).exists():
            continue
        x = draw(pool, d, with_parts=True)
        le = llr(x, 1.0, mask, prior, P)[0]
        X0 = np.stack([x["parts"][c] for c in PARTS[:-1]] + [le], axis=2)
        _, a, S = pseudo_rounds(comp, x["gallery"], x["query"], X0, x["ks"], W8, codes)
        pp = components(x["query"], x["query"])
        thq = question_thetas(PHI, B, SLOT, positions_of, x["query"], prior, mask, P)
        lpp, nm = pair_llr(PHI, B, SLOT, positions_of, x["query"], thq, prior, mask)
        if first:
            first = False
            diag_rank = float(np.mean([(lpp[i] > lpp[i, i]).sum() for i in range(len(lpp))]))
            report["pair_llr_diag_rank_mean"] = diag_rank
            print("pair LLR: mean number of q' scoring above q itself:", diag_rank, flush=True)
        np.savez(path(pool, d), S=S, assign=a, owner=x["owner"], k=x["ks"], n_cand=len(x["candidates"]),
                 LLRp=lpp.astype(np.float32), n_moves=nm, **{n: pp[n].astype(np.float32) for n in QM2_PAIR_PARTS})
        print(f"pool {pool} draw {d} done", flush=True)
    (QM2_OUT / "report_gpu.json").write_text(json.dumps(report, indent=1, default=float))

    # ---- verifier
    load = lambda pool, d: dict(np.load(path(pool, d)))
    fns, states = qm2_fit(R11_FIT)
    tune = {x: load(*x) for x in R11_TUNE}
    ap = {}
    for name, v in fns.items():
        ys, ps = [], []
        for dd in tune.values():
            iu = np.triu_indices(len(dd["k"]), 1)
            ys.append((dd["owner"][:, None] == dd["owner"][None, :])[iu])
            ps.append(qm2_pair_m(v, dd)[iu])
        y, p = np.concatenate(ys), np.concatenate(ps)
        ap[name] = {"ap": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p))}
    chosen_v = max(ap, key=lambda n: ap[n]["ap"])
    verifier = fns[chosen_v]
    (QM2_OUT / "verifier.pkl").write_bytes(pickle.dumps({"name": chosen_v, **states[chosen_v]}))
    out = {"verifier": ap, "chosen_verifier": chosen_v}
    print("verifier on tuning draws:", ap, "chosen", chosen_v, flush=True)

    target_of = lambda S: posterior_assign(S, 0.5)[1] < 0.5

    # ---- tuning
    rows = []
    for (pool, dr), dd in tune.items():
        m = qm2_pair_m(verifier, dd)
        tg = target_of(dd["S"])
        rows.append({"draw": dr, "arm": "base", **buckets(official(dd["S"], dd["owner"]), dd["k"])})
        for tau, theta in itertools.product(QM2_TAUS, QM2_THETAS):
            from scripts.identity.pid2_qmerge import merge_term
            t = merge_term(dd["S"], m, tau, theta)
            for beta in QM2_BETAS:
                S2 = dd["S"].copy()
                S2[tg] += beta * t[tg]
                rows.append({"draw": dr, "arm": f"t{tau:g}_b{beta:g}_th{theta:g}", **buckets(official(S2, dd["owner"]), dd["k"])})
                if beta == QM2_BETAS[0] and theta == QM2_THETAS[0] and tau == QM2_TAUS[0]:
                    pass
        print(f"tune draw {dr} done", flush=True)
    tframe = pd.DataFrame(rows)
    tframe.to_csv(QM2_OUT / "tune.csv", index=False)
    ttab = paired(tframe, "base").sort_values("d_all", ascending=False)
    best = ttab[ttab["arm"] != "base"].iloc[0]["arm"]
    tau, beta, theta = (float(s[1:].lstrip("bh")) for s in best.split("_"))
    edge = {"tau": tau in (QM2_TAUS[0], QM2_TAUS[-1]), "beta": beta in (QM2_BETAS[0], QM2_BETAS[-1]),
            "theta": theta == QM2_THETAS[-1]}
    out.update({"chosen_arm": best, "params": {"tau": tau, "beta": beta, "theta": theta}, "grid_edge": edge,
                "tune_top": ttab.head(10).to_dict("records")})
    print(ttab.head(10)[["arm", "all", "d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(index=False),
          f"\nchosen {best} (grid edge: {edge})", flush=True)
    rows = []                                # information only: the same parameters on every question
    for (pool, dr), dd in tune.items():
        m = qm2_pair_m(verifier, dd)
        rows.append({"draw": dr, "arm": "base", **buckets(official(dd["S"], dd["owner"]), dd["k"])})
        rows.append({"draw": dr, "arm": "all_targets", **buckets(official(
            qm2_apply(dd["S"], m, np.ones(len(dd["k"]), bool), tau, beta, theta), dd["owner"]), dd["k"])})
    out["tune_all_targets"] = paired(pd.DataFrame(rows), "base").to_dict("records")
    (QM2_OUT / "report.json").write_text(json.dumps(out, indent=1, default=float))

    # ---- sealed
    out["sealed"], out["separation"], out["targets"] = {}, {}, {}
    for label, draws in (("pool0", R11_SEAL0), ("pool1", R11_SEAL1)):
        rows, sep, ys, ps, nt = [], [], [], [], []
        for pool, dr in draws:
            dd = load(pool, dr)
            m = qm2_pair_m(verifier, dd)
            tg = target_of(dd["S"])
            nt.append(float(tg.mean()))
            truth = (dd["owner"][:, None] == dd["owner"][None, :]).astype(np.float64)
            np.fill_diagonal(truth, 0.0)
            rows.append({"draw": dr, "arm": "base", **buckets(official(dd["S"], dd["owner"]), dd["k"])})
            rows.append({"draw": dr, "arm": "merge", **buckets(official(qm2_apply(dd["S"], m, tg, tau, beta, theta), dd["owner"]), dd["k"])})
            for ob in (0.5, 1.0):
                rows.append({"draw": dr, "arm": f"oracle_b{ob:g}", **buckets(official(
                    qm2_apply(dd["S"], truth, tg, tau, ob, 0.5), dd["owner"]), dd["k"])})
            sep.append(separation(m, dd))
            iu = np.triu_indices(len(m), 1)
            ys.append(truth[iu] > 0)
            ps.append(m[iu])
        frame = pd.DataFrame(rows)
        frame["pool"] = int(label[-1])
        frame.to_csv(QM2_OUT / f"sealed_{label}.csv", index=False)
        out["sealed"][label] = paired(frame, "base").to_dict("records")
        y, p = np.concatenate(ys), np.concatenate(ps)
        out["separation"][label] = {**pd.DataFrame(sep).mean().to_dict(), "ap": float(average_precision_score(y, p)),
                                    "auc": float(roc_auc_score(y, p))}
        out["targets"][label] = float(np.mean(nt))
        print(f"sealed {label}:\n", paired(frame, "base").round(4).to_string(index=False),
              f"\nseparation {label}: {out['separation'][label]}, target share {out['targets'][label]:.3f}", flush=True)
    t1, t0 = pd.DataFrame(out["sealed"]["pool1"]), pd.DataFrame(out["sealed"]["pool0"])
    v = verdict(t1, t0, "merge")
    v["probe"] = bool(not v["pass"] and v["pool1_delta"] > 0 and v["pool0_delta"] > 0)
    out["verdict"] = v
    print("verdict:", json.dumps(v), flush=True)
    (QM2_OUT / "report.json").write_text(json.dumps(out, indent=1, default=float))


def items12_selfcheck() -> None:
    """synthetic: fit_free keeps the fixed weights and recovers a planted free one; reverse_llr /
    pair_llr are 0 at theta_pop and match a direct computation; qm2_apply leaves non-targets alone."""
    import torch
    rng = np.random.default_rng(5)
    true = np.array([1.0, 0.5, 2.0, 1.0, 0.8, 0.3, 1.2, 3.0, 0.7])
    sets = []
    for _ in range(3):
        X = rng.normal(size=(1500, 60, 9)) * 0.5
        s = X @ true
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        sets.append((X, np.array([rng.choice(60, p=r) for r in p])))
    w0 = np.r_[true[:8], 0.0]
    w = fit_free(sets, w0, [8], torch.device("cpu"))
    assert np.array_equal(w[:8], true[:8]) and abs(w[8] - 0.7) < 0.15, w
    n, d = 600, 5
    phi = torch.from_numpy(rng.normal(size=(n, OPTIONS, d)).astype(np.float16))
    b = torch.log_softmax(torch.from_numpy(rng.normal(size=(n, OPTIONS)).astype(np.float32)), dim=1)
    slot = torch.from_numpy(rng.integers(0, OPTIONS, n))
    games = [np.arange(i * 50, i * 50 + 50) for i in range(12)]

    def positions_of(gl):
        pos = np.concatenate([np.concatenate([games[g] for g in grp]) for grp in gl])
        own = np.concatenate([np.full(sum(len(games[g]) for g in grp), i) for i, grp in enumerate(gl)])
        return torch.from_numpy(pos), torch.from_numpy(own)

    prior = torch.zeros(d, dtype=torch.float64)
    th = torch.from_numpy(rng.normal(size=(3, d)))
    gallery, query = [[0, 1], [2, 3, 4], [5]], [[6], [7, 8], [9, 10, 11]]
    r = reverse_llr(phi, b, slot, positions_of, gallery, th, prior, None)
    assert np.abs(reverse_llr(phi, b, slot, positions_of, gallery, prior[None].repeat(3, 1), prior, None)).max() < 1e-9
    direct = np.zeros((3, 3))
    for qi in range(3):
        for ci, grp in enumerate(gallery):
            pos = torch.from_numpy(np.concatenate([games[g] for g in grp]))
            lp = played_logp(phi, b, slot, pos, torch.stack([th[qi], prior]))
            direct[qi, ci] = float((lp[:, 0] - lp[:, 1]).sum())
    assert np.abs(r - direct).max() < 1e-9
    pl, nm = pair_llr(phi, b, slot, positions_of, query, th, prior, None)
    assert pl.shape == (3, 3) and nm.tolist() == [50, 100, 150]
    S = rng.normal(size=(6, 4))
    m = np.ones((6, 6)) - np.eye(6)
    tg = np.array([True, False, True, False, False, False])
    S2 = qm2_apply(S, m, tg, 1.0, 0.5, 0.0)
    assert np.array_equal(S2[~tg], S[~tg]) and not np.allclose(S2[tg], S[tg])
    dd = {"S": S, "k": np.array([1, 2, 3, 5, 1, 2]), "assign": np.array([0, 0, -1, 1, -1, 1]),
          "LLRp": rng.normal(size=(6, 6)), "n_moves": np.full(6, 100.0),
          **{n_: rng.normal(size=(6, 6)) for n_ in QM2_PAIR_PARTS}}
    f = qm2_features(dd)
    assert f.shape[:2] == (6, 6) and np.allclose(f, f.transpose(1, 0, 2))
    assert f[0, 1, -3] == 1 and f[2, 4, -3] == 0
    print("items 1/2 selfcheck ok", {"add1_weight": round(float(w[8]), 3), "features": int(f.shape[2])}, flush=True)


def official_builder(part: str, choice: dict, cache_dir: Path):
    """(comp7, local category codes) for one official added-set part: the 7 non-LLR fusion parts
    for arbitrary gallery / query lists in local game rows (pool rows, then question rows), rebuilt
    from the stores exactly as predict2 / pid2_style_export / pid2_multilevel_export built them
    (PID_PSEUDO_COLOUR_PLAN.md plan 1)."""
    from multiprocessing import Pool

    from go_aicup.data.preprocess import load_store
    from go_aicup.tasks.identity import data as idata, opening as op, twocolor
    from go_aicup.tasks.identity.predict import PLIES, store_prefix, train_prefix
    from scripts.identity.pid2_multilevel_export import _init as ml_init, _keys_of as ml_keys, game_list
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, PopTable, level_scores
    from scripts.identity.pid2_qmerge_export import DAY0, ENC
    from scripts.identity.pid2_style_export import REF_HEAD, REF_RUN, STORE, head_vectors

    s = json.loads((DAY0 / f"{part}2_d4_9600_seqB" / "info.json").read_text())["summary"]
    params = s["opening"]
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    train_meta, train_pre = train_prefix("d4")
    pool_m, pool_c, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
    q_m, q_c, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
    pool_pre, ok_p = store_prefix(pool_m, pool_c, pool_meta, "d4", min_plies)
    q_pre, ok_q = store_prefix(q_m, q_c, q_meta, "d4", min_plies)
    assert ok_p.all() and ok_q.all()
    n_train = len(train_meta)
    prefix = np.concatenate([train_pre[:, :PLIES], pool_pre, q_pre])
    hashes = op.prefix_hashes(prefix)
    train_color = train_meta["player_color"].to_numpy().astype(np.int64)
    populations = (np.flatnonzero(train_color == 0), np.flatnonzero(train_color == 1))
    is_white = lambda meta: (meta["color"].astype(str).to_numpy() == "W").astype(np.int64)
    color_full = np.concatenate([train_color, is_white(pool_meta), is_white(q_meta)])
    codes_full = np.concatenate([np.zeros(n_train, np.int64),
                                 np.minimum(pool_meta["hd"].to_numpy(), 1) + 2 * is_white(pool_meta),
                                 np.minimum(q_meta["hd"].to_numpy(), 1) + 2 * is_white(q_meta)])
    games = game_list(pool_m, pool_c, pool_meta) + game_list(q_m, q_c, q_meta)
    with Pool(12, initializer=ml_init, initargs=(games,)) as pool:
        keys = np.stack(pool.map(ml_keys, range(len(games)), chunksize=64))
    color_ml = np.array([g[1] for g in games], np.int64)
    meta_train = idata.load_meta()
    popfile = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popfile["keys"], meta_train["player_color"].to_numpy().astype(np.int64)[popfile["game_index"]])
    runs = {"B": (REF_RUN, REF_HEAD)}
    for name, (suffix, _) in ENC.items():
        if name != "B":
            info = json.loads((DAY0 / f"{part}2_d4_9600{suffix}" / "info.json").read_text())
            runs[name] = (info["run"], ROOT / info["head"])
    vecs = {}
    for name, (run, head) in runs.items():
        path = cache_dir / f"vectors_{name}_{part}.npy"
        if not path.exists():
            np.save(path, head_vectors(run, Path(head), part))
        vecs[name] = np.load(path)
        assert len(vecs[name]) == len(games), name

    def comp7(gal, qry, cat_prior=None):
        G, Q = [np.asarray(g) + n_train for g in gal], [np.asarray(q) + n_train for q in qry]
        out = {"opening": twocolor.opening_scores(prefix, hashes, G, Q, color_full, populations, params),
               "category": category_scores_fixed(codes_full, G, Q, 4, cat_prior)}
        for lv, lam in (("L1", choice["lambda1"]), ("L2", choice["lambda2"])):
            out[lv] = lam * level_scores(lambda g: keys[g], color_ml, pop, gal, qry, choice["n"], lv)[choice["beta"]]
        for name, (_, w) in ENC.items():
            out[name] = 16 * w * s["w"] * twocolor.style_scores(vecs[name], gal, qry)
        return out

    return comp7, codes_full[n_train:]


def export(args) -> None:
    """official added set: S(ogsa12w) + lambda * LLR with the sealed-chosen sigma / lambda."""
    import hashlib
    import os

    import torch

    from go_aicup.data.preprocess import load_store
    from go_aicup.tasks.identity.assign import top5
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1
    from scripts.identity.pid2_qmerge_export import BASE, COLS, DAY0, ENC, SUB
    from scripts.identity.pid2_style_export import STORE

    job = os.environ.get("SLURM_JOB_ID", "local")
    rep = json.loads((OUT / "report.json").read_text())
    assert rep["sealed"]["verdict"]["pass"], "stage 0 did not pass the switch rule"
    sigma, lam = rep["dev"]["chosen"]["sigma"], rep["dev"]["chosen"]["lambda"]
    mk = {k: BASE_M for k in (1, 2, 3, 5)}
    tag = "lik"
    if args.round2:                        # round-2 choice (wider grid / per-k), only if it replaced B0
        r2 = json.loads((R2 / "report.json").read_text())
        assert r2["final"] != "B0", "round 2 kept B0; the stage-0 export stands"
        fp = r2["final_params"]
        sigma, lam, mk = fp["sigma"], fp["lambda"], {int(k): float(v) for k, v in fp["m_k"].items()}
        tag = f"lik2{r2['final'].lower()}"
    rich_mask = None
    if args.rich:                          # rich-feature choice (PID_LIKELIHOOD_RICH_PLAN.md), only if it passed
        r3 = json.loads((RICH_OUT / "report.json").read_text())
        assert r3["sealed"]["verdict"]["pass"], "rich features did not pass the switch rule"
        fp = r3["final_params"]
        sigma, lam, mk = fp["sigma"], fp["lambda"], {int(k): float(v) for k, v in fp["m_k"].items()}
        tag = "likrich"
    precision = None
    if args.eb:                            # empirical-Bayes prior (PID_EB_COMBINER_PLAN.md), only if it passed
        r4 = json.loads((EB_OUT / "report.json").read_text())
        assert r4["sealed"]["verdict"]["pass"], "the empirical-Bayes prior did not pass the switch rule"
        fp = r4["final_params"]
        sigma, lam, mk = 1.0, fp["lambda"], {int(k): float(v) for k, v in fp["m_k"].items()}
        tag = "likeb"
        if args.eb_lambda:                 # widened lambda grid (evaluate6), only if it passed
            r6 = json.loads((EB2_OUT / "report.json").read_text())
            assert r6["sealed"]["verdict"]["pass"], "the widened lambda did not pass the switch rule"
            lam = float(r6["chosen"]["lambda"])
            tag = f"likeb-l{lam:g}"
    comb = None
    if args.combine:                       # per-k learned fusion on top of EB (plan 2), only if it passed
        assert args.eb, "--combine stacks on --eb"
        r5 = json.loads((COMB_OUT / "report.json").read_text())
        assert args.probe or args.colour or args.pseudo or r5["sealed"]["verdict"]["pass"], "the learned fusion did not pass the switch rule"  # colour / pseudo stack on the adopted 821010 base
        comb = {int(k): np.array(v, np.float64) for k, v in r5["chosen"]["weights"].items()}
        tag = "comb-probe" if args.probe else "comb"
    colw = None
    if args.colour:                        # colour-dependent weights on top of the probe fusion (821010 base)
        assert args.combine and not args.probe, "--colour stacks on --combine (the 821010-comb-probe weights)"
        r7 = json.loads((COL_OUT / "report.json").read_text())
        v7 = r7["sealed"]["verdict"]
        assert v7["pass"] or v7["probe"], "colour weights neither passed nor qualify as a probe"
        colw = {int(k): np.array(v, np.float64) for k, v in r7["chosen"]["weights"].items()}
        tag = "combcol" if v7["pass"] else "combcol-probe"
    pseudo = None
    if args.pseudo:                        # pseudo-label expansion (plan 1) on top of the base fusion it was tested on
        assert args.combine and not args.probe, "--pseudo stacks on --combine [--colour]"
        r8 = json.loads((PS_OUT / "report.json").read_text())
        v8 = r8["sealed"]["verdict"]
        assert v8["pass"] or v8["probe"], "the expansion neither passed nor qualifies as a probe"
        assert (r8["base"] == "colour") == bool(args.colour), "export base differs from the tested base"
        pseudo = {"tau": float(r8["chosen"]["tau"]), "rounds": int(r8["chosen"]["rounds"])}
        tag = tag + ("-pseudo" if v8["pass"] else "-pseudo-probe")
        if args.pseudo2:                   # retuned tau / rounds and refitted weights (evaluate9)
            r9 = json.loads((PS2_OUT / "report.json").read_text())
            v9 = r9["sealed"]["verdict"]
            assert v9["pass"] or v9["probe"], "the retune neither passed nor qualifies as a probe"
            pseudo["second"] = {"tau": float(r9["chosen_config"]["tau"]), "rounds": int(r9["chosen_config"]["rounds"]),
                                "W": {int(k): np.array(w_, np.float64) for k, w_ in r9["refit"]["weights"].items()}}
            tag = tag.replace("-pseudo-probe", "-pseudo").replace("-pseudo", "-pseudo2" if v9["pass"] else "-pseudo2-probe")
    target = SUB / f"{job}-{tag}.csv"
    assert not target.exists(), target
    device = torch.device("cuda")
    theta_pop = torch.tensor(rep["theta_pop"], device=device, dtype=torch.float64)
    if args.rich or args.eb:
        theta_pop = torch.tensor(fp["theta_pop"], device=device, dtype=torch.float64)
        rich_mask = torch.tensor(fp["mask"], device=device, dtype=torch.float32)
    if args.eb:
        precision = torch.from_numpy(np.load(EB_OUT / "precision.npy")).to(device)
    official = OUT / "official"
    official.mkdir(parents=True, exist_ok=True)

    # 1. positions of both parts (ids: part * 1e6 + row; pool rows first, then question rows)
    stores, groups = {}, {0: ([], []), 1: ([], [])}
    for p_i, part in enumerate(("public", "private")):
        pool_m, pool_c, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
        q_m, q_c, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
        games = {}
        for base_row, (m, c, meta) in ((0, (pool_m, pool_c, pool_meta)), (len(pool_meta), (q_m, q_c, q_meta))):
            for r, (off, length, col, hd) in enumerate(zip(meta["offset"], meta["length"],
                                                          meta["color"].astype(str), meta["hd"])):
                gid = p_i * 1_000_000 + base_row + r
                moves = np.asarray(m[off:off + length], np.int64)
                colors = np.asarray(c[off:off + length], np.int64)
                assert int(hd) <= 1, "official added set has HD 0/1 only"
                games[gid] = moves
                groups[int(hd)][0].append((moves, colors, colors == int(col == "W")))
                groups[int(hd)][1].append(gid)
        stores[part] = (pool_meta, q_meta, games)
    cache = official / "positions.npz"
    if cache.exists():
        arr = dict(np.load(cache))
    else:
        arr, meta = extract_positions([(hd, groups[hd][0], np.array(groups[hd][1])) for hd in (0, 1)],
                                      args.workers)
        np.savez(cache, **arr)
        (official / "policy_meta.json").write_text(json.dumps(meta, indent=1))
    all_games = {g: m for _, _, gm in stores.values() for g, m in gm.items()}
    B, PHI, SLOT, span = position_tensors(arr, all_games.__getitem__, device,
                                          dict(np.load(RICH / "official.npz")) if args.rich or args.eb else None)
    print(f"official positions {len(SLOT)}, outside top-32 {float((SLOT == TOP).float().mean()):.4f}", flush=True)

    def positions_of(game_lists):
        pos, own = [], []
        for i, gl in enumerate(game_lists):
            for g in gl:
                s_, e_ = span[int(g)]
                pos.append(np.arange(s_, e_))
                own.append(np.full(e_ - s_, i))
        return (torch.from_numpy(np.concatenate(pos)).to(device), torch.from_numpy(np.concatenate(own)).to(device))

    # 2. per part: base S (= ogsa12w, verified), candidate thetas, LLR, fused top-5
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")
    out = base.copy()
    report = {"sigma": sigma, "lambda": lam, "m_k": mk, "eb": bool(args.eb), "base": str(BASE.relative_to(ROOT)), "parts": {}}
    for p_i, (part, prefix) in enumerate((("public", "pub2"), ("private", "priv2"))):
        pool_meta, q_meta, _ = stores[part]
        parts = {}
        for name, (suffix, _) in ENC.items():
            d = DAY0 / f"{part}2_d4_9600{suffix}"
            parts[name] = dict(np.load(d / "parts.npz"))
            if name == "B":
                cand = np.load(d / "candidates.npy", allow_pickle=True).astype(str)
                qids = pd.read_parquet(d / "predictions.parquet").question_id.astype(str).tolist()
        ml = dict(np.load(DAY0 / f"{part}2_d4_9600_seqB_ml" / "parts.npz"))
        rest = (parts["B"]["opening"] + parts["B"]["category"]
                + choice["lambda1"] * ml["ml_L1"] + choice["lambda2"] * ml["ml_L2"])
        style_term = 16 * sum(w * parts[n]["style"] for n, (_, w) in ENC.items())     # multiplier 1.5 inside
        s_base = rest + style_term
        np.testing.assert_array_equal(cand[top5(s_base)], base.loc[qids, COLS].to_numpy())
        players = pool_meta["player_id"].astype(str).to_numpy()
        assert np.array_equal(np.array(sorted(set(players))), cand)
        gallery = [p_i * 1_000_000 + np.flatnonzero(players == c) for c in cand]
        rows = q_meta.groupby("question_id", sort=False).indices
        query = [p_i * 1_000_000 + len(pool_meta) + np.sort(rows[q]) for q in qids]
        ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[qids].to_numpy().astype(int)
        pos_g, own_g = positions_of(gallery)
        th = fit(PHI, B, SLOT, pos_g, own_g, len(gallery), theta_pop, sigma, mask=rich_mask, precision=precision)
        pos_q, own_q = positions_of(query)
        lp = played_logp(PHI, B, SLOT, pos_q, torch.cat([th, theta_pop[None]]), mask=rich_mask)
        llr = torch.zeros(len(query), len(gallery), device=device, dtype=torch.float64)
        llr.index_add_(0, own_q, lp[:, :-1] - lp[:, -1:])
        llr = llr.cpu().numpy()
        np.savez(official / f"{prefix}.npz", llr=llr, theta=th.cpu().numpy(), base=s_base, k=ks,
                 candidates=cand, qids=np.array(qids))
        factor = np.array([mk[int(k)] / BASE_M for k in ks])[:, None]
        ranked = cand[top5(rest + factor * style_term + lam * llr)]
        if comb is not None:               # components in base units; current weights must give the EB file
            comps = {"opening": parts["B"]["opening"], "category": parts["B"]["category"],
                     "L1": choice["lambda1"] * ml["ml_L1"], "L2": choice["lambda2"] * ml["ml_L2"],
                     **{n: 16 * w * parts[n]["style"] for n, (_, w) in ENC.items()}, "LLR": llr}
            X = np.stack([comps[c] for c in PARTS], axis=2)
            eb_file = pd.read_csv(SUB / "820880-likeb.csv", dtype=str).set_index("question_id")
            np.testing.assert_array_equal(cand[top5(apply_weights(X, ks, current_weights(mk, lam)))],
                                          eb_file.loc[qids, COLS].to_numpy())
            np.testing.assert_array_equal(ranked, eb_file.loc[qids, COLS].to_numpy())
            ranked = cand[top5(apply_weights(X, ks, comb))]
            if colw is not None:           # the probe weights must give the current base file first
                base_file = pd.read_csv(SUB / "821010-comb-probe.csv", dtype=str).set_index("question_id")
                np.testing.assert_array_equal(ranked, base_file.loc[qids, COLS].to_numpy())
                qc = q_meta["color"].astype(str).to_numpy()
                z = np.array([2.0 * (qc[rows[q]] == "B").mean() - 1.0 for q in qids])
                report.setdefault("colour_groups", {})[prefix] = {
                    "black": int((z == 1).sum()), "white": int((z == -1).sum()), "mixed": int((np.abs(z) < 1).sum())}
                ranked = cand[top5(apply_colour(X, ks, z, colw))]
            else:
                probe_file = pd.read_csv(SUB / "821010-comb-probe.csv", dtype=str).set_index("question_id")
                if not args.probe:
                    np.testing.assert_array_equal(ranked, probe_file.loc[qids, COLS].to_numpy())
                qc = q_meta["color"].astype(str).to_numpy()
                z = np.array([2.0 * (qc[rows[q]] == "B").mean() - 1.0 for q in qids])
            if pseudo is not None:
                base_f = {"kind": "colour", "W": colw} if colw is not None else {"kind": "comb", "W": comb}
                comp7, codes_local = official_builder(part, choice, official)
                n_pool = len(pool_meta)
                gal_l = [g - p_i * 1_000_000 for g in gallery]
                qry_l = [q - p_i * 1_000_000 for q in query]

                def comp(gal, qry, cat_prior=None):
                    parts7 = comp7(gal, qry, cat_prior)
                    pg, og = positions_of([np.asarray(g) + p_i * 1_000_000 for g in gal])
                    th_ = fit(PHI, B, SLOT, pg, og, len(gal), theta_pop, sigma, mask=rich_mask, precision=precision)
                    pq, oq = positions_of([np.asarray(q) + p_i * 1_000_000 for q in qry])
                    lp_ = played_logp(PHI, B, SLOT, pq, torch.cat([th_, theta_pop[None]]), mask=rich_mask)
                    le = torch.zeros(len(qry), len(gal), device=device, dtype=torch.float64)
                    le.index_add_(0, oq, lp_[:, :-1] - lp_[:, -1:])
                    return np.stack([parts7[c] for c in PARTS[:-1]] + [le.cpu().numpy()], axis=2)

                Xr = comp(gal_l, qry_l)
                rdiff = {c: float(np.abs(Xr[:, :, j] - X[:, :, j]).max()) for j, c in enumerate(PARTS)}
                print(prefix, "official comp() vs saved parts, max abs diff:", rdiff, flush=True)
                assert max(rdiff[c] for c in ("opening", "category", "L1", "L2", "LLR")) <= 1e-9, rdiff
                assert max(rdiff[c] for c in ("B", "A1", "A2")) <= 2e-3, rdiff
                S0 = fuse(base_f, X, ks, z)
                assert np.array_equal(cand[top5(S0)], ranked)
                S, a = S0, None
                for _ in range(pseudo["rounds"]):
                    a, _p = posterior_assign(S, pseudo["tau"])
                    Xp, _ = expand(comp, gal_l, qry_l, a, X, codes_local)
                    S = fuse(base_f, Xp, ks, z)
                ranked = cand[top5(S)]
                if "second" in pseudo:     # the first pass must give the current base file
                    cur_file = pd.read_csv(SUB / "824896-comb-pseudo.csv", dtype=str).set_index("question_id")
                    np.testing.assert_array_equal(ranked, cur_file.loc[qids, COLS].to_numpy())
                    sec = pseudo["second"]
                    S = S0
                    for _ in range(sec["rounds"]):
                        a, _p = posterior_assign(S, sec["tau"])
                        Xp, _ = expand(comp, gal_l, qry_l, a, X, codes_local)
                        S = fuse(base_f, Xp, ks, z)
                    ranked = cand[top5(apply_weights(Xp, ks, sec["W"]))]
                report.setdefault("pseudo", {})[prefix] = {
                    "comp_diff": rdiff, "picked": int((a >= 0).sum()),
                    "picked_by_k": {int(k): int((a[ks == k] >= 0).sum()) for k in (1, 2, 3, 5)},
                    "expanded_candidates": int(len(np.unique(a[a >= 0])))}
        out.loc[qids, COLS] = ranked
        ch = ranked[:, 0] != base.loc[qids, "top_1"].to_numpy()
        alone_top1 = cand[llr.argmax(1)] == base.loc[qids, "top_1"].to_numpy()
        report["parts"][prefix] = {"top1_changed": int(ch.sum()),
                                   "by_k": {int(k): int(ch[ks == k].sum()) for k in (1, 2, 3, 5)},
                                   "llr_alone_agrees_with_base_top1": float(alone_top1.mean()),
                                   "theta_mean_norm": float((th - theta_pop).norm(dim=1).mean())}
        print(prefix, json.dumps(report["parts"][prefix]), flush=True)
    orig = base.index[base.index.str.startswith(("pub_", "priv_"))]
    pd.testing.assert_frame_equal(out.loc[orig], base.loc[orig])
    assert len(out) == 1600
    for q, row in out.iterrows():
        assert len(set(row)) == 5 and all(v.split("_")[0] == q.split("_")[0] for v in row)
    out.to_csv(target)
    pd.testing.assert_frame_equal(pd.read_csv(target, dtype=str).set_index("question_id"), out)
    report["path"] = str(target.relative_to(ROOT))
    report["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (official / f"report_{job}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


# ---------------------------------------------------------------- CPU self-checks
def selfcheck(_args) -> None:
    import torch
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    n = 12000
    top = np.stack([rng.permutation(361)[:TOP] for _ in range(n)])
    raw = rng.normal(size=(n, 4, 362))
    lp_all = raw - np.logaddexp.reduce(raw, axis=2, keepdims=True)
    lp_top = np.take_along_axis(lp_all, top[:, None, :].repeat(4, 1), axis=2)
    other = np.log1p(-np.exp(np.logaddexp.reduce(lp_top, axis=2)))
    lp = np.concatenate([lp_top, other[..., None]], axis=2)
    ply = rng.integers(0, 250, size=n)
    last = rng.integers(-1, 361, size=n)
    b, phi = features(top, lp, ply, last, np.roll(last, 1))
    assert phi.shape == (n, OPTIONS, DIM)
    assert np.allclose(np.logaddexp.reduce(b, axis=1), 0.0, atol=1e-5), "b is a distribution over options"
    assert (phi[:, TOP, 7:19] == 0).all(), "other has no geometry"
    assert np.allclose(phi[:, :, 19:23], phi[:, :, 3:7] * (ply >= LATE_PLY)[:, None, None])
    # MAP recovers a planted theta; LLR is 0 at theta_pop
    device = torch.device("cpu")
    PHI, B = torch.from_numpy(phi), torch.from_numpy(b)
    true = torch.zeros(2, DIM, dtype=torch.float64)
    true[0, :3] = torch.tensor([0.8, -0.5, 0.3])
    true[1, 3:7] = torch.tensor([-0.6, -1.0, -1.5, -2.0])
    owner = torch.from_numpy(np.arange(n) % 2)
    z = B + torch.einsum("pjd,pd->pj", PHI.float(), true[owner].float())
    slot = torch.distributions.Categorical(logits=z).sample()
    pos = torch.arange(n)
    est = fit(PHI, B, slot, pos, owner, 2, torch.zeros(DIM, dtype=torch.float64), 10.0)
    assert (est[0, :3] - true[0, :3]).abs().max() < 0.25, est[0, :3]
    assert (est[1, 3:7] - true[1, 3:7]).abs().max() < 0.35, est[1, 3:7]
    lp2 = played_logp(PHI, B, slot, pos[:50], torch.stack([est[0], est[0]]))
    assert float((lp2[:, 0] - lp2[:, 1]).abs().max()) == 0.0
    # strong prior pins theta to the prior mean
    pinned = fit(PHI, B, slot, pos, owner, 2, torch.ones(DIM, dtype=torch.float64), 1e-4)
    assert (pinned - 1).abs().max() < 1e-3
    # rich (50-dim) layout, mask equivalence and recovery of planted new-dim parameters
    rich = (rng.random((n, TOP, 8)) < 0.15).astype(np.uint8)
    b50, phi50 = features(top, lp, ply, last, np.roll(last, 1), rich)
    assert phi50.shape == (n, OPTIONS, DIM_RICH) and np.array_equal(phi50[:, :, :DIM], phi) and np.allclose(b50, b)
    assert np.allclose(phi50[:, :, 27], b, atol=2e-2) and (phi50[:, TOP, 29:] == 0).all()
    assert np.array_equal(phi50[:, :TOP, 37:42], rich[:, :, :5]) and np.array_equal(phi50[:, :TOP, 47:50], rich[:, :, 5:])
    zone_sum = phi50[:, :TOP, 29:37].astype(np.float32).sum(2)
    assert zone_sum.max() <= 1
    P50 = torch.from_numpy(phi50)
    base = torch.zeros(DIM_RICH)
    base[:DIM] = 1
    t27 = fit(PHI, B, slot, pos, owner, 2, torch.zeros(DIM, dtype=torch.float64), 3.0)
    t50 = fit(P50, B, slot, pos, owner, 2, torch.zeros(DIM_RICH, dtype=torch.float64), 3.0, mask=base)
    assert (t50[:, :DIM] - t27).abs().max() < 1e-4 and t50[:, DIM:].abs().max() == 0
    m = 30000                                   # one owner, enough positions for the early / late pairs
    top_m = np.stack([rng.permutation(361)[:TOP] for _ in range(m)])
    raw_m = rng.normal(size=(m, 4, 362))
    lpa = raw_m - np.logaddexp.reduce(raw_m, axis=2, keepdims=True)
    lpt = np.take_along_axis(lpa, top_m[:, None, :].repeat(4, 1), axis=2)
    lp_m = np.concatenate([lpt, np.log1p(-np.exp(np.logaddexp.reduce(lpt, axis=2)))[..., None]], axis=2)
    ply_m = rng.integers(0, 250, size=m)
    last_m = rng.integers(-1, 361, size=m)
    b_m, phi_m = features(top_m, lp_m, ply_m, last_m, np.roll(last_m, 1), (rng.random((m, TOP, 8)) < 0.15).astype(np.uint8))
    Pm, Bm = torch.from_numpy(phi_m), torch.from_numpy(b_m)
    true50 = torch.zeros(DIM_RICH, dtype=torch.float64)
    true50[37:42] = torch.tensor([1.0, -0.8, 0.6, 0.0, 0.5])
    true50[47:50] = torch.tensor([1.5, -1.0, 0.7])
    true50[27] = 0.3
    z = Bm + torch.einsum("pjd,d->pj", Pm.float(), true50.float())
    slot50 = torch.distributions.Categorical(logits=z).sample()
    own_m = torch.zeros(m, dtype=torch.long)
    est50 = fit(Pm, Bm, slot50, torch.arange(m), own_m, 1, torch.zeros(DIM_RICH, dtype=torch.float64), 10.0)[0]
    early = est50[37:42] - true50[37:42]
    late = est50[37:42] + est50[42:47] - true50[37:42]
    err = max(float(early.abs().max()), float(late.abs().max()), float((est50[47:50] - true50[47:50]).abs().max()))
    assert err < 0.3, (err, est50[27:50])
    # full prior precision = I / sigma^2 is the isotropic fit
    iso = fit(PHI, B, slot, pos, owner, 2, torch.zeros(DIM, dtype=torch.float64), 1.0,
              precision=torch.eye(DIM, dtype=torch.float64) / 9.0)
    assert (iso - t27).abs().max() < 1e-9
    eb_selfcheck()
    combiner_selfcheck()
    colour_selfcheck()
    pseudo_selfcheck()
    print("selfcheck ok", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("games")
    p = sub.add_parser("policy")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=11)
    sub.add_parser("evaluate")
    sub.add_parser("evaluate2")
    sub.add_parser("rich-features")
    sub.add_parser("evaluate3")
    sub.add_parser("evaluate4")
    sub.add_parser("evaluate5")
    sub.add_parser("evaluate6")
    sub.add_parser("evaluate7")
    sub.add_parser("evaluate8")
    sub.add_parser("evaluate9")
    e = sub.add_parser("export")
    e.add_argument("--workers", type=int, default=11)
    e.add_argument("--round2", action="store_true", help="use the round-2 choice (refuses if it kept B0)")
    e.add_argument("--rich", action="store_true", help="use the rich-feature choice (refuses unless it passed)")
    e.add_argument("--eb", action="store_true", help="use the empirical-Bayes prior (refuses unless it passed)")
    e.add_argument("--combine", action="store_true", help="per-k learned fusion on top of --eb (refuses unless it passed)")
    e.add_argument("--probe", action="store_true", help="with --combine: LB probe even though the gate failed")
    e.add_argument("--pseudo2", action="store_true", help="with --pseudo: retuned tau / rounds and refitted weights (evaluate9)")
    e.add_argument("--pseudo", action="store_true", help="with --combine [--colour]: pseudo-label expansion (passed or probe)")
    e.add_argument("--colour", action="store_true", help="with --combine: colour-dependent weights (passed or probe)")
    e.add_argument("--eb-lambda", action="store_true", help="with --eb: the widened-grid lambda (refuses unless it passed)")
    sub.add_parser("selfcheck")
    args = parser.parse_args()
    {"games": games, "policy": policy, "evaluate": evaluate, "evaluate2": evaluate2, "export": export,
     "rich-features": rich_features_cmd, "evaluate3": evaluate3, "evaluate4": evaluate4, "evaluate5": evaluate5, "evaluate6": evaluate6, "evaluate7": evaluate7, "evaluate8": evaluate8, "evaluate9": evaluate9,
     "selfcheck": selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
