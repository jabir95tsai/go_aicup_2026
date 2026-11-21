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
                base_mask=base_mask)


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
        assert args.probe or r5["sealed"]["verdict"]["pass"], "the learned fusion did not pass the switch rule"
        comb = {int(k): np.array(v, np.float64) for k, v in r5["chosen"]["weights"].items()}
        tag = "comb-probe" if args.probe else "comb"
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
    e = sub.add_parser("export")
    e.add_argument("--workers", type=int, default=11)
    e.add_argument("--round2", action="store_true", help="use the round-2 choice (refuses if it kept B0)")
    e.add_argument("--rich", action="store_true", help="use the rich-feature choice (refuses unless it passed)")
    e.add_argument("--eb", action="store_true", help="use the empirical-Bayes prior (refuses unless it passed)")
    e.add_argument("--combine", action="store_true", help="per-k learned fusion on top of --eb (refuses unless it passed)")
    e.add_argument("--probe", action="store_true", help="with --combine: LB probe even though the gate failed")
    e.add_argument("--eb-lambda", action="store_true", help="with --eb: the widened-grid lambda (refuses unless it passed)")
    sub.add_parser("selfcheck")
    args = parser.parse_args()
    {"games": games, "policy": policy, "evaluate": evaluate, "evaluate2": evaluate2, "export": export,
     "rich-features": rich_features_cmd, "evaluate3": evaluate3, "evaluate4": evaluate4, "evaluate5": evaluate5, "evaluate6": evaluate6,
     "selfcheck": selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
