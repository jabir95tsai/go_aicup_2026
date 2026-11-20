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


def features(top: np.ndarray, lp: np.ndarray, ply: np.ndarray, last_opp: np.ndarray, last_own: np.ndarray):
    """top int [N, 32], lp float [N, 4, 33] (conditions C, A, 2D, 5D), ply [N], last moves [N]
    (-1 = none) -> (b [N, 33] float32, phi [N, 33, 27] float16)."""
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
    return b.astype(np.float32), phi.astype(np.float16)


# ---------------------------------------------------------------- MAP fit and scores (torch)
def fit(phi, b, slot, pos, owner, n_owner, prior, sigma, iters=25, chunk=40_000, max_step=1.0):
    """batched Newton MAP of theta [n_owner, D] for positions pos (owner[i] = owner of pos[i])."""
    import torch
    device = phi.device
    theta = prior[None].repeat(n_owner, 1).clone()
    eye = torch.eye(DIM, device=device, dtype=torch.float64)
    for _ in range(iters):
        grad = torch.zeros(n_owner, DIM, device=device, dtype=torch.float64)
        hess = torch.zeros(n_owner, DIM, DIM, device=device, dtype=torch.float64)
        for s in range(0, len(pos), chunk):
            p, o = pos[s:s + chunk], owner[s:s + chunk]
            f = phi[p].float()
            z = b[p] + torch.einsum("pjd,pd->pj", f, theta[o].float())
            pi = torch.softmax(z, dim=1)
            mean = torch.einsum("pj,pjd->pd", pi, f)
            g = f[torch.arange(len(p), device=device), slot[p]] - mean
            h = torch.einsum("pj,pjd,pje->pde", pi, f, f) - mean[:, :, None] * mean[:, None, :]
            grad.index_add_(0, o, g.double())
            hess.index_add_(0, o, h.double())
        grad -= (theta - prior[None]) / sigma ** 2
        hess += eye / sigma ** 2
        step = torch.linalg.solve(hess, grad[..., None])[..., 0]
        size = step.abs().max(dim=1, keepdim=True).values
        step = step * torch.clamp(max_step / size.clamp(min=1e-12), max=1.0)   # damped Newton
        theta = theta + step
        if float(step.abs().max()) < 1e-5:
            break
    return theta


def played_logp(phi, b, slot, pos, theta, chunk=2_000):
    """[len(pos), n_theta] log p(played option) under every theta row."""
    import torch
    device = phi.device
    out = torch.empty(len(pos), len(theta), device=device, dtype=torch.float64)
    t = theta.float()
    for s in range(0, len(pos), chunk):
        p = pos[s:s + chunk]
        f = phi[p].float()
        z = b[p][:, :, None] + torch.einsum("pjd,cd->pjc", f, t)              # [p, 33, C]
        lse = torch.logsumexp(z, dim=1)
        pick = z[torch.arange(len(p), device=device), slot[p]]
        out[s:s + chunk] = (pick - lse).double()
    return out


# ---------------------------------------------------------------- evaluation
def _context(theta_fixed=None) -> dict:
    """shared evaluation setup: bench positions and features, theta_pop, base fusion, draw and LLR."""
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
    b_all, phi_all = [], []
    for s in range(0, len(order), 500_000):
        b, phi = features(arr["top"][s:s + 500_000], arr["lp"][s:s + 500_000], arr["ply"][s:s + 500_000],
                          last_opp[s:s + 500_000], last_own[s:s + 500_000])
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
    zero = torch.zeros(DIM, device=device, dtype=torch.float64)
    theta_pop = fit(PHI, B, SLOT, pos_p, own_p, 1, zero, 10.0)[0]
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

    def draw(pool, d):
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
        return dict(candidates=candidates, q=q, owner=owner, ks=q["k"].to_numpy(), gallery=gallery,
                    query=query, rest=rest, style=styl, base=rest + scale * BASE_M * styl)

    x = draw(*CHECK)
    got = bucket_means(points(x["base"], x["candidates"], x["q"]), x["ks"])
    ref = pd.read_csv(A2OUT / "sealed_per_draw.csv")
    ref = ref[(ref["pool"] == CHECK[0]) & (ref["draw"] == CHECK[1]) & (ref["arm"] == "A12w")].iloc[0]
    report["reproduction_max_diff"] = max(abs(got[c] - ref[c]) for c in got)
    print(f"reproduction max diff {report['reproduction_max_diff']:.2e}", flush=True)
    assert report["reproduction_max_diff"] <= 1e-9

    def llr(x, sigma, mask=None):
        """[Q, C] LLR with candidate thetas fitted on the draw's reference games; mask zeroes the
        features outside the kept dims (ablation)."""
        phi = PHI if mask is None else PHI * mask
        pos_g, own_g = positions_of(x["gallery"])
        th = fit(phi, B, SLOT, pos_g, own_g, len(x["gallery"]), theta_pop, sigma)
        pos_q, own_q = positions_of(x["query"])
        lp = played_logp(phi, B, SLOT, pos_q, torch.cat([th, theta_pop[None]]))
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
                env=env, PHI=PHI, B=B, SLOT=SLOT, positions_of=positions_of)


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


# ---------------------------------------------------------------- official export
def position_tensors(arr: dict, moves_of, device):
    """sorted positions -> (B, PHI, SLOT tensors, first / end position per game id)."""
    import torch
    order = np.lexsort((arr["ply"], arr["game"]))
    arr = {k: v[order] for k, v in arr.items()}
    n = len(order)
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
                          last_opp[s:s + 500_000], last_own[s:s + 500_000])
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
    target = SUB / f"{job}-{tag}.csv"
    assert not target.exists(), target
    device = torch.device("cuda")
    theta_pop = torch.tensor(rep["theta_pop"], device=device, dtype=torch.float64)
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
    B, PHI, SLOT, span = position_tensors(arr, all_games.__getitem__, device)
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
    report = {"sigma": sigma, "lambda": lam, "m_k": mk, "base": str(BASE.relative_to(ROOT)), "parts": {}}
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
        th = fit(PHI, B, SLOT, pos_g, own_g, len(gallery), theta_pop, sigma)
        pos_q, own_q = positions_of(query)
        lp = played_logp(PHI, B, SLOT, pos_q, torch.cat([th, theta_pop[None]]))
        llr = torch.zeros(len(query), len(gallery), device=device, dtype=torch.float64)
        llr.index_add_(0, own_q, lp[:, :-1] - lp[:, -1:])
        llr = llr.cpu().numpy()
        np.savez(official / f"{prefix}.npz", llr=llr, theta=th.cpu().numpy(), base=s_base, k=ks,
                 candidates=cand, qids=np.array(qids))
        factor = np.array([mk[int(k)] / BASE_M for k in ks])[:, None]
        ranked = cand[top5(rest + factor * style_term + lam * llr)]
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
    e = sub.add_parser("export")
    e.add_argument("--workers", type=int, default=11)
    e.add_argument("--round2", action="store_true", help="use the round-2 choice (refuses if it kept B0)")
    sub.add_parser("selfcheck")
    args = parser.parse_args()
    {"games": games, "policy": policy, "evaluate": evaluate, "evaluate2": evaluate2, "export": export,
     "selfcheck": selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
