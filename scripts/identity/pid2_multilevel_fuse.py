"""Multi-level opening back-off on the added competition-1 set, stage 1 (2026-11-17).

Adds count-likelihood terms of the two local states of src/go_aicup/tasks/identity/multilevel.py
(L1 corner, L2 reply) to the current base fusion (hung-seqB: opening + lambda * w * cosine of
B s2026 + mu * category). For every own move among a game's first N moves and level l,
candidate c scores

    log((n_c(outcome) + beta * p_pop) / (n_c(context) + beta)),
    p_pop = (N_pop(outcome) + 1/361) / (N_pop(context) + 1),

n_c counted over c's gallery games of the same colour (first N own moves), N_pop over a fixed
sample of 100,000 population games per colour (training players outside the bench). A
candidate that never met the context scores p_pop, the same for everyone, so the term cancels.

    python -m scripts.identity.pid2_multilevel_fuse popkeys   # CPU: keys of the population sample
    python -m scripts.identity.pid2_multilevel_fuse run       # 1 GPU (B vectors) + CPU

Selection on development draws pool 0 100-119 over N, beta, lambda1, lambda2; the chosen
setting is scored once on the sealed draws 800- (pool 0 800-819, pool 1 800-829). Gate
(registered in RESULTS.md before the run): pool 1 delta >= +0.005, z >= 2.5, every k >= -0.005,
pool 0 delta > 0.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import multilevel, twocolor
from go_aicup.tasks.identity.predict import train_prefix
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_multilevel import N_OWN, _combine, _with_color
from scripts.identity.pid2_paired import points

RUN = "pid_ge2e_v4_d4_hd_9600"
BENCH = ROOT / "outputs" / "pid" / "pid2_bench"
HEAD = ROOT / "outputs" / "pid" / "pid2_seqagg" / "head_B" / "best.pt"
CACHE = Path("/home/jason1122/pid2_seqagg_cache/bench")
STAGE0 = ROOT / "outputs" / "pid" / "pid2_multilevel"
OUT = STAGE0 / "stage1"
POP_PER_COLOR = 100_000
POP_SEED = 2026
NS = (8, 15)
BETAS = (0.1, 1.0, 4.0)
LAMS = (0.0, 0.25, 0.5, 1.0, 2.0)
DEV = (0, range(100, 120))
SEALED = ((0, range(800, 820)), (1, range(800, 830)))
GATE = {"pool1": 0.005, "z": 2.5, "bucket": -0.005, "pool0": 0.0}
LEVELS = {"L1": (0, 1), "L2": (2, 3)}          # (context column, outcome column) in game_keys

_PREFIX = None
_COLOR = None


def _init(prefix, color):
    global _PREFIX, _COLOR
    _PREFIX, _COLOR = prefix, color


def _keys_of(game: int) -> np.ndarray:
    return multilevel.game_keys(_PREFIX[game].astype(np.int64), int(_COLOR[game]), N_OWN)


def popkeys(args) -> None:
    from scripts.identity.pid2_transductive import load
    env = load(RUN)
    rng = np.random.default_rng(POP_SEED)
    games = np.sort(np.concatenate([rng.choice(p, size=min(POP_PER_COLOR, len(p)), replace=False)
                                    for p in env["populations"]]))
    started = time.time()
    with Pool(args.workers, initializer=_init, initargs=(env["prefix"], env["color"])) as pool:
        keys = np.stack(pool.map(_keys_of, games.tolist(), chunksize=512))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pop_keys.npz", game_index=games, keys=keys)
    print(f"population keys {keys.shape} in {time.time() - started:.0f}s", flush=True)


class PopTable:
    """N_pop(context) and N_pop(outcome) per (N, level), keyed with the colour."""

    def __init__(self, keys: np.ndarray, color: np.ndarray, ns=NS, levels=None):
        self.tables = {}
        for n in ns:
            col = np.repeat(color, n)
            for level, cols in (levels or LEVELS).items():
                for j in cols:
                    k = keys[:, :n, j].ravel()
                    ok = k != 0
                    u, c = np.unique(_with_color(k[ok], col[ok]), return_counts=True)
                    self.tables[n, j] = (u, c)

    def count(self, n, j, key, col):
        u, c = self.tables[n, j]
        x = _with_color(key, col)
        pos = np.minimum(np.searchsorted(u, x), len(u) - 1)
        return np.where(u[pos] == x, c[pos], 0)


def _lookup(sorted_keys, counts, x):
    pos = np.minimum(np.searchsorted(sorted_keys, x), len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == x, counts[pos], 0)


def level_scores(keys_of, color, pop: PopTable, gallery, query, n: int, level: str, levels=None):
    """{beta: [questions, candidates]} for one level and move count n."""
    jc, jo = (levels or LEVELS)[level]
    n_cand = len(gallery)
    cand_ids = np.arange(n_cand, dtype=np.int64)
    g_games = np.concatenate(gallery)
    g_cand = np.repeat(cand_ids, [len(g) for g in gallery])
    gk = keys_of(g_games)[:, :n]                                  # [g, n, 4]
    g_col = np.repeat(color[g_games], n)
    g_c = np.repeat(g_cand, n)
    tables = {}
    for j in (jc, jo):
        k = gk[:, :, j].ravel()
        ok = k != 0
        tables[j] = np.unique(_combine(k[ok], g_c[ok], g_col[ok]), return_counts=True)
    q_games = np.concatenate(query)
    q_owner = np.repeat(np.arange(len(query)), [len(x) for x in query])
    qk = keys_of(q_games)[:, :n]
    ctx, out = qk[:, :, jc].ravel(), qk[:, :, jo].ravel()
    q_col = np.repeat(color[q_games], n)
    owner = np.repeat(q_owner, n)
    ok = out != 0
    ctx, out, q_col, owner = ctx[ok], out[ok], q_col[ok], owner[ok]
    p_pop = (pop.count(n, jo, out, q_col) + 1.0 / 361) / (pop.count(n, jc, ctx, q_col) + 1.0)
    n_ctx = _lookup(*tables[jc], _combine(ctx[:, None], cand_ids[None, :], q_col[:, None])).astype(np.float64)
    n_out = _lookup(*tables[jo], _combine(out[:, None], cand_ids[None, :], q_col[:, None])).astype(np.float64)
    # moves are ordered by question (owner non-decreasing): per-question sums by cumulative sums
    bounds = np.searchsorted(owner, np.arange(len(query) + 1))
    result = {}
    for beta in BETAS:
        term = np.log((n_out + beta * p_pop[:, None]) / (n_ctx + beta))
        cs = np.vstack([np.zeros((1, n_cand)), np.cumsum(term, axis=0)])
        result[beta] = cs[bounds[1:]] - cs[bounds[:-1]]
    return result


def run(args) -> None:
    import torch
    from go_aicup.tasks.identity import data
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_transductive import load

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    s = env["summary"]
    scale = s["lambda"] * s["w"]
    color = env["color"]

    # B s2026 game vectors (as hung-seqB / pid2_b3em)
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    store = MoveStore(CACHE, meta["player_color"].to_numpy().astype(np.int8))
    assert np.array_equal(store.games, saved["game_index"]) and store.finite
    device = torch.device("cuda")
    head, step = load_head(HEAD, device)
    vec_b = game_vectors(head, store, np.arange(len(store.games)), device)
    print(f"B vectors (step {step}) in {time.time() - started:.0f}s", flush=True)

    # local keys: bench games (stage 0) and the population sample
    bench = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[bench["game_index"]] = np.arange(len(bench["game_index"]))
    local = bench["keys"]

    def keys_of(games):
        r = row[games]
        assert (r >= 0).all(), "game without local keys"
        return local[r]

    popfile = np.load(OUT / "pop_keys.npz")
    assert not np.isin(popfile["game_index"], bench["game_index"]).any(), "population overlaps the bench"
    pop = PopTable(popfile["keys"], color[popfile["game_index"]])
    print(f"population tables in {time.time() - started:.0f}s", flush=True)

    pos = env["position"]

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        other = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                         env["populations"], s["opening"])
                 + s["mu"] * twocolor.category_scores(env["codes"], gallery, query, 4))
        base = other + scale * twocolor.style_scores(vec_b, [pos[g] for g in gallery], [pos[g] for g in query])
        ml = {(n, level): level_scores(keys_of, color, pop, gallery, query, n, level)
              for n in NS for level in LEVELS}
        return candidates, q, base, ml

    # development selection
    combos = [(n, b, l1, l2) for n in NS for b in BETAS for l1 in LAMS for l2 in LAMS]
    dev_rows = []
    pool, draws = DEV
    for d in draws:
        candidates, q, base, ml = draw(pool, d)
        row_ = {"draw": d}
        for n, b, l1, l2 in combos:
            sc = base + l1 * ml[n, "L1"][b] + l2 * ml[n, "L2"][b]
            row_[f"n{n}_b{b:g}_l{l1:g}_{l2:g}"] = points(sc, candidates, q).mean()
        dev_rows.append(row_)
        print(f"dev draw {d} ({time.time() - started:.0f}s)", flush=True)
    dev = pd.DataFrame(dev_rows).set_index("draw")
    dev.to_csv(OUT / "dev_grid.csv")
    means = dev.mean()
    base_name = f"n{NS[0]}_b{BETAS[0]:g}_l0_0"
    key_of = {f"n{n}_b{b:g}_l{l1:g}_{l2:g}": (n, b, l1, l2) for n, b, l1, l2 in combos}
    best = max(means.index, key=lambda a: (round(means[a], 12), -(key_of[a][2] + key_of[a][3])))
    n, beta, l1, l2 = key_of[best]
    choice = {"chosen": {"n": n, "beta": beta, "lambda1": l1, "lambda2": l2},
              "dev_mean": float(means[best]), "dev_base": float(means[base_name]),
              "dev_gain": float(means[best] - means[base_name]),
              "dev_top": (means - means[base_name]).sort_values(ascending=False).head(10).to_dict(),
              "on_boundary": bool(max(l1, l2) == LAMS[-1])}
    (OUT / "dev_choice.json").write_text(json.dumps(choice, indent=1))
    print("dev choice:", json.dumps(choice, indent=1), flush=True)
    if l1 == 0 and l2 == 0:
        print("development chose no multi-level term; sealed draws not opened", flush=True)
        return

    # sealed evaluation, once
    rows = []
    for pool, draws in SEALED:
        for d in draws:
            candidates, q, base, ml = draw(pool, d)
            ks = q["k"].to_numpy()
            arms = {"base": base,
                    "ml": base + l1 * ml[n, "L1"][beta] + l2 * ml[n, "L2"][beta],
                    "L1_only": base + l1 * ml[n, "L1"][beta],
                    "L2_only": base + l2 * ml[n, "L2"][beta]}
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"sealed pool {pool} done ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"choice": choice, "tables": {}}
    for pool, _ in SEALED:
        table = paired(frame[frame["pool"] == pool], "base")
        report["tables"][f"pool{pool}"] = table.to_dict(orient="records")
        print(f"pool {pool}:\n{table.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"]["pool1"]).set_index("arm").loc["ml"]
    t0 = pd.DataFrame(report["tables"]["pool0"]).set_index("arm").loc["ml"]
    z = float(t1["d_all"] / t1["se_all"])
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    passed = (t1["d_all"] >= GATE["pool1"] and z >= GATE["z"]
              and min(buckets.values()) >= GATE["bucket"] and t0["d_all"] > GATE["pool0"])
    report["gate"] = {"pool1_delta": float(t1["d_all"]), "pool1_se": float(t1["se_all"]), "z": z,
                      "buckets": buckets, "pool0_delta": float(t0["d_all"]), "pass": bool(passed),
                      "rule": GATE}
    report["seconds"] = time.time() - started
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("GATE:", "PASS" if passed else "FAIL", json.dumps(report["gate"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("popkeys")
    p.add_argument("--workers", type=int, default=12)
    p.set_defaults(func=popkeys)
    sub.add_parser("run").set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
