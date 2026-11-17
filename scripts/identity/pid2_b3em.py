"""seqB3 sealed evaluation + candidate-centroid EM (H2-A) on B vectors (2026-11-17).

Part 1: fresh sealed draws (pool 0 700-719, pool 1 700-729): A, B s2026, B3 (style cosine
averaged over B s2026 / s2027 / s2028, as hung-seqB3.csv). Fusion as the submission: opening +
16 * w * cosine + category, w = 9,600-step value.
Part 2: H2-A (pid2_transductive.expanded: each candidate centroid is expanded by the question
vectors weighted by posterior p = softmax(S / tau), leave-one-out) applied per seed, p from the
B3 fused score. tau / alpha / iters chosen on development draws pool 0 100-119, then the chosen
setting is evaluated once on the sealed pool 1 700-729 (and pool 0 700-719) against B3.
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, twocolor
from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
from scripts.identity.pid2_diag_d123a import COLS, bucket_means
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_transductive import load, softmax, unit_sums

RUN = "pid_ge2e_v4_d4_hd_9600"
BENCH = ROOT / "outputs" / "pid" / "pid2_bench"
HEADS = ROOT / "outputs" / "pid" / "pid2_seqagg"
from pathlib import Path
CACHE = Path("/home/jason1122/pid2_seqagg_cache/bench")
OUT = ROOT / "outputs" / "pid" / "pid2_b3em"
SEEDS = {"s2026": "head_B", "s2027": "head_B_s2027", "s2028": "head_B_s2028"}
GRID = list(itertools.product((1.0, 2.0, 4.0, 8.0), (0.25, 0.5, 1.0, 2.0), (1, 2)))


def expanded_multi(units, g, qq, other, scale, tau, alpha, iters):
    """per-seed centroid expansion with a shared posterior from the averaged fused score."""
    a = [unit_sums(u, g) for u in units]
    qs = [unit_sums(u, qq) for u in units]
    cos = [twocolor.normalise(q) @ twocolor.normalise(x).T for q, x in zip(qs, a)]
    for _ in range(iters):
        p = softmax(other + scale * np.mean(cos, axis=0), tau)
        new = []
        for x, qsum in zip(a, qs):
            qmean = twocolor.normalise(qsum)
            qdot = (qmean * qsum).sum(axis=1, keepdims=True)
            qsq = (qsum ** 2).sum(axis=1, keepdims=True)
            full = x + alpha * (p.T @ qsum)
            ap = alpha * p
            num = qmean @ full.T - ap * qdot
            norm2 = (full ** 2).sum(axis=1)[None, :] - 2 * ap * (qsum @ full.T) + ap ** 2 * qsq
            new.append(num / np.sqrt(np.maximum(norm2, 1e-12)))
        cos = new
    return np.mean(cos, axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    env = load(RUN)
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    store = MoveStore(CACHE, meta["player_color"].to_numpy().astype(np.int8))
    assert np.array_equal(store.games, saved["game_index"]) and store.finite
    rows = np.arange(len(store.games))
    vec = {"A": game_vectors(None, store, rows, device)}
    for name, d in SEEDS.items():
        head, step = load_head(HEADS / d / "best.pt", device)
        vec[name] = game_vectors(head, store, rows, device)
        print(name, "step", step, flush=True)
    units = {k: twocolor.normalise(v.astype(np.float64)) for k, v in vec.items()}
    w = float(saved["w"])
    scale = 16.0 * w
    s = env["summary"]

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        other = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                         env["populations"], s["opening"])
                 + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
        pos = env["position"]
        return candidates, [pos[x] for x in gallery], [pos[x] for x in query], q, other

    def b3_cos(g, qq):
        return np.mean([twocolor.style_scores(vec[k], g, qq) for k in SEEDS], axis=0)

    # development tuning of H2-A on B3
    dev = []
    for d in range(100, 120):
        candidates, g, qq, q, other = draw(0, d)
        ks = q["k"].to_numpy()
        row = {"base": points(other + scale * b3_cos(g, qq), candidates, q).mean()}
        for tau, alpha, iters in GRID:
            c = expanded_multi([units[k] for k in SEEDS], g, qq, other, scale, tau, alpha, iters)
            row[f"t{tau:g}_a{alpha:g}_i{iters}"] = points(other + scale * c, candidates, q).mean()
        dev.append(row)
        print(f"dev draw {d}", flush=True)
    dev = pd.DataFrame(dev)
    gain = (dev.drop(columns="base").sub(dev["base"], axis=0)).mean().sort_values(ascending=False)
    best = gain.index[0]
    tau, alpha, iters = next(x for x in GRID if f"t{x[0]:g}_a{x[1]:g}_i{x[2]}" == best)
    print("dev top:\n", gain.head(8).round(4).to_string(), flush=True)

    # sealed evaluation
    out = []
    for pool, n in ((0, 20), (1, 30)):
        for d in range(700, 700 + n):
            candidates, g, qq, q, other = draw(pool, d)
            ks = q["k"].to_numpy()
            arms = {"A": twocolor.style_scores(vec["A"], g, qq), "B": twocolor.style_scores(vec["s2026"], g, qq),
                    "B3": b3_cos(g, qq),
                    "B3_em": expanded_multi([units[k] for k in SEEDS], g, qq, other, scale, tau, alpha, iters)}
            for name, c in arms.items():
                out.append({"pool": pool, "draw": d, "arm": name,
                            **bucket_means(points(other + scale * c, candidates, q), ks)})
        print(f"sealed pool {pool} done", flush=True)
    frame = pd.DataFrame(out)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"dev_choice": {"tau": tau, "alpha": alpha, "iters": iters, "dev_gain": float(gain.iloc[0])},
              "dev_top": gain.head(8).to_dict(), "sealed": {}}
    for pool in (0, 1):
        wide = frame[frame.pool == pool].pivot(index="draw", columns="arm", values=COLS)
        for arm, base in (("B", "A"), ("B3", "A"), ("B3", "B"), ("B3_em", "B3")):
            d = wide.xs(arm, axis=1, level=1) - wide.xs(base, axis=1, level=1)
            key = f"pool{pool} {arm}-{base}"
            report["sealed"][key] = {"abs": wide.xs(arm, axis=1, level=1).mean().to_dict(),
                                     "delta": d.mean().to_dict(), "se_all": float(d["all"].std(ddof=1) / np.sqrt(len(d))),
                                     "pos": float((d["all"] > 0).mean())}
            print(key, {k: round(v, 4) for k, v in d.mean().items()},
                  "se", round(report["sealed"][key]["se_all"], 4), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
