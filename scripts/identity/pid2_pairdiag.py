"""Question-to-question same-player diagnostic for the added set (2026-11-14, CPU).

pid2_transductive showed that merging the questions of one player is worth +0.099 with
the true pairs and nothing with the pair scores tried there. This measures what a pair
score has to deliver and whether cheap ones do:

pair scores   cos     cosine of the two questions' mean style vectors
              open    opening model with each question as a gallery, both directions:
                      LR(i | j) + LR(j | i), LR relative to an empty gallery
              cand    log sum_c P(c | i) P(c | j), the candidate-mediated score
              lr_dir  logistic on cos, open and k            (no candidate information)
              lr_all  lr_dir plus cand
merge rule    edges = the n best pairs of a draw (direct), their connected components
              (closure), or only pairs that are each other's best match (mutual);
              T = S + beta * sum_{q' joined to q} log P(c | q')
degraded truth  the same rule fed with a share of the true pairs plus false pairs at a
              set precision; "random" picks both at random, "hard" keeps the true pairs
              lr_all ranks highest and adds the false pairs it ranks highest.

Verifier fit: pool 0 draws 200-219. Rule choice: pool 0 draws 100-119, then locked.
Report: pool 1 draws 100-129 (the draws of the paired 9,600-step run).

    python -m scripts.identity.pid2_pairdiag --run pid_ge2e_v4_d4_hd_9600
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from go_aicup.data import cv
from go_aicup.tasks.identity import twocolor
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_transductive import OUT, draw_cache, load, log_posterior, softmax, unit_sums

SCORES = ("cos", "open", "cand", "lr_dir", "lr_all")
EDGES = (10, 20, 40, 80, 160)
RULES = ("direct", "closure", "mutual")
BETAS = (0.25, 0.5, 1.0)
RECALLS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
PRECISIONS = (1.0, 0.9, 0.7, 0.5, 0.3)
COLS = [str(k) for k in cv.PID2_K] + ["all"]


def pair_opening(env, pool: int, draw: int, query) -> np.ndarray:
    path = OUT / "cache" / f"pair_open_pool{pool}_draw{draw}.npz"
    if path.exists():
        return np.load(path)["m"]
    gallery = list(query) + [np.zeros(0, dtype=np.int64)]
    s = twocolor.opening_scores(env["prefix"], env["hashes"], gallery, list(query), env["color"],
                                env["populations"], env["summary"]["opening"])
    lr = s[:, :-1] - s[:, -1:]
    m = lr + lr.T
    np.savez(path, m=m)
    return m


def draw_data(env, pool: int, draw: int) -> dict:
    candidates, g, qq, q, other = draw_cache(env, pool, draw)
    s = other + env["scale"] * twocolor.style_scores(env["unit"], g, qq)
    qmean = twocolor.normalise(unit_sums(env["unit"], qq))
    p = softmax(s, 1.0)
    owner = q["player_id"].to_numpy()
    same = owner[:, None] == owner[None, :]
    np.fill_diagonal(same, False)
    ks = q["k"].to_numpy()
    lk = np.log(ks.astype(float))
    a, b = np.meshgrid(lk, lk, indexing="ij")
    raw = {"cos": qmean @ qmean.T, "open": pair_opening(env, pool, draw, list(q["games"])),
           "cand": np.log(np.maximum(p @ p.T, 1e-12))}
    feats = np.stack([raw["cos"], raw["cos"] * (a + b), raw["cos"] * a * b, raw["open"],
                      raw["open"] / np.exp(np.maximum(a, b)), a + b, a * b, raw["cand"]], axis=-1)
    return dict(candidates=candidates, q=q, s=s, logp=log_posterior(s), same=same, ks=ks, raw=raw, feats=feats,
                base=points(s, candidates, q))


def fit_verifiers(env, draws):
    from sklearn.linear_model import LogisticRegression
    xs, ys = [], []
    for draw in draws:
        d = draw_data(env, 0, draw)
        iu = np.triu_indices(len(d["ks"]), 1)
        xs.append(d["feats"][iu]); ys.append(d["same"][iu])
    x, y = np.concatenate(xs), np.concatenate(ys)
    mu, sd = x.mean(0), x.std(0)
    models = {}
    for name, width in (("lr_dir", x.shape[1] - 1), ("lr_all", x.shape[1])):
        model = LogisticRegression(C=1.0, max_iter=5000).fit(((x - mu) / sd)[:, :width], y)
        print(f"{name}: {len(y)} pairs, {int(y.sum())} same, coef {np.round(model.coef_[0], 3)}", flush=True)
        models[name] = (model, width)

    def apply(d):
        z = (d["feats"].reshape(-1, x.shape[1]) - mu) / sd
        for name, (model, width) in models.items():
            d["raw"][name] = model.decision_function(z[:, :width]).reshape(d["same"].shape)
        return d
    return apply


def top_edges(score: np.ndarray, n: int, mutual: bool) -> np.ndarray:
    """bool [Q, Q] symmetric: the n best pairs (among mutual best matches if `mutual`)."""
    size = len(score)
    m = (score + score.T) / 2
    np.fill_diagonal(m, -np.inf)
    iu = np.triu_indices(size, 1)
    values = m[iu]
    if mutual:
        best = m.argmax(axis=1)
        ok = (best[iu[0]] == iu[1]) & (best[iu[1]] == iu[0])
        values = np.where(ok, values, -np.inf)
    n = min(n, int(np.isfinite(values).sum()))
    pick = np.argpartition(-values, n - 1)[:n] if n else np.zeros(0, dtype=int)
    edges = np.zeros((size, size), dtype=bool)
    edges[iu[0][pick], iu[1][pick]] = True
    return edges | edges.T


def closure(edges: np.ndarray) -> np.ndarray:
    _, label = connected_components(coo_matrix(edges), directed=False)
    joined = label[:, None] == label[None, :]
    np.fill_diagonal(joined, False)
    return joined


def merge_points(d, joined: np.ndarray, beta: float) -> np.ndarray:
    return points(d["s"] + beta * joined.astype(float) @ d["logp"], d["candidates"], d["q"])


def edge_stats(d, edges: np.ndarray, joined: np.ndarray) -> dict:
    n = edges.sum() / 2
    return {"edges": n, "precision": (edges & d["same"]).sum() / max(edges.sum(), 1),
            "recall": (edges & d["same"]).sum() / max(d["same"].sum(), 1),
            "coverage": joined.any(axis=1).mean(),
            "joined_precision": (joined & d["same"]).sum() / max(joined.sum(), 1),
            "largest_group": int(joined.sum(axis=1).max()) + 1}


def row(d, pts, **extra) -> dict:
    return {**extra, **{str(k): pts[d["ks"] == k].mean() for k in cv.PID2_K}, "all": pts.mean()}


def run_grid(data: dict, arms) -> pd.DataFrame:
    rows = []
    for draw, d in data.items():
        rows.append(row(d, d["base"], draw=draw, arm="base"))
        for score, n, rule, beta in arms:
            edges = top_edges(d["raw"][score], n, rule == "mutual")
            joined = closure(edges) if rule == "closure" else edges
            rows.append(row(d, merge_points(d, joined, beta), draw=draw, arm=f"{score}|{n}|{rule}|{beta:g}",
                            **edge_stats(d, edges, joined)))
    return pd.DataFrame(rows)


def deltas(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame[frame["arm"] == "base"].set_index("draw")[COLS]
    extra = [c for c in ("edges", "precision", "recall", "coverage", "joined_precision", "largest_group")
             if c in frame]
    out = []
    for arm, part in frame[frame["arm"] != "base"].groupby("arm", sort=False):
        diff = part.set_index("draw")[COLS] - base
        out.append({"arm": arm, **{f"d{c}": diff[c].mean() for c in COLS},
                    "se": diff["all"].std(ddof=1) / np.sqrt(len(diff)), "pos": (diff["all"] > 0).mean(),
                    "min": diff["all"].min(), "max": diff["all"].max(), **part[extra].mean().to_dict()})
    return pd.DataFrame(out)


def pr_table(data: dict) -> pd.DataFrame:
    """Per score, pooled over draws: precision at fixed recall and recall at fixed false-positive rate,
    for all pairs and by the smaller k of the pair."""
    rows = []
    cells = {"all": lambda lo: np.ones_like(lo, dtype=bool)} | {f"kmin{k}": (lambda lo, k=k: lo == k) for k in cv.PID2_K}
    for score in SCORES:
        vals, labs, lows = [], [], []
        for d in data.values():
            iu = np.triu_indices(len(d["ks"]), 1)
            m = (d["raw"][score] + d["raw"][score].T) / 2
            vals.append(m[iu]); labs.append(d["same"][iu])
            lows.append(np.minimum(d["ks"][iu[0]], d["ks"][iu[1]]))
        v, y, lo = np.concatenate(vals), np.concatenate(labs), np.concatenate(lows)
        for cell, pick in cells.items():
            keep = pick(lo)
            vv, yy = v[keep], y[keep]
            order = np.argsort(-vv)
            hit = np.cumsum(yy[order])
            total = max(int(yy.sum()), 1)
            rank = np.arange(1, len(vv) + 1)
            false = rank - hit
            item = {"score": score, "cell": cell, "pairs": len(vv), "same": int(yy.sum())}
            for r in (0.05, 0.1, 0.2, 0.3):
                at = int(np.searchsorted(hit, np.ceil(r * total)))
                item[f"prec@rec{r:g}"] = hit[at] / rank[at] if at < len(vv) else np.nan
            for f in (1e-4, 1e-3):
                at = int(np.searchsorted(false, np.ceil(f * (len(vv) - total)), side="right")) - 1
                item[f"rec@fpr{f:g}"] = hit[max(at, 0)] / total
            rows.append(item)
    return pd.DataFrame(rows)


def degraded(data: dict, seed: int = 0) -> pd.DataFrame:
    rows = []
    for draw, d in data.items():
        rng = np.random.default_rng([seed, draw])
        rows.append(row(d, d["base"], draw=draw, arm="base"))
        size = len(d["ks"])
        iu = np.triu_indices(size, 1)
        y = d["same"][iu]
        true, false = np.flatnonzero(y), np.flatnonzero(~y)
        v = ((d["raw"]["lr_all"] + d["raw"]["lr_all"].T) / 2)[iu]
        orders = {"random": (rng.permutation(true), rng.permutation(false)),
                  "hard": (true[np.argsort(-v[true])], false[np.argsort(-v[false])])}
        for mode, (t_order, f_order) in orders.items():
            for recall, precision in itertools.product(RECALLS, PRECISIONS):
                n_true = int(round(recall * len(true)))
                n_false = int(round(n_true * (1 - precision) / precision))
                pick = np.concatenate([t_order[:n_true], f_order[:n_false]])
                edges = np.zeros((size, size), dtype=bool)
                edges[iu[0][pick], iu[1][pick]] = True
                edges |= edges.T
                for rule, beta in itertools.product(("direct", "closure"), (0.5, 1.0)):
                    joined = closure(edges) if rule == "closure" else edges
                    rows.append(row(d, merge_points(d, joined, beta), draw=draw,
                                    arm=f"{mode}|{recall:g}|{precision:g}|{rule}|{beta:g}"))
    return pd.DataFrame(rows)


def show(frame: pd.DataFrame, n: int | None = None) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 400):
        print((frame if n is None else frame.head(n)).round(4).to_string(index=False), flush=True)


def self_check() -> None:
    rng = np.random.default_rng(0)
    score = rng.normal(size=(8, 8))
    score = score + score.T
    edges = top_edges(score, 3, False)
    assert edges.sum() == 6 and (edges == edges.T).all() and not edges.diagonal().any()
    m = score.copy()
    np.fill_diagonal(m, -np.inf)
    assert np.isclose(m[edges].min(), np.sort(m[np.triu_indices(8, 1)])[-3])
    mutual = top_edges(score, 99, True)
    best = m.argmax(axis=1)
    assert all(best[i] == j and best[j] == i for i, j in zip(*np.nonzero(mutual)))
    chain = np.zeros((5, 5), dtype=bool)
    chain[0, 1] = chain[1, 0] = chain[1, 2] = chain[2, 1] = True
    joined = closure(chain)
    assert joined[0, 2] and joined.sum() == 6 and not joined[3].any()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fit-draws", type=int, default=20)
    parser.add_argument("--tune-draws", type=int, default=20)
    parser.add_argument("--eval-draws", type=int, default=30)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("self-check passed")
        return
    env = load(args.run)
    env["scale"] = env["summary"]["lambda"] * env["summary"]["w"]
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    bank = env["bank"]
    players = [set(bank.loc[bank["pool"] == p, "player_id"]) for p in (0, 1)]
    print(f"pool players {len(players[0])} / {len(players[1])}, shared {len(players[0] & players[1])}", flush=True)

    apply = fit_verifiers(env, range(200, 200 + args.fit_draws))
    tune = {d: apply(draw_data(env, 0, d)) for d in range(100, 100 + args.tune_draws)}
    print("pool 0 tuning draws loaded; same-player pairs per draw %.1f"
          % np.mean([d["same"].sum() / 2 for d in tune.values()]), flush=True)

    print("\n== pair scores, pool 0 (tuning draws) ==")
    pr0 = pr_table(tune)
    show(pr0)

    print("\n== degraded truth, pool 0: delta of the all-k score ==")
    deg = deltas(degraded(tune))
    deg[["mode", "recall", "precision", "rule", "beta"]] = deg["arm"].str.split("|", expand=True)
    deg.to_csv(OUT / f"{args.run}_pairdiag_degraded_pool0.csv", index=False)
    for (mode, rule, beta), part in deg.groupby(["mode", "rule", "beta"], sort=False):
        print(f"{mode}, {rule}, beta {beta} (rows recall, columns precision)")
        table = part.pivot(index="recall", columns="precision", values="dall")
        show(table.loc[[f"{r:g}" for r in RECALLS], [f"{p:g}" for p in PRECISIONS]].reset_index())

    print("\n== merge rules with the real pair scores, pool 0 ==")
    grid = list(itertools.product(SCORES, EDGES, RULES, BETAS))
    table = deltas(run_grid(tune, grid)).sort_values("dall", ascending=False)
    table.to_csv(OUT / f"{args.run}_pairdiag_tune_pool0.csv", index=False)
    show(table, 25)
    print("best arm per score:")
    show(table.groupby(table["arm"].str.split("|").str[0], sort=False).head(1))
    chosen = table.iloc[0]["arm"]
    score, n, rule, beta = chosen.split("|")
    locked = [(score, int(n), rule, float(beta))]
    print(f"\nlocked rule: {chosen} (pool 0 delta {table.iloc[0]['dall']:+.4f}, se {table.iloc[0]['se']:.4f})", flush=True)
    del tune

    held = {d: apply(draw_data(env, 1, d)) for d in range(100, 100 + args.eval_draws)}
    print("\n== pair scores, pool 1 ==")
    pr1 = pr_table(held)
    show(pr1)
    result = deltas(run_grid(held, locked))
    result.to_csv(OUT / f"{args.run}_pairdiag_eval_pool1.csv", index=False)
    print(f"\n== locked rule on pool 1, {args.eval_draws} draws ==")
    show(result)
    base = float(np.mean([d["base"].mean() for d in held.values()]))
    passed = bool(result.iloc[0]["dall"] >= 0.005 and result.iloc[0]["pos"] >= 0.7)
    print(f"pool 1 base {base:.4f}; gate (delta >= +0.005 and >= 70% of draws positive): {'pass' if passed else 'fail'}")
    (OUT / f"{args.run}_pairdiag.json").write_text(json.dumps({
        "locked": chosen, "pool0_top": table.head(10).to_dict("records"), "pool1_base": base,
        "pool1": result.to_dict("records"), "gate_pass": passed,
        "shared_players": len(players[0] & players[1]),
        "pair_scores_pool0": pr0.to_dict("records"), "pair_scores_pool1": pr1.to_dict("records")}, indent=1,
        default=float))


if __name__ == "__main__":
    main()
