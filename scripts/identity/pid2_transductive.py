"""Transductive gallery expansion for the added set (2026-11-14).

One player can own several questions and each candidate has only 10 gallery games,
so the other questions are unlabelled evidence about the candidates. Soft EM, style
term only:

    P      = softmax(S / tau) over candidates, S the fused bench score
             (opening + lambda * w * cosine + mu * HD-and-colour)
    A_c    = sum of the unit game vectors of c's gallery
    B_c    = sum_q P[q, c] * (sum of the unit game vectors of question q)
    centre = normalise(A_c + alpha * (B_c - P[q, c] * sum_q))   (question q left out)

the cosine to that centre replaces the style term of S; repeated `iters` times.

Parameters are chosen on pool 0 (the pool the bench summary was tuned on) and the
chosen arm is compared with the unchanged fused score on pool 1, draws 100-129 (the
draws of the 9,600-step paired run). CPU only; the opening/HD terms are cached per draw.

    python -m scripts.identity.pid2_transductive --run pid_ge2e_v4_d4_hd_9600
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import opening, twocolor
from go_aicup.tasks.identity.predict import train_prefix
from scripts.identity.pid2_paired import BANK, BENCH, HANDICAP, points

OUT = ROOT / "outputs" / "pid" / "pid2_transductive"
TAUS = (2.0, 4.0, 8.0, 16.0, 32.0)
ALPHAS = (0.25, 0.5, 1.0, 2.0)
ITERS = (1, 2)


def softmax(s: np.ndarray, tau: float) -> np.ndarray:
    z = (s - s.max(axis=1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def unit_sums(unit: np.ndarray, groups) -> np.ndarray:
    return np.stack([unit[g].sum(axis=0) for g in groups])


def load(run: str):
    meta, prefix = train_prefix("d4")
    bank = pd.read_parquet(BANK)
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy().astype(np.int64)
    color = meta["player_color"].to_numpy().astype(np.int64)
    pid = meta["player_id"].astype(str).to_numpy()
    hashes = opening.prefix_hashes(prefix)
    codes = np.minimum(handicap, 1) + 2 * color
    outside = ~np.isin(pid, bank["player_id"].unique())
    populations = (np.flatnonzero(outside & (color == 0)), np.flatnonzero(outside & (color == 1)))
    summary = json.loads((BENCH / run / "summary.json").read_text())
    if summary.get("use_per_k", False):
        raise ValueError("per-k lambda is not supported here")
    saved = np.load(BENCH / run / "embeddings.npz")
    position = np.full(len(meta), -1)
    position[saved["game_index"]] = np.arange(len(saved["game_index"]))
    unit = twocolor.normalise(saved[summary["style"].split("-")[1]].astype(np.float64))
    return dict(prefix=prefix, hashes=hashes, codes=codes, color=color, populations=populations,
                bank=bank, summary=summary, unit=unit, position=position)


def draw_cache(env, pool: int, draw: int):
    path = OUT / "cache" / f"pool{pool}_draw{draw}.npz"
    candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
    query = list(q["games"])
    if path.exists():
        saved = np.load(path)
        other = saved["other"]
    else:
        s = env["summary"]
        op = twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                     env["populations"], s["opening"])
        other = op + s["mu"] * twocolor.category_scores(env["codes"], gallery, query, 4)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, other=other)
    pos = env["position"]
    g = [pos[x] for x in gallery]
    qq = [pos[x] for x in query]
    assert all((x >= 0).all() for x in g + qq), "game without an embedding"
    return candidates, g, qq, q, other


def evaluate(env, pool: int, draws, arms):
    s = env["summary"]
    scale = s["lambda"] * s["w"]
    unit = env["unit"]
    rows = []
    for draw in draws:
        candidates, g, qq, q, other = draw_cache(env, pool, draw)
        cosine = twocolor.style_scores(unit, g, qq)
        ks = q["k"].to_numpy()
        for name, (tau, alpha, iters) in arms.items():
            if name == "base":
                cos = cosine
            elif name.startswith("oracle"):
                truth = (candidates[None, :] == q["player_id"].to_numpy()[:, None]).astype(float)
                cos = expanded(unit, g, qq, other, scale, cosine, tau, alpha, 1, truth)
            else:
                cos = expanded(unit, g, qq, other, scale, cosine, tau, alpha, iters)
            pts = points(other + scale * cos, candidates, q)
            rows.append({"draw": draw, "arm": name, **{str(k): pts[ks == k].mean() for k in cv.PID2_K},
                         "all": pts.mean()})
        print(f"pool {pool} draw {draw} done", flush=True)
    return pd.DataFrame(rows)


def expanded(unit, gallery, query, other, scale, cosine, tau, alpha, iters, truth=None) -> np.ndarray:
    a = unit_sums(unit, gallery)
    qsum = unit_sums(unit, query)
    qmean = twocolor.normalise(qsum)
    qdot = (qmean * qsum).sum(axis=1, keepdims=True)          # qmean . qsum
    qsq = (qsum ** 2).sum(axis=1, keepdims=True)
    cos = cosine
    for _ in range(iters):
        p = softmax(other + scale * cos, tau) if truth is None else truth
        full = a + alpha * (p.T @ qsum)                       # [C, D]
        ap = alpha * p
        num = qmean @ full.T - ap * qdot
        norm2 = (full ** 2).sum(axis=1)[None, :] - 2 * ap * (qsum @ full.T) + ap ** 2 * qsq
        cos = num / np.sqrt(np.maximum(norm2, 1e-12))
    return cos


def log_posterior(s: np.ndarray) -> np.ndarray:
    m = s.max(axis=1, keepdims=True)
    return s - m - np.log(np.exp(s - m).sum(axis=1, keepdims=True))


def merged(s: np.ndarray, tau: float, beta: float, same=None) -> np.ndarray:
    """S + beta * sum_{q' != q} m(q, q') * log P(c | q'), m = probability that q and q' share
    a player under the posteriors softmax(S / tau); `same` replaces m with the truth."""
    if same is None:
        p = softmax(s, tau)
        same = p @ p.T
    same = same.copy()
    np.fill_diagonal(same, 0.0)
    return s + beta * same @ log_posterior(s)


def evaluate_merge(env, pool: int, draws, arms):
    s_ = env["summary"]
    scale = s_["lambda"] * s_["w"]
    rows = []
    for draw in draws:
        candidates, g, qq, q, other = draw_cache(env, pool, draw)
        s = other + scale * twocolor.style_scores(env["unit"], g, qq)
        ks = q["k"].to_numpy()
        owner = q["player_id"].to_numpy()
        for name, (tau, beta) in arms.items():
            if name == "base":
                t = s
            elif name.startswith("oracle"):
                t = merged(s, 1.0, beta, (owner[:, None] == owner[None, :]).astype(float))
            else:
                t = merged(s, tau, beta)
            pts = points(t, candidates, q)
            rows.append({"draw": draw, "arm": name, **{str(k): pts[ks == k].mean() for k in cv.PID2_K},
                         "all": pts.mean()})
        print(f"pool {pool} draw {draw} done", flush=True)
    return pd.DataFrame(rows)


def pair_features(env, s: np.ndarray, qq, ks: np.ndarray, tau: float) -> np.ndarray:
    """[Q, Q, F] features for 'questions q and q' share a player'."""
    qmean = twocolor.normalise(unit_sums(env["unit"], qq))
    cos = qmean @ qmean.T
    p = softmax(s, tau)
    agree = np.log(np.maximum(p @ p.T, 1e-12))
    lk = np.log(ks.astype(float))
    a, b = np.meshgrid(lk, lk, indexing="ij")
    return np.stack([cos, cos * (a + b), cos * a * b, agree, a + b, a * b], axis=-1)


def fit_verifier(env, draws, tau):
    from sklearn.linear_model import LogisticRegression
    xs, ys = [], []
    for draw in draws:
        candidates, g, qq, q, other = draw_cache(env, 0, draw)
        s = other + env["scale"] * twocolor.style_scores(env["unit"], g, qq)
        owner = q["player_id"].to_numpy()
        f = pair_features(env, s, qq, q["k"].to_numpy(), tau)
        iu = np.triu_indices(len(owner), 1)
        xs.append(f[iu]); ys.append((owner[:, None] == owner[None, :])[iu])
    x, y = np.concatenate(xs), np.concatenate(ys)
    mu, sd = x.mean(0), x.std(0)
    model = LogisticRegression(C=1.0, max_iter=2000).fit((x - mu) / sd, y)
    print(f"verifier: {len(y)} pairs, positive rate {y.mean():.5f}, coef {np.round(model.coef_[0], 3)}", flush=True)
    return lambda f: model.predict_proba(((f.reshape(-1, f.shape[-1]) - mu) / sd))[:, 1].reshape(f.shape[:2])


def evaluate_verified(env, pool, draws, verifier, tau, betas, power=1.0):
    rows = []
    for draw in draws:
        candidates, g, qq, q, other = draw_cache(env, pool, draw)
        s = other + env["scale"] * twocolor.style_scores(env["unit"], g, qq)
        ks = q["k"].to_numpy()
        owner = q["player_id"].to_numpy()
        m = verifier(pair_features(env, s, qq, ks, tau))
        same = (owner[:, None] == owner[None, :])
        np.fill_diagonal(same, False)
        off = ~np.eye(len(ks), dtype=bool)
        for beta in betas:
            t = s if beta == 0 else merged(s, tau, beta, m)
            pts = points(t, candidates, q)
            rows.append({"draw": draw, "arm": f"v_b{beta:g}", **{str(k): pts[ks == k].mean() for k in cv.PID2_K},
                         "all": pts.mean(), "auc_note": float(m[same].mean()) if same.any() else np.nan,
                         "neg_mean": float(m[off & ~same].mean())})
        print(f"pool {pool} draw {draw} done", flush=True)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, base: str = "base") -> pd.DataFrame:
    cols = [str(k) for k in cv.PID2_K] + ["all"]
    b = frame[frame["arm"] == base].set_index("draw")[cols]
    out = []
    for arm, part in frame.groupby("arm"):
        d = part.set_index("draw")[cols] - b
        out.append({"arm": arm, **part[cols].mean().round(4).to_dict(),
                    "delta": round(d["all"].mean(), 4),
                    "se": round(d["all"].std(ddof=1) / np.sqrt(len(d)), 4) if arm != base else 0.0,
                    "pos": round((d["all"] > 0).mean(), 3)})
    return pd.DataFrame(out).sort_values("delta", ascending=False)


def self_check() -> None:
    """expanded() equals the explicit leave-one-out centre computation on random data."""
    rng = np.random.default_rng(0)
    unit = twocolor.normalise(rng.normal(size=(60, 8)))
    gallery = [np.arange(i * 5, i * 5 + 5) for i in range(6)]
    query = [np.array([30 + i, 40 + i]) if i % 2 else np.array([30 + i]) for i in range(10)]
    other = rng.normal(size=(10, 6))
    cosine = twocolor.style_scores(unit, gallery, query)
    got = expanded(unit, gallery, query, other, 3.0, cosine, 2.0, 0.7, 1)
    p = softmax(other + 3.0 * cosine, 2.0)
    qsum = unit_sums(unit, query)
    want = np.empty_like(got)
    for qi in range(10):
        for c in range(6):
            centre = unit[gallery[c]].sum(0) + 0.7 * sum(p[j, c] * qsum[j] for j in range(10) if j != qi)
            want[qi, c] = twocolor.normalise(qsum[qi]) @ twocolor.normalise(centre)
    assert np.allclose(got, want), np.abs(got - want).max()
    alpha0 = expanded(unit, gallery, query, other, 3.0, cosine, 2.0, 0.0, 1)
    assert np.allclose(alpha0, cosine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--tune-draws", type=int, default=20)
    parser.add_argument("--eval-draws", type=int, default=30)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="pool 1 ceiling: true owners of the other questions")
    parser.add_argument("--merge", action="store_true", help="query merging: tune on pool 0, evaluate on pool 1")
    parser.add_argument("--verify", action="store_true", help="query merging with a pair verifier fitted on pool 0")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("self-check passed")
        return
    env = load(args.run)
    OUT.mkdir(parents=True, exist_ok=True)
    env["scale"] = env["summary"]["lambda"] * env["summary"]["w"]
    if args.verify:
        tau, betas = 1.0, (0.0, 0.25, 0.5, 1.0)
        verifier = fit_verifier(env, range(100, 100 + args.tune_draws), tau)
        tune = evaluate_verified(env, 0, range(100, 100 + args.tune_draws), verifier, tau, betas)
        print("pool 0 (in-sample verifier):")
        print(summarise(tune.drop(columns=["auc_note", "neg_mean"]), base="v_b0").to_string(index=False))
        print("mean m on same-player pairs %.4f, on other pairs %.6f" % (tune.auc_note.mean(), tune.neg_mean.mean()))
        t = summarise(tune.drop(columns=["auc_note", "neg_mean"]), base="v_b0")
        best = t[t["arm"] != "v_b0"].iloc[0]["arm"]
        beta = float(best.split("_b")[1])
        ev = evaluate_verified(env, 1, range(100, 100 + args.eval_draws), verifier, tau, (0.0, beta))
        ev.to_csv(OUT / f"{args.run}_verify_eval_pool1.csv", index=False)
        cols = [str(k) for k in cv.PID2_K] + ["all"]
        a = ev[ev["arm"] == "v_b0"].set_index("draw")[cols]
        d = ev[ev["arm"] == best].set_index("draw")[cols] - a
        print(f"pool 1, chosen {best}:")
        print(pd.DataFrame({"base": a.mean(), "arm": a.mean() + d.mean(), "delta": d.mean(),
                            "se": d.std(ddof=1) / np.sqrt(len(d)), "pos": (d > 0).mean()}).round(4).to_string())
        print("pool 1 mean m same %.4f other %.6f" % (ev.auc_note.mean(), ev.neg_mean.mean()))
        return
    if args.merge:
        oracle = {f"oracle_b{b:g}": (1.0, b) for b in (0.5, 1.0)}
        grid = {"base": (0.0, 0.0)} | {f"m_t{t:g}_b{b:g}": (t, b) for t in (1, 2, 4, 8, 16, 32)
                                       for b in (0.125, 0.25, 0.5, 1.0)}
        tune = evaluate_merge(env, 0, range(100, 100 + args.tune_draws), grid)
        tune.to_csv(OUT / f"{args.run}_merge_tune_pool0.csv", index=False)
        table = summarise(tune)
        print("pool 0 merge tuning:")
        print(table.head(12).to_string(index=False), flush=True)
        best = table[table["arm"] != "base"].iloc[0]["arm"]
        ev = evaluate_merge(env, 1, range(100, 100 + args.eval_draws), {"base": grid["base"], best: grid[best]} | oracle)
        ev.to_csv(OUT / f"{args.run}_merge_eval_pool1.csv", index=False)
        print(f"pool 1 (chosen {best}):")
        cols = [str(k) for k in cv.PID2_K] + ["all"]
        a = ev[ev["arm"] == "base"].set_index("draw")[cols]
        for arm in [best] + list(oracle):
            d = ev[ev["arm"] == arm].set_index("draw")[cols] - a
            print(arm)
            print(pd.DataFrame({"base": a.mean(), "arm": a.mean() + d.mean(), "delta": d.mean(),
                                "se": d.std(ddof=1) / np.sqrt(len(d)), "pos": (d > 0).mean()}).round(4).to_string())
        return
    if args.oracle:
        arms = {"base": (0.0, 0.0, 0), **{f"oracle_a{a:g}": (1.0, a, 1) for a in (0.5, 1.0, 2.0)}}
        frame = evaluate(env, 1, range(100, 100 + args.eval_draws), arms)
        frame.to_csv(OUT / f"{args.run}_oracle_pool1.csv", index=False)
        print(summarise(frame).to_string(index=False))
        return
    grid = {"base": (0.0, 0.0, 0)}
    for tau, alpha, iters in itertools.product(TAUS, ALPHAS, ITERS):
        grid[f"t{tau:g}_a{alpha:g}_i{iters}"] = (tau, alpha, iters)
    tune = evaluate(env, 0, range(100, 100 + args.tune_draws), grid)
    tune.to_csv(OUT / f"{args.run}_tune_pool0.csv", index=False)
    table = summarise(tune)
    print("pool 0 tuning (delta vs base, all k):")
    print(table.head(15).to_string(index=False), flush=True)
    best = table[table["arm"] != "base"].iloc[0]["arm"]
    arms = {"base": grid["base"], best: grid[best]}
    ev = evaluate(env, 1, range(100, 100 + args.eval_draws), arms)
    ev.to_csv(OUT / f"{args.run}_eval_pool1.csv", index=False)
    cols = [str(k) for k in cv.PID2_K] + ["all"]
    a = ev[ev["arm"] == "base"].set_index("draw")[cols]
    b = ev[ev["arm"] == best].set_index("draw")[cols]
    d = b - a
    print(f"pool 1, {args.eval_draws} draws; chosen arm {best}")
    print(pd.DataFrame({"base": a.mean(), best: b.mean(), "delta": d.mean(),
                        "se": d.std(ddof=1) / np.sqrt(len(d)), "pos": (d > 0).mean()}).round(4).to_string())
    (OUT / f"{args.run}_result.json").write_text(json.dumps(
        {"chosen": best, "params": dict(zip(("tau", "alpha", "iters"), grid[best])),
         "pool0_top": table.head(10).to_dict("records"),
         "pool1": {"base": a.mean().round(4).to_dict(), "arm": b.mean().round(4).to_dict(),
                   "delta": d.mean().round(4).to_dict(),
                   "se": (d.std(ddof=1) / np.sqrt(len(d))).round(4).to_dict(),
                   "pos": (d > 0).mean().round(3).to_dict()}}, indent=1))


if __name__ == "__main__":
    main()
