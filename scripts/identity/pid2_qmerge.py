"""E1: question merging with a same-player verifier built on the current fusion (2026-11-20,
registered in RESULTS.md before the run).

One added-set player can own several questions. If questions q and q' share a player, the
posterior of q' is evidence about q's answer. Base score (= ogsa12w added-set form)
    S = rest + 16 w 1.5 (0.25 cos_seqB + 0.75 mean(cos_A1, cos_A2)),  rest = opening + category + L1 + L2
Merge rule
    S'_q = S_q + beta * sum_{q' != q} [m >= theta] log(1 - m + m C P_tau(c | q'))
m = verifier probability that q and q' share a player, C = number of candidates,
P_tau = softmax(S / tau). m = 1 multiplies in q' (merging two questions), m = 0 changes nothing.

Verifier features per question pair (symmetrised): question-centre cosine of seqB / A1 / A2; the
opening, L1, L2 and category scores of q against q' used as a one-person gallery (row-centred);
posterior overlap at tau 1 and 8; same top-1; log k terms. Logistic regression and
HistGradientBoosting, chosen by average precision on the tuning draws.

Draws: pool 0 2200-2219 fit the verifier, 2220-2239 choose verifier and (tau, beta, theta),
2240-2259 sealed; pool 1 2200-2229 sealed. Switch rule (11/20): sealed pool 1 delta > 0 and
z >= 2.5, every k >= -0.005, sealed pool 0 delta > 0.

  python -m scripts.identity.pid2_qmerge compute   # GPU: per-draw base scores and pair matrices
  python -m scripts.identity.pid2_qmerge analyse   # CPU: verifier, tuning, sealed evaluation
  python -m scripts.identity.pid2_qmerge selfcheck # CPU: merge-rule checks on synthetic data
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv

OUT = Path("outputs/pid/qmerge")
FIT = [(0, d) for d in range(2200, 2220)]
TUNE = [(0, d) for d in range(2220, 2240)]
SEAL0 = [(0, d) for d in range(2240, 2260)]
SEAL1 = [(1, d) for d in range(2200, 2230)]
CHECK = (1, 2000)                      # reproduces A12w of pid2_ogs_a2_eval on this draw
TAUS = (1.0, 2.0, 4.0, 8.0, 16.0)
BETAS = (0.25, 0.5, 1.0)
THETAS = (0.0, 0.1, 0.3, 0.5)
PAIRS = ("cos_B", "cos_A1", "cos_A2", "open", "L1", "L2", "cat")


def path(pool: int, draw: int) -> Path:
    return OUT / "draws" / f"pool{pool}_draw{draw}.npz"


# ---------------------------------------------------------------- GPU stage
def compute() -> None:
    import torch

    from go_aicup.tasks.identity import data, twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_diag_d123a import bucket_means
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
    from scripts.identity.pid2_ogs_a2_eval import ENC, OUT as A2OUT, RUN
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    (OUT / "draws").mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    vec = {}
    for name, (cache, head_file) in ENC.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, _ = load_head(Path(head_file), device)
        vec[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
    print("vectors done", flush=True)
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda games: b1["keys"][row[games]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popf["keys"], color[popf["game_index"]])

    def rest_parts(gallery, query):
        return {"open": twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                                env["populations"], env["summary"]["opening"]),
                "cat": twocolor.category_scores(env["codes"], gallery, query, 4),
                "L1": level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]],
                "L2": level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]]}

    def base(parts, c):
        rest = parts["open"] + 1.0 * parts["cat"] + s1["lambda1"] * parts["L1"] + s1["lambda2"] * parts["L2"]
        return rest + scale * 1.5 * (0.25 * c["B"] + 0.75 * 0.5 * (c["A1"] + c["A2"]))

    def one(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
        s = base(rest_parts(gallery, query), {k: twocolor.style_scores(v, g, qq) for k, v in vec.items()})
        return candidates, query, qq, q, s

    # reproduction gate: A12w of the A2 evaluation on one sealed draw
    candidates, _, _, q, s = one(*CHECK)
    got = bucket_means(points(s, candidates, q), q["k"].to_numpy())
    ref = pd.read_csv(A2OUT / "sealed_per_draw.csv")
    ref = ref[(ref["pool"] == CHECK[0]) & (ref["draw"] == CHECK[1]) & (ref["arm"] == "A12w")].iloc[0]
    diff = max(abs(got[c] - ref[c]) for c in got)
    print(f"reproduction of A12w pool {CHECK[0]} draw {CHECK[1]}: max diff {diff:.2e}", flush=True)
    assert diff <= 1e-9, (got, ref.to_dict())

    for pool, d in FIT + TUNE + SEAL0 + SEAL1:
        if path(pool, d).exists():
            continue
        candidates, query, qq, q, s = one(pool, d)
        pair = rest_parts(query, query)                       # q' games as a one-person gallery
        for k, v in vec.items():
            u = twocolor.normalise(np.stack([twocolor.normalise(twocolor.normalise(v[x].astype(np.float64)).mean(0))
                                             for x in qq]))
            pair[f"cos_{k}"] = u @ u.T
        owner = np.searchsorted(candidates, q["player_id"].to_numpy())
        assert (candidates[owner] == q["player_id"].to_numpy()).all()
        np.savez(path(pool, d), S=s, owner=owner, k=q["k"].to_numpy(), n_cand=len(candidates),
                 **{k: pair[k].astype(np.float32) for k in PAIRS})
        print(f"pool {pool} draw {d} done", flush=True)


# ---------------------------------------------------------------- CPU stage
def softmax(s: np.ndarray, tau: float) -> np.ndarray:
    z = (s - s.max(axis=1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def centred(x: np.ndarray) -> np.ndarray:
    """row-centre without the diagonal, then symmetrise."""
    x = x.astype(np.float64).copy()
    np.fill_diagonal(x, np.nan)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    np.fill_diagonal(x, 0.0)
    return x + x.T


def features(d: dict) -> np.ndarray:
    """[Q, Q, F] pair features."""
    s, k = d["S"], d["k"].astype(np.float64)
    lk = np.log(k)
    a, b = np.meshgrid(lk, lk, indexing="ij")
    cos = (d["cos_B"] + d["cos_A1"] + d["cos_A2"]).astype(np.float64) / 3
    out = [d["cos_B"], d["cos_A1"], d["cos_A2"], cos * (a + b), cos * a * b]
    out += [centred(d[x]) for x in ("open", "L1", "L2", "cat")]
    for tau in (1.0, 8.0):
        p = softmax(s, tau)
        out.append(np.log(np.maximum(p @ p.T, 1e-300)))
    top = s.argmax(axis=1)
    out += [(top[:, None] == top[None, :]).astype(np.float64), a + b, a * b]
    return np.stack([np.asarray(x, np.float64) for x in out], axis=-1)


def load_draw(pool: int, d: int) -> dict:
    return dict(np.load(path(pool, d)))


def upper(d: dict):
    iu = np.triu_indices(len(d["k"]), 1)
    same = d["owner"][:, None] == d["owner"][None, :]
    return features(d)[iu], same[iu]


def fit_verifiers(draws):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    xs, ys = zip(*(upper(load_draw(*x)) for x in draws))
    x, y = np.concatenate(xs), np.concatenate(ys)
    mu, sd = x.mean(0), x.std(0) + 1e-12
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((x - mu) / sd, y)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=2026).fit(x, y)
    print(f"verifier fit: {len(y)} pairs, positive rate {y.mean():.5f}", flush=True)
    return {"lr": lambda f: lr.predict_proba((f - mu) / sd)[:, 1],
            "hgb": lambda f: hgb.predict_proba(f)[:, 1]}


def pair_m(verifier, d: dict) -> np.ndarray:
    f = features(d)
    q = len(d["k"])
    m = verifier(f.reshape(-1, f.shape[-1])).reshape(q, q)
    m = 0.5 * (m + m.T)
    np.fill_diagonal(m, 0.0)
    return m


def merge_term(s: np.ndarray, m: np.ndarray, tau: float, theta: float) -> np.ndarray:
    """sum_{q'} [m >= theta] log(1 - m + m C P_tau(c | q')), [Q, C]."""
    c = s.shape[1]
    p = np.maximum(softmax(s, tau), 1e-300)          # m = 1 with P underflowing to 0 would give -inf
    m = np.where(m >= theta, m, 0.0) if theta > 0 else m
    out = np.zeros_like(s)
    for i in range(len(s)):
        j = np.flatnonzero(m[i] > 0)
        if len(j):
            mi = m[i, j][:, None]
            out[i] = np.log1p(mi * (c * p[j] - 1.0)).sum(axis=0)
    return out


def score(d: dict, s: np.ndarray) -> dict:
    from go_aicup.tasks.identity import assign
    from scripts.identity.pid2_diag_d123a import bucket_means
    w = np.exp(-np.arange(5.0))
    top = assign.top5(s)
    pts = ((top == d["owner"][:, None]) * w).sum(axis=1)
    return bucket_means(pts, d["k"])


def separation(m: np.ndarray, d: dict) -> dict:
    same = d["owner"][:, None] == d["owner"][None, :]
    np.fill_diagonal(same, False)
    off = ~np.eye(len(m), dtype=bool)
    has = same.any(axis=1)
    return {"m_same": float(m[same].mean()) if same.any() else np.nan,
            "m_other": float(m[off & ~same].mean()),
            "mass_true": float((m * same).sum(axis=1)[has].mean()) if has.any() else np.nan,
            "mass_false": float((m * (off & ~same)).sum(axis=1).mean())}


def analyse() -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    from scripts.identity.pid2_diag_d123a import paired

    verifiers = fit_verifiers(FIT)
    tune = {x: load_draw(*x) for x in TUNE}
    ap = {}
    for name, v in verifiers.items():
        ys, ps = [], []
        for d in tune.values():
            iu = np.triu_indices(len(d["k"]), 1)
            ys.append((d["owner"][:, None] == d["owner"][None, :])[iu])
            ps.append(pair_m(v, d)[iu])
        y, p = np.concatenate(ys), np.concatenate(ps)
        ap[name] = {"ap": float(average_precision_score(y, p)), "auc": float(roc_auc_score(y, p))}
    chosen_v = max(ap, key=lambda n: ap[n]["ap"])
    verifier = verifiers[chosen_v]
    print("verifier on tuning draws:", ap, "chosen", chosen_v, flush=True)

    rows = []
    for (pool, dr), d in tune.items():
        m = pair_m(verifier, d)
        rows.append({"draw": dr, "arm": "base", **score(d, d["S"])})
        for tau, theta in itertools.product(TAUS, THETAS):
            t = merge_term(d["S"], m, tau, theta)
            for beta in BETAS:
                rows.append({"draw": dr, "arm": f"t{tau:g}_b{beta:g}_th{theta:g}",
                             **score(d, d["S"] + beta * t)})
        print(f"tune draw {dr} done", flush=True)
    tframe = pd.DataFrame(rows)
    tframe.to_csv(OUT / "tune_pool0.csv", index=False)
    ttab = paired(tframe, "base").sort_values("d_all", ascending=False)
    best = ttab[ttab["arm"] != "base"].iloc[0]["arm"]
    tau, beta, theta = (float(x[1:].lstrip("bh")) for x in best.split("_"))
    edge = {"tau": tau in (TAUS[0], TAUS[-1]), "beta": beta in (BETAS[0], BETAS[-1]),
            "theta": theta == THETAS[-1]}
    print(ttab.head(10).round(4).to_string(index=False), flush=True)
    print(f"chosen {best} (grid edge: {edge})", flush=True)

    report = {"verifier": ap, "chosen_verifier": chosen_v, "chosen_arm": best,
              "params": {"tau": tau, "beta": beta, "theta": theta}, "grid_edge": edge,
              "tune_top": ttab.head(10).to_dict("records"), "sealed": {}, "separation": {}}
    for label, draws in (("pool0", SEAL0), ("pool1", SEAL1)):
        rows, sep, ys, ps = [], [], [], []
        for pool, dr in draws:
            d = load_draw(pool, dr)
            m = pair_m(verifier, d)
            truth = (d["owner"][:, None] == d["owner"][None, :]).astype(np.float64)
            np.fill_diagonal(truth, 0.0)
            rows.append({"draw": dr, "arm": "base", **score(d, d["S"])})
            rows.append({"draw": dr, "arm": "merge", **score(d, d["S"] + beta * merge_term(d["S"], m, tau, theta))})
            for ob in (0.5, 1.0):
                rows.append({"draw": dr, "arm": f"oracle_b{ob:g}",
                             **score(d, d["S"] + ob * merge_term(d["S"], truth, tau, 0.5))})
            sep.append(separation(m, d))
            iu = np.triu_indices(len(m), 1)
            ys.append(truth[iu] > 0)
            ps.append(m[iu])
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / f"sealed_{label}.csv", index=False)
        tab = paired(frame, "base")
        report["sealed"][label] = tab.to_dict("records")
        y, p = np.concatenate(ys), np.concatenate(ps)
        report["separation"][label] = {**pd.DataFrame(sep).mean().to_dict(),
                                       "ap": float(average_precision_score(y, p)),
                                       "auc": float(roc_auc_score(y, p))}
        print(f"sealed {label}:\n{tab.round(4).to_string(index=False)}", flush=True)
        print(f"separation {label}: {report['separation'][label]}", flush=True)
    t1 = pd.DataFrame(report["sealed"]["pool1"]).set_index("arm").loc["merge"]
    t0 = pd.DataFrame(report["sealed"]["pool0"]).set_index("arm").loc["merge"]
    z = float(t1["d_all"] / t1["se_all"])
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    report["switch"] = {"pool1_delta": float(t1["d_all"]), "z": z, "buckets": buckets,
                        "pool0_delta": float(t0["d_all"]),
                        "pass": bool(t1["d_all"] > 0 and z >= 2.5 and min(buckets.values()) >= -0.005
                                     and t0["d_all"] > 0)}
    print("SWITCH:", json.dumps(report["switch"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


def selfcheck() -> None:
    rng = np.random.default_rng(0)
    q, c = 30, 12
    s = rng.normal(size=(q, c)) * 3
    zero = merge_term(s, np.zeros((q, q)), 2.0, 0.0)
    assert np.allclose(zero, 0.0)
    m = np.zeros((q, q)); m[0, 1] = m[1, 0] = 1.0
    t = merge_term(s, m, 2.0, 0.0)
    want = np.log(c * softmax(s, 2.0)[1])
    assert np.allclose(t[0], want) and np.allclose(t[2:], 0.0)
    m2 = np.full((q, q), 0.05); np.fill_diagonal(m2, 0.0)
    assert np.allclose(merge_term(s, m2, 2.0, 0.1), 0.0)        # theta gate removes all pairs
    x = rng.normal(size=(q, q)); x = x + 7.0 * np.arange(q)[:, None]
    cx = centred(x)
    assert np.allclose(cx, cx.T) and np.allclose(np.diag(cx), 0.0)
    # synthetic: true merges help, random merges do not
    owner = rng.integers(0, c, size=q)
    s = rng.normal(size=(q, c)); s[np.arange(q), owner] += 1.0
    truth = (owner[:, None] == owner[None, :]).astype(float); np.fill_diagonal(truth, 0.0)
    acc = lambda x: (x.argmax(1) == owner).mean()
    assert acc(s + merge_term(s, truth, 1.0, 0.0)) >= acc(s)
    print("selfcheck passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("compute", "analyse", "selfcheck"))
    args = parser.parse_args()
    if args.command == "compute":
        selfcheck()
        compute()
        analyse()
    elif args.command == "analyse":
        analyse()
    else:
        selfcheck()


if __name__ == "__main__":
    main()
