"""Opponent identity link v2, learned per-game evidence (2026-11-23, user "升基底 並往這個方向訓練"; RESULTS.md)

Base = 845986 (SOFT tau 3.5, alpha 0.35, DELTA kernel on all Valid-S; Public 0.7726). The soft kernel is hand-made.
LEARN replaces it by a multinomial logistic regression trained on the opponent sides of every Valid-S HD<=1 game
(62,867, owner rank known) that maps the game's top-10 link to P(owner rank):
  phi = [ h_tau(r_j) for tau in TAUS ]                    rank histograms of the top-10 weighted softmax((f_i-f_1)/tau)
        + h_3.5 x gap-bin one-hot (5 bins)                 trust depends on how clear the top-1 is
        + h_3.5 x [HD1 & owner black], h_3.5 x [HD1 & owner white]   handicap shift
        + gap-bin one-hot + (HD, owner colour) one-hot
  l_g(r) = log P_LR(r | phi_g) - log prior(r),  prior = Valid-S owner-rank shares
  P(r | q) ~ P_B2(r | q) * exp(alpha * sum_g l_g(r))
C chosen by player-grouped 5-fold log-loss on Valid-S (C in CS). Cells: LEARN alpha in ALPHAS, plus a small soft grid
(SOFT tau in SOFT_TAUS x alpha in SOFT_ALPHAS). Selection on E1 (five-seed mean Equal-k against the base cell
SOFT3.5_0.35); E2 opened for the chosen cell only. Export `<job>-opp3-l10.csv` if chosen - base > 0 on both E1 and
E2 (LB probe; the user decides promotion). Export fits the LR on Valid-S + u_opp.

    python -m scripts.strength.rank_opp_learn --job <jobid> [--force-cell LEARN_0.5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as more
import scripts.strength.rank_nonlin as nl
import scripts.strength.rank_opp_fuse as fu
import scripts.strength.rank_opp_refine as rf
import scripts.strength.rank_prod_a0_fusion as prod
import scripts.strength.rank_qmerge as qm

OUT = fu.OUT
TAUS = (1.0, 3.5, 7.0, 14.0)
CS = (0.01, 0.1, 1.0)
ALPHAS = (0.25, 0.35, 0.5, 0.7)
SOFT_TAUS = (2.0, 5.0)
SOFT_ALPHAS = (0.35, 0.45)
BASE = "SOFT3.5_0.35"
SEED = 2039


def features(rows: pd.DataFrame, top: dict) -> np.ndarray:
    i = rows.index.to_numpy()
    val, idx = top["val"][i].astype(np.float64), top["idx"][i]
    rj = top["cand_rank"][idx]
    onehot = np.eye(10)[rj]                                                  # [n, 10 cand, 10]
    hist = {}
    for tau in TAUS:
        w = np.exp((val - val[:, :1]) / tau)
        w /= w.sum(1, keepdims=True)
        hist[tau] = (w[:, :, None] * onehot).sum(1)
    b = np.eye(fu.N_BIN)[rows["bin"].to_numpy()]
    hd, oc = rows["hd"].to_numpy(), rows["oc"].to_numpy()
    h1b = ((hd == 1) & (oc == 0)).astype(float)[:, None]
    h1w = ((hd == 1) & (oc == 1)).astype(float)[:, None]
    hdoc = np.eye(4)[2 * hd + oc]
    blocks = [hist[t] for t in TAUS] + [(hist[3.5][:, None, :] * b[:, :, None]).reshape(len(rows), -1),
                                         hist[3.5] * h1b, hist[3.5] * h1w, b, hdoc]
    return np.hstack(blocks)


def fit_lr(rows: pd.DataFrame, top: dict, c: float):
    from sklearn.linear_model import LogisticRegression
    x = features(rows, top)
    y = rows["owner_rank"].to_numpy()
    model = LogisticRegression(C=c, max_iter=2000)
    model.fit(x, y)
    prior = np.bincount(y, minlength=10) / len(y)
    return model, prior


def choose_c(rows: pd.DataFrame, top: dict) -> tuple[float, dict]:
    from sklearn.linear_model import LogisticRegression
    x, y = features(rows, top), rows["owner_rank"].to_numpy()
    players = rows["owner"].astype(str).to_numpy()
    uniq = np.array(sorted(set(players)))
    fold = pd.Series(np.random.default_rng(SEED).integers(5, size=len(uniq)), index=uniq).loc[players].to_numpy()
    loss = {}
    for c in CS:
        total = 0.0
        for f in range(5):
            m = LogisticRegression(C=c, max_iter=2000).fit(x[fold != f], y[fold != f])
            p = m.predict_proba(x[fold == f])
            total += -np.log(np.clip(p[np.arange((fold == f).sum()), y[fold == f]], 1e-12, 1)).sum()
        loss[str(c)] = total / len(y)
    prior = np.bincount(y, minlength=10) / len(y)
    loss["prior_only"] = float(-np.log(prior[y]).mean())
    best = min(CS, key=lambda c: loss[str(c)])
    return best, loss


def learn_ll(model, prior, rows: pd.DataFrame, top: dict) -> np.ndarray:
    p = model.predict_proba(features(rows, top))
    return np.log(np.clip(p, 1e-12, 1)) - np.log(prior)


def local(t: pd.DataFrame, top: dict) -> dict:
    s_all = t[t["set"] == "s_opp"]
    u = t[t["set"] == "u_opp"]
    key = u["game_index"].to_numpy()
    c, cv = choose_c(s_all, top)
    print("C", c, json.dumps(cv), flush=True)
    model, prior = fit_lr(s_all, top, c)
    prob_big = rf._delta_prob(s_all)
    evidence = {"LEARN": pd.DataFrame(learn_ll(model, prior, u, top), index=key)}
    for tau in (3.5, *SOFT_TAUS):
        evidence[f"SOFT{tau:g}"] = pd.DataFrame(rf.soft_ll(prob_big, u, top, tau), index=key)
    cells = {BASE: ("SOFT3.5", 0.35)} | {f"LEARN_{a:g}": ("LEARN", a) for a in ALPHAS} | \
            {f"SOFT{tau:g}_{a:g}": (f"SOFT{tau:g}", a) for tau in SOFT_TAUS for a in SOFT_ALPHAS}
    fit, eval_games, x_fit, x_eval = prod.train_data()
    o, (e,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.hstack([x_fit, o]), np.hstack([x_eval, e])
    pri = more.player_prior(fit[fit["role"] == "S"])
    sets = prod.question_sets()
    qll = {name: {n: fu.question_ll(q["games"], ev) for n, q in sets.items()} for name, ev in evidence.items()}
    probs, e1 = {}, {}
    for seed in prod.SEEDS:
        m = prod._fit(fit, xf, pri, seed)
        for n, q in sets.items():
            probs[seed, n] = m.probabilities(xe, eval_games["game_index"].to_numpy(), q)
        q = sets["E1"]
        k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
        ref = rf.cell_scores(probs[seed, "E1"], qll["SOFT3.5"]["E1"], 0.35, truth)
        e1[seed] = {}
        for cell, (ev, a) in cells.items():
            s = rf.cell_scores(probs[seed, "E1"], qll[ev]["E1"], a, truth)
            e1[seed][cell] = {**{str(b): float(s[k == b].mean() - ref[k == b].mean()) for b in K_BUCKETS},
                              "equal_k": hf.equal_k(s, k) - hf.equal_k(ref, k)}
        print(seed, json.dumps({c_: round(v["equal_k"], 4) for c_, v in e1[seed].items()}), flush=True)
    mean_e1 = {c_: {kk: round(float(np.mean([e1[s][c_][kk] for s in prod.SEEDS])), 4)
                    for kk in [str(b) for b in K_BUCKETS] + ["equal_k"]} for c_ in cells}
    chosen = max([c_ for c_ in cells if c_ != BASE], key=lambda c_: mean_e1[c_]["equal_k"])
    q = sets["E2"]
    k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
    ev, a = cells[chosen]
    e2 = {}
    for seed in prod.SEEDS:
        ref = rf.cell_scores(probs[seed, "E2"], qll["SOFT3.5"]["E2"], 0.35, truth)
        s = rf.cell_scores(probs[seed, "E2"], qll[ev]["E2"], a, truth)
        e2[seed] = {**{str(b): float(s[k == b].mean() - ref[k == b].mean()) for b in K_BUCKETS},
                    "equal_k": hf.equal_k(s, k) - hf.equal_k(ref, k)}
    mean_e2 = {kk: round(float(np.mean([e2[s][kk] for s in prod.SEEDS])), 4)
               for kk in [str(b) for b in K_BUCKETS] + ["equal_k"]}
    return {"C": c, "cv_logloss": cv, "mean_E1_vs_base": mean_e1, "chosen": chosen, "chosen_E2_vs_base": mean_e2,
            "E1_per_seed_chosen": {str(s): round(e1[s][chosen]["equal_k"], 4) for s in prod.SEEDS},
            "E2_per_seed_chosen": {str(s): round(e2[s]["equal_k"], 4) for s in prod.SEEDS},
            "pass": bool(mean_e1[chosen]["equal_k"] > 0 and mean_e2["equal_k"] > 0)}


def export(t: pd.DataFrame, top: dict, cell: str, c: float, job: str) -> dict:
    name, alpha = cell.split("_")
    pool = t[t["set"].isin(["s_opp", "u_opp"])]
    test_index = t.index[t["set"] == "test"].to_numpy()
    test_rows = t.loc[t["set"] == "test"].set_index(["part", "store_row"])
    if name == "LEARN":
        model, prior = fit_lr(pool, top, c)
        per = lambda rows: learn_ll(model, prior, rows, top)
    else:
        prob = rf._delta_prob(pool)
        tau = float(name[4:])
        per = lambda rows: rf.soft_ll(prob, rows, top, tau)

    def evidence(games, sub, _table, key):
        pos = test_rows.index.get_indexer(pd.MultiIndex.from_arrays([sub["part"], sub["store_row"]]))
        rows = sub.assign(_row=test_index[pos]).set_index("_row")
        return fu.question_ll(games, pd.DataFrame(per(rows), index=sub[key].to_numpy()))

    return fu.export(t.copy(), f"DELTA-ALL_{alpha}", job, evidence=evidence, tag="opp3-l10")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    parser.add_argument("--force-cell", help="skip the local round and export this cell")
    parser.add_argument("--c", type=float, default=None)
    args = parser.parse_args()
    t, top = rf.full_table()
    if args.force_cell:
        report = {"local": None, "chosen": args.force_cell}
        c = args.c if args.c is not None else choose_c(t[t["set"] == "s_opp"], top)[0]
        report["export"] = export(t, top, args.force_cell, c, args.job)
    else:
        loc = local(t, top)
        report = {"local": loc}
        print(json.dumps(loc, indent=1), flush=True)
        if loc["pass"]:
            report["export"] = export(t, top, loc["chosen"], loc["C"], args.job)
        else:
            print("no cell beats the base on both E1 and E2: no file", flush=True)
    (OUT / f"learn_{args.job}.json").write_text(json.dumps(report, indent=1, default=int))


if __name__ == "__main__":
    main()
