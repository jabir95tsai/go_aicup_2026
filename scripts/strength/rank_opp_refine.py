"""Opponent identity link v2, refinement round (2026-11-23, user "做 1–4"; RESULTS.md, registered before results)

Current = rank_opp_fuse DELTA-ALL, alpha 0.25, table on the 20,000 s_opp games (843411, E1 +0.0198 / E2 +0.0135).
845534 (another session, same day) already showed alpha < 0.25 and one top-10 form are worse, so this round tests:
  BIG    the same DELTA table on every Valid-S HD<=1 game (s_opp + 42,867 extra, scores_extra.npz), alpha in
         ALPHAS (0.25 upward)
  SOFT   top-10 soft link: weights softmax((f_i - f_1) / tau) over the top-10 candidates, per-game likelihood
         log sum_i w_i P_delta(r_j(i) | r, bin, hd, oc) (BIG table), tau in TAUS
Selection on E1 only (five-seed mean Equal-k over current). E2 stays sealed except for the chosen cell and current.
Adopt (export `<job>-opp2-l10.csv`) if chosen - current >= +0.003 on E1 and >= +0.002 on E2.

    python -m scripts.strength.rank_opp_refine --job <jobid>
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
import scripts.strength.rank_prod_a0_fusion as prod
import scripts.strength.rank_qmerge as qm

OUT = fu.OUT
ALPHAS = (0.25, 0.3, 0.35, 0.4)
TAUS = (3.5, 7.0)
SOFT_ALPHAS = (0.25, 0.35)
E1_GAIN, E2_GAIN = 0.003, 0.002


def full_table() -> tuple[pd.DataFrame, dict]:
    """link table of the main queries plus the extra Valid-S games, same arm and bin edges; top-10 kept."""
    t = fu.link_table()
    arm, edges = t.attrs["arm"], t.attrs["edges"]
    z = np.load(OUT / "scores.npz")
    top = {"idx": z[f"{arm}_idx"], "val": z[f"{arm}_val"]}
    q = pd.read_parquet(OUT / "queries_extra.parquet")
    ze = np.load(OUT / "scores_extra.npz")
    val, idx = ze[f"{arm}_val"], ze[f"{arm}_idx"]
    if not np.isfinite(val[:, 1]).all():
        raise ValueError("extra scores incomplete")
    gap = val[:, 0] - val[:, 1]
    colour = q["qcolor"].to_numpy()
    b = np.zeros(len(q), np.int64)
    for c in (0, 1):
        m = colour == c
        b[m] = np.searchsorted(np.array(edges[c]), gap[m], side="right")
    extra = q.assign(bin=b, rj=z["cand_rank"][idx[:, 0]], gap=gap, oc=(1 - colour).astype(np.int64))
    out = pd.concat([t, extra], ignore_index=True)
    out.attrs.update(t.attrs)
    top = {"idx": np.vstack([top["idx"], idx]), "val": np.vstack([top["val"], val]), "cand_rank": z["cand_rank"]}
    return out, top


def _delta_prob(t: pd.DataFrame, *, smooth: bool = False) -> np.ndarray:
    """P(r_j | r, bin, hd, oc) as in fit_delta, as probabilities [2, 2, r, bin, rj]."""
    c = np.zeros((2, 2, fu.N_BIN, 19))
    np.add.at(c, (t["hd"].to_numpy(), t["oc"].to_numpy(), t["bin"].to_numpy(),
                  t["rj"].to_numpy() - t["owner_rank"].to_numpy() + 9), 1.0)
    if smooth:
        nearby = np.full_like(c, 0.5)
        nearby[:, :, 1:] += c[:, :, :-1]
        nearby[:, :, :-1] += c[:, :, 1:]
        c += (0.5 * c.shape[-1]) * nearby / nearby.sum(axis=-1, keepdims=True)
    else:
        c += 0.5
    r, rj = np.arange(10)[:, None], np.arange(10)[None, :]
    p = c[..., rj - r + 9]
    p = p / p.sum(-1, keepdims=True)
    return np.moveaxis(p, 3, 2)


def soft_ll(prob: np.ndarray, rows: pd.DataFrame, top: dict, tau: float) -> np.ndarray:
    """[n, 10] log sum_i w_i P(r_j(i) | r, bin, hd, oc) minus its mean over r."""
    i = rows.index.to_numpy()
    val, idx = top["val"][i].astype(np.float64), top["idx"][i]
    w = np.exp((val - val[:, :1]) / tau)
    w /= w.sum(1, keepdims=True)
    rj = top["cand_rank"][idx]                                              # [n, 10 candidates]
    hd, oc, b = rows["hd"].to_numpy(), rows["oc"].to_numpy(), rows["bin"].to_numpy()
    kern = prob[hd[:, None], oc[:, None], :, b[:, None], rj]               # [n, 10 cand, 10 r]
    like = (w[:, :, None] * kern).sum(1)
    ll = np.log(like)
    return ll - ll.mean(1, keepdims=True)


def cell_scores(p, ll, alpha, truth):
    return qm.official(score_rank(fu.posterior(p, ll, alpha)), truth)


def local(t: pd.DataFrame, top: dict) -> dict:
    fit, eval_games, x_fit, x_eval = prod.train_data()
    o, (e,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.hstack([x_fit, o]), np.hstack([x_eval, e])
    prior = more.player_prior(fit[fit["role"] == "S"])
    s20 = t[(t["set"] == "s_opp") & t.index.isin(np.flatnonzero((t["set"] == "s_opp").to_numpy())[:20000])]
    s_all = t[t["set"] == "s_opp"]
    if len(s20) != 20000 or len(s_all) != 62867:
        raise ValueError(f"unexpected s_opp sizes {len(s20)} / {len(s_all)}")
    u = t[t["set"] == "u_opp"]
    key = u["game_index"].to_numpy()
    prob_big = _delta_prob(s_all)
    evidence = {"CUR": pd.DataFrame(fu.game_ll(fu.fit_delta(s20), u, False), index=key),
                "BIG": pd.DataFrame(fu.game_ll(fu.fit_delta(s_all), u, False), index=key)}
    for tau in TAUS:
        evidence[f"SOFT{tau:g}"] = pd.DataFrame(soft_ll(prob_big, u, top, tau), index=key)
    cells = {"CUR_0.25": ("CUR", 0.25)} | {f"BIG_{a:g}": ("BIG", a) for a in ALPHAS} | \
            {f"SOFT{tau:g}_{a:g}": (f"SOFT{tau:g}", a) for tau in TAUS for a in SOFT_ALPHAS}
    sets = prod.question_sets()
    qll = {name: {n: fu.question_ll(q["games"], ev) for n, q in sets.items()} for name, ev in evidence.items()}
    probs = {}
    e1 = {}
    for seed in prod.SEEDS:
        model = prod._fit(fit, xf, prior, seed)
        for n, q in sets.items():
            probs[seed, n] = model.probabilities(xe, eval_games["game_index"].to_numpy(), q)
        q = sets["E1"]
        k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
        cur = cell_scores(probs[seed, "E1"], qll["CUR"]["E1"], 0.25, truth)
        e1[seed] = {c: {**{str(b): float(s[k == b].mean() - cur[k == b].mean()) for b in K_BUCKETS},
                        "equal_k": hf.equal_k(s, k) - hf.equal_k(cur, k)}
                    for c, (ev, a) in cells.items()
                    for s in [cell_scores(probs[seed, "E1"], qll[ev]["E1"], a, truth)]}
        print(seed, json.dumps({c: round(v["equal_k"], 4) for c, v in e1[seed].items()}), flush=True)
    mean_e1 = {c: {key_: round(float(np.mean([e1[s][c][key_] for s in prod.SEEDS])), 4)
                   for key_ in [str(b) for b in K_BUCKETS] + ["equal_k"]} for c in cells}
    candidates = [c for c in cells if c != "CUR_0.25"]
    chosen = max(candidates, key=lambda c: mean_e1[c]["equal_k"])
    # E2: unseal only the chosen cell (and current as its reference)
    q = sets["E2"]
    k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
    ev, a = cells[chosen]
    e2 = {}
    for seed in prod.SEEDS:
        cur = cell_scores(probs[seed, "E2"], qll["CUR"]["E2"], 0.25, truth)
        s = cell_scores(probs[seed, "E2"], qll[ev]["E2"], a, truth)
        base = qm.official(score_rank(probs[seed, "E2"]), truth)
        e2[seed] = {"vs_current": {**{str(b): float(s[k == b].mean() - cur[k == b].mean()) for b in K_BUCKETS},
                                   "equal_k": hf.equal_k(s, k) - hf.equal_k(cur, k)},
                    "current_vs_base": hf.equal_k(cur, k) - hf.equal_k(base, k)}
    mean_e2 = {key_: round(float(np.mean([e2[s]["vs_current"][key_] for s in prod.SEEDS])), 4)
               for key_ in [str(b) for b in K_BUCKETS] + ["equal_k"]}
    passed = mean_e1[chosen]["equal_k"] >= E1_GAIN and mean_e2["equal_k"] >= E2_GAIN
    return {"cells": list(cells), "mean_E1_vs_current": mean_e1, "chosen": chosen, "chosen_E2_vs_current": mean_e2,
            "E2_current_vs_base": round(float(np.mean([e2[s]["current_vs_base"] for s in prod.SEEDS])), 4),
            "E1_per_seed_chosen": {str(s): round(e1[s][chosen]["equal_k"], 4) for s in prod.SEEDS},
            "E2_per_seed_chosen": {str(s): round(e2[s]["vs_current"]["equal_k"], 4) for s in prod.SEEDS},
            "pass": bool(passed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    parser.add_argument("--force-cell", help="skip the local round and export this cell (LB probe at user request)")
    args = parser.parse_args()
    t, top = full_table()
    if args.force_cell:
        report = {"local": None}
        loc = {"pass": True, "chosen": args.force_cell}
    else:
        report = {"local": local(t, top)}
        print(json.dumps(report["local"], indent=1), flush=True)
        loc = report["local"]
    if loc["pass"]:
        name, alpha = loc["chosen"].split("_")
        t_export = t.copy()
        if name == "BIG":
            report["export"] = fu.export(t_export, f"DELTA-ALL_{alpha}", args.job, tag="opp2-l10")
        else:
            tau = float(name[4:])
            prob = _delta_prob(t_export[t_export["set"].isin(["s_opp", "u_opp"])])

            def evidence(games, sub, _table, key):
                rows = t_export.loc[t_export["set"] == "test"].set_index(["part", "store_row"])
                idx = rows.index.get_indexer(pd.MultiIndex.from_arrays([sub["part"], sub["store_row"]]))
                srows = sub.assign(_row=t_export.index[t_export["set"] == "test"].to_numpy()[idx]).set_index("_row")
                per_game = pd.DataFrame(soft_ll(prob, srows, top, tau), index=sub[key].to_numpy())
                return fu.question_ll(games, per_game)

            report["export"] = fu.export(t_export, f"DELTA-ALL_{alpha}", args.job, evidence=evidence, tag="opp2-l10")
    else:
        print("refinement not adopted: keep 843411-opp-l10.csv", flush=True)
    (OUT / f"refine_{args.job}.json").write_text(json.dumps(report, indent=1, default=int))


if __name__ == "__main__":
    main()
