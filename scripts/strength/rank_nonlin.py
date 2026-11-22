"""Competition 2: extend the nonlinear per-game feature of 839169 (2026-11-22, user "好，做非線性延伸"; RESULTS.md)

Base (839169-ygbm-l10): hc 52 + yH = HistGBR regressor on H 30 -> rank_id, (own, swapped) = 2 columns.
  yHC  replace yH by the same regressor on H 30 + C v2 20 (own / opponent swapped together) -> 2 columns
  cHC  replace yH by a HistGBR classifier on H + C -> 10 class log-probabilities (own side)
Every new column is player-grouped 5-fold out-of-fold on the pool it is fitted on (local: Valid-S + train20 HD<=1;
export: + Valid-U HD<=1); eval / test games use the model fitted on the whole pool. Local: Valid-U E1 / E2, B2 seeds
2027-2031, each arm minus base. An arm with both five-seed mean Equal-k > 0 gets an export (same recipe as 839169:
prior-matched seed 2027, 9 merge groups, L10; the base arm must reproduce 839169-ygbm-l10.csv).

    python -m scripts.strength.rank_nonlin --job <jobid>
    python -m scripts.strength.rank_nonlin --job <jobid> --export cHC   # export only (LB probe)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as more
import scripts.strength.rank_ogs as ro
import scripts.strength.rank_prod_a0_fusion as prod
import scripts.strength.rank_qmerge as qm
import scripts.strength.rank_stack as rs

OUT = ROOT / "outputs" / "rank" / "nonlin"
SEED = 2036
TAGS = {"yHC": "yhc-l10", "cHC": "chc-l10"}


def swap50(x: np.ndarray) -> np.ndarray:
    return np.hstack([x[:, 15:30], x[:, :15], x[:, 40:50], x[:, 30:40]])


def classifier():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=SEED)


TRANSFORMS = {
    "yH": (lambda: ro.regressor(), lambda x: x[:, :30], lambda m, x: np.column_stack([m.predict(x[:, :30]),
                                                                                    m.predict(ro.swap(x[:, :30]))])),
    "yHC": (lambda: ro.regressor(), lambda x: x[:, :50], lambda m, x: np.column_stack([m.predict(x[:, :50]),
                                                                                     m.predict(swap50(x[:, :50]))])),
    "cHC": (classifier, lambda x: x[:, :50], lambda m, x: np.log(np.clip(m.predict_proba(x[:, :50]), 1e-4, 1))),
}


def oof(name: str, pool: pd.DataFrame, x_pool: np.ndarray, others: list[np.ndarray], folds: int = 5):
    """(out-of-fold columns for the pool, columns of each array in `others` from the whole-pool model).
    yH uses rank_ogs.yugen_oof unchanged so the base arm reproduces 839169."""
    if name == "yH":
        o, _, full = ro.yugen_oof(pool, x_pool, x_pool[:1])
        return o, [ro.g_features(full, x[:, :30]) for x in others]
    make, inputs, feats = TRANSFORMS[name]
    players = pool["player_id"].astype(str).to_numpy()
    uniq = np.array(sorted(set(players)))
    fold_of = pd.Series(np.random.default_rng(SEED).integers(folds, size=len(uniq)), index=uniq).loc[players].to_numpy()
    y = pool["rank_id"].to_numpy()
    y = y.astype(float) if name.startswith("y") else y.astype(int)
    out = None
    for f in range(folds):
        m = make().fit(inputs(x_pool[fold_of != f]), y[fold_of != f])
        part = feats(m, x_pool[fold_of == f])
        if out is None:
            out = np.zeros((len(pool), part.shape[1]))
        out[fold_of == f] = part
    full = make().fit(inputs(x_pool), y)
    return out, [feats(full, x) for x in others]


def local(fit, eval_games, x_fit, x_eval) -> dict:
    cols = {}
    for name in ("yH", "yHC", "cHC"):
        o, (e,) = oof(name, fit, x_fit, [x_eval])
        cols[name] = (o, e)
        print(f"{name}: oof corr with rank {np.corrcoef(o[:, 0] if name != 'cHC' else (np.exp(o) * np.arange(10)).sum(1), fit['rank_id'])[0, 1]:.3f}", flush=True)
    arms = {"base": "yH", "yHC": "yHC", "cHC": "cHC"}
    xf = {a: np.hstack([x_fit, cols[n][0]]) for a, n in arms.items()}
    xe = {a: np.hstack([x_eval, cols[n][1]]) for a, n in arms.items()}
    sets = prod.question_sets()
    prior = more.player_prior(fit[fit["role"] == "S"])
    seeds = {}
    for seed in prod.SEEDS:
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {a: hf.predict(prod._fit(fit, xf[a], prior, seed), eval_games, xe[a], q) for a in arms}
            entry[name] = {a: {**{str(b): round(float(scores[a][k == b].mean() - scores["base"][k == b].mean()), 4)
                                  for b in K_BUCKETS},
                               "equal_k": round(hf.equal_k(scores[a], k) - hf.equal_k(scores["base"], k), 4)}
                           for a in TAGS}
            entry[name]["base_equal_k"] = round(hf.equal_k(scores["base"], k), 4)
        seeds[str(seed)] = entry
        print(seed, json.dumps({n: {a: e[a]["equal_k"] for a in TAGS} for n, e in entry.items()}), flush=True)
    mean = {a: {n: {key: round(float(np.mean([seeds[str(s)][n][a][key] for s in prod.SEEDS])), 4)
                    for key in [str(b) for b in K_BUCKETS] + ["equal_k"]} for n in sets} for a in TAGS}
    return {"seeds": seeds, "mean_vs_base": mean}


def export(fit, eval_games, x_fit, x_eval, arms: list[str], job: str) -> dict:
    u = (eval_games["handicap"] <= 1).to_numpy()
    pool = pd.concat([fit, eval_games[u]], ignore_index=True)
    x_pool = np.vstack([x_fit, x_eval[u]])
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    xs, gis, qs, shift = [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        xs.append(x), gis.append(gi + shift), qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True)
    groups = qm.groups_from_pairs(pd.read_csv(qm.PAIRS))
    by_id = questions.set_index("question_id")
    l10 = pd.read_csv(qm.L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    current = pd.read_csv(rs.CURRENT, dtype=str).set_index("question_id")["pred_rank"]
    report = {}
    for arm in ["yH"] + arms:
        o, (t,) = oof(arm, pool, x_pool, [x])
        model = prod._fit(pool, np.hstack([x_pool, o]), prior, 2027)
        pred = rs.predict(model, np.hstack([x, t]), gi, questions, groups, by_id, l10).loc[current.index]
        diff = pred != current
        if arm == "yH":
            if diff.any():
                raise ValueError(f"base arm does not reproduce 839169-ygbm-l10.csv ({int(diff.sum())})")
            continue
        report[arm] = {"public": int((diff & diff.index.str.startswith("pub")).sum()),
                       "private": int((diff & diff.index.str.startswith("priv")).sum()),
                       "file": write_rank_csv(pred.rename("pred_rank").reset_index(), TAGS[arm], job).name}
        print(arm, json.dumps(report[arm]), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    parser.add_argument("--export", nargs="+", choices=list(TAGS),
                        help="skip the local check and export these arms (LB probe at user request)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fit, eval_games, x_fit, x_eval = prod.train_data()
    if args.export:
        report = {"export": export(fit, eval_games, x_fit, x_eval, args.export, args.job)}
        (OUT / f"export_{args.job}.json").write_text(json.dumps(report, indent=1))
        return
    report = {"local": local(fit, eval_games, x_fit, x_eval)}
    print(json.dumps(report["local"]["mean_vs_base"], indent=1), flush=True)
    m = report["local"]["mean_vs_base"]
    passing = [a for a in TAGS if m[a]["E1"]["equal_k"] > 0 and m[a]["E2"]["equal_k"] > 0]
    report["passing"] = passing
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))
    if passing:
        report["export"] = export(fit, eval_games, x_fit, x_eval, passing, args.job)
    else:
        print("no arm improves both E1 and E2: no file", flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
