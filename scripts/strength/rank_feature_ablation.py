"""H / C ablation under the current rank base (RESULTS 2026-11-11).

Base = noA-t20-prior: B2 on HD<=1 games of Valid-S + train20, x = [H 30 | C 20 | HD1 flags 2],
synthetic questions drawn with the Valid-S rank prior (rank_more_players.local). Arms drop one
feature block, everything else fixed; Valid-U E1 / E2, five seeds, diff vs H+C.

    python -m scripts.strength.rank_feature_ablation
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as mp

OUT = mp.OUT / "feature_ablation.json"
COLUMNS = {"hc": np.arange(52), "c": np.arange(30, 52), "h": np.r_[0:30, 50:52]}


def main() -> None:
    games, fit, eval_games, x_fit, x_eval, _ = mp.data(with_ex=False)
    assert x_fit.shape[1] == 52 and x_eval.shape[1] == 52, (x_fit.shape, x_eval.shape)
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    prior = mp.player_prior(fit[valid_s])
    fit_rows = fit[hd01].reset_index(drop=True)
    result = {"fit_games": int(hd01.sum()), "columns": {a: len(c) for a, c in COLUMNS.items()}, "seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        models = {a: mp.PriorMatched(prior).fit(fit_rows, pd.DataFrame(), x_fit[hd01][:, c]) for a, c in COLUMNS.items()}
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {a: hf.predict(m, eval_games, x_eval[:, COLUMNS[a]], q) for a, m in models.items()}
            cell = {a: hf.by_k(s, k) for a, s in scores.items()}
            for a in ("c", "h"):
                cell[f"{a}-hc"] = {"by_k": {str(b): round(float(scores[a][k == b].mean() - scores["hc"][k == b].mean()), 4)
                                            for b in K_BUCKETS},
                                   "equal_k": round(hf.equal_k(scores[a], k) - hf.equal_k(scores["hc"], k), 4)}
                if seed == hf.SEED:
                    cell[f"{a}-hc"]["bootstrap_95"] = hf.bootstrap(scores["hc"], scores[a], q)
            entry[name] = cell
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {d: e[d]["equal_k"] for d in ("c-hc", "h-hc")} | {"hc": e["hc"]["equal_k"]}
                                for n, e in entry.items()}), flush=True)
    result["mean"] = {n: {d: round(float(np.mean([result["seeds"][str(s)][n][d]["equal_k"] for s in hf.SEEDS])), 4)
                          for d in ("c-hc", "h-hc")} for n in sets}
    result["mean_by_k"] = {n: {d: {str(b): round(float(np.mean([result["seeds"][str(s)][n][d]["by_k"][str(b)]
                                                                 for s in hf.SEEDS])), 4) for b in K_BUCKETS}
                               for d in ("c-hc", "h-hc")} for n in sets}
    print(json.dumps({"mean": result["mean"], "mean_by_k": result["mean_by_k"]}))
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
