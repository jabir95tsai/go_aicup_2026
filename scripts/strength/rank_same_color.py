"""B2 synthetic questions drawn from one colour, as the test questions are (RESULTS 2026-11-13).

PriorMatched draws the k games of a synthetic question from all games of a player, black and white
mixed; the official questions (and E1 / E2) hold games of one colour. With mixed draws both HD1-share
flags are non-zero in most fitting questions, while one of them is always zero at test time.

Arms (same features, prior, number of synthetic questions; cvec2 base, fit Valid-S + train20 HD<=1):
  mixed       the base: players with >= k games, k of them at random
  same        (player, colour) groups with >= k games: rank by the prior, a player of that rank with
              such a group, one of the player's eligible colours, k of its games
  mixed_elig  mixed draws restricted to the players eligible for `same` (separates the smaller
              player set from the colour effect)
Reported: eligible players per k and rank, Valid-U E1 / E2 over five seeds, player bootstrap (seed
2027) of each arm against mixed. Decision on E2 (user 11/13): five-seed mean Equal-k >= +0.005, no k
bucket clearly worse, bootstrap interval reported.

    python -m scripts.strength.rank_same_color [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT

OUT = ROOT / "outputs" / "rank" / "humansl" / "same_color"
ARMS = ("mixed", "same", "mixed_elig")


def make_model(arm: str, prior: dict):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    class Sampled(more.PriorMatched):
        def _fit_synthetic(self, games, x, y) -> None:
            rng = np.random.default_rng(hs.SEED)
            player = games["player_id"].astype(str).to_numpy()
            color = games["player_color"].to_numpy()
            by_player = {p: rows for p, rows in pd.Series(np.arange(len(games))).groupby(player).apply(np.asarray).items()}
            self.per_k, self.eligible = {}, {}
            for k in K_BUCKETS:
                groups = {p: [rows[color[rows] == c] for c in (0, 1) if (color[rows] == c).sum() >= k]
                          for p, rows in by_player.items()}
                if arm == "mixed":
                    pool = [p for p, rows in by_player.items() if len(rows) >= k]
                else:
                    pool = [p for p, g in groups.items() if g]
                by_rank = [[p for p in pool if y[by_player[p][0]] == r] for r in range(10)]
                self.eligible[k] = [len(b) for b in by_rank]
                ranks = rng.choice(10, hs.SYNTHETIC, p=self.prior[k])
                features = []
                for r in ranks:
                    p = by_rank[r][rng.integers(len(by_rank[r]))]
                    rows = groups[p][rng.integers(len(groups[p]))] if arm == "same" else by_player[p]
                    features.append(x[rng.choice(rows, k, replace=False)].mean(axis=0))
                features = np.stack(features)
                scaler = StandardScaler().fit(features)
                model = LogisticRegression(C=1.0, max_iter=5000)
                self.per_k[k] = (scaler, model.fit(scaler.transform(features), ranks))

    return Sampled(prior)


def task(seed, arm, fit, x_fit, prior, eval_games, x_eval, sets):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.humansl_hd_filter as hf
    hs.SEED = seed
    model = make_model(arm, prior).fit(fit, pd.DataFrame(), x_fit)
    return {n: hf.predict(model, eval_games, x_eval, q) for n, q in sets.items()}, model.eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_more_players as more

    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    sets = {n: q.reset_index(drop=True) for n, q in hf.question_sets(games, rebuild=False).items() if n in ("E1", "E2")}
    tasks = [(seed, arm) for seed in hf.SEEDS for arm in ARMS]
    done = dict(zip(tasks, Parallel(n_jobs=args.jobs)(
        delayed(task)(s, a, fit, x_fit, prior, eval_games, x_eval, sets) for s, a in tasks)))
    result = {"fit_games": int(len(fit)), "eligible_players": {a: done[hf.SEEDS[0], a][1] for a in ARMS},
              "seeds": {}, "mean": {}, "mean_by_k": {}, "bootstrap_seed": hf.SEED}
    for name, q in sets.items():
        k = q["k"].to_numpy()
        table = {a: {s: done[s, a][0][name] for s in hf.SEEDS} for a in ARMS}
        result["mean"][name] = {a: round(float(np.mean([hf.equal_k(v, k) for v in table[a].values()])), 4) for a in ARMS}
        result["seeds"][name] = {a: [round(hf.equal_k(v, k), 4) for v in table[a].values()] for a in ARMS}
        result["mean_by_k"][name] = {a: {str(b): round(float(np.mean([v[k == b].mean() for v in table[a].values()])), 4)
                                         for b in K_BUCKETS} for a in ARMS}
        for a in ARMS[1:]:
            diff = [hf.equal_k(table[a][s], k) - hf.equal_k(table["mixed"][s], k) for s in hf.SEEDS]
            result["mean"][name][f"{a}-mixed"] = round(float(np.mean(diff)), 4)
            result["seeds"][name][f"{a}-mixed"] = [round(d, 4) for d in diff]
            result["mean_by_k"][name][f"{a}-mixed"] = {
                str(b): round(result["mean_by_k"][name][a][str(b)] - result["mean_by_k"][name]["mixed"][str(b)], 4)
                for b in K_BUCKETS}
            result["mean"][name][f"{a}-mixed bootstrap_95"] = hf.bootstrap(table["mixed"][hf.SEED], table[a][hf.SEED], q)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
