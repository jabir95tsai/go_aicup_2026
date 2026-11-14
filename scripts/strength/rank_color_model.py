"""Colour-specific B2 (RESULTS 2026-11-13).

Error split of the base on E2: outside A-3D, black questions score ~0.05 below white ones (D-B 0.786
vs 0.843, 4D-6D 0.800 vs 0.843) and regress toward the middle; on the fitting pool the white-minus-
black offset of the opponent block runs from +0.2 ranks at D to -0.15 at 6D (HD0 games), i.e. the
feature-to-rank map differs by colour. A question holds one colour, the base fits one map on mixed
questions.

Arms (cvec2 base features and prior; Valid-U E1 / E2, five seeds):
  mixed      the base
  sep        one B2 per colour, each on that colour's games only (players with >= k such games)
  inter      mixed draws, per game x plus s * x[:, :50] with s = +1 white / -1 black (102 columns):
             a pure-colour question gets its own linear map, all players stay in the fit
  inter_opp  the same with s * opponent block only (77 columns)
Decision on E2 (user 11/13): five-seed mean Equal-k >= +0.005, no k bucket clearly worse, player
bootstrap reported; E1 as a stress check.

    python -m scripts.strength.rank_color_model [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs" / "rank" / "humansl" / "color_model"
ARMS = ("mixed", "sep", "inter", "inter_opp")
OPP = np.r_[15:30, 40:50]


def transform(arm: str, x: np.ndarray, color: np.ndarray) -> np.ndarray:
    sign = np.where(color == 1, 1.0, -1.0)[:, None]
    if arm == "inter":
        return np.hstack([x, sign * x[:, :50]])
    if arm == "inter_opp":
        return np.hstack([x, sign * x[:, OPP]])
    return x


def task(seed, arm, fit, x_fit, prior, eval_games, x_eval, sets):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    hs.SEED = seed
    fit_color, eval_color = fit["player_color"].to_numpy(), eval_games["player_color"].to_numpy()
    index = eval_games["game_index"].to_numpy()
    question_color = {n: eval_games.set_index("game_index")["player_color"].loc[[g[0] for g in q["games"]]].to_numpy()
                      for n, q in sets.items()}
    out = {n: np.zeros(len(q)) for n, q in sets.items()}
    if arm == "sep":
        for color in (0, 1):
            rows = fit_color == color
            model = more.PriorMatched(prior).fit(fit[rows].reset_index(drop=True), pd.DataFrame(), x_fit[rows])
            for n, q in sets.items():
                m = question_color[n] == color
                p = model.probabilities(x_eval, index, q[m].reset_index(drop=True))
                out[n][m] = official_scores(score_rank(p), q["rank_id"].to_numpy()[m])
        return out
    xf, xe = transform(arm, x_fit, fit_color), transform(arm, x_eval, eval_color)
    model = more.PriorMatched(prior).fit(fit, pd.DataFrame(), xf)
    for n, q in sets.items():
        out[n] = official_scores(score_rank(model.probabilities(xe, index, q)), q["rank_id"].to_numpy())
    return out


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
    color_of = eval_games.set_index("game_index")["player_color"]
    result = {"mean": {}, "seeds": {}, "mean_by_k": {}, "by_band_color": {}}
    for name, q in sets.items():
        k, rank = q["k"].to_numpy(), q["rank_id"].to_numpy()
        color = color_of.loc[[g[0] for g in q["games"]]].to_numpy()
        band = np.where(rank < 3, "D-B", np.where(rank <= 6, "A-3D", "4D-6D"))
        table = {a: {s: done[s, a][name] for s in hf.SEEDS} for a in ARMS}
        result["mean"][name] = {a: round(float(np.mean([hf.equal_k(v, k) for v in table[a].values()])), 4) for a in ARMS}
        result["mean_by_k"][name], result["seeds"][name] = {}, {}
        result["by_band_color"][name] = {
            a: {f"{b} {'black' if c == 0 else 'white'}": round(float(np.mean([v[(band == b) & (color == c)].mean()
                                                                               for v in table[a].values()])), 4)
                for b in ("D-B", "A-3D", "4D-6D") for c in (0, 1)} for a in ARMS}
        for a in ARMS[1:]:
            diff = [hf.equal_k(table[a][s], k) - hf.equal_k(table["mixed"][s], k) for s in hf.SEEDS]
            result["mean"][name][f"{a}-mixed"] = round(float(np.mean(diff)), 4)
            result["seeds"][name][f"{a}-mixed"] = [round(d, 4) for d in diff]
            result["mean_by_k"][name][f"{a}-mixed"] = {
                str(b): round(float(np.mean([table[a][s][k == b].mean() - table["mixed"][s][k == b].mean()
                                             for s in hf.SEEDS])), 4) for b in K_BUCKETS}
            result["mean"][name][f"{a}-mixed bootstrap_95"] = hf.bootstrap(table["mixed"][hf.SEED], table[a][hf.SEED], q)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
