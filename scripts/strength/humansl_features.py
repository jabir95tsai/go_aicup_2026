"""HumanSL per-game feature sets (RESULTS A-046), CPU only; no new extraction.

Feature sets from features/humansl.shard_features, HumanSL only (no A-024):
  F0 mean (current H), F1 phase, F2 vote, F3 full.
B2 fitted on the fitting games, scored on the 1,000 S-select questions.
Pre-registered: the best set replaces F0 if it beats F0 on S-select by >= 0.005.
Valid-U is not used.

    python -m scripts.strength.humansl_features                # fit = q1000 S-fit (A-041 setting)
    python -m scripts.strength.humansl_features --with-rest    # + A-044 S-fit and S-extra games
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.strength.humansl_questions import by_k
from go_aicup.features.humansl import run_features
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs"
SETS = {"F0": "mean", "F1": "phase", "F2": "vote", "F3": "full"}
MIN_GAIN = 0.005


def games_with(run, feature_set, roles):
    games = pd.read_parquet(run / "games.parquet")
    games = games[games["role"].isin(roles)]
    table = games.merge(run_features(run, feature_set), on="game_index", validate="one_to_one")
    if len(table) != len(games):
        raise ValueError(f"{run}: features missing")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-rest", action="store_true")
    args = parser.parse_args()
    questions = pd.read_parquet(OUT / "humansl_qsel" / "questions.parquet")
    k = questions["k"].to_numpy()
    result = {"fit": "q1000 S-fit" + (" + vs_rest S-fit, S-extra" if args.with_rest else ""), "S-select": {}}
    for label, feature_set in SETS.items():
        fit = [games_with(OUT / "humansl_q1000", feature_set, ["S-fit"])]
        if args.with_rest:
            fit.append(games_with(OUT / "humansl_vs_rest", feature_set, ["S-fit", "S-extra"]))
        fit = pd.concat(fit, ignore_index=True)
        test = games_with(OUT / "humansl_qsel", feature_set, ["S-select"])
        columns = [c for c in fit.columns if c.startswith(("own_", "opp_"))]
        model = RankModel("B2").fit(fit, pd.DataFrame(), fit[columns].to_numpy())
        pred = score_rank(model.probabilities(test[columns].to_numpy(), test["game_index"].to_numpy(), questions))
        result["S-select"][label] = {"features": len(columns), **by_k(official_scores(pred, questions["rank_id"].to_numpy()), k)}
        print(label, result["S-select"][label], flush=True)
    scores = {label: value["equal_k"] for label, value in result["S-select"].items()}
    best = max(scores, key=scores.get)
    result.update({"best": best, "adopted": best if scores[best] - scores["F0"] >= MIN_GAIN else "F0",
                   "fit_games": int(len(fit)), "fit_players": int(fit["player_id"].nunique())})
    print(json.dumps(result, indent=1))
    name = "humansl_features_rest.json" if args.with_rest else "humansl_features.json"
    (OUT / name).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
