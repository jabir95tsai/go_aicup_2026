"""Komi downstream test (RESULTS A-043), CPU only.

B2 fitted on S-fit and scored on the 1,000 S-select questions, once with the
komi-6.5 features (outputs/humansl_q1000, outputs/humansl_qsel) and once with
komi-0 features of the same games (outputs/humansl_komi0/{fit,sel}). A-024
features are shared. Rule: komi 0 is adopted if its S-select Equal-k beats
komi 6.5 by >= 0.005. The player-grouped bootstrap interval is informative only.

    python scripts/humansl_komi_downstream.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from humansl_aggregation import load, part, score
from humansl_questions import by_k, equal_k
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import SEED

MIN_GAIN, BOOTSTRAP = 0.005, 2000
OUT = ROOT / "outputs"


def main() -> int:
    runs = {"6.5": (part(load(OUT / "humansl_q1000"), "S-fit"), load(OUT / "humansl_qsel")),
            "0": (load(OUT / "humansl_komi0" / "fit"), load(OUT / "humansl_komi0" / "sel"))}
    for komi, (fit, select) in runs.items():
        if not fit[0]["game_index"].equals(runs["6.5"][0][0]["game_index"]) or \
                not select[0]["game_index"].equals(runs["6.5"][1][0]["game_index"]):
            raise ValueError(f"komi {komi}: games differ from the komi-6.5 sets")
    questions = runs["6.5"][1][1]
    k = questions["k"].to_numpy()
    scores = {komi: score("B2", fit, select) for komi, (fit, select) in runs.items()}
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(scores["0"][rows], k[rows]) - equal_k(scores["6.5"][rows], k[rows]))
    gain = equal_k(scores["0"], k) - equal_k(scores["6.5"], k)
    result = {"S-select": {komi: by_k(value, k) for komi, value in scores.items()},
              "gain_komi0": round(gain, 4), "bootstrap_95": [round(float(v), 4) for v in np.quantile(draws, [0.025, 0.975])],
              "min_gain": MIN_GAIN, "adopt_komi": "0" if gain >= MIN_GAIN else "6.5"}
    print(json.dumps(result, indent=1))
    (OUT / "humansl_komi0" / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
