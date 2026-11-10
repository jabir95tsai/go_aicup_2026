"""Strong-engine feature pilot (RESULTS A-048), CPU only.

B2 fitted on the q1000 S-fit games, scored on the 1,000 S-select questions:
  AH    A-024 + HumanSL mean (A-041/A-044 features; reference)
  AH_E  AH + engine move quality (14)
  H_E   HumanSL + engine (diagnostic)
  E     engine only (diagnostic)
Pre-registered gate: AH_E - AH >= +0.01 -> extend engine features to all fitting
data and Valid-U.

    python -m scripts.strength.engine_pilot
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.strength.humansl_aggregation import load, part
from scripts.strength.humansl_questions import by_k
from go_aicup.features.katago_engine import run_features as engine_features
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs"
MIN_GAIN = 0.01


def with_engine(data, engine: pd.DataFrame):
    games, questions, x = data
    table = games[["game_index"]].merge(engine, on="game_index", how="left", validate="one_to_one")
    if table.isna().any().any():
        raise ValueError("engine features missing for some games")
    return games, questions, x, table.drop(columns="game_index").to_numpy()


def main() -> int:
    engine = engine_features(OUT / "engine_pilot")
    fit = with_engine(part(load(OUT / "humansl_q1000"), "S-fit"), engine)
    test = with_engine(load(OUT / "humansl_qsel"), engine)
    k = test[1]["k"].to_numpy()
    views = {"AH": lambda d: d[2], "AH_E": lambda d: np.hstack([d[2], d[3]]),
             "H_E": lambda d: np.hstack([d[2][:, 10:], d[3]]), "E": lambda d: d[3]}
    result = {"engine_features": [c for c in engine.columns if c != "game_index"], "S-select": {}}
    for name, view in views.items():
        model = RankModel("B2").fit(fit[0], fit[1], view(fit))
        pred = score_rank(model.probabilities(view(test), test[0]["game_index"].to_numpy(), test[1]))
        result["S-select"][name] = by_k(official_scores(pred, test[1]["rank_id"].to_numpy()), k)
        print(name, result["S-select"][name], flush=True)
    gain = result["S-select"]["AH_E"]["equal_k"] - result["S-select"]["AH"]["equal_k"]
    result["gate"] = {"gain_AH_E_minus_AH": round(gain, 4), "min_gain": MIN_GAIN, "pass": bool(gain >= MIN_GAIN)}
    print(json.dumps(result["gate"]))
    (OUT / "engine_pilot" / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
