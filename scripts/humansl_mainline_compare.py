"""Competition-2 main-line comparison (RESULTS A-045), CPU only; diagnostics, no gate.

Candidates on the same S-select (1,000) and Valid-U (1,000) questions:
  A024_C2  A-024 + C2 (current submission path)
  B2_H     B2 on HumanSL features only (30)
  B2_AH    B2 on A-024 + HumanSL (40; A-041 choice)
B2 is fitted on the S-fit games only (A-041 setting), so S-select is out of
sample for every candidate's fitting step. Reported per k, per rank, per colour,
the exact / one-off / worse shares, and G = Equal-k(S-select) - Equal-k(U).
Valid-U numbers of B2_H are an extra look at U, reported as a diagnostic only.

    python scripts/humansl_mainline_compare.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from humansl_aggregation import load, part
from humansl_questions import by_k, equal_k, fit_players
from v2_candidate_gate import c2_scores
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"


def a024_c2(questions: dict) -> dict:
    """A-024 + C2 predictions for each question set, T_k on all S-fit questions (A-025)."""
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    s_fit = seen[seen["player_id"].astype(str).isin(fit_players(seen))].reset_index(drop=True)
    categories = {name: np.load(OUT / "v2_strength" / file, allow_pickle=False) for name, file in
                  (("U", "validation_unseen_category.npy"), ("S", "validation_seen_category.npy"))}
    # c2_scores scores its "S-select" part on Valid-S results and "U" on Valid-U results.
    scores, _ = c2_scores(categories, {"S-fit": s_fit, "S-select": questions["S-select"], "U": questions["U"]})
    return scores


def predictions(model: RankModel, data, columns) -> np.ndarray:
    games, questions, x = data
    return score_rank(model.probabilities(x[:, columns], games["game_index"].to_numpy(), questions))


def summary(scores: np.ndarray, pred, questions: pd.DataFrame) -> dict:
    k, true = questions["k"].to_numpy(), questions["rank_id"].to_numpy()
    out = by_k(scores, k)
    out["per_rank"] = dict(zip(RANKS, pd.Series(scores).groupby(true).mean().round(3).tolist()))
    out["per_colour"] = pd.Series(scores).groupby(questions["color"].astype(str).to_numpy()).mean().round(4).to_dict()
    if pred is not None:
        error = np.abs(np.asarray(pred) - true)
        out["exact_oneoff_worse"] = [round(float((error == 0).mean()), 3), round(float((error == 1).mean()), 3),
                                     round(float((error >= 2).mean()), 3)]
    return out


def main() -> int:
    data = load(OUT / "humansl_q1000")
    fit, u = part(data, "S-fit"), part(data, "U")
    select = load(OUT / "humansl_qsel")
    sets = {"S-select": select, "U": u}
    reference = a024_c2({name: value[1] for name, value in sets.items()})
    all_columns, human_columns = np.arange(fit[2].shape[1]), np.arange(10, fit[2].shape[1])
    result = {}
    for name, value in sets.items():
        result.setdefault("A024_C2", {})[name] = summary(reference[name], None, value[1])
    for label, columns in (("B2_H", human_columns), ("B2_AH", all_columns)):
        model = RankModel("B2").fit(fit[0], fit[1], fit[2][:, columns])
        for name, value in sets.items():
            pred = predictions(model, value, columns)
            result.setdefault(label, {})[name] = summary(
                official_scores(pred, value[1]["rank_id"].to_numpy()), pred, value[1])
    for label, value in result.items():
        value["G"] = round(value["S-select"]["equal_k"] - value["U"]["equal_k"], 4)
    print(json.dumps(result, indent=1, default=str))
    (OUT / "humansl_mainline_compare.json").write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
