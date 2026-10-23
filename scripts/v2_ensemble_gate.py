"""T2 competition-2 ensemble gate (EXECUTION_PLAN T2, RESULTS D-002), CPU only.

Rule, fixed before the component runs finished: per game, average the mean
log-probability of every model; a question averages its games (1/k each);
per-k temperatures are fitted on S-fit (the A-025 half of Valid-S, same
player draw); decode with metrics.score_rank. The reference is the A-024
model alone under the same C2 rule and the same S-fit fit.

Gate (confirmed 2026-10-22): ensemble Equal-k on Valid-U >= reference + 0.005
AND the 95% interval of a player-grouped bootstrap of the difference has a
lower bound > 0. Valid-U is never used for fitting. Each component alone is
reported as a diagnostic only.

    python scripts/v2_ensemble_gate.py --runs outputs/v2_strength outputs/v2_strength_seed2028_cosine \\
        outputs/v2_strength_seed2029_cosine
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.v2.aggregate import (decode_questions, fit_temperature, official_scores,
                                                  tempered_probabilities)

DATASET = PROCESSED / "strength_v2" / "seed2027"
FILES = {"U": "validation_unseen_category.npy", "S": "validation_seen_category.npy"}
MIN_GAIN = 0.005
SPLIT_SEED = 2027   # A-025 S-fit / S-select player draw
BOOTSTRAP = 2000


def per_game_logp(category: np.ndarray) -> np.ndarray:
    return category["log_probability_sum"] / category["num_moves"][:, None]


def question_logits(per_game, game_index, questions) -> np.ndarray:
    position = np.searchsorted(game_index, np.concatenate(list(questions["games"])))
    sizes = questions["games"].map(len).to_numpy()
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return np.add.reduceat(per_game[position], starts, axis=0) / sizes[:, None]


def c2_scores(logits: dict, parts: dict) -> tuple[np.ndarray, dict]:
    """Per-question official score on U after fitting T_k on S-fit."""
    temps, edges, probabilities = {}, {}, np.empty_like(logits["U"])
    for k in K_BUCKETS:
        fit = parts["S-fit"]["k"].to_numpy() == k
        temps[k], edges[k] = fit_temperature(logits["S-fit"][fit], parts["S-fit"]["rank_id"].to_numpy()[fit])
        mask = parts["U"]["k"].to_numpy() == k
        probabilities[mask] = tempered_probabilities(logits["U"][mask], temps[k])
    pred = decode_questions(probabilities, "score_rank")
    return official_scores(pred, parts["U"]["rank_id"].to_numpy()), {"temperatures": temps, "grid_edge": edges}


def equal_k(scores: np.ndarray, k: np.ndarray) -> float:
    return float(np.mean([scores[k == b].mean() for b in K_BUCKETS]))


def by_k(scores, k) -> dict:
    return {str(b): round(float(scores[k == b].mean()), 4) for b in K_BUCKETS} | {"equal_k": round(equal_k(scores, k), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="first run = reference (A-024)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "v2_ensemble_gate" / "result.json")
    args = parser.parse_args()

    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    unseen = pd.read_parquet(DATASET / "strength_v2_valid_unseen_questions.parquet")
    players = np.sort(seen["player_id"].astype(str).unique())
    fit_players = set(np.random.default_rng(SPLIT_SEED).permutation(players)[:len(players) // 2])
    parts = {"S-fit": seen[seen["player_id"].astype(str).isin(fit_players)].reset_index(drop=True), "U": unseen}
    source = {"S-fit": "S", "U": "U"}

    per_run = []
    for run in args.runs:
        categories = {name: np.load(run / file, allow_pickle=False) for name, file in FILES.items()}
        per_run.append(categories)
    for name in FILES:
        first = per_run[0][name]
        for categories in per_run[1:]:
            if not (np.array_equal(categories[name]["game_index"], first["game_index"])
                    and np.array_equal(categories[name]["num_moves"], first["num_moves"])):
                raise ValueError(f"runs differ in their {name} games; they must share one split")

    def logits_for(members) -> dict:
        out = {}
        for part, frame in parts.items():
            name = source[part]
            mean = np.mean([per_game_logp(per_run[m][name]) for m in members], axis=0)
            out[part] = question_logits(mean, per_run[0][name]["game_index"], frame)
        return out

    k = unseen["k"].to_numpy()
    result = {"runs": [str(r) for r in args.runs], "components": {}}
    component_scores = []
    for m, run in enumerate(args.runs):
        scores, fit = c2_scores(logits_for([m]), parts)
        component_scores.append(scores)
        result["components"][run.name] = {"U": by_k(scores, k), **fit}
    ensemble, fit = c2_scores(logits_for(range(len(args.runs))), parts)
    result["ensemble"] = {"U": by_k(ensemble, k), **fit}

    reference = component_scores[0]
    gain = equal_k(ensemble, k) - equal_k(reference, k)
    u_players = unseen["player_id"].astype(str).to_numpy()
    groups = pd.Series(np.arange(len(unseen))).groupby(u_players).apply(np.asarray).to_list()
    rng = np.random.default_rng(SPLIT_SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(ensemble[rows], k[rows]) - equal_k(reference[rows], k[rows]))
    low, high = np.quantile(draws, [0.025, 0.975])
    passed = gain >= MIN_GAIN and low > 0
    result["gate"] = {"gain_equal_k": round(gain, 4), "bootstrap_95": [round(float(low), 4), round(float(high), 4)],
                      "players": len(groups), "min_gain": MIN_GAIN, "pass": bool(passed)}
    print(json.dumps(result, indent=1, default=str))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
