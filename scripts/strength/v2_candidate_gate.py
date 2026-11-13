"""Single-model replacement gate for competition 2 (CPU only), RESULTS A-037.

Rule, fixed before the 14-epoch run finished: candidate and reference (A-024)
each get C2 on their own outputs (per-game mean log-prob, questions 1/k per
game, per-k temperature fitted on S-fit, metrics.score_rank). Gate: candidate
Valid-U Equal-k >= reference + 0.005, the 95% interval of a player-grouped
bootstrap of the difference has a lower bound > 0, and the candidate's
G = Equal-k(S-select) - Equal-k(U) <= 0.08. Valid-U is never used for fitting.

    python -m scripts.strength.v2_candidate_gate --reference outputs/v2_strength --candidate outputs/v2_strength_e14
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.strength.v2_ensemble_gate import (BOOTSTRAP, DATASET, FILES, MIN_GAIN, SPLIT_SEED, by_k, equal_k, per_game_logp,
                              question_logits)
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import (decode_questions, fit_temperature, official_scores,
                                                  tempered_probabilities)

MAX_GAP = 0.08


def c2_scores(categories: dict, parts: dict) -> tuple[dict, dict]:
    """Per-question official scores on S-select and U with T_k fitted on S-fit."""
    source = {"S-fit": "S", "S-select": "S", "U": "U"}
    logits = {part: question_logits(per_game_logp(categories[source[part]]),
                                    categories[source[part]]["game_index"], frame)
              for part, frame in parts.items()}
    temps, edges, scores = {}, {}, {}
    for k in K_BUCKETS:
        fit = parts["S-fit"]["k"].to_numpy() == k
        temps[k], edges[k] = fit_temperature(logits["S-fit"][fit], parts["S-fit"]["rank_id"].to_numpy()[fit])
    for part in ("S-select", "U"):
        probabilities = np.empty_like(logits[part])
        for k in K_BUCKETS:
            mask = parts[part]["k"].to_numpy() == k
            probabilities[mask] = tempered_probabilities(logits[part][mask], temps[k])
        scores[part] = official_scores(decode_questions(probabilities, "score_rank"), parts[part]["rank_id"].to_numpy())
    return scores, {"temperatures": temps, "grid_edge": edges}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs" / "v2_strength")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    unseen = pd.read_parquet(DATASET / "strength_v2_valid_unseen_questions.parquet")
    players = np.sort(seen["player_id"].astype(str).unique())
    fit_players = set(np.random.default_rng(SPLIT_SEED).permutation(players)[:len(players) // 2])
    in_fit = seen["player_id"].astype(str).isin(fit_players)
    parts = {"S-fit": seen[in_fit].reset_index(drop=True), "S-select": seen[~in_fit].reset_index(drop=True),
             "U": unseen}

    result, scores = {}, {}
    for role, run in (("reference", args.reference), ("candidate", args.candidate)):
        categories = {name: np.load(run / file, allow_pickle=False) for name, file in FILES.items()}
        scores[role], fit = c2_scores(categories, parts)
        u = equal_k(scores[role]["U"], unseen["k"].to_numpy())
        s = equal_k(scores[role]["S-select"], parts["S-select"]["k"].to_numpy())
        result[role] = {"run": str(run), "U": by_k(scores[role]["U"], unseen["k"].to_numpy()),
                        "S-select": by_k(scores[role]["S-select"], parts["S-select"]["k"].to_numpy()),
                        "G": round(s - u, 4), **fit}

    k = unseen["k"].to_numpy()
    candidate, reference = scores["candidate"]["U"], scores["reference"]["U"]
    gain = equal_k(candidate, k) - equal_k(reference, k)
    u_players = unseen["player_id"].astype(str).to_numpy()
    groups = pd.Series(np.arange(len(unseen))).groupby(u_players).apply(np.asarray).to_list()
    rng = np.random.default_rng(SPLIT_SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(candidate[rows], k[rows]) - equal_k(reference[rows], k[rows]))
    low, high = np.quantile(draws, [0.025, 0.975])
    gap = result["candidate"]["G"]
    passed = gain >= MIN_GAIN and low > 0 and gap <= MAX_GAP
    result["gate"] = {"gain_equal_k": round(gain, 4), "bootstrap_95": [round(float(low), 4), round(float(high), 4)],
                      "players": len(groups), "min_gain": MIN_GAIN, "candidate_G": gap, "max_G": MAX_GAP,
                      "pass": bool(passed)}
    print(json.dumps(result, indent=1, default=str))
    out = args.out or args.candidate.parent / f"{args.candidate.name}_gate.json"
    out.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
