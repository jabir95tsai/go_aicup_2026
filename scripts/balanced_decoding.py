"""Rank-balanced decoding simulation (RESULTS A-049), CPU only; preparation for 11/04.

Question probabilities: B2 on A-024 + HumanSL fitted on the A-044 (i) games
(q1000 S-fit + vs_rest S-fit / S-extra), applied to the 1,000 S-select
questions. Valid-U is not used. Each draw takes 400 S-select questions with a
given rank mix (10 per rank per k when balanced) and decodes them with
  score_rank   current per-question Bayes decoder
  sinkhorn     probabilities rescaled so each rank's total equals Q/10, then score_rank
  hungarian    exactly Q/10 questions per rank, maximising total expected official score
Scenarios: balanced; and unbalanced mixes (rank shares from a Dirichlet(alpha))
to measure the cost of assuming balance when it is false.

    python scripts/balanced_decoding.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from humansl_aggregation import load, part
from humansl_more_data import combine, rest_data
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.balanced import hungarian, sinkhorn
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs"
DRAWS, PER_FILE, SEED = 300, 400, 2027


def draw(rng, questions: pd.DataFrame, shares) -> np.ndarray:
    """PER_FILE question rows: 100 per k, rank counts within each k from shares."""
    rows = []
    for k in K_BUCKETS:
        counts = rng.multinomial(PER_FILE // len(K_BUCKETS), shares) if shares is not None \
            else np.full(10, PER_FILE // len(K_BUCKETS) // 10)
        for rank, count in enumerate(counts):
            pool = np.flatnonzero((questions["k"].to_numpy() == k) & (questions["rank_id"].to_numpy() == rank))
            rows.append(rng.choice(pool, count, replace=count > len(pool)))
    return np.concatenate(rows)


def main() -> int:
    data = load(OUT / "humansl_q1000")
    new_games, new_x = rest_data()
    fit_rows = new_games["role"].isin(["S-fit", "S-extra"]).to_numpy()
    fit = combine(part(data, "S-fit"), (new_games[fit_rows], new_x[fit_rows]))
    test = load(OUT / "humansl_qsel")
    probabilities = RankModel("B2").fit(*fit).probabilities(test[2], test[0]["game_index"].to_numpy(), test[1])
    questions, truth = test[1], test[1]["rank_id"].to_numpy()
    rng = np.random.default_rng(SEED)
    scenarios = {"balanced": None, "dirichlet_alpha_50": 50.0, "dirichlet_alpha_10": 10.0, "dirichlet_alpha_3": 3.0}
    result = {"S-select_all_score_rank": round(float(official_scores(score_rank(probabilities), truth).mean()), 4)}
    for name, alpha in scenarios.items():
        totals = {"score_rank": [], "sinkhorn": [], "hungarian": []}
        for _ in range(DRAWS):
            shares = None if alpha is None else rng.dirichlet(np.full(10, alpha))
            rows = draw(rng, questions, shares)
            p, y = probabilities[rows], truth[rows]
            totals["score_rank"].append(official_scores(score_rank(p), y).mean())
            totals["sinkhorn"].append(official_scores(score_rank(sinkhorn(p)), y).mean())
            totals["hungarian"].append(official_scores(hungarian(p), y).mean())
        base = np.array(totals["score_rank"])
        result[name] = {method: {"mean": round(float(np.mean(v)), 4),
                                 "gain_vs_score_rank": round(float(np.mean(np.array(v) - base)), 4),
                                 "gain_p05_p95": [round(float(q), 4) for q in np.quantile(np.array(v) - base, [0.05, 0.95])]}
                        for method, v in totals.items()}
        print(name, json.dumps(result[name]), flush=True)
    (OUT / "balanced_decoding.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
