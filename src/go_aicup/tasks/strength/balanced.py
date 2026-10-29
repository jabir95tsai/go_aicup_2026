"""Competition-2 rank-balanced decoding (RESULTS A-049, A-064), CPU only.

If every rank has the same number of test questions, decoding can use that:
hungarian() gives each rank at most `capacity` questions (default ceil(n/10))
and maximises the total expected official score sum_q E[score | p_q, guess],
instead of score_rank's per-question maximum. sinkhorn() is the soft version
(probabilities rescaled so each rank's total is n/10). A-049: about +0.006 if
the test is balanced, -0.01 to -0.07 if not.

The command reads the probabilities of an aggregate run (p_<rank> columns,
e.g. the AH+C primary rankvec_predictions.parquet) and writes a submission:

    python -m go_aicup.tasks.strength.balanced --predictions <work>/rankvec_predictions.parquet \\
        --submission <out>/submissions/rank_public_balanced.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import NEAR


def expected_scores(p: np.ndarray) -> np.ndarray:
    """Expected official score of guessing each rank: p[r] + NEAR * (p[r-1] + p[r+1])."""
    padded = np.pad(p, ((0, 0), (1, 1)))
    return p + NEAR * (padded[:, :-2] + padded[:, 2:])


def sinkhorn(p: np.ndarray, iterations: int = 200) -> np.ndarray:
    q = p.copy()
    target = len(p) / p.shape[1]
    for _ in range(iterations):
        q *= target / q.sum(axis=0, keepdims=True)
        q /= q.sum(axis=1, keepdims=True)
    return q


def hungarian(p: np.ndarray, capacity=None) -> np.ndarray:
    """One rank per question, at most capacity[r] questions per rank (default ceil(n/10) each)."""
    capacity = np.full(p.shape[1], -(-len(p) // p.shape[1])) if capacity is None else np.asarray(capacity)
    columns = np.repeat(np.arange(p.shape[1]), capacity)
    rows, cols = linear_sum_assignment(-expected_scores(p)[:, columns])
    pred = np.empty(len(p), dtype=np.int64)
    pred[rows] = columns[cols]
    return pred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True, help="aggregate output with p_<rank> columns")
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.predictions)
    rank_id = hungarian(table[[f"p_{r}" for r in RANKS]].to_numpy())
    result = table[["question_id"]].assign(rank=[RANKS[r] for r in rank_id])
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    # Provisional format; replace once the official sample_submission is out.
    result.to_csv(args.submission, index=False)
    changed = int((rank_id != table["rank_id"].to_numpy()).sum())
    print(f"balanced (hungarian): {len(result)} questions, {changed} differ from the input decoding, "
          f"predicted ranks={result['rank'].value_counts().reindex(RANKS, fill_value=0).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
