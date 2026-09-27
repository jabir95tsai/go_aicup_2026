"""Evaluation. Every report is split by k bucket (5/10/15/20) plus overall.

There is deliberately no function that returns only an overall number: the
test set weights the four buckets equally, and an average hides a k=5
collapse (CLAUDE.md section 3).

Official metrics (confirmed 2026-09-27, CLAUDE.md section 0):
  rank      -> `score`: 1 for exact, 1/e for |pred - true| == 1, else 0
  player id -> `score`: exp(-(r - 1)) for the truth at position r in the top 5
The other columns (acc / MAE / QWK / top-5 / MAP@5) are kept as diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

K_BUCKETS = (5, 10, 15, 20)
N_RANKS = 10
NEAR = float(np.exp(-1.0))                     # rank: credit for an off-by-one guess
PID_WEIGHTS = np.exp(-np.arange(5, dtype=float))  # player id: 1, .368, .135, .050, .018


def _by_bucket(k: np.ndarray, fn) -> pd.DataFrame:
    rows = []
    for bucket in K_BUCKETS:
        mask = k == bucket
        if mask.any():
            rows.append({"k": str(bucket), "n": int(mask.sum()), **fn(mask)})
    rows.append({"k": "all", "n": int(len(k)), **fn(np.ones(len(k), bool))})
    return pd.DataFrame(rows).set_index("k")


def rank_report(pred, true, k) -> pd.DataFrame:
    """`pred`, `true`: rank ids 0..9 per question; `k`: games per question."""
    pred, true, k = np.asarray(pred), np.asarray(true), np.asarray(k)

    def fn(mask):
        p, t = pred[mask], true[mask]
        qwk = cohen_kappa_score(t, p, weights="quadratic", labels=list(range(N_RANKS)))
        err = np.abs(p - t)
        return {
            "score": float(np.where(err == 0, 1.0, np.where(err == 1, NEAR, 0.0)).mean()),
            "acc": float((p == t).mean()),
            "mae": float(np.abs(p - t).mean()),
            "qwk": float(qwk),
            "within1": float((np.abs(p - t) <= 1).mean()),
        }

    return _by_bucket(k, fn)


def score_rank(proba: np.ndarray) -> np.ndarray:
    """Bayes-optimal decode for the official rank score.

    E[score | guess r] = p[r] + (p[r-1] + p[r+1]) / e, so the best guess is not
    the argmax: it favours ranks with probable neighbours, and the edge ranks
    (D, 6D) are penalised because they have only one neighbour to share credit.
    """
    proba = np.asarray(proba, dtype=float)
    gain = proba.copy()
    gain[:, 1:] += NEAR * proba[:, :-1]
    gain[:, :-1] += NEAR * proba[:, 1:]
    return gain.argmax(axis=1)


def expected_rank(proba: np.ndarray) -> np.ndarray:
    """Expected-value decode, rounded; the MAE-optimal decode is the median."""
    return np.rint(proba @ np.arange(proba.shape[1])).astype(int)


def median_rank(proba: np.ndarray) -> np.ndarray:
    return (np.cumsum(proba, axis=1) < 0.5).sum(axis=1).clip(0, proba.shape[1] - 1)


def pid_report(ranked, true, k) -> pd.DataFrame:
    """`ranked`: (n_questions, >=5) candidate ids best-first; `true`: id per question."""
    ranked, true, k = np.asarray(ranked)[:, :5], np.asarray(true), np.asarray(k)
    hit = ranked == true[:, None]
    position = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)

    def fn(mask):
        pos = position[mask]
        return {
            "score": float(np.where(pos >= 0, PID_WEIGHTS[np.maximum(pos, 0)], 0.0).mean()),
            "top1": float((pos == 0).mean()),
            "top5": float((pos >= 0).mean()),
            "map5": float(np.where(pos >= 0, 1.0 / (np.maximum(pos, 0) + 1), 0.0).mean()),
        }

    return _by_bucket(k, fn)


def top5_from_scores(scores: np.ndarray, candidate_ids) -> np.ndarray:
    """(n_questions, n_candidates) scores -> (n_questions, 5) ids, best first."""
    order = np.argsort(-scores, axis=1)[:, :5]
    return np.asarray(candidate_ids)[order]


def format_report(report: pd.DataFrame, digits: int = 4) -> str:
    return report.round(digits).to_string()
