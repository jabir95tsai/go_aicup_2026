"""Pool-level post-processing of a (questions x candidates) score matrix (L6).

Each question alone ranks candidates by its row. When questions and candidates
are one-to-one (the synthetic pools are by construction; the real test is
unconfirmed), the columns carry information too: a "hub" candidate that scores
high for many questions can be the answer to at most one of them.

row_posterior: softmax of scores / temperature per question.
sinkhorn:      alternately normalise rows and columns of the posterior so each
               candidate's total mass is 1 (soft one-to-one).
hungarian:     the best one-to-one assignment becomes rank 1; ranks 2-5 follow
               the Sinkhorn order.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax, logsumexp


def row_posterior(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """log P(candidate | question)."""
    return log_softmax(scores / temperature, axis=1)


def sinkhorn(log_p: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Log of a (near) doubly stochastic matrix; columns sum to n_rows / n_cols."""
    log_q = log_p.copy()
    col_target = np.log(log_q.shape[0] / log_q.shape[1])
    for _ in range(iterations):
        log_q -= logsumexp(log_q, axis=0, keepdims=True) - col_target
        log_q -= logsumexp(log_q, axis=1, keepdims=True)
    return log_q


def hungarian_top5(log_q: np.ndarray) -> np.ndarray:
    """(n_questions, 5) candidate positions: assignment first, then by log_q."""
    rows, cols = linear_sum_assignment(-log_q)
    first = np.full(log_q.shape[0], -1)
    first[rows] = cols
    order = np.argsort(-log_q, axis=1)
    out = np.empty((log_q.shape[0], 5), np.int64)
    for i in range(log_q.shape[0]):
        rest = order[i][order[i] != first[i]] if first[i] >= 0 else order[i]
        out[i] = np.concatenate([[first[i]], rest[:4]]) if first[i] >= 0 else rest[:5]
    return out


def top5(log_q: np.ndarray) -> np.ndarray:
    return np.argsort(-log_q, axis=1)[:, :5]
