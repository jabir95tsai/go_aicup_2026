"""Competition-1 opening drift: label-free detection and a robust opening aggregate (RESULTS A-061, A-062).

Detection (Day-0, no labels):
  match_depth     longest common prefix (plies) of each game with any game of a
                  reference set (the candidate pool); leave-one-out when the
                  game is itself in the reference
  drift_estimate  p_hat = (D_pool - D_query) / (D_pool - D_foreign), clipped to
                  [0, 1]: where the query games sit between "the pool's own
                  habits" and "openings of players outside the pool"
  agreement       share of questions whose opening-only and style-only top-1 agree

Robust aggregate: per game, the candidate posterior softmax_c(s_g) is mixed with
uniform, o(q, c) = sum_g log((1 - eps) * pi_g(c) + eps / N). eps = 0 ranks as
the plain sum of opening scores; eps > 0 caps what one unfamiliar game can take
away from the true candidate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax

from go_aicup.tasks.identity import opening


def match_depth(prefix: np.ndarray, reference: np.ndarray, games: np.ndarray, in_reference: bool = False,
                hashes: np.ndarray | None = None) -> np.ndarray:
    """int [len(games)]: plies shared with the closest reference game.

    in_reference: every game is itself a reference row and must not match itself.
    """
    hashes = opening.prefix_hashes(prefix) if hashes is None else hashes
    need = 2 if in_reference else 1
    depth = np.zeros(len(games), np.int64)
    for t in range(1, hashes.shape[1]):
        keys, counts = np.unique(hashes[reference, t], return_counts=True)
        h = hashes[games, t]
        pos = np.minimum(np.searchsorted(keys, h), len(keys) - 1)
        hit = (keys[pos] == h) & (counts[pos] >= need)
        if not hit.any():
            break
        depth[hit] = t      # a match at t implies a match at every shorter prefix
    return depth


def drift_estimate(pool_depth: np.ndarray, query_depth: np.ndarray, foreign_depth: np.ndarray) -> dict:
    pool, query, foreign = (float(np.mean(d)) for d in (pool_depth, query_depth, foreign_depth))
    span = pool - foreign
    p_hat = float(np.clip((pool - query) / span, 0.0, 1.0)) if span > 0 else float("nan")
    return {"mean_depth_pool": pool, "mean_depth_query": query, "mean_depth_foreign": foreign, "p_hat": p_hat}


def agreement(opening_scores: np.ndarray, style_scores: np.ndarray) -> float:
    return float(np.mean(opening_scores.argmax(axis=1) == style_scores.argmax(axis=1)))


def robust_opening(per_game: np.ndarray, question_games: list[np.ndarray], eps: float) -> np.ndarray:
    """[n_questions, n_cand] from per-game opening scores [n_games, n_cand]."""
    log_pi = log_softmax(per_game, axis=1)
    if eps > 0:
        log_pi = np.logaddexp(np.log1p(-eps) + log_pi, np.log(eps / per_game.shape[1]))
    return np.stack([log_pi[g].sum(axis=0) for g in question_games])
