"""Opening-fingerprint scorers for competition 1: (questions x candidates) log-likelihoods.

Both models score only the queried player's own (black, even-ply) moves among
the first `plies` plies; white moves enter only as history.

slot_nb:   P_c(m_t) per ply slot t, a candidate's smoothed categorical
           (A-005 E5). Treats consecutive opening moves as independent, so a
           joseki the candidate always plays is counted once per move.
prefix:    P_c(m_t | m_<t) = (N_c(m_<=t) + beta * slot_nb_c(t, m_t)) / (N_c(m_<t) + beta),
           N_c(h) = number of the candidate's gallery games whose opening starts
           with h. Deep exact-prefix matches carry the habit; once the query
           leaves every gallery line N_c is 0 and it falls back to slot_nb.
"""
from __future__ import annotations

import numpy as np

N_POINTS = 361
_MULT = np.uint64(0x9E3779B97F4A7C15)
_CAND = np.uint64(0xC2B2AE3D27D4EB4F)


def population_prior(prefix: np.ndarray, games: np.ndarray, plies: np.ndarray) -> np.ndarray:
    """[n_slot, 361] point frequencies (add-one) over `games`."""
    counts = np.stack([np.bincount(prefix[games, t], minlength=N_POINTS) for t in plies])
    counts = counts + 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def slot_log_prob(prefix, gallery_games, plies, pop, alpha) -> np.ndarray:
    """[n_cand, n_slot, 361] log P_c(m at slot)."""
    logp = np.empty((len(gallery_games), len(plies), N_POINTS), np.float32)
    for i, games in enumerate(gallery_games):
        for j, t in enumerate(plies):
            counts = np.bincount(prefix[games, t], minlength=N_POINTS)
            logp[i, j] = np.log((counts + alpha * pop[j]) / (len(games) + alpha))
    return logp


def _game_slot_scores(logp, points) -> np.ndarray:
    """[n_games, n_cand] summed slot log-probs; points [n_games, n_slot]."""
    slots = np.arange(points.shape[1])
    # logp[:, slot, point] for every game -> [n_cand, n_games, n_slot]
    return logp[:, slots[None, :], points].sum(axis=2).T


def slot_nb(prefix, gallery_games, query_games, plies, pop, alpha=5.0) -> np.ndarray:
    logp = slot_log_prob(prefix, gallery_games, plies, pop, alpha)
    flat = np.concatenate(query_games)
    per_game = _game_slot_scores(logp, prefix[flat][:, plies].astype(np.int64))
    return _sum_by_question(per_game, query_games)


def prefix_hashes(prefix: np.ndarray) -> np.ndarray:
    """uint64 [n_games, plies + 1]; column t hashes the first t plies (column 0 = empty)."""
    hashes = np.zeros((len(prefix), prefix.shape[1] + 1), np.uint64)
    with np.errstate(over="ignore"):
        for t in range(prefix.shape[1]):
            hashes[:, t + 1] = hashes[:, t] * _MULT + prefix[:, t].astype(np.uint64) + np.uint64(1)
    return hashes


def _keyed(hashes: np.ndarray, cand: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return hashes ^ ((cand.astype(np.uint64) + np.uint64(1)) * _CAND)


class _Counter:
    """Counts of (candidate, prefix hash) over the gallery."""

    def __init__(self, hashes, gallery_games, depth):
        cand = np.concatenate([np.full(len(g), i) for i, g in enumerate(gallery_games)])
        rows = hashes[np.concatenate(gallery_games), :depth + 1]
        keys = _keyed(rows, cand[:, None]).ravel()
        self.keys, self.counts = np.unique(keys, return_counts=True)

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, len(self.keys) - 1)
        return np.where(self.keys[pos] == keys, self.counts[pos], 0)


def population_conditional(hashes, prefix, population_games, query_flat, plies, pop, gamma=1.0):
    """[n_query_games, n_slot] P_pop(m_t | m_<t) from `population_games`, backed off to `pop`."""
    depth = int(np.max(plies)) + 1
    counter = _Counter(hashes, [population_games], depth)
    zero = np.zeros(1, np.int64)
    out = np.empty((len(query_flat), len(plies)))
    for j, t in enumerate(plies):
        before = counter(_keyed(hashes[query_flat, t], zero))
        after = counter(_keyed(hashes[query_flat, t + 1], zero))
        base = pop[j, prefix[query_flat, t].astype(np.int64)]
        out[:, j] = (after + gamma * base) / (before + gamma)
    return out


def prefix_backoff(prefix, gallery_games, query_games, plies, pop, alpha=5.0, beta=1.0,
                   hashes=None, base=None) -> np.ndarray:
    """`plies` must be the black plies 0, 2, ..., in order; history includes white.

    `base` [n_query_games, n_slot] replaces the candidate slot back-off with a
    candidate-independent one (population_conditional): candidates that never
    played the query's line then all score the same and the term cancels.
    """
    plies = np.asarray(plies)
    depth = int(plies.max()) + 1
    hashes = prefix_hashes(prefix[:, :depth]) if hashes is None else hashes
    logp = slot_log_prob(prefix, gallery_games, plies, pop, alpha) if base is None else None
    counter = _Counter(hashes, gallery_games, depth)
    flat = np.concatenate(query_games)
    n_cand = len(gallery_games)
    cand = np.arange(n_cand)
    points = prefix[flat][:, plies].astype(np.int64)
    per_game = np.zeros((len(flat), n_cand))
    for j, t in enumerate(plies):
        before = counter(_keyed(hashes[flat, t][:, None], cand[None, :]))
        after = counter(_keyed(hashes[flat, t + 1][:, None], cand[None, :]))
        if base is None:
            backoff = np.exp(logp[:, j, points[:, j]]).T      # [n_games, n_cand]
        else:
            backoff = base[:, j:j + 1]
        per_game += np.log((after + beta * backoff) / (before + beta))
    return _sum_by_question(per_game, query_games)


def _sum_by_question(per_game: np.ndarray, query_games) -> np.ndarray:
    sizes = np.array([len(g) for g in query_games])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return np.add.reduceat(per_game, starts, axis=0)
