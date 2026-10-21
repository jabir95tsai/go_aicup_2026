"""Competition-1 opening scorers and pool post-processing."""
import numpy as np
from scipy.special import logsumexp

from go_aicup.tasks.identity import assign, data, opening


def _games(lines):
    return np.array(lines, dtype=np.int16)


def test_slot_nb_matches_direct_count():
    prefix = _games([[1, 9, 2, 9], [1, 9, 3, 9], [5, 9, 2, 9], [1, 9, 2, 9]])
    gallery = [np.array([0, 1]), np.array([2])]
    plies = np.array([0, 2])
    pop = np.full((2, opening.N_POINTS), 1 / opening.N_POINTS)
    scores = opening.slot_nb(prefix, gallery, [np.array([3])], plies, pop, alpha=1.0)
    p = 1 / opening.N_POINTS
    first = np.log((2 + p) / 3) + np.log((1 + p) / 3)       # candidate 0: 1 at slot 0 twice, 2 once
    second = np.log((0 + p) / 2) + np.log((1 + p) / 2)      # candidate 1 played 5 then 2
    np.testing.assert_allclose(scores[0], [first, second], rtol=1e-5)


def test_prefix_hashes_distinguish_order_and_length():
    h = opening.prefix_hashes(_games([[1, 2, 3], [2, 1, 3], [1, 2, 4]]))
    assert h.shape == (3, 4)
    assert (h[:, 0] == 0).all()
    assert h[0, 2] != h[1, 2] and h[0, 2] == h[2, 2] and h[0, 3] != h[2, 3]


def test_population_backoff_cancels_for_unmatched_candidates():
    # candidate 0 always plays line A; candidate 1 line B; query follows A
    a, b = [1, 9, 2, 9, 3, 9], [4, 9, 5, 9, 6, 9]
    prefix = _games([a, a, b, b, a, b])
    gallery = [np.array([0, 1]), np.array([2, 3])]
    plies = np.array([0, 2, 4])
    hashes = opening.prefix_hashes(prefix)
    pop = np.full((3, opening.N_POINTS), 1 / opening.N_POINTS)
    base = opening.population_conditional(hashes, prefix, np.array([4, 5]), np.array([4]), plies, pop)
    s = opening.prefix_backoff(prefix, gallery, [np.array([4])], plies, pop, beta=1.0,
                               hashes=hashes, base=base)
    assert s[0, 0] > s[0, 1]
    # candidate 1 never played the line: its first move is judged against its own
    # 2 gallery games (empty history matches all), later factors are the population term
    expected = np.log(base[0, 0] / 3) + np.log(base[0, 1:]).sum()
    np.testing.assert_allclose(s[0, 1], expected, rtol=1e-6)


def test_sinkhorn_is_doubly_stochastic_and_hungarian_first():
    rng = np.random.default_rng(0)
    log_q = assign.sinkhorn(assign.row_posterior(rng.normal(size=(6, 6)) * 3), iterations=500)
    np.testing.assert_allclose(logsumexp(log_q, axis=1), 0, atol=1e-6)
    np.testing.assert_allclose(logsumexp(log_q, axis=0), 0, atol=1e-4)
    # a hub column best for every row can win only one row under the assignment
    scores = np.eye(4) * 2
    scores[:, 0] += 5
    ranked = assign.hungarian_top5(assign.row_posterior(np.pad(scores, ((0, 0), (0, 1)))))
    assert sorted(ranked[:, 0]) == [0, 1, 2, 3]
    assert all(len(set(r)) == 5 for r in ranked)


def test_canonical_d4_is_invariant_and_keeps_shared_prefixes():
    rng = np.random.default_rng(0)
    tables = data.d4_tables()
    diag = [0, 20, 40, 60, 80, 100]                     # on the main diagonal: symmetric under transpose
    prefix = np.array([diag + [1, 2], diag + [19, 38], list(rng.integers(0, 361, 8))], np.int16)
    canon = data.canonical_d4(prefix)
    for g in range(8):
        np.testing.assert_array_equal(data.canonical_d4(tables[g][prefix]), canon)
    assert any((tables[g][prefix[0]] == canon[0]).all() for g in range(8))
    # rows 0 and 1 are transposes after 6 shared plies: one canonical form, prefix kept
    np.testing.assert_array_equal(canon[0], canon[1])
    np.testing.assert_array_equal(data.canonical_d4(prefix[:, :6])[0], canon[0, :6])
