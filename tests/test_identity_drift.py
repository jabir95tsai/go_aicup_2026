import numpy as np

from go_aicup.tasks.identity import drift


def test_match_depth_leave_one_out_and_external():
    prefix = np.array([[1, 2, 3, 4], [1, 2, 3, 5], [1, 9, 9, 9], [7, 7, 7, 7], [1, 2, 3, 4]], np.int16)
    reference = np.arange(4)
    assert drift.match_depth(prefix, reference, reference, in_reference=True).tolist() == [3, 3, 1, 0]
    assert drift.match_depth(prefix, reference, np.array([4])).tolist() == [4]


def test_drift_estimate_interpolates_and_clips():
    pool, foreign = np.full(10, 8.0), np.full(10, 6.0)
    assert drift.drift_estimate(pool, np.full(10, 7.0), foreign)["p_hat"] == 0.5
    assert drift.drift_estimate(pool, np.full(10, 9.0), foreign)["p_hat"] == 0.0


def test_robust_opening_eps0_ranks_as_sum_and_floors_each_game():
    scores = np.random.default_rng(0).normal(size=(6, 5)) * 10
    games = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    plain = np.stack([scores[g].sum(axis=0) for g in games])
    robust = drift.robust_opening(scores, games, 0.0)
    assert np.allclose(robust - robust.max(axis=1, keepdims=True), plain - plain.max(axis=1, keepdims=True))
    assert drift.robust_opening(scores, games, 0.1).min() >= 3 * np.log(0.1 / 5) - 1e-9
