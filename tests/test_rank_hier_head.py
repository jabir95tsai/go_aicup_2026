import numpy as np

import scripts.strength.rank_hier_head as hh


def _synthetic(seed=0, d=4, per_rank=400, max_games=40, garbage=None):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(size=(d, d)), rng.normal(size=(d, d))
    between, within = a @ a.T / d + 0.2 * np.eye(d), 4 * (b @ b.T / d + 0.2 * np.eye(d))
    step = rng.normal(size=d)
    x, player, rank, color = [], [], [], []
    for p in range(hh.RANKS * per_rank):
        r, n = p % hh.RANKS, int(rng.integers(1, max_games + 1))
        centre = r * step + rng.multivariate_normal(np.zeros(d), between)
        x.append(rng.multivariate_normal(centre, within, n))
        player += [p] * n
        rank += [r] * n
        color += list(rng.integers(0, 2, n))
    x = np.concatenate(x)
    if garbage is not None:
        x[np.isin(player, garbage)] += 1000.0
    data = hh.Data(x, np.array(player), np.array(rank), np.array(color), np.ones(len(x), bool))
    return data, between, within


def test_variance_components_recovered_with_unequal_games():
    data, between, within = _synthetic()
    parts = hh.fit_components(data, np.arange(len(data.n)))
    scale = np.outer(parts["scale"], parts["scale"])
    for estimate, truth in ((parts["within"] * scale, within), (parts["between"] * scale, between)):
        assert np.linalg.norm(estimate - truth) / np.linalg.norm(truth) < 0.1
    naive = np.cov((data.mean - data.mean.mean(0))[data.rank == 0].T)       # no finite-n correction
    assert np.trace(naive) > 1.2 * np.trace(between)


def test_shared_gaussian_score_is_linear():
    data, _, _ = _synthetic(seed=1, per_rank=60)
    parts = hh.fit_components(data, np.arange(len(data.n)))
    rng = np.random.default_rng(2)
    mean, k = rng.normal(size=(50, 4)) * 3, np.repeat(hh.K_BUCKETS, 50)[:50]
    k = rng.choice(hh.K_BUCKETS, 50)
    ll = hh.loglik(parts, 0.1, mean, k)
    z = (mean - parts["center"]) / parts["scale"]
    for bucket in hh.K_BUCKETS:
        rows = k == bucket
        chol, _ = hh.covariance(parts, bucket, 0.1)
        inverse = np.linalg.inv(chol @ chol.T)
        for r in range(1, hh.RANKS):
            w = inverse @ (parts["mu"][r] - parts["mu"][0])
            bias = -0.5 * (parts["mu"][r] @ inverse @ parts["mu"][r] - parts["mu"][0] @ inverse @ parts["mu"][0])
            assert np.allclose(ll[rows, r] - ll[rows, 0], z[rows] @ w + bias, atol=1e-8)


def test_held_out_players_do_not_enter_estimation_or_calibration():
    clean, _, _ = _synthetic(seed=3, per_rank=200, max_games=90)
    players = np.arange(len(clean.n))
    held = np.sort(np.random.default_rng(9).permutation(players)[:len(players) // 5])
    dirty, _, _ = _synthetic(seed=3, per_rank=200, max_games=90, garbage=held)
    train = np.setdiff1d(players, held)
    hh.TEMPERATURES, saved = np.geomspace(0.01, 1000, 41), hh.TEMPERATURES
    try:
        a, b = hh.fold_model(clean, train, (0,)), hh.fold_model(dirty, train, (0,))
    finally:
        hh.TEMPERATURES = saved
    assert a["gamma"] == b["gamma"] and a["temps"] == b["temps"]
    for key in ("center", "scale", "mu", "within", "between"):
        assert np.array_equal(a["parts"][key], b["parts"][key])
    assert all(np.array_equal(a["prior"][k], b["prior"][k]) for k in hh.K_BUCKETS)


def test_bootstrap_keeps_equal_k_definition():
    rng = np.random.default_rng(0)
    k, player = np.repeat(hh.K_BUCKETS, 200), rng.integers(0, 50, 800)
    diff = np.where(k == 5, 0.02, 0.0) + rng.normal(0, 1e-4, 800)
    out = hh.bootstrap(diff, k, player, np.random.default_rng(1))
    assert abs(np.mean(out["bootstrap_95"]) - 0.005) < 5e-4 and out["prob_positive"] == 1.0
