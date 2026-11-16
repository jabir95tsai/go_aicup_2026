import numpy as np

import scripts.strength.rank_hc_s_fusion as f


def test_fuse_beta_zero_is_hc_temperature():
    rng = np.random.default_rng(0)
    lp = np.log(rng.dirichlet(np.ones(10), 50))
    other = np.log(rng.dirichlet(np.ones(10), 50))
    assert np.allclose(f._fuse(lp, other, 1.0, 0.0), np.exp(lp))
    assert np.allclose(f._fuse(lp, other, 1.0, 0.0).sum(1), 1)


def test_select_tie_rule_prefers_smaller_beta_then_alpha_near_one():
    lp = np.log(np.tile(np.eye(10)[3] * 0.9 + 0.01, (20, 1)))
    truth = np.full(20, 3)
    alpha, beta = f._select(lp, lp, truth, f.BETAS)      # every candidate scores 1.0
    assert (alpha, beta) == (1.0, 0.0)


def test_select_uses_partner_when_it_helps():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 10, 400)
    hc = np.log(rng.dirichlet(np.ones(10), 400))          # uninformative
    good = np.full((400, 10), 0.01)
    good[np.arange(400), truth] = 0.91
    alpha, beta = f._select(hc, np.log(good), truth, f.BETAS)
    assert beta > 0
