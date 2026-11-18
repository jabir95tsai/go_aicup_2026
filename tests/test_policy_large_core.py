"""Architecture, shared start, regularization and confirmed patience semantics."""
from dataclasses import asdict
import numpy as np
import pytest
import torch
from torch import nn
from go_aicup.tasks.strength.policy_large import models, b3
from go_aicup.tasks.strength.policy_large.stopping import LossStopper


def test_patience_warmup_ties_reset_and_restore():
    s = LossStopper(2500)
    for step in range(0, 3000, 500):
        assert not s.observe(1., step)
        assert s.bad == 0
    for step in (3000, 3500, 4000, 4500):
        assert not s.observe(1., step)
    assert LossStopper(**asdict(s)).observe(1., 5000)
    assert not s.observe(.9, 5000)
    assert s.bad == 0 and s.best == .9
    for invalid in (float('nan'), float('inf'), -float('inf')):
        with pytest.raises(ValueError):
            s.observe(invalid, 5500)


@pytest.mark.parametrize('family,p,totals', [
    ('H', 9, [190, 10760, 12895, 77639, 79774]),
    ('C', 10, [210, 11005, 13075, 80316, 82386])])
def test_deeper_models_shape_counts_shared_body_and_start(family, p, totals):
    torch.set_num_threads(1)
    module = models.module_for(family)
    initial = (module.Head() if family == 'H' else module.OriginalHead()).state_dict()
    q = torch.randn(3, p) - 4
    target = torch.tensor([2, 361, 8])
    lp = torch.randn(3, p, 362).log_softmax(-1)
    heads, networks = {}, {}
    for arm, total in zip(models.ARMS, totals):
        net = models.build(family, arm, 2027)
        head = models.module_for(family).BranchHead(net.width, initial)
        assert sum(x.numel() for x in net.parameters()) + sum(x.numel() for x in head.parameters()) == total
        evidence = net(lp, target, q)
        assert evidence.shape == (3, net.width)
        features = torch.cat((evidence, evidence), 1)
        heads[arm], networks[arm] = head(features), net
        if arm in ('A0', 'R10', 'CNN-R10'):
            torch.testing.assert_close(evidence, q, atol=0, rtol=0)
        torch.testing.assert_close(heads[arm], heads['A0'], atol=1e-6, rtol=1e-6)
    for r, z, body in [('R10', 'Z32', 'phi'), ('CNN-R10', 'CNN-Z32', 'conv')]:
        for a, b in zip(getattr(networks[r], body).parameters(), getattr(networks[z], body).parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert networks['CNN-R10'].conv(torch.zeros(1, 3*p, 19, 19)).shape[-2:] == (7, 7)


def test_b3_layout_all_arms_and_common_start():
    rng = np.random.default_rng(2)
    raw_h, raw_c = rng.normal(size=(100, 18)).astype('f'), rng.normal(size=(100, 20)).astype('f')
    raw = np.column_stack((raw_h, raw_c))
    y = np.tile(np.arange(10), 10)
    reference = None
    for arm in b3.ARMS:
        if arm in ('Z32', 'CNN-Z32'):
            h = np.concatenate((raw_h.reshape(-1, 2, 9), rng.normal(size=(100, 2, 32)).astype('f')), 2).reshape(100, 82)
            c = np.concatenate((raw_c.reshape(-1, 2, 10), rng.normal(size=(100, 2, 32)).astype('f')), 2).reshape(100, 84)
        else:
            h, c = raw_h + .1, raw_c + .1
        values = b3.assemble(arm, h, c, raw_h, raw_c)
        np.testing.assert_array_equal(values[:, b3.raw_columns(arm)], raw)
        head = b3.Head(values.shape[1])
        head.load_state_dict(b3.initial_state(arm, values, raw, y))
        got = head(torch.as_tensor(values)).detach()
        if reference is None:
            reference = got
        torch.testing.assert_close(got, reference, atol=2e-6, rtol=2e-6)
