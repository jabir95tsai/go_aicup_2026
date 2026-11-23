import copy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.functional import cross_entropy

from go_aicup.tasks.strength.cvec2_policy.encoder import (
    backward_questions,
    build,
    game_features,
)


def inputs():
    generator = torch.Generator().manual_seed(91)
    policy = torch.randn(8, 19, 362, generator=generator)
    policy[:, :, 70] = -torch.inf
    target = torch.tensor([2, 3, 361, 5, 6, 1, 2, 3])
    q = policy.log_softmax(-1).gather(2, target[:, None, None].expand(-1, 19, 1)).squeeze(-1)
    return policy, target, q


@pytest.mark.parametrize('architecture', ['R10', 'CNN-R10'])
def test_common_zero_start_and_hidden_scores(architecture):
    scalar = build(architecture, False, 2027)
    full = build(architecture, True, 2027)
    policy, target, q = inputs()
    for name, value in scalar.state_dict().items():
        torch.testing.assert_close(value, full.state_dict()[name], atol=0, rtol=0)
    for model in (scalar, full):
        torch.testing.assert_close(model(policy, target, q), torch.zeros_like(q), atol=0, rtol=0)

    changed = policy.clone()
    changed[:, :, 55] += 3
    scalar_inputs = scalar.policy_inputs(policy, target, q)
    changed_inputs = scalar.policy_inputs(changed, target, q)
    for before, after in zip(scalar_inputs, changed_inputs):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(full.policy_inputs(policy, target, q)[0], full.policy_inputs(changed, target, q)[0])


class Cache:
    device = torch.device('cpu')
    spans = [(0, 3), (3, 5), (5, 8)]
    sides = torch.tensor([0, 0, 1, 0, 1, 0, 1, 1])

    def __init__(self):
        self.policy, self.target, self.q = inputs()

    def counts(self, rows):
        return torch.cat([torch.bincount(self.sides[a:b], minlength=2) for a, b in
                          (self.spans[row] for row in rows)]).float()

    def chunks(self, rows, chunk):
        indices, slots = [], []
        for ordinal, row in enumerate(rows):
            start, end = self.spans[row]
            indices.extend(range(start, end))
            slots.extend(2 * ordinal + self.sides[start:end])
        for start in range(0, len(indices), chunk):
            ids = indices[start:start + chunk]
            yield self.policy[ids], self.target[ids], self.q[ids], torch.tensor(slots[start:start + chunk])


@pytest.mark.parametrize('architecture', ['R10', 'CNN-R10'])
def test_chunked_question_gradient_matches_direct_autograd(architecture):
    cache = Cache()
    raw = np.random.default_rng(5).normal(size=(3, 40)).astype(np.float32)
    model = build(architecture, True, 7)
    torch.testing.assert_close(game_features(model, cache, raw, np.arange(3), 2), torch.from_numpy(raw),
                               atol=0, rtol=0)
    with torch.no_grad():
        model.h.weight.fill_(0.03)
        model.c.weight.normal_(generator=torch.Generator().manual_seed(33))
    direct = copy.deepcopy(model)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        linear = nn.Linear(40, 10).requires_grad_(False)
    questions = [np.array([0, 2]), np.array([1, 2, 0])]
    labels, k = np.array([2, 4]), np.array([2, 3])

    def head(x, current_k):
        np.testing.assert_array_equal(current_k, k)
        return linear(x)

    chunked_loss = backward_questions(model, cache, raw, questions, labels, k, head, 2)

    residual = direct(cache.policy, cache.target, cache.q)
    features = []
    for row, (start, stop) in enumerate(cache.spans):
        own = residual[start:stop][cache.sides[start:stop] == 0].mean(0)
        opp = residual[start:stop][cache.sides[start:stop] == 1].mean(0)
        features.append(torch.from_numpy(raw[row]) + torch.cat((own[:9], opp[:9], own[9:], opp[9:], torch.zeros(2))))
    stacked = torch.stack(features)
    direct_loss = cross_entropy(head(torch.stack([stacked[rows].mean(0) for rows in questions]), k),
                                torch.from_numpy(labels))
    direct_loss.backward()
    assert chunked_loss == pytest.approx(float(direct_loss.detach()), abs=2e-6)
    for chunked_param, direct_param in zip(model.parameters(), direct.parameters()):
        torch.testing.assert_close(chunked_param.grad, direct_param.grad, atol=2e-6, rtol=2e-5)
