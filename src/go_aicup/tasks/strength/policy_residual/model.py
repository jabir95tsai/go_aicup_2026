from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('A0', 'S', 'F', 'F-mask', 'C-conv')


def probabilities(logp):
    """Cached illegal entries are -inf; renormalise FP16 log probabilities in FP32."""
    legal = torch.isfinite(logp[:, 0])
    lp = logp.float().masked_fill(~legal[:, None], -torch.inf)
    lp = lp - lp.logsumexp(-1, keepdim=True)
    return lp.exp(), lp.masked_fill(~legal[:, None], 0), legal


def statistics(p, lp, legal, q):
    mu = (p * lp).sum(-1)
    var = (p * (lp - mu[..., None]).square()).sum(-1)
    top = lp.masked_fill(~legal[:, None], -torch.inf).amax(-1)
    n = legal.sum(-1).float().log()[:, None].expand_as(q)
    return torch.stack((-(p * lp).sum(-1), p.amax(-1), p.square().sum(-1), n,
                        var, q - top, (q - mu) / (var + 1e-6).sqrt()), -1).flatten(1)


def candidate_inputs(p, lp, legal, target, masked=False, q=None):
    n, _, points = p.shape
    visible = legal.clone()
    if masked:
        visible.zero_()
        visible.scatter_(1, target[:, None], True)
    values = torch.cat((p, lp), 1).transpose(1, 2) * visible[..., None]
    if q is not None:
        # Keep the played token identical; FP16 normalisation must not leak hidden candidates.
        values = values.clone()
        values[torch.arange(n, device=p.device), target] = torch.cat((q.exp(), q), 1)
    is_pass = p.new_zeros(n, points, 1)
    is_pass[:, -1] = 1
    return torch.cat((values, is_pass, visible[..., None].to(p.dtype)), -1)


def policy_image(p, lp, target, symmetry=0):
    marker = F.one_hot(target, num_classes=362)[:, None, :361].expand(-1, 10, -1)
    x = torch.cat((p[..., :361], lp[..., :361], marker), 1).reshape(-1, 30, 19, 19)
    if symmetry >= 4:
        x = x.flip(-1)
    return torch.rot90(x, symmetry % 4, (-2, -1))


class Residual(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.register_buffer('mean', torch.zeros(80))
        self.register_buffer('scale', torch.ones(80))
        if arm == 'A0':
            return
        if arm == 'S':
            width = 80
        elif arm in ('F', 'F-mask'):
            self.phi = nn.Sequential(nn.Linear(22, 32), nn.ReLU(), nn.Linear(32, 32))
            self.scorer = nn.Sequential(nn.Linear(74, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            width = 75
        else:
            self.conv = nn.Sequential(nn.Conv2d(30, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
            width = 64
        self.delta = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 10))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self, logp, target, q, symmetry=0):
        if self.arm == 'A0':
            return q
        p, lp, legal = probabilities(logp)
        if self.arm == 'S':
            x = torch.cat((q, statistics(p, lp, legal, q)), 1)
            x = (x - self.mean) / self.scale
        elif self.arm in ('F', 'F-mask'):
            v = self.phi(candidate_inputs(p, lp, legal, target, self.arm == 'F-mask', q))
            chosen = v[torch.arange(len(v), device=v.device), target]
            selected = chosen[:, None].expand(-1, 362, -1)
            evidence = q[:, None].expand(-1, 362, -1)
            weights = self.scorer(torch.cat((v, selected, evidence), -1)).squeeze(-1)
            weights = weights.masked_fill(~legal, -torch.inf).softmax(-1)
            summary = (weights[..., None] * v).sum(1)
            x = torch.cat((q, chosen, summary, legal.sum(-1).float().log()[:, None]), 1)
        else:
            image = policy_image(p, lp, target, symmetry).contiguous(memory_format=torch.channels_last)
            c = self.conv(image).mean((-2, -1))
            x = torch.cat((c, q, p[..., -1], lp[..., -1], target.eq(361).float()[:, None],
                           legal.sum(-1).float().log()[:, None]), 1)
        return q + self.delta(x).float()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(20))
        self.register_buffer('scale', torch.ones(20))
        self.linear = nn.Linear(20, 10)

    def forward(self, x):
        return self.linear((x - self.mean) / self.scale)


def side_mean(values, slot, groups):
    counts = torch.bincount(slot, minlength=groups)
    if (counts == 0).any():
        raise ValueError('empty game side')
    return values.new_zeros(groups, 10).index_add(0, slot, values) / counts[:, None]


def question_mean(features, rows):
    return torch.stack([features[r].mean(0) for r in rows])
