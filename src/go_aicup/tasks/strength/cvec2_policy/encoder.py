"""Joint H9/C10 residual variants with cvec2 game and question aggregation."""
from __future__ import annotations

from threading import Lock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.tasks.strength.policy_residual.model import candidate_inputs, probabilities, question_mean

H, C = 9, 10
PROFILES, POINTS = H + C, 362
INITIALIZATION_LOCK = Lock()


class JointEncoder(nn.Module):
    def __init__(self, architecture: str, full_policy: bool):
        super().__init__()
        self.architecture = architecture
        self.full_policy = full_policy
        if architecture == 'R10':
            self.phi = nn.Sequential(nn.Linear(2 * PROFILES + 2, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
            self.scorer = nn.Sequential(nn.Linear(64 + PROFILES, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            width = PROFILES + 65
        elif architecture == 'CNN-R10':
            self.conv = nn.Sequential(nn.Conv2d(3 * PROFILES, 32, 5), nn.ReLU(),
                                      nn.Conv2d(32, 32, 5), nn.ReLU(),
                                      nn.Conv2d(32, 32, 5), nn.ReLU())
            width = 32 + 3 * PROFILES + 2
        else:
            raise ValueError(f'unknown policy architecture: {architecture}')
        self.hidden = nn.Sequential(nn.Linear(width, 64), nn.ReLU())
        self.h = nn.Linear(64, H)
        self.c = nn.Linear(64, C)
        for head in (self.h, self.c):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def policy_inputs(self, logp, target, q):
        if self.full_policy:
            p, lp, legal = probabilities(logp)
            at = target[:, None, None].expand(-1, PROFILES, 1)
            return p.scatter(2, at, q.exp()[..., None]), lp.scatter(2, at, q[..., None]), legal
        # Broadcast the exact played feature after normalization. Hidden scores cannot affect it.
        legal = torch.isfinite(logp[:, 0])
        lp = q[..., None].masked_fill(~legal[:, None], 0)
        p = q.exp()[..., None].masked_fill(~legal[:, None], 0)
        return p, lp, legal

    def delta(self, logp, target, q):
        p, lp, legal = self.policy_inputs(logp, target, q)
        log_count = legal.sum(-1).float().log()[:, None]
        if self.architecture == 'R10':
            v = self.phi(candidate_inputs(p, lp, legal, target))
            chosen = v[torch.arange(len(v), device=v.device), target]
            attention = self.scorer(torch.cat((v, chosen[:, None].expand(-1, POINTS, -1),
                                                q[:, None].expand(-1, POINTS, -1)), -1)).squeeze(-1)
            weights = attention.masked_fill(~legal, -torch.inf).softmax(-1)
            summary = (weights[..., None] * v).sum(1)
            features = torch.cat((q, chosen, summary, log_count), 1)
        else:
            marker = F.one_hot(target, num_classes=POINTS)[:, None, :361].expand(-1, PROFILES, -1)
            image = torch.cat((p[..., :361], lp[..., :361], marker), 1).reshape(-1, 3 * PROFILES, 19, 19)
            board = self.conv(image.contiguous(memory_format=torch.channels_last)).mean((-2, -1))
            features = torch.cat((board, q, q.exp(), lp[..., -1], target.eq(361).float()[:, None], log_count), 1)
        hidden = self.hidden(features)
        return self.h(hidden).float(), self.c(hidden).float()

    def forward(self, logp, target, q):
        """Change relative to the selected baseline, in H9/C10 order."""
        delta = self.delta(logp, target, q)
        # Any retained baseline logsumexp cancels from this difference.
        return torch.cat([F.log_softmax(values + shift, -1) - F.log_softmax(values, -1)
                          for values, shift in zip(q.split((H, C), -1), delta)], -1)


def build(architecture: str, full_policy: bool, seed: int) -> JointEncoder:
    with INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return JointEncoder(architecture, full_policy)


@torch.no_grad()
def game_features(model, cache, raw, rows, chunk):
    features = torch.as_tensor(raw[rows], dtype=torch.float32, device=cache.device)
    sums = features.new_zeros((len(rows) * 2, PROFILES))
    for policy, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            sums.index_add_(0, slot, model(policy, target, q))
    means = sums / cache.counts(rows)[:, None]
    h, c = means.split((H, C), dim=1)
    return torch.cat((features[:, :2 * H] + h.reshape(-1, 2 * H),
                      features[:, 2 * H:2 * PROFILES] + c.reshape(-1, 2 * C),
                      features[:, 2 * PROFILES:]), dim=1)


def backward_questions(model, cache, raw, questions, labels, k, head, chunk):
    """Apply the question loss across full games while retaining one move chunk at a time."""
    unique, inverse = np.unique(np.concatenate(questions), return_inverse=True)
    bounds = np.r_[0, np.cumsum([len(rows) for rows in questions])]
    local = [inverse[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    features = game_features(model, cache, raw, unique, chunk).requires_grad_()
    targets = torch.as_tensor(labels, device=cache.device, dtype=torch.long)
    loss = F.cross_entropy(head(question_mean(features, local), k), targets)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite policy question loss')
    (grad,) = torch.autograd.grad(loss, features)
    side_grad = torch.cat((grad[:, :2 * H].reshape(-1, H),
                           grad[:, 2 * H:2 * PROFILES].reshape(-1, C)), 1) / cache.counts(unique)[:, None]
    for policy, target, q, slot in cache.chunks(unique, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            correction = model(policy, target, q)
        correction.backward(side_grad[slot])
    return float(loss.detach())
