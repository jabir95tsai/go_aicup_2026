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
    def __init__(self, architecture: str, full_policy: bool, raw_a: bool = False):
        super().__init__()
        self.architecture = architecture
        self.full_policy = full_policy
        self.raw_a = raw_a
        self.h_profiles = 15 if raw_a else H
        profiles = self.h_profiles + C
        if architecture == 'R10':
            self.phi = nn.Sequential(nn.Linear(2 * profiles + 2, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32))
            self.scorer = nn.Sequential(nn.Linear(64 + profiles, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.zeros_(self.scorer[-1].weight)
            nn.init.zeros_(self.scorer[-1].bias)
            width = profiles + 65
        elif architecture == 'CNN-R10':
            self.conv = nn.Sequential(nn.Conv2d(3 * profiles, 32, 5), nn.ReLU(),
                                      nn.Conv2d(32, 32, 5), nn.ReLU(),
                                      nn.Conv2d(32, 32, 5), nn.ReLU())
            width = 32 + 3 * profiles + 2
        else:
            raise ValueError(f'unknown policy architecture: {architecture}')
        self.hidden = nn.Sequential(nn.Linear(width, 64), nn.ReLU())
        self.h = nn.Linear(64, self.h_profiles)
        self.c = nn.Linear(64, C)
        for head in (self.h, self.c):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def policy_inputs(self, logp, target, q):
        if self.full_policy:
            if self.raw_a:
                legal = torch.isfinite(logp[:, 0])
                lp = logp.float()
                p = lp.exp().masked_fill(~legal[:, None], 0)
                lp = lp.masked_fill(~legal[:, None], 0)
            else:
                p, lp, legal = probabilities(logp)
            at = target[:, None, None].expand(-1, q.shape[1], 1)
            return p.scatter(2, at, q.exp()[..., None]), lp.scatter(2, at, q[..., None]), legal
        # Broadcast the exact played feature after normalization. Hidden scores cannot affect it.
        legal = torch.isfinite(logp[:, 0])
        lp = q[..., None].masked_fill(~legal[:, None], 0)
        p = q.exp()[..., None].masked_fill(~legal[:, None], 0)
        return p, lp, legal

    def delta(self, logp, target, q, inputs=None):
        p, lp, legal = self.policy_inputs(logp, target, q) if inputs is None else inputs
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
            marker = F.one_hot(target, num_classes=POINTS)[:, None, :361].expand(-1, q.shape[1], -1)
            image = torch.cat((p[..., :361], lp[..., :361], marker), 1).reshape(-1, 3 * q.shape[1], 19, 19)
            board = self.conv(image.contiguous(memory_format=torch.channels_last)).mean((-2, -1))
            features = torch.cat((board, q, q.exp(), lp[..., -1], target.eq(361).float()[:, None], log_count), 1)
        hidden = self.hidden(features)
        return self.h(hidden).float(), self.c(hidden).float()

    def forward(self, logp, target, q, inputs=None):
        """Change relative to the selected baseline, in H9/C10 order."""
        shift_h, shift_c = self.delta(logp, target, q, inputs)
        h, c = q.split((self.h_profiles, C), -1)
        residual_h = F.log_softmax(h + shift_h, -1) - F.log_softmax(h, -1)
        residual_c = shift_c if self.raw_a else F.log_softmax(c + shift_c, -1) - F.log_softmax(c, -1)
        return torch.cat((residual_h, residual_c), -1)


def build(architecture: str, full_policy: bool, seed: int, *, raw_a: bool = False) -> JointEncoder:
    with INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return JointEncoder(architecture, full_policy, raw_a)


def add_side_features(base, residual):
    h_profiles = residual.shape[1] - C
    h, c = residual.split((h_profiles, C), dim=1)
    c_start, flags = 2 * h_profiles, 2 * residual.shape[1]
    return torch.cat((base[:, :c_start] + h.reshape(-1, c_start),
                      base[:, c_start:flags] + c.reshape(-1, 2 * C),
                      base[:, flags:]), dim=1)


@torch.no_grad()
def game_features(model, cache, raw, rows, chunk):
    features = torch.as_tensor(raw[rows], dtype=torch.float32, device=cache.device)
    sums = features.new_zeros((len(rows) * 2, model.h_profiles + C))
    for policy, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            sums.index_add_(0, slot, model(policy, target, q))
    return add_side_features(features, sums / cache.counts(rows)[:, None])


def question_indices(questions):
    unique, inverse = np.unique(np.concatenate(questions), return_inverse=True)
    bounds = np.r_[0, np.cumsum([len(rows) for rows in questions])]
    return unique, [inverse[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def question_ce(features, questions, labels, k, head):
    targets = torch.as_tensor(labels, device=features.device, dtype=torch.long)
    return F.cross_entropy(head(question_mean(features, questions), k), targets)


def backward_questions(model, cache, raw, questions, labels, k, head, chunk, penalty=None):
    """Apply CE and optional head regularization with bounded move activations."""
    unique, local = question_indices(questions)
    features = game_features(model, cache, raw, unique, chunk).requires_grad_()
    ce = question_ce(features, local, labels, k, head)
    loss = ce if penalty is None else ce + penalty
    if not torch.isfinite(loss):
        raise ValueError('nonfinite policy question loss')
    loss.backward()
    grad = features.grad
    c_start = 2 * model.h_profiles
    side_grad = torch.cat((grad[:, :c_start].reshape(-1, model.h_profiles),
                           grad[:, c_start:-2].reshape(-1, C)), 1) / cache.counts(unique)[:, None]
    for policy, target, q, slot in cache.chunks(unique, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            correction = model(policy, target, q)
        correction.backward(side_grad[slot])
    return float(ce.detach())
