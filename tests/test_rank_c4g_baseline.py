import numpy as np
import torch

import scripts.strength.rank_c4g_baseline as bl


def test_baseline_terms_masked_and_centred():
    torch.manual_seed(0)
    n, ranks, area = 7, 10, 30
    logits = torch.randn(n * ranks, area) * 3
    legal = (torch.rand(n, area) > 0.4).repeat_interleave(ranks, 0)
    legal[:, -1] = True
    target = torch.stack([torch.nonzero(row)[0, 0] for row in legal])
    raw, base, checks = bl.baseline_terms(logits, legal, target, ranks)
    assert checks["finite"] and max(checks["policy_sum_error"], checks["q_sum_error"]) < 1e-9
    assert checks["centred_expectation"] < 1e-9
    p = torch.softmax(logits.double().masked_fill(~legal, float("-inf")), -1).reshape(n, ranks, area)
    q = p.mean(1)
    entropy = -(q * torch.where(q > 0, q.log(), torch.zeros_like(q))).sum(-1)
    ratio = torch.where(p > 0, q[:, None, :] / p, torch.ones_like(p)).log()
    kl = (q[:, None, :] * ratio).sum(-1)
    assert torch.allclose(-base, entropy[:, None] + kl, atol=1e-9)        # baselined = raw + H(q) + KL(q || p_r)
    assert torch.allclose(raw, p.log().reshape(n * ranks, area).gather(1, target[:, None])[:, 0].reshape(n, ranks))


def test_shard_parser():
    assert bl._shards("0") == [0] and bl._shards("1-7") == list(range(1, 8)) and bl._shards("0,3-4") == [0, 3, 4]
