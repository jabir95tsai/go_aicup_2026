import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.cvec2_policy.full_cache import JointCache
from go_aicup.tasks.strength.policy_large.cache import PolicyCache, Shard


def cache(profiles, shift=0):
    games = pd.DataFrame({"game_index": [8, 19], "length": [2, 3]})
    shards = []
    for first, ids in enumerate((torch.arange(2), torch.arange(2, 5))):
        q = (ids[:, None] + torch.arange(profiles)[None] + shift).float()
        logp = q[:, :, None].expand(-1, -1, 362).half()
        shards.append(Shard(first, first + 1, logp, q, ids.short(), (ids % 2).byte()))
    return PolicyCache(games, shards, np.zeros((2, 2 * profiles), np.float32),
                       np.array([[1, 1], [2, 1]]))


def test_joint_cache_preserves_game_order_and_channel_alignment():
    joint = JointCache(cache(9), cache(10, shift=50)).for_device("cpu")
    rows = [1, 0, 1]
    lp, target, q, side = (torch.cat(parts) for parts in zip(*joint.chunks(rows, 3)))
    torch.testing.assert_close(target, torch.tensor([2, 3, 4, 0, 1, 2, 3, 4]))
    torch.testing.assert_close(side, torch.tensor([0, 1, 0, 2, 3, 4, 5, 4]))
    torch.testing.assert_close(q[:, 0], target.float())
    torch.testing.assert_close(q[:, 9], target.float() + 50)
    torch.testing.assert_close(lp[:, :, 0].float(), q)
    torch.testing.assert_close(joint.counts(rows), torch.tensor([2, 1, 1, 1, 2, 1.]))


def test_joint_cache_rejects_misaligned_played_moves():
    c = cache(10)
    c.shards[0].target[1] = 9
    with pytest.raises(ValueError, match="played moves differ"):
        JointCache(cache(9), c)
