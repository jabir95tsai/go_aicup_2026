import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.cvec2_policy.full_cache import JointCache, a_config
from go_aicup.tasks.strength.policy_large.cache import PolicyCache, Shard, profile_offsets


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



def test_a_teacher_uses_raw_profile_order_microbatches_and_side_offsets():
    from go_aicup.features.humansl import PROFILES

    config = a_config()
    assert tuple(config["profiles_H"]) == PROFILES
    assert config["inference_positions"] == {"H": 128, "C": 256}
    assert config["cache_fraction"] == .85

    q = torch.randn(7, 25)
    side = torch.tensor([0, 1, 0, 1, 1, 0, 1])
    for family in (q[:, :15], q[:, 15:]):
        direct = family - family.log_softmax(-1)
        reference = torch.stack([direct[side == player_side, 0].mean() for player_side in (0, 1)])
        torch.testing.assert_close(profile_offsets(family, side), reference)


def test_h15_joint_view_accepts_stricter_h_mask_and_rejects_reverse_difference():
    from go_aicup.tasks.strength.cvec2_policy.encoder import build

    h, c = cache(15), cache(10)
    h.offsets = c.offsets = np.zeros((2, 2), dtype=np.float32)
    for shard in h.shards:
        shard.board = shard.target[:, None].expand(-1, 91).byte().clone()
    h.shards[0].logp[0, :, 9] = -torch.inf
    joint = JointCache(h, c).for_device("cpu")
    lp, target, q, slot, board = (torch.cat(parts) for parts in zip(*joint.chunks([1, 0], 3, board=True)))
    assert lp.shape == (5, 25, 362)
    assert q.shape == (5, 25)
    assert slot.tolist() == [0, 1, 0, 2, 3]
    assert board.shape == (5, 91)
    torch.testing.assert_close(board[:, 0], target.byte())
    assert torch.isfinite(lp[3, 15:, 9]).all()
    p, features, legal = build("CNN-R10", True, 2027, raw_a=True).policy_inputs(lp, target, q)
    assert not legal[3, 9]
    assert not p[3, :, 9].any() and not features[3, :, 9].any()
    c.shards[0].logp[0, :, 10] = -torch.inf
    with pytest.raises(ValueError, match="outside the C10 mask"):
        JointCache(h, c)
