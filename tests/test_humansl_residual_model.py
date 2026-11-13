"""CPU checks for stage isolation, pre-action heads and equal-game objectives."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.tasks.strength.humansl_residual_b2.loss import grouped_losses
from go_aicup.tasks.strength.humansl_residual_b2.model import ResidualClassifier


class SyntheticTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.bn = nn.BatchNorm2d(1)

    def forward_features(self, spatial, global_inputs, metadata):
        return self.bn(spatial[:, :1]).repeat(1, 384, 1, 1) * self.scale


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(n=4):
    torch.manual_seed(62)
    return (torch.randn(n, 1, 19, 19), torch.zeros(n, 2), torch.zeros(n, 3),
            torch.arange(n) * 7, torch.arange(n) % 2)


def model(stage=1):
    return ResidualClassifier(SyntheticTrunk(), stage=stage, channels=2, blocks=1)


def test_stage_one_rank_and_value_both_update_shared_projection():
    net = model()
    before = {k: v.clone() for k, v in net.humansl.state_dict().items()}
    out = net(*inputs())
    assert out['rank_logits'].shape == (4, 10)
    assert out['value'].shape == (4,)
    rank_gradient = torch.autograd.grad(out['rank_logits'].square().mean(),
                                        net.projection[0].weight, retain_graph=True)[0]
    value_gradient = torch.autograd.grad(out['value'].square().mean(),
                                         net.projection[0].weight)[0]
    assert rank_gradient.abs().sum() > 0
    assert value_gradient.abs().sum() > 0
    assert not net.humansl.training
    assert all(not p.requires_grad and p.grad is None for p in net.humansl.parameters())
    assert all(torch.equal(before[k], v) for k, v in net.humansl.state_dict().items())


def test_stage_two_zero_residual_and_frozen_stage_one_statistics():
    first, second = model(), model(2)
    second.load_stage1(first.state_dict())
    first.eval()
    expected = first(*inputs())['rank_logits']
    second.train()
    frozen = {k: v.clone() for k, v in second.state_dict().items()
              if k.startswith(('humansl.', 'projection.', 'rank_heads.', 'value_head.'))}
    out = second(*inputs())
    assert torch.equal(out['rank_logits'], expected)
    assert out['policy_logits'].shape == (4, 362)
    (out['rank_logits'].square().mean() + out['policy_logits'].square().mean()
     + out['value'].square().mean()).backward()
    assert second.blocks[0].layers[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for m in second._frozen_modules() for p in m.parameters())
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in frozen.items())
    assert all(not m.training for m in second._frozen_modules())


@pytest.mark.parametrize('stage', [1, 2])
def test_action_changes_classification_only(stage):
    net = model(stage).eval()
    args = inputs()
    if stage == 2:
        with torch.no_grad():
            for head in net.residual_heads:
                head.weight.normal_()
    out = net(*args)
    changed = net(*args[:3], (args[3] + 100) % 361, args[4])
    assert not torch.equal(out['rank_logits'], changed['rank_logits'])
    assert torch.equal(out['value'], changed['value'])
    if stage == 2:
        assert torch.equal(out['policy_logits'], changed['policy_logits'])
    with pytest.raises(ValueError, match='pass unsupported'):
        net(*args[:3], torch.full((4,), 361), args[4])


def test_checkpoint_identity_and_strict_stage_one_copy():
    first, second = model(), model(2)
    second.load_stage1(first.state_dict())
    clone = model(2)
    clone.load_state_dict(second.state_dict())
    with pytest.raises(ValueError, match='identity'):
        first.load_state_dict(second.state_dict())
    broken = first.state_dict()
    del broken['projection.0.weight']
    with pytest.raises(ValueError, match='keys'):
        second.load_stage1(broken)


def loss_inputs(dtype=torch.float32):
    torch.manual_seed(8)
    out = {'rank_logits': torch.randn(60, 10, dtype=dtype, requires_grad=True),
           'policy_logits': torch.randn(60, 362, dtype=dtype, requires_grad=True),
           'value': torch.zeros(60, dtype=dtype, requires_grad=True)}
    return out, torch.tensor([3, 6]), torch.linspace(-1, 1, 60), torch.arange(60)


@pytest.mark.parametrize('stage', [1, 2])
def test_losses_match_manual_game_side_means_and_microbatch_weighting(stage):
    out, labels, targets, actions = loss_inputs()
    losses = grouped_losses(out, labels, targets, actions, stage=stage, games=2)
    rank = F.cross_entropy(out['rank_logits'], labels.repeat_interleave(30), reduction='none')
    value = (out['value'] - targets).square()
    policy = F.cross_entropy(out['policy_logits'], actions, reduction='none')
    reference_rank = sum(rank[g * 30 + s * 15:g * 30 + (s + 1) * 15].mean()
                         for g in range(2) for s in range(2)) / 4
    assert torch.allclose(losses['rank'], reference_rank)
    expected = (2 if stage == 1 else 3) * rank.mean() + value.mean()
    if stage == 2:
        expected = expected + policy.mean()
    assert torch.allclose(losses['total'], expected)
    split = sum(grouped_losses({k: v[s:s + 30] for k, v in out.items()},
                               labels[s // 30:s // 30 + 1], targets[s:s + 30],
                               actions[s:s + 30], stage=stage)['total'] / 2
                for s in (0, 30))
    assert torch.allclose(losses['total'], split)
    whole_grad = torch.autograd.grad(losses['total'], out['rank_logits'], retain_graph=True)[0]
    split_grad = torch.autograd.grad(split, out['rank_logits'])[0]
    assert torch.allclose(whole_grad, split_grad)


def test_bfloat16_loss_is_fp32_and_invalid_teacher_or_labels_rejected():
    out, labels, targets, actions = loss_inputs(torch.bfloat16)
    losses = grouped_losses(out, labels, targets, actions, stage=2)
    assert all(v.dtype == torch.float32 and torch.isfinite(v) for v in losses.values())
    targets[12] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        grouped_losses(out, labels, targets, actions, stage=2)
    targets[12] = 0
    per_position = labels.repeat_interleave(30)
    per_position[15] = 5
    with pytest.raises(ValueError, match='same target player'):
        grouped_losses(out, per_position, targets, actions, stage=2)
    with pytest.raises(ValueError, match='complete games'):
        grouped_losses({k: v[:-1] for k, v in out.items()}, per_position[:-1],
                       targets[:-1], actions[:-1], stage=2)
