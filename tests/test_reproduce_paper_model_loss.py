"""Synthetic FP32 architecture, listwise formula and two-process DDP checks."""
from datetime import timedelta
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.model import ResidualBlock, StrengthEstimator
from go_aicup.tasks.strength.reproduce_paper.losses import grouped_bt_loss, grouped_losses


@pytest.fixture(scope='module', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_author_architecture_and_config():
    model = StrengthEstimator(**read_config()['model'])
    assert len(model.blocks) == 20 and model.stem[0].out_channels == 256
    assert model.policy[0].out_channels == 2 and model.policy[-1].in_features == 722
    assert model.policy[-1].out_features == 362
    assert model.score[-1].out_features == 1 and model.score[-3].out_features == 256
    assert isinstance(model.value[-1], nn.Tanh)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert module.bias is not None
        elif isinstance(module, nn.BatchNorm2d):
            assert module.eps == 1e-5 and module.momentum == 0.1
    block = ResidualBlock(2).eval()
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
        # A missing post-add activation would preserve negative inputs here.
        assert torch.count_nonzero(block(-torch.ones(1, 2, 19, 19))) == 0


def test_shapes_heads_and_default_initialization():
    torch.manual_seed(4)
    model = StrengthEstimator(channels=2, blocks=1).eval()
    inputs = torch.randn(1, 10, 7, 18, 19, 19)
    output = model(inputs)
    flat = model(inputs.flatten(0, 2))
    for name in output:
        assert output[name].shape == ((1, 10, 7, 362) if name == 'policy_logits' else (1, 10, 7))
        torch.testing.assert_close(output[name].flatten(0, 2), flat[name], rtol=0, atol=0)
    with torch.no_grad():
        model.score[-1].weight.zero_()
        model.score[-1].bias.fill_(3)
        model.value[-2].weight.zero_()
        model.value[-2].bias.fill_(-2)
        bounded = model(inputs)
    assert torch.all(bounded['score'] == 3)  # Strength is not a probability.
    torch.testing.assert_close(bounded['value'], torch.full((1, 10, 7), math.tanh(-2)))
    assert torch.all(output['value'].abs() <= 1)


def test_listwise_direct_formula_normalization_and_group_boundary():
    torch.manual_seed(3)
    scores = torch.randn(3, 10, 7, requires_grad=True)
    means = scores.float().mean(-1).flip(-1)  # Paper order: strongest first.
    direct = sum(torch.logsumexp(means[:, i:], dim=-1) - means[:, i] for i in range(9)).mean() / 9
    actual = grouped_bt_loss(scores)
    assert abs((actual - direct).item()) <= 1e-6
    torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                               torch.autograd.grad(direct, scores)[0], rtol=1e-5, atol=1e-8)
    assert grouped_bt_loss(torch.zeros(1, 10, 7)).item() == pytest.approx(math.lgamma(11) / 9, abs=1e-6)
    ramp = torch.arange(10.).view(1, 10, 1).expand(1, 10, 7)
    assert grouped_bt_loss(ramp) < grouped_bt_loss(-ramp)
    opposite = torch.cat([ramp, -ramp])
    assert grouped_bt_loss(opposite) > grouped_bt_loss(opposite.mean(0, keepdim=True)) + 1
    torch.testing.assert_close(grouped_bt_loss(scores + 10), actual, rtol=1e-6, atol=1e-6)


def test_extreme_scores_and_bfloat16_are_finite_fp32():
    scores = torch.linspace(10000, -10000, 70).reshape(1, 10, 7).requires_grad_()
    loss = grouped_bt_loss(scores)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()
    low_precision = scores.detach().to(torch.bfloat16).requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = grouped_bt_loss(low_precision)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, grouped_bt_loss(low_precision.float()), rtol=0, atol=0)
    result.backward()
    assert torch.isfinite(low_precision.grad).all()


def synthetic_batch():
    torch.manual_seed(26)
    return (torch.randn(4, 10, 7, 18, 19, 19),
            torch.randint(362, (4, 10, 7)), torch.rand(4, 10, 7) * 2 - 1)


def test_three_losses_match_reference_and_all_heads_backpropagate():
    inputs, actions, targets = synthetic_batch()
    torch.manual_seed(7)
    model = StrengthEstimator(channels=2, blocks=1)
    output = model(inputs[:1])
    actions = actions[:1].clone()
    actions[0, 0, 0] = 361  # Pass participates in the same 362-way CE.
    losses = grouped_losses(output, actions, targets[:1], torch.ones_like(targets[:1]))
    torch.testing.assert_close(losses['policy'],
        -(F.one_hot(actions, 362) * output['policy_logits'].log_softmax(-1)).sum(-1).mean())
    torch.testing.assert_close(losses['value'], (output['value'] - targets[:1]).square().mean())
    torch.testing.assert_close(losses['total'], losses['bt'] + losses['policy'] + losses['value'])
    losses['total'].backward()
    for name in ('stem', 'blocks', 'score', 'policy', 'value'):
        grads = [p.grad for p in getattr(model, name).parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert sum(grad.abs().sum().item() for grad in grads) > 0
    low = {key: tensor.detach().to(torch.bfloat16) for key, tensor in output.items()}
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = grouped_losses(low, actions, targets[:1].double())
    assert all(tensor.dtype == torch.float32 and torch.isfinite(tensor) for tensor in result.values())


def _ddp_worker(rank, store, destination):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(7)
        # Frozen BN removes training batch-statistic differences from this loss check.
        model = DistributedDataParallel(StrengthEstimator(channels=2, blocks=1).eval())
        inputs, actions, targets = synthetic_batch()
        rows = slice(rank * 2, rank * 2 + 2)
        losses = grouped_losses(model(inputs[rows]), actions[rows], targets[rows])
        losses['total'].backward()
        metrics = torch.stack([losses[name].detach() for name in ('total', 'bt', 'policy', 'value')])
        dist.all_reduce(metrics)
        if rank == 0:
            torch.save({'metrics': metrics / 2,
                        'grads': {name: p.grad for name, p in model.module.named_parameters()}}, destination)
    finally:
        dist.destroy_process_group()


def test_two_process_group_loss_and_parameter_gradients_match(tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo unavailable')
    torch.manual_seed(7)
    model = StrengthEstimator(channels=2, blocks=1).eval()
    inputs, actions, targets = synthetic_batch()
    losses = grouped_losses(model(inputs), actions, targets)
    losses['total'].backward()
    destination = tmp_path / 'ddp.pt'
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / 'store'), str(destination)), nprocs=2)
    result = torch.load(destination, weights_only=True)
    metrics = torch.stack([losses[name].detach() for name in ('total', 'bt', 'policy', 'value')])
    torch.testing.assert_close(result['metrics'], metrics, rtol=1e-5, atol=1e-7)
    reference = torch.cat([parameter.grad.flatten() for parameter in model.parameters()])
    actual = torch.cat([result['grads'][name].flatten() for name, _ in model.named_parameters()])
    difference = actual - reference
    relative = (difference.norm() / reference.norm()).item()
    absolute = difference.abs().max().item()
    assert relative <= 1e-5 and absolute <= 1e-6
    near_zero = reference.abs() <= 1e-7
    near_zero_absolute = difference[near_zero].abs().max().item() if near_zero.any() else 0.
    assert near_zero_absolute <= 1e-7
    print(f'Gloo gradient vector: relative L2={relative:.3g}, max absolute={absolute:.3g}, '
          f'near-zero max absolute={near_zero_absolute:.3g}')



@pytest.mark.parametrize('shape', [(1, 9, 7), (1, 10, 6), (0, 10, 7), (70,)])
def test_incomplete_score_groups_rejected(shape):
    with pytest.raises(ValueError, match='scores'):
        grouped_bt_loss(torch.zeros(shape))


@pytest.mark.parametrize('kind', ['score_nan', 'logit_inf', 'policy_float', 'policy_out_of_range',
                                  'value_nan', 'target_out_of_range', 'missing_value', 'weight_mask'])
def test_invalid_predictions_and_supervision_rejected(kind):
    output = {'score': torch.zeros(1, 10, 7), 'policy_logits': torch.zeros(1, 10, 7, 362),
              'value': torch.zeros(1, 10, 7)}
    actions, target, weight = torch.zeros(1, 10, 7, dtype=torch.long), torch.zeros(1, 10, 7), None
    if kind == 'score_nan':
        output['score'][0, 0, 0] = float('nan')
    elif kind == 'logit_inf':
        output['policy_logits'][0, 0, 0, 0] = float('inf')
    elif kind == 'policy_float':
        actions = actions.float()
    elif kind == 'policy_out_of_range':
        actions[0, 0, 0] = 362
    elif kind == 'value_nan':
        target[0, 0, 0] = float('nan')
    elif kind == 'target_out_of_range':
        target[0, 0, 0] = -1.1
    elif kind == 'missing_value':
        target = target[..., :6]
    else:
        weight = torch.zeros_like(target)
    with pytest.raises(ValueError):
        grouped_losses(output, actions, target, weight)


@pytest.mark.parametrize('kwargs', [dict(channels=0), dict(blocks=0), dict(policy_size=361),
                                   dict(input_encoding='post_action')])
def test_invalid_model_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        StrengthEstimator(**kwargs)


@pytest.mark.parametrize('shape', [(1, 17, 19, 19), (1, 10, 6, 18, 19, 19), (0, 18, 19, 19)])
def test_invalid_model_input_rejected(shape):
    with pytest.raises(ValueError):
        StrengthEstimator(channels=2, blocks=1)(torch.zeros(shape))
