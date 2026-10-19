"""V2 logits/CE contracts and actual DDP parameter-gradient equivalence."""
import math
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from go_aicup.tasks.strength.v2.losses import global_cross_entropy
from go_aicup.tasks.strength.v2.model import StrengthResNet


def test_model_shape_class_order_and_production_parameter_count():
    torch.set_num_threads(1)
    model = StrengthResNet(channels=2, blocks=1, hidden_size=4).eval()
    assert model.class_names == ('D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D')
    logits = model(torch.zeros(1, 18, 19, 19))
    assert logits.shape == (1, 10)
    torch.testing.assert_close(logits.softmax(-1).sum(-1), torch.ones(1))
    with torch.device('meta'):
        production = StrengthResNet()
    assert sum(p.numel() for p in production.parameters()) == 23750924
    with pytest.raises(ValueError, match='expected input'):
        model(torch.zeros(1, 17, 19, 19))


def test_ce_fp32_sanity_mask_and_gradient():
    logits = torch.zeros(3, 10, requires_grad=True)
    labels = torch.tensor([0, 9, -1])
    mask = torch.tensor([True, True, False])
    loss, total, count = global_cross_entropy(logits, labels, mask)
    assert loss.item() == pytest.approx(math.log(10))
    assert total.item() == pytest.approx(2 * math.log(10))
    assert count.item() == 2
    loss.backward()
    assert torch.count_nonzero(logits.grad[2]).item() == 0
    improved = logits.detach().clone()
    improved[0, 0] = improved[1, 9] = 2
    improved[2] = float('nan')  # Padding must be excluded before CE.
    assert global_cross_entropy(improved, labels, mask)[0] < loss
    low_precision = torch.zeros(2, 10, dtype=torch.bfloat16, requires_grad=True)
    fp32_loss = global_cross_entropy(low_precision, labels[:2])[0]
    assert fp32_loss.dtype == torch.float32
    torch.testing.assert_close(fp32_loss, F.cross_entropy(low_precision.float(), labels[:2]))
    fp32_loss.backward()
    assert low_precision.grad is not None


@pytest.mark.parametrize('labels,mask', [
    (torch.tensor([10]), None),
    (torch.tensor([-1]), None),
    (torch.tensor([0]), torch.tensor([False])),
])
def test_invalid_or_globally_empty_samples_rejected(labels, mask):
    with pytest.raises(ValueError):
        global_cross_entropy(torch.zeros(1, 10), labels, mask)


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(2027)
        model = DDP(torch.nn.Linear(3, 10))
        x = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10
        labels = torch.tensor([0, 0, 9, 4, 4])
        # Unequal valid counts, zero-valid tail process, then physically empty input.
        for scenario in ('unequal', 'masked_empty', 'empty'):
            model.zero_grad(set_to_none=True)
            reference = torch.nn.Linear(3, 10)
            reference.load_state_dict(model.module.state_dict())
            if scenario == 'unequal':
                indices = torch.tensor([0, 1, 2] if rank == 0 else [3, 4, 0])
                mask = torch.tensor([True, True, True] if rank == 0 else [True, True, False])
                used = torch.arange(5)
            else:
                indices = torch.tensor([0, 1] if rank == 0 or scenario == 'masked_empty' else [], dtype=torch.long)
                mask = torch.full((len(indices),), rank == 0, dtype=torch.bool)
                used = torch.arange(2)
            logits = model(x[indices])
            loss, total, count = global_cross_entropy(logits, labels[indices], mask)
            expected = F.cross_entropy(reference(x[used]), labels[used])
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(total, expected.detach() * len(used))
            assert count.item() == len(used)
            loss.backward()
            expected.backward()
            for actual, target in zip(model.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, target.grad, atol=1e-6, rtol=1e-5)
        # Synchronization can be disabled even inside an initialized group.
        local = torch.zeros(rank + 1, 10, requires_grad=True)
        _, _, count = global_cross_entropy(local, torch.zeros(rank + 1, dtype=torch.long), synchronize=False)
        assert count.item() == rank + 1
        for bad_mode in ('invalid', 'empty'):
            labels = torch.tensor([10 if rank == 1 and bad_mode == 'invalid' else 0])
            mask = torch.tensor([bad_mode != 'empty'])
            with pytest.raises(ValueError):
                global_cross_entropy(torch.zeros(1, 10), labels, mask)
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_loss_and_gradients_match_single_process(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=((tmp_path / 'gloo').as_uri(),), nprocs=2, join=True)
