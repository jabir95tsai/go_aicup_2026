"""FP32 local means; equal local group counts give global means under DDP."""
import torch
from torch.nn import functional as F


def grouped_losses(output, rank_target, policy_target, value_target, value_weight):
    rank, policy, value = (output[k] for k in ('rank_logits', 'policy_logits', 'value'))
    shape = value.shape
    if (value.ndim != 3 or shape[0] < 1 or shape[1:] != (10, 20)
            or rank.shape != (*shape, 10) or policy.shape != (*shape, 362)
            or any(t.shape != shape for t in (rank_target, policy_target, value_target, value_weight))):
        raise ValueError('expected aligned [G,10,20] groups and targets')
    tensors = (rank, policy, value, rank_target, policy_target, value_target, value_weight)
    if any(t.device != value.device for t in tensors):
        raise ValueError('outputs and targets must share a device')
    for target, classes in ((rank_target, 10), (policy_target, 362)):
        if target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('class targets must be integer IDs')
        if torch.any((target < 0) | (target >= classes)):
            raise ValueError('class target out of range')
    if (any(not t.is_floating_point() or not torch.isfinite(t).all() for t in (rank, policy, value, value_target))
            or torch.any(value.abs() > 1) or torch.any(value_target.abs() > 1)
            or not torch.all(value_weight == 1)):
        raise ValueError('invalid logits, value or teacher weight')
    strength = F.cross_entropy(rank.float().reshape(-1, 10), rank_target.long().reshape(-1))
    policy_loss = F.cross_entropy(policy.float().reshape(-1, 362), policy_target.long().reshape(-1))
    value_loss = F.mse_loss(value.float(), value_target.float())
    return dict(total=strength + policy_loss + value_loss, strength=strength, policy=policy_loss, value=value_loss)
