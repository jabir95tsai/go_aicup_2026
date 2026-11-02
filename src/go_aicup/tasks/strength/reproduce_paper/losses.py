"""FP32 losses over complete, equally sized local SE groups.

DDP averages local parameter gradients: with equal group counts on every rank,
local means give the global mean gradient without an extra world-size factor.
These losses do not communicate; logging across ranks belongs to the trainer.
"""
import torch
from torch.nn import functional as F


def grouped_bt_loss(scores):
    """Mean normalized listwise loss; scores [G,10,7], rank 0 weakest."""
    if (scores.ndim != 3 or scores.shape[1:] != (10, 7) or scores.shape[0] < 1
            or not scores.is_floating_point() or not torch.isfinite(scores).all()):
        raise ValueError('expected finite floating scores [G,10,7]')
    means = scores.float().mean(dim=-1)
    # Prefixes in weak-to-strong order equal suffixes in the paper's reverse order.
    terms = torch.logcumsumexp(means, dim=1)[:, 1:] - means[:, 1:]
    return terms.mean()


def grouped_losses(output, policy_target, value_target, value_weight=None):
    """Return differentiable total/bt/policy/value means, all weights fixed at 1.

    Consume StrengthEstimator's grouped outputs and collate_groups' [G,10,7]
    targets. Teacher values are already in player perspective. Missing/quality
    masks are unsupported: optional value_weight must contain only ones.
    """
    scores, logits, values = (output[name] for name in ('score', 'policy_logits', 'value'))
    shape = scores.shape
    if (logits.shape != (*shape, 362) or values.shape != shape
            or policy_target.shape != shape or value_target.shape != shape):
        raise ValueError('outputs and targets must align with [G,10,7] groups')
    tensors = (logits, values, policy_target, value_target)
    if any(tensor.device != scores.device for tensor in tensors):
        raise ValueError('outputs and targets must share a device')
    if policy_target.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError('policy_target must contain integer action IDs')
    if (not logits.is_floating_point() or not values.is_floating_point()
            or not value_target.is_floating_point()
            or not torch.isfinite(logits).all() or not torch.isfinite(values).all()
            or not torch.isfinite(value_target).all()
            or torch.any(value_target.abs() > 1) or torch.any(values.abs() > 1)
            or torch.any((policy_target < 0) | (policy_target > 361))):
        raise ValueError('invalid policy logits/actions or value predictions/targets')
    if value_weight is not None:
        if (value_weight.shape != shape or value_weight.device != scores.device
                or not torch.all(value_weight == 1)):
            raise ValueError('value_weight must be aligned and all ones; masking is unsupported')
    bt = grouped_bt_loss(scores)
    policy = F.cross_entropy(logits.float().reshape(-1, 362), policy_target.long().reshape(-1))
    value = F.mse_loss(values.float(), value_target.float())
    return {'total': bt + policy + value, 'bt': bt, 'policy': policy, 'value': value}
