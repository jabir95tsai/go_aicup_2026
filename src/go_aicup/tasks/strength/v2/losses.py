"""FP32 per-position CE, averaged over globally valid samples.

All DDP processes must call this function and backward, including processes
with zero valid samples. Differentiable SUM multiplies each local gradient
by world_size; DDP's parameter-gradient average cancels that factor. Do not
scale the returned loss by world_size again.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce
from torch.nn import functional as F

from go_aicup.tasks.strength.shared.runtime import distributed


def global_cross_entropy(logits: torch.Tensor, rank_ids: torch.Tensor,
                         valid: torch.Tensor | None = None, *,
                         synchronize: bool = True):
    """Return (differentiable mean, detached FP32 sum, detached int64 count).

    Inputs are raw [B,10] logits and integer labels [B], in D..6D order.
    Optional valid[B] excludes padding before CE and label/finite checks.
    With synchronize=False, statistics and gradients are strictly local.
    Empty local selections are legal; an empty global selection is an error.
    No class weights, label smoothing, rank balancing or missing-rank rule.
    """
    if logits.ndim != 2 or logits.shape[1] != 10 or not logits.is_floating_point():
        raise ValueError("expected floating logits [B,10]")
    if rank_ids.shape != logits.shape[:1] or rank_ids.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("expected integer rank_ids [B]")
    if rank_ids.device != logits.device:
        raise ValueError("rank_ids and logits must be on the same device")
    if valid is not None:
        if valid.shape != rank_ids.shape or valid.dtype != torch.bool or valid.device != logits.device:
            raise ValueError("expected boolean valid [B] on the logits device")
        logits, rank_ids = logits[valid], rank_ids[valid]
    logits, rank_ids = logits.float(), rank_ids.long()
    synchronize = synchronize and distributed()
    bad = ((rank_ids < 0).any() | (rank_ids >= 10).any() | (~torch.isfinite(logits)).any()).int()
    if synchronize:
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError("invalid rank IDs or nonfinite logits on at least one process")
    count = torch.tensor(rank_ids.numel(), dtype=torch.int64, device=logits.device)
    if synchronize:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        raise ValueError("cross entropy requires at least one valid sample globally")
    # Empty [0,10] CE with reduction=sum remains connected to the local model.
    loss_sum = F.cross_entropy(logits, rank_ids, reduction="sum", label_smoothing=0.0)
    if synchronize:
        loss_sum = all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    return loss_sum / count, loss_sum.detach(), count.detach()
