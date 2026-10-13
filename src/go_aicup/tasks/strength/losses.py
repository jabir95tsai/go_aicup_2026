"""FP32 global rank means and strong-to-weak listwise ranking loss.

Differentiable SUM contributes world_size times each local gradient; DDP's
parameter-gradient average cancels that factor. Do not scale the loss again.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce


class MissingRanksError(ValueError):
    pass


def distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank_statistics(scores, rank_ids, valid=None, *, synchronize: bool = True,
                    differentiable: bool = True):
    scores = scores.reshape(-1).float()
    rank_ids = rank_ids.reshape(-1).long()
    if scores.shape != rank_ids.shape:
        raise ValueError("scores and rank_ids must have equal lengths")
    if valid is not None:
        mask = valid.reshape(-1).bool()
        if mask.shape != scores.shape:
            raise ValueError("valid mask must match scores")
        scores, rank_ids = scores[mask], rank_ids[mask]
    bad = ((rank_ids < 0) | (rank_ids > 9)).any() | (~torch.isfinite(scores)).any()
    bad = bad.to(torch.int32)
    if synchronize and distributed():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError("invalid rank IDs or nonfinite scores on at least one process")
    sums = torch.zeros(10, device=scores.device, dtype=torch.float32).scatter_add(0, rank_ids, scores)
    counts = torch.bincount(rank_ids, minlength=10).to(dtype=torch.float32)
    if synchronize and distributed():
        if differentiable:
            sums = all_reduce(sums, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return sums, counts


def ranking_loss_from_means(means: torch.Tensor) -> torch.Tensor:
    if means.shape != (10,):
        raise ValueError("means must have shape [10], weak-to-strong")
    if not torch.isfinite(means).all():
        raise ValueError("rank means must be finite")
    strong_first = means.float().flip(0)
    suffix = torch.logcumsumexp(strong_first.flip(0), dim=0).flip(0)
    return (suffix[:-1] - strong_first[:-1]).mean()


def global_rank_loss(scores, rank_ids, valid=None, *, synchronize: bool = True):
    sums, counts = rank_statistics(scores, rank_ids, valid, synchronize=synchronize)
    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise MissingRanksError(f"global batch is missing ranks {missing}")
    means = sums / counts
    return ranking_loss_from_means(means), means, counts
