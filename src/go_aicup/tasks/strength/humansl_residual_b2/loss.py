"""FP32 equal-game/equal-side losses for complete 2 x 15-position games."""
from __future__ import annotations

import torch
from torch.nn import functional as F

_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
POSITIONS_PER_SIDE = 15
POSITIONS_PER_GAME = 2 * POSITIONS_PER_SIDE


def grouped_losses(outputs, labels, value_targets, actions, *, stage, games=None):
    """Return local game means; flat order is game, side(own/opp), position.

    Each microbatch must contain complete games. The trainer multiplies these
    means by local_games * world_size / global_update_games before backward;
    DDP's gradient averaging then gives the full update's global game mean.
    No collective runs here, so uneven microbatch counts cannot deadlock.
    """
    if type(stage) is not int or stage not in (1, 2):
        raise ValueError('stage must be 1 or 2')
    logits, values = outputs['rank_logits'], outputs['value']
    n = len(logits)
    game_count = n // POSITIONS_PER_GAME
    if (n == 0 or n % POSITIONS_PER_GAME or logits.shape != (n, 10)
            or values.shape != (n,) or value_targets.shape != (n,)
            or actions.shape != (n,) or labels.shape not in ((n,), (game_count,))
            or (games is not None and (type(games) is not int or games != game_count))):
        raise ValueError('expected complete games with 2 sides x 15 positions')
    if any(t.device != logits.device for t in (values, value_targets, actions, labels)):
        raise ValueError('loss tensors must share a device')
    if (labels.dtype not in _INTEGER_DTYPES or actions.dtype not in _INTEGER_DTYPES
            or torch.any((labels < 0) | (labels > 9))
            or torch.any((actions < 0) | (actions >= 361))):
        raise ValueError('invalid rank labels or board actions (pass unsupported)')
    if (any(not t.is_floating_point() or not torch.isfinite(t).all()
            for t in (logits, values, value_targets))
            or torch.any(values.abs() > 1) or torch.any(value_targets.abs() > 1)):
        raise ValueError('predictions/teacher must be finite with value in [-1,1]')
    if labels.shape == (game_count,):
        labels = labels.repeat_interleave(POSITIONS_PER_GAME)
    if not torch.all(labels.reshape(game_count, POSITIONS_PER_GAME) ==
                     labels.reshape(game_count, POSITIONS_PER_GAME)[:, :1]):
        raise ValueError('both branches must predict the same target player rank per game')

    def mean_games(per_position):
        return per_position.reshape(game_count, 2, POSITIONS_PER_SIDE).mean(2).mean(1).mean()

    rank = mean_games(F.cross_entropy(logits.float(), labels.long(), reduction='none'))
    value = mean_games(F.mse_loss(values.float(), value_targets.float(), reduction='none'))
    policy = rank.new_zeros(())
    if stage == 2:
        policy_logits = outputs['policy_logits']
        if (policy_logits.shape != (n, 362) or policy_logits.device != logits.device
                or not policy_logits.is_floating_point() or not torch.isfinite(policy_logits).all()):
            raise ValueError('invalid stage-two policy logits')
        policy = mean_games(F.cross_entropy(policy_logits.float(), actions.long(), reduction='none'))
    return {'total': (2 if stage == 1 else 3) * rank + value + policy,
            'rank': rank, 'value': value, 'policy': policy}
