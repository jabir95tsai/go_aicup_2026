"""Fresh H/C models with one additional candidate MLP or valid 5x5 convolution."""
from threading import RLock
import torch
from torch import nn
from go_aicup.tasks.strength.policy_h import five_arms as h
from go_aicup.tasks.strength.policy_residual import five_arms as c

ARMS = ('A0', 'R10', 'Z32', 'CNN-R10', 'CNN-Z32')
INITIALIZATION_LOCK = RLock()


def module_for(family):
    if family not in ('H', 'C'):
        raise ValueError(f'unknown family {family}')
    return h if family == 'H' else c


def build(family, arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    with INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        model = module_for(family).build(arm, seed)
        torch.manual_seed(seed + 1000)
        if arm in ('R10', 'Z32'):
            if len(model.phi) != 3:
                raise ValueError('unexpected original set architecture')
            model.phi.extend([nn.ReLU(), nn.Linear(32, 32)])
        elif arm.startswith('CNN'):
            if len(model.conv) != 4:
                raise ValueError('unexpected original CNN architecture')
            model.conv.extend([nn.Conv2d(32, 32, 5), nn.ReLU()])
    return model
