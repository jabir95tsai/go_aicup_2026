"""SE residual semantics with a ten-class strength head."""
import torch
from torch import nn
from go_aicup.tasks.strength.reproduce_paper.model import ResidualBlock, scalar_head


class StrengthEstimator(nn.Module):
    def __init__(self, channels=256, blocks=18, policy_size=362,
                 input_encoding='pre_action_own_opponent'):
        super().__init__()
        if type(channels) is not int or channels < 1 or type(blocks) is not int or blocks < 1:
            raise ValueError('channels and blocks must be positive integers')
        if policy_size != 362 or input_encoding != 'pre_action_own_opponent':
            raise ValueError('SE3 requires 362 actions and pre-action own/opponent inputs')
        self.stem = nn.Sequential(nn.Conv2d(18, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.rank = scalar_head(channels)
        self.rank[-1] = nn.Linear(256, 10)
        self.policy = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                                    nn.Flatten(1), nn.Linear(722, 362))
        self.value = nn.Sequential(*scalar_head(channels), nn.Tanh())

    def forward(self, x):
        if (x.ndim not in (4, 6) or x.shape[-3:] != (18, 19, 19) or x.shape[0] == 0
                or (x.ndim == 6 and x.shape[1:3] != (10, 30)) or not x.is_floating_point()):
            raise ValueError('expected floating [B,18,19,19] or [G,10,30,18,19,19]')
        shape = x.shape[:-3]
        features = self.blocks(self.stem(x.reshape(-1, 18, 19, 19)))
        return dict(rank_logits=self.rank(features).reshape(*shape, 10),
                    policy_logits=self.policy(features).reshape(*shape, 362),
                    value=self.value(features).reshape(shape))
