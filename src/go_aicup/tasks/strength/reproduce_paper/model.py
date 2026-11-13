"""Three-head SE architecture checked against author commit d9498964.

Dedicated blocks preserve post-add ReLU and convolution biases; shared V1/V2
blocks have different semantics. Layers retain PyTorch's default initialization.
"""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.layers(x))


def scalar_head(channels):
    return nn.Sequential(
        nn.Conv2d(channels, 1, 1), nn.BatchNorm2d(1), nn.ReLU(), nn.Flatten(1),
        nn.Linear(361, 256), nn.ReLU(), nn.Linear(256, 1),
    )


class StrengthEstimator(nn.Module):
    """Input [B,18,19,19] or [G,10,7,18,19,19]; retain leading axes.

    Outputs: score/value [...], policy_logits [...,362]. Rank axes are weak to
    strong. No learned weighting head: the confirmed SE setting is unweighted.
    """
    def __init__(self, channels=256, blocks=20, policy_size=362,
                 input_encoding='pre_action_own_opponent'):
        super().__init__()
        if type(channels) is not int or channels < 1 or type(blocks) is not int or blocks < 1:
            raise ValueError('channels and blocks must be positive integers')
        if policy_size != 362 or input_encoding != 'pre_action_own_opponent':
            raise ValueError('SE requires 362 actions and pre-action own/opponent inputs')
        self.stem = nn.Sequential(nn.Conv2d(18, channels, 3, padding=1),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(), nn.Flatten(1),
            nn.Linear(722, 362),
        )
        self.value = nn.Sequential(*scalar_head(channels), nn.Tanh())
        self.score = scalar_head(channels)

    def forward(self, x):
        if (x.ndim not in (4, 6) or x.shape[-3:] != (18, 19, 19)
                or (x.ndim == 6 and x.shape[1:3] != (10, 7))
                or x.shape[0] == 0 or not x.is_floating_point()):
            raise ValueError('expected floating [B,18,19,19] or [G,10,7,18,19,19]')
        shape = x.shape[:-3]
        features = self.blocks(self.stem(x.reshape(-1, 18, 19, 19)))
        return {'score': self.score(features).reshape(shape),
                'policy_logits': self.policy(features).reshape(*shape, 362),
                'value': self.value(features).reshape(shape)}
