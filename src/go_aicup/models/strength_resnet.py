"""Strength V1 scalar ResNet, as specified in STRENGTH_V1_PLAN.md.

No pretrained weights, pooling, sigmoid or post-addition activation. The
default network has 23,748,611 trainable parameters. Smaller configurations
are explicit development settings, not the production V1 architecture.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, *, bn_eps: float, bn_momentum: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=bn_eps, momentum=bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))


class StrengthResNet(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 20,
                 hidden_size: int = 256, bn_eps: float = 1e-5,
                 bn_momentum: float = 0.1):
        super().__init__()
        if min(channels, blocks, hidden_size) < 1 or bn_eps <= 0:
            raise ValueError("channels, blocks, hidden_size and bn_eps must be positive")
        if not 0 < bn_momentum <= 1:
            raise ValueError("bn_momentum must be in (0,1]")
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels, bn_eps=bn_eps, bn_momentum=bn_momentum)
            for _ in range(blocks)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=False),
            nn.Flatten(start_dim=1),
            nn.Linear(361, hidden_size),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                bound = module.in_features ** -0.5
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (18, 19, 19):
            raise ValueError("expected input [B,18,19,19]")
        return self.head(self.blocks(self.stem(x))).squeeze(-1)
