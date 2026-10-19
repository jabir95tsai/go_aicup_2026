"""V1 scalar output head over the shared strength ResNet backbone."""
import torch
from torch import nn

from go_aicup.tasks.strength.shared.model import ResidualBlock, StrengthResNetBackbone


class StrengthResNet(StrengthResNetBackbone):
    def __init__(self, channels: int = 256, blocks: int = 20,
                 hidden_size: int = 256, bn_eps: float = 1e-5,
                 bn_momentum: float = 0.1):
        super().__init__(channels, blocks, hidden_size, bn_eps, bn_momentum)
        self.head.add_module("6", nn.Linear(hidden_size, 1))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).squeeze(-1)
