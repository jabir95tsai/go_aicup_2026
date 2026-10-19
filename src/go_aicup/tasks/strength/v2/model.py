"""Ten-class logits over the shared strength ResNet; no pretrained weights."""
import torch
from torch import nn

from go_aicup.data.board_cache import RANKS
from go_aicup.tasks.strength.shared.model import StrengthResNetBackbone


class StrengthResNet(StrengthResNetBackbone):
    """Output [B,10] in RANKS order; apply softmax only during inference."""
    class_names = RANKS

    def __init__(self, channels: int = 256, blocks: int = 20,
                 hidden_size: int = 256, bn_eps: float = 1e-5,
                 bn_momentum: float = 0.1):
        super().__init__(channels, blocks, hidden_size, bn_eps, bn_momentum)
        self.head.add_module("6", nn.Linear(hidden_size, len(self.class_names)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
