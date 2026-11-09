"""Frozen HumanSL features, shared stage-one projection and stage-two correction."""
from __future__ import annotations

import torch
from torch import nn

from go_aicup.tasks.strength.reproduce_paper.model import ResidualBlock, scalar_head

MODEL_VERSION = 1
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class ResidualClassifier(nn.Module):
    """Flat pre-action inputs; sides are 0=own and 1=opponent.

    Production HumanSL exposes KataGo's ExtraOutputs("trunkfinal"). Tests may
    inject a module with forward_features(spatial, global_inputs, metadata).
    Actual actions enter only the two classification readouts.
    """

    def __init__(self, humansl: nn.Module, stage: int = 1, channels=256, blocks=10):
        super().__init__()
        if type(stage) is not int or stage not in (1, 2):
            raise ValueError('stage must be 1 or 2')
        if any(type(n) is not int or n < 1 for n in (channels, blocks)):
            raise ValueError('channels and blocks must be positive integers')
        self.stage, self.channels, self.block_count = stage, channels, blocks
        self.register_buffer('_model_identity', torch.tensor(
            [MODEL_VERSION, stage, channels, blocks if stage == 2 else 0], dtype=torch.int64))
        self.humansl = humansl
        self.projection = nn.Sequential(nn.Conv2d(384, channels, 1),
                                        nn.BatchNorm2d(channels), nn.ReLU())
        self.rank_heads = nn.ModuleList([nn.Linear(2 * channels, 10) for _ in range(2)])
        self.value_head = nn.Sequential(*scalar_head(channels), nn.Tanh())
        if stage == 2:
            self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
            self.residual_heads = nn.ModuleList([nn.Linear(2 * channels, 10) for _ in range(2)])
            for head in self.residual_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                nn.Flatten(1), nn.Linear(722, 362))
            self.residual_value_head = nn.Sequential(*scalar_head(channels), nn.Tanh())
        for module in self._frozen_modules():
            module.requires_grad_(False)
        self.train()

    def _frozen_modules(self):
        return ((self.humansl,) if self.stage == 1 else
                (self.humansl, self.projection, self.rank_heads, self.value_head))

    def train(self, mode=True):
        super().train(mode)
        for module in self._frozen_modules():
            module.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        identity = state_dict.get('_model_identity')
        if identity is None or not torch.equal(identity.cpu(), self._model_identity.cpu()):
            raise ValueError('checkpoint model identity does not match stage/architecture')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def load_stage1(self, state_dict):
        """Strictly copy all stage-one parameters/statistics, leaving correction fresh."""
        if self.stage != 2:
            raise ValueError('load_stage1 requires a stage-two model')
        expected = torch.tensor([MODEL_VERSION, 1, self.channels, 0], dtype=torch.int64)
        identity = state_dict.get('_model_identity')
        if identity is None or not torch.equal(identity.cpu(), expected):
            raise ValueError('source must be a matching stage-one model')
        current = self.state_dict()
        prefixes = ('humansl.', 'projection.', 'rank_heads.', 'value_head.')
        required = {name for name in current if name.startswith(prefixes)}
        if set(state_dict) != required | {'_model_identity'}:
            raise ValueError('stage-one checkpoint keys do not match')
        current.update({name: state_dict[name] for name in required})
        self.load_state_dict(current)
        self.train(self.training)
        return self

    def _features(self, spatial, global_inputs, metadata):
        with torch.no_grad():
            if callable(getattr(self.humansl, 'forward_features', None)):
                features = self.humansl.forward_features(spatial, global_inputs, metadata)
            else:
                from katago.train.model_pytorch import ExtraOutputs
                extra = ExtraOutputs(['trunkfinal'])
                self.humansl(spatial, global_inputs, input_meta=metadata, extra_outputs=extra)
                features = extra.returned['trunkfinal']
        if features.shape != (len(spatial), 384, 19, 19):
            raise ValueError('HumanSL trunkfinal must have shape [N,384,19,19]')
        return features

    @staticmethod
    def _rank(features, heads, actions, sides):
        flattened = features.flatten(2)
        played = flattened.gather(2, actions[:, None, None].expand(-1, features.shape[1], 1)).squeeze(2)
        readout = torch.cat((flattened.mean(2), played), dim=1)
        return torch.where(sides[:, None] == 0, heads[0](readout), heads[1](readout))

    def forward(self, spatial, global_inputs, metadata, actions, sides):
        n = len(spatial)
        if (spatial.ndim != 4 or spatial.shape[-2:] != (19, 19) or n == 0
                or not spatial.is_floating_point() or global_inputs.ndim != 2
                or metadata.ndim != 2 or len(global_inputs) != n or len(metadata) != n
                or actions.shape != (n,) or sides.shape != (n,)):
            raise ValueError('expected flat aligned spatial/global/metadata/actions/sides inputs')
        if any(t.device != spatial.device for t in (global_inputs, metadata, actions, sides)):
            raise ValueError('model inputs must share a device')
        if (actions.dtype not in _INTEGER_DTYPES or sides.dtype not in _INTEGER_DTYPES
                or torch.any((actions < 0) | (actions >= 361))
                or torch.any((sides < 0) | (sides > 1))):
            raise ValueError('actions must be board points 0..360 and sides must be 0/1; pass unsupported')
        actions = actions.long()
        features = self._features(spatial, global_inputs, metadata)
        if self.stage == 1:
            projected = self.projection(features)
            return {'rank_logits': self._rank(projected, self.rank_heads, actions, sides),
                    'value': self.value_head(projected).flatten()}
        with torch.no_grad():
            projected = self.projection(features)
            base_logits = self._rank(projected, self.rank_heads, actions, sides)
        corrected = self.blocks(projected)
        return {'rank_logits': base_logits + self._rank(corrected, self.residual_heads, actions, sides),
                'policy_logits': self.policy_head(corrected),
                'value': self.residual_value_head(corrected).flatten()}
