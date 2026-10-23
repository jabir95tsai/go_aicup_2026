"""Competition-1 insurance model: board encoder -> per-game vector -> GE2E (D-002 D2).

Each judged (black) move at ply >= MIN_PLY is encoded by the A-024 V2 ResNet
trunk (stem + residual blocks; the 1-channel strength head is dropped), pooled
over the board (mean and max) and projected to an embedding. A game vector is
the mean of its move embeddings; plies before MIN_PLY never enter, so the
model cannot use the opening fingerprint by construction.

GE2E (Wan et al. 2018, softmax variant): for N players x M games, normalised
game vectors e_ji, centroids c_k (own centroid excludes e_ji),
S_ji,k = w cos(e_ji, c_k) + b, loss = cross entropy of S_ji over k.

Batches are drawn statelessly from (seed, rank, step), so a resumed run sees
exactly the batches it would have seen.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.tasks.strength.v2.model import StrengthResNet

MIN_PLY = 20


class IdentityEncoder(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 20, embedding: int = 256,
                 bn_eps: float = 1e-5, bn_momentum: float = 0.1):
        super().__init__()
        trunk = StrengthResNet(channels, blocks, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.stem, self.blocks = trunk.stem, trunk.blocks
        self.project = nn.Sequential(nn.Linear(2 * channels, 2 * channels), nn.ReLU(inplace=False),
                                     nn.Linear(2 * channels, embedding))
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def load_trunk(self, state: dict) -> None:
        """Copy stem/blocks weights from a StrengthResNet state dict (A-024 model.pt)."""
        own = {k: v for k, v in state.items() if k.startswith(("stem.", "blocks."))}
        missing = self.load_state_dict(own, strict=False)
        if any(k.startswith(("stem.", "blocks.")) for k in missing.missing_keys) or missing.unexpected_keys:
            raise ValueError("trunk state does not match the encoder")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.project(pooled)


def ge2e_loss(games: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
    """games [N, M, D] game vectors -> (loss, accuracy of the own-centroid argmax)."""
    n, m, _ = games.shape
    e = F.normalize(games.float(), dim=-1)
    total = e.sum(dim=1)                                            # [N, D]
    centroids = F.normalize(total / m, dim=-1)
    exclusive = F.normalize((total[:, None] - e) / (m - 1), dim=-1)  # [N, M, D]
    sim = torch.einsum("jid,kd->jik", e, centroids)                 # [N, M, N]
    own = (e * exclusive).sum(-1)                                   # [N, M]
    index = torch.arange(n, device=games.device)
    sim[index, :, index] = own
    logits = w.clamp(min=1e-6) * sim + b
    target = index[:, None].expand(n, m)
    loss = F.cross_entropy(logits.reshape(n * m, n), target.reshape(-1))
    accuracy = (logits.argmax(-1) == target).float().mean()
    return loss, accuracy


def own_plies(length: int) -> np.ndarray:
    """Black (judged) plies from MIN_PLY on: MIN_PLY, MIN_PLY + 2, ..."""
    return np.arange(MIN_PLY, length, 2)


class GE2EBatches(torch.utils.data.Dataset):
    """Item `step` = one GPU batch: planes uint8 [N*M*S,18,19,19], N players x M games x S moves."""

    def __init__(self, cache_dir, player_games: list[np.ndarray], lengths: np.ndarray, *, players: int,
                 games: int, moves: int, steps: int, seed: int, rank: int, start_step: int = 0,
                 max_open_shards: int = 8):
        self.cache_dir, self.player_games, self.lengths = cache_dir, player_games, lengths
        self.shape = (players, games, moves)
        self.steps, self.seed, self.rank, self.start = steps, seed, rank, start_step
        self.max_open_shards = max_open_shards
        self._cache = None

    def __len__(self):
        return self.steps - self.start

    def sample(self, step: int):
        n, m, s = self.shape
        rng = np.random.default_rng([self.seed, self.rank, step])
        chosen = rng.choice(len(self.player_games), size=n, replace=False)
        game_index, plies = [], []
        for player in chosen:
            for game in rng.choice(self.player_games[player], size=m, replace=False):
                candidates = own_plies(int(self.lengths[game]))
                game_index.append(np.full(s, game))
                plies.append(rng.choice(candidates, size=s, replace=len(candidates) < s))
        return np.concatenate(game_index), np.concatenate(plies)

    def __getitem__(self, item: int):
        if self._cache is None:
            self._cache = BoardCache(self.cache_dir, max_open_shards=self.max_open_shards)
        games, plies = self.sample(self.start + item)
        histories, colors = self._cache.history_batch(games, plies)
        if np.any(colors != 0):
            raise ValueError("GE2E sample is not a black move")
        return torch.from_numpy(assemble_inputs(histories, colors))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state
