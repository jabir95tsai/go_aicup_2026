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
from go_aicup.tasks.strength.shared.dataset import apply_d4
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


def own_plies(length: int, color: int = 0, min_ply: int = MIN_PLY) -> np.ndarray:
    """The plies of `color` (0 black, 1 white) from min_ply on; default = black from MIN_PLY."""
    return np.arange(min_ply + (color - min_ply) % 2, length, 2)


def as_black(histories: np.ndarray, colors: np.ndarray):
    """Exchange the stone planes of white moves and mark every move as black to play."""
    white = colors == 1
    histories[white] = histories[white][:, :, ::-1]
    return histories, np.zeros_like(colors)


class GE2EBatches(torch.utils.data.Dataset):
    """Item `step` = one GPU batch: planes uint8 [N*M*S,18,19,19], N players x M games x S moves."""

    def __init__(self, cache_dir, player_games: list[np.ndarray], lengths: np.ndarray, *, players: int,
                 games: int, moves: int, steps: int, seed: int, rank: int, start_step: int = 0,
                 max_open_shards: int = 8, player_color: np.ndarray | None = None, min_ply: int = MIN_PLY,
                 player_d4: bool = False, world: int = 1, extra_games: list[np.ndarray] | None = None,
                 extra_per_group: int = 0):
        # player_color[game] (0 black, 1 white): sample that colour's moves, white ones colour-swapped
        # (added test set, 2026-11-12). None = the original black-only sampling.
        # player_d4: every board of one sampled player gets the same random D4 transform, so a
        # player seen under 8 orientations acts as 8 training players while the within-player
        # orientation habits stay consistent (2026-11-13).
        self.player_color, self.min_ply, self.player_d4 = player_color, min_ply, player_d4
        # world > 1: the ranks' groups are pooled into one GE2E group (train_ge2e cross_gpu), so the
        # players of a step are drawn once for all ranks and never repeat across ranks.
        self.world = world
        # extra_games: a second list of players (candidate galleries, 2026-11-14); every group
        # holds exactly extra_per_group of them and players - extra_per_group ordinary players.
        self.extra_games, self.extra_per_group = extra_games or [], int(extra_per_group)
        if self.extra_per_group and (world > 1 or not 0 < self.extra_per_group < players
                                     or self.extra_per_group > len(self.extra_games)):
            raise ValueError("extra_per_group needs world == 1 and 0 < extra_per_group < players <= extra players")
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
        if self.world > 1:
            shared = np.random.default_rng([self.seed, step, 7])
            chosen = shared.choice(len(self.player_games), size=n * self.world,
                                   replace=False)[self.rank * n:(self.rank + 1) * n]
        elif self.extra_per_group:
            e = self.extra_per_group
            chosen = rng.choice(len(self.player_games), size=n - e, replace=False)
            extra = rng.choice(len(self.extra_games), size=e, replace=False)
        else:
            chosen = rng.choice(len(self.player_games), size=n, replace=False)
        groups = [self.player_games[p] for p in chosen]
        if self.extra_per_group:
            groups += [self.extra_games[p] for p in extra]
        game_index, plies = [], []
        for games in groups:
            for game in rng.choice(games, size=m, replace=False):
                color = 0 if self.player_color is None else int(self.player_color[game])
                candidates = own_plies(int(self.lengths[game]), color, self.min_ply)
                game_index.append(np.full(s, game))
                plies.append(rng.choice(candidates, size=s, replace=len(candidates) < s))
        return np.concatenate(game_index), np.concatenate(plies)

    def __getitem__(self, item: int):
        if self._cache is None:
            self._cache = BoardCache(self.cache_dir, max_open_shards=self.max_open_shards)
        games, plies = self.sample(self.start + item)
        histories, colors = self._cache.history_batch(games, plies)
        if self.player_color is None:
            if np.any(colors != 0):
                raise ValueError("GE2E sample is not a black move")
        else:
            if not np.array_equal(colors, self.player_color[games]):
                raise ValueError("GE2E sample is not the player's move")
            histories, colors = as_black(histories, colors)
        planes = assemble_inputs(histories, colors)
        if self.player_d4:
            n, m, s = self.shape
            transforms = np.random.default_rng([self.seed, self.rank, self.start + item, 4]).integers(8, size=n)
            blocks = planes.reshape(n, m * s, *planes.shape[1:])
            planes = np.concatenate([apply_d4(block, int(t)) for block, t in zip(blocks, transforms)])
        return torch.from_numpy(planes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state
