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
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.tasks.strength.shared.dataset import apply_d4
from go_aicup.tasks.strength.v2.model import StrengthResNet

MIN_PLY = 20


class IdentityEncoder(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 20, embedding: int = 256,
                 bn_eps: float = 1e-5, bn_momentum: float = 0.1, pick: bool = False,
                 checkpoint_segments: int = 0):
        super().__init__()
        # checkpoint_segments > 0 (2026-11-17, E2E-scratch): recompute the residual blocks in that
        # many segments during backward (activation checkpointing) so 48-move groups fit in memory.
        # Training only; the recompute updates BatchNorm running statistics a second time.
        self.checkpoint_segments = int(checkpoint_segments)
        trunk = StrengthResNet(channels, blocks, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.stem, self.blocks = trunk.stem, trunk.blocks
        # pick (2026-11-15): the trunk feature at the point the player is about to play joins the
        # board mean and max. The input then carries a 19th plane, one-hot at that point
        # (with_pick_plane), so colour swaps and D4 transforms move it with the board.
        self.pick = bool(pick)
        width = (3 if self.pick else 2) * channels
        self.project = nn.Sequential(nn.Linear(width, 2 * channels), nn.ReLU(inplace=False),
                                     nn.Linear(2 * channels, embedding))
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def load_trunk(self, state: dict) -> None:
        """Copy stem/blocks weights from a StrengthResNet state dict (A-024 model.pt)."""
        own = {k: v for k, v in state.items() if k.startswith(("stem.", "blocks."))}
        missing = self.load_state_dict(own, strict=False)
        if any(k.startswith(("stem.", "blocks.")) for k in missing.missing_keys) or missing.unexpected_keys:
            raise ValueError("trunk state does not match the encoder")

    def trunk(self, h: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_segments and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint_sequential(self.blocks, self.checkpoint_segments, h,
                                                                use_reentrant=False)
        return self.blocks(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pick:
            h = self.trunk(self.stem(x))
            return self.project(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))
        if x.shape[1] != 19:
            raise ValueError("a pick encoder needs the move plane (with_pick_plane)")
        h = self.trunk(self.stem(x[:, :18]))
        picked = (h * x[:, 18:19].to(h.dtype)).sum(dim=(2, 3))
        return self.project(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), picked], dim=1))


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


class MovePoints:
    """Point (row * 19 + col) of the move played at (game, ply), read from a board cache's
    shard_XXXXX_moves.npy files."""

    def __init__(self, cache: BoardCache):
        self.cache, self._open = cache, {}

    def __call__(self, games, plies) -> np.ndarray:
        games, plies = np.asarray(games, np.int64), np.asarray(plies, np.int64)
        points = np.empty(len(games), np.int64)
        shard_ids = self.cache.shard_ids[games]
        for shard_id in np.unique(shard_ids):
            if shard_id not in self._open:
                self._open[shard_id] = np.load(self.cache.directory / f"shard_{int(shard_id):05d}_moves.npy",
                                               mmap_mode="r", allow_pickle=False)
            rows = np.flatnonzero(shard_ids == shard_id)
            points[rows] = self._open[shard_id][self.cache.offsets[games[rows]] + plies[rows]]
        return points


def with_pick_plane(planes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """uint8 [B, 19, 19, 19]: the 18 input planes plus a one-hot plane at each move's point
    (all zero for a pass)."""
    out = np.zeros((len(planes), 19, 19, 19), np.uint8)
    out[:, :18] = planes
    valid = np.flatnonzero((points >= 0) & (points < 361))
    out.reshape(len(planes), 19, 361)[valid, 18, points[valid]] = 1
    return out


class GE2EBatches(torch.utils.data.Dataset):
    """Item `step` = one GPU batch: planes uint8 [N*M*S,18,19,19], N players x M games x S moves."""

    def __init__(self, cache_dir, player_games: list[np.ndarray], lengths: np.ndarray, *, players: int,
                 games: int, moves: int, steps: int, seed: int, rank: int, start_step: int = 0,
                 max_open_shards: int = 8, player_color: np.ndarray | None = None, min_ply: int = MIN_PLY,
                 player_d4: bool = False, world: int = 1, extra_games: list[np.ndarray] | None = None,
                 extra_per_group: int = 0, pick: bool = False, player_rank: np.ndarray | None = None,
                 rank_focus: dict | None = None, player_p: np.ndarray | None = None):
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
        # rank_focus = {"share": x, "radius": d} (2026-11-15): round(x * players) of a group come
        # from the ranks within d of a centre rank, the centre being the rank of a uniformly drawn
        # player; the others from all remaining players. player_rank[i] = rank id of player i.
        self.pick, self.rank_focus, self._points = bool(pick), rank_focus, None
        if rank_focus:
            if world > 1 or self.extra_per_group or player_rank is None:
                raise ValueError("rank_focus needs player ranks and excludes cross-GPU groups and gallery players")
            self.player_rank = np.asarray(player_rank, np.int64)
            self.focus = int(round(rank_focus["share"] * players))
            bands = [int((np.abs(self.player_rank - r) <= rank_focus["radius"]).sum())
                     for r in np.unique(self.player_rank)]
            if len(self.player_rank) != len(player_games) or not 0 < self.focus < players or min(bands) < self.focus:
                raise ValueError("rank_focus: bad share, or a rank band smaller than the focused part")
        self.cache_dir, self.player_games, self.lengths = cache_dir, player_games, lengths
        # player_p: drawing probabilities of the players (external data mix, 2026-11-18); None = uniform
        self.player_p = None if player_p is None else np.asarray(player_p, np.float64)
        if self.player_p is not None and (len(self.player_p) != len(player_games) or self.extra_per_group
                                          or rank_focus):
            raise ValueError("player_p needs one probability per player and no gallery or rank focus")
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
            chosen = shared.choice(len(self.player_games), size=n * self.world, replace=False,
                                   p=self.player_p)[self.rank * n:(self.rank + 1) * n]
        elif self.extra_per_group:
            e = self.extra_per_group
            chosen = rng.choice(len(self.player_games), size=n - e, replace=False)
            extra = rng.choice(len(self.extra_games), size=e, replace=False)
        elif self.rank_focus:
            chosen = self.focused_players(rng)
        else:
            chosen = rng.choice(len(self.player_games), size=n, replace=False, p=self.player_p)
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

    def focused_players(self, rng) -> np.ndarray:
        """Player indices of one group under rank_focus: the focused part first, no repeats."""
        centre = self.player_rank[rng.integers(len(self.player_games))]
        band = np.flatnonzero(np.abs(self.player_rank - centre) <= self.rank_focus["radius"])
        first = rng.choice(band, size=self.focus, replace=False)
        rest = np.setdiff1d(np.arange(len(self.player_games)), first)
        return np.concatenate([first, rng.choice(rest, size=self.shape[0] - self.focus, replace=False)])

    def rank_exposure(self, world: int) -> dict:
        """Group slots per rank over the whole run (every GPU rank and step), replayed from the
        stateless sampler: {"centre": counts, "slots": counts, "players": counts} by rank id."""
        size = int(self.player_rank.max()) + 1
        centre, slots = np.zeros(size, np.int64), np.zeros(size, np.int64)
        for rank in range(world):
            for step in range(self.steps):
                rng = np.random.default_rng([self.seed, rank, step])
                chosen = self.focused_players(rng)
                # focused_players draws the centre first; replay that draw to count it
                centre[self.player_rank[np.random.default_rng([self.seed, rank, step])
                                        .integers(len(self.player_games))]] += 1
                slots += np.bincount(self.player_rank[chosen], minlength=size)
        return {"centre": centre.tolist(), "slots": slots.tolist(),
                "players": np.bincount(self.player_rank, minlength=size).tolist()}

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
        if self.pick:
            if self._points is None:
                self._points = MovePoints(self._cache)
            planes = with_pick_plane(planes, self._points(games, plies))
        if self.player_d4:
            n, m, s = self.shape
            transforms = np.random.default_rng([self.seed, self.rank, self.start + item, 4]).integers(8, size=n)
            blocks = planes.reshape(n, m * s, *planes.shape[1:])
            planes = np.concatenate([apply_d4(block, int(t)) for block, t in zip(blocks, transforms)])
        return torch.from_numpy(planes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = state["_points"] = None
        return state
