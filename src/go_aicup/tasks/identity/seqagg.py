"""Cross-move aggregation head on per-move identity vectors (2026-11-16, moved here 2026-11-17).

Arm B: per-move pre-LN residual feed-forward blocks (256 -> 512 -> 256, ReLU) x 2 plus a learned
embedding of the move's ply, then a mean over the valid moves. Arm C adds 4-head self-attention
(nn.TransformerEncoderLayer, norm_first). Used frozen by scripts/identity/pid2_seqagg.py and
trained jointly with the encoder by train_e2e.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.tasks.identity.ge2e import as_black
from go_aicup.tasks.strength.shared.runtime import autocast_context

MAX_PLY = 512


def own_plies(color: np.ndarray, length: np.ndarray, min_ply: int) -> list[np.ndarray]:
    first = min_ply + (color.astype(np.int64) - min_ply) % 2
    return [np.arange(f, n, 2) for f, n in zip(first, length)]


class _Planes(torch.utils.data.Dataset):
    """Item i = assembled input planes of samples [i * batch, (i + 1) * batch), white colour-swapped."""

    def __init__(self, cache_dir, games, plies, colors, batch):
        self.cache_dir, self.games, self.plies, self.colors, self.batch = cache_dir, games, plies, colors, batch
        self._cache = None

    def __len__(self):
        return (len(self.games) + self.batch - 1) // self.batch

    def __getitem__(self, i):
        if self._cache is None:
            self._cache = BoardCache(self.cache_dir)
        span = slice(i * self.batch, (i + 1) * self.batch)
        histories, side = self._cache.history_batch(self.games[span], self.plies[span])
        if not np.array_equal(side, self.colors[span]):
            raise ValueError("sampled ply is not the player's move")
        histories, side = as_black(histories, side)
        return torch.from_numpy(assemble_inputs(histories, side)), i

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


WORKERS = 10


@torch.no_grad()
def move_vectors(model, config, cache, games, colors, lengths, device, min_ply, batch=8192):
    """(float32 [moves, D] per-move vectors in game order then ply order, int64 [games + 1] offsets);
    white moves colour-swapped as in twocolor.embed_games(swap=True)."""
    plies = own_plies(colors, lengths, min_ply)
    sizes = np.array([len(p) for p in plies])
    owner = np.repeat(np.arange(len(games)), sizes)
    sample_game, sample_ply = games[owner], np.concatenate(plies)
    out = np.empty((len(owner), config["model"]["embedding"]), np.float32)
    model.eval()
    dataset = _Planes(cache.directory, sample_game, sample_ply, colors[owner], batch)
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                                         multiprocessing_context="spawn", pin_memory=True)
    for x, i in loader:
        with autocast_context(config, device):
            vec = model(x.to(device, dtype=torch.float32, non_blocking=True))
        out[i * batch:(i + 1) * batch] = vec.float().cpu().numpy()
    return out, np.concatenate([[0], np.cumsum(sizes)])


SEED = 2026
HOLDOUT_SHARE = 0.05


class MoveStore:
    """All per-move vectors of one cache part in RAM, with per-game offsets and plies."""

    def __init__(self, directory: Path, colors: np.ndarray):
        record = json.loads((directory / "chunks.json").read_text())
        self.games = np.load(directory / "games.npy")
        self.offsets = np.load(directory / "offsets.npy")
        if record["moves"] != self.offsets[-1] or record["games"] != len(self.games):
            raise ValueError(f"{directory}: chunks.json does not match offsets")
        self.vectors = np.empty((int(self.offsets[-1]), 256), np.float16)
        for chunk in record["chunks"]:
            start = self.offsets[chunk["first_game"]]
            part = np.load(directory / chunk["file"])
            if len(part) != chunk["moves"] or start + len(part) != self.offsets[chunk["first_game"] + chunk["games"]]:
                raise ValueError(f"{chunk['file']}: size does not match its games")
            self.vectors[start:start + len(part)] = part
        self.finite = bool(np.isfinite(self.vectors).all())
        self.first_ply = colors[self.games].astype(np.int64) + record["min_ply"]   # min_ply 0 -> colour
        if record["min_ply"] != 0:
            raise ValueError("ply positions assume min_ply 0")
        self.extra = {p.stem: np.load(p) for p in directory.glob("*.npy")
                      if p.stem in ("players", "rank_ids")}

    def batch(self, rows: np.ndarray, device):
        """rows = positions in self.games -> (x [G, L, D] float32, ply [G, L], valid [G, L])."""
        sizes = self.offsets[rows + 1] - self.offsets[rows]
        width = int(sizes.max())
        index = np.concatenate([np.arange(a, b) for a, b in zip(self.offsets[rows], self.offsets[rows + 1])])
        game = np.repeat(np.arange(len(rows)), sizes)
        step = np.arange(len(index)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        x = torch.zeros(len(rows), width, self.vectors.shape[1], device=device)
        x[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = \
            torch.from_numpy(self.vectors[index].astype(np.float32)).to(device)
        ply = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        ply[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = \
            torch.from_numpy(self.first_ply[rows][game] + 2 * step).to(device)
        valid = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        valid[torch.from_numpy(game).to(device), torch.from_numpy(step).to(device)] = True
        return x, ply, valid


def holdout_players(rank_ids: np.ndarray) -> np.ndarray:
    """bool [players]: 5% of the players of every rank (rounded), fixed seed; shared by both arms."""
    rng = np.random.default_rng([SEED, 5])
    out = np.zeros(len(rank_ids), bool)
    for r in np.unique(rank_ids):
        members = np.flatnonzero(rank_ids == r)
        out[rng.choice(members, size=int(round(HOLDOUT_SHARE * len(members))), replace=False)] = True
    return out


def monitor_set(store: MoveStore, held: np.ndarray, draws: int = 3):
    """Fixed monitor questions on the held-out players: per draw every such player with >= 20 games
    gets a 10-game gallery, and 400 questions (k = 1/2/3/5 x 100, player drawn with replacement)
    take unused non-gallery games. -> list of (gallery rows per player, question rows, k, truth)."""
    players = store.extra["players"]
    by_player = {p: np.flatnonzero(players == p) for p in np.flatnonzero(held)}
    eligible = [p for p, rows in by_player.items() if len(rows) >= 20]
    out = []
    for draw in range(draws):
        rng = np.random.default_rng([SEED, 9, draw])
        order = {p: rng.permutation(by_player[p]) for p in eligible}
        used = {p: 10 for p in eligible}
        gallery = [np.sort(order[p][:10]) for p in eligible]
        questions, ks, truth = [], [], []
        for k in rng.permutation(np.repeat([1, 2, 3, 5], 100)):
            while True:
                i = int(rng.integers(len(eligible)))
                if used[eligible[i]] + k <= len(order[eligible[i]]):
                    break
            p = eligible[i]
            questions.append(np.sort(order[p][used[p]:used[p] + k]))
            used[p] += int(k)
            ks.append(int(k))
            truth.append(i)
        rows = np.concatenate(gallery + questions)
        assert len(np.unique(rows)) == len(rows), "a monitor game is used twice"
        out.append((gallery, questions, np.array(ks), np.array(truth)))
    return out, len(eligible)


@torch.no_grad()
def game_vectors(head, store: MoveStore, rows: np.ndarray, device, batch: int = 2048) -> np.ndarray:
    out = np.empty((len(rows), 256), np.float32)
    if head is not None:
        head.eval()
    for s in range(0, len(rows), batch):
        part = rows[s:s + batch]
        x, ply, valid = store.batch(part, device)
        if head is None:
            mask = valid[..., None].float()
            v = (x * mask).sum(1) / mask.sum(1)
        else:
            v = head(x, ply, valid)
        out[s:s + len(part)] = v.float().cpu().numpy()
    if head is not None:
        head.train()
    return out


def monitor_score(head, store, sets, device) -> float:
    """Pure-style official score over the monitor draws (k buckets equally weighted)."""
    from go_aicup.metrics import PID_WEIGHTS
    from go_aicup.tasks.identity import assign
    rows = np.unique(np.concatenate([np.concatenate(g + q) for g, q, _, _ in sets]))
    vec = game_vectors(head, store, rows, device)
    where = {r: i for i, r in enumerate(rows)}
    totals = []
    for gallery, questions, ks, truth in sets:
        cos = twocolor_style([[where[r] for r in g] for g in gallery], [[where[r] for r in q] for q in questions], vec)
        ranked = assign.top5(cos)
        pts = ((ranked == truth[:, None]) * PID_WEIGHTS).sum(axis=1)
        totals.append(np.mean([pts[ks == k].mean() for k in (1, 2, 3, 5)]))
    return float(np.mean(totals))


def twocolor_style(gallery, questions, vec):
    from go_aicup.tasks.identity import twocolor
    return twocolor.style_scores(vec, [np.asarray(g) for g in gallery], [np.asarray(q) for q in questions])




class Head(torch.nn.Module):
    def __init__(self, attention: bool, dim: int = 256, hidden: int = 512, layers: int = 2, heads: int = 4,
                 position: bool = True):
        super().__init__()
        self.attention = attention
        self.position = torch.nn.Embedding(MAX_PLY, dim) if position else None
        if position:
            torch.nn.init.normal_(self.position.weight, std=0.02)
        if attention:
            self.blocks = torch.nn.ModuleList(torch.nn.TransformerEncoderLayer(
                dim, heads, hidden, dropout=0.0, batch_first=True, norm_first=True) for _ in range(layers))
        else:
            self.blocks = torch.nn.ModuleList(torch.nn.ModuleDict({
                "norm": torch.nn.LayerNorm(dim),
                "ff": torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, dim))})
                for _ in range(layers))
        self.w = torch.nn.Parameter(torch.tensor(10.0))
        self.b = torch.nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor, ply: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x [G, L, D] padded per-move vectors, ply [G, L], valid [G, L] bool -> game vectors [G, D]
        (mean over valid moves only)."""
        h = x + self.position(ply) if self.position is not None else x
        for block in self.blocks:
            if self.attention:
                h = block(h, src_key_padding_mask=~valid)
            else:
                h = h + block["ff"](block["norm"](h))
        mask = valid[..., None].to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1)


def load_head(path, device):
    """(head, step) from a best.pt / last.pt / head.pt written by pid2_seqagg or train_e2e."""
    state = torch.load(path, map_location="cpu")
    head = Head(attention=state["arm"] == "C", position=state.get("position", True)).to(device)
    head.load_state_dict(state["model"])
    return head, int(state["step"])
