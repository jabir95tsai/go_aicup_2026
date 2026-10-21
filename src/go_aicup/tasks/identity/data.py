"""Competition-1 inputs: per-game opening prefixes and synthetic pool splits.

Reads the existing board cache (`data/processed/boards/strength_v1`): its
`games.parquet` row i is game_index i, and `shard_XXXXX_moves.npy` holds the
int16 point ids (row*19+col, 0..360) of every game in the shard back to back.
Every training game has >= 100 plies, so prefixes up to 100 need no padding.

Synthetic pools come from `data.cv.make_pid_splits` (400 players x 100 black
gallery games, one disjoint query per player, k balanced). Splits 0-2 are
player-disjoint; 3-4 overlap them and are not independent samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import PROCESSED

CACHE_DIR = PROCESSED / "boards" / "strength_v1"
OUT_DIR = PROCESSED / "identity"
MIN_GAME_PLIES = 100


def load_meta(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    meta = pd.read_parquet(cache_dir / "games.parquet")
    if not np.array_equal(meta["game_index"].to_numpy(), np.arange(len(meta))):
        raise ValueError("games.parquet rows must be game_index order")
    return meta


def opening_prefix(meta: pd.DataFrame, plies: int, cache_dir: Path = CACHE_DIR,
                   out_dir: Path = OUT_DIR) -> np.ndarray:
    """int16 [n_games, plies]; cached as `train_prefix_{plies}.npy`."""
    if plies > MIN_GAME_PLIES:
        raise ValueError(f"prefixes longer than {MIN_GAME_PLIES} plies would need padding")
    path = out_dir / f"train_prefix_{plies}.npy"
    if path.exists():
        prefix = np.load(path)
        if prefix.shape == (len(meta), plies):
            return prefix
    if (meta["length"].to_numpy() < plies).any():
        raise ValueError("a game is shorter than the requested prefix")
    prefix = np.empty((len(meta), plies), np.int16)
    offsets = np.arange(plies)
    for shard, rows in meta.groupby("shard_id", sort=True).indices.items():
        moves = np.load(cache_dir / f"shard_{shard:05d}_moves.npy", mmap_mode="r")
        start = meta["local_offset"].to_numpy()[rows]
        prefix[rows] = moves[start[:, None] + offsets]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, prefix)
    tmp.replace(path)
    return prefix


def d4_tables() -> np.ndarray:
    """int16 [8, 361] point maps of the 8 board symmetries (row 0 = identity)."""
    r, c = np.divmod(np.arange(361), 19)
    s = 18
    maps = [(r, c), (c, r), (s - r, c), (r, s - c), (s - r, s - c),
            (c, s - r), (s - c, r), (s - c, s - r)]
    return np.stack([rr * 19 + cc for rr, cc in maps]).astype(np.int16)


def canonical_d4(prefix: np.ndarray, chunk: int = 100_000) -> np.ndarray:
    """Map each row to its lexicographically smallest D4 image over the whole prefix.

    Exactly invariant to any D4 transform of a row (all 8 images form the same
    set). Rows sharing their first t plies keep identical first t plies, so
    exact-prefix matching is preserved. E0 keyed on 6 plies only, which is not
    invariant when those plies are symmetric.
    """
    tables = d4_tables()
    out = np.empty_like(prefix)
    for lo in range(0, len(prefix), chunk):
        images = tables[:, prefix[lo:lo + chunk]]               # [8, n, plies]
        alive = np.ones(images.shape[:2], bool)
        for t in range(images.shape[2]):
            vals = np.where(alive, images[:, :, t], np.int16(32767))
            alive &= vals == vals.min(axis=0)
        best = alive.argmax(axis=0)
        out[lo:lo + chunk] = images[best, np.arange(images.shape[1])]
    return out


def opening_prefix_d4(meta: pd.DataFrame, plies: int, out_dir: Path = OUT_DIR) -> np.ndarray:
    """`canonical_d4(opening_prefix(...))`, cached as `train_prefix_{plies}_d4.npy`."""
    path = out_dir / f"train_prefix_{plies}_d4.npy"
    if path.exists():
        prefix = np.load(path)
        if prefix.shape == (len(meta), plies):
            return prefix
    prefix = canonical_d4(opening_prefix(meta, plies, out_dir=out_dir))
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, prefix)
    tmp.replace(path)
    return prefix


def pid_splits(meta: pd.DataFrame, out_dir: Path = OUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(gallery, questions) for all synthetic pools, cached as parquet."""
    gallery_path, question_path = out_dir / "pid_gallery.parquet", out_dir / "pid_questions.parquet"
    if gallery_path.exists() and question_path.exists():
        gallery, questions = pd.read_parquet(gallery_path), pd.read_parquet(question_path)
        questions["games"] = questions["games"].map(np.asarray)
        return gallery, questions
    gallery, questions = cv.make_pid_splits(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    gallery.to_parquet(gallery_path, index=False)
    questions.to_parquet(question_path, index=False)
    return gallery, questions


def split_view(gallery: pd.DataFrame, questions: pd.DataFrame, split: int):
    """Candidates (sorted ids), their gallery game lists, and the split's questions."""
    g = gallery[gallery["split"] == split]
    q = questions[questions["split"] == split].sort_values("question_id").reset_index(drop=True)
    candidates = np.array(sorted(g["player_id"].unique()))
    by_player = g.groupby("player_id")["game"].apply(np.asarray)
    return candidates, [by_player[p] for p in candidates], q
