"""DataLoader adapter for on-demand packed Strength V1 board decompression.

Pass sample game/ply indices selected by the caller (only the labelled
player's actual moves, including passes). This module makes no CV splits,
sample permutation or augmentation and never reads raw CSV/SGF.

Example, inside a script with an ``if __name__ == '__main__'`` guard::

    from torch.utils.data import DataLoader
    dataset = PackedBoardDataset('train_game.npy', 'train_ply.npy')
    loader = DataLoader(dataset, batch_size=2048, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    for batch in loader:
        x = batch['planes'].to('cuda', non_blocking=True).float()
        # x: [B,18,19,19]; batch['rank_id']: [B]

Prefer index file paths for large datasets: each worker lazily memmaps them,
rather than pickling all sample indices under Windows spawn. Index files are
external inputs; this decoder does not generate train/validation indices.
"""

from __future__ import annotations

import operator
import os
from pathlib import Path

import numpy as np

from go_aicup.data.board_cache import BoardCache, DEFAULT_OUTPUT, _close_mmap


def _open_index(value):
    array = np.load(value, mmap_mode="r", allow_pickle=False) if isinstance(value, (str, Path)) else np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        _close_mmap(array)
        raise ValueError("sample indices must be 1D integer arrays")
    return array


class PackedBoardDataset:
    """PyTorch map-style protocol with batch ``__getitems__`` support.

    Output: planes uint8[18,19,19], rank_id, game_index and ply. History
    includes the opponent's plies, but the selected ply must be the labelled
    player's actual move. Rank and IDs are metadata, never input channels.
    """

    def __init__(self, game_indices, plies, directory: Path = DEFAULT_OUTPUT, *, max_open_shards: int = 8):
        self.directory = Path(directory).resolve()
        self.max_open_shards = max_open_shards
        self._sources = tuple(Path(value).resolve() if isinstance(value, (str, Path)) else np.asarray(value)
                              for value in (game_indices, plies))
        indices = []
        try:
            for source in self._sources:
                indices.append(_open_index(source))
            if indices[0].shape != indices[1].shape:
                raise ValueError("game and ply sample indices must have equal lengths")
            self._length = len(indices[0])
        finally:
            for array, source in zip(indices, self._sources):
                if isinstance(source, Path):
                    _close_mmap(array)
        self._cache = None
        self._indices = None
        self._pid = None

    def __len__(self):
        return self._length

    def close(self):
        if self._cache is not None:
            self._cache.close()
        if self._indices is not None:
            for array, source in zip(self._indices, self._sources):
                if isinstance(source, Path):
                    _close_mmap(array)
        self._cache = None
        self._indices = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=None, _indices=None, _pid=None)
        return state

    def _ensure_open(self):
        if self._pid != os.getpid() or self._cache is None:
            self.close()
            self._cache = BoardCache(self.directory, max_open_shards=self.max_open_shards)
            self._indices = tuple(_open_index(source) for source in self._sources)
            self._pid = os.getpid()

    def __getitems__(self, sample_indices):
        samples = np.asarray([operator.index(index) for index in sample_indices], dtype=np.int64)
        if np.any(samples < 0) or np.any(samples >= len(self)):
            raise IndexError("sample index out of range")
        self._ensure_open()
        games, plies = (array[samples] for array in self._indices)
        histories, colors = self._cache.history_batch(games, plies)
        if np.any(colors != self._cache.player_colors[games]):
            raise ValueError("sample index contains an opponent ply")
        inputs = np.empty((len(samples), 18, 19, 19), np.uint8)
        inputs[:, :16] = histories.reshape(len(samples), 16, 19, 19)
        inputs[:, 16] = (colors == 0)[:, None, None]
        inputs[:, 17] = (colors == 1)[:, None, None]
        return [{"planes": inputs[i], "rank_id": int(self._cache.rank_ids[game]),
                 "game_index": int(game), "ply": int(ply)}
                for i, (game, ply) in enumerate(zip(games, plies))]

    def __getitem__(self, sample_index):
        return self.__getitems__([sample_index])[0]
