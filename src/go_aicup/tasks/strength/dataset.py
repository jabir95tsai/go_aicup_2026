"""DataLoader adapter for on-demand packed Strength V1 board decompression.

Pass sample game/ply indices selected by the caller (only the labelled
player's moves). CV splits are made only by data.cv.
This module also supplies explicit global batch boundaries, disk-backed
permutations and stateless D4. It never reads raw CSV/SGF.

Example, inside a script with an ``if __name__ == '__main__'`` guard::

    import torch
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

from go_aicup.data.board_cache import BoardCache, DEFAULT_OUTPUT, _close_mmap, assemble_inputs
from go_aicup.artifacts import file_hash, write_json


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

    def __init__(self, game_indices, plies, directory: Path = DEFAULT_OUTPUT, *,
                 max_open_shards: int = 8, augment: bool = False, seed: int = 2026):
        self.directory = Path(directory).resolve()
        self.max_open_shards = max_open_shards
        self.augment = augment
        self.seed = seed
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
        # Epoch travels with the task: persistent workers do not need mutable
        # dataset.epoch, and prefetch cannot change augmentation on resume.
        tasks = [index if isinstance(index, tuple) else (index, 0, True)
                 for index in sample_indices]
        samples = np.asarray([operator.index(task[0]) for task in tasks], dtype=np.int64)
        if np.any(samples < 0) or np.any(samples >= len(self)):
            raise IndexError("sample index out of range")
        self._ensure_open()
        games, plies = (array[samples] for array in self._indices)
        histories, colors = self._cache.history_batch(games, plies)
        if np.any(colors != self._cache.player_colors[games]):
            raise ValueError("sample index contains an opponent ply")
        inputs = assemble_inputs(histories, colors)
        if self.augment:
            for i, (sample, epoch, _) in enumerate(tasks):
                transform = d4_id(self.seed, int(epoch), int(sample))
                inputs[i] = apply_d4(inputs[i], transform)
        return [{"planes": inputs[i], "rank_id": int(self._cache.rank_ids[game]),
                 "game_index": int(game), "ply": int(ply),
                 "sample_index": int(samples[i]), "valid": bool(tasks[i][2])}
                for i, (game, ply) in enumerate(zip(games, plies))]

    def __getitem__(self, sample_index):
        return self.__getitems__([sample_index])[0]


def d4_id(seed: int, epoch: int, sample: int) -> int:
    """SplitMix64 with fixed integer arithmetic, independent of worker RNG."""
    mask = (1 << 64) - 1
    value = (seed + 0x9E3779B97F4A7C15 * (epoch + 1) + sample) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return int((value ^ (value >> 31)) & 7)


def apply_d4(planes: np.ndarray, transform: int) -> np.ndarray:
    if not 0 <= transform < 8:
        raise ValueError("D4 transform must be 0..7")
    result = np.flip(planes, axis=-1) if transform >= 4 else planes
    return np.ascontiguousarray(np.rot90(result, transform % 4, axes=(-2, -1)))


def create_permutation(directory: Path, length: int, seed: int, epoch: int) -> Path:
    """One shared uint32 permutation per epoch, published atomically.

    Call on process zero, then barrier before readers open the result. No
    Python list containing the entire dataset is materialized.
    """
    import json

    if length < 1 or length > np.iinfo(np.uint32).max:
        raise ValueError("permutation length exceeds uint32 format or is empty")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{epoch:04d}.npy"
    record_path = path.with_suffix(".json")
    identity = {"length": length, "seed": seed, "epoch": epoch,
                "rng": "PCG64", "numpy": np.__version__}
    if path.exists() or record_path.exists():
        if not path.exists() or not record_path.exists():
            raise ValueError("incomplete permutation; remove its files or use a new run directory")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if any(record.get(key) != value for key, value in identity.items()) or file_hash(path) != record["sha256"]:
            raise ValueError("permutation identity/hash mismatch")
        return path
    temporary = path.with_name(path.name + ".tmp")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint32, shape=(length,))
    try:
        for start in range(0, length, 1_000_000):
            end = min(start + 1_000_000, length)
            array[start:end] = np.arange(start, end, dtype=np.uint32)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
        rng.shuffle(array)
        array.flush()
    finally:
        _close_mmap(array)
    os.replace(temporary, path)
    write_json(record_path, dict(identity, sha256=file_hash(path)))
    return path


class GlobalBatchSampler:
    """One epoch order, explicit global batches, disjoint per-rank slices.

    Tail placeholders carry valid=False and are removed BEFORE the model,
    so duplicated boards do not enter SyncBN. Each task is (sample,epoch,valid).
    start_batch is the consumer cursor, never DataLoader's prefetched cursor.
    """
    def __init__(self, length: int, global_batch_size: int, *, rank: int = 0,
                 world_size: int = 1, permutation=None, epoch: int = 0,
                 start_batch: int = 0):
        if length < 1 or global_batch_size < world_size or global_batch_size % world_size:
            raise ValueError("positive length and global batch divisible by world_size required")
        if not 0 <= rank < world_size or start_batch < 0:
            raise ValueError("invalid rank or start_batch")
        self.length, self.global_batch_size = length, global_batch_size
        self.rank, self.world_size = rank, world_size
        self.permutation, self.epoch = permutation, epoch
        self.total_batches = (length + global_batch_size - 1) // global_batch_size
        if start_batch > self.total_batches:
            raise ValueError("start_batch exceeds epoch length")
        self.start_batch = start_batch

    def __len__(self):
        return self.total_batches - self.start_batch

    def __iter__(self):
        order = _open_index(self.permutation) if self.permutation is not None else None
        try:
            if order is not None and len(order) != self.length:
                raise ValueError("permutation length mismatch")
            for step in range(self.start_batch, self.total_batches):
                begin = step * self.global_batch_size
                size = min(self.global_batch_size, self.length - begin)
                quotient, remainder = divmod(size, self.world_size)
                local_size = quotient + int(self.rank < remainder)
                local_begin = begin + self.rank * quotient + min(self.rank, remainder)
                indices = (np.arange(local_begin, local_begin + local_size, dtype=np.int64)
                           if order is None else order[local_begin:local_begin + local_size])
                tasks = [(int(index), self.epoch, True) for index in indices]
                slots = quotient + int(remainder > 0)
                tasks.extend([(0, self.epoch, False)] * (slots - local_size))
                yield tasks
        finally:
            if isinstance(self.permutation, (str, Path)) and order is not None:
                _close_mmap(order)


def make_loader(dataset: PackedBoardDataset, sampler: GlobalBatchSampler, *,
                num_workers: int = 2, prefetch_factor: int = 2,
                pin_memory: bool = True, persistent_workers: bool = True):
    import torch
    from torch.utils.data import DataLoader
    if num_workers < 0 or prefetch_factor < 1:
        raise ValueError("invalid DataLoader worker/prefetch configuration")
    kwargs = dict(batch_sampler=sampler, num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=persistent_workers and num_workers > 0,
                  generator=torch.Generator().manual_seed(dataset.seed + sampler.rank))
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, multiprocessing_context="spawn")
    return DataLoader(dataset, **kwargs)
