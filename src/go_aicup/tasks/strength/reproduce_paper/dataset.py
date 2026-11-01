"""Pre-action MiniZero-style samples on top of the existing packed cache.

History pairs are current player/opponent, newest first. Policy IDs use the
repository's row-major coordinates; 361 is pass. No teacher labels are invented.
"""
from collections import OrderedDict
import json
import operator
import os
from pathlib import Path
import numpy as np
from go_aicup.artifacts import file_hash
from go_aicup.data.board_cache import BoardCache, _close_mmap, assemble_inputs
from go_aicup.tasks.strength.shared.dataset import apply_d4, d4_id
from .config import DEFAULT_TRAINING_SEED, SPLIT_SEED


def transform_policy(action, transform):
    """Apply exactly the board D4 transform, with pass invariant."""
    action = operator.index(action)
    if not 0 <= action <= 361:
        raise ValueError("policy action must be 0..361")
    if not 0 <= transform < 8:
        raise ValueError("D4 transform must be 0..7")
    if action == 361:
        return action
    row, col = divmod(action, 19)
    if transform >= 4:
        col = 18 - col
    for _ in range(transform % 4):
        row, col = 18 - col, row
    return row * 19 + col


def soft_value_target(p_black, player_color):
    """Convert a validated teacher black win probability to sample perspective."""
    probability = np.asarray(p_black, dtype=np.float64)
    color = np.asarray(player_color)
    if not np.all(np.isfinite(probability) & (probability >= 0) & (probability <= 1)):
        raise ValueError("p_black must be finite and within [0,1]")
    if not np.all((color == 0) | (color == 1)):
        raise ValueError("player color must be black=0 or white=1")
    return np.asarray((2 * probability - 1) * (1 - 2 * color.astype(np.int8)), dtype=np.float32)


def pre_action_inputs(cache, games, plies):
    """Decode only boards preceding the selected actions; preserve true history."""
    games, plies = np.asarray(games), np.asarray(plies)
    if games.ndim != 1 or games.shape != plies.shape or games.dtype.kind not in 'iu' or plies.dtype.kind not in 'iu':
        raise ValueError("games and plies must be equal-length 1D integer arrays")
    if np.any(games < 0) or np.any(games >= len(cache)):
        raise IndexError("game index out of range")
    if np.any(plies < 0) or np.any(plies >= cache.lengths[games]):
        raise IndexError("ply index out of range")
    histories = np.zeros((len(games), 8, 2, 19, 19), np.uint8)
    colors = np.empty(len(games), np.int8)
    # Actual selected-action colors, never infer turn from the previous action.
    for game in np.unique(games):
        rows = np.flatnonzero(games == game)
        colors[rows] = cache.colors(int(game))[plies[rows]]
    if np.any(colors != cache.player_colors[games]):
        raise ValueError("sample contains an opponent ply")
    previous = plies > 0
    if previous.any():
        histories[previous] = cache.history_batch(games[previous], plies[previous].astype(np.int64) - 1)[0]
    white = colors == 1
    histories[white] = histories[white, :, ::-1]
    return assemble_inputs(histories, colors), colors


class PaperDataset:
    """Lazy map-style dataset over prepare.py's compact game/offset/ply arrays.

    Tasks may be (sample_index, augmentation_round); round is carried in each
    task for deterministic persistent-worker use. This is not a group sampler.
    Optional exported teacher labels follow the same sample_index ordering.
    Without label_dir, value supervision remains absent.
    """
    def __init__(self, directory, split='train', *, augment=False, seed=None,
                 max_open_shards=8, cache_dir=None, label_dir=None):
        if split not in ('train', 'valid'):
            raise ValueError("split must be train or valid")
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.directory = Path(directory).resolve()
        self.record = json.loads((self.directory / 'manifest.json').read_text())
        self.split, self.augment = split, augment
        if self.record['identity'].get('split_seed') != SPLIT_SEED:
            raise ValueError('SE dataset requires a shared split with split_seed=2026')
        self.seed = DEFAULT_TRAINING_SEED if seed is None else operator.index(seed)
        self.cache_dir = Path(cache_dir or self.record['cache_dir']).resolve()
        self.max_open_shards = max_open_shards
        self._length = self.record['splits'][split]['positions']
        self._cache = None
        self._indices = None
        self._moves = OrderedDict()
        self._pid = None
        self.label_dir = Path(label_dir).resolve() if label_dir is not None else None
        self._values = None

    def __len__(self):
        return self._length

    def close(self):
        if self._values is not None:
            _close_mmap(self._values)
            self._values = None
        if self._cache is not None:
            self._cache.close()
        if self._indices is not None:
            for array in self._indices:
                _close_mmap(array)
        for array in self._moves.values():
            _close_mmap(array)
        self._cache, self._indices = None, None
        self._moves.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_cache=None, _indices=None, _moves=OrderedDict(), _pid=None, _values=None)
        return state

    def _ensure_open(self):
        if self._pid == os.getpid() and self._cache is not None:
            return
        self.close()
        try:
            self._cache = BoardCache(self.cache_dir, max_open_shards=self.max_open_shards)
            if file_hash(self.cache_dir / 'manifest.json') != self.record['identity']['cache_manifest_sha256']:
                raise ValueError("prepared data belongs to another board cache")
            paths = [self.directory / f'{self.split}_{suffix}.npy' for suffix in ('game', 'offsets', 'ply')]
            for path in paths:
                if file_hash(path) != self.record['files'][path.name]:
                    raise ValueError(f"prepared index checksum mismatch: {path.name}")
            self._indices = tuple(np.load(path, mmap_mode='r', allow_pickle=False) for path in paths)
            games, offsets, plies = self._indices
            if (games.dtype != np.uint32 or offsets.dtype != np.int64 or plies.dtype != np.uint16
                    or any(array.ndim != 1 for array in self._indices)
                    or len(offsets) != len(games) + 1 or offsets[0] != 0
                    or offsets[-1] != len(plies) or len(plies) != len(self)
                    or np.any(np.diff(offsets) < 7)):
                raise ValueError("invalid compact sample indices")
            if self.label_dir is not None:
                path = self.label_dir / 'value.npy'
                self._values = np.load(path, mmap_mode='r', allow_pickle=False)
                if self._values.dtype != np.float32 or self._values.shape != (len(self),):
                    raise ValueError('invalid teacher value shape/dtype')
            self._pid = os.getpid()
        except Exception:
            self.close()
            raise

    def _actions(self, games, plies):
        actions = np.empty(len(games), np.int64)
        shards = self._cache.shard_ids[games]
        for shard in np.unique(shards):
            shard = int(shard)
            if shard not in self._moves:
                path = self.cache_dir / f'shard_{shard:05d}_moves.npy'
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.dtype != np.int16 or array.ndim != 1:
                    _close_mmap(array)
                    raise ValueError("invalid move shard")
                self._moves[shard] = array
            self._moves.move_to_end(shard)
            rows = np.flatnonzero(shards == shard)
            actions[rows] = self._moves[shard][self._cache.offsets[games[rows]] + plies[rows]]
            if len(self._moves) > self.max_open_shards:
                _, old = self._moves.popitem(last=False)
                _close_mmap(old)
        if np.any((actions < 0) | (actions > 361)):
            raise ValueError("unsupported action ID in cache")
        return actions

    def __getitems__(self, tasks):
        tasks = [task if isinstance(task, tuple) else (task, 0) for task in tasks]
        indices = np.asarray([operator.index(task[0]) for task in tasks], np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("sample index out of range")
        self._ensure_open()
        game_ids, offsets, ply_ids = self._indices
        games = game_ids[np.searchsorted(offsets, indices, side='right') - 1]
        plies = ply_ids[indices]
        if np.any(plies < self.record['identity']['positions']['min_ply']):
            raise ValueError("sample precedes minimum effective ply")
        inputs, colors = pre_action_inputs(self._cache, games, plies)
        actions = self._actions(games, plies)
        output = []
        for row, (index, task) in enumerate(zip(indices, tasks)):
            transform = d4_id(self.seed, operator.index(task[1]), int(index)) if self.augment else 0
            output.append({'planes': apply_d4(inputs[row], transform),
                'policy_target': transform_policy(int(actions[row]), transform),
                'rank_id': int(self._cache.rank_ids[games[row]]),
                'player_color': int(colors[row]), 'game_index': int(games[row]),
                'ply': int(plies[row]), 'sample_index': int(index), 'd4': transform})
            if self._values is not None:
                value = float(self._values[index])
                if not np.isfinite(value) or abs(value) > 1:
                    raise ValueError('invalid teacher value')
                output[-1]['value_target'] = value
                output[-1]['value_weight'] = 1.0
        return output

    def __getitem__(self, task):
        return self.__getitems__([task])[0]
