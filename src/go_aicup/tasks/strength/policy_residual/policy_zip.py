"""Read the persistent per-move policy cache straight from its ZIP into RAM.

The 792691 cache (TRAIN + monitor, 24,000 games, 5,343,778 positions) is stored only as
``policy_cnn/cached_AB_mondev_seed2027.zip``; nothing is extracted to disk.  Members are
inflated in memory and copied into one preallocated array per field.  With ``verify``
(default) every array is checked through the recorded hash chain:
policy/complete.json -> shard complete.json (file SHA-256) -> each .npy (SHA-256).

Fields per position: logp float16 (10, 362) = log p of all 362 moves under the 10 rank
hypotheses, q float32 (10,), target int16, side uint8 (1 = opponent move).
Full logp needs about 38.7 GB of RAM; pass ``fields`` / ``roles`` to load less.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT

NAME = 'cached_AB_mondev_seed2027'
ARCHIVE = ROOT / 'data/processed/strength_features/humansl/policy_cnn' / f'{NAME}.zip'
IDENTITY = 'cc1ca4f6da211644301c16ed9c04aaae038baf5f6b64f41fd56fdf71cde94f72'
FIELDS = {'logp': (np.float16, (10, 362)), 'q': (np.float32, (10,)),
          'target': (np.int16, ()), 'side': (np.uint8, ())}
SELECTION_FILES = ('games', 'reuse_games', 'supplement_games',
                   'train_questions', 'monitor_questions', 'dev_questions')


@dataclass
class PolicyCache:
    games: pd.DataFrame    # one row per loaded game: selection columns + start/length into arrays
    arrays: dict           # field -> array with one row per loaded position


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Reader:
    """One ZipFile handle per thread; ZipFile objects are not safe to share."""

    def __init__(self, path: Path):
        self.path = path
        self.local = threading.local()

    def read(self, member: str) -> bytes:
        if not hasattr(self.local, 'zip'):
            self.local.zip = zipfile.ZipFile(self.path)
        return self.local.zip.read(f'{NAME}/{member}')


def _npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def selection(name: str = 'games', path: Path = ARCHIVE) -> pd.DataFrame:
    """Selection / question tables (also covers dev, which has no policy shards)."""
    if name not in SELECTION_FILES:
        raise ValueError(f'unknown selection table: {name}')
    return pd.read_parquet(io.BytesIO(_Reader(path).read(f'{name}.parquet')))


def load(path: Path = ARCHIVE, *, fields=tuple(FIELDS), roles=('train', 'monitor'),
         verify: bool = True, threads: int = 16) -> PolicyCache:
    fields = tuple(fields)
    if set(fields) - set(FIELDS):
        raise ValueError(f'unknown fields: {set(fields) - set(FIELDS)}')
    reader = _Reader(Path(path))
    complete = json.loads(reader.read('policy/complete.json'))
    if complete['status'] != 'complete' or complete['identity'] != IDENTITY:
        raise ValueError('policy cache incomplete or identity changed')
    table = pd.read_parquet(io.BytesIO(reader.read('games.parquet')))
    wanted = table[table.role.isin(roles)].set_index('game_index')
    names = sorted(complete['shards'])

    def index(shard):
        marker_bytes = reader.read(f'policy/{shard}/complete.json')
        if verify and _sha(marker_bytes) != complete['shards'][shard]:
            raise ValueError(f'shard marker hash mismatch: {shard}')
        marker = json.loads(marker_bytes)
        small = {}
        for name in ('offsets', 'game_index'):
            data = reader.read(f'policy/{shard}/{name}.npy')
            if verify and _sha(data) != marker['files'][name]:
                raise ValueError(f'hash mismatch: {shard}/{name}')
            small[name] = _npy(data)
        return marker, small['offsets'], small['game_index']

    with ThreadPoolExecutor(threads) as pool:
        indexed = list(pool.map(index, names))
    plan, rows, total = [], [], 0
    for shard, (marker, offsets, ids) in zip(names, indexed):
        keep = np.isin(ids, wanted.index)
        if not keep.any():
            continue
        lengths = np.diff(offsets)
        position_mask = np.repeat(keep, lengths)
        plan.append((shard, marker, position_mask, total))
        for g, n in zip(ids[keep], lengths[keep]):
            rows.append((int(g), total, int(n)))
            total += int(n)
    games = pd.DataFrame(rows, columns=['game_index', 'start', 'length_cached'])
    if len(games) != len(wanted) or not games.game_index.is_unique:
        raise ValueError('requested games missing from or duplicated in the cache')
    games = games.join(wanted, on='game_index')
    if not np.array_equal(games.length, games.length_cached):
        raise ValueError('cached position counts differ from selection lengths')
    games = games.drop(columns='length_cached')

    arrays = {f: np.empty((total, *FIELDS[f][1]), dtype=FIELDS[f][0]) for f in fields}

    def fill(item):
        shard, marker, mask, start = item
        stop = start + int(mask.sum())
        for f in fields:
            data = reader.read(f'policy/{shard}/{f}.npy')
            if verify and _sha(data) != marker['files'][f]:
                raise ValueError(f'hash mismatch: {shard}/{f}')
            array = _npy(data)
            del data
            arrays[f][start:stop] = array[mask]

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(fill, plan))
    return PolicyCache(games=games, arrays=arrays)
