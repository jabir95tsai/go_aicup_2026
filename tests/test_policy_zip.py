import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from go_aicup.tasks.strength.policy_residual import policy_zip as pz


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _build(path, corrupt=False):
    rng = np.random.default_rng(0)
    games = pd.DataFrame(dict(game_index=[10, 11, 12, 13, 14], length=np.array([3, 2, 4, 1, 2], np.int32),
                              role=['train', 'monitor', 'train', 'dev', 'monitor'], rank_id=np.arange(5, dtype=np.uint8)))
    shards, layout, truth = {}, {'unit_000_shard_000': [10, 11], 'unit_001_shard_000': [12, 14]}, {}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        def put(name, data):
            archive.writestr(f'{pz.NAME}/{name}', data)
        buffer = io.BytesIO()
        games.to_parquet(buffer)
        put('games.parquet', buffer.getvalue())
        for shard, ids in layout.items():
            lengths = games.set_index('game_index').length.loc[ids].to_numpy()
            n = int(lengths.sum())
            values = dict(logp=rng.standard_normal((n, 10, 362)).astype(np.float16),
                          q=rng.standard_normal((n, 10)).astype(np.float32),
                          target=rng.integers(0, 362, n).astype(np.int16), side=(np.arange(n) % 2).astype(np.uint8),
                          offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64), game_index=np.array(ids, np.int64))
            truth[shard] = values
            blobs = {k: _npy(v) for k, v in values.items()}
            marker = json.dumps(dict(files={k: hashlib.sha256(b).hexdigest() for k, b in blobs.items()})).encode()
            if corrupt and shard == 'unit_001_shard_000':
                blobs['q'] = _npy(values['q'] + 1)
            for k, b in blobs.items():
                put(f'policy/{shard}/{k}.npy', b)
            put(f'policy/{shard}/complete.json', marker)
            shards[shard] = hashlib.sha256(marker).hexdigest()
        put('policy/complete.json', json.dumps(dict(status='complete', identity=pz.IDENTITY, shards=shards)))
    return games, truth


def test_load_from_zip(tmp_path):
    path = tmp_path / 'cache.zip'
    games, truth = _build(path)
    cache = pz.load(path, threads=2)
    assert cache.games.game_index.tolist() == [10, 11, 12, 14]
    assert cache.games.role.tolist() == ['train', 'monitor', 'train', 'monitor']
    every = {f: np.concatenate([truth[s][f] for s in sorted(truth)]) for f in pz.FIELDS}
    for f in pz.FIELDS:
        np.testing.assert_array_equal(cache.arrays[f], every[f])
    for row in cache.games.itertuples():
        assert cache.arrays['logp'][row.start:row.start + row.length].shape == (row.length, 10, 362)

    monitor = pz.load(path, roles=('monitor',), fields=('q',), threads=2)
    assert monitor.games.game_index.tolist() == [11, 14]
    assert set(monitor.arrays) == {'q'}
    np.testing.assert_array_equal(monitor.arrays['q'], np.concatenate([truth['unit_000_shard_000']['q'][3:],
                                                                       truth['unit_001_shard_000']['q'][4:]]))
    assert pz.selection('games', path).equals(games)


def test_hash_mismatch_rejected(tmp_path):
    path = tmp_path / 'cache.zip'
    _build(path, corrupt=True)
    with pytest.raises(ValueError, match='hash mismatch'):
        pz.load(path, threads=2)
    pz.load(path, threads=2, verify=False)
