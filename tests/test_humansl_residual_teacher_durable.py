"""Durable teacher writes: chunk commits, legacy-row revalidation and disk re-read verification."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from go_aicup.tasks.strength.humansl_residual_b2 import teacher as module
from go_aicup.tasks.strength.humansl_residual_b2.teacher import TeacherLabels, value_digest


def setup(tmp_path, monkeypatch, n=6, chunk_games=2):
    table = pd.DataFrame([dict(game_index=i, rank='D', game_id=f'g{i}', store_index=i, hd=0, length=34,
                               player_color=0, role='D', oof_fold=1) for i in range(n)])
    config = dict(data=dict(teacher_dir=str(tmp_path / 'labels'), store_dir='unused'),
                  teacher=dict(num_shards=1, visits=1, chunk_games=chunk_games))
    for key in ('engine', 'model_path', 'analysis_config'):
        path = tmp_path / key; path.write_text(key); config['teacher'][key] = str(path)
    monkeypatch.setattr(module, 'MODEL', tmp_path / 'model_path')
    monkeypatch.setattr(module, 'load_prepared', lambda cfg, **kwargs: dict(identity=dict(sources={}), dataset_id='p'))
    monkeypatch.setattr(module, 'games', lambda cfg, phase: table.copy())
    monkeypatch.setattr(module, 'MoveStore', lambda _: SimpleNamespace(game=lambda _: (np.arange(34), np.arange(34) % 2)))
    queries = []

    class Engine:
        fail_on = None
        def __init__(self, *args): pass
        def close(self): pass
        def analyze(self, query):
            if query['id'] == Engine.fail_on:
                Engine.fail_on = None
                raise RuntimeError('node died')
            queries.append(query['id'])
            for ply in query['analyzeTurns']:
                yield ply, dict(rootInfo=dict(winrate=.75, scoreLead=0., visits=1))
    monkeypatch.setattr(module, 'Engine', Engine)
    return config, table, Engine, queries


def test_crash_keeps_committed_chunks_and_resume_does_not_requery(tmp_path, monkeypatch):
    config, table, Engine, queries = setup(tmp_path, monkeypatch)
    Engine.fail_on = 'D/g3'
    with pytest.raises(RuntimeError, match='node died'):
        module.label(config)
    root = tmp_path / 'labels' / 'shard_00000'
    # g0,g1 committed as a full chunk, g2 as the partial chunk flushed on failure; arrays published.
    assert np.load(root / 'done.npy').tolist() == [True, True, True, False, False, False]
    assert not list((root / 'chunks').glob('*.npz'))
    metrics = module.label(config)
    assert metrics['status'] == 'complete' and metrics['new_games'] == 3 and metrics['verified_games'] == 6
    assert queries == ['D/g0', 'D/g1', 'D/g2', 'D/g3', 'D/g4', 'D/g5']
    assert module.verify_shard(config)['complete']
    assert np.array_equal(TeacherLabels(config).values(table.iloc[4], np.arange(30, 34)),
                          np.asarray([.5, -.5, .5, -.5], np.float32))


def test_chunks_survive_lost_publish_and_are_replayed(tmp_path, monkeypatch):
    config, table, Engine, queries = setup(tmp_path, monkeypatch)
    real = module._publish_arrays
    monkeypatch.setattr(module, '_publish_arrays', lambda *a: (_ for _ in ()).throw(OSError('quota')))
    with pytest.raises(OSError, match='quota'):
        module.label(config)
    root = tmp_path / 'labels' / 'shard_00000'
    assert len(list((root / 'chunks').glob('chunk_*.npz'))) == 3
    monkeypatch.setattr(module, '_publish_arrays', real)
    metrics = module.label(config)
    assert metrics['status'] == 'complete' and metrics['new_games'] == 0 and len(queries) == 6
    assert not list((root / 'chunks').glob('*.npz'))


def test_legacy_unverifiable_rows_are_discarded_and_relabelled(tmp_path, monkeypatch):
    config, table, Engine, queries = setup(tmp_path, monkeypatch)
    module.label(config)
    root = tmp_path / 'labels' / 'shard_00000'
    done = np.load(root / 'done.npy'); checksums = np.load(root / 'checksums.npy')
    # Mimic the lost-memmap state: flags persisted, checksum never did.
    checksums[[1, 4]] = 0
    np.save(root / 'checksums.npy', checksums); np.save(root / 'done.npy', done)
    metrics = module.label(config)
    assert metrics['discarded_games'] == 2 and metrics['new_games'] == 2 and metrics['status'] == 'complete'
    assert queries[-2:] == ['D/g1', 'D/g4']


def test_disk_reread_mismatch_is_never_complete(tmp_path, monkeypatch):
    config, table, Engine, queries = setup(tmp_path, monkeypatch)
    real = module._publish_arrays

    def lossy(out, values, done, checksums):
        broken = done.copy(); broken[2] = False
        real(out, values, broken, checksums)
    monkeypatch.setattr(module, '_publish_arrays', lossy)
    with pytest.raises(ValueError, match='re-read'):
        module.label(config)
    root = tmp_path / 'labels' / 'shard_00000'
    assert __import__('json').loads((root / 'metrics.json').read_text())['status'] != 'complete'
    # Chunks are kept because verification failed, so an honest rerun recovers without new queries.
    monkeypatch.setattr(module, '_publish_arrays', real)
    assert module.label(config)['status'] == 'complete' and len(queries) == 6
