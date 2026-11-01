"""Pure protocol tests; added without execution at the user's request."""
import numpy as np
import pytest

from go_aicup.tasks.strength.reproduce_paper.label_value import make_query, parse_root


def test_pre_action_turns_keep_history_and_exclude_future():
    moves = np.array([0, 20, 40, 60, 80])
    colors = np.array([0, 1, 0, 1, 0])
    query = make_query('g', moves, colors, [1, 3], 32, 'japanese', 6.5)
    assert query['analyzeTurns'] == [1, 3]
    assert query['moves'] == [['B', 'A19'], ['W', 'B18'], ['B', 'C17']]
    assert query['overrideSettings']['reportAnalysisWinratesAs'] == 'BLACK'
    assert query['maxVisits'] == 32


def test_real_pass_preserved_no_synthetic_pass():
    query = make_query('g', [361, 0, 20], [0, 1, 0], [2], 1, 'japanese', 6.5)
    assert query['moves'] == [['B', 'pass'], ['W', 'A19']]


def test_value_perspective_and_optional_uncertainty():
    result = {'rootInfo': {'winrate': 0.8, 'scoreLead': 5.0, 'visits': 32}}
    black = parse_root(result, 0)
    white = parse_root(result, 1)
    assert black[1] == pytest.approx(0.6)
    assert white[1] == pytest.approx(-0.6)
    assert black[2] == white[2] == 5.0
    assert np.isnan(black[3])


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_probabilities_are_not_filled(probability):
    with pytest.raises(ValueError):
        parse_root({'rootInfo': {'winrate': probability, 'scoreLead': 0, 'visits': 1}}, 0)


def test_duplicate_or_post_game_turn_rejected():
    for plies in ([1, 1], [2]):
        with pytest.raises(ValueError):
            make_query('g', [0, 20], [0, 1], plies, 1, 'japanese', 6.5)


def test_sharded_retry_export_and_mixed_teacher_rejection(tmp_path, monkeypatch):
    """Small fake-engine integration: never starts a real engine, even if run."""
    from argparse import Namespace
    import json
    import pandas as pd
    from go_aicup.artifacts import file_hash
    from go_aicup.tasks.strength.reproduce_paper import label_value as module

    prepared, cache = tmp_path / 'prepared', tmp_path / 'cache'
    prepared.mkdir()
    cache.mkdir()
    (cache / 'manifest.json').write_text('{}')
    (prepared / 'manifest.json').write_text('{}')
    engine_file = tmp_path / 'engine'
    engine_file.write_text('fake engine/model/config')
    monkeypatch.setattr(module, 'MODEL', engine_file)
    games = np.array([0, 1], np.uint32)
    offsets = np.array([0, 7, 14], np.int64)
    plies = np.tile(np.arange(30, 44, 2, dtype=np.uint16), 2)
    table = pd.DataFrame({'game_index': games, 'rank': ['D', 'C'],
        'game_id': ['same-id', 'same-id'], 'split': ['train', 'train'],
        'shard_id': [0, 0], 'local_offset': [0, 44], 'length': [44, 44],
        'player_color': [0, 1]}).set_index('game_index')
    record = {'build_id': 'test', 'identity': {
        'cache_manifest_sha256': file_hash(cache / 'manifest.json'),
        'positions': {'min_ply': 30}}}
    monkeypatch.setattr(module, 'source_data', lambda *_: (record, table, games, offsets, plies))
    np.save(cache / 'shard_00000_moves.npy', np.tile(np.arange(44, dtype=np.int16), 2))
    np.save(cache / 'shard_00000_colors.npy',
            np.r_[np.arange(44) % 2, 1 - np.arange(44) % 2].astype(np.int8))
    calls = []
    fail_once = [True]

    class FakeEngine:
        def __init__(self, *args):
            pass

        def analyze(self, query):
            calls.append(query)
            for turn in query['analyzeTurns']:
                yield turn, {'rootInfo': {'winrate': 0.75, 'scoreLead': 4., 'visits': 8}}
                if fail_once[0]:
                    fail_once[0] = False
                    raise RuntimeError('simulated interruption after one result')

        def close(self):
            pass

    monkeypatch.setattr(module, 'Engine', FakeEngine)
    config = {'paths': {'dataset_dir': str(prepared), 'cache_dir': str(cache)},
              'teacher': {'target': 'per_effective_pre_action_position',
                          'visits': None, 'rules': 'japanese', 'komi': 6.5}}
    args = Namespace(split='train', visits=8, num_shards=2, shard=0,
        retries=1, timeout=10, limit_games=None, model=engine_file,
        katago=engine_file, engine_config=engine_file, out=tmp_path / 'labels')
    module.label(config, args)
    assert calls[0]['analyzeTurns'] == list(range(30, 44, 2))
    assert calls[1]['analyzeTurns'] == list(range(32, 44, 2))
    before = len(calls)
    module.label(config, args)
    assert len(calls) == before  # successful positions never re-analyzed
    with pytest.raises(ValueError, match='missing/duplicate'):
        module.export_labels(config, args)
    args.shard = 1
    module.label(config, args)
    manifest_path = args.out / 'train/shard_00001/manifest.json'
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest['identity']['visits'] = 16
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='mix teacher'):
        module.export_labels(config, args)
    manifest_path.write_text(original)
    real_remove = module.shutil.rmtree
    removals = []

    def interrupted_cleanup(path):
        removals.append(path)
        if len(removals) == 2:
            raise OSError('simulated cleanup interruption')
        real_remove(path)

    monkeypatch.setattr(module.shutil, 'rmtree', interrupted_cleanup)
    with pytest.raises(OSError, match='cleanup interruption'):
        module.export_labels(config, args)
    assert (args.out / 'train/value.npy').is_file()
    assert (args.out / 'train/labels_manifest.json').is_file()
    monkeypatch.setattr(module.shutil, 'rmtree', real_remove)
    output = module.export_labels(config, args)
    np.testing.assert_array_equal(np.load(output), [0.5] * 7 + [-0.5] * 7)
    assert sorted(p.name for p in output.parent.iterdir()) == ['value.npy']
    assert module.export_labels(config, args) == output
    with pytest.raises(ValueError, match='already published'):
        module.label(config, args)
    assert sorted(p.name for p in output.parent.iterdir()) == ['value.npy']
