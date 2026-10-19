"""Whole-game V2 monitoring coverage and shared preparation compatibility."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from go_aicup.tasks.strength.shared.prepare import prepare as prepare_shared, select_monitoring_games
from go_aicup.tasks.strength.shared.runtime import check_index_hashes, load_dataset_manifest
from go_aicup.tasks.strength.v2.config import DEFAULT_CONFIG, read_config
from go_aicup.tasks.strength.v2.prepare import prepare


def test_production_config_and_version_validation(tmp_path):
    config = read_config()
    assert config['seed'] == 2027
    assert config['training']['global_batch_size'] == 65536
    assert config['training']['epochs'] == 1
    assert config['monitoring'] == {'games_per_rank': 5000, 'score_unit': 'game', 'decoder': 'argmax'}
    assert config['evaluation']['question_decoder'] == 'score_rank'
    assert Path(config['paths']['output_dir']).name == 'v2_strength'
    raw = json.loads(DEFAULT_CONFIG.read_text())
    raw['monitoring']['games_per_rank'] = 20001
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='fit the validation split'):
        read_config(path)
    with pytest.raises(ValueError, match='V2 monitoring'):
        prepare({'strength_version': 1})


def test_balanced_sampling_is_by_game_fixed_and_order_independent():
    table = pd.DataFrame({'game_index': np.arange(800), 'rank_id': np.repeat(np.arange(10), 80),
                          'split': np.tile(['train'] * 40 + ['valid'] * 40, 10)})
    selected = select_monitoring_games(table, 2027, 5)
    assert len(selected) == len(np.unique(selected)) == 50
    assert (table.set_index('game_index').loc[selected, 'split'] == 'valid').all()
    np.testing.assert_array_equal(np.bincount(table.loc[selected, 'rank_id']), [5] * 10)
    np.testing.assert_array_equal(selected, select_monitoring_games(table.sample(frac=1), 2027, 5))
    assert not np.array_equal(selected, select_monitoring_games(table, 2028, 5))
    with pytest.raises(ValueError, match='needs at least'):
        select_monitoring_games(table, 2027, 41)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import go_aicup.tasks.strength.shared.prepare as shared
    ranks = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
    meta = pd.DataFrame([
        dict(game_index=rank * 80 + game, rank=ranks[rank], rank_id=rank,
             game_id=f'g{game}', player_id=f'p{rank}', color='B' if rank % 2 == 0 else 'W',
             length=4 + game % 7,
             player_ply_count=len(range(rank % 2, 4 + game % 7, 2)))
        for rank in range(10) for game in range(80)])
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'build_id': 'synthetic-v2'}))
    meta.to_parquet(cache / 'games.parquet', index=False)

    class SyntheticCache:
        def __init__(self, directory, **kwargs):
            self.directory = directory
            self.manifest = {'build_id': 'synthetic-v2'}
            self.rank_ids = meta.rank_id.to_numpy()
            self.player_colors = self.rank_ids % 2

        def colors(self, game):
            return np.arange(meta.iloc[game].length, dtype=np.int8) % 2

        def close(self):
            pass

    monkeypatch.setattr(shared, 'BoardCache', SyntheticCache)
    config = read_config()
    config['paths'].update(cache_dir=str(cache), dataset_dir=str(tmp_path / 'v2'))
    config['split'] = {'expected_games_per_rank': 80, 'train_games_per_rank': 40}
    config['questions'] = {'validation': 40}
    config['monitoring']['games_per_rank'] = 5
    prepare(config)
    return config, meta


def test_indices_cover_every_player_move_and_reuse_saved_selection(prepared, monkeypatch):
    config, meta = prepared
    directory = Path(config['paths']['dataset_dir'])
    manifest = load_dataset_manifest(config)
    check_index_hashes(config, manifest)
    selected = np.load(directory / 'monitor_games.npy', allow_pickle=False)
    games = np.load(directory / 'monitor_game.npy', allow_pickle=False)
    plies = np.load(directory / 'monitor_ply.npy', allow_pickle=False)
    np.testing.assert_array_equal(np.unique(games), selected)
    for game in selected:
        row = meta.iloc[int(game)]
        np.testing.assert_array_equal(plies[games == game], np.arange(row.rank_id % 2, row.length, 2))
    assert len(plies) == meta.iloc[selected].player_ply_count.sum()
    assert manifest['sets']['monitor']['num_games'] == 50
    assert manifest['sets']['monitor']['num_samples'] == len(plies)
    assert manifest['player_overlap']['valid_player_overlap_fraction'] == 1.0
    assert set(manifest['sets']) == {'train', 'valid', 'monitor'}
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    def unexpected_sampling(*args, **kwargs):
        raise AssertionError('completed preparation must not resample')
    monkeypatch.setattr('go_aicup.tasks.strength.shared.prepare.select_monitoring_games', unexpected_sampling)
    prepare(config)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    changed = copy.deepcopy(config)
    changed['monitoring']['games_per_rank'] = 4
    with pytest.raises(ValueError, match='mismatch'):
        load_dataset_manifest(changed)
    with pytest.raises(ValueError, match='settings/code changed'):
        prepare(changed)
    selected[0] += 1
    np.save(directory / 'monitor_games.npy', selected, allow_pickle=False)
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare(config)


def test_same_split_and_fixed_questions_as_shared_v1(prepared):
    config, _ = prepared
    v1 = copy.deepcopy(config)
    v1.pop('monitoring')
    v1.pop('strength_version')
    v1['paths']['dataset_dir'] = str(Path(config['paths']['dataset_dir']).parent / 'v1')
    prepare_shared(v1)
    first, second = Path(config['paths']['dataset_dir']), Path(v1['paths']['dataset_dir'])
    for name in ('strength_v1_games.parquet', 'strength_v1_validation_questions.parquet',
                 'train_game.npy', 'train_ply.npy', 'valid_game.npy', 'valid_ply.npy'):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = load_dataset_manifest(v1)
    assert set(manifest['sets']) == {'train', 'valid'}
    assert not list(second.glob('monitor*'))
