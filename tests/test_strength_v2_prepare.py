"""V2 player-disjoint split, whole-game monitors and shared preparation."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from go_aicup.data.cv import make_strength_v2_split, strength_v2_overlap
from go_aicup.tasks.strength.shared.prepare import prepare as prepare_shared, select_monitoring_games
from go_aicup.tasks.strength.shared.runtime import check_index_hashes, load_dataset_manifest
from go_aicup.tasks.strength.v2.config import DEFAULT_CONFIG, read_config
from go_aicup.tasks.strength.v2.prepare import prepare

RANKS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']


def test_production_config_and_version_validation(tmp_path):
    config = read_config()
    assert config['seed'] == 2027
    assert config['training']['global_batch_size'] == 65536
    assert config['training']['epochs'] == 1
    assert config['split'] == {'policy': 'player_disjoint', 'seen_holdout_fraction': 0.1}
    assert config['monitoring'] == {'games_per_rank': 2500, 'score_unit': 'game', 'decoder': 'argmax'}
    assert config['evaluation']['question_decoder'] == 'score_rank'
    assert Path(config['paths']['output_dir']).name == 'v2_strength'
    for change, message in ((('split', 'policy', 'game'), 'player-disjoint'),
                            (('split', 'seen_holdout_fraction', 1.0), 'player-disjoint'),
                            (('monitoring', 'games_per_rank', 0), 'positive integer')):
        raw = json.loads(DEFAULT_CONFIG.read_text())
        raw[change[0]][change[1]] = change[2]
        path = tmp_path / 'invalid.json'
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match=message):
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
    train = select_monitoring_games(table, 2027, 5, 'train')
    assert (table.set_index('game_index').loc[train, 'split'] == 'train').all()
    with pytest.raises(ValueError, match='needs at least'):
        select_monitoring_games(table, 2027, 41)


def synthetic_meta():
    """Per rank: 10 heavy players (40 games per colour) and 15 light (12)."""
    rows = []
    for rank_id, rank in enumerate(RANKS):
        for player in range(25):
            per_color = 40 if player < 10 else 12
            for color in ('B', 'W'):
                for game in range(per_color):
                    length = 4 + (game + player) % 7
                    start = 0 if color == 'B' else 1
                    rows.append(dict(rank=rank, rank_id=rank_id, game_id=f'{player}{color}{game}',
                                     player_id=f'p{rank}_{player}', color=color,
                                     player_color=start, length=length,
                                     player_ply_count=len(range(start, length, 2))))
    meta = pd.DataFrame(rows)
    meta.insert(0, 'game_index', np.arange(len(meta)))
    return meta


def test_player_disjoint_split_properties():
    meta = synthetic_meta()
    table = make_strength_v2_split(meta, 2027, seen_holdout_fraction=0.5)
    assert set(table['split']) == {'train', 'valid_seen', 'valid_unseen', 'excluded'}
    players = {name: set(group['player_id']) for name, group in table.groupby('split')}
    assert not players['train'] & players['valid_unseen']
    assert not players['excluded'] & (players['train'] | players['valid_unseen'] | players['valid_seen'])
    assert players['valid_seen'] <= players['train']
    np.testing.assert_array_equal(table.loc[table.split == 'valid_unseen', 'fold'].unique(), [0])
    np.testing.assert_array_equal(table.loc[table.split == 'excluded', 'fold'].unique(), [-1])
    trained = table[table.split.isin(['train', 'valid_seen'])]
    counts = trained.groupby(['player_id', 'color']).agg(
        total=('split', 'size'), held=('split', lambda s: (s == 'valid_seen').sum()))
    np.testing.assert_array_equal(counts['held'], np.floor(counts['total'] * 0.5))
    overlap = strength_v2_overlap(table)
    assert overlap['valid_unseen_overlap_fraction'] == 0.0
    assert overlap['valid_seen_overlap_fraction'] == 1.0
    pd.testing.assert_frame_equal(table, make_strength_v2_split(meta.sample(frac=1, random_state=1), 2027,
                                                                seen_holdout_fraction=0.5))
    assert not table['split'].equals(make_strength_v2_split(meta, 2028, seen_holdout_fraction=0.5)['split'])
    with pytest.raises(ValueError, match='between zero and one'):
        make_strength_v2_split(meta, 2027, seen_holdout_fraction=0)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import go_aicup.tasks.strength.shared.prepare as shared
    meta = synthetic_meta()
    cache = tmp_path / 'cache'
    cache.mkdir()
    (cache / 'manifest.json').write_text(json.dumps({'build_id': 'synthetic-v2'}))
    meta.to_parquet(cache / 'games.parquet', index=False)

    class SyntheticCache:
        def __init__(self, directory, **kwargs):
            self.directory = directory
            self.manifest = {'build_id': 'synthetic-v2'}
            self.rank_ids = meta.rank_id.to_numpy()
            self.player_colors = meta.player_color.to_numpy()

        def colors(self, game):
            return np.arange(meta.iloc[game].length, dtype=np.int8) % 2

        def close(self):
            pass

    monkeypatch.setattr(shared, 'BoardCache', SyntheticCache)
    config = read_config()
    config['paths'].update(cache_dir=str(cache), dataset_dir=str(tmp_path / 'v2'))
    config['split']['seen_holdout_fraction'] = 0.5
    config['questions'] = {'validation': 40}
    config['monitoring']['games_per_rank'] = 5
    prepare(config)
    return config, meta


def test_indices_questions_and_monitors_follow_their_split(prepared, monkeypatch):
    config, meta = prepared
    directory = Path(config['paths']['dataset_dir'])
    manifest = load_dataset_manifest(config)
    check_index_hashes(config, manifest)
    table = pd.read_parquet(directory / 'strength_v2_games.parquet').set_index('game_index')
    assert set(manifest['sets']) == {'train', 'valid_unseen', 'valid_seen', 'monitor_unseen', 'monitor_seen'}
    assert manifest['player_overlap']['valid_unseen_overlap_fraction'] == 0.0
    used = set()
    for name, source in (('train', 'train'), ('valid_unseen', 'valid_unseen'), ('valid_seen', 'valid_seen'),
                         ('monitor_unseen', 'valid_unseen'), ('monitor_seen', 'valid_seen')):
        games = np.load(directory / f'{name}_game.npy', allow_pickle=False)
        plies = np.load(directory / f'{name}_ply.npy', allow_pickle=False)
        unique = np.unique(games)
        assert (table.loc[unique, 'split'] == source).all()
        if not name.startswith('monitor'):
            np.testing.assert_array_equal(unique, np.sort(table.index[table.split == source]))
            used.update(unique.tolist())
        for game in unique[:20]:
            row = meta.iloc[int(game)]
            np.testing.assert_array_equal(plies[games == game],
                                          np.arange(row.player_color, row.length, 2))
        assert manifest['sets'][name]['num_samples'] == len(plies)
    assert not used & set(table.index[table.split == 'excluded'])
    for name in ('monitor_unseen', 'monitor_seen'):
        selected = np.load(directory / f'{name}_games.npy', allow_pickle=False)
        np.testing.assert_array_equal(np.bincount(table.loc[selected, 'rank_id']), [5] * 10)
    for name in ('valid_unseen', 'valid_seen'):
        questions = pd.read_parquet(directory / f'strength_v2_{name}_questions.parquet')
        assert len(questions) == 40
        for row in questions.itertuples():
            assert (table.loc[row.games, 'split'] == name).all()
            assert table.loc[row.games, 'player_id'].nunique() == 1
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    def unexpected_sampling(*args, **kwargs):
        raise AssertionError('completed preparation must not resample')
    monkeypatch.setattr('go_aicup.tasks.strength.shared.prepare.select_monitoring_games', unexpected_sampling)
    prepare(config)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    changed = copy.deepcopy(config)
    changed['split']['seen_holdout_fraction'] = 0.25
    with pytest.raises(ValueError, match='mismatch'):
        load_dataset_manifest(changed)
    with pytest.raises(ValueError, match='settings/code changed'):
        prepare(changed)


def test_game_disjoint_policy_still_writes_v1_layout(prepared):
    config, _ = prepared
    v1 = copy.deepcopy(config)
    v1.pop('monitoring')
    v1.pop('strength_version')
    v1['split'] = {'expected_games_per_rank': 1160, 'train_games_per_rank': 300}
    v1['paths']['dataset_dir'] = str(Path(config['paths']['dataset_dir']).parent / 'v1')
    prepare_shared(v1)
    directory = Path(v1['paths']['dataset_dir'])
    manifest = load_dataset_manifest(v1)
    assert set(manifest['sets']) == {'train', 'valid'}
    assert (directory / 'strength_v1_games.parquet').exists()
    assert (directory / 'strength_v1_validation_questions.parquet').exists()
    assert not list(directory.glob('monitor*'))
