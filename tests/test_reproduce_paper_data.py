"""Synthetic-only SE preparation and state/action alignment checks."""
import csv
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from go_aicup.data.board_cache import BoardCache, RANKS, convert_training
from go_aicup.features.sgf_planes import sgf_to_planes
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.prepare import prepare
from go_aicup.tasks.strength.reproduce_paper.dataset import (
    PaperDataset, pre_action_inputs, soft_value_target, transform_policy)
from go_aicup.tasks.strength.shared.dataset import apply_d4


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp('paper')
    raw = root / 'raw'
    raw.mkdir()
    prefix = ['ba', 'aa', 'pp', 'qq', 'ab', 'pd', 'aa']
    coords = prefix + [chr(97 + col) + chr(97 + row) for row in range(8, 12) for col in range(12)]
    sgf = '(;SZ[19]' + ''.join(f';{"B" if ply % 2 == 0 else "W"}[{coord}]'
                               for ply, coord in enumerate(coords)) + ')'
    for rank in RANKS:
        with (raw / f'train_{rank}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
            writer.writeheader()
            for game in range(4):
                writer.writerow(dict(player_id=f'{rank}_p', game_id=str(game), rank=rank,
                                     color='B' if game % 2 == 0 else 'W', sgf_content=sgf))
    cache_dir = root / 'cache'
    convert_training(raw, cache_dir, workers=1, games_per_shard=3, expected_games_per_rank=4)
    config = read_config()
    config['paths']['cache_dir'] = str(cache_dir)
    config['paths']['dataset_dir'] = str(root / 'prepared')
    config['split'] = dict(expected_games_per_rank=4, train_games_per_rank=2)
    path = prepare(config)
    return config, path, sgf


def test_default_seed_and_split_indices(prepared):
    config, path, _ = prepared
    assert read_config()['seed'] == 2026
    record = json.loads(path.read_text())
    assert record['identity']['seed'] == 2026
    assert record['player_overlap']['shared_players'] == 10
    table = pd.read_parquet(path.parent / 'games.parquet')
    assert not set(table.loc[table.split == 'train', 'game_index']) & set(table.loc[table.split == 'valid', 'game_index'])
    cache = BoardCache(Path(config['paths']['cache_dir']))
    try:
        for split in ('train', 'valid'):
            games, offsets, plies = [np.load(path.parent / f'{split}_{suffix}.npy') for suffix in ('game', 'offsets', 'ply')]
            assert record['splits'][split]['per_rank_games'] == [2] * 10
            assert len(plies) == offsets[-1]
            for row, game in enumerate(games):
                expected = np.flatnonzero((cache.colors(int(game)) == cache.player_colors[game]) & (np.arange(cache.lengths[game]) >= 30))
                np.testing.assert_array_equal(plies[offsets[row]:offsets[row + 1]], expected)
    finally:
        cache.close()
    assert prepare(config) == path
    changed = json.loads(json.dumps(config))
    changed['seed'] = 2027
    with pytest.raises(ValueError, match='identity changed'):
        prepare(changed)


def test_pre_action_replay_padding_capture_and_white_view(prepared):
    config, _, sgf = prepared
    cache = BoardCache(Path(config['paths']['cache_dir']))
    replay = sgf_to_planes(sgf)
    try:
        games = np.array([0, 1, 0, 1, 0, 1])
        plies = np.array([0, 1, 6, 7, 30, 31])
        inputs, colors = pre_action_inputs(cache, games, plies)
        for sample, (ply, color) in enumerate(zip(plies, colors)):
            for history in range(8):
                before = ply - 1 - history
                expected = replay[before] if before >= 0 else np.zeros((2, 19, 19), np.uint8)
                if color == 1:
                    expected = expected[::-1]
                np.testing.assert_array_equal(inputs[sample, 2 * history:2 * history + 2], expected)
            assert np.all(inputs[sample, 16] == (color == 0))
            assert np.all(inputs[sample, 17] == (color == 1))
        assert not inputs[0].any(axis=(1, 2))[:16].any()
        # White aa was captured at ply 4; before ply 6 it is empty.
        assert replay[3, 1, 0, 0] == 1 and inputs[2, 1, 0, 0] == 0
        assert replay[6, 0, 0, 0] == 1 and inputs[2, 0, 0, 0] == 0
        with pytest.raises(ValueError, match='opponent'):
            pre_action_inputs(cache, np.array([0]), np.array([31]))
    finally:
        cache.close()


def test_all_d4_policy_coordinates_and_pass():
    for transform in range(8):
        for action in range(361):
            board = np.zeros((19, 19), np.uint8)
            board.flat[action] = 1
            assert np.flatnonzero(apply_d4(board, transform))[0] == transform_policy(action, transform)
        assert transform_policy(361, transform) == 361
    with pytest.raises(ValueError):
        transform_policy(0, 8)


def test_dataset_lazy_batch_pickle_and_augmentation(prepared):
    config, path, _ = prepared
    dataset = PaperDataset(path.parent, max_open_shards=1)
    augmented = PaperDataset(path.parent, augment=True, max_open_shards=1)
    try:
        indices = [len(dataset) - 1, 0, len(dataset) // 2]
        batch = dataset.__getitems__(indices)
        for index, plain in zip(indices, batch):
            actual = augmented[(index, 3)]
            np.testing.assert_array_equal(actual['planes'], apply_d4(plain['planes'], actual['d4']))
            assert actual['policy_target'] == transform_policy(plain['policy_target'], actual['d4'])
            assert 'value_target' not in actual
            np.testing.assert_array_equal(augmented[(index, 3)]['planes'], actual['planes'])
        restored = pickle.loads(pickle.dumps(augmented))
        try:
            np.testing.assert_array_equal(restored[(0, 3)]['planes'], augmented[(0, 3)]['planes'])
        finally:
            restored.close()
        with pytest.raises(IndexError):
            dataset[-1]
    finally:
        dataset.close()
        augmented.close()


def test_value_perspective_and_invalid_teacher():
    np.testing.assert_allclose(soft_value_target([0.8, 0.8, 0.5], [0, 1, 0]), [0.6, -0.6, 0])
    for probability in (float('nan'), float('inf'), -0.1, 1.1):
        with pytest.raises(ValueError):
            soft_value_target(probability, 0)
    with pytest.raises(ValueError):
        soft_value_target(0.8, 2)


def test_effective_position_shortage_fails(prepared, tmp_path):
    config, _, _ = prepared
    changed = json.loads(json.dumps(config))
    changed['paths']['dataset_dir'] = str(tmp_path / 'short')
    changed['positions']['min_ply'] = 50
    with pytest.raises(ValueError, match='need seven'):
        prepare(changed)


def test_dataloader_spawn_matches_serial(prepared):
    torch = pytest.importorskip('torch')
    _, path, _ = prepared
    dataset = PaperDataset(path.parent, augment=True, max_open_shards=1)
    tasks = [(0, 2), (len(dataset) - 1, 2), (1, 2)]
    try:
        serial = list(torch.utils.data.DataLoader(dataset, batch_size=2, sampler=tasks))
        parallel = list(torch.utils.data.DataLoader(dataset, batch_size=2, sampler=tasks,
                            num_workers=2, multiprocessing_context='spawn'))
        for expected, actual in zip(serial, parallel):
            for key in expected:
                torch.testing.assert_close(actual[key], expected[key])
    finally:
        dataset.close()
