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
    assert record['identity']['split_seed'] == 2026
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
    changed['split']['train_games_per_rank'] = 1
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


def test_training_seed_reuses_shared_split_without_regeneration(prepared, monkeypatch):
    import importlib
    from go_aicup.artifacts import file_hash
    module = importlib.import_module('go_aicup.tasks.strength.reproduce_paper.prepare')
    config, path, _ = prepared
    before = json.loads(path.read_text())
    tracked = [path, *(path.parent / name for name in before['files'])]
    unchanged = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in tracked}
    changed = json.loads(json.dumps(config))
    changed['seed'] = 9090
    changed['model']['channels'] = 128
    changed['paths']['output_dir'] = str(path.parent / 'another_run')
    changed['loader']['max_open_shards'] = 1

    def forbidden_split(*args, **kwargs):
        raise AssertionError('existing shared split must not be regenerated')

    monkeypatch.setattr(module, 'make_strength_v1_split', forbidden_split)
    assert prepare(changed) == path
    for p in tracked:
        assert (file_hash(p), p.stat().st_mtime_ns) == unchanged[str(p)]
    assert 'seed' not in before['identity']
    assert before['identity']['split_seed'] == 2026
    dataset = PaperDataset(path.parent, seed=changed['seed'], augment=True)
    try:
        assert dataset.seed == 9090
        assert dataset[0]['game_index'] == PaperDataset(path.parent)[0]['game_index']
    finally:
        dataset.close()


def test_fresh_build_does_not_use_training_seed(prepared, tmp_path):
    from go_aicup.artifacts import file_hash
    config, path, _ = prepared
    changed = json.loads(json.dumps(config))
    changed['seed'] = 3030
    changed['paths']['dataset_dir'] = str(tmp_path / 'rebuild')
    rebuilt = prepare(changed)
    old = json.loads(path.read_text())
    new = json.loads(rebuilt.read_text())
    assert old['identity'] == new['identity']
    assert old['build_id'] == new['build_id']
    for name in old['files']:
        assert file_hash(path.parent / name) == file_hash(rebuilt.parent / name)


@pytest.mark.parametrize('split_seed', [2027, 2026.0, True])
def test_split_seed_is_fixed(prepared, split_seed):
    config, _, _ = prepared
    changed = json.loads(json.dumps(config))
    changed['split_seed'] = split_seed
    with pytest.raises(ValueError, match='split_seed must be fixed'):
        prepare(changed)


def test_corrupt_shared_index_is_rejected(prepared, tmp_path):
    import shutil
    config, path, _ = prepared
    changed = json.loads(json.dumps(config))
    directory = tmp_path / 'corrupt'
    shutil.copytree(path.parent, directory)
    changed['paths']['dataset_dir'] = str(directory)
    changed['seed'] = 3030
    (directory / 'train_game.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare(changed)


def test_prepare_and_questions_share_data_across_training_seeds(tmp_path):
    from go_aicup.artifacts import file_hash
    from go_aicup.tasks.strength.reproduce_paper.questions import prepare_questions
    raw = tmp_path / 'raw'
    raw.mkdir()
    coords = [chr(97 + col) + chr(97 + row) for row in range(8, 12) for col in range(12)]
    sgf = '(;SZ[19]' + ''.join(f';{"B" if ply % 2 == 0 else "W"}[{coord}]'
                               for ply, coord in enumerate(coords)) + ')'
    for rank in RANKS:
        with (raw / f'train_{rank}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
            writer.writeheader()
            for game in range(80):
                writer.writerow(dict(player_id=f'{rank}_p', game_id=str(game), rank=rank,
                                     color='B' if game % 2 == 0 else 'W', sgf_content=sgf))
    cache = tmp_path / 'cache'
    convert_training(raw, cache, workers=1, games_per_shard=80, expected_games_per_rank=80)
    config = read_config()
    config['paths'].update(cache_dir=str(cache), dataset_dir=str(tmp_path / 'prepared'))
    config['split'] = dict(expected_games_per_rank=80, train_games_per_rank=40)
    config['questions']['validation'] = 40
    # The very first build must use split_seed, even for a different run seed.
    config['seed'] = 3030
    path = prepare(config)
    questions_path = prepare_questions(config)
    before = {str(p): (file_hash(p), p.stat().st_mtime_ns) for p in
              [path, questions_path, path.parent / 'games.parquet', path.parent / 'validation_questions.parquet']}
    config['seed'] = 4040
    assert prepare(config) == path
    assert prepare_questions(config) == questions_path
    for name, expected in before.items():
        p = Path(name)
        assert (file_hash(p), p.stat().st_mtime_ns) == expected
    table = pd.read_parquet(path.parent / 'games.parquet')
    assert len(table[table.split == 'train']) == len(table[table.split == 'valid']) == 400


@pytest.fixture
def labelled(prepared, tmp_path):
    _, path, _ = prepared
    count = json.loads(path.read_text())['splits']['train']['positions']
    values = np.linspace(-1, 1, count, dtype=np.float32)
    np.save(tmp_path / 'value.npy', values)
    dataset = PaperDataset(path.parent, augment=True, label_dir=tmp_path)
    yield dataset, values
    dataset.close()


def test_group_batch_structure_and_value_alignment(labelled):
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler, collate_groups
    dataset, values = labelled
    tasks = next(iter(GroupBatchSampler(dataset, groups_per_update=3, updates=1)))
    samples = dataset.__getitems__(tasks)
    batch = collate_groups(samples)
    assert batch['planes'].shape == (3, 10, 7, 18, 19, 19)
    assert batch['policy_target'].shape == (3, 10, 7)
    indices = batch['sample_index'].numpy()
    np.testing.assert_array_equal(batch['value_target'].numpy(), values[indices])
    assert set(batch['player_color'].flatten().tolist()) == {0, 1}
    assert batch['ply'].min() >= 30
    games = set(dataset._indices[0].tolist())
    assert set(batch['game_index'].flatten().tolist()) <= games
    # Values are already in the player's perspective, including white samples.
    plain = PaperDataset(dataset.directory, label_dir=dataset.label_dir)
    try:
        for sample in samples:
            original = plain[sample['sample_index']]
            np.testing.assert_array_equal(sample['planes'], apply_d4(original['planes'], sample['d4']))
            assert sample['policy_target'] == transform_policy(original['policy_target'], sample['d4'])
            assert sample['value_target'] == values[sample['sample_index']]
    finally:
        plain.close()



def test_group_resume_ddp_and_prefetch_are_reproducible(labelled):
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler
    dataset, _ = labelled
    serial = list(GroupBatchSampler(dataset, groups_per_update=128, updates=3))
    resumed = GroupBatchSampler(dataset, groups_per_update=128, updates=3, start_step=1)
    assert len(resumed) == 2
    assert list(resumed) == serial[1:]
    # Advancing a producer/prefetch iterator must not mutate the consumer cursor.
    iterator = iter(resumed)
    next(iterator)
    next(iterator)
    assert list(resumed) == serial[1:]
    for step in range(3):
        combined = []
        for rank in range(8):
            local = next(iter(GroupBatchSampler(dataset, updates=3, start_step=step,
                                               rank=rank, world_size=8)))
            assert len(local) == 16 * 10 * 7
            combined.extend(local)
        assert combined == serial[step]
    assert serial[0] != serial[1]
    assert list(GroupBatchSampler(dataset, updates=3, start_step=3)) == []


def test_group_sampling_is_uniform_by_game_not_position():
    from types import SimpleNamespace
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler
    # Two games per rank with 7 and 70 positions: either game must get half the draws.
    offsets = np.r_[0, np.cumsum([7, 70] * 10)]
    dataset = SimpleNamespace(seed=2026, split='train', label_dir='labels',
        _ensure_open=lambda: None, _indices=(np.arange(20), offsets, None),
        _cache=SimpleNamespace(rank_ids=np.repeat(np.arange(10), 2)))
    tasks = next(iter(GroupBatchSampler(dataset, groups_per_update=2000, updates=1)))
    indices = np.array([index for index, _ in tasks]).reshape(2000, 10, 7)
    rows = np.searchsorted(offsets, indices, side='right') - 1
    assert np.all(rows == rows[..., :1])
    assert np.all(np.abs((rows[..., 0] % 2).mean(axis=0) - 0.5) < 0.05)
    assert np.all(np.diff(np.sort(indices, axis=-1), axis=-1) > 0)


def test_group_loader_spawn_and_resume_match(labelled):
    torch = pytest.importorskip('torch')
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler, collate_groups
    dataset, _ = labelled
    def loader(workers, start_step=0):
        kwargs = dict(num_workers=workers, collate_fn=collate_groups,
            batch_sampler=GroupBatchSampler(dataset, groups_per_update=2, updates=3, start_step=start_step))
        if workers:
            kwargs.update(multiprocessing_context='spawn', prefetch_factor=2, persistent_workers=True)
        return torch.utils.data.DataLoader(dataset, **kwargs)
    serial = list(loader(0))
    for actual_batches, expected_batches in ((list(loader(2)), serial), (list(loader(2, 1)), serial[1:])):
        assert len(actual_batches) == len(expected_batches)
        for actual, expected in zip(actual_batches, expected_batches):
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_group_invalid_inputs_and_labels_are_rejected(labelled, prepared):
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler, collate_groups
    dataset, _ = labelled
    for kwargs in (dict(world_size=3), dict(rank=8, world_size=8), dict(world_size=0),
                   dict(groups_per_update=0), dict(start_step=-1), dict(updates=0),
                   dict(start_step=4, updates=3)):
        with pytest.raises(ValueError):
            GroupBatchSampler(dataset, **kwargs)
    _, path, _ = prepared
    for unlabelled in (PaperDataset(path.parent), PaperDataset(path.parent, split='valid', label_dir=dataset.label_dir)):
        with pytest.raises(ValueError, match='training data with teacher labels'):
            GroupBatchSampler(unlabelled)
    samples = dataset.__getitems__(next(iter(GroupBatchSampler(dataset, groups_per_update=1, updates=1))))
    with pytest.raises(ValueError, match='complete'):
        collate_groups(samples[:-1])
    for key, value, message in (('rank_id', 9, 'ranks'), ('game_index', -1, 'share a game'),
                                ('player_color', 2, 'share a game'), ('value_target', np.nan, 'teacher'),
                                ('value_target', 2, 'teacher'), ('value_weight', 0, 'teacher'),
                                ('ply', samples[1]['ply'], 'distinct')):
        broken = [dict(sample) for sample in samples]
        broken[0][key] = value
        with pytest.raises(ValueError, match=message):
            collate_groups(broken)
    del samples[0]['value_target']
    with pytest.raises(ValueError, match='teacher labels'):
        collate_groups(samples)
    values = dataset._values
    try:
        dataset._values = np.full(len(dataset), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match='invalid teacher value'):
            dataset[0]
    finally:
        dataset._values = values


@pytest.mark.parametrize('dtype,length_delta', [(np.float64, 0), (np.float32, -1)])
def test_group_sampler_rejects_misaligned_label_file(labelled, dtype, length_delta):
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler
    dataset, values = labelled
    dataset.close()
    np.save(dataset.label_dir / 'value.npy', values[:len(values) + length_delta].astype(dtype))
    with pytest.raises(ValueError, match='teacher value shape/dtype'):
        GroupBatchSampler(dataset)


def test_group_batch_runs_three_head_model_and_losses(labelled):
    import torch
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler, collate_groups
    from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
    from go_aicup.tasks.strength.reproduce_paper.losses import grouped_losses
    dataset, _ = labelled
    tasks = next(iter(GroupBatchSampler(dataset, groups_per_update=1, updates=1)))
    batch = collate_groups(dataset.__getitems__(tasks))
    model = StrengthEstimator(channels=2, blocks=1)
    output = model(batch['planes'].float())
    losses = grouped_losses(output, batch['policy_target'], batch['value_target'], batch['value_weight'])
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses['total'].backward()
    assert model.stem[0].weight.grad is not None
