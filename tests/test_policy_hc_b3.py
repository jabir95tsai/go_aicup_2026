"""CPU checks for frozen feature composition, per-k fitting and exact epoch replay."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.policy_hc_b3 import features as f, train as t
from go_aicup.tasks.strength.policy_residual.data import Cache


def test_composition_uses_original_q_and_only_32_latents_per_side():
    left = {'q': np.full((2, 18), 1.), 'r': np.full((2, 18), 2.), 'z': np.full((2, 64), 3.)}
    right = {'q': np.full((2, 20), 4.), 'r': np.full((2, 20), 5.), 'z': np.full((2, 64), 6.)}
    result = f.assemble(left, right)
    for arm, width in zip(f.ARMS, (38, 76, 166)):
        assert result[arm].shape == (2, width)
        np.testing.assert_array_equal(result[arm][:, t.raw_columns(arm)], result['A0'])
    combined = np.concatenate((left['q'].reshape(2, 2, 9), left['z'].reshape(2, 2, 32)), 2).reshape(2, 82)
    split = f.split_features(left['q'], left['r'], combined, 9)
    np.testing.assert_array_equal(split['z'], left['z'])
    combined[:, 0] += 1
    with pytest.raises(AssertionError):
        f.split_features(left['q'], left['r'], combined, 9)


def test_alignment_reorders_and_rejects_bad_inputs():
    np.testing.assert_array_equal(f.align([9, 2], [[90], [20]], [2, 9], 1), [[20], [90]])
    for ids, values, wanted in (([2, 2], [[1], [2]], [2, 9]), ([2], [[np.nan]], [2]),
                                ([2], [[1]], [3])):
        with pytest.raises(ValueError):
            f.align(ids, values, wanted, 1)


@pytest.mark.parametrize('profiles', [9, 10])
def test_frozen_chunk_means_and_models_get_no_gradients(profiles):
    class Encoder(torch.nn.Module):
        def __init__(self, latent):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(2.))
            self.latent = latent

        def forward(self, lp, target, q):
            if self.latent:
                return torch.cat((q, target.float()[:, None].expand(-1, 32) * self.bias), 1)
            return q + self.bias

    models = {'CNN-R10': Encoder(False), 'CNN-Z32': Encoder(True)}
    q = torch.arange(7 * profiles).reshape(7, profiles).float()
    sides = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.uint8)
    arrays = {'q': q, 'side': sides, 'target': torch.arange(7),
              'logp': torch.zeros(7, profiles, 362), 'counts': torch.tensor([[2, 1], [1, 3]])}
    arrays['c'] = torch.stack([torch.cat((q[a:b][sides[a:b] == 0].mean(0),
                                         q[a:b][sides[a:b] == 1].mean(0))) for a, b in ((0, 3), (3, 7))])
    cache = Cache(pd.DataFrame({'game_index': [8, 2]}), np.array([0, 3, 7]), arrays, torch.device('cpu'))
    result = f.frozen_features(models, cache, 2, profiles)
    np.testing.assert_allclose(result['r'], arrays['c'].numpy() + 2)
    np.testing.assert_allclose(result['z'][:, [0, 32]], [[2, 2], [8, 28 / 3]])
    assert all(p.grad is None for m in models.values() for p in m.parameters())


def synthetic():
    rng = np.random.default_rng(44)
    n = 80
    raw = rng.normal(size=(n, 38)).astype(np.float32)
    left = {'q': raw[:, :18], 'r': rng.normal(size=(n, 18)).astype(np.float32),
            'z': rng.normal(size=(n, 64)).astype(np.float32)}
    right = {'q': raw[:, 18:], 'r': rng.normal(size=(n, 20)).astype(np.float32),
             'z': rng.normal(size=(n, 64)).astype(np.float32)}
    values = f.assemble(left, right)
    questions = pd.DataFrame({'k': np.repeat([5, 10, 15, 20], 20), 'rank_id': np.tile(np.arange(10), 8)})
    return values, {'train': questions, 'monitor': questions.copy()}, raw


def test_all_heads_share_step_zero_and_scalers_ignore_monitor():
    values, questions, raw = synthetic()
    y = questions['train'].rank_id.to_numpy()
    reference = None
    for arm in f.ARMS:
        head = t.initialize(values[arm], raw, y, arm, 'cpu')
        output = head(torch.from_numpy(values[arm])).detach().numpy()
        if reference is None:
            reference = output
        else:
            np.testing.assert_allclose(output, reference, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(head.mean.numpy(), values[arm].mean(0), atol=1e-6)


def test_per_k_training_and_interrupted_epoch_replay(tmp_path):
    torch.set_num_threads(1)
    values, questions, raw = synthetic()
    config = {'seed': 2027, 'lr': .0003, 'epochs': 2, 'questions_per_k': 5, 'evaluate_every': 2}
    inputs = {'train': values['R-HC'], 'monitor': values['R-HC'] + 50}
    args = ('R-HC', 5, inputs, questions, raw, config, 'test')
    expected, expected_report = t.fit_head(*args, tmp_path / 'reference', 'cpu')
    save = t.atomic_save
    writes = 0

    def interrupted(payload, path):
        nonlocal writes
        if path.name == 'last.pt':
            writes += 1
            if writes == 2:
                raise RuntimeError('interrupted before epoch commit')
        save(payload, path)

    with patch.object(t, 'atomic_save', side_effect=interrupted), pytest.raises(RuntimeError):
        t.fit_head(*args, tmp_path / 'resumed', 'cpu')
    got, report = t.fit_head(*args, tmp_path / 'resumed', 'cpu')
    assert report == expected_report
    assert report['completed_updates'] == 8 and report['train_questions'] == 20
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, got.state_dict()[name], atol=0, rtol=0)
    # Other k buckets cannot influence this k's fitting or scaler.
    changed = deepcopy(inputs)
    changed['train'][20:] += 1000
    independent, other_report = t.fit_head('R-HC', 5, changed, questions, raw, config, 'test', tmp_path / 'independent', 'cpu')
    assert other_report == expected_report
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(value, independent.state_dict()[name], atol=0, rtol=0)
    with pytest.raises(ValueError, match='identity'):
        t.fit_head('R-HC', 5, inputs, questions, raw, config, 'changed', tmp_path / 'resumed', 'cpu')


def test_equal_game_weight_and_complete_question_order():
    games = pd.DataFrame({'game_index': [6, 1, 9]})
    q = pd.DataFrame({'games': [[1, 6], [9, 1]]})
    np.testing.assert_array_equal(f.question_values(games, q, np.array([[10.], [20.], [40.]])), [[15], [30]])
    a = t.batches(2000, 8, 2027, 1)
    assert a.shape == (250, 8)
    np.testing.assert_array_equal(np.sort(a.flatten()), np.arange(2000))
    np.testing.assert_array_equal(a, t.batches(2000, 8, 2027, 1))
    assert not np.array_equal(a, t.batches(2000, 8, 2027, 2))


def test_compare_uses_all_four_k_buckets():
    frame = pd.DataFrame({'question_id': np.arange(40), 'player_id': np.tile(np.arange(10), 4),
                          'rank_id': np.tile(np.arange(10), 4), 'k': np.repeat([5, 10, 15, 20], 10), 'score': .5})
    frames = {arm: frame.copy() for arm in f.ARMS}
    frames['Z-HC']['score'] += .01
    result = t.comparisons(frames, 2027)
    assert result['Z-HC-A0']['gate_pass']
    assert not result['R-HC-A0']['gate_pass']
    assert result['Z-HC-A0']['bootstrap_draws'] == 2000


@pytest.mark.parametrize('smoke', [False, True])
def test_complete_launcher_on_cpu_with_mocked_gpu_workers(tmp_path, smoke):
    """Exercise all twelve fits and dev publication; only hardware/extraction are replaced."""
    from go_aicup.artifacts import file_hash, write_json
    torch.set_num_threads(1)
    rng = np.random.default_rng(9)
    games, questions = [], {}
    side_values = {s: {} for s in ('H', 'C')}
    for role_index, role in enumerate(('train', 'monitor', 'dev')):
        rows, qs = [], []
        for rank in range(10):
            ids = list(range(role_index * 200 + rank * 20, role_index * 200 + rank * 20 + 20))
            rows.extend({'game_index': g, 'player_id': role_index * 10 + rank,
                         'rank_id': rank, 'role': role, 'player_color': 0} for g in ids)
            qs.extend({'question_id': f'{role}_{rank}_{k}', 'player_id': role_index * 10 + rank,
                       'rank_id': rank, 'k': k, 'games': ids[:k]} for k in (5, 10, 15, 20))
        games.extend(rows)
        questions[role] = pd.DataFrame(qs)
        for side, p in (('H', 9), ('C', 10)):
            side_values[side][role] = {key: rng.normal(size=(200, width)).astype(np.float32)
                                       for key, width in (('q', 2*p), ('r', 2*p), ('z', 64))}
    games = pd.DataFrame(games)
    config = {'seed': 2027, 'lr': .0003, 'epochs': 1, 'questions_per_k': 5, 'evaluate_every': 2}
    if smoke:
        config['smoke_updates'] = 1
    original_device = torch.device

    def spawn(worker, args, nprocs, join):
        config, identity, out = args
        assert nprocs == 2 and join and worker is t.extract_worker
        for side in ('H', 'C'):
            path = out / 'features' / side / 'features.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, game_index=games.loc[~games.role.eq('dev'), 'game_index'].to_numpy(),
                     **{key: np.concatenate([side_values[side][role][key] for role in ('train', 'monitor')])
                        for key in ('q', 'r', 'z')})
            write_json(path.parent / 'complete.json', {'status': 'complete', 'identity': identity,
                                                       'sha256': file_hash(path)})

    def dev_input(cfg, side, ids):
        assert not smoke, 'smoke must never read dev feature values'
        return side_values[side]['dev']

    with patch.object(t, 'preflight', return_value=(games, questions, {})), \
         patch.object(t.torch.multiprocessing, 'spawn', side_effect=spawn), \
         patch.object(t.torch.cuda, 'device_count', return_value=2), \
         patch.object(t.torch.cuda, 'set_device'), \
         patch.object(t.torch.cuda, 'get_device_name', return_value='mock CPU'), \
         patch.object(t.torch, 'device', side_effect=lambda *a: original_device('cpu')), \
         patch.object(t, 'dev_side', side_effect=dev_input):
        t.run(config, tmp_path)
        summary = t.read_json(tmp_path / 'summary.json')
        assert t.read_json(tmp_path / 'complete.json')['status'] == 'complete'
        assert len(summary['heads']) == 3
        for arm in f.ARMS:
            assert set(summary['heads'][arm]) == {'5', '10', '15', '20'}
            assert summary['smoke'] is smoke
            if smoke:
                assert 'dev' not in summary['scores'][arm]
                assert not (tmp_path / arm / 'dev_predictions.parquet').exists()
                assert summary['differences'] == {}
            else:
                assert len(pd.read_parquet(tmp_path / arm / 'dev_predictions.parquet')) == 40
                assert set(summary['scores'][arm]['dev']['by_k']) == {'5', '10', '15', '20'}
        # Reopening a complete run must not spawn extraction or train again.
        with patch.object(t.torch.multiprocessing, 'spawn', side_effect=AssertionError('unexpected extraction')):
            t.run(config, tmp_path)


@pytest.mark.parametrize('profiles,module', [(9, f.h), (10, f.c)])
def test_real_cnn_extraction_matches_existing_game_features(profiles, module):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    lp = torch.randn(7, profiles, 362).log_softmax(-1)
    target = torch.arange(7)
    q = lp.gather(-1, target[:, None, None].expand(-1, profiles, 1)).squeeze(-1)
    side = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.uint8)
    raw = torch.stack([torch.cat([q[a:b][side[a:b] == s].mean(0) for s in (0, 1)])
                       for a, b in ((0, 3), (3, 7))])
    cache = Cache(pd.DataFrame({'game_index': [6, 3]}), np.array([0, 3, 7]),
                  {'logp': lp, 'target': target, 'q': q, 'side': side, 'c': raw,
                   'counts': torch.tensor([[2, 1], [1, 3]])}, torch.device('cpu'))
    models = {arm: module.build(arm, 2027).requires_grad_(False).eval() for arm in f.MODELS}
    with torch.no_grad():
        models['CNN-R10'].out.bias.fill_(.3)
    result = f.frozen_features(models, cache, 3, profiles)
    residual = module.game_features(models['CNN-R10'], cache, [0, 1], 3).numpy()
    combined = module.game_features(models['CNN-Z32'], cache, [0, 1], 3).numpy()
    expected = f.split_features(raw.numpy(), residual, combined, profiles)
    for key in ('q', 'r', 'z'):
        np.testing.assert_allclose(result[key], expected[key], atol=1e-6, rtol=1e-6)


def test_smoke_caps_updates_without_changing_training_pool(tmp_path):
    values, questions, raw = synthetic()
    config = {'seed': 2027, 'lr': .0003, 'epochs': 2, 'questions_per_k': 5,
              'evaluate_every': 2, 'smoke_updates': 3}
    inputs = {'train': values['A0'], 'monitor': values['A0']}
    _, result = t.fit_head('A0', 5, inputs, questions, raw, config, 'smoke', tmp_path, 'cpu')
    assert result['completed_updates'] == 3 and result['train_questions'] == 20
    history = t.read_json(tmp_path / 'A0/k5/learning_curve.json')
    assert [entry['step'] for entry in history] == [0, 2, 3]
    _, resumed = t.fit_head('A0', 5, inputs, questions, raw, config, 'smoke', tmp_path, 'cpu')
    assert resumed == result
