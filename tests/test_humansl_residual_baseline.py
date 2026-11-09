"""Controlled baseline checks without loading external HumanSL or running a GPU."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from go_aicup.tasks.strength.humansl_residual_b2 import baseline as b
from go_aicup.tasks.strength.humansl_residual_b2.b2 import load_model


class FakeDataset:
    def __init__(self, config, rows, model_config, **kwargs):
        self.rows = rows.reset_index(drop=True)
        self.minima = []

    def __getitem__(self, task):
        index, _, plies = task
        row = self.rows.iloc[index]; plies = np.asarray(plies, np.int64)
        n = len(plies)
        return dict(spatial=np.zeros((n, 22, 19, 19), np.uint8),
                    global_inputs=np.full((n, 1), row.rank_id / 10, np.float32),
                    actions=plies, sides=(plies % 2 != row.player_color).astype(np.int64),
                    labels=np.full(n, row.rank_id, np.int64),
                    game_index=np.full(n, row.game_index), ply=plies)

    def iter_positions(self, row, batch_size, min_ply=30):
        self.minima.append(min_ply)
        for start in range(min_ply, self.rows.length.iloc[row], batch_size):
            yield self[(row, 0, np.arange(start, min(start + batch_size, self.rows.length.iloc[row])))]

    def close(self):
        pass


def fake_policy(model, spatial, global_inputs, meta, actions):
    return -(meta[:, 0] - global_inputs[:, 0]).square() - 1


def fake_game(model, vectors, bins, global_inputs, actions, own, batch):
    sides = torch.as_tensor(~np.asarray(own)).long()
    target = torch.as_tensor(global_inputs)[:, 0]
    return -(vectors[:, sides, 0].T - target[:, None]).square() - 1


@pytest.fixture
def controlled(tmp_path, monkeypatch):
    rows = []
    for rank in range(10):
        for fold in range(1, 5):
            rows.append(dict(game_index=len(rows), rank_id=rank, player_id=f'd{rank}_{fold}',
                             player_color=rank % 2, role='D', oof_fold=fold, length=6, hd=fold % 2))
        for role in ('M', 'U', 'R'):
            rows.append(dict(game_index=len(rows), rank_id=rank, player_id=f'{role}{rank}',
                             player_color=0, role=role, oof_fold=0, length=6, hd=0))
    rows = pd.DataFrame(rows)
    checkpoint = tmp_path / 'human.ckpt'; checkpoint.write_text('synthetic identity')
    config = dict(run_dir=str(tmp_path / 'run'), device='cpu', seed=2026, split_seed=2027,
                  model={'human_checkpoint': str(checkpoint)}, baseline=dict(updates=5, batch_positions=12, lr=.01),
                  training={'checkpoint_every': 2})
    monkeypatch.setattr(b, 'games', lambda config: rows)
    monkeypatch.setattr(b, 'load_prepared', lambda config: {'dataset_id': 'synthetic'})
    monkeypatch.setattr(b, 'SelectedDataset', FakeDataset)
    monkeypatch.setattr(b, 'load_runtime', lambda config: (SimpleNamespace(config={}), torch.zeros(384), torch.device('cpu')))
    monkeypatch.setattr(b, 'policy_logp', fake_policy)
    monkeypatch.setattr(b, 'game_logp', fake_game)
    return config, rows


def test_training_players_are_D_and_exclude_whole_held_out_fold(controlled):
    config, rows = controlled
    trained = b.training_rows(config, 2)
    assert set(trained.role) == {'D'}
    assert set(trained.rank_id) == set(range(10))
    assert not trained.oof_fold.eq(2).any()
    assert len(b.training_rows(config, 'all')) == 40
    with pytest.raises(ValueError, match='fold'):
        b.training_rows(config, 5)


def test_c_training_resume_is_exact_and_never_uses_M_U_R(controlled):
    config, rows = controlled
    partial = b.train_c(config, 2, stop_after=2)
    assert not partial['complete']
    final = b.train_c(config, 2)
    assert final['complete'] and final['updates'] == 5
    payload = torch.load(final['path'], weights_only=False)
    assert all(player.startswith('d') and not player.endswith('_2')
               for player in payload['identity']['training_players'])
    other = copy.deepcopy(config); other['run_dir'] += '_uninterrupted'
    reference = b.train_c(other, 2)
    expected = torch.load(reference['path'], weights_only=False)
    assert torch.equal(payload['model']['vectors'], expected['model']['vectors'])
    assert payload['state'] == expected['state']
    assert not torch.equal(payload['model']['vectors'], torch.zeros(10, 2, 384))
    config['baseline']['lr'] = .02
    with pytest.raises(ValueError, match='source changed'):
        b.train_c(config, 2)


def test_baseline_streams_all_plies_and_is_batch_invariant(controlled):
    config, rows = controlled
    rows = rows.head(2)
    dataset = FakeDataset(config, rows, {})
    profiles = torch.zeros(2, len(b.PROFILES), 384)
    profiles[:, :, 0] = torch.arange(len(b.PROFILES))[None] / 10
    vectors = torch.zeros(10, 2, 384)
    vectors[:, :, 0] = torch.arange(10)[:, None] / 10
    first = b.infer_likelihoods(None, vectors, profiles, dataset, rows, 2, 'cpu')
    second = b.infer_likelihoods(None, vectors, profiles, dataset, rows, 5, 'cpu')
    assert np.allclose(first[b.COLUMNS], second[b.COLUMNS])
    assert dataset.minima == [0, 0, 0, 0]
    assert (first[['own_count', 'opp_count']].to_numpy() == 3).all()
    assert np.allclose(first[b.H_COLUMNS].iloc[0, :15], -(np.arange(15) / 10) ** 2 - 1)
    assert np.allclose(first[b.C_COLUMNS].iloc[0, :10], -(np.arange(10) / 10) ** 2 - 1)


def test_uniform_position_sampling_has_no_duplicates_and_keeps_opening(controlled):
    config, rows = controlled
    rows = rows.head(2)
    sample = b.sample_positions(FakeDataset(config, rows, {}), rows, 12, np.random.default_rng(2), 0)
    assert len(set(zip(sample['game_index'], sample['ply']))) == 12
    assert sample['ply'].min() == 0 and sample['ply'].max() == 5
    with pytest.raises(ValueError, match='batch'):
        b.sample_positions(FakeDataset(config, rows, {}), rows, 13, np.random.default_rng(2), 0)


def test_old_or_in_sample_C_vectors_are_rejected(controlled):
    _, rows = controlled
    with pytest.raises(ValueError, match='identity'):
        b.validate_ancestry(rows.head(1), {'training_players': []}, 1)
    identity = dict(format='controlled_hc_vectors_v1', phase='dev', fold='1', training_players=[rows.player_id.iloc[0]])
    with pytest.raises(ValueError, match='seen'):
        b.validate_ancestry(rows.head(1), identity, 1)
    identity['training_players'] = []
    b.validate_ancestry(rows.head(1), identity, 1)
    with pytest.raises(ValueError, match='identity'):
        b.validate_ancestry(rows.head(1), identity, 2)


def test_H_uses_original_profile_encoder_despite_C_identity_override(monkeypatch):
    class Encoder(nn.Module):
        def forward(self, rows, extra_outputs=None):
            return rows[:, :1].repeat(1, 384)
    encoder = Encoder(); encoder.forward = lambda rows, extra_outputs=None: rows
    model = SimpleNamespace(metadata_encoder=encoder)
    monkeypatch.setattr(b, '_katago', lambda: (SimpleNamespace(Board=SimpleNamespace(BLACK=1, WHITE=2)), None, None, None))
    monkeypatch.setattr(b, 'profile_metadata', lambda profile: SimpleNamespace(
        get_metadata_row=lambda nextPlayer, boardArea: np.full(192, b.PROFILES.index(profile) + nextPlayer)))
    values = b.human_vectors(model, 'cpu')
    assert values.shape == (2, 15, 384)
    assert values[0, 0, 0] == 1 and values[1, 14, 0] == 16


def test_fixed_question_B2_is_52D_and_monitor_does_not_refit_scaler(controlled, monkeypatch):
    config, _ = controlled
    all_rows, question_rows, feature_rows = [], {'fit': [], 'monitor': []}, {'fit': [], 'monitor': []}
    for pool in ('fit', 'monitor'):
        for rank in range(10):
            ids = []
            for game in range(20):
                index = len(all_rows); ids.append(index)
                all_rows.append(dict(game_index=index, player_id=f'{pool}{rank}', rank_id=rank,
                                     player_color=rank % 2, role='D' if pool == 'fit' else 'M', hd=rank % 2))
                values = -np.abs(np.arange(50) % 10 - rank).astype(float)
                if pool == 'monitor':
                    values += .5
                feature_rows[pool].append(dict(game_index=index, own_count=10, opp_count=10,
                                              **dict(zip(b.COLUMNS, values))))
            for k in b.BUCKETS:
                question_rows[pool].append(dict(question_id=f'{pool}_{rank}_{k}', player_id=f'{pool}{rank}',
                                               rank_id=rank, color=rank % 2, k=k, games=np.array(ids[:k])))
    canonical = pd.DataFrame(all_rows)
    frames = {key: pd.DataFrame(value) for key, value in feature_rows.items()}
    qs = {key: pd.DataFrame(value) for key, value in question_rows.items()}
    monkeypatch.setattr(b, 'read_pool', lambda config, pool: (canonical, qs[pool], frames[pool], []))
    path = b.fit(config)
    fitted, identity = load_model(path)
    assert identity['dimensions'] == 52
    means = {k: fitted.per_k[k][0].mean_.copy() for k in b.BUCKETS}
    result = b.evaluate(config)
    assert set(result['by_k']) == {'5', '10', '15', '20'}
    assert b.fit(config) == path
    again, _ = load_model(path)
    assert all(np.array_equal(means[k], again.per_k[k][0].mean_) for k in b.BUCKETS)
    output = b.root(config) / 'evaluation' / 'monitor.parquet'
    assert output.exists()
    assert json.loads(output.with_suffix('.json').read_text())['report']['pool'] == 'monitor'
    assert len(pd.read_parquet(output)) == 40
