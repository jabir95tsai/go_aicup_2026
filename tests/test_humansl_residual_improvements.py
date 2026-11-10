"""2026-11-10 revision: log aggregation, fold-average upstream, capped fit_all pools, LR decay, probe."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.artifacts import file_hash, object_hash
from go_aicup.tasks.strength.SE_H_C_B2.common import publish_table
from go_aicup.tasks.strength.SE_H_C_B2.data import sample_questions
from go_aicup.tasks.strength.humansl_residual_b2 import b2, data, features, probe
from go_aicup.tasks.strength.humansl_residual_b2 import train as training
from go_aicup.tasks.strength.humansl_residual_b2.config import read_config, validate_config


def test_infer_games_stores_mean_log_softmax_next_to_mean_probability():
    class Model(torch.nn.Module):
        def forward(self, spatial, global_inputs, metadata, actions, sides):
            return {'rank_logits': torch.nn.functional.one_hot(actions % 10, 10).float() * 4}
    class Dataset:
        def iter_positions(self, row, batch_size):
            for start in range(0, 40, batch_size):
                pos = np.arange(start, min(start + batch_size, 40))
                yield dict(spatial=np.zeros((len(pos), 1, 1, 1)), global_inputs=np.zeros((len(pos), 1)),
                           metadata=np.zeros((len(pos), 1)), actions=pos, sides=pos % 2)
    frame = features.infer_games(Model(), Dataset(), pd.DataFrame(dict(game_index=[7])), batch_size=9)
    logits = torch.nn.functional.one_hot(torch.arange(40) % 10, 10).float() * 4
    expected = features.mean_log_probabilities(logits, torch.arange(40) % 2)
    np.testing.assert_allclose(frame[features.LOGP_COLUMNS].to_numpy()[0], expected, atol=1e-6)
    np.testing.assert_allclose(frame[features.COLUMNS].to_numpy()[0],
                               features.mean_probabilities(logits, torch.arange(40) % 2), atol=1e-6)
    # Log aggregation is a different statistic from the log of the probability mean.
    assert not np.allclose(expected, np.log(frame[features.COLUMNS].to_numpy()[0]))


def test_p2_values_selects_aggregation_and_rejects_old_frames():
    games = pd.DataFrame(dict(game_index=[0], hd=[1], player_color=[0]))
    frame = pd.DataFrame([dict(game_index=0, own_count=3, opp_count=3,
                               **{c: .1 for c in features.COLUMNS}, **{c: -2.3 for c in features.LOGP_COLUMNS})])
    assert features.p2_values(games, frame, 'logp')[0, :20].tolist() == [-2.3] * 20
    assert features.p2_values(games, frame, 'logp')[0, 20:].tolist() == [1, 0]
    with pytest.raises(ValueError, match='re-extract'):
        features.p2_values(games, frame.drop(columns=features.LOGP_COLUMNS), 'logp')
    bad = frame.copy(); bad[features.LOGP_COLUMNS[0]] = .5
    with pytest.raises(ValueError, match='nonpositive'):
        features.validate_features(bad)


def test_cosine_learning_rate_and_step_schedule_compose():
    options = dict(lr=.1, lr_schedule=[], lr_decay='cosine', min_lr_ratio=.05, updates=100)
    assert training.learning_rate(options, 0) == pytest.approx(.1)
    assert training.learning_rate(options, 50) == pytest.approx(.1 * (.05 + .95 * .5))
    assert training.learning_rate(options, 100) == pytest.approx(.005)
    stepped = dict(options, lr_schedule=[[50, .01]])
    assert training.learning_rate(stepped, 50) == pytest.approx(.01 * (.05 + .95 * .5))
    assert training.learning_rate(dict(lr=.1, lr_schedule=[[10, .01]]), 10) == .01


def test_production_config_budgets_and_validation():
    config = read_config(Path(__file__).parents[1] / 'configs/strength/humansl_residual_train.json')
    assert config['stage1']['updates'] == 3000 and config['stage1']['monitor_every'] == 300
    assert config['stage2']['updates'] == 6000 and config['stage2']['monitor_every'] == 600
    assert config['b2']['aggregation'] == 'logp' and config['b2']['fit_pool'] == 'fit_all'
    bad = json.loads(json.dumps(config)); bad['b2']['fit_all_cap_per_color'] = 10
    with pytest.raises(ValueError, match='k=20'):
        validate_config(bad)
    bad = json.loads(json.dumps(config)); bad['stage1']['lr_decay'] = 'linear'
    with pytest.raises(ValueError, match='lr_decay'):
        validate_config(bad)
    with pytest.raises(ValueError, match='release retraining'):
        training.train(config, 1, phase='release')


def _games(roles=('D',), players=3, per_color=25):
    rows = []
    for role in roles:
        for rank in range(10):
            for p in range(players):
                for color in (0, 1):
                    for _ in range(per_color):
                        rows.append(dict(game_index=len(rows), player_id=f'{role}{rank}-{p}', rank_id=rank,
                                         player_color=color, hd=0, role=role, oof_fold=p % 4 + 1 if role == 'D' else 0))
    return pd.DataFrame(rows)


def test_cap_per_color_is_seeded_and_bounded():
    games = _games()
    capped = data.cap_per_color(games, 20, 2027, 91)
    assert capped.groupby(['player_id', 'player_color']).size().max() == 20
    assert capped.game_index.tolist() == data.cap_per_color(games.sample(frac=1, random_state=1), 20, 2027, 91).game_index.tolist()
    assert capped.game_index.tolist() != data.cap_per_color(games, 20, 2027, 92).game_index.tolist()


@pytest.fixture
def extension(tmp_path, monkeypatch):
    games = _games(('D', 'M', 'U'), players=4)
    config = dict(run_dir=str(tmp_path / 'run'), split_seed=2027, data={'prepared_dir': str(tmp_path / 'prepared')},
                  b2=dict(questions_per_k=300, fit_all_cap_per_color=20, fit_pool='fit_all', aggregation='logp'))
    (tmp_path / 'prepared').mkdir()
    fit = sample_questions(games[games.role.eq('D')], 300, 2027, prefix='fit')
    fit.to_parquet(tmp_path / 'prepared/questions_fit.parquet', index=False)
    monitor = sample_questions(games[games.role.eq('M')], 2, 2027, balanced=True, prefix='monitor')
    monitor.to_parquet(tmp_path / 'prepared/questions_monitor.parquet', index=False)
    monkeypatch.setattr(data, 'games', lambda *a, **kw: games)
    monkeypatch.setattr(data, 'load_prepared', lambda *a, **kw: {'dataset_id': 'synthetic-base'})
    return config, games


def test_extension_pools_are_additive_capped_and_release_needs_lock(extension, monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2 import evaluate
    config, games = extension
    record = data.prepare_extension(config)
    assert data.prepare_extension(config) == record
    ids = data.pool_game_ids(config, 'fit_all')
    pool = games[games.game_index.isin(ids)]
    assert set(pool.role) == {'D'} and pool.groupby(['player_id', 'player_color']).size().max() == 20
    q = data.questions(config, 'fit_all')
    assert set(np.concatenate(q.games.to_numpy())) <= ids and set(q.k) == {5, 10, 15, 20}
    with pytest.raises(FileNotFoundError):
        data.prepare_extension(config, release=True)  # no U lock yet
    monkeypatch.setattr(evaluate, 'load_lock', lambda *a: {})
    data.prepare_extension(config, release=True)
    mu = games[games.game_index.isin(data.pool_game_ids(config, 'mu_all'))]
    assert set(mu.role) == {'M', 'U'}
    release = data.questions(config, 'release_fit_all')
    used = games.set_index('game_index').loc[np.concatenate(release.games.to_numpy())]
    assert set(used.role) == {'D', 'M', 'U'}
    changed = json.loads(json.dumps(config)); changed['b2']['fit_all_cap_per_color'] = 25
    with pytest.raises(ValueError, match='identity changed'):
        data.questions(changed, 'fit_all')


def _publish(config, games, pool, fold, frame, stage=1, candidate='best'):
    checkpoint = Path(config['run_dir']) / 'dev' / f'fold_{fold}' / f'stage{stage}' / f'{candidate}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        checkpoint.write_bytes(f'fold{fold}'.encode())
    upstream = dict(stage=stage, phase='dev', fold=str(fold),
                    training_players=games.loc[games.role.eq('D') & games.oof_fold.ne(fold), 'player_id'].unique().tolist())
    identity = dict(prepared=object_hash({'dataset_id': 'synthetic-base'}), checkpoint=file_hash(checkpoint),
                    upstream=upstream, candidate=candidate, pool=pool)
    publish_table(features.feature_path(config, 'dev', stage, candidate, pool, fold), frame, identity)


def _frame(games, ids, shift=0.):
    rows = games[games.game_index.isin(ids)].sort_values('game_index')
    logits = np.full((len(rows), 2, 10), -3.) + shift
    logits[np.arange(len(rows)), :, rows.rank_id.to_numpy()] = 2.
    logp = logits - np.log(np.exp(logits).sum(2, keepdims=True))
    frame = pd.DataFrame(np.exp(logp).reshape(len(rows), 20), columns=features.COLUMNS)
    frame[features.LOGP_COLUMNS] = logp.reshape(len(rows), 20)
    frame.insert(0, 'game_index', rows.game_index.to_numpy())
    frame['own_count'] = 9; frame['opp_count'] = 8
    return frame


def test_fit_all_oof_b2_and_fold_average_monitor_evaluation(extension):
    from go_aicup.tasks.strength.humansl_residual_b2 import evaluate
    config, games = extension
    data.prepare_extension(config)
    fit_ids = data.pool_game_ids(config, 'fit_all')
    monitor_ids = data.pool_game_ids(config, 'monitor')
    for fold in range(1, 5):
        held = games[games.oof_fold.eq(fold) & games.game_index.isin(fit_ids)].game_index
        _publish(config, games, 'fit_all', fold, _frame(games, held))
        _publish(config, games, 'monitor', fold, _frame(games, monitor_ids, shift=fold * .1))
    path = b2.fit(config, 1, arm='new', aggregation='logp')
    assert path.parent.name == 'new_logp_fit_all'
    _, identity = b2.load_model(path)
    assert identity['dimensions'] == 22 and identity['upstream'] == 'four-fold-oof/fold-average'
    _, _, averaged, sources = b2.read_pool(config, 1, 'best', 'monitor')
    assert len(sources) == 4
    singles = [b2.read_pool(config, 1, 'best', 'monitor', inference_fold=f)[2] for f in range(1, 5)]
    np.testing.assert_allclose(averaged[features.LOGP_COLUMNS].to_numpy(),
                               np.mean([s[features.LOGP_COLUMNS].to_numpy() for s in singles], axis=0))
    result = evaluate.evaluate(config, 1, 'best', 'monitor', 'new', 'logp')
    assert result['upstream'] == 'fold-average' and result['equal_k'] > .9
    with pytest.raises(ValueError, match='held-out fold'):
        features.extraction_rows(config, 'fit_all', fold='all')


def test_probe_head_learns_separable_readout_and_game_features_are_branch_means():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 10, 4000)
    x = rng.normal(size=(4000, 768)).astype(np.float16)
    x[np.arange(4000), labels] += 6
    side = rng.integers(0, 2, 4000)
    options = dict(probe.DEFAULTS, epochs=25, batch_positions=256, lr=3e-3)
    for kind in ('linear', 'mlp'):
        predict = probe.train_head(kind, x, side, labels, options, torch.device('cpu'))
        logp = predict(x, side)
        assert (logp.argmax(1) == labels).mean() > .95
        np.testing.assert_allclose(np.exp(logp).sum(1), 1, atol=1e-5)
    games = np.repeat(np.arange(4), 30)
    sides = np.tile(np.repeat([0, 1], 15), 4)
    logp = np.log(np.full((120, 10), .1))
    frame = probe.game_features(games, sides, logp)
    assert frame.shape == (4, 43) and np.allclose(frame[features.COLUMNS], .1)
    features.validate_features(frame)


def test_probe_readout_matches_stage_one_gather():
    torch.manual_seed(0)
    f = torch.randn(3, 384, 19, 19)
    actions = torch.tensor([0, 200, 360])
    out = probe.readout(f, actions)
    assert out.shape == (3, 768)
    torch.testing.assert_close(out[:, :384], f.flatten(2).mean(2))
    torch.testing.assert_close(out[1, 384:], f[1, :, 200 // 19, 200 % 19])
