"""The new OOF/B2 seams; the neural network has separate focused tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.tasks.strength.SE_H_C_B2.data import sample_questions
from go_aicup.tasks.strength.SE_H_C_B2.predict import export
from go_aicup.tasks.strength.humansl import RankModel, question_mean
from go_aicup.tasks.strength.humansl_residual_b2 import b2, evaluate, features


@pytest.fixture
def example():
    records = []
    for rank in range(10):
        for color in (0, 1):
            for _ in range(21):
                records.append(dict(game_index=len(records), player_id=f'p{rank}', rank_id=rank,
                                    player_color=color, hd=color))
    games = pd.DataFrame(records)
    q = sample_questions(games, 1, 2027, balanced=True)
    values = np.full((len(games), 20), .01)
    for i, rank in enumerate(games.rank_id):
        values[i, rank] = .91
        values[i, 10 + rank] = .91
    frame = pd.DataFrame(values, columns=features.COLUMNS)
    frame.insert(0, 'game_index', games.game_index)
    frame['own_count'] = 17; frame['opp_count'] = 35
    return games, q, frame


def test_arithmetic_probability_means_and_empty_side():
    logits = torch.tensor([[8., 0.] + [0.] * 8, [0., 0.] + [0.] * 8, [0., 8.] + [0.] * 8])
    result = features.mean_probabilities(logits, [0, 0, 1])
    np.testing.assert_allclose(result[:10], logits[:2].softmax(-1).mean(0), atol=1e-7)
    assert not np.allclose(result[:10], logits[:2].mean(0).softmax(-1))
    np.testing.assert_allclose(result.reshape(2, 10).sum(1), [1, 1], atol=1e-7)
    with pytest.raises(ValueError, match='both'):
        features.mean_probabilities(logits, [0, 0, 0])


def test_b2_uses_fixed_same_color_questions_and_fit_only_scaler(example, monkeypatch):
    games, q, frame = example
    monkeypatch.setattr(RankModel, '_fit_synthetic', lambda *args: pytest.fail('old sampler called'))
    model = b2.fit_questions(games, q, frame)
    x = features.p2_values(games, frame)
    assert x.shape == (len(games), 22)
    for k in features.BUCKETS:
        expected = question_mean(x, frame.game_index, q[q.k.eq(k)]).mean(0)
        np.testing.assert_allclose(model.per_k[k][0].mean_, expected)
    p = b2.probabilities(model, games, q, frame)
    np.testing.assert_allclose(p.sum(1), 1)
    assert (p.argmax(1) == q.rank_id.to_numpy()).all()
    corrupt = q.copy(); corrupt.at[0, 'color'] = 1 - corrupt.color.iloc[0]
    with pytest.raises(ValueError, match='one player, color and rank'):
        b2.fit_questions(games, corrupt, frame)


def test_oof_rejects_seen_and_wrong_stage1_ancestry():
    rows = pd.DataFrame(dict(player_id=['held']))
    identity = dict(phase='dev', fold=2, stage=1, training_players=['train'])
    features.validate_oof(rows, identity, phase='dev', fold=2)
    with pytest.raises(ValueError, match='trained'):
        features.validate_oof(rows, dict(identity, training_players=['held']))
    with pytest.raises(ValueError, match='ancestry'):
        features.validate_oof(rows, dict(identity, stage=2))
    parent = dict(identity, fold=1)
    with pytest.raises(ValueError, match='ancestry'):
        features.validate_oof(rows, dict(identity, stage=2, stage1_identity=parent))
    features.validate_oof(rows, dict(identity, stage=2, stage1_identity=identity))


def test_bounded_position_batches_have_identical_game_mean():
    class Model(torch.nn.Module):
        def forward(self, spatial, global_inputs, metadata, actions, sides):
            return {'rank_logits': torch.nn.functional.one_hot(actions % 10, 10).float() * 5}
    class Dataset:
        def iter_positions(self, row, batch_size):
            for start in range(0, 31, batch_size):
                pos = np.arange(start, min(start + batch_size, 31))
                yield dict(spatial=np.zeros((len(pos), 1, 1, 1)), global_inputs=np.zeros((len(pos), 1)),
                           metadata=np.zeros((len(pos), 1)), actions=pos, sides=pos % 2)
    rows = pd.DataFrame(dict(game_index=[123]))
    left = features.infer_games(Model(), Dataset(), rows, batch_size=7)
    right = features.infer_games(Model(), Dataset(), rows, batch_size=300)
    np.testing.assert_allclose(left[features.COLUMNS], right[features.COLUMNS], atol=1e-7)
    assert left.own_count.iloc[0] == 16 and left.opp_count.iloc[0] == 15


def test_each_game_equal_weight_regardless_of_position_count(example):
    games, q, frame = example
    a = question_mean(frame[features.COLUMNS].to_numpy(), frame.game_index, q)
    frame['own_count'] = np.arange(len(frame)) + 1
    b = question_mean(frame[features.COLUMNS].to_numpy(), frame.game_index, q)
    np.testing.assert_array_equal(a, b)


def test_gate_and_player_bootstrap(example):
    _, q, _ = example
    p = np.eye(10)[q.rank_id]
    same = evaluate.compare(q, p, p, repeats=20)
    assert not same['passed'] and same['equal_k_gain'] == 0
    poor = np.eye(10)[(q.rank_id + 3) % 10]
    better = evaluate.compare(q, poor, p, repeats=20)
    assert better['passed'] and all(v > 0 for v in better['by_k_gain'].values())
    assert better['paired_bootstrap']['intervals']['equal_k'][0] > 0


def test_best_final_tie_keeps_best(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, 'evaluate', lambda *args: {'equal_k': .5})
    assert evaluate.select({'run_dir': str(tmp_path)}, 1)['selected'] == 'best'


def test_b2_atomic_publication_checks_hash(example, tmp_path):
    games, q, frame = example
    model = b2.fit_questions(games, q, frame)
    path = tmp_path / 'model.pkl'
    b2.save_model(path, model, {'fit_players': ['p0']})
    restored, identity = b2.load_model(path)
    assert identity == {'fit_players': ['p0']}
    np.testing.assert_allclose(b2.probabilities(restored, games, q, frame), b2.probabilities(model, games, q, frame))
    with path.open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        b2.load_model(path)


def test_existing_export_accepts_new_filename_without_fake_fallback(tmp_path):
    ids = []
    for part in ('public', 'private'):
        qid = f'{part}-1'; ids.append(qid)
        p = tmp_path / f'rank_{part}' / 'new.parquet'; p.parent.mkdir()
        frame = pd.DataFrame(dict(question_id=[qid], rank_id=[0], rank=[RANKS[0]]))
        for i, rank in enumerate(RANKS):
            frame[f'p_{rank}'] = float(i == 0)
        frame.to_parquet(p, index=False)
        write_json(p.with_suffix('.json'), dict(model='same-release-model', sha256=file_hash(p)))
    sample = tmp_path / 'sample.csv'
    pd.DataFrame(dict(question_id=ids[::-1], pred_rank=['', ''])).to_csv(sample, index=False)
    output = tmp_path / 'submission.csv'
    export(tmp_path, sample, output, prediction_name='new.parquet')
    assert pd.read_csv(output).question_id.tolist() == ids[::-1]


def test_complete_oof_fit_rejects_stale_checkpoint(example, tmp_path, monkeypatch):
    from go_aicup.artifacts import object_hash
    from go_aicup.tasks.strength.SE_H_C_B2.common import publish_table
    from go_aicup.tasks.strength.humansl_residual_b2 import data
    games, q, frame = example
    games['role'] = 'D'
    games['oof_fold'] = games.rank_id % 4 + 1
    manifest = {'dataset_id': 'synthetic'}
    config = {'run_dir': str(tmp_path)}
    monkeypatch.setenv('SLURM_GPUS_ON_NODE', '1')
    monkeypatch.setattr(data, 'games', lambda *a, **kw: games)
    monkeypatch.setattr(data, 'questions', lambda *a, **kw: q)
    monkeypatch.setattr(data, 'load_prepared', lambda *a, **kw: manifest)
    ids = set(np.concatenate(q.games))
    for fold in range(1, 5):
        checkpoint = tmp_path / 'dev' / f'fold_{fold}' / 'stage1' / 'best.pt'
        checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(f'fold{fold}'.encode())
        rows = games[games.oof_fold.eq(fold) & games.game_index.isin(ids)]
        upstream = dict(stage=1, phase='dev', fold=str(fold),
                        training_players=games.loc[games.oof_fold.ne(fold), 'player_id'].unique().tolist())
        identity = dict(prepared=object_hash(manifest), checkpoint=file_hash(checkpoint),
                        upstream=upstream, candidate='best', pool='fit')
        publish_table(features.feature_path(config, 'dev', 1, 'best', 'fit', fold),
                      frame[frame.game_index.isin(rows.game_index)], identity)
    path = b2.fit(config, 1)
    _, identity = b2.load_model(path)
    assert identity['dimensions'] == 22 and len(identity['sources']) == 4
    metrics_path = path.with_suffix('.metrics.json')
    metrics = json.loads(metrics_path.read_text())
    original_metrics = metrics_path.read_bytes()
    assert not metrics['reused'] and metrics['dimensions'] == 22 and metrics['questions'] == len(q)
    assert metrics['allocated_gpus'] == 1 and metrics['compute_gpu_hours'] == 0
    assert metrics['gpu_hours'] == pytest.approx(metrics['elapsed_seconds'] / 3600)
    assert b2.fit(config, 1) == path
    assert metrics_path.read_bytes() == original_metrics
    reuse = json.loads(path.with_suffix('.reuse.metrics.json').read_text())
    assert reuse['reused'] and reuse['questions_per_second'] is None
    checkpoint.write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='checkpoint'):
        b2.fit(config, 1)


@pytest.mark.parametrize('stage', [1, 2])
def test_release_prepare_predict_real_move_store_and_cached_reuse(example, tmp_path, monkeypatch, stage):
    from types import SimpleNamespace
    from go_aicup.artifacts import object_hash
    from go_aicup.tasks.strength.humansl_residual_b2 import data, dataset, predict, train
    from go_aicup.tasks.strength.v2.predict import prepare
    games, q, frame = example
    store = tmp_path / 'store'; store.mkdir()
    lengths = np.arange(65, 70)
    np.save(store / 'rank_public_moves.npy', np.concatenate([np.arange(n, dtype=np.int16) for n in lengths]))
    np.save(store / 'rank_public_colors.npy', np.concatenate([(np.arange(n) % 2).astype(np.int8) for n in lengths]))
    metadata = pd.DataFrame(dict(offset=np.r_[0, np.cumsum(lengths)[:-1]], length=lengths,
                                 color=['B'] * 5, hd=[0, 1, 0, 1, 0], question_id=['q-release'] * 5, num_games=5))
    metadata.to_parquet(store / 'rank_public_meta.parquet')
    work = tmp_path / 'rank_public'
    prepare(store, 'rank_public', work)
    config = dict(run_dir=str(tmp_path / 'run'), seed=2026, device='cpu', precision='fp32',
                  training={'positions_per_side': 15}, data={'store_dir': str(store)})
    manifest = {'dataset_id': 'release-test'}
    monkeypatch.setattr(data, 'load_prepared', lambda *a, **kw: manifest)
    monkeypatch.setattr(predict, 'load_lock', lambda *a: {'fixed_steps': {'s1': 7, 's2': 9}})
    checkpoint = tmp_path / 'final.pt'; checkpoint.write_bytes(b'mocked-human-and-classifier')
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.humansl = SimpleNamespace(config={})
        def forward(self, spatial, global_inputs, metadata, actions, sides):
            return {'rank_logits': torch.nn.functional.one_hot(actions % 10, 10).float()}
    payload = dict(identity=dict(phase='release', fold='all', stage=stage), state={'optimizer_steps': 7 if stage == 1 else 9},
                   dataset_id=manifest['dataset_id'], config_sha256=object_hash(config))
    if stage == 2:
        parent = Path(config['run_dir']) / 'release/fold_all/stage1/final.pt'
        parent.parent.mkdir(parents=True)
        torch.save({'state': {'optimizer_steps': 7}}, parent)
        payload['identity']['stage1_source_sha256'] = file_hash(parent)
    monkeypatch.setattr(train, 'load_candidate', lambda *a, **kw: (Model(), payload))
    encoded = []
    def encode(moves, colors, plies, model_config, hd, **kwargs):
        encoded.append((len(moves), hd, tuple(plies)))
        assert np.array_equal(moves, np.arange(len(moves)))
        assert np.array_equal(colors, np.arange(len(colors)) % 2)
        return dict(spatial=np.zeros((len(plies), 22, 19, 19), np.uint8),
                    global_inputs=np.zeros((len(plies), 19), np.float32),
                    metadata=np.zeros((len(plies), 192), np.float32), actions=moves[plies])
    monkeypatch.setattr(dataset, 'encode_selected', encode)
    bundle = b2.model_path(config, stage, 'final', phase='release')
    b2.save_model(bundle, b2.fit_questions(games, q, frame),
                  dict(phase='release', stage=stage, config=object_hash(config)))
    result = predict.predict(config, work, checkpoint, stage=stage)
    assert result['questions'] == 1
    output = pd.read_parquet(work / predict.FILENAME)
    assert output.question_id.tolist() == ['q-release']
    assert not any('fallback' in c or 'humansl_games' in c for c in output)
    saved = pd.read_parquet(work / 'humansl_residual_b2_games.parquet')
    assert (saved.own_count + saved.opp_count).tolist() == (lengths - 30).tolist()
    assert [hd for _, hd, _ in encoded] == metadata.hd.tolist()
    monkeypatch.setattr(dataset, 'encode_selected', lambda *a, **kw: pytest.fail('completed features recomputed'))
    assert predict.predict(config, work, checkpoint, stage=stage) == result
    if stage == 2:
        torch.save({'state': {'optimizer_steps': 8}}, parent)
        with pytest.raises(ValueError, match='locked s1'):
            predict.predict(config, work, checkpoint, stage=stage)


def test_u_lock_binds_selection_models_dataset_and_stage1_ancestry(tmp_path, monkeypatch):
    from go_aicup.artifacts import object_hash
    from go_aicup.tasks.strength.humansl_residual_b2 import data
    config = {'run_dir': str(tmp_path)}
    monkeypatch.setattr(data, 'load_prepared', lambda *a, **kw: {'dataset_id': 'fixed-data'})
    previous = None
    for stage in (1, 2):
        selection = tmp_path / 'selection' / f'stage{stage}.json'
        write_json(selection, {'selected': 'best'})
        checkpoint = tmp_path / 'dev' / 'fold_all' / f'stage{stage}' / 'best.pt'
        checkpoint.parent.mkdir(parents=True)
        identity = dict(phase='dev', fold='all', stage=stage, stage1_source_sha256=previous)
        torch.save(dict(identity=identity, state={'optimizer_steps': stage * 10}), checkpoint)
        previous = file_hash(checkpoint)
        b2.save_model(b2.model_path(config, stage, 'best'), RankModel('B2'), {'config': object_hash(config)})
    with pytest.raises(ValueError, match='selected'):
        evaluate.lock(config, {'s1': {'stage': 1, 'candidate': 'final'}})
    arms = {'s1': {'stage': 1, 'candidate': 'best'}, 's2': {'stage': 2, 'candidate': 'best'}}
    result = evaluate.lock(config, arms)
    assert result['fixed_steps'] == {'s1': 10, 's2': 20}
    assert evaluate.load_lock(config) == result
    write_json(tmp_path / 'selection/stage1.json', {'selected': 'final'})
    with pytest.raises(ValueError, match='selection changed'):
        evaluate.load_lock(config)


def test_extraction_cost_and_reuse_keep_success_metrics(example, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from go_aicup.artifacts import object_hash
    from go_aicup.tasks.strength.humansl_residual_b2 import data, train
    games, _, frame = example
    games = games.iloc[:2].copy(); frame = frame.iloc[:2].copy()
    config = dict(run_dir=str(tmp_path), device='cpu', seed=2026, data={}, training={'positions_per_side': 15})
    manifest = {'dataset_id': 'measurement-test'}
    checkpoint = tmp_path / 'best.pt'; checkpoint.write_bytes(b'weights')
    payload = dict(identity=dict(stage=1, phase='dev', fold='1', training_players=['different-player']),
                   dataset_id=manifest['dataset_id'], config_sha256=object_hash(config))
    monkeypatch.setattr(data, 'load_prepared', lambda *a, **kw: manifest)
    monkeypatch.setattr(features, 'extraction_rows', lambda *a, **kw: games)
    monkeypatch.setattr(train, 'load_candidate', lambda *a, **kw: (SimpleNamespace(humansl=SimpleNamespace(config={})), payload))
    calls = []
    def infer(*a, **kw):
        calls.append(True)
        return frame
    monkeypatch.setattr(features, 'infer_games', infer)
    features.extract(config, checkpoint, fold=1)
    path = features.feature_path(config, 'dev', 1, 'best', 'fit', 1)
    measured = path.with_suffix('.metrics.json')
    original = measured.read_bytes()
    metrics = json.loads(original)
    assert metrics['positions'] == 104 and metrics['positions_per_second'] > 0
    assert metrics['gpu_hours'] == 0 and not metrics['reused']
    features.extract(config, checkpoint, fold=1)
    assert measured.read_bytes() == original and len(calls) == 1
    reuse = json.loads(path.with_suffix('.reuse.metrics.json').read_text())
    assert reuse['reused'] and reuse['computed_positions'] == 0 and reuse['positions_per_second'] is None


def test_smoke_cannot_lock_u_or_release(tmp_path):
    with pytest.raises(ValueError, match='smoke'):
        evaluate.lock({'run_dir': str(tmp_path), 'smoke': True}, {})
