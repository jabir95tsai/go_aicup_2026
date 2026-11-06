"""Synthetic CPU checks for frozen-feature B2; never loads official weights."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import RANKS, write_store_cache
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.SE_H_C_B2 import common, data, features, model, train, evaluate, predict
from go_aicup.tasks.strength.reproduce_paper_3.evaluate import infer_games


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def metadata():
    rows = []
    for rank in range(10):
        for role in ('dev', 'select', 'u', 'c'):
            for n in range(50):
                rows.append(dict(game_index=len(rows), rank_id=rank, player_id=f'{rank}_{role}',
                                 player_color=n % 2, length=100, game_id=str(len(rows)),
                                 effective_position_count=35, split='valid'))
    return pd.DataFrame(rows)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = common.read_config()
    paths = config['paths']
    for key in ('output', 'dataset', 'cache', 'human_root', 'a_run', 'se3_evaluation'):
        paths[key] = str(tmp_path / key); Path(paths[key]).mkdir()
    canonical = metadata()
    canonical.to_parquet(Path(paths['dataset']) / 'games.parquet', index=False)
    canonical.to_parquet(Path(paths['cache']) / 'games.parquet', index=False)
    for key in ('cache', 'dataset'):
        write_json(Path(paths[key]) / 'manifest.json', {'synthetic': True})
    roles = canonical.copy()
    roles['role'] = roles.player_id.str.split('_').str[-1].map({'dev': 'S-fit', 'select': 'S-select', 'u': 'U'})
    c_games = canonical[canonical.player_id.str.endswith('_c')]
    root = Path(paths['human_root'])
    for name, selected in [('humansl_q1000', roles[roles.role.isin(['S-fit', 'U'])]),
                           ('humansl_qsel', roles[roles.role.eq('S-select')]),
                           ('humansl_vs_rest', roles.iloc[:0]), ('humansl_rankvec', c_games)]:
        (root / name).mkdir(); selected.to_parquet(root / name / 'games.parquet', index=False)
    qs = data.sample_questions(roles[roles.role.eq('S-select')], 200, 12, prefix='sel').assign(role='S-select')
    qu = data.sample_questions(roles[roles.role.eq('U')], 200, 13, prefix='u').assign(role='U')
    qs.to_parquet(root / 'humansl_qsel/questions.parquet', index=False)
    qu.to_parquet(root / 'humansl_q1000/questions.parquet', index=False)
    paths['se3_monitor'] = str(tmp_path / 'monitor.npz'); np.savez(paths['se3_monitor'], games=np.array([100]))
    config.update(synthetic_per_k=200, evaluation_per_cell=2, bootstrap=30)
    config['training'].update(epochs=3, batch_size=64, patience=2)
    source = dict(models={}, dataset=file_hash(Path(paths['dataset']) / 'games.parquet'), prepared='synthetic', cache='synthetic', code='synthetic')
    monkeypatch.setattr(data, 'source_identity', lambda _: source)
    return config, canonical


def test_architecture_and_ce():
    head = model.Head()
    assert sum(p.numel() for p in head.parameters()) == 2282
    assert [type(v) for v in head] == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear]
    x = torch.randn(13, 60)
    logits = head(x)
    assert logits.shape == (13, 10)
    truth = torch.arange(13) % 10
    loss = torch.nn.functional.cross_entropy(logits, truth)
    torch.testing.assert_close(loss, -logits.log_softmax(-1)[torch.arange(13), truth].mean())
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_questions_and_pool_leakage():
    games = metadata()
    q = data.sample_questions(games[games.player_id.str.endswith('_u')], 2, 3, balanced=True)
    assert len(q) == 80 and q.groupby(['k', 'rank_id']).size().eq(2).all()
    data.check_questions(q, games)
    again = data.sample_questions(games[games.player_id.str.endswith('_u')], 2, 3, balanced=True)
    for left, right in zip(q.games, again.games):
        np.testing.assert_array_equal(left, right)
    bad = q.copy(); bad.at[0, 'games'] = np.repeat(q.games.iloc[0][0], 5)
    with pytest.raises(ValueError, match='duplicate'):
        data.check_questions(bad, games)
    bad = q.copy(); bad.at[0, 'color'] = 1 - bad.color.iloc[0]
    with pytest.raises(ValueError, match='one player'):
        data.check_questions(bad, games)
    roles = games[~games.player_id.str.endswith('_c')].copy()
    roles['role'] = roles.player_id.str.split('_').str[-1].map({'dev': 'S-fit', 'select': 'S-select', 'u': 'U'})
    c = games[games.player_id.str.endswith('_c')]
    marked = games.copy(); marked.loc[marked.game_index.eq(101), 'split'] = 'train'
    dev, final, pool = data.make_pools(marked, roles, c, np.array([100]))
    assert not {100, 101} & set(pool.game_index)
    assert not set(pool.player_id) & set(final.player_id)
    with pytest.raises(ValueError, match='C learning'):
        data.make_pools(games, roles, games[games.player_id.str.endswith('_u')], [])
    corrupted = roles.copy(); corrupted.loc[corrupted.index[0], 'length'] += 1
    with pytest.raises(ValueError, match='metadata mismatch'):
        data.make_pools(games, corrupted, c, [])


def test_fixed_prepare_reuse_and_corruption(experiment):
    config, _ = experiment
    result = data.prepare(config)
    assert result['pools']['dev']['players'] == 10
    root = Path(config['paths']['output'])
    digest = file_hash(root / 'questions/newgames.parquet')
    before = (root / 'questions/newgames.parquet').stat().st_mtime_ns
    assert data.prepare(config) == result
    assert digest == file_hash(root / 'questions/newgames.parquet')
    assert before == (root / 'questions/newgames.parquet').stat().st_mtime_ns
    changed = deepcopy(config); changed['seed'] += 1
    with pytest.raises(ValueError, match='identity changed'):
        data.prepare(changed)
    (root / 'questions/newgames.parquet').write_bytes(b'broken')
    with pytest.raises(ValueError, match='changed'):
        data.load_prepared(config)


def fitting_arrays():
    rng = np.random.default_rng(19)
    y = np.tile(np.arange(10), 8)
    x = rng.normal(0, .05, (len(y), 60)).astype(np.float32)
    x[np.arange(len(y)), y] += 4
    return x, y


def settings():
    return dict(epochs=5, batch_size=23, lr=.01, min_lr=.00001, weight_decay=.0001, patience=10, threads=1)


def test_training_resume_scaler_and_fixed_schedule(tmp_path):
    x, y = fitting_arrays(); t = settings()
    full = train.train_head(x, y, x + .1, y, tmp_path / 'full', t, 24, 'id')
    assert train.train_head(x, y, x + .1, y, tmp_path / 'resume', t, 24, 'id', stop_after_updates=3) is None
    resumed = train.train_head(x, y, x + .1, y, tmp_path / 'resume', t, 24, 'id')
    assert full['best_epoch'] == resumed['best_epoch']
    for k, value in full['model'].items():
        torch.testing.assert_close(value, resumed['model'][k], rtol=0, atol=0)
    np.testing.assert_allclose(full['mean'], x.mean(axis=0, dtype=np.float64), rtol=0, atol=1e-14)
    state = torch.load(tmp_path / 'full/latest.pt', weights_only=False)
    assert state['history'][-1]['lr'] == pytest.approx(t['min_lr'])
    assert state['updates'] == 20  # 80 examples / 23, keeping the tail batch.
    assert state['history'][-1]['loss'] < state['history'][0]['loss']
    final = train.train_head(x, y, None, None, tmp_path / 'final', t, 24, 'final', fixed_epochs=2)
    state = torch.load(tmp_path / 'final/latest.pt', weights_only=False)
    assert final['updates'] == 8 and final['best_epoch'] == 2
    assert state['scheduler']['T_max'] == 20
    assert state['history'][-1]['lr'] > t['min_lr']
    p = model.probabilities({5: full}, x, np.full(len(x), 5))
    p2 = model.probabilities({5: torch.load(tmp_path / 'full/model.pt', weights_only=False)}, x, np.full(len(x), 5))
    np.testing.assert_array_equal(p, p2)
    with pytest.raises(ValueError, match='identity mismatch'):
        train.train_head(x, y, x, y, tmp_path / 'full', t, 24, 'changed')


def test_patience_keeps_earliest_tie(tmp_path, monkeypatch):
    x, y = fitting_arrays()
    monkeypatch.setattr(train, 'score', lambda *_: .5)
    t = settings(); t['patience'] = 2
    result = train.train_head(x, y, x, y, tmp_path, t, 10, 'early')
    assert result['best_epoch'] == 1 and result['stopped_epoch'] == 3


@pytest.mark.parametrize('failure', ['nan', 'missing_class', 'width'])
def test_invalid_training_inputs(tmp_path, failure):
    x, y = fitting_arrays()
    if failure == 'nan': x[0, 0] = np.nan
    if failure == 'missing_class': y[:] = 0
    if failure == 'width': x = x[:, :-1]
    with pytest.raises(ValueError, match='fitting requires'):
        train.train_head(x, y, x, y, tmp_path, settings(), 1, 'bad')


def test_feature_order_missing_and_equal_game_weight():
    cols = common.S + common.H + common.C
    frame = pd.DataFrame(np.vstack([np.ones(60) * n for n in range(5)]), columns=cols)
    q = pd.DataFrame([dict(k=5, games=np.arange(5))])
    np.testing.assert_array_equal(train.question_values(frame, q, cols), np.full((1, 60), 2))
    with pytest.raises(KeyError):
        train.question_values(frame.iloc[:4], q, cols)
    with pytest.raises(ValueError, match='schema'):
        train.question_values(frame[cols[::-1]], q, cols)
    a = frame.reset_index(names='game_index')
    bad = a.iloc[:1].copy(); bad.iloc[0, 1] = 100
    with pytest.raises(ValueError, match='conflicting'):
        features.merge_features([a, bad], cols)


def test_frozen_buffers_and_artifact_checksum(tmp_path):
    upstream = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3)).eval().requires_grad_(False)
    before = common.state_hash(upstream)
    with torch.inference_mode(): upstream(torch.randn(9, 3))
    common.assert_frozen(upstream, before)
    upstream[1].running_mean[0] += 1
    with pytest.raises(ValueError, match='frozen'):
        common.assert_frozen(upstream, before)
    frame = pd.DataFrame({'game_index': [1], 'x': [2.]})
    path = tmp_path / 'part.parquet'; common.publish_table(path, frame, {'model': 'one'})
    pd.testing.assert_frame_equal(common.load_table(path, {'model': 'one'}), frame)
    with pytest.raises(ValueError, match='mismatch'):
        common.load_table(path, {'model': 'other'})
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        common.load_table(path, {'model': 'one'})


def test_release_preaction_black_white_and_logp_means(tmp_path):
    lengths = [50, 61]
    # Sparse separated rows avoid captures and exercise different game lengths.
    moves = np.concatenate([np.arange(n) + 19 * 8 for n in lengths]).astype(np.int16)
    colors = np.concatenate([np.arange(n) % 2 for n in lengths]).astype(np.int8)
    meta = pd.DataFrame(dict(offset=[0, lengths[0]], length=lengths, color=['B', 'W']))
    write_store_cache(moves, colors, meta, tmp_path / 'cache')
    dataset = features.ReleaseDataset(tmp_path / 'cache')
    class Extreme(torch.nn.Module):
        def forward(self, x):
            # Make probabilities depend on the pre-action board's stone count.
            count = x[:, :2].sum((1, 2, 3))
            return {'rank_logits': count[:, None] * torch.arange(10)[None, :]}
    config = dict(device='cpu', precision='fp32', training={'channels_last': False}, evaluation={'per_gpu_batch_size': 7})
    try:
        dataset._ensure_open()
        assert np.diff(dataset._indices[1]).tolist() == [10, 15]
        assert all(p >= 30 for p in dataset._indices[2])
        samples = dataset.__getitems__([0, 10])
        assert samples[0]['planes'][16].all() and samples[1]['planes'][17].all()
        frame = infer_games(Extreme(), dataset, np.array([0, 1]), config, torch.device('cpu'))
        assert frame.position_count.tolist() == [10, 15]
        for game in range(2):
            start, stop = dataset._indices[1][game:game + 2]
            planes = torch.tensor(np.stack([v['planes'] for v in dataset.__getitems__(np.arange(start, stop))]), dtype=torch.float32)
            expected = Extreme()(planes)['rank_logits'].log_softmax(-1).double().mean(0).numpy()
            np.testing.assert_allclose(frame.iloc[game][[f'logp_{i}' for i in range(10)]].to_numpy(float), expected, atol=1e-6)
    finally:
        dataset.close()


def fake_bundle():
    head = model.Head()
    return dict(format='se_h_c_b2_bundle_v1', arm='se3', stage='final', columns=common.S + common.H + common.C,
                heads={k: dict(model=head.state_dict(), mean=np.zeros(60), scale=np.ones(60), best_epoch=1) for k in common.BUCKETS})


def test_aggregate_falls_back_whole_question_and_export(tmp_path):
    bundle = fake_bundle()
    frames = []
    for part in ('public', 'private'):
        q = pd.DataFrame([dict(question_id=f'{part}_{k}', k=k, color='B' if k < 15 else 'W', games=np.arange(k))
                          for k in common.BUCKETS])
        values = pd.DataFrame(np.zeros((19, 60)), columns=bundle['columns'])  # k=20 is incomplete.
        backup = pd.DataFrame([dict(question_id=f'{part}_20', num_games=20, humansl_games=20,
                                   fallback_v2=False, fallback_other=False,
                                   **{f'p_{r}': float(i == 4) for i, r in enumerate(RANKS)})])
        out = predict.aggregate_frames(bundle, q, values, backup)
        assert out.fallback_ahc.tolist() == [False, False, False, True]
        assert out.loc[3, 'rank'] == '1D'
        with pytest.raises(ValueError, match='existing AH'):
            predict.aggregate_frames(bundle, q, values)
        backup.loc[0, 'humansl_games'] = 19
        with pytest.raises(ValueError, match='complete AH'):
            predict.aggregate_frames(bundle, q, values, backup)
        directory = tmp_path / f'rank_{part}'; directory.mkdir()
        common.atomic_table(directory / 'se3_hc_b2_predictions.parquet', out)
        write_json(directory / 'se3_hc_b2_predictions.json', {'model': 'same-frozen-heads', 'sha256': file_hash(directory / 'se3_hc_b2_predictions.parquet')})
        frames.append(out)
    template = pd.DataFrame({'question_id': list(reversed(pd.concat(frames).question_id.tolist())), 'pred_rank': ['D'] * 8})
    template.to_csv(tmp_path / 'sample.csv', index=False)
    result = predict.export(tmp_path, tmp_path / 'sample.csv', tmp_path / 'submission.csv')
    assert result['questions'] == 8
    output = pd.read_csv(tmp_path / 'submission.csv', dtype=str)
    assert output.question_id.tolist() == template.question_id.tolist()
    assert list(output) == ['question_id', 'pred_rank']
    template.iloc[:4].to_csv(tmp_path / 'short.csv', index=False)
    with pytest.raises(ValueError, match='Public and Private'):
        predict.export(tmp_path, tmp_path / 'short.csv', tmp_path / 'bad.csv')


def test_bootstrap_clusters_players_and_same_pair():
    games = metadata()
    q = data.sample_questions(games[games.player_id.str.endswith('_u')], 2, 7, balanced=True)
    baseline = np.zeros(len(q)); candidate = np.ones(len(q)) * .1
    result = evaluate.paired_bootstrap(q, baseline, candidate, 30, 8)
    np.testing.assert_allclose(result['intervals']['equal_k'], [.1, .1])
    result = evaluate.paired_bootstrap(q, baseline, baseline, 30, 8)
    assert result['intervals']['equal_k'] == [0., 0.]


def test_full_cpu_training_and_evaluation(experiment, monkeypatch):
    config, canonical = experiment
    data.prepare(config)
    rng = np.random.default_rng(93)
    raw = rng.normal(0, .05, (len(canonical), 60)).astype(np.float32)
    raw[np.arange(len(canonical)), canonical.rank_id.to_numpy()] += 5
    def cached_features(config, arm, ids, **kwargs):
        columns = (common.S if arm == 'se3' else common.A) + common.H + common.C
        return pd.DataFrame(raw[ids], columns=columns, index=ids)
    monkeypatch.setattr(features, 'assemble', cached_features)
    monkeypatch.setattr(evaluate, 'assemble', cached_features)
    for arm in ('a', 'se3'):
        train.train(config, 'dev', arm)
        train.train(config, 'final', arm)
    result = evaluate.evaluate(config, 'final', 'newgames')
    assert set(result['delta']) == {'5', '10', '15', '20', 'equal_k'}
    assert result['delta']['equal_k'] == 0
    assert not result['candidate_gate']
    assert result['bootstrap']['intervals']['equal_k'] == [0., 0.]


def test_selected_se3_extract_cpu_and_reuse(tmp_path, monkeypatch):
    lengths = [50, 61]
    moves = np.concatenate([np.arange(n) + 19 * 8 for n in lengths]).astype(np.int16)
    colors = np.concatenate([np.arange(n) % 2 for n in lengths]).astype(np.int8)
    store_meta = pd.DataFrame(dict(offset=[0, 50], length=lengths, color=['B', 'W']))
    cache = write_store_cache(moves, colors, store_meta, tmp_path / 'cache')
    metadata = pd.read_parquet(cache / 'games.parquet')
    se_config = dict(model=dict(channels=2, blocks=1, policy_size=362, input_encoding='pre_action_own_opponent'),
                     device='cpu', precision='fp32', training={'channels_last': True}, evaluation={'per_gpu_batch_size': 8})
    from go_aicup.tasks.strength.reproduce_paper_3.model import StrengthEstimator
    upstream = StrengthEstimator(**se_config['model'])
    path = tmp_path / 'source.pt'; torch.save(dict(config=se_config, model=upstream.state_dict()), path)
    config = common.read_config(); config['paths']['se3_model'] = str(path)
    config['inference']['batch_size'] = 8
    directory = tmp_path / 'features'
    identity = dict(source={'models': {'se3_model': file_hash(path)}}, cache='synthetic')
    monkeypatch.setattr(features, 'context', lambda *_: (metadata, cache, directory, identity))
    def read(config, kind, **kwargs):
        parts = []
        for p in (directory / 'se3').glob('*.parquet'):
            parts.append(common.load_table(p, dict(**identity, component='se3')).drop(columns='position_count'))
        return common.checked_features(features.merge_features(parts, common.S), common.S)
    monkeypatch.setattr(features, 'read_component', read)
    result = features.extract(config, 'se3', release=tmp_path, device='cpu')
    assert result['games'] == 2 and result['positions'] == 25
    frame = read(config, 'se3')
    assert frame.shape == (2, 10) and np.isfinite(frame.to_numpy()).all()
    assert features.extract(config, 'se3', release=tmp_path, device='cpu')['games'] == 0
    shard = next((directory / 'se3').glob('*.parquet'))
    shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        features.extract(config, 'se3', release=tmp_path, device='cpu')


def test_c_means_side_order_and_failed_release_game(monkeypatch):
    from go_aicup.features import humansl_rankvec as rv
    games = [(np.arange(n, dtype=np.int16), (np.arange(n) % 2).astype(np.int8)) for n in (7, 10)]
    table = pd.DataFrame(dict(game_index=[10, 11], player_color=[0, 1]))
    monkeypatch.setattr(features.human, 'cache_games', lambda *_: games)
    def logp(forward, vectors, bins, globals_, played, own, batch):
        return torch.from_numpy(played.astype(np.float32))[:, None] + torch.arange(10)[None, :]
    monkeypatch.setattr(rv, 'game_logp', logp)
    class Pool:
        failed = False
        def map(self, fn, data, **kwargs):
            return [(np.zeros((len(m), 1), np.uint8), np.zeros((len(m), 1), np.float32), m)
                    if not self.failed or i == 0 else None for i, (m, _) in enumerate(data)]
    pool = Pool(); runtime = (None, None, torch.zeros(10, 2, 384), pool)
    frame = features.compute_c(runtime, table, Path('.'), batch=4)
    for i, ((moves, colors), row) in enumerate(zip(games, table.itertuples())):
        own = colors == row.player_color
        np.testing.assert_allclose(frame.iloc[i][common.C[:10]].to_numpy(float), moves[own].mean() + np.arange(10))
        np.testing.assert_allclose(frame.iloc[i][common.C[10:]].to_numpy(float), moves[~own].mean() + np.arange(10))
    pool.failed = True
    assert features.compute_c(runtime, table, Path('.'), batch=4, allow_failures=True).game_index.tolist() == [10]
    with pytest.raises(ValueError, match='C replay'):
        features.compute_c(runtime, table, Path('.'), batch=4)


def test_uncommitted_feature_shard_is_retried_not_reused(tmp_path):
    config = common.read_config()
    directory = tmp_path / 'features'; (directory / 'h').mkdir(parents=True)
    metadata = pd.DataFrame({'game_index': [1], 'player_color': [0], 'length': [50]})
    identity = {'source': 'same'}
    ctx = (metadata, tmp_path / 'cache', directory, identity)
    frame = pd.DataFrame(np.zeros((1, 30)), columns=common.H); frame.insert(0, 'game_index', [1])
    path = directory / 'h/shard_partial.parquet'; common.atomic_table(path, frame)
    assert features.read_component(config, 'h', release=tmp_path, ctx=ctx).empty
    common.publish_table(path, frame, dict(**identity, component='h'))
    assert features.read_component(config, 'h', release=tmp_path, ctx=ctx).index.tolist() == [1]
    path.write_bytes(b'damaged-complete-shard')
    with pytest.raises(ValueError, match='checksum'):
        features.read_component(config, 'h', release=tmp_path, ctx=ctx)
