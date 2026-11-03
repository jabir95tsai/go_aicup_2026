"""SE3 checks on synthetic data only; never launches GPU jobs."""
import copy
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash
from go_aicup.data.board_cache import RANKS, convert_training
from go_aicup.tasks.strength.reproduce_paper import config as base_config
from go_aicup.tasks.strength.reproduce_paper.prepare import prepare
from go_aicup.tasks.strength.reproduce_paper.questions import prepare_questions
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.reproduce_paper_3.config import read_config, validate_config
from go_aicup.tasks.strength.reproduce_paper_3.model import StrengthEstimator
from go_aicup.tasks.strength.reproduce_paper_3.losses import grouped_losses
from go_aicup.tasks.strength.reproduce_paper_3.dataset import GroupBatchSampler, collate_groups, validate_prepared
from go_aicup.tasks.strength.reproduce_paper_3.evaluate import (
    LOGP, prepare_monitor, predict_questions, official_scores, evaluate, monitor)
from go_aicup.tasks.strength.reproduce_paper_3 import train as trainer


@pytest.fixture(scope='module')
def prepared3(tmp_path_factory):
    root = tmp_path_factory.mktemp('se3')
    raw = root / 'raw'
    raw.mkdir()
    coords = [chr(97 + col) + chr(97 + row) for row in range(8, 17) for col in range(12)]
    sgf = '(;SZ[19]' + ''.join(f';{"B" if p % 2 == 0 else "W"}[{c}]' for p, c in enumerate(coords)) + ')'
    for rank in RANKS:
        with (raw / f'train_{rank}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
            writer.writeheader()
            for game in range(80):
                writer.writerow(dict(player_id=f'p{rank}', game_id=game, rank=rank,
                                     color='B' if game % 2 == 0 else 'W', sgf_content=sgf))
    cache = root / 'cache'
    convert_training(raw, cache, workers=1, games_per_shard=400, expected_games_per_rank=80)
    legacy = base_config.read_config()
    legacy['paths'].update(cache_dir=str(cache), dataset_dir=str(root / 'prepared'))
    legacy['split'] = dict(expected_games_per_rank=80, train_games_per_rank=40)
    legacy['questions']['validation'] = 40
    prepare(legacy)
    prepare_questions(legacy)
    labels = root / 'labels'
    labels.mkdir()
    count = json.loads((root / 'prepared/manifest.json').read_text())['splits']['train']['positions']
    np.save(labels / 'value.npy', np.linspace(-.8, .8, count, dtype=np.float32))
    return legacy, labels


def small_config(prepared3, tmp_path):
    legacy, labels = prepared3
    c = read_config()
    c['device'], c['precision'] = 'cpu', 'fp32'
    c['paths'].update(legacy['paths'])
    c['paths'].update(label_dir=str(labels), output_dir=str(tmp_path / 'run'))
    c['split'] = copy.deepcopy(legacy['split'])
    c['questions'] = copy.deepcopy(legacy['questions'])
    c['model'].update(channels=2, blocks=1)
    c['training'].update(updates=4, lr_drop_after_updates=2, groups_per_update=2,
        microbatch_groups=2, workers=0, checkpoint_every=1, sync_batch_norm=False,
        deterministic=True, channels_last=False)
    c['monitoring'].update(every=2, questions=40)
    c['evaluation']['per_gpu_batch_size'] = 128
    return c


def test_config_and_lr():
    c = read_config()
    assert c['positions']['positions_per_rank'] == 30 and c['model']['blocks'] == 18
    assert c['training']['updates'] == 100000
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.01)
    scheduler = trainer.scheduler_for(optimizer, c['training'])
    for step in (0, 69999, 70000, 99999, 100000):
        assert scheduler.lr_lambdas[0](step) * .01 == (.01 if step < 70000 else .005)
    scheduler.last_epoch = 69999
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == .005
    restored = trainer.scheduler_for(optimizer, c['training'])
    restored.load_state_dict(scheduler.state_dict())
    assert restored.last_epoch == 70000
    for field, value in [('positions_per_rank', 7)]:
        bad = copy.deepcopy(c)
        bad['positions'][field] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_model_loss_gradients_and_invalid_targets():
    torch.manual_seed(7)
    model = StrengthEstimator(channels=2, blocks=1)
    x = torch.randn(1, 10, 30, 18, 19, 19)
    out = model(x)
    assert out['rank_logits'].shape == (1, 10, 30, 10)
    assert out['policy_logits'].shape == (1, 10, 30, 362)
    rank = torch.arange(10).reshape(1, 10, 1).expand(1, 10, 30)
    policy = torch.randint(362, (1, 10, 30))
    value = torch.rand(1, 10, 30) * 2 - 1
    weights = torch.ones_like(value)
    terms = grouped_losses(out, rank, policy, value, weights)
    assert all(v.dtype == torch.float32 and torch.isfinite(v) for v in terms.values())
    torch.testing.assert_close(terms['strength'], torch.nn.functional.cross_entropy(out['rank_logits'].reshape(-1, 10), rank.reshape(-1)))
    terms['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.randn(1, 10, 7, 18, 19, 19))
    for target in (rank.float(), rank + 10):
        with pytest.raises(ValueError):
            grouped_losses(out, target, policy, value, weights)
    with pytest.raises(ValueError):
        grouped_losses(out, rank, policy, value * float('nan'), weights)


def fake_dataset(count=16, positions=35):
    games = np.arange(10 * count, dtype=np.uint32)
    offsets = np.arange(len(games) + 1, dtype=np.int64) * positions
    return SimpleNamespace(seed=2026, split='train', label_dir=Path('/synthetic'),
        _ensure_open=lambda: None, _indices=(games, offsets, None),
        _cache=SimpleNamespace(rank_ids=np.repeat(np.arange(10), count)))


def test_eight_rank_exhaustion_positions_and_resume():
    data = fake_dataset()
    all_ranks = [list(GroupBatchSampler(data, groups_per_update=8, updates=4, rank=r, world_size=8)) for r in range(8)]
    epochs = []
    for epoch in range(2):
        rows = []
        for step in range(epoch * 2, epoch * 2 + 2):
            for rank in range(8):
                indices = np.array([i for i, _ in all_ranks[rank][step]]).reshape(-1, 30)
                assert all(len(set(row)) == 30 for row in indices)
                games = indices // 35
                assert np.all(games == games[:, :1])
                rows.extend(games[:, 0])
        assert len(rows) == len(set(rows)) == 160
        epochs.append(rows)
    assert epochs[0] != epochs[1]
    for rank in range(8):
        assert list(GroupBatchSampler(data, groups_per_update=8, updates=4, start_step=1, rank=rank, world_size=8)) == all_ranks[rank][1:]
    with pytest.raises(ValueError, match='thirty'):
        GroupBatchSampler(fake_dataset(positions=29), groups_per_update=8)


def test_prepared_monitor_and_spawn_resume(prepared3, tmp_path):
    c = small_config(prepared3, tmp_path)
    root = Path(c['paths']['dataset_dir'])
    before = {p: (p.stat().st_mtime_ns, file_hash(p)) for p in root.iterdir()}
    identity = validate_prepared(c)
    assert identity['available_positions']['positions_per_rank'] == 7
    assert identity['sampled_positions_per_game'] == 30
    first = prepare_monitor(c)
    assert prepare_monitor(c) == first and first['questions'] == 40
    questions = pd.read_parquet(Path(c['paths']['output_dir']) / 'monitor/questions.parquet')
    assert (questions.groupby(['k', 'rank_id']).size() == 1).all()
    position = np.load(Path(c['paths']['output_dir']) / 'monitor/positions.npz')
    assert all(len(set(row)) == 30 for row in position['samples'])
    assert all((p.stat().st_mtime_ns, file_hash(p)) == value for p, value in before.items())
    data = PaperDataset(root, augment=True, label_dir=c['paths']['label_dir'], cache_dir=c['paths']['cache_dir'])
    def batches(start):
        loader = DataLoader(data, batch_sampler=GroupBatchSampler(data, groups_per_update=2, updates=3, start_step=start),
                            collate_fn=collate_groups, num_workers=2, multiprocessing_context='spawn', prefetch_factor=2)
        return [{k: b[k] for k in ('sample_index', 'd4', 'value_target')} for b in loader]
    try:
        full, resumed = batches(0), batches(1)
        for left, right in zip(full[1:], resumed):
            for name in left:
                assert torch.equal(left[name], right[name])
    finally:
        data.close()


def score_result(score):
    return dict(equal_k=score, buckets={str(k): dict(n=1000, exact=int(score * 1000) if math.isfinite(score) else 0, adjacent=0, score=score) for k in (5, 10, 15, 20)})


def test_patience_ties_retry_and_failures():
    state = dict(optimizer_steps=5000, evaluated_step=0, validation_pending=True,
                 best_score=None, best_step=None, lower_count=0)
    state, accepted = trainer.validation_transition(state, score_result(.6), 5000)
    assert accepted and state['lower_count'] == 0
    for step, score, lower in [(10000, .5, 1), (15000, .6, 0), (20000, .5, 1), (25000, .4, 2), (30000, .3, 3)]:
        state.update(optimizer_steps=step, validation_pending=True)
        state, _ = trainer.validation_transition(state, score_result(score), step)
        assert state['lower_count'] == lower
        retried, accepted = trainer.validation_transition(state, score_result(score), step)
        assert retried == state and not accepted
    state.update(optimizer_steps=35000, validation_pending=True)
    original = dict(state)
    with pytest.raises(ValueError):
        trainer.validation_transition(state, score_result(float('nan')), 35000)
    assert state == original
    result = score_result(.2)
    result['buckets'].pop('5')
    with pytest.raises(ValueError):
        trainer.validation_transition(state, result, 35000)
    assert state == original


def test_game_equal_logp_argmax_and_scoring():
    questions = pd.DataFrame([dict(question_id=str(k), k=k, rank_id=0, games=np.arange(k)) for k in (5, 10, 15, 20)])
    # A long game votes for rank 1; each other game votes for rank 0.
    values = np.full((20, 10), -20.)
    values[:, 0] = -.2
    values[:, 1] = -1.
    values[0, :2] = [-2., 0.]
    games = pd.DataFrame(values, columns=LOGP)
    games['game_index'] = range(20)
    games['position_count'] = [10000] + [30] * 19
    predicted = predict_questions(games, questions)
    assert (predicted.predicted_rank_id == 0).all()
    report = official_scores(predicted)
    assert report['equal_k'] == 1
    games[LOGP] = 0.
    assert (predict_questions(games, questions).predicted_rank_id == 0).all()
    predicted['predicted_rank_id'] = 1
    assert official_scores(predicted)['equal_k'] == 1 / math.e


def test_training_exact_resume_and_full_evaluation(prepared3, tmp_path):
    config = small_config(prepared3, tmp_path)
    full = copy.deepcopy(config)
    full['paths']['output_dir'] += '_full'
    summary = trainer.train(full)
    assert summary['status'] == 'completed' and summary['completed_updates'] == 4
    trainer.train(config, stop_after=1)
    with (Path(config['paths']['output_dir']) / 'train.jsonl').open('a') as handle:
        handle.write('{"step": 2,')  # Interrupted writes must not prevent checkpoint recovery.
    resumed = trainer.train(config, resume=Path(config['paths']['output_dir']) / 'last.json')
    assert resumed['status'] == 'completed'
    left = trainer.load_artifact(resolve_checkpoint(Path(full['paths']['output_dir']) / 'last.json'), full)
    right = trainer.load_artifact(resolve_checkpoint(Path(config['paths']['output_dir']) / 'last.json'), config)
    for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], rtol=0, atol=0)
    assert left['state'] == right['state']
    assert right['state']['evaluated_step'] == 4
    assert len(list((Path(config['paths']['output_dir']) / 'checkpoints').glob('*.pt'))) == 1
    final = trainer.load_artifact(Path(config['paths']['output_dir']) / 'model.pt', config)
    assert final['best_step'] == right['state']['best_step']
    report = evaluate(config)
    assert set(report['buckets']) == {'5', '10', '15', '20'}
    assert evaluate(config, report_only=True) == report
    saved = pd.read_parquet(Path(config['paths']['output_dir']) / 'evaluation/validation_games.parquet')
    assert saved.position_count.min() > 30 and len(saved) == 400
    assert not (Path(config['paths']['output_dir']) / 'evaluation/centroids.npy').exists()


@pytest.mark.parametrize('failure', ['inference', 'best', 'record', 'commit'])
def test_validation_failure_retries_without_double_patience(prepared3, tmp_path, monkeypatch, failure):
    c = small_config(prepared3, tmp_path)
    original_monitor, original_atomic, original_json = trainer.monitor, trainer.atomic_payload, trainer.write_json
    failed = False
    def fail_once():
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError('injected transaction failure')
    def fake_monitor(*args):
        if failure == 'inference':
            fail_once()
        return score_result(.6)
    def atomic(path, payload):
        if failure == 'best' and path.parent.name == 'best':
            fail_once()
        if failure == 'commit' and path.parent.name == 'checkpoints' and payload['state']['evaluated_step'] == 2:
            fail_once()
        return original_atomic(path, payload)
    def write(path, value):
        if failure == 'record' and path.parent.name == 'validation':
            fail_once()
        return original_json(path, value)
    monkeypatch.setattr(trainer, 'monitor', fake_monitor)
    monkeypatch.setattr(trainer, 'atomic_payload', atomic)
    monkeypatch.setattr(trainer, 'write_json', write)
    with pytest.raises(RuntimeError, match='injected'):
        trainer.train(c)
    run = Path(c['paths']['output_dir'])
    pending = trainer.load_artifact(resolve_checkpoint(run / 'last.json'), c)
    assert pending['state']['optimizer_steps'] == 2
    assert pending['state']['validation_pending'] and pending['state']['evaluated_step'] == 0
    assert trainer.train(c, resume=run / 'last.json')['status'] == 'completed'
    state = trainer.load_artifact(resolve_checkpoint(run / 'last.json'), c)['state']
    assert state['best_step'] == 4 and state['lower_count'] == 0
    assert len((run / 'validation.jsonl').read_text().splitlines()) == 2


def test_early_stop_publishes_best_and_evaluates(prepared3, tmp_path, monkeypatch):
    c = small_config(prepared3, tmp_path)
    c['training']['updates'] = 12
    scores = iter([.6, .5, .4, .3])
    monkeypatch.setattr(trainer, 'monitor', lambda *args: score_result(next(scores)))
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(c))
    trainer.main(['--config', str(config_path)])
    run = Path(c['paths']['output_dir'])
    final = trainer.load_artifact(run / 'model.pt', c)
    assert final['best_step'] == 2 and final['stopped_step'] == 8 and final['stop_reason'] == 'patience'
    assert final['state']['optimizer_steps'] == 2
    assert (run / 'evaluation/result.json').is_file()
    before = file_hash(run / 'model.pt')
    trainer.main(['--config', str(config_path)])
    assert file_hash(run / 'model.pt') == before
    assert len((run / 'train.jsonl').read_text().splitlines()) == 8


def test_monitor_restores_rng_and_batchnorm(prepared3, tmp_path):
    from go_aicup.tasks.strength.shared.training import rng_state
    c = small_config(prepared3, tmp_path)
    prepare_monitor(c)
    data = PaperDataset(c['paths']['dataset_dir'], split='valid', cache_dir=c['paths']['cache_dir'])
    model = StrengthEstimator(channels=2, blocks=1).train()
    torch.manual_seed(8)
    before = rng_state(torch.device('cpu'), model)
    monitor(model, data, c, torch.device('cpu'), 0, 1)
    after = rng_state(torch.device('cpu'), model)
    assert model.training
    assert torch.equal(before['torch'], after['torch'])
    assert np.array_equal(before['numpy'][1], after['numpy'][1])
    for name in before['buffers']:
        torch.testing.assert_close(before['buffers'][name], after['buffers'][name], rtol=0, atol=0)
    data.close()


def _gradient_batch():
    torch.manual_seed(43)
    return (torch.randn(2, 10, 30, 18, 19, 19),
            torch.arange(10).reshape(1, 10, 1).expand(2, 10, 30),
            torch.randint(362, (2, 10, 30)), torch.rand(2, 10, 30) * 2 - 1, torch.ones(2, 10, 30))


def _gloo_worker(rank, store, destination):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(7)
        model = DistributedDataParallel(StrengthEstimator(channels=2, blocks=1).eval())
        batch = _gradient_batch()
        rows = slice(rank, rank + 1)
        terms = grouped_losses(model(batch[0][rows]), *(t[rows] for t in batch[1:]))
        terms['total'].backward()
        means = torch.stack([terms[k].detach() for k in ('total', 'strength', 'policy', 'value')])
        dist.all_reduce(means)
        if rank == 0:
            torch.save(dict(means=means / 2, grads=torch.cat([p.grad.flatten() for p in model.parameters()])), destination)
    finally:
        dist.destroy_process_group()


def test_gloo_global_loss_and_gradient(tmp_path):
    torch.manual_seed(7)
    model = StrengthEstimator(channels=2, blocks=1).eval()
    batch = _gradient_batch()
    terms = grouped_losses(model(batch[0]), *batch[1:])
    terms['total'].backward()
    result = tmp_path / 'gloo.pt'
    torch.multiprocessing.spawn(_gloo_worker, args=(str(tmp_path / 'store'), str(result)), nprocs=2)
    loaded = torch.load(result, weights_only=True)
    torch.testing.assert_close(loaded['means'], torch.stack([terms[k] for k in ('total', 'strength', 'policy', 'value')]), rtol=1e-5, atol=1e-7)
    reference = torch.cat([p.grad.flatten() for p in model.parameters()])
    difference = loaded['grads'] - reference
    relative = float(difference.norm() / reference.norm())
    absolute = float(difference.abs().max())
    assert relative < 1e-5 and absolute < 1e-6
    print(f'SE3 Gloo gradient relative L2={relative:.3g}, max absolute={absolute:.3g}')


def test_gloo_trainer_resume(prepared3, tmp_path):
    import subprocess
    import sys
    c = small_config(prepared3, tmp_path)
    c['training'].update(updates=2, lr_drop_after_updates=1)
    path = tmp_path / 'ddp.json'
    path.write_text(json.dumps(c))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2',
               '-m', 'go_aicup.tasks.strength.reproduce_paper_3.train', '--config', str(path)]
    for extra in (['--stop-after', '1'], ['--resume', str(Path(c['paths']['output_dir']) / 'last.json')]):
        result = subprocess.run(command + extra, capture_output=True, text=True, timeout=120,
                                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        assert result.returncode == 0, result.stdout + result.stderr
    payload = trainer.load_artifact(Path(c['paths']['output_dir']) / 'model.pt', c)
    assert payload['world_size'] == 2 and len(payload['rng_by_rank']) == 2
    assert payload['state']['best_step'] == 2
    assert (Path(c['paths']['output_dir']) / 'evaluation/result.json').exists()


def test_persisted_bucket_order_keeps_exact_equal_k():
    from go_aicup.artifacts import object_hash
    predictions = pd.DataFrame(dict(k=[5, 5, 5, 10, 10, 15, 15, 15, 20],
        rank_id=[0] * 9, predicted_rank_id=[0, 1, 2, 0, 2, 1, 0, 1, 2]))
    result = official_scores(predictions)
    persisted = json.loads(json.dumps(result, sort_keys=True))
    state = dict(optimizer_steps=5000, evaluated_step=0, validation_pending=True,
                 best_score=None, best_step=None, lower_count=0)
    first, _ = trainer.validation_transition(state, result, 5000)
    retry, _ = trainer.validation_transition(state, persisted, 5000)
    assert object_hash(first) == object_hash(retry)


def test_config_mismatch_and_readonly_shortage(prepared3, tmp_path):
    import shutil
    from go_aicup.artifacts import write_json, object_hash
    c = small_config(prepared3, tmp_path)
    bad = copy.deepcopy(c)
    bad['split']['train_games_per_rank'] = 39
    with pytest.raises(ValueError, match='identity'):
        validate_prepared(bad)
    root = tmp_path / 'short'
    shutil.copytree(c['paths']['dataset_dir'], root)
    offsets = np.load(root / 'train_offsets.npy')
    offsets[1] = 29
    np.save(root / 'train_offsets.npy', offsets)
    record = json.loads((root / 'manifest.json').read_text())
    record['files']['train_offsets.npy'] = file_hash(root / 'train_offsets.npy')
    write_json(root / 'manifest.json', record)
    c['paths']['dataset_dir'] = str(root)
    before = file_hash(root / 'manifest.json')
    with pytest.raises(ValueError, match='thirty'):
        validate_prepared(c)
    assert file_hash(root / 'manifest.json') == before


def test_fp32_logsoftmax_and_position_means(prepared3, tmp_path):
    from go_aicup.tasks.strength.reproduce_paper_3.evaluate import infer_games
    c = small_config(prepared3, tmp_path)
    dataset = PaperDataset(c['paths']['dataset_dir'], split='valid', cache_dir=c['paths']['cache_dir'])
    dataset._ensure_open()
    class Extreme(torch.nn.Module):
        def forward(self, x):
            logits = torch.arange(10, dtype=torch.float32).mul(1000).expand(len(x), 10)
            return dict(rank_logits=logits)
    try:
        games = dataset._indices[0][:2]
        frame = infer_games(Extreme(), dataset, games, c, torch.device('cpu'))
        expected = torch.arange(10, dtype=torch.float32).mul(1000).log_softmax(-1).numpy()
        np.testing.assert_allclose(frame[LOGP], np.broadcast_to(expected, (2, 10)), rtol=0, atol=0)
        offsets = dataset._indices[1]
        assert frame.position_count.tolist() == np.diff(offsets[:3]).tolist()
    finally:
        dataset.close()


def test_smoke_metrics_and_no_full_evaluation(prepared3, tmp_path, monkeypatch):
    import go_aicup.tasks.strength.reproduce_paper_3.evaluate as evaluator
    c = small_config(prepared3, tmp_path)
    c['smoke'] = True
    c['training']['updates'] = 2
    path = tmp_path / 'smoke.json'
    path.write_text(json.dumps(c))
    def unexpected(*args, **kwargs):
        raise AssertionError('smoke must not run full validation')
    monkeypatch.setattr(evaluator, 'evaluate', unexpected)
    trainer.main(['--config', str(path)])
    run = Path(c['paths']['output_dir'])
    events = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
    assert len(events) == 2
    assert all(row['update_norm'] > 0 and row['grad_norm'] > 0 and len(row['ranks']) == 1 for row in events)
    assert json.loads((run / 'checkpoint_metrics.json').read_text())['bytes'] > 0
    assert not (run / 'evaluation').exists()


def test_final_publication_failure_retries_without_updates(prepared3, tmp_path, monkeypatch):
    c = small_config(prepared3, tmp_path)
    c['training']['updates'] = 2
    original = trainer.atomic_payload
    def fail_model(path, payload):
        if path.name == 'model.pt':
            raise RuntimeError('injected final publication failure')
        return original(path, payload)
    monkeypatch.setattr(trainer, 'atomic_payload', fail_model)
    with pytest.raises(RuntimeError, match='final publication'):
        trainer.train(c)
    run = Path(c['paths']['output_dir'])
    assert (run / 'best.json').is_file() and not (run / 'model.complete.json').exists()
    monkeypatch.setattr(trainer, 'atomic_payload', original)
    summary = trainer.train(c, resume=run / 'last.json')
    assert summary['status'] == 'completed' and summary['updates_this_session'] == 0
    assert len((run / 'train.jsonl').read_text().splitlines()) == 2
