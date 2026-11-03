"""Synthetic checks for AdamW continuation and globally exhaustive sampling."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_reproduce_paper_data import prepared, labelled
from test_reproduce_paper_train import probe_config
from test_reproduce_paper_evaluate import final_run
from go_aicup.artifacts import file_hash
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.reproduce_paper.train import train as train_parent
from go_aicup.tasks.strength.reproduce_paper.dataset import collate_groups
from go_aicup.tasks.strength.reproduce_paper_2.config import read_config, validate_config
from go_aicup.tasks.strength.reproduce_paper_2.dataset import GroupBatchSampler
from go_aicup.tasks.strength.reproduce_paper_2 import train as trainer


def fake_dataset(count=16):
    games = np.arange(10 * count, dtype=np.uint32)
    offsets = np.arange(len(games) + 1, dtype=np.int64) * 10
    return SimpleNamespace(seed=2026, split='train', label_dir=Path('/synthetic'),
        _ensure_open=lambda: None, _indices=(games, offsets, None),
        _cache=SimpleNamespace(rank_ids=np.repeat(np.arange(10), count)))


def game_rows(tasks, offsets):
    rows = np.searchsorted(offsets, [index for index, _ in tasks], side='right') - 1
    rows = rows.reshape(-1, 10, 7)
    assert np.all(rows == rows[..., :1])
    for group in np.asarray([index for index, _ in tasks]).reshape(-1, 7):
        assert len(set(group)) == 7
    return rows[..., 0].ravel().tolist()


def test_global_exhaustion_reshuffle_and_resume():
    dataset = fake_dataset()
    all_ranks = [list(GroupBatchSampler(dataset, groups_per_update=8, updates=4, rank=r, world_size=8)) for r in range(8)]
    epochs = []
    for epoch in range(2):
        rows = [row for step in range(epoch * 2, epoch * 2 + 2) for rank in range(8)
                for row in game_rows(all_ranks[rank][step], dataset._indices[1])]
        assert len(rows) == len(set(rows)) == 160
        assert set(rows) == set(range(160))
        epochs.append(rows)
    assert epochs[0] != epochs[1]
    for rank in range(8):
        resumed = list(GroupBatchSampler(dataset, groups_per_update=8, updates=4,
            start_step=1, rank=rank, world_size=8))
        assert resumed == all_ranks[rank][1:]
    single = list(GroupBatchSampler(dataset, groups_per_update=8, updates=4))
    assert single == [sum([all_ranks[r][s] for r in range(8)], []) for s in range(4)]
    with pytest.raises(ValueError, match='divisible'):
        GroupBatchSampler(fake_dataset(17), groups_per_update=8)
    dataset._indices[0][1] = dataset._indices[0][0]
    with pytest.raises(ValueError, match='unique'):
        GroupBatchSampler(dataset, groups_per_update=8)


def test_spawn_workers_and_prefetch_resume(labelled):
    dataset, _ = labelled
    def batches(start):
        sampler = GroupBatchSampler(dataset, groups_per_update=1, updates=4, start_step=start)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_groups,
            num_workers=2, multiprocessing_context='spawn', prefetch_factor=2)
        return [{k: b[k].clone() for k in ('sample_index', 'game_index', 'd4', 'value_target')} for b in loader]
    full, resumed = batches(0), batches(1)
    for left, right in zip(full[1:], resumed):
        for key in left:
            assert torch.equal(left[key], right[key])


def test_config_and_lr_boundaries():
    config = read_config()
    assert config['training']['updates'] == 70000
    assert config['source']['optimizer_steps'] == 35000
    assert config['split']['train_games_per_rank'] // config['training']['groups_per_update'] == 625
    assert config['training']['updates'] // 625 == 112
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=0.002)
    scheduler = trainer.scheduler_for(optimizer, config['training'])
    for completed in (0, 49999, 50000, 69999, 70000):
        assert scheduler.lr_lambdas[0](completed) * 0.002 == (0.002 if completed < 50000 else 0.001)
    scheduler.last_epoch = 49999
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0.001
    state = scheduler.state_dict()
    restored = trainer.scheduler_for(optimizer, config['training'])
    restored.load_state_dict(state)
    assert restored.last_epoch == 50000
    bad = copy.deepcopy(config)
    bad['training']['eps'] = 0
    with pytest.raises(ValueError, match='eps'):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad['paths']['output_dir'] = str(Path(config['paths']['source_checkpoint']).parent)
    with pytest.raises(ValueError, match='separate'):
        validate_config(bad)


@pytest.fixture
def continuation(labelled, tmp_path):
    parent = probe_config(labelled, tmp_path)
    train_parent(parent)
    path = resolve_checkpoint(Path(parent['paths']['output_dir']) / 'last.json')
    config = read_config()
    config.update(device='cpu', precision='fp32')
    config['paths'].update(parent['paths'])
    config['paths'].update(source_checkpoint=str(path), output_dir=str(tmp_path / 'adamw'))
    config['source'] = dict(optimizer_steps=4, sha256=file_hash(path))
    config['model'] = copy.deepcopy(parent['model'])
    config['split'] = copy.deepcopy(parent['split'])
    config['training'].update(updates=4, lr_drop_after_updates=2, groups_per_update=1,
        microbatch_groups=1, workers=0, checkpoint_every=1, deterministic=True,
        sync_batch_norm=False, channels_last=False)
    return config


def finish(config, resume=None):
    summary = trainer.train(config, resume=resume)
    if summary['status'] == 'milestone':
        summary = trainer.train(config, resume=Path(config['paths']['output_dir']) / 'last.json')
    return summary


def test_adamw_reset_and_exact_resume(continuation):
    config = continuation
    full = copy.deepcopy(config)
    full['paths']['output_dir'] += '_full'
    assert finish(full)['cumulative_updates'] == 8
    assert trainer.train(config, stop_after=1)['status'] == 'stopped'
    run = Path(config['paths']['output_dir'])
    payload = torch.load(resolve_checkpoint(run / 'last.json'), weights_only=False)
    assert payload['state']['optimizer_steps'] == 1
    assert payload['state']['epoch'] == 0 and payload['state']['next_batch'] == 1
    assert all(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and v['step'] == 1
               for v in payload['optimizer']['state'].values())
    with (run / 'train.jsonl').open('a') as handle:
        handle.write('{"step":2,"total":999}\n{"step":')
    finish(config, run / 'last.json')
    expected = torch.load(Path(full['paths']['output_dir']) / 'model.pt', weights_only=False)
    actual = torch.load(run / 'model.pt', weights_only=False)
    for key, value in expected['model'].items():
        torch.testing.assert_close(actual['model'][key], value, rtol=0, atol=0)
    rows = [json.loads(s) for s in (run / 'train.jsonl').read_text().splitlines()]
    expected_rows = [json.loads(s) for s in (Path(full['paths']['output_dir']) / 'train.jsonl').read_text().splitlines()]
    assert [r['lr'] for r in rows] == [0.002, 0.002, 0.001, 0.001]
    assert [r['total'] for r in rows] == [r['total'] for r in expected_rows]
    assert [r['cumulative_step'] for r in rows] == [5, 6, 7, 8]
    assert (run / 'best_1.pt').exists() and (run / 'best_2.pt').exists()
    assert not (run / 'checkpoints').exists()
    assert trainer.train(config)['updates_this_session'] == 0
    assert not (run / '.writer.lock').exists()


def test_refuse_changed_source_and_resume(continuation):
    config = continuation
    bad = copy.deepcopy(config)
    bad['source']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        trainer.train(bad)
    # Failed initialization leaves an empty directory after its lock is released.
    assert not (Path(config['paths']['output_dir']) / '.writer.lock').exists()
    trainer.train(config, stop_after=1)
    with pytest.raises(ValueError, match='nonempty'):
        trainer.train(config)
    bad = copy.deepcopy(config)
    bad['training']['eps'] = 1e-7
    with pytest.raises(ValueError, match='mismatch'):
        trainer.train(bad, resume=Path(config['paths']['output_dir']) / 'last.json')


def test_midpoint_publication_failure_retries_without_updates(continuation, monkeypatch):
    config = continuation
    publish = trainer.publish_milestone
    def fail(*args):
        raise OSError('simulated publication interruption')
    monkeypatch.setattr(trainer, 'publish_milestone', fail)
    with pytest.raises(OSError, match='publication'):
        trainer.train(config)
    run = Path(config['paths']['output_dir'])
    assert torch.load(resolve_checkpoint(run / 'last.json'), weights_only=False)['state']['optimizer_steps'] == 2
    monkeypatch.setattr(trainer, 'publish_milestone', publish)
    summary = trainer.train(config, resume=run / 'last.json')
    assert summary['status'] == 'milestone' and summary['updates_this_session'] == 0
    assert len((run / 'train.jsonl').read_text().splitlines()) == 2
    assert (Path(trainer.milestone_config(config)['paths']['output_dir']) / 'model.complete.json').exists()


def full_config(parent, output):
    config = read_config()
    config.update(device='cpu', precision='fp32')
    config['paths'].update(parent['paths'])
    source = Path(parent['paths']['output_dir']) / 'best_1.pt'
    payload = torch.load(resolve_checkpoint(source), weights_only=False)
    config['paths'].update(source_checkpoint=str(source), output_dir=str(output))
    config['source'] = dict(optimizer_steps=payload['state']['optimizer_steps'], sha256=file_hash(source))
    for key in ('model', 'split', 'questions'):
        config[key] = copy.deepcopy(parent[key])
    config['training'].update(updates=4, lr_drop_after_updates=2, groups_per_update=2,
        microbatch_groups=1, workers=0, checkpoint_every=1, deterministic=True,
        sync_batch_norm=False, channels_last=False)
    config['evaluation']['per_gpu_batch_size'] = 128
    return config


def test_full_cli_evaluation_failure_resume_and_completed_retry(final_run, tmp_path, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper_2 import evaluate as evaluator
    config = full_config(final_run, tmp_path / 'full_adamw')
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    original = evaluator.evaluate
    def fail(*args):
        raise OSError('simulated evaluation interruption')
    monkeypatch.setattr(evaluator, 'evaluate', fail)
    with pytest.raises(OSError, match='evaluation interruption'):
        trainer.main(['--config', str(path)])
    run = Path(config['paths']['output_dir'])
    checkpoint = torch.load(resolve_checkpoint(run / 'last.json'), weights_only=False)
    assert checkpoint['state']['optimizer_steps'] == 2
    monkeypatch.setattr(evaluator, 'evaluate', original)
    (run / '.stop-requested').touch()
    assert trainer.main(['--config', str(path), '--resume', str(run / 'last.json')]) == 0
    for folder in (run, Path(trainer.milestone_config(config)['paths']['output_dir'])):
        report = json.loads((folder / 'evaluation/report.json').read_text())
        assert report['status'] == 'completed' and all(str(k) in report['by_k'] for k in (5, 10, 15, 20))
    assert len((run / 'train.jsonl').read_text().splitlines()) == 4
    digest = file_hash(run / 'model.pt')
    def no_inference(*args, **kwargs):
        raise AssertionError('completed evaluation must reuse score shards')
    monkeypatch.setattr(evaluator.base, 'infer_positions', no_inference)
    assert trainer.main(['--config', str(path)]) == 0
    assert file_hash(run / 'model.pt') == digest
    assert len((run / 'train.jsonl').read_text().splitlines()) == 4


def test_two_process_full_workflow(final_run, tmp_path):
    import os
    import subprocess
    import sys
    parent = copy.deepcopy(final_run)
    parent['paths']['output_dir'] = str(tmp_path / 'ddp_parent')
    parent.pop('evaluation')
    parent_path = tmp_path / 'parent.json'
    parent_path.write_text(json.dumps(parent))
    prefix = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc-per-node=2', '-m']
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run(prefix + ['go_aicup.tasks.strength.reproduce_paper.train', '--config', str(parent_path)],
        capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    config = full_config(parent, tmp_path / 'ddp_adamw')
    path = tmp_path / 'ddp.json'
    path.write_text(json.dumps(config))
    command = prefix + ['go_aicup.tasks.strength.reproduce_paper_2.train', '--config', str(path)]
    result = subprocess.run(command + ['--stop-after', '2'],
        capture_output=True, text=True, timeout=120, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    run = Path(config['paths']['output_dir'])
    assert torch.load(resolve_checkpoint(run / 'last.json'), weights_only=False)['state']['optimizer_steps'] == 2
    result = subprocess.run(command + ['--resume', str(run / 'last.json')],
        capture_output=True, text=True, timeout=180, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = torch.load(run / 'best_1.pt', weights_only=False)
    assert payload['world_size'] == len(payload['rng_by_rank']) == 2
    assert json.loads((run / 'evaluation/report.json').read_text())['status'] == 'completed'
    snapshot = Path(trainer.milestone_config(config)['paths']['output_dir'])
    assert json.loads((snapshot / 'evaluation/report.json').read_text())['status'] == 'completed'
