"""CPU checks for the bounded probe; no official data or GPU allocation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_reproduce_paper_data import prepared, labelled
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.train import (
    DEFAULT_PROBE_CONFIG, scheduler_for, train)
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint


def probe_config(labelled, tmp_path):
    dataset, _ = labelled
    config = read_config(DEFAULT_PROBE_CONFIG)
    config['device'], config['precision'] = 'cpu', 'fp32'
    config['paths'].update(dataset_dir=str(dataset.directory), cache_dir=str(dataset.cache_dir),
                           label_dir=str(dataset.label_dir), output_dir=str(tmp_path / 'probe'))
    config['split'] = dict(expected_games_per_rank=4, train_games_per_rank=2)
    config['model'].update(channels=2, blocks=1)
    config['training'].update(groups_per_update=2, microbatch_groups=1, workers=0,
                              probe_updates=4, warmup_steps=1, checkpoint_every=2,
                              deterministic=True, sync_batch_norm=False, channels_last=False)
    return config


def test_lr_boundaries_and_scheduler_restore():
    training = read_config()['training']
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=training['lr'])
    scheduler = scheduler_for(optimizer, training)
    for step in (0, 24999, 25000, 32499, 32500, 34999, 35000):
        scheduler.last_epoch = step
        expected = 0.04 if step < 25000 else 0.02 if step < 32500 else 0.002
        assert scheduler.lr_lambdas[0](step) * training['lr'] == pytest.approx(expected)
    optimizer = torch.optim.SGD([parameter], lr=training['lr'])
    scheduler = scheduler_for(optimizer, training)
    scheduler.last_epoch = 24999
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == 0.02
    state = scheduler.state_dict()
    restored = scheduler_for(optimizer, training)
    restored.load_state_dict(state)
    restored.last_epoch = 32499
    optimizer.step()
    restored.step()
    assert optimizer.param_groups[0]['lr'] == 0.002


def load_probe(config):
    directory = Path(config['paths']['output_dir'])
    payload = torch.load(resolve_checkpoint(directory / 'last.json'), weights_only=False)
    rows = [json.loads(line)['ranks'][0] for line in (directory / 'train.jsonl').read_text().splitlines()]
    return payload, rows


def test_probe_metrics_resume_and_refuse_overwrite(labelled, tmp_path):
    config = probe_config(labelled, tmp_path)
    summary = train(config)
    payload, rows = load_probe(config)
    assert summary['completed_updates'] == 4 and summary['measured_steps'] == 3
    assert summary['finite_checks_passed']
    assert summary['steps_per_second'] > 0 and summary['peak_node_rss_bytes'] > 0
    assert summary['gpu_hours'] == 0
    assert [row['step'] for row in rows] == [1, 2, 3, 4]
    assert rows[-1]['checkpoint_bytes'] > 0
    assert rows[-1]['grad_norm'] > 0 and rows[-1]['update_norm'] > 0
    assert rows[-1]['total'] == pytest.approx(sum(rows[-1][key] for key in ('bt', 'policy', 'value')))
    assert len(rows[-1]['score_mean_by_rank']) == 10
    with pytest.raises(ValueError, match='nonempty'):
        train(config)
    resumed_config = copy.deepcopy(config)
    resumed_config['paths']['output_dir'] = str(tmp_path / 'resumed')
    train(resumed_config, stop_after=2)
    log = Path(resumed_config['paths']['output_dir']) / 'train.jsonl'
    with log.open('a') as handle:
        handle.write(json.dumps(dict(step=3, ranks=[])) + '\n')
    train(resumed_config, resume=Path(resumed_config['paths']['output_dir']) / 'last.json')
    resumed_payload, resumed_rows = load_probe(resumed_config)
    assert len(resumed_rows) == 4
    for name, value in payload['model'].items():
        torch.testing.assert_close(value, resumed_payload['model'][name], rtol=0, atol=0)
    for left, right in zip(rows, resumed_rows):
        for key in ('lr', 'next_lr', 'total', 'bt', 'policy', 'value', 'update_norm'):
            assert left[key] == right[key]
    assert resumed_payload['state']['optimizer_steps'] == 4
    changed = copy.deepcopy(resumed_config)
    changed['training']['microbatch_groups'] = 2
    with pytest.raises(ValueError, match='mismatch'):
        train(changed, resume=Path(resumed_config['paths']['output_dir']) / 'last.json')
    assert not (Path(changed['paths']['output_dir']) / '.writer.lock').exists()


def test_probe_bounds(labelled, tmp_path):
    config = probe_config(labelled, tmp_path)
    config['training']['probe_updates'] = 501
    with pytest.raises(ValueError, match='1..500'):
        train(config)


def test_cuda_probe_requires_eight_processes(monkeypatch):
    config = read_config(DEFAULT_PROBE_CONFIG)
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='exactly eight'):
        train(config)


def test_two_process_cpu_trainer(labelled, tmp_path):
    import os
    import subprocess
    import sys
    config = probe_config(labelled, tmp_path)
    config['paths']['output_dir'] = str(tmp_path / 'ddp')
    config['training']['probe_updates'] = 2
    config['training']['warmup_steps'] = 0
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'go_aicup.tasks.strength.reproduce_paper.train',
        '--config', str(path)], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    payload, rows = load_probe(config)
    assert payload['world_size'] == 2 and len(payload['rng_by_rank']) == 2
    assert payload['state']['optimizer_steps'] == 2
    events = [json.loads(line) for line in (Path(config['paths']['output_dir']) / 'train.jsonl').read_text().splitlines()]
    assert len(events) == 2 and all(len(event['ranks']) == 2 for event in events)
    assert all(event['ranks'][0]['total'] == event['ranks'][1]['total'] for event in events)


def test_channels_last_training_and_resume(labelled, tmp_path):
    config = probe_config(labelled, tmp_path)
    config['training']['channels_last'] = True
    summary = train(config, stop_after=2)
    assert summary['channels_last']
    train(config, resume=Path(config['paths']['output_dir']) / 'last.json')
    actual, _ = load_probe(config)
    continuous = copy.deepcopy(config)
    continuous['paths']['output_dir'] = str(tmp_path / 'continuous_channels_last')
    train(continuous)
    expected, _ = load_probe(continuous)
    for name, value in actual['model'].items():
        torch.testing.assert_close(value, expected['model'][name], rtol=0, atol=0)



def test_channels_last_preserves_controlled_bn_update(labelled):
    from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler, collate_groups
    from go_aicup.tasks.strength.reproduce_paper.losses import grouped_losses
    from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
    dataset, _ = labelled
    tasks = next(iter(GroupBatchSampler(dataset, groups_per_update=2, updates=1)))
    batch = collate_groups(dataset.__getitems__(tasks))
    torch.manual_seed(2026)
    baseline = StrengthEstimator(channels=2, blocks=1).eval()
    candidate = copy.deepcopy(baseline).to(memory_format=torch.channels_last)
    losses = []
    for model, layout in ((baseline, torch.contiguous_format), (candidate, torch.channels_last)):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=0.0001)
        planes = batch['planes'].reshape(-1, 18, 19, 19).to(dtype=torch.float32, memory_format=layout)
        output = {name: value.reshape(2, 10, 7, *value.shape[1:]) for name, value in model(planes).items()}
        terms = grouped_losses(output, batch['policy_target'], batch['value_target'], batch['value_weight'])
        terms['total'].backward()
        optimizer.step()
        losses.append(terms)
    for key in ('bt', 'policy', 'value', 'total'):
        torch.testing.assert_close(losses[0][key], losses[1][key], atol=1e-5, rtol=1e-5)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[name], atol=1e-5, rtol=1e-5)
