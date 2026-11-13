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


def full_config(labelled, tmp_path):
    config = probe_config(labelled, tmp_path)
    config['training'].pop('probe_updates')
    config['training'].pop('warmup_steps')
    config['training'].update(updates=4, lr_drop_after_updates=2, final_lr_after_updates=3)
    return config


def test_full_training_resume_final_model_and_no_resource_monitoring(labelled, tmp_path, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import train as trainer
    from go_aicup.tasks.strength.shared.runtime import model_state_hash
    from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
    config = full_config(labelled, tmp_path)

    def resource_monitor_called(*args, **kwargs):
        raise AssertionError('production training must not collect resource usage')

    monkeypatch.setattr(trainer, 'process_stats', resource_monitor_called)
    finalized_checkpoints = []
    finalize = trainer.finalize_training
    def capture_final_checkpoint(directory, *args):
        finalized_checkpoints.append(torch.load(resolve_checkpoint(directory / 'last.json'), weights_only=False))
        finalize(directory, *args)
    monkeypatch.setattr(trainer, 'finalize_training', capture_final_checkpoint)
    first = train(config, stop_after=2)
    directory = Path(config['paths']['output_dir'])
    assert first['status'] == 'stopped' and not (directory / 'model.pt').exists()
    log = directory / 'train.jsonl'
    with log.open('a') as handle:
        handle.write('{"step":3')
    train(config, resume=directory / 'last.json')
    payload = finalized_checkpoints[-1]
    assert not (directory / 'last.json').exists() and not (directory / 'checkpoints').exists()
    assert not (directory / 'best.json').exists()
    retained = json.loads((directory / 'best_checkpoints.json').read_text())['checkpoints']
    assert len(retained) == 2 and retained[0]['loss'] <= retained[1]['loss']
    for record in retained:
        checkpoint = torch.load(resolve_checkpoint(directory / record['file']), weights_only=False)
        assert checkpoint['state']['checkpoint_loss'] == record['loss']
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [row['step'] for row in events] == [1, 2, 3, 4]
    assert [row['lr'] for row in events] == [0.04, 0.04, 0.02, 0.002]
    assert not any('rss_bytes' in row or 'ranks' in row for row in events)
    assert payload['state']['optimizer_steps'] == 4
    assert payload['state']['checkpoint_loss_updates'] == 2
    assert payload['state']['checkpoint_loss'] == pytest.approx(sum(row['total'] for row in events[2:]) / 2)
    model_path = resolve_checkpoint(directory / 'model.pt')
    final = torch.load(model_path, weights_only=False)
    verifier = StrengthEstimator(**config['model'])
    verifier.load_state_dict(final['model'])
    assert model_state_hash(verifier) == final['model_state_sha256']
    assert final['optimizer_steps'] == 4
    for name, value in payload['model'].items():
        torch.testing.assert_close(value, final['model'][name], rtol=0, atol=0)
    continuous = copy.deepcopy(config)
    continuous['paths']['output_dir'] = str(tmp_path / 'full_continuous')
    train(continuous)
    expected = finalized_checkpoints[-1]
    for name, value in payload['model'].items():
        torch.testing.assert_close(value, expected['model'][name], rtol=0, atol=0)
    assert payload['scheduler'] == expected['scheduler']
    for parameter, state in payload['optimizer']['state'].items():
        torch.testing.assert_close(state['momentum_buffer'], expected['optimizer']['state'][parameter]['momentum_buffer'], rtol=0, atol=0)
    retry = train(config, resume=directory / 'last.json')
    assert retry['status'] == 'completed' and retry['updates_this_session'] == 0
    assert len(log.read_text().splitlines()) == 4


def test_full_training_stop_request_and_final_publication_retry(labelled, tmp_path, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import train as trainer
    config = full_config(labelled, tmp_path)
    directory = Path(config['paths']['output_dir'])
    original_log = trainer.log_event

    def stop_after_first(directory, rank, event):
        original_log(directory, rank, event)
        (directory / '.stop-requested').touch()

    monkeypatch.setattr(trainer, 'log_event', stop_after_first)
    stopped = train(config)
    assert stopped['completed_updates'] == 1 and stopped['status'] == 'stopped'
    assert not (directory / 'model.pt').exists()
    monkeypatch.setattr(trainer, 'log_event', original_log)
    finalize = trainer.finalize_training

    def fail_finalize(*args):
        raise OSError('simulated final publication failure')

    monkeypatch.setattr(trainer, 'finalize_training', fail_finalize)
    with pytest.raises(OSError, match='final publication failure'):
        train(config, resume=directory / 'last.json')
    payload = torch.load(resolve_checkpoint(directory / 'last.json'), weights_only=False)
    assert payload['state']['optimizer_steps'] == 4
    assert not (directory / '.writer.lock').exists()
    monkeypatch.setattr(trainer, 'finalize_training', finalize)
    final = train(config, resume=directory / 'last.json')
    assert final['status'] == 'completed' and final['updates_this_session'] == 0
    assert resolve_checkpoint(directory / 'model.pt').exists()


def test_production_config_has_full_budget_and_checkpoint_interval():
    config = read_config()
    assert config['training']['updates'] == 35000
    assert config['training']['checkpoint_every'] == 1000
    assert 'probe_updates' not in config['training']
    assert config['training']['microbatch_groups'] == 16
    assert config['training']['channels_last'] and not config['training']['cudnn_benchmark']
    assert config['precision'] == 'bf16' and config['device'] == 'cuda'
    assert Path(config['paths']['label_dir']).name == 'train'


def test_two_process_full_training(labelled, tmp_path):
    import os
    import subprocess
    import sys
    config = full_config(labelled, tmp_path)
    path = tmp_path / 'full_config.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'go_aicup.tasks.strength.reproduce_paper.train',
        '--config', str(path)], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    directory = Path(config['paths']['output_dir'])
    payload = torch.load(resolve_checkpoint(directory / 'model.pt'), weights_only=False)
    assert payload['optimizer_steps'] == 4
    assert not (directory / 'last.json').exists() and not (directory / 'checkpoints').exists()
    assert resolve_checkpoint(directory / 'model.pt').exists()
    assert len((directory / 'train.jsonl').read_text().splitlines()) == 4


def test_completed_training_cleanup_can_resume_after_partial_deletion(labelled, tmp_path, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import train as trainer
    config = full_config(labelled, tmp_path)
    directory = Path(config['paths']['output_dir'])
    remove = trainer.shutil.rmtree

    def fail_after_removing_checkpoints(path):
        remove(path)
        raise OSError('simulated cleanup interruption')

    monkeypatch.setattr(trainer.shutil, 'rmtree', fail_after_removing_checkpoints)
    with pytest.raises(OSError, match='cleanup interruption'):
        train(config)
    assert resolve_checkpoint(directory / 'model.pt').exists()
    assert not (directory / 'checkpoints').exists()
    assert (directory / 'last.json').exists()
    assert not (directory / '.writer.lock').exists()
    for name in ('last.json.tmp', '.stop-requested', 'train.tmp', 'model.pt.tmp', 'model.complete.json.tmp'):
        (directory / name).touch()
    preserved = {path.name: path.read_bytes() for path in directory.iterdir()
                 if path.name in ('config.json', 'data_identity.json', 'train.jsonl', 'environment.json',
                                  'model.pt', 'model.complete.json', 'training_summary.json')}
    monkeypatch.setattr(trainer.shutil, 'rmtree', remove)
    result = train(config, resume=directory / 'last.json')
    assert result['status'] == 'completed' and result['updates_this_session'] == 0
    assert not (directory / '.writer.lock').exists()
    assert not (directory / 'last.json').exists()
    assert not list(directory.glob('*.tmp')) and not (directory / '.stop-requested').exists()
    for name, contents in preserved.items():
        assert (directory / name).read_bytes() == contents
    assert train(config)['updates_this_session'] == 0


def test_cleanup_refuses_invalid_or_incomplete_final_model(labelled, tmp_path):
    from go_aicup.tasks.strength.reproduce_paper.train import cleanup_training
    config = full_config(labelled, tmp_path)
    directory = Path(config['paths']['output_dir'])
    train(config, stop_after=2)
    previous = resolve_checkpoint(directory / 'last.json')
    with pytest.raises(FileNotFoundError):
        cleanup_training(directory, config)
    assert previous.exists()
    train(config, resume=directory / 'last.json')
    (directory / 'checkpoints').mkdir()
    retained = directory / 'checkpoints' / 'retained.pt'
    retained.touch()
    (directory / 'last.json').write_text('{}')
    with (directory / 'model.pt').open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        cleanup_training(directory, config)
    assert retained.exists() and (directory / 'last.json').exists()


def test_best_checkpoint_copy_failure_preserves_resume_and_retries(labelled, tmp_path, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import train as trainer
    config = full_config(labelled, tmp_path)
    directory = Path(config['paths']['output_dir'])
    original = trainer.shutil.copyfile
    copies = []
    def fail_second_copy(source, destination):
        copies.append(source)
        if len(copies) == 2:
            Path(destination).write_bytes(b'incomplete')
            raise OSError('simulated checkpoint copy failure')
        return original(source, destination)
    monkeypatch.setattr(trainer.shutil, 'copyfile', fail_second_copy)
    with pytest.raises(OSError, match='copy failure'):
        train(config)
    assert resolve_checkpoint(directory / 'last.json').exists()
    assert len(list((directory / 'checkpoints').glob('*.pt'))) == 2
    assert not (directory / 'best_checkpoints.json').exists()
    monkeypatch.setattr(trainer.shutil, 'copyfile', original)
    assert train(config, resume=directory / 'last.json')['updates_this_session'] == 0
    assert not (directory / 'checkpoints').exists() and not (directory / 'best.json').exists()
    records = json.loads((directory / 'best_checkpoints.json').read_text())['checkpoints']
    assert len(records) == 2
    for record in records:
        assert resolve_checkpoint(directory / record['file']).exists()
    assert not list(directory.glob('*.tmp'))


def test_single_saved_checkpoint_and_corrupt_archive_guard(labelled, tmp_path):
    from go_aicup.tasks.strength.reproduce_paper.train import cleanup_training
    config = full_config(labelled, tmp_path)
    config['training']['checkpoint_every'] = 1000
    directory = Path(config['paths']['output_dir'])
    train(config)
    assert (directory / 'best_1.pt').exists() and not (directory / 'best_2.pt').exists()
    with (directory / 'best_1.pt').open('ab') as handle:
        handle.write(b'corrupt')
    (directory / 'checkpoints').mkdir()
    remaining = directory / 'checkpoints' / 'remaining.pt'
    remaining.touch()
    with pytest.raises(ValueError, match='hash mismatch'):
        cleanup_training(directory, config)
    assert remaining.exists()
