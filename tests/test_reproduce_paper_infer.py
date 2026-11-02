"""Inference batching preserves sample coverage and eval-mode scalar scores."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_reproduce_paper_data import prepared, labelled
from test_reproduce_paper_train import probe_config
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
from go_aicup.tasks.strength.reproduce_paper.infer import infer_positions, benchmark
from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
from go_aicup.tasks.strength.reproduce_paper.train import train


def test_batch_layout_and_partition_invariance(labelled, tmp_path):
    source, _ = labelled
    config = probe_config(labelled, tmp_path)
    dataset = PaperDataset(source.directory, cache_dir=source.cache_dir)
    model = StrengthEstimator(channels=2, blocks=1)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    try:
        reference, report = infer_positions(model, dataset, config, torch.device('cpu'),
                                            batch_size=7, positions_per_rank=83, warmup_batches=1)
        assert report['sample_start'] == 0 and report['sample_stop'] == 83
        assert report['batches'] == 12 and report['measured_seconds'] > 0
        assert model.training
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        larger, report = infer_positions(model, dataset, config, torch.device('cpu'),
                                         batch_size=19, positions_per_rank=83)
        np.testing.assert_allclose(larger, reference, atol=1e-6, rtol=1e-6)
        model.to(memory_format=torch.channels_last)
        nhwc, _ = infer_positions(model, dataset, config, torch.device('cpu'), batch_size=19,
                                  positions_per_rank=83, channels_last=True)
        np.testing.assert_allclose(nhwc, reference, atol=1e-6, rtol=1e-6)
        parts = []
        for rank in range(3):
            scores, item = infer_positions(model, dataset, config, torch.device('cpu'),
                                           batch_size=31, rank=rank, world=3, channels_last=True)
            assert len(scores) == item['sample_stop'] - item['sample_start']
            parts.append(scores)
        all_scores, _ = infer_positions(model, dataset, config, torch.device('cpu'), batch_size=37,
                                        channels_last=True)
        np.testing.assert_allclose(np.concatenate(parts), all_scores, atol=1e-6, rtol=1e-6)
        empty, item = infer_positions(model, dataset, config, torch.device('cpu'),
                                      batch_size=3, rank=0, world=len(dataset) + 1)
        assert len(empty) == 0 and item['batches'] == 0
        augmented = PaperDataset(source.directory, augment=True)
        with pytest.raises(ValueError, match='augment=False'):
            infer_positions(model, augmented, config, torch.device('cpu'), batch_size=3)
    finally:
        dataset.close()


def test_checkpoint_inference_benchmark(labelled, tmp_path, monkeypatch):
    config = probe_config(labelled, tmp_path)
    train(config, stop_after=2)
    checkpoint = str(Path(config['paths']['output_dir']) / 'last.json')
    config['paths']['output_dir'] = str(tmp_path / 'inference')
    config['inference'] = dict(per_gpu_batch_sizes=[7, 19], positions_per_rank=83,
                               warmup_batches=1, split='valid', channels_last=False,
                               checkpoint=checkpoint)
    summary = benchmark(config)
    assert [row['status'] for row in summary['candidates']] == ['ok', 'ok']
    assert summary['candidates'][0]['positions'] == summary['candidates'][1]['positions'] == 83
    assert summary['candidates'][1]['max_abs_error'] < 1e-6
    assert (Path(config['paths']['output_dir']) / 'reference_rank_00.npy').exists()
    saved = json.loads((Path(config['paths']['output_dir']) / 'benchmark.json').read_text())
    assert saved['checkpoint_sha256'] == summary['checkpoint_sha256']
    with pytest.raises(ValueError, match='nonempty'):
        benchmark(config)
    wrong = copy.deepcopy(config)
    wrong['paths']['output_dir'] = str(tmp_path / 'wrong_model')
    wrong['model']['channels'] = 3
    with pytest.raises(ValueError, match='mismatch'):
        benchmark(wrong)
    assert not (Path(wrong['paths']['output_dir']) / '.writer.lock').exists()
    from go_aicup.tasks.strength.reproduce_paper import infer
    original = infer.infer_positions
    def fail_large(*args, **kwargs):
        if kwargs['batch_size'] > 7:
            raise torch.cuda.OutOfMemoryError('synthetic OOM')
        return original(*args, **kwargs)
    monkeypatch.setattr(infer, 'infer_positions', fail_large)
    oom_config = copy.deepcopy(config)
    oom_config['paths']['output_dir'] = str(tmp_path / 'oom')
    oom_config['inference']['per_gpu_batch_sizes'] = [7, 19, 37]
    result = benchmark(oom_config)
    assert [item['status'] for item in result['candidates']] == ['ok', 'oom']


def test_inference_rejects_bad_batches_and_cuda_world(monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper.config import read_config
    from go_aicup.tasks.strength.reproduce_paper.infer import DEFAULT_CONFIG
    config = read_config(DEFAULT_CONFIG)
    monkeypatch.setenv('WORLD_SIZE', '1')
    with pytest.raises(ValueError, match='exactly eight'):
        benchmark(config)
    config['inference']['per_gpu_batch_sizes'] = [19, 7]
    with pytest.raises(ValueError, match='increasing'):
        benchmark(config)


def test_two_process_inference_benchmark(labelled, tmp_path):
    import os
    import subprocess
    import sys
    config = probe_config(labelled, tmp_path)
    train(config, stop_after=2)
    checkpoint = str(Path(config['paths']['output_dir']) / 'last.json')
    config['paths']['output_dir'] = str(tmp_path / 'ddp_infer')
    config['inference'] = dict(per_gpu_batch_sizes=[7, 19], positions_per_rank=83,
                               warmup_batches=1, split='train', channels_last=False,
                               checkpoint=checkpoint)
    path = tmp_path / 'infer.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'go_aicup.tasks.strength.reproduce_paper.infer',
        '--config', str(path)], capture_output=True, text=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((Path(config['paths']['output_dir']) / 'benchmark.json').read_text())
    assert summary['world_size'] == 2
    assert all(item['positions'] == 166 for item in summary['candidates'])
    assert summary['candidates'][1]['max_abs_error'] < 1e-6
    first, second = summary['candidates'][0]['ranks']
    assert first['sample_stop'] <= second['sample_start']
