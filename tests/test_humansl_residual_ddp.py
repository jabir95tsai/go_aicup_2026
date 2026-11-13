"""Actual trainer: two-process Gloo, complete-group accumulation and exact resume."""
import copy
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from go_aicup.tasks.strength.humansl_residual_b2 import data, dataset, train as training
from go_aicup.tasks.strength.humansl_residual_b2.config import read_config
from go_aicup.tasks.strength.humansl_residual_b2.model import ResidualClassifier


class Human(nn.Module):
    config = {}
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.bn = nn.BatchNorm2d(1)
    def forward_features(self, spatial, global_inputs, metadata):
        return (self.bn(spatial[:, :1]) * self.scale).expand(-1, 384, -1, -1)


class FixedBatchStatsClassifier(ResidualClassifier):
    def train(self, mode=True):
        super().train(mode)
        # Fixed statistics make single-process and distributed microbatches comparable.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


class SyntheticDataset:
    def __init__(self, config, games, model_config, **kwargs):
        self.games, self.seed = games.reset_index(drop=True), config['seed']
    def __len__(self):
        return len(self.games)
    def __getitem__(self, task):
        row, step = task
        game = self.games.iloc[row]
        rng = np.random.default_rng([self.seed, int(game.game_index), step])
        return dict(spatial=rng.integers(0, 2, (30, 1, 19, 19), dtype=np.uint8),
                    global_inputs=np.zeros((30, 1), np.float32), metadata=np.zeros((30, 1), np.float32),
                    actions=np.arange(30, dtype=np.int64), sides=np.repeat([0, 1], 15),
                    labels=np.full(30, game.rank_id, np.int64),
                    values=np.full(30, (game.rank_id - 4.5) / 10, np.float32),
                    ply=np.arange(30, 60, dtype=np.int64))


def load_synthetic(config, stage, source=None, device='cpu'):
    model = FixedBatchStatsClassifier(Human(), stage=stage, channels=2, blocks=1)
    if source:
        model.load_stage1(torch.load(source, map_location='cpu', weights_only=False)['model'])
    return model.to(device)


def rows():
    return pd.DataFrame([dict(game_index=r * 8 + g, rank_id=r, player_id=f'{r}-{g}',
                              role='D', oof_fold=g % 4 + 1) for r in range(10) for g in range(8)])


def setup_patches():
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch.object(data, 'games', lambda *args, **kwargs: rows()))
    stack.enter_context(patch.object(data, 'load_prepared', lambda *args, **kwargs: {'dataset_id': 'synthetic-ddp'}))
    stack.enter_context(patch.object(dataset, 'SelectedDataset', SyntheticDataset))
    stack.enter_context(patch.object(training, 'load_model', load_synthetic))
    stack.enter_context(patch.object(training, 'code_identity', lambda: {'synthetic-ddp': 'fixed'}))
    stack.enter_context(patch.object(training, 'environment', lambda *args: {}))
    stack.enter_context(patch.object(training, 'monitor', lambda *args: {'fused': {'equal_k': .25}}))
    return stack


def candidate(config, stage=1, name='final.pt'):
    return Path(config['run_dir']) / 'dev' / 'fold_all' / f'stage{stage}' / name


def stage_one_and_two(config, rank, ports=None):
    if rank == 0:
        path = Path(config['run_dir']) / 'selection' / 'stage1.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'selected': 'final'}))
    if ports:
        os.environ['MASTER_PORT'] = str(ports[0])
    training.train(config, 1)
    if ports:
        os.environ['MASTER_PORT'] = str(ports[1])
    # The next training invocation creates a new process group and synchronizes initialization.
    training.train(config, 2, source=candidate(config))


def worker(rank, config, ports):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(ports[0]), WORLD_SIZE='2',
                      LOCAL_WORLD_SIZE='2', RANK=str(rank), LOCAL_RANK=str(rank))
    with setup_patches():
        complete = copy.deepcopy(config); complete['run_dir'] += '_ddp_full'
        stage_one_and_two(complete, rank, ports[:2])
        resumed = copy.deepcopy(config); resumed['run_dir'] += '_ddp_resumed'
        os.environ['MASTER_PORT'] = str(ports[2])
        assert training.train(resumed, 1, stop_after=1)['status'] == 'paused'
        os.environ['MASTER_PORT'] = str(ports[3])
        assert training.train(resumed, 1)['status'] == 'completed'


def config_for_test(tmp_path):
    config = read_config()
    source = tmp_path / 'human.ckpt'; source.write_bytes(b'ddp-fixture-source')
    config.update(run_dir=str(tmp_path / 'run'), device='cpu', precision='fp32', smoke=False)
    config['model']['human_checkpoint'] = str(source)
    config['training'].update(groups_per_update=4, microbatch_groups=1, workers=0, channels_last=False,
                              sync_batch_norm=False, checkpoint_every=1, optimizer='sgd', weight_decay=0.,
                              allow_tf32=False)
    config['stage1'].update(updates=2, lr=.01, lr_schedule=[])
    config['stage2'].update(updates=2, lr=.01, lr_schedule=[])
    return config


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Gloo unavailable')
def test_actual_trainer_gloo_matches_global_mean_and_resumes_all_rank_rng(tmp_path, monkeypatch):
    config = config_for_test(tmp_path)
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    for name in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'RANK', 'LOCAL_RANK'):
        monkeypatch.delenv(name, raising=False)
    try:
        single = copy.deepcopy(config); single['run_dir'] += '_single'
        with setup_patches():
            stage_one_and_two(single, 0)
        sockets = [socket.socket() for _ in range(4)]
        for sock in sockets:
            sock.bind(('127.0.0.1', 0))
        ports = [sock.getsockname()[1] for sock in sockets]
        for sock in sockets:
            sock.close()
        mp.spawn(worker, args=(config, ports), nprocs=2, join=True)
        complete = copy.deepcopy(config); complete['run_dir'] += '_ddp_full'
        resumed = copy.deepcopy(config); resumed['run_dir'] += '_ddp_resumed'
        for stage in (1, 2):
            reference = torch.load(candidate(single, stage), weights_only=False)
            actual = torch.load(candidate(complete, stage), weights_only=False)
            assert actual['world_size'] == 2 and len(actual['rng']) == 2
            assert not torch.equal(actual['rng'][0]['torch'], actual['rng'][1]['torch'])
            assert actual['sampler']['completed_step'] == 2
            assert actual['sampler'] == reference['sampler']
            for key in reference['model']:
                torch.testing.assert_close(reference['model'][key], actual['model'][key], rtol=2e-5, atol=2e-7)
            assert all(stream['cursor'] == 0 and stream['round'] == 1 and not stream['pending']
                       for stream in actual['sampler']['ranks'])
        full = torch.load(candidate(complete), weights_only=False)
        retry = torch.load(candidate(resumed), weights_only=False)
        assert full['state'] == retry['state'] and full['sampler'] == retry['sampler']
        for key, value in full['model'].items():
            torch.testing.assert_close(value, retry['model'][key], rtol=0, atol=0)
        for rank in (0, 1):
            assert torch.equal(full['rng'][rank]['torch'], retry['rng'][rank]['torch'])
            assert set(full['rng'][rank]['buffers']) == set(retry['rng'][rank]['buffers'])
        second = torch.load(candidate(complete, 2), weights_only=False)
        for key, value in full['model'].items():
            if key.startswith(('humansl.', 'projection.', 'rank_heads.', 'value_head.')):
                torch.testing.assert_close(value, second['model'][key], rtol=0, atol=0)
        assert any(second['model'][f'residual_heads.{side}.weight'].abs().sum() > 0 for side in (0, 1))
        single_log = [json.loads(line) for line in candidate(single).with_name('train.jsonl').read_text().splitlines()]
        ddp_log = [json.loads(line) for line in candidate(complete).with_name('train.jsonl').read_text().splitlines()]
        assert len(single_log) == len(ddp_log) == 2
        for expected, actual in zip(single_log, ddp_log):
            assert actual['total'] == pytest.approx(expected['total'], rel=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
