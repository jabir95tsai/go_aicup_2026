"""Small real optimizer runs verify resume/publication rather than model quality."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from go_aicup.tasks.strength.humansl_residual_b2 import train as training
from go_aicup.tasks.strength.humansl_residual_b2.config import read_config, runtime_config
from go_aicup.tasks.strength.humansl_residual_b2.model import ResidualClassifier


class Human(nn.Module):
    config = {}
    def forward_features(self, spatial, global_inputs, metadata):
        return spatial[:, :1].expand(-1, 384, -1, -1)


class SyntheticDataset:
    def __init__(self, config, games, model_config, **kwargs):
        self.games, self.seed = games, config['seed']
    def __len__(self):
        return len(self.games)
    def __getitem__(self, task):
        row, step = task
        game = self.games.iloc[row]
        rng = np.random.default_rng([self.seed, row, step])
        return dict(spatial=rng.integers(0, 2, size=(30, 1, 19, 19), dtype=np.uint8),
                    global_inputs=np.zeros((30, 1), np.float32), metadata=np.zeros((30, 1), np.float32),
                    actions=np.arange(30, dtype=np.int64), sides=np.repeat([0, 1], 15),
                    labels=np.full(30, game.rank_id, np.int64), values=np.zeros(30, np.float32),
                    ply=np.arange(30, 60, dtype=np.int64))


@pytest.fixture
def run(tmp_path, monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2 import data, dataset
    torch.set_num_threads(1)
    config = read_config()
    config.update(run_dir=str(tmp_path / 'run'), device='cpu', precision='fp32', smoke=False)
    source = tmp_path / 'human.ckpt'; source.write_bytes(b'fixture source')
    config['model']['human_checkpoint'] = str(source)
    config['training'].update(groups_per_update=2, workers=0, channels_last=False,
                              sync_batch_norm=False, checkpoint_every=1)
    config['stage1'].update(updates=3)
    rows = pd.DataFrame([dict(game_index=r * 4 + g, rank_id=r, player_id=f'{r}-{g}',
                              role='D', oof_fold=g % 4 + 1) for r in range(10) for g in range(4)])
    monkeypatch.setattr(data, 'games', lambda *args, **kwargs: rows)
    monkeypatch.setattr(data, 'load_prepared', lambda *args, **kwargs: {'dataset_id': 'synthetic'})
    monkeypatch.setattr(dataset, 'SelectedDataset', SyntheticDataset)
    monkeypatch.setattr(training, 'environment', lambda *args: {})
    monkeypatch.setattr(training, 'code_identity', lambda: {'synthetic': 'fixed'})
    monkeypatch.setattr(training, 'load_model', lambda *args, **kwargs: ResidualClassifier(Human(), channels=2, blocks=1))
    monkeypatch.setattr(training, 'monitor', lambda *args: {'fused': {'equal_k': .25}})
    return config


def payload(config, filename='latest.pt'):
    return torch.load(Path(config['run_dir']) / 'dev/fold_all/stage1' / filename,
                      map_location='cpu', weights_only=False)


def test_resume_matches_uninterrupted_optimizer_and_sampler(run):
    complete = copy.deepcopy(run); complete['run_dir'] += '_complete'
    training.train(complete, 1)
    first = training.train(run, 1, stop_after=1)
    assert first['status'] == 'paused'
    assert payload(run)['sampler']['completed_step'] == 1
    training.train(run, 1)
    left, right = payload(complete), payload(run)
    assert left['state'] == right['state']
    assert left['sampler'] == right['sampler']
    for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], rtol=0, atol=0)
    assert len((Path(run['run_dir']) / 'dev/fold_all/stage1/train.jsonl').read_text().splitlines()) == 3


def test_pending_monitor_retry_does_not_repeat_update(run, monkeypatch):
    original = training.monitor
    monkeypatch.setattr(training, 'monitor', lambda *args: (_ for _ in ()).throw(RuntimeError('monitor failed')))
    with pytest.raises(RuntimeError, match='monitor failed'):
        training.train(run, 1)
    saved = payload(run)
    assert saved['state']['optimizer_steps'] == 3 and saved['state']['pending_monitor']
    monkeypatch.setattr(training, 'monitor', original)
    result = training.train(run, 1)
    assert result['status'] == 'completed'
    assert payload(run)['state']['best_step'] == 3
    for key, value in saved['model'].items():
        torch.testing.assert_close(value, payload(run)['model'][key], rtol=0, atol=0)


def test_spawn_prefetch_resume_keeps_consumed_updates(run):
    run['training']['workers'] = 2
    run['stage1']['updates'] = 2
    complete = copy.deepcopy(run); complete['run_dir'] += '_workers_complete'
    training.train(complete, 1)
    training.train(run, 1, stop_after=1)
    training.train(run, 1)
    left, right = payload(complete), payload(run)
    assert left['sampler'] == right['sampler']
    for key, value in left['model'].items():
        torch.testing.assert_close(value, right['model'][key], rtol=0, atol=0)


def test_final_publication_retry_and_changed_config_rejected(run, monkeypatch):
    original = training.atomic_payload
    def fail_final(path, value):
        if path.name == 'final.pt':
            raise OSError('publication failed')
        return original(path, value)
    monkeypatch.setattr(training, 'atomic_payload', fail_final)
    with pytest.raises(OSError, match='publication failed'):
        training.train(run, 1)
    monkeypatch.setattr(training, 'atomic_payload', original)
    assert training.train(run, 1)['status'] == 'completed'
    assert payload(run, 'final.pt')['state']['optimizer_steps'] == 3
    changed = copy.deepcopy(run); changed['stage1']['lr'] *= 2
    with pytest.raises(ValueError, match='resume config'):
        training.train(changed, 1)


def test_strict_improvement_ties_and_lr_boundaries():
    state = dict(best_score=None, best_step=None, non_improvements=0)
    state, accepted = training.validation_transition(state, .5, 5)
    assert accepted and state['best_step'] == 5
    for step in (10, 15, 20):
        state, accepted = training.validation_transition(state, .5, step)
        assert not accepted and state['best_step'] == 5
    assert state['early_stopped']
    with pytest.raises(ValueError, match='finite'):
        training.validation_transition(state, float('nan'), 25)
    options = dict(lr=.1, lr_schedule=[[10, .01], [20, .001]])
    assert [training.learning_rate(options, s) for s in (9, 10, 19, 20)] == [.1, .01, .01, .001]


def test_config_and_complete_group_divisibility():
    config = read_config()
    assert runtime_config(config, 8)['training']['global_batch_size'] == 128
    config['training']['groups_per_update'] = 127
    with pytest.raises(ValueError, match='divide'):
        runtime_config(config, 8)
