"""Experiment configuration; long-run budgets must be supplied explicitly."""
import copy
import json
import math
from pathlib import Path

from go_aicup.paths import ROOT

DEFAULT_CONFIG = ROOT / 'configs/strength/humansl_residual_smoke100.json'


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    for key in ('run_dir',):
        config[key] = str((ROOT / config[key]).resolve())
    for section in ('data', 'model', 'teacher'):
        for key, value in config[section].items():
            if key.endswith('_paths'):
                config[section][key] = [str((ROOT / item).resolve()) for item in value]
            if isinstance(value, str) and (key.endswith(('_dir', '_path', '_checkpoint')) or key in ('engine', 'analysis_config')):
                config[section][key] = str((ROOT / value).resolve())
    validate_config(config)
    return config


def validate_config(config):
    if config['seed'] != 2026 or config['split_seed'] != 2027:
        raise ValueError('this experiment fixes model seed=2026 and split/question seed=2027')
    if config['device'] not in ('cpu', 'cuda') or config['precision'] not in ('fp32', 'bf16'):
        raise ValueError('expected cpu/cuda and fp32/bf16')
    options = config['training']
    for key in ('groups_per_update', 'microbatch_groups', 'checkpoint_every', 'positions_per_side'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if options['positions_per_side'] != 15 or options['prefetch_factor'] != 1:
        raise ValueError('fifteen positions per side and bounded prefetch_factor=1 are required')
    if type(options['workers']) is not int or options['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    if options['optimizer'] not in ('adamw', 'sgd'):
        raise ValueError('optimizer must be adamw or sgd')
    if config['model']['channels'] != 256 or config['model']['blocks'] != 10:
        raise ValueError('production architecture requires 256 channels and ten residual blocks')
    for stage in ('stage1', 'stage2'):
        budget = config[stage]
        if budget['updates'] is not None and (type(budget['updates']) is not int or budget['updates'] < 1):
            raise ValueError('updates must be a positive integer or unset')
        for key in ('lr',):
            value = budget[key]
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                raise ValueError('learning rate must be finite and positive or unset')
        drops = budget['lr_schedule']
        if any(type(row[0]) is not int or row[0] < 1 or not math.isfinite(row[1]) or row[1] <= 0 for row in drops):
            raise ValueError('schedule entries are [completed_update, positive_lr]')
        if [row[0] for row in drops] != sorted(set(row[0] for row in drops)):
            raise ValueError('schedule boundaries must increase')
    if config['monitoring']['every'] != 5000 or config['monitoring']['patience'] != 3:
        raise ValueError('monitor every 5000 updates with patience=3')
    if config.get('smoke') and any(config[s]['updates'] != 100 for s in ('stage1', 'stage2')):
        raise ValueError('resource smoke runs require 100 optimizer updates per stage')
    return config


def runtime_config(config, world):
    """Adapt existing distributed helpers without changing the saved experiment."""
    result = copy.deepcopy(config)
    groups = result['training']['groups_per_update']
    micro = result['training']['microbatch_groups']
    if groups % world or (groups // world) % micro:
        raise ValueError('global groups must divide ranks and complete microbatches')
    result['training']['global_batch_size'] = groups
    result['evaluation']['global_batch_size'] = max(world, world * result['evaluation']['batch_size'])
    return result
