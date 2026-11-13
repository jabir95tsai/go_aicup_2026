"""Configuration for the second SE experiment; steps count new AdamW updates."""
import json
import math
from pathlib import Path

from go_aicup.paths import ROOT
from go_aicup.tasks.strength.reproduce_paper.config import validate_config as validate_base

DEFAULT_CONFIG = ROOT / 'configs/strength/se_seed2026_adamw70k.json'


def validate_config(config):
    validate_base(config)
    t = config['training']
    for key in ('updates', 'groups_per_update', 'microbatch_groups', 'checkpoint_every', 'lr_drop_after_updates'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (t['optimizer'] != 'adamw' or t['sampling'] != 'shuffle_without_replacement'
            or not 0 < t['lr_drop_after_updates'] < t['updates']):
        raise ValueError('AdamW, exhaustive sampling and two LR stages are required')
    for key in ('lr', 'lr_after_drop', 'eps', 'weight_decay'):
        if not math.isfinite(t[key]) or t[key] < 0 or (key != 'weight_decay' and t[key] == 0):
            raise ValueError(f'invalid {key}')
    if t['lr_after_drop'] > t['lr'] or len(t['betas']) != 2 or any(not 0 <= b < 1 for b in t['betas']):
        raise ValueError('invalid AdamW betas or learning rates')
    if type(t['workers']) is not int or t['workers'] < 0:
        raise ValueError('workers must be a nonnegative integer')
    if config['precision'] not in ('fp32', 'bf16') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('unsupported precision/device')
    if config['device'] == 'cpu' and config['precision'] != 'fp32':
        raise ValueError('CPU checks require fp32')
    if config['loss_weights'] != dict(bt=1.0, policy=1.0, value=1.0):
        raise ValueError('equal loss weights are required')
    source = config['source']
    if (type(source['optimizer_steps']) is not int or source['optimizer_steps'] < 1
            or len(source['sha256']) != 64 or any(c not in '0123456789abcdef' for c in source['sha256'])):
        raise ValueError('invalid source step/checksum')
    output = Path(config['paths']['output_dir']).resolve()
    parent = Path(config['paths']['source_checkpoint']).resolve().parent
    if output == parent or parent in output.parents or output in parent.parents:
        raise ValueError('source and output directories must be separate')
    if (type(config['evaluation']['per_gpu_batch_size']) is not int
            or config['evaluation']['per_gpu_batch_size'] < 1):
        raise ValueError('evaluation batch must be positive')


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    for name, value in config['paths'].items():
        target = Path(value)
        config['paths'][name] = str((target if target.is_absolute() else ROOT / target).resolve())
    validate_config(config)
    return config
