"""SE3 configuration, separate from the seven-position prepared index identity."""
import copy
import json
import math
from pathlib import Path
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.reproduce_paper.config import validate_config as validate_indices

DEFAULT_CONFIG = ROOT / 'configs/strength/se3_seed2026.json'


def validate_config(config):
    legacy = copy.deepcopy(config)
    legacy['positions']['positions_per_rank'] = 7
    validate_indices(legacy)
    if type(config['smoke']) is not bool:
        raise ValueError('smoke must be a boolean')
    if config['positions']['min_ply'] != 30:
        raise ValueError('SE3 requires min_ply=30')
    if type(config['positions']['positions_per_rank']) is not int or config['positions']['positions_per_rank'] != 30:
        raise ValueError('SE3 requires thirty positions per game')
    t, m = config['training'], config['monitoring']
    for key in ('updates', 'groups_per_update', 'microbatch_groups', 'checkpoint_every', 'lr_drop_after_updates'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if t['optimizer'] != 'adamw' or t['sampling'] != 'shuffle_without_replacement':
        raise ValueError('AdamW and exhaustive sampling are required')
    if t['groups_per_update'] % t['microbatch_groups']:
        raise ValueError('microbatch must divide groups_per_update')
    for key in ('lr', 'lr_after_drop', 'eps', 'weight_decay'):
        if not math.isfinite(t[key]) or t[key] < 0 or (key != 'weight_decay' and t[key] == 0):
            raise ValueError(f'invalid {key}')
    if len(t['betas']) != 2 or any(not math.isfinite(b) or not 0 <= b < 1 for b in t['betas']):
        raise ValueError('invalid AdamW betas')
    if type(t['workers']) is not int or t['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    if config['device'] not in ('cpu', 'cuda') or config['precision'] not in ('fp32', 'bf16'):
        raise ValueError('unsupported device/precision')
    if config['device'] == 'cpu' and config['precision'] != 'fp32':
        raise ValueError('CPU checks require fp32')
    for key in ('every', 'questions', 'patience', 'seed'):
        if type(m[key]) is not int or m[key] < 1:
            raise ValueError(f'monitoring {key} must be positive')
    if m['questions'] > 5000 or m['questions'] % 40 or m['patience'] != 3:
        raise ValueError('monitoring requires <=5000 questions in balanced cells and patience=3')
    if config['loss_weights'] != dict(strength=1., policy=1., value=1.):
        raise ValueError('equal loss weights are required')
    if type(config['evaluation']['per_gpu_batch_size']) is not int or config['evaluation']['per_gpu_batch_size'] < 1:
        raise ValueError('evaluation batch must be positive')
    if config['device'] == 'cuda':
        if config['smoke'] and t['updates'] != 100:
            raise ValueError('CUDA smoke requires exactly 100 updates')
        if (config['model']['channels'] != 256 or config['model']['blocks'] != 18
                or t['groups_per_update'] != 128 or t['microbatch_groups'] != 16
                or not t['sync_batch_norm'] or not t['channels_last'] or t['cudnn_benchmark']
                or config['precision'] != 'bf16'):
            raise ValueError('CUDA requires the confirmed 18-block, 128-group, full-batch BF16/SyncBN settings')
    output = Path(config['paths']['output_dir']).resolve()
    for key in ('dataset_dir', 'cache_dir', 'label_dir'):
        source = Path(config['paths'][key]).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError('output and input directories must be separate')


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    for name, value in config['paths'].items():
        target = Path(value)
        config['paths'][name] = str((target if target.is_absolute() else ROOT / target).resolve())
    validate_config(config)
    return config
