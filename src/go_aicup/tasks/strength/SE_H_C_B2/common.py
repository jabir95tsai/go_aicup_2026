"""Artifact boundaries for this experiment; no import-time I/O."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import sklearn

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features.humansl import PROFILES
from go_aicup.paths import ROOT

BUCKETS = (5, 10, 15, 20)
H = [f'{side}_{profile}' for side in ('own', 'opp') for profile in PROFILES]
C = [f'c_{side}_r{i}' for side in ('own', 'opp') for i in range(10)]
S = [f'se3_{i}' for i in range(10)]
A = [f'a024_{i}' for i in range(10)]
DEFAULT_CONFIG = ROOT / 'configs/strength/se3_hc_b2.json'


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    config['paths'] = {k: str((ROOT / v).resolve()) for k, v in config['paths'].items()}
    t = config['training']
    for key in ('epochs', 'batch_size', 'patience', 'threads'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f'invalid training.{key}')
    if not 0 < t['min_lr'] <= t['lr'] or t['weight_decay'] < 0:
        raise ValueError('invalid learning rate / weight decay')
    for key in ('synthetic_per_k', 'evaluation_per_cell', 'seed', 'bootstrap'):
        if type(config[key]) is not int or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'invalid {key}')
    if config['inference']['device'] not in ('cpu', 'cuda'):
        raise ValueError('invalid inference device')
    if config['inference']['batch_size'] < 1 or config['inference']['games_per_shard'] < 1:
        raise ValueError('invalid inference batch sizes')
    return config


def code_hashes():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def source_identity(config):
    paths = config['paths']
    names = ('se3_model', 'a_model', 'vectors', 'human_model', 'human_checkpoint', 'engine', 'analysis_config')
    models = {k: file_hash(Path(paths[k])) for k in names}
    for k, digest in config['expected_sha256'].items():
        if models[k] != digest:
            raise ValueError(f'{k}: source checksum mismatch')
    upstream = ('data/board_cache.py', 'tasks/strength/reproduce_paper/dataset.py',
                'tasks/strength/reproduce_paper/model.py', 'tasks/strength/reproduce_paper_3/model.py',
                'tasks/strength/reproduce_paper_3/evaluate.py', 'features/humansl.py',
                'features/humansl_torch.py', 'features/humansl_rankvec.py')
    return dict(models=models, upstream={name: file_hash(ROOT / 'src/go_aicup' / name) for name in upstream},
                dataset=file_hash(Path(paths['dataset']) / 'games.parquet'),
                prepared=file_hash(Path(paths['dataset']) / 'manifest.json'),
                cache=file_hash(Path(paths['cache']) / 'manifest.json'), code=code_hashes(),
                versions=dict(torch=str(torch.__version__), numpy=np.__version__, sklearn=sklearn.__version__))


@contextmanager
def writer_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer.lock'
    with lock.open('x'):
        pass
    try:
        yield
    finally:
        lock.unlink()


def atomic_table(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def publish_table(path, frame, identity):
    atomic_table(path, frame)
    write_json(Path(path).with_suffix('.json'), dict(identity=identity, sha256=file_hash(path)))


def load_table(path, identity):
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    if record != dict(identity=identity, sha256=file_hash(path)):
        raise ValueError(f'cache identity/checksum mismatch: {path}')
    return pd.read_parquet(path)


def checked_features(frame, columns):
    if list(frame.columns) != ['game_index', *columns] or not frame.game_index.is_unique:
        raise ValueError('feature columns/order or game IDs differ')
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite features')
    return frame.set_index('game_index')


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(model, before):
    if model.training or any(p.requires_grad for p in model.parameters()) or state_hash(model) != before:
        raise ValueError('frozen upstream model parameters/buffers changed')
