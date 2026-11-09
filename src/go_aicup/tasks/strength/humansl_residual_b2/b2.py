"""Four linear B2 heads fitted only on the prepared same-color OOF questions."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.humansl import GameLR, RankModel, question_mean
from go_aicup.tasks.strength.SE_H_C_B2.data import check_questions
from .features import BUCKETS, feature_path, p2_values, read_features, validate_oof


def fit_questions(games, questions, frame):
    """Do not call RankModel.fit: its old B2 sampler ignores fixed questions."""
    check_questions(questions, games)
    values = question_mean(p2_values(games, frame), frame.game_index, questions)
    model = RankModel('B2'); model.per_k = {}
    for k in BUCKETS:
        mask = questions.k.eq(k).to_numpy()
        labels = questions.loc[mask, 'rank_id'].to_numpy()
        if set(labels) != set(range(10)):
            raise ValueError(f'k={k}: all ten B2 classes are required')
        fit = GameLR(values[mask], labels)
        model.per_k[k] = (fit.scaler, fit.model)
    return model


def read_pool(config, stage, candidate, pool, phase='dev', inference_fold='all'):
    from .data import games, questions, load_prepared
    prepared = object_hash(load_prepared(config, release=phase == 'release'))
    canonical = games(config, phase=phase)
    name = ('fit' if phase == 'dev' else 'release_fit') if pool == 'fit' else pool
    q = questions(config, name, phase=phase)
    ids = np.unique(np.concatenate(q.games.to_numpy()))
    selected = canonical[canonical.game_index.isin(ids)]
    frames, sources = [], []
    for fold in range(1, 5) if pool == 'fit' else (inference_fold,):
        path = feature_path(config, phase, stage, candidate, pool, fold)
        paths = [path] if path.exists() else sorted(path.parent.glob(f'{path.stem}.rank*.parquet'))
        if not paths:
            raise ValueError(f'missing extracted features: {path}')
        for shard in paths:
            frame, identity = read_features(shard)
            checkpoint = Path(config['run_dir']) / phase / f'fold_{fold}' / f'stage{stage}' / f'{candidate}.pt'
            if identity['checkpoint'] != file_hash(checkpoint):
                raise ValueError('feature checkpoint no longer matches its declared candidate')
            if identity.get('prepared') != prepared or int(identity['upstream']['stage']) != stage:
                raise ValueError('feature prepared dataset or stage mismatch')
            rows = selected[selected.game_index.isin(frame.game_index)]
            if len(rows) != len(frame):
                raise ValueError('feature games do not belong to the prepared question pool')
            validate_oof(rows, identity['upstream'], phase=phase, fold=fold)
            if pool == 'fit' and not rows.oof_fold.eq(fold).all():
                raise ValueError('OOF features are assigned to the wrong fold')
            if identity['pool'] != pool or identity['candidate'] != candidate:
                raise ValueError('feature role/candidate mismatch')
            frames.append(frame); sources.append(dict(path=str(shard), sha256=file_hash(shard), identity=identity))
    frame = pd.concat(frames, ignore_index=True)
    if not frame.game_index.is_unique or set(frame.game_index) != set(ids):
        raise ValueError('missing or duplicate question game features')
    return canonical, q, frame, sources


def model_path(config, stage, candidate, phase='dev'):
    return Path(config['run_dir']) / 'b2' / phase / f'stage{stage}' / candidate / 'model.pkl'


def save_model(path, model, identity):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    model.save(temporary, identity=identity)
    _, info = RankModel.load(temporary)
    if info != {'identity': identity}:
        raise ValueError('B2 publication failed reload validation')
    temporary.replace(path)
    write_json(path.with_suffix('.json'), dict(identity=identity, sha256=file_hash(path)))


def load_model(path):
    path = Path(path)
    marker = json.loads(path.with_suffix('.json').read_text())
    if marker['sha256'] != file_hash(path):
        raise ValueError('B2 model checksum mismatch')
    model, info = RankModel.load(path)
    if info['identity'] != marker['identity']:
        raise ValueError('B2 model identity mismatch')
    return model, info['identity']


def fit(config, stage, candidate='best', phase='dev'):
    from .data import load_prepared
    started, cpu_started = time.perf_counter(), time.process_time()
    manifest = load_prepared(config, release=phase == 'release')
    if phase == 'release':
        from .evaluate import load_lock
        load_lock(config)
    games, q, frame, sources = read_pool(config, stage, candidate, 'fit', phase)
    if phase == 'dev' and not games.set_index('game_index').loc[frame.game_index, 'role'].eq('D').all():
        raise ValueError('development B2 can only fit D players')
    identity = dict(format='humansl_residual_b2_v1', phase=phase, stage=stage, candidate=candidate,
                    prepared=object_hash(manifest), config=object_hash(config), sources=sources,
                    questions=object_hash(q.to_json(orient='records')), training_players=sorted(q.player_id.astype(str).unique()),
                    dimensions=22, regularization_C=1, max_iter=5000, question_seed=2027)
    path = model_path(config, stage, candidate, phase)
    reused = path.exists()
    if reused:
        _, existing = load_model(path)
        if existing != identity:
            raise ValueError('B2 fit identity changed; use a new run directory')
    else:
        model = fit_questions(games, q, frame)
        save_model(path, model, identity)
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    allocation = os.environ.get('SLURM_GPUS_ON_NODE', '0')
    allocated_gpus = int(allocation) if allocation.isdigit() else None
    write_json(path.with_suffix('.reuse.metrics.json' if reused else '.metrics.json'),
               dict(reused=reused, elapsed_seconds=elapsed, cpu_seconds=cpu, cpu_hours=cpu / 3600, compute_gpu_hours=0, allocated_gpus=allocated_gpus,
                    gpu_hours=None if allocated_gpus is None else allocated_gpus * elapsed / 3600,
                    games=len(frame), dimensions=22, questions=len(q),
                    questions_per_k={str(k): int(q.k.eq(k).sum()) for k in BUCKETS},
                    questions_per_second=None if reused else len(q) / elapsed,
                    scope='source validation, feature read, scaler/LR fitting and serialization'))
    return path


def probabilities(model, games, q, frame):
    return model.probabilities(p2_values(games, frame), frame.game_index.to_numpy(), q)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--stage', type=int, choices=(1, 2), required=True)
    p.add_argument('--candidate', choices=('best', 'final'), default='best')
    p.add_argument('--phase', choices=('dev', 'release'), default='dev')
    a = p.parse_args()
    from .config import read_config
    print(fit(read_config(a.config), a.stage, a.candidate, a.phase))


if __name__ == '__main__':
    main()
