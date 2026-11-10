"""Four linear B2 heads fitted only on the prepared same-color OOF questions.

Arms (all StandardScaler -> multinomial LR per k, C=1, score_rank decoding):
  new   : new-model 20 dims (prob or logp per branch) + 2 P2 flags = 22 dims
  stack : controlled H(30) + C(20) + new-model 20 dims + 2 P2 flags = 72 dims

Upstream features for fit pools are player-OOF (each game from the fold model that never saw
its player). M, U and test games use the average of the four fold models' per-game features;
there is no separately trained all-player upstream model.
"""
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
from .features import (BUCKETS, COLUMNS, FIT_POOLS, LOGP_COLUMNS, feature_path,
                       p2_values, read_features, validate_features, validate_oof)

FOLDS = (1, 2, 3, 4)


def arm_options(config, arm=None, aggregation=None):
    """Resolve the arm, its aggregation and its fitting pool from the config defaults."""
    b2 = config.get('b2', {})
    arm = arm or 'new'
    aggregation = aggregation or b2.get('aggregation', 'prob')
    if arm not in ('new', 'stack') or aggregation not in ('prob', 'logp'):
        raise ValueError('arm must be new/stack and aggregation prob/logp')
    pool = b2.get('fit_pool', 'fit') if arm == 'new' else b2.get('stack_fit_pool', 'fit')
    if pool not in FIT_POOLS:
        raise ValueError('B2 fitting pool must be fit or fit_all')
    return arm, aggregation, pool


def design(games, frame, aggregation='prob', baseline=None):
    """Rows follow frame.game_index; stack prepends the controlled H/C likelihood columns."""
    values = p2_values(games, frame, aggregation)
    if baseline is None:
        return values
    from .baseline import COLUMNS as HC_COLUMNS
    hc = baseline.set_index('game_index').reindex(frame.game_index)
    if hc[HC_COLUMNS].isna().any().any():
        raise ValueError('stack arm requires controlled H/C features for every new-model game')
    return np.column_stack([hc[HC_COLUMNS].to_numpy(np.float64), values])


def fit_questions(games, questions, frame, aggregation='prob', baseline=None):
    """Do not call RankModel.fit: its old B2 sampler ignores fixed questions."""
    check_questions(questions, games)
    values = question_mean(design(games, frame, aggregation, baseline), frame.game_index, questions)
    model = RankModel('B2'); model.per_k = {}
    for k in BUCKETS:
        mask = questions.k.eq(k).to_numpy()
        labels = questions.loc[mask, 'rank_id'].to_numpy()
        if set(labels) != set(range(10)):
            raise ValueError(f'k={k}: all ten B2 classes are required')
        fit = GameLR(values[mask], labels)
        model.per_k[k] = (fit.scaler, fit.model)
    return model


def _fold_frames(config, stage, candidate, pool, fold, selected, prepared):
    path = feature_path(config, 'dev', stage, candidate, pool, fold)
    paths = [path] if path.exists() else sorted(path.parent.glob(f'{path.stem}.rank*.parquet'))
    if not paths:
        raise ValueError(f'missing extracted features: {path}')
    frames, sources = [], []
    checkpoint = Path(config['run_dir']) / 'dev' / f'fold_{fold}' / f'stage{stage}' / f'{candidate}.pt'
    for shard in paths:
        frame, identity = read_features(shard)
        if identity['checkpoint'] != file_hash(checkpoint):
            raise ValueError('feature checkpoint no longer matches its declared candidate')
        if identity.get('prepared') != prepared or int(identity['upstream']['stage']) != stage:
            raise ValueError('feature prepared dataset or stage mismatch')
        rows = selected[selected.game_index.isin(frame.game_index)]
        if len(rows) != len(frame):
            raise ValueError('feature games do not belong to the prepared question pool')
        validate_oof(rows, identity['upstream'], phase='dev', fold=fold)
        if pool in FIT_POOLS and not rows.oof_fold.eq(fold).all():
            raise ValueError('OOF features are assigned to the wrong fold')
        if identity['pool'] != pool or identity['candidate'] != candidate:
            raise ValueError('feature role/candidate mismatch')
        frames.append(frame); sources.append(dict(path=str(shard), sha256=file_hash(shard), identity=identity))
    return pd.concat(frames, ignore_index=True), sources


def average_frames(frames):
    """Per-game mean of the fold models' branch features; counts must agree across folds."""
    columns = [c for c in COLUMNS + LOGP_COLUMNS if all(c in f for f in frames)]
    indexed = [f.set_index('game_index').sort_index() for f in frames]
    first = indexed[0]
    for other in indexed[1:]:
        if not first.index.equals(other.index) or not first[['own_count', 'opp_count']].equals(other[['own_count', 'opp_count']]):
            raise ValueError('fold-average requires identical games and position counts')
    mean = sum(f[columns].to_numpy(np.float64) for f in indexed) / len(indexed)
    result = pd.DataFrame(mean, columns=columns, index=first.index)
    result.insert(0, 'opp_count', first.opp_count)
    result.insert(0, 'own_count', first.own_count)
    return validate_features(result.reset_index()[['game_index', 'own_count', 'opp_count', *columns]])


def read_pool(config, stage, candidate, pool, phase='dev', inference_fold='avg', questions_name=None, trim=True):
    """Fit pools join the four OOF folds; inference pools average folds 1..4 (or use one fold).

    trim=True keeps only games referenced by the pool's questions; mu_all has no question table
    of its own and is always returned whole.
    """
    from .data import games, questions, load_prepared, pool_game_ids
    if phase != 'dev':
        raise ValueError('features only exist for development fold models')
    prepared = object_hash(load_prepared(config))
    canonical = games(config)
    ids = pool_game_ids(config, pool)
    q = None if pool == 'mu_all' and questions_name is None else questions(config, questions_name or pool)
    needed = ids if q is None else set(int(i) for row in q.games for i in row)
    if not needed <= ids:
        raise ValueError('question games fall outside the extracted pool')
    selected = canonical[canonical.game_index.isin(ids)]
    if pool in FIT_POOLS:
        parts = [_fold_frames(config, stage, candidate, pool, fold, selected, prepared) for fold in FOLDS]
        frame = pd.concat([p[0] for p in parts], ignore_index=True)
        sources = [s for p in parts for s in p[1]]
    elif inference_fold == 'avg':
        parts = [_fold_frames(config, stage, candidate, pool, fold, selected, prepared) for fold in FOLDS]
        frame = average_frames([p[0] for p in parts])
        sources = [s for p in parts for s in p[1]]
    else:
        frame, sources = _fold_frames(config, stage, candidate, pool, inference_fold, selected, prepared)
    if not frame.game_index.is_unique or set(frame.game_index) != ids:
        raise ValueError('missing or duplicate pool game features')
    if trim:
        frame = frame[frame.game_index.isin(needed)].reset_index(drop=True)
    return canonical, q, frame, sources


def baseline_frame(config, pool, questions_name=None):
    from . import baseline
    _, _, frame, sources = baseline.read_pool(config, pool, questions_name=questions_name)
    return frame, sources


def model_path(config, stage, candidate, phase='dev', arm=None, aggregation=None):
    arm, aggregation, pool = arm_options(config, arm, aggregation)
    return Path(config['run_dir']) / 'b2' / phase / f'stage{stage}' / candidate / f'{arm}_{aggregation}_{pool}' / 'model.pkl'


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


def release_pool(config, stage, candidate, fit_pool):
    """Dev OOF features of the fit pool plus fold-averaged M+U features (both unseen by upstream)."""
    from .data import questions
    q = questions(config, 'release_fit_all' if fit_pool == 'fit_all' else 'release_fit')
    canonical, _, fit_frame, fit_sources = read_pool(config, stage, candidate, fit_pool, trim=False)
    _, _, mu_frame, mu_sources = read_pool(config, stage, candidate, 'mu_all')
    frame = pd.concat([fit_frame, mu_frame], ignore_index=True)
    needed = set(int(i) for row in q.games for i in row)
    if not frame.game_index.is_unique or not needed <= set(frame.game_index):
        raise ValueError('release questions reference games without unique features')
    return canonical, q, frame[frame.game_index.isin(needed)].reset_index(drop=True), fit_sources + mu_sources


def fit(config, stage, candidate='best', phase='dev', arm=None, aggregation=None):
    from .data import load_prepared
    started, cpu_started = time.perf_counter(), time.process_time()
    arm, aggregation, fit_pool = arm_options(config, arm, aggregation)
    manifest = load_prepared(config)
    if phase == 'release':
        from .evaluate import load_lock
        load_lock(config)
        games, q, frame, sources = release_pool(config, stage, candidate, fit_pool)
        baseline = None
        if arm == 'stack':
            from . import baseline as controlled
            parts = [controlled.read_pool(config, pool, trim=False)[2] for pool in (fit_pool, 'mu_all')]
            baseline = pd.concat(parts, ignore_index=True)
    elif phase == 'dev':
        games, q, frame, sources = read_pool(config, stage, candidate, fit_pool)
        if not games.set_index('game_index').loc[frame.game_index, 'role'].eq('D').all():
            raise ValueError('development B2 can only fit D players')
        baseline = baseline_frame(config, fit_pool)[0] if arm == 'stack' else None
    else:
        raise ValueError('phase must be dev or release')
    dimensions = 22 + (50 if arm == 'stack' else 0)
    identity = dict(format='humansl_residual_b2_v2', phase=phase, stage=stage, candidate=candidate,
                    arm=arm, aggregation=aggregation, fit_pool=fit_pool, upstream='four-fold-oof/fold-average',
                    prepared=object_hash(manifest), config=object_hash(config), sources=sources,
                    questions=object_hash(q.to_json(orient='records')), training_players=sorted(q.player_id.astype(str).unique()),
                    dimensions=dimensions, regularization_C=1, max_iter=5000, question_seed=2027)
    path = model_path(config, stage, candidate, phase, arm, aggregation)
    reused = path.exists()
    if reused:
        _, existing = load_model(path)
        if existing != identity:
            raise ValueError('B2 fit identity changed; use a new run directory')
    else:
        model = fit_questions(games, q, frame, aggregation, baseline)
        save_model(path, model, identity)
    elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
    allocation = os.environ.get('SLURM_GPUS_ON_NODE', '0')
    allocated_gpus = int(allocation) if allocation.isdigit() else None
    write_json(path.with_suffix('.reuse.metrics.json' if reused else '.metrics.json'),
               dict(reused=reused, elapsed_seconds=elapsed, cpu_seconds=cpu, cpu_hours=cpu / 3600, compute_gpu_hours=0, allocated_gpus=allocated_gpus,
                    gpu_hours=None if allocated_gpus is None else allocated_gpus * elapsed / 3600,
                    games=len(frame), dimensions=dimensions, questions=len(q),
                    questions_per_k={str(k): int(q.k.eq(k).sum()) for k in BUCKETS},
                    questions_per_second=None if reused else len(q) / elapsed,
                    scope='source validation, feature read, scaler/LR fitting and serialization'))
    return path


def probabilities(model, games, q, frame, aggregation='prob', baseline=None):
    return model.probabilities(design(games, frame, aggregation, baseline), frame.game_index.to_numpy(), q)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--stage', type=int, choices=(1, 2), required=True)
    p.add_argument('--candidate', choices=('best', 'final'), default='best')
    p.add_argument('--phase', choices=('dev', 'release'), default='dev')
    p.add_argument('--arm', choices=('new', 'stack'), default=None)
    p.add_argument('--aggregation', choices=('prob', 'logp'), default=None)
    a = p.parse_args()
    from .config import read_config
    print(fit(read_config(a.config), a.stage, a.candidate, a.phase, a.arm, a.aggregation))


if __name__ == '__main__':
    main()
