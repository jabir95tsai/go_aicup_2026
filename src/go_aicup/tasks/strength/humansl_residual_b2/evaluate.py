"""Complete-pipeline M selection, player-paired gates and locked U evaluation (fold-average upstream)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.humansl import question_mean
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table
from go_aicup.tasks.strength.SE_H_C_B2.evaluate import official_rows, paired_bootstrap, report
from .b2 import FOLDS, arm_options, baseline_frame, load_model, model_path, probabilities, read_pool
from .features import BUCKETS, COLUMNS


def monitor_report(q, frame):
    x = question_mean(frame[COLUMNS].to_numpy(), frame.game_index, q)
    result = {}
    for name, p in [('own', x[:, :10]), ('opp', x[:, 10:]), ('fused', (x[:, :10] + x[:, 10:]) / 2)]:
        result[name] = report(q, p)
        result[name]['cross_entropy'] = float(-np.log(np.maximum(p[np.arange(len(q)), q.rank_id], 1e-30)).mean())
    return result


def compare(q, reference, candidate, *, repeats=2000, seed=2027):
    baseline, new = report(q, reference), report(q, candidate)
    gains = {str(k): new['by_k'][str(k)]['score'] - baseline['by_k'][str(k)]['score'] for k in BUCKETS}
    equal = new['equal_k'] - baseline['equal_k']
    return dict(reference=baseline, candidate=new, by_k_gain=gains, equal_k_gain=equal,
                passed=bool(equal >= .005 - 1e-12 and min(gains.values()) >= -.005 - 1e-12),
                paired_bootstrap=paired_bootstrap(q, official_rows(score_rank(reference), q.rank_id),
                                                 official_rows(score_rank(candidate), q.rank_id), repeats, seed))


def prediction_path(config, stage, candidate, pool, arm=None, aggregation=None, fit_pool=None):
    arm, aggregation, fit_pool = arm_options(config, arm, aggregation, fit_pool)
    return (Path(config['run_dir']) / 'evaluation' / f'stage{stage}' / candidate
            / f'{arm}_{aggregation}_{fit_pool}' / f'{pool}.parquet')


def evaluate(config, stage, candidate='best', pool='monitor', arm=None, aggregation=None, fit_pool=None):
    """Score one fitted B2 arm on M (or locked U) with fold-averaged upstream features."""
    if pool not in ('monitor', 'U'):
        raise ValueError('development evaluation is restricted to M or locked U')
    arm, aggregation, fit_pool = arm_options(config, arm, aggregation, fit_pool)
    bundle = model_path(config, stage, candidate, 'dev', arm, aggregation, fit_pool)
    model, identity = load_model(bundle)
    if identity['config'] != object_hash(config):
        raise ValueError('development config changed after B2 fit')
    if pool == 'U':
        locked = load_lock(config)
        if not any(a['model'] == str(bundle) and a['sha256'] == file_hash(bundle) for a in locked['arms'].values()):
            raise ValueError('U evaluation arm was not predeclared in lock')
    games, q, frame, sources = read_pool(config, stage, candidate, pool)
    if set(q.player_id.astype(str)) & set(identity['training_players']):
        raise ValueError('B2 fitting/evaluation players overlap')
    baseline = baseline_frame(config, pool)[0] if arm == 'stack' else None
    proba = probabilities(model, games, q, frame, aggregation, baseline)
    result = report(q, proba)
    result.update(pool=pool, arm=arm, aggregation=aggregation, fit_pool=fit_pool, upstream='fold-average', model=file_hash(bundle),
                  sources=sources,
                  interpretation='development selection' if pool == 'monitor' else 'locked one-time U evaluation')
    path = prediction_path(config, stage, candidate, pool, arm, aggregation, fit_pool)
    output = q.drop(columns='games').assign(rank_prediction=score_rank(proba))
    for i, rank in enumerate(RANKS):
        output[f'p_{rank}'] = proba[:, i]
    if path.exists():
        old = json.loads(path.with_suffix('.json').read_text())
        if old != dict(report=result, sha256=file_hash(path)):
            raise ValueError('evaluation identity changed; refusing to replace an earlier U/M result')
        return result
    atomic_table(path, output)
    write_json(path.with_suffix('.json'), dict(report=result, sha256=file_hash(path)))
    return result


def read_predictions(path):
    path = Path(path); marker = json.loads(path.with_suffix('.json').read_text())
    if marker['sha256'] != file_hash(path):
        raise ValueError('prediction checksum mismatch')
    return pd.read_parquet(path)


def select(config, stage):
    """Best vs final upstream candidate, judged by the configured main arm (new) on M."""
    reports = {name: evaluate(config, stage, name) for name in ('best', 'final')}
    selected = 'final' if reports['final']['equal_k'] > reports['best']['equal_k'] else 'best'
    arm, aggregation, fit_pool = arm_options(config)
    result = dict(stage=stage, selected=selected, arm=arm, aggregation=aggregation, fit_pool=fit_pool,
                  reports=reports)
    write_json(Path(config['run_dir']) / 'selection' / f'stage{stage}.json', result)
    return result


def gate_paths(reference, candidate, *, repeats=2000):
    left, right = read_predictions(reference), read_predictions(candidate)
    if (left.question_id.tolist() != right.question_id.tolist()
            or not left[['player_id', 'rank_id', 'k']].equals(right[['player_id', 'rank_id', 'k']])):
        raise ValueError('paired gate requires the identical question table')
    columns = [f'p_{r}' for r in RANKS]
    return compare(left, left[columns].to_numpy(), right[columns].to_numpy(), repeats=repeats)


def scale_diagnostics(config, stage, candidate, arm=None, aggregation=None, fit_pool=None):
    """Same M and B2: each single fold model versus the four-fold average used for M/U/test."""
    arm, aggregation, fit_pool = arm_options(config, arm, aggregation, fit_pool)
    model, _ = load_model(model_path(config, stage, candidate, 'dev', arm, aggregation, fit_pool))
    baseline = baseline_frame(config, 'monitor')[0] if arm == 'stack' else None
    result = {}
    for fold in ('avg', 1, 2, 3, 4):
        games, q, frame, _ = read_pool(config, stage, candidate, 'monitor', inference_fold=fold)
        x = question_mean(frame[COLUMNS].to_numpy(), frame.game_index, q)
        summary = {'b2': report(q, probabilities(model, games, q, frame, aggregation, baseline))}
        for name, p in [('own', x[:, :10]), ('opp', x[:, 10:])]:
            logp = np.log(np.maximum(p, 1e-30))
            summary[name] = dict(entropy=float(-(p * logp).sum(1).mean()),
                                 nll=float(-logp[np.arange(len(q)), q.rank_id].mean()))
        result[str(fold)] = summary
    path = Path(config['run_dir']) / 'diagnostics' / f'stage{stage}_{candidate}_{arm}_{aggregation}_{fit_pool}.json'
    write_json(path, result)
    return result


def lock(config, arms):
    """Freeze predeclared B2 arms and their four fold checkpoints before reading U.

    Release no longer retrains upstream models, so no fixed step counts are recorded: the
    locked fold checkpoints themselves are the release upstream.
    """
    if config.get('smoke'):
        raise ValueError('smoke candidates cannot lock U evaluation or release settings')
    root = Path(config['run_dir'])
    frozen, selections, folds = {}, {}, {}
    import torch
    for name, spec in arms.items():
        if 'stage' not in spec:
            path = Path(spec['model'])
            frozen[name] = dict(model=str(path), sha256=file_hash(path))
            continue
        stage, candidate = int(spec['stage']), spec['candidate']
        arm, aggregation, fit_pool = arm_options(config, spec.get('arm'), spec.get('aggregation'), spec.get('fit_pool'))
        selection = root / 'selection' / f'stage{stage}.json'
        if json.loads(selection.read_text())['selected'] != candidate:
            raise ValueError('lock must use the selected M complete-B2 candidate')
        selections[str(stage)] = dict(path=str(selection), sha256=file_hash(selection))
        path = model_path(config, stage, candidate, 'dev', arm, aggregation, fit_pool)
        _, identity = load_model(path)
        if identity['config'] != object_hash(config):
            raise ValueError('arm config differs from current experiment')
        checkpoints = {}
        for fold in FOLDS:
            checkpoint = root / 'dev' / f'fold_{fold}' / f'stage{stage}' / f'{candidate}.pt'
            payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if (payload['identity']['phase'] != 'dev' or str(payload['identity']['fold']) != str(fold)
                    or payload['identity']['stage'] != stage):
                raise ValueError('lock requires the four selected development fold checkpoints')
            checkpoints[str(fold)] = dict(path=str(checkpoint), sha256=file_hash(checkpoint),
                                          stage1_source_sha256=payload['identity'].get('stage1_source_sha256'))
        if str(stage) in folds and folds[str(stage)] != checkpoints:
            raise ValueError('all arms of one stage must share the selected fold checkpoints')
        folds[str(stage)] = checkpoints
        frozen[name] = dict(model=str(path), sha256=file_hash(path), stage=stage, candidate=candidate,
                            arm=arm, aggregation=aggregation, fit_pool=fit_pool)
    if '2' in folds and '1' in folds:
        for fold in FOLDS:
            if folds['2'][str(fold)]['stage1_source_sha256'] != folds['1'][str(fold)]['sha256']:
                raise ValueError('stage-two arm does not inherit the selected stage-one candidate')
    from .data import load_prepared
    value = dict(config=object_hash(config), arms=frozen, fold_checkpoints=folds,
                 dataset_id=load_prepared(config)['dataset_id'], selections=selections)
    path = root / 'selection_lock.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('development settings are already locked')
    write_json(path, value)
    return value


def load_lock(config):
    value = json.loads((Path(config['run_dir']) / 'selection_lock.json').read_text())
    from .data import load_prepared
    if value['dataset_id'] != load_prepared(config)['dataset_id']:
        raise ValueError('prepared dataset differs from U/release lock')
    if value['config'] != object_hash(config):
        raise ValueError('config differs from U/release lock')
    for selection in value['selections'].values():
        if file_hash(Path(selection['path'])) != selection['sha256']:
            raise ValueError('development selection changed after locking')
    for arm in value['arms'].values():
        if file_hash(Path(arm['model'])) != arm['sha256']:
            raise ValueError('locked model changed')
    for checkpoints in value.get('fold_checkpoints', {}).values():
        for record in checkpoints.values():
            if file_hash(Path(record['path'])) != record['sha256']:
                raise ValueError('locked fold checkpoint changed')
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--stage', type=int, choices=(1, 2))
    p.add_argument('--candidate', choices=('best', 'final'), default='best')
    p.add_argument('--pool', choices=('monitor', 'U'), default='monitor')
    p.add_argument('--arm', choices=('new', 'stack')); p.add_argument('--aggregation', choices=('prob', 'logp'))
    p.add_argument('--fit-pool', choices=('fit', 'fit_all'))
    p.add_argument('--select', action='store_true'); p.add_argument('--diagnostics', action='store_true'); p.add_argument('--lock-arms')
    p.add_argument('--reference'); p.add_argument('--candidate-predictions'); p.add_argument('--output')
    a = p.parse_args()
    from .config import read_config
    config = read_config(a.config)
    if a.lock_arms:
        result = lock(config, json.loads(Path(a.lock_arms).read_text()))
    elif a.reference:
        result = gate_paths(a.reference, a.candidate_predictions)
    elif a.diagnostics:
        result = scale_diagnostics(config, a.stage, a.candidate, a.arm, a.aggregation, a.fit_pool)
    elif a.select:
        result = select(config, a.stage)
    else:
        result = evaluate(config, a.stage, a.candidate, a.pool, a.arm, a.aggregation, a.fit_pool)
    if a.output:
        write_json(Path(a.output), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
