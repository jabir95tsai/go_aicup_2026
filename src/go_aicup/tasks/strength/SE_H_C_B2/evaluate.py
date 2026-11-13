"""Paired per-k official scores and rank-stratified player bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import write_json
from go_aicup.metrics import score_rank, NEAR
from .common import A, S, H, C, BUCKETS, atomic_table
from .data import load_prepared, questions
from .features import assemble
from .model import probabilities
from .train import load_bundle, question_values


def official_rows(pred, truth):
    error = np.abs(np.asarray(pred) - np.asarray(truth))
    return np.where(error == 0, 1.0, np.where(error == 1, NEAR, 0.0))


def report(q, proba):
    pred = score_rank(proba)
    scores = official_rows(pred, q.rank_id)
    result = {'by_k': {}}
    for k in BUCKETS:
        mask = q.k.to_numpy() == k
        if not mask.any():
            raise ValueError('all k buckets are required for Equal-k')
        err = np.abs(pred[mask] - q.rank_id.to_numpy()[mask])
        result['by_k'][str(k)] = dict(score=float(scores[mask].mean()), n=int(mask.sum()),
            players=int(q.loc[mask, 'player_id'].nunique()), exact=int((err == 0).sum()), adjacent=int((err == 1).sum()))
    result['equal_k'] = float(np.mean([result['by_k'][str(k)]['score'] for k in BUCKETS]))
    return result


def paired_bootstrap(q, reference, candidate, repeats, seed):
    players = q[['player_id', 'rank_id']].drop_duplicates().reset_index(drop=True)
    if players.player_id.duplicated().any():
        raise ValueError('bootstrap player must have one rank')
    player_index = dict(zip(players.player_id, range(len(players))))
    rows = np.array([player_index[p] for p in q.player_id])
    counts = np.zeros((len(players), 4)); gains = np.zeros_like(counts)
    for column, k in enumerate(BUCKETS):
        mask = q.k.to_numpy() == k
        np.add.at(counts[:, column], rows[mask], 1)
        np.add.at(gains[:, column], rows[mask], (candidate - reference)[mask])
    rng = np.random.default_rng(seed)
    strata = [group.index.to_numpy() for _, group in players.groupby('rank_id', sort=True)]
    samples, skipped = [], 0
    for _ in range(repeats):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        weight = np.bincount(selected, minlength=len(players))
        denominator = weight @ counts
        if (denominator == 0).any():
            skipped += 1; continue
        by_k = weight @ gains / denominator
        samples.append([*by_k, by_k.mean()])
    if not samples:
        raise ValueError('no valid player bootstrap replicates')
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    return dict(repeats=len(samples), skipped=skipped,
                intervals={str(k): intervals[:, i].tolist() for i, k in enumerate((*BUCKETS, 'equal_k'))})


def evaluate(config, stage, role):
    load_prepared(config)
    if (stage, role) not in [('dev', 'select'), ('final', 'newgames'), ('final', 'old_u')]:
        raise ValueError('dev evaluates select; final evaluates newgames or diagnostic old_u')
    root = Path(config['paths']['output'])
    q = questions(config, role)
    ids = np.unique(np.concatenate(q.games))
    result = dict(stage=stage, role=role, limitation='SE3 player overlap remains; old_u also overlaps training games.')
    scored = {}
    directory = root / f'reports/{stage}_{role}'
    directory.mkdir(parents=True, exist_ok=True)
    for arm in ('a', 'se3'):
        bundle = load_bundle(root / f'models/{stage}/{arm}.pt', config, arm)
        frame = assemble(config, arm, ids)
        values = question_values(frame, q, bundle['columns'])
        proba = probabilities(bundle['heads'], values, q.k.to_numpy())
        result[arm] = report(q, proba)
        pred = score_rank(proba); scored[arm] = official_rows(pred, q.rank_id.to_numpy())
        output = q.drop(columns='games').assign(predicted_rank_id=pred, score=scored[arm])
        for i in range(10):
            output[f'p_{i}'] = proba[:, i]
        atomic_table(directory / f'{arm}_predictions.parquet', output)
        confusion = np.zeros((4, 10, 10), np.int64)
        rank_rows = []
        for i, k in enumerate(BUCKETS):
            mask = q.k.to_numpy() == k
            np.add.at(confusion[i], (q.rank_id.to_numpy()[mask], pred[mask]), 1)
            for rank in range(10):
                selected = mask & q.rank_id.eq(rank).to_numpy()
                if selected.any():
                    rank_rows.append(dict(k=k, rank_id=rank, n=int(selected.sum()), score=float(scored[arm][selected].mean())))
        np.save(directory / f'{arm}_confusion.npy', confusion, allow_pickle=False)
        atomic_table(directory / f'{arm}_per_rank.parquet', pd.DataFrame(rank_rows))
    result['delta'] = {str(k): result['se3']['by_k'][str(k)]['score'] - result['a']['by_k'][str(k)]['score'] for k in BUCKETS}
    result['delta']['equal_k'] = result['se3']['equal_k'] - result['a']['equal_k']
    result['bootstrap'] = paired_bootstrap(q, scored['a'], scored['se3'], config['bootstrap'], config['seed'])
    result['candidate_gate'] = (result['delta']['equal_k'] >= 0.005 and min(result['delta'][str(k)] for k in BUCKETS) >= -0.005)
    result['positive_interval'] = result['bootstrap']['intervals']['equal_k'][0] > 0
    result['continue_development'] = not (all(result['delta'][str(k)] < 0 for k in BUCKETS) and result['delta']['equal_k'] <= 0)
    write_json(directory / 'result.json', result)
    return result
