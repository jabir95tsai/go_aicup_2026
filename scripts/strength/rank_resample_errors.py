"""Paired resampling of unseen players: cvec2, adj swap and cvec2+adj.

Reuse existing features and B2 recipes; never fit Valid-U players. Each fixed
B2 seed predicts the same 200 same-colour subsets per player and k. The main
cohort has >=20 games; >=40 games is a sensitivity check for pool exhaustion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

OUT = ROOT / 'outputs/rank/humansl/resample_errors_20261113'
ARMS = ('base', 'swap', 'add')
DRAWS = 200
SEED = 20261113


def measures(pred, truth):
    """Per-group measures; stability is defined within one fixed classifier."""
    counts = np.stack([np.bincount(p, minlength=10) for p in pred])
    mode = counts.argmax(axis=1)
    frequency = counts.max(axis=1) / pred.shape[1]
    distance = np.abs(pred - truth[:, None])
    score = np.where(distance == 0, 1., np.where(distance == 1, np.exp(-1.), 0.))
    return {'exact': (distance == 0).mean(axis=1), 'off1': (distance == 1).mean(axis=1),
            'score': score.mean(axis=1), 'signed_error': (pred - truth[:, None]).mean(axis=1),
            'mode': mode, 'mode_frequency': frequency,
            'stable_correct': (mode == truth) & (frequency >= .9),
            'stable_wrong': (mode != truth) & (frequency >= .9),
            'ever_correct_and_wrong': (distance == 0).any(axis=1) & (distance != 0).any(axis=1)}


def summarize(pred, truth, mask):
    pred, truth = pred[mask], truth[mask]
    if not len(truth):
        return {'groups': 0}
    m = measures(pred, truth)
    loss = 1 - m['score']
    stable = m['stable_wrong']
    return {'groups': len(truth),
            **{k: float(np.mean(m[k])) for k in ('score', 'exact', 'off1', 'stable_correct',
                                                'stable_wrong', 'ever_correct_and_wrong')},
            'variable_prediction': float(np.mean(~(m['stable_correct'] | stable))),
            'stable_wrong_loss_share': float(loss[stable].sum() / loss.sum()) if loss.sum() else 0.}


def paired_ci(delta, players):
    """Bootstrap players, keeping both colours together and averaging draw scores first."""
    groups = list(pd.Series(np.arange(len(players))).groupby(players).apply(np.asarray))
    rng = np.random.default_rng(SEED)
    values = [delta[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean()
              for _ in range(2000)]
    return np.quantile(values, [.025, .975]).tolist()


def self_check():
    pred = np.array([[2, 2, 2, 2], [1, 1, 1, 1], [1, 2, 1, 2], [0, 1, 0, 1]])
    truth = np.full(4, 2)
    m = measures(pred, truth)
    assert m['stable_correct'].tolist() == [True, False, False, False]
    assert m['stable_wrong'].tolist() == [False, True, False, False]
    assert m['ever_correct_and_wrong'].tolist() == [False, False, True, False]
    assert np.allclose(m['score'], [1, np.exp(-1), (1 + np.exp(-1))/2, np.exp(-1)/2])
    assert summarize(pred, truth, np.ones(4, bool))['variable_prediction'] == .5
    assert np.allclose(paired_ci(np.zeros(4), np.array(['a', 'a', 'b', 'c'])), 0)


def task(seed, arm, fit, xf, prior, views, draws, full, truth, sizes, eval_games, qs):
    from threadpoolctl import threadpool_limits
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_c4x as c4
    import scripts.strength.humansl_hd_filter as hf
    with threadpool_limits(limits=1):
        hs.SEED = seed
        model = more.PriorMatched(prior).fit(fit, pd.DataFrame(), xf)
        predictions, full_predictions = [], []
        for k in K_BUCKETS:
            scaler, lr = model.per_k[k]
            def predict(features):
                probability = np.mean([lr.predict_proba(scaler.transform(f)) for f in features], axis=0)
                return score_rank(probability).astype(np.int8)
            features = [np.stack([x[index].mean(axis=1) for index in draws[k]])
                        .reshape(-1, x.shape[1]) for x in views]
            predictions.append(predict(features).reshape(len(truth), DRAWS))
            full_predictions.append(predict(full))
        # Reproduce the existing fixed-question results before interpreting new draws.
        check = {name: hf.by_k(c4.view_scores(model, eval_games, views, q), q['k'].to_numpy())
                 for name, q in qs.items()}
        print(json.dumps({'finished': [seed, arm], 'E2': check['E2']['equal_k']}), flush=True)
        return np.stack(predictions), np.stack(full_predictions), check


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('self-check passed')
        return
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits
    import scripts.strength.rank_c4x as c4
    import scripts.strength.rank_more_players as more
    import scripts.strength.humansl_hd_filter as hf
    OUT.mkdir(parents=True, exist_ok=True)
    games, fit, ev, xf, xe = c4.base_data('v2')
    cf, cv = c4.c4_data('adj', fit, ev)
    keep = (fit['handicap'] <= 1).to_numpy()
    prior = more.player_prior(fit[keep & (fit['role'] == 'S').to_numpy()])
    assert not set(fit.player_id) & set(ev.player_id), 'fitting/evaluation player overlap'
    all_x = {'base': (xf, [xe])}
    for arm in ('swap', 'add'):
        all_x[arm] = (c4.arm_x(xf, cf, arm), [c4.arm_x(xe, cv[h], arm) for h in c4.HALVES])
    assert all(np.isfinite(x[keep]).all() and all(np.isfinite(v).all() for v in vs)
               for x, vs in all_x.values())
    qs = {n: q for n, q in hf.question_sets(games, rebuild=False).items() if n in ('E1', 'E2')}
    ug = ev[ev.handicap <= 1]
    groups = [rows for rows in ug.groupby(['player_id', 'player_color']).groups.values() if len(rows) >= 20]
    groups = [np.asarray(rows) for rows in groups]
    meta = ev.loc[[rows[0] for rows in groups], ['player_id', 'player_color', 'rank_id']].reset_index(drop=True)
    meta['games'] = [len(rows) for rows in groups]
    truth, sizes = meta.rank_id.to_numpy(), meta.games.to_numpy()
    assert all(ev.loc[rows, 'rank_id'].nunique() == 1 for rows in groups)
    rng = np.random.default_rng(SEED)
    draws = {k: [np.stack([rng.choice(rows, k, replace=False) for _ in range(DRAWS)]) for rows in groups]
             for k in K_BUCKETS}
    assert all(all(len(np.unique(r)) == k for r in index) for k, ds in draws.items() for index in ds)
    full = {a: [np.stack([v[rows].mean(axis=0) for rows in groups]) for v in vs] for a, (_, vs) in all_x.items()}
    tasks = [(s, a) for s in hf.SEEDS for a in ARMS]
    print(json.dumps({'groups': len(groups), 'players': meta.player_id.nunique(), 'draws': DRAWS,
                      'tasks': len(tasks), 'fit_games': int(keep.sum())}), flush=True)
    with threadpool_limits(limits=1):
        done = dict(zip(tasks, Parallel(n_jobs=6)(delayed(task)(s, a, fit[keep].reset_index(drop=True),
            all_x[a][0][keep], prior, all_x[a][1], draws, full[a], truth, sizes, ev, qs) for s, a in tasks)))
    old = json.loads((c4.OUT / 'result_adj_v2.json').read_text())
    for s, a in tasks:
        for name in qs:
            assert all(abs(done[s, a][2][name][str(k)] - old['seeds'][str(s)][name][a][str(k)]) <= .0002
                       for k in K_BUCKETS), f'fixed-question reproduction failed: {s}/{a}/{name}'
    pred = {a: np.stack([done[s, a][0] for s in hf.SEEDS]) for a in ARMS}
    full_pred = {a: np.stack([done[s, a][1] for s in hf.SEEDS]) for a in ARMS}
    np.savez_compressed(OUT / 'predictions.npz', **pred, **{f'{a}_all_games': v for a, v in full_pred.items()})
    meta.to_csv(OUT / 'groups.csv', index=False)
    result = {'protocol': {'seeds': list(hf.SEEDS), 'draw_seed': SEED, 'draws_per_group': DRAWS,
        'group': 'Valid-U player and colour, HD<=1, >=20 games', 'stable_threshold': .9,
        'arms': {'base': 'H+C v2', 'swap': 'H+adj', 'add': 'H+C v2+adj'},
        'view_rule': 'mean of B2 probabilities, then score_rank',
        'limits': ['subsets overlap; draws are not independent new players',
                   'all-games predictions use finite pools and are not a true infinite-game limit',
                   'the final submission fits Valid-U; this diagnostic excludes it to avoid leakage']},
        'reproduced_E1_E2': True, 'groups': len(groups), 'players': int(meta.player_id.nunique()),
        'summary': {}, 'paired': {}, 'seed_summary': {}}
    rows = []
    for ki, k in enumerate(K_BUCKETS):
        masks = {'all': np.ones(len(meta), bool), 'mid': np.isin(truth, [3, 4, 5, 6]),
                 'rich': sizes >= 40, 'mid_rich': (sizes >= 40) & np.isin(truth, [3, 4, 5, 6])}
        result['summary'][str(k)], result['paired'][str(k)] = {}, {}
        for cohort, mask in masks.items():
            result['summary'][str(k)][cohort] = {}
            for a in ARMS:
                summaries = [summarize(pred[a][si, ki], truth, mask) for si in range(len(hf.SEEDS))]
                result['seed_summary'][f'{k}/{cohort}/{a}'] = summaries
                result['summary'][str(k)][cohort][a] = {key: float(np.mean([r[key] for r in summaries]))
                                                         for key in summaries[0]}
            paired = {}
            bm = [measures(p, truth) for p in pred['base'][:, ki]]
            for a in ('swap', 'add'):
                am = [measures(p, truth) for p in pred[a][:, ki]]
                delta = np.mean([x['score'] - y['score'] for x, y in zip(am, bm)], axis=0)
                bcorrect = pred['base'][:, ki, mask] == truth[mask][None, :, None]
                acorrect = pred[a][:, ki, mask] == truth[mask][None, :, None]
                paired[a] = {'score_delta': float(delta[mask].mean()),
                    'player_bootstrap_95': paired_ci(delta[mask], meta.player_id.to_numpy()[mask]),
                    'wrong_to_exact': float((~bcorrect & acorrect).mean()),
                    'exact_to_wrong': float((bcorrect & ~acorrect).mean()),
                    'persistent_wrong_score_delta': float(delta[mask & (np.mean([m['stable_wrong'] for m in bm], axis=0) >= .8)].mean())
                        if (mask & (np.mean([m['stable_wrong'] for m in bm], axis=0) >= .8)).any() else None}
            result['paired'][str(k)][cohort] = paired
        for a in ARMS:
            ms = [measures(p, truth) for p in pred[a][:, ki]]
            for gi, group in meta.iterrows():
                row = group.to_dict() | {'arm': a, 'k': k}
                for key in ('exact', 'off1', 'score', 'signed_error', 'mode_frequency', 'stable_wrong', 'stable_correct'):
                    row[key] = float(np.mean([m[key][gi] for m in ms]))
                modes = [int(m['mode'][gi]) for m in ms]
                row['mode_by_seed'] = json.dumps(modes)
                row['exact_by_seed'] = json.dumps([round(float(m['exact'][gi]), 3) for m in ms])
                row['all_games_by_seed'] = json.dumps(full_pred[a][:, ki, gi].tolist())
                row['status'] = ('persistent_wrong' if row['stable_wrong'] >= .8 and max(modes.count(m) for m in modes) >= 4
                                 else 'stable_correct' if row['stable_correct'] >= .8 else 'variable')
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'players.csv', index=False)
    (OUT / 'result.json').write_text(json.dumps(result, indent=1))
    print(json.dumps({'summary': result['summary'], 'paired': result['paired']}), flush=True)


if __name__ == '__main__':
    main()
