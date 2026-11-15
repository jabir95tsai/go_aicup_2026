"""C-only B2 comparison on unseen HD0 players, after attention selection is fixed.

Two new rank-balanced, same-colour question draws; these are HD0 diagnostics,
not the existing HD<=1 E1/E2 benchmarks or a production promotion gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from go_aicup.artifacts import write_json
from go_aicup.data.cv import make_rank_questions
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores


def fit_seed(seed, fit, valid, arms, questions):
    import go_aicup.tasks.strength.humansl as hs
    from scripts.strength.rank_more_players import PriorMatched, player_prior
    hs.SEED = seed
    prior = player_prior(fit)
    scores = {}
    for name, (xf, xu) in arms.items():
        model = PriorMatched(prior).fit(fit, pd.DataFrame(), xf)
        scores[name] = {tag: official_scores(score_rank(model.probabilities(xu, valid.game_index.to_numpy(), q)),
                                             q.rank_id.to_numpy()) for tag, q in questions.items()}
    return scores


def compare(out, jobs=5):
    from scripts.strength.humansl_hd_filter import by_k, bootstrap
    if not (out / 'complete.json').exists():
        raise ValueError('training must finish and freeze its best model before scoring U')
    table = pd.read_parquet(out / 'games.parquet')
    feature = np.load(out / 'features.npz')
    if not np.array_equal(feature['game_index'], table.game_index):
        raise ValueError('features and metadata are not aligned')
    mean = np.load(ROOT / 'outputs/rank/humansl/permove/agg_train.npz')
    lookup = pd.Series(np.arange(len(mean['game_index'])), index=mean['game_index'])
    mean_c = mean['mean'][lookup.loc[table.game_index].to_numpy()]
    if mean_c.shape != feature['c'].shape or not np.isfinite(feature['c']).all():
        raise ValueError('invalid learned C features')
    fit_mask = table['set'].isin(('train', 'hold')).to_numpy()
    u = table['set'].eq('U').to_numpy()
    fit, valid = table[fit_mask].reset_index(drop=True), table[u].reset_index(drop=True)
    if set(fit.player_id) & set(valid.player_id):
        raise ValueError('U players overlap B2 fit players')
    if (table.handicap != 0).any():
        raise ValueError('this evaluation is specifically the first HD0 experiment')
    # Keep the two flag columns for compatibility; both are zero in this HD0 pilot.
    arms = {name: (np.c_[c[fit_mask], np.zeros((fit_mask.sum(), 2))],
                   np.c_[c[u], np.zeros((u.sum(), 2))])
            for name, c in (('mean', mean_c), ('weighted', feature['c']))}
    meta = valid.set_index('game_index').assign(color=valid.set_index('game_index').player_color.map({0:'B', 1:'W'}))
    questions = {}
    for seed in (2027, 2028):
        tag = f'HD0_Q{seed}'
        q = make_rank_questions(meta, meta.player_id.unique(), n_questions=4000, seed=seed)
        q.to_parquet(out / f'{tag}.parquet', index=False)
        questions[tag] = q
    seeds = list(range(2027, 2032))
    scores = Parallel(n_jobs=jobs)(delayed(fit_seed)(s, fit, valid, arms, questions) for s in seeds)
    result = {'fit_games':len(fit), 'fit_players':int(fit.player_id.nunique()),
              'u_games':len(valid), 'u_players':int(valid.player_id.nunique()),
              'questions':{t:q.groupby('k').size().to_dict() for t,q in questions.items()},
              'limitation':'HD0 only, new question draws; not the existing E1/E2. Attention and B2 share fit players; U is player-disjoint.',
              'seeds':{}}
    for seed, pair in zip(seeds, scores):
        result['seeds'][str(seed)] = {}
        for tag,q in questions.items():
            a,b = pair['mean'][tag],pair['weighted'][tag]
            k = q.k.to_numpy()
            entry = {'mean':by_k(a,k), 'weighted':by_k(b,k), 'difference':by_k(b-a,k)}
            if seed == seeds[0]:
                entry['player_bootstrap_95'] = bootstrap(a,b,q)
            result['seeds'][str(seed)][tag] = entry
    result['five_seed_mean'] = {}
    for tag in questions:
        result['five_seed_mean'][tag] = {arm:{key:float(np.mean([result['seeds'][str(s)][tag][arm][key] for s in seeds]))
                                            for key in ('5','10','15','20','equal_k')}
                                       for arm in ('mean','weighted','difference')}
    write_json(out / 'b2_result.json', result)
    print(json.dumps(result['five_seed_mean'],indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT / 'configs/strength/board_attention.json')
    p.add_argument('--jobs',type=int,default=5)
    args=p.parse_args()
    compare(ROOT / json.loads(args.config.read_text())['out'],args.jobs)


if __name__ == '__main__':
    main()
