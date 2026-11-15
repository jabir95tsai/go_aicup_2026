"""Post-training diagnostics, no attention updates or production selection.

Calibrators and linear probes fit TRAIN only. Hold/U are reported separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from go_aicup.artifacts import write_json
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores
from scripts.strength.rank_move_weights import d_prime


def ce(z, y):
    z = np.asarray(z, dtype=np.float64)
    return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))


def report(z, y):
    p = softmax(z, axis=1)
    score = official_scores(score_rank(p),y)
    exact = z.argmax(1) == y
    return {'ce':ce(z,y), 'accuracy':float(exact.mean()),
            'official':float(score.mean()),
            'by_rank':{str(r):{'ce':ce(z[y==r],y[y==r]),
                                'accuracy':float(exact[y==r].mean()),
                                'official':float(score[y==r].mean())} for r in np.unique(y)}}


def self_check():
    z = np.array([[1.,2.,3.],[4.,1.,2.]])
    y = np.array([2,0])
    assert np.isclose(ce(z,y),ce(z + np.array([[100.],[-100.]]),y))
    assert ce(z*2,y) < ce(z,y)
    assert np.array_equal(z.argmax(1),(z*2).argmax(1))


def diagnose(out):
    self_check()
    complete = json.loads((out/'complete.json').read_text())
    table = pd.read_parquet(out/'games.parquet')
    f = np.load(out/'features.npz')
    assert np.array_equal(table.game_index,f['game_index'])
    raw = np.load(ROOT/'outputs/rank/humansl/permove/agg_train.npz')
    lookup = pd.Series(np.arange(len(raw['game_index'])),index=raw['game_index'])
    arms = {'mean':raw['mean'][lookup.loc[table.game_index].to_numpy()].astype(float),
            'weighted':f['c'].astype(float)}
    train = table['set'].eq('train').to_numpy()
    labels = table.rank_id.to_numpy()
    splits = {s:table['set'].eq(s).to_numpy() for s in ('train','hold','U')}
    assert not set(table.loc[train,'player_id']) & set(table.loc[splits['U'],'player_id'])
    result = {'training':complete,'raw':{},'linear_probe':{},'separation':{},
              'limitation':'Observational diagnostics; no retraining attention or causal ablation. Temperature/probes use TRAIN only; game-level samples are correlated within players.'}
    for arm,c in arms.items():
        result['raw'][arm] = {}
        for side,sl in [('own',slice(0,10)),('opp',slice(10,20))]:
            z = c[:,sl]
            fitted = minimize_scalar(lambda a:ce(z[train]*np.exp(a),labels[train]),bounds=(-5,5),method='bounded')
            scale = float(np.exp(fitted.x))
            result['raw'][arm][side] = {'train_fitted_multiplier':scale,'splits':{}}
            for name,mask in splits.items():
                zz = z[mask]
                centred = zz - zz.mean(1,keepdims=True)
                result['raw'][arm][side]['splits'][name] = {
                    'unscaled':report(zz,labels[mask]), 'scaled':report(zz*scale,labels[mask]),
                    'mean_rank_spread':float(zz.std(1).mean()),
                    'common_logp_quantiles':np.quantile(zz.mean(1),[.1,.5,.9]).tolist(),
                    'contrast_norm_quantiles':np.quantile(np.linalg.norm(centred,axis=1),[.1,.5,.9]).tolist()}
        result['linear_probe'][arm] = {}
        result['separation'][arm] = {}
        for block,sl in [('own',slice(0,10)),('opp',slice(10,20)),('both',slice(0,20))]:
            x = c[:,sl]
            scaler = StandardScaler().fit(x[train])
            lr = LogisticRegression(C=1.,max_iter=5000).fit(scaler.transform(x[train]),labels[train])
            result['linear_probe'][arm][block] = {}
            result['separation'][arm][block] = {}
            standard = scaler.transform(x)
            for name,mask in splits.items():
                logits = lr.decision_function(scaler.transform(x[mask]))
                result['linear_probe'][arm][block][name] = report(logits,labels[mask])
                zz = standard[mask]
                by_player = pd.DataFrame(zz).groupby(table.loc[mask,'player_id'].to_numpy()).transform('mean').to_numpy()
                within = np.square(zz-by_player).sum() / np.square(zz-zz.mean(0)).sum()
                result['separation'][arm][block][name] = {
                    'within_player_variance_fraction':float(within),
                    'middle_dprime':d_prime(standard[train],labels[train],standard[mask],labels[mask]),
                    'all_adjacent_dprime':d_prime(standard[train],labels[train],standard[mask],labels[mask],pairs=tuple((r,r+1) for r in range(9)))}
    write_json(out/'diagnostic.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-check',action='store_true')
    args=p.parse_args()
    if args.self_check:
        self_check()
        print('self-check passed')
    else:
        c=json.loads((ROOT/'configs/strength/board_attention.json').read_text())
        diagnose(ROOT/c['out'])
