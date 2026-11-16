"""Summarize frozen TRAIN predictions on fixed, historically sampled question subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import object_hash, write_json
from .data import Cache, question_rows
from .evaluate_train import ARMS, evaluate_checkpoint
from .train import sample_questions


def subset_report(frame, mask, head_l2):
    part = frame.loc[mask]
    p = part[[f'p{r}' for r in range(10)]].to_numpy()
    truth = p[np.arange(len(part)), part.rank_id.to_numpy()]
    if not len(part) or not np.isfinite(p).all() or (truth <= 0).any():
        raise ValueError('invalid diagnostic probabilities')
    ce = float(-np.log(truth).mean())
    by_k = {str(k): float(g.score.mean()) for k, g in part.groupby('k')}
    if set(by_k) != {'5', '10', '15', '20'}:
        raise ValueError('subset misses a k bucket')
    return {'questions': len(part), 'players': int(part.player_id.nunique()), 'ce': ce,
            'objective': ce + head_l2, 'equal_k': float(np.mean(list(by_k.values()))),
            'by_k': by_k, 'questions_by_k': {str(k): len(g) for k, g in part.groupby('k')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    args = parser.parse_args()
    out = args.diagnostic
    identity = json.loads((out / 'identity.json').read_text())
    if json.loads((out / 'complete.json').read_text())['status'] != 'complete':
        raise ValueError('diagnostic incomplete')
    source = Path(identity['source'])
    config = json.loads((source / 'config.json').read_text())
    questions = pd.read_parquet(source / 'train_questions.parquet')
    games = pd.read_parquet(source / 'games.parquet')
    train_games = games[games.role.eq('train')].reset_index(drop=True)
    rng = np.random.default_rng(config['seed'])
    seen, seen_games = np.zeros(len(questions), dtype=bool), set()
    groups = {'all_fixed_train': np.ones(len(questions), dtype=bool)}
    for step in range(1, 201):
        chosen = sample_questions(questions, rng, config['questions_per_k'])
        rng.integers(8)
        seen[chosen] = True
        for row in questions.iloc[chosen].itertuples():
            seen_games.update(row.games)
        if step in (100, 200):
            groups[f'sampled_through_step_{step}'] = seen.copy()
    for arm in ARMS:
        saved = torch.load(source / arm / 'last.pt', map_location='cpu', weights_only=False)
        if (object_hash(saved['rng']) != object_hash(rng.bit_generator.state)
                or set(games.loc[saved['seen'], 'game_index']) != seen_games):
            raise ValueError('historical sampling replay mismatch')
    raw = np.load(source / 'raw_c.npz')
    indices = pd.Index(raw['game_index']).get_indexer(train_games.game_index)
    if (indices < 0).any():
        raise ValueError('missing original raw C')
    cache = Cache(train_games, None, {'c': torch.from_numpy(raw['c'][indices].copy())}, torch.device('cpu'))
    initial, _, frame = evaluate_checkpoint(source / 'A0/initial.pt', 'A0', cache, questions,
                                           question_rows(train_games, questions),
                                           identity['source_identity'], config['position_chunk'])
    initial['subsets'] = {name: subset_report(frame, mask, initial['head_l2'])
                          for name, mask in groups.items()}
    result = json.loads((out / 'result.json').read_text())
    arms = {}
    for arm in ARMS:
        arms[arm] = {}
        for label in ('selected', 'last'):
            frame = pd.read_parquet(out / f'{arm}_{label}_predictions.parquet')
            if not frame[['question_id', 'player_id', 'rank_id', 'k']].equals(
                    questions[['question_id', 'player_id', 'rank_id', 'k']]):
                raise ValueError('unpaired diagnostic questions')
            report = result['arms'][arm][label]
            subsets = {name: subset_report(frame, mask, report['head_l2'])
                       for name, mask in groups.items()}
            np.testing.assert_allclose(subsets['all_fixed_train']['ce'], report['ce'], atol=2e-6, rtol=0)
            np.testing.assert_allclose(subsets['all_fixed_train']['equal_k'], report['equal_k'], atol=1e-12, rtol=0)
            arms[arm][label] = {'step': report['step'], 'subsets': subsets}
    report = {'initial_A0': initial, 'arms': arms,
              'limitations': ['all 8000 TRAIN labels were used by the original sklearn head warm-start',
                              'sampled subsets describe subsequent AdamW updates only',
                              'fixed subsets reused for every checkpoint; these are in-sample diagnostics']}
    write_json(out / 'sampling_analysis.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
