import json

import numpy as np
import pandas as pd
import pytest
import torch

from test_policy_residual import one_thread, policies
from go_aicup.tasks.strength.policy_residual.data import Cache
from go_aicup.tasks.strength.policy_residual.epoch import (
    epoch_batches, gradient_snapshot, run_arm, update_statistics,
)
from go_aicup.tasks.strength.policy_residual.model import Head, Residual
from go_aicup.tasks.strength.policy_residual.run import finalize_cache


def test_epoch_visits_every_question_once_with_paired_balanced_batches():
    q = pd.DataFrame({'k': np.repeat([5, 10, 15, 20], 2000)})
    a = epoch_batches(q, 2027, 1, 8)
    b = epoch_batches(q, 2027, 1, 8)
    c = epoch_batches(q, 2027, 2, 8)
    assert len(a) == 250
    np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(8000))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
        assert q.iloc[x].groupby('k').size().to_dict() == {5: 8, 10: 8, 15: 8, 20: 8}
    assert not np.array_equal(np.concatenate(a), np.concatenate(c))
    with pytest.raises(ValueError, match='divisible'):
        epoch_batches(q.iloc[:-1], 2027, 1, 8)


def test_gradient_summary_measures_updates_and_rejects_nonfinite_values():
    model, head = Residual('A0'), Head()
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    head.linear.weight.square().sum().backward()
    snapshots, report = gradient_snapshot(model, head)
    expected = .01 * report['head']['gradient_norm']
    optimizer.step()
    report = update_statistics(snapshots, report)
    assert report['head']['update_norm'] == pytest.approx(expected, rel=1e-5)
    assert report['head']['update_ratio'] > 0
    assert report['head']['nonfinite_gradients'] == 0
    head.linear.weight.grad[0, 0] = torch.nan
    with pytest.raises(ValueError, match='nonfinite'):
        gradient_snapshot(model, head)


def epoch_cache():
    games = pd.DataFrame({'game_index': np.arange(60), 'length': [2] * 60,
                          'role': np.repeat(['train', 'monitor', 'dev'], 20),
                          'player_id': np.repeat(['train-p', 'monitor-p', 'dev-p'], 20),
                          'player_color': [0] * 60, 'rank_id': [2] * 60})
    lp, target, q = policies(120)
    arrays = {'logp': lp, 'q': q, 'target': target.short(),
              'side': torch.tensor([0, 1] * 60, dtype=torch.uint8),
              'done': torch.ones(60, dtype=torch.bool)}
    bounds = np.arange(61) * 2
    finalize_cache(games, bounds, arrays)
    questions = {}
    for role in ['train', 'monitor', 'dev']:
        ids = games.loc[games.role.eq(role), 'game_index'].to_numpy()
        questions[role] = pd.DataFrame([
            {'question_id': f'{role}-{k}-{j}', 'player_id': f'{role}-p', 'rank_id': 2,
             'k': k, 'games': ids[:k]} for k in [5, 10, 15, 20] for j in range(2)])
    return Cache(games, bounds, arrays, torch.device('cpu')), questions


@pytest.mark.parametrize('arm', ['A0', 'S', 'F'])
def test_complete_epoch_reports_and_resume_without_extra_updates(tmp_path, arm):
    cache, questions = epoch_cache()
    config = {'seed': 2027, 'epochs': 2, 'questions_per_k': 1,
              'position_chunk': 17, 'weight_decay': .0001}
    initial = Head().state_dict()
    case = (f'{arm}_test', arm, .0003)
    run_arm(case, cache, questions, initial, config, 'source-fixed', tmp_path)
    out = tmp_path / case[0]
    history = json.loads((out / 'learning_curve.json').read_text())
    assert [r['step'] for r in history] == [0, 2, 4]
    assert [r['questions_seen_this_epoch'] for r in history[1:]] == [8, 8]
    for row in history:
        assert set(row['train']['by_k']) == {'5', '10', '15', '20'}
        assert np.isfinite(row['train']['objective'])
    result = json.loads((out / 'result.json').read_text())
    assert result['completed_updates'] == 4
    assert result['selected_step'] in [0, 2, 4]
    assert set(result['roles']) == {'train', 'monitor', 'dev'}
    for epoch in (1, 2):
        log = [json.loads(line) for line in (out / f'gradients_epoch_{epoch}.jsonl').read_text().splitlines()]
        assert len(log) == 2
        assert all(v['nonfinite_gradients'] == 0 for row in log for v in row['groups'].values())
        if arm == 'F':
            assert set(log[0]['groups']) == {'head', 'phi', 'scorer', 'delta'}
    before = (out / 'last.pt').read_bytes()
    (out / 'complete.json').unlink()
    run_arm(case, cache, questions, initial, config, 'source-fixed', tmp_path)
    assert (out / 'last.pt').read_bytes() == before
    assert json.loads((out / 'result.json').read_text())['completed_updates'] == 4
    with pytest.raises(ValueError, match='identity'):
        run_arm(case, cache, questions, initial, config, 'different-source', tmp_path)
