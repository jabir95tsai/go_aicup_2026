"""CPU checks for joint gradients, KL, question isolation and checkpoint resume."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from go_aicup.tasks.strength.board_attention import joint
from go_aicup.tasks.strength.board_attention.train import Games, MoveWeights, pool


def synthetic():
    table = pd.DataFrame([{'rank_id': r, 'player_id': f'{s}_{r}', 'set': s,
                           'game_index': 10000 + len_s * 200 + r * 20 + g,
                           'player_color': r % 2}
                          for len_s, s in enumerate(('train', 'hold', 'U'))
                          for r in range(10) for g in range(20)])
    n = len(table) * 4
    rng = np.random.default_rng(7)
    lp = rng.normal(-2, .2, (len(table), 4, 10)).astype(np.float32)
    for i, rank in enumerate(table.rank_id):
        lp[i, :, rank] += .4
    data = Games(table, torch.tensor(rng.integers(0, 256, (n, 2, 46), dtype=np.uint8)),
                 torch.tensor(lp.reshape(-1, 10)), torch.tensor([0, 1, 0, 1] * len(table)),
                 torch.tensor([0, 1, 2, 3] * len(table)), np.arange(len(table) + 1) * 4)
    return data


def test_kl_uniform_concentrated_and_gradient():
    slot = torch.tensor([0, 0, 1, 1, 1, 2, 3])
    lp = torch.randn(7, 10)
    _, w = pool(torch.zeros(7), lp, slot, 4)
    torch.testing.assert_close(joint.weight_kl(w, slot, 4), torch.zeros(2), atol=1e-7, rtol=0)
    scores = torch.tensor([2., -2., 3., 0., -3., 0., 0.], requires_grad=True)
    _, w = pool(scores, lp, slot, 4)
    penalty = joint.weight_kl(w, slot, 4).mean()
    penalty.backward()
    assert penalty > 0 and torch.isfinite(scores.grad).all()
    assert scores.grad[0] > 0 and scores.grad[1] < 0
    with pytest.raises(ValueError, match='empty'):
        joint.weight_kl(w, slot, 5)


def test_question_mean_and_kl_are_question_weighted():
    values = torch.tensor([[1., 2.], [3., 4.], [9., 10.]], requires_grad=True)
    out = joint.aggregate(values, [np.array([0, 1]), np.array([2])])
    torch.testing.assert_close(out, torch.tensor([[2., 3.], [9., 10.]]))
    out.mean().backward()
    torch.testing.assert_close(values.grad[2], 2 * values.grad[0])


def test_question_isolation_balancing_and_sampling_resume():
    data = synthetic()
    q, rows = joint.questions(data.table, 'train', 400, 2027)
    assert (q.groupby(['k', 'rank_id']).size() == 10).all()
    assert len(q) == len(rows) == 400
    rng = np.random.default_rng(10)
    selected = joint.sample_questions(q, rng, 2)
    assert (q.iloc[selected].groupby(['k', 'rank_id']).size() == 2).all()
    state = copy.deepcopy(rng.bit_generator.state)
    expected = joint.sample_questions(q, rng, 2)
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    np.testing.assert_array_equal(joint.sample_questions(q, restored, 2), expected)


def test_sklearn_transfer_joint_gradients_and_mean_control():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    data = synthetic()
    attention = MoveWeights()
    c, stats = joint.game_features(attention, data, np.arange(len(data.table)), uniform=True)
    assert stats['effective_moves'] == 2 and stats['max_weight'] == .5
    q, rows = joint.questions(data.table, 'train', 400, 2027)
    head = joint.fit_b2(c, q, rows)
    selected = joint.sample_questions(q, np.random.default_rng(1), 1)
    buckets = torch.tensor([joint.K_BUCKETS.index(int(k)) for k in q.iloc[selected].k])
    labels = torch.tensor(q.iloc[selected].rank_id.to_numpy())
    selected_rows = [rows[i] for i in selected]
    x = torch.tensor(np.stack([c[r].mean(0) for r in selected_rows]))
    expected = F.cross_entropy(head(x, buckets), labels) + head.penalty(100)
    loss, ce, kl = joint.question_forward(attention, head, data, selected_rows, buckets, labels, .1, 100)
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kl, torch.tensor(0.))
    loss.backward()
    assert attention.head[-1].weight.grad.abs().sum() > 0
    assert head.weight.grad.abs().sum() > 0
    attention.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    loss, _, _ = joint.question_forward(attention, head, data, selected_rows, buckets, labels, None, 100)
    loss.backward()
    assert all(p.grad is None for p in attention.parameters())
    assert head.weight.grad.abs().sum() > 0


def test_training_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(3)
    data = synthetic()
    train_q, train_rows = joint.questions(data.table, 'train', 400, 2027)
    hold_q, hold_rows = joint.questions(data.table, 'hold', 80, 2028)
    attention = MoveWeights()
    c, _ = joint.game_features(attention, data, np.arange(len(data.table)), uniform=True)
    head = joint.fit_b2(c, train_q, train_rows)
    initial = {'attention': copy.deepcopy(attention.state_dict()), 'b2': copy.deepcopy(head.state_dict())}
    config = {'seed': 2027, 'lr': .001, 'weight_decay': .0001, 'updates': 2,
              'every': 1, 'min_updates': 0, 'patience': 10, 'questions_per_cell': 1}
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    full.mkdir()
    resumed.mkdir()
    for directory, updates in ((full, 2), (resumed, 1), (resumed, 2)):
        joint.train_arm(data, dict(config, updates=updates), 'joint_kl01', directory, 'same', initial,
                        train_q, train_rows, hold_q, hold_rows, c)
    a = torch.load(full / 'last.pt', weights_only=False)
    b = torch.load(resumed / 'last.pt', weights_only=False)
    for group in ('attention', 'b2'):
        for key in a[group]:
            torch.testing.assert_close(a[group][key], b[group][key], atol=0, rtol=0)
    assert a['rng'] == b['rng']
    assert json.loads((resumed / 'training_complete.json').read_text())['updates'] == 2
    with pytest.raises(ValueError, match='identity'):
        joint.train_arm(data, config, 'joint_kl01', resumed, 'changed', initial,
                        train_q, train_rows, hold_q, hold_rows, c)


def test_run_publishes_frozen_u_predictions_and_completed_retry(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = synthetic()

    def load(out, device):
        data.table.to_parquet(out / 'games.parquet', index=False)
        joint.write_json(out / 'data.json', {'seconds': 0., 'synthetic': True})
        return data

    monkeypatch.setattr(joint.base, 'load_data', load)
    monkeypatch.setattr(joint.base, 'normalisation', lambda data: (torch.zeros(13), torch.ones(13)))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'synthetic CPU')
    config = {'out': str(tmp_path), 'seed': 2027, 'lr': .001, 'weight_decay': .0001,
              'updates': 1, 'every': 1, 'min_updates': 0, 'patience': 10,
              'questions_per_cell': 1, 'train_questions': 400, 'hold_questions': 80, 'u_questions': 80}
    joint.run(config, 'joint_kl001')
    out = tmp_path / 'joint_kl001'
    result = json.loads((out / 'result.json').read_text())
    assert set(result['U']) == {'2027', '2028'}
    assert set(result['hold']['by_k']) == {'5', '10', '15', '20'}
    for seed in (2027, 2028):
        pred = pd.read_parquet(out / f'predictions_Q{seed}.parquet')
        assert len(pred) == 80
        assert np.allclose(pred[[f'p_{r}' for r in range(10)]].sum(axis=1), 1)
    timestamp = (out / 'result.json').stat().st_mtime_ns
    joint.run(config, 'joint_kl001')
    assert (out / 'result.json').stat().st_mtime_ns == timestamp
