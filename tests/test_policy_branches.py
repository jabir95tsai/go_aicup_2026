import copy
import json

import numpy as np
import pytest
import torch

from test_policy_residual import one_thread, policies, small_cache
from test_policy_residual_epoch import epoch_cache
from go_aicup.tasks.strength.policy_residual import branches as b
from go_aicup.tasks.strength.policy_residual.model import Head, question_mean


def initialized(arm):
    torch.manual_seed(2027)
    model = b.Encoder(arm)
    initial = Head().state_dict()
    head = b.BranchHead(model.width, initial)
    return model, head, initial


@pytest.mark.parametrize('arm,width', [('A0', 10), ('R10', 10), ('Z10', 20), ('Z32', 42)])
def test_initial_predictions_scaling_and_shared_encoder(arm, width):
    cache = small_cache()
    model, head, initial = initialized(arm)
    c = b.game_features(model, cache, np.arange(3), 4)
    assert c.shape == (3, 2 * width)
    ref = Head()
    ref.load_state_dict(initial)
    torch.testing.assert_close(head(c), ref(cache.arrays['c']), atol=2e-6, rtol=2e-5)
    for side in (0, 1):
        torch.testing.assert_close(c[:, side * width:side * width + 10], cache.arrays['c'][:, side * 10:side * 10 + 10])
        assert head.mean[side * width + 10:(side + 1) * width].eq(0).all()
        assert head.scale[side * width + 10:(side + 1) * width].eq(1).all()
    if arm != 'A0':
        other, _, _ = initialized('R10')
        for key, value in model.state_dict().items():
            if key.startswith(('phi.', 'scorer.', 'delta.0.')):
                torch.testing.assert_close(value, other.state_dict()[key], atol=0, rtol=0)
    if arm.startswith('Z'):
        assert model.delta[-1].weight.abs().sum() > 0


@pytest.mark.parametrize('arm', ['R10', 'Z10', 'Z32'])
def test_chunked_matches_direct_autograd_with_unequal_lengths_reused_games(arm):
    cache = small_cache()
    # Repartition the same positions into unequal game/side lengths.
    cache.bounds = np.array([0, 4, 10, 18])
    cache.arrays.pop('counts')
    cache.arrays['c'] = torch.stack([torch.cat([
        cache.arrays['q'][start:end][cache.arrays['side'][start:end].eq(side)].mean(0)
        for side in (0, 1)]) for start, end in zip(cache.bounds[:-1], cache.bounds[1:])])
    model, head, _ = initialized(arm)
    with torch.no_grad():
        model.delta[-1].weight.normal_(std=.01)
        if arm.startswith('Z'):
            head.linear.weight.normal_(std=.1)
    direct_model, direct_head = copy.deepcopy(model), copy.deepcopy(head)
    rows, y = [np.array([0, 1]), np.array([1, 2, 0])], torch.tensor([2, 5])
    b.backward_questions(model, head, cache, rows, y, 100, 3)
    lp, target, q, slot = next(cache.chunks(np.arange(3), 100))
    evidence = direct_model(lp, target, q)
    c = torch.zeros(6, model.width).index_add(0, slot, evidence) / cache.counts(np.arange(3))[:, None]
    logits = direct_head(question_mean(c.reshape(3, -1), rows))
    loss = torch.nn.functional.cross_entropy(logits, y) + direct_head.linear.weight.square().sum() / 200
    loss.backward()
    for a, z in zip(list(model.parameters()) + list(head.parameters()),
                    list(direct_model.parameters()) + list(direct_head.parameters())):
        torch.testing.assert_close(a.grad, z.grad, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('arm', ['R10', 'Z10', 'Z32'])
def test_expected_first_step_and_encoder_activation(arm):
    cache = small_cache()
    model, head, _ = initialized(arm)
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'weight_decay': .0001},
                                  {'params': head.parameters(), 'weight_decay': 0.}], lr=.0003)
    rows, y = [np.array([0, 1]), np.array([1, 2])], torch.tensor([2, 5])
    b.backward_questions(model, head, cache, rows, y, 100, 4)
    assert model.phi[0].weight.grad.eq(0).all()
    if arm.startswith('Z'):
        assert model.delta[-1].weight.grad.eq(0).all()
        assert head.linear.weight.grad[:, 10:model.width].abs().sum() > 0
    else:
        assert model.delta[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    b.backward_questions(model, head, cache, rows, y, 100, 4)
    assert model.phi[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('arm', ['R10', 'Z10', 'Z32'])
def test_candidate_permutation_and_d4_invariance(arm):
    model, _, _ = initialized(arm)
    torch.nn.init.normal_(model.delta[-1].weight, std=.1)
    lp, target, q = policies()
    perm = torch.cat((torch.randperm(361), torch.tensor([361])))
    inverse = torch.argsort(perm)
    torch.testing.assert_close(model(lp, target, q), model(lp[:, :, perm], inverse[target], q), atol=2e-6, rtol=2e-5)
    points = torch.arange(361).reshape(19, 19)
    perm = torch.cat((torch.rot90(points, 1).flatten(), torch.tensor([361])))
    torch.testing.assert_close(model(lp, target, q), model(lp[:, :, perm], torch.argsort(perm)[target], q), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('arm', ['A0', 'R10', 'Z10', 'Z32'])
def test_half_epoch_evaluation_resume_and_best_rollback(tmp_path, monkeypatch, arm):
    cache, questions = epoch_cache()
    initial = Head().state_dict()
    config = {'seed': 2027, 'epochs': 2, 'questions_per_k': 1, 'evaluate_every': 1,
              'position_chunk': 17, 'weight_decay': .0001}
    case = (arm, arm, .0003)
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    full.mkdir(), resumed.mkdir()
    b.run_arm(case, cache, questions, initial, config, 'fixed', full)
    original = b.backward_questions
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError('simulated interruption after half epoch')
        return original(*args, **kwargs)
    monkeypatch.setattr(b, 'backward_questions', interrupted)
    with pytest.raises(RuntimeError, match='simulated'):
        b.run_arm(case, cache, questions, initial, config, 'fixed', resumed)
    monkeypatch.setattr(b, 'backward_questions', original)
    b.run_arm(case, cache, questions, initial, config, 'fixed', resumed)
    for filename in ('last.pt', 'model.pt'):
        a = torch.load(full / arm / filename, weights_only=False)
        z = torch.load(resumed / arm / filename, weights_only=False)
        assert a['step'] == z['step']
        for group in ('model', 'head'):
            for key in a[group]:
                torch.testing.assert_close(a[group][key], z[group][key], atol=0, rtol=0)
    history = json.loads((resumed / arm / 'learning_curve.json').read_text())
    assert [r['step'] for r in history] == [0, 1, 2, 3, 4]
    for r in history:
        assert r['train']['added_logits_rms'] >= 0
        assert set(r['train']['by_k']) == {'5', '10', '15', '20'}
    last_before = (resumed / arm / 'last.pt').read_bytes()
    (resumed / arm / 'complete.json').unlink()
    b.run_arm(case, cache, questions, initial, config, 'fixed', resumed)
    assert (resumed / arm / 'last.pt').read_bytes() == last_before
    assert len(list((resumed / arm).glob('*_step_*.npz'))) == 10
    with pytest.raises(ValueError, match='identity'):
        b.run_arm(case, cache, questions, initial, config, 'changed', resumed)


def test_gate_exact_boundaries():
    diff = {'equal_k': .005, 'prob_positive': .9, 'by_k': {'5': -.005, '10': .015, '15': .005, '20': .005}}
    assert b.gate(diff)
    for key, value in [('equal_k', .004999), ('prob_positive', .899999)]:
        assert not b.gate(dict(diff, **{key: value}))
    assert not b.gate(dict(diff, by_k={'5': -.005001}))


def test_branch_logits_decomposition_and_nonfinite_rejection():
    cache = small_cache()
    rows = [np.array([0]), np.array([0, 1, 2])]
    for arm in ['A0', 'R10', 'Z10', 'Z32']:
        model, head, _ = initialized(arm)
        c = b.game_features(model, cache, np.arange(3), 4)
        report = b.branch_rms(model, head, c, cache.arrays['c'], rows)
        assert report['added_logits_rms'] == 0
        if arm != 'A0':
            with torch.no_grad():
                model.delta[-1].bias.add_(.1)
                if arm.startswith('Z'):
                    head.linear.weight.fill_(.1)
            c = b.game_features(model, cache, np.arange(3), 4)
            assert b.branch_rms(model, head, c, cache.arrays['c'], rows)['added_logits_rms'] > 0
        with torch.no_grad():
            head.linear.weight.fill_(float('nan'))
        with pytest.raises(ValueError, match='nonfinite'):
            b.backward_questions(model, head, cache, rows, torch.tensor([2, 5]), 100, 4)


def test_paired_bootstrap_recomputes_equal_k_per_player_draw():
    k = np.tile([5, 10, 15, 20], 4)
    player = np.repeat(np.arange(4), 4)
    a = np.repeat([.1, -.05, .2, .03], 4)
    b0 = np.zeros(16)
    report = b.paired_difference(a, b0, k, player, np.random.default_rng(17))
    draws = np.random.default_rng(17).integers(0, 4, (2000, 4))
    expected = np.array([.1, -.05, .2, .03])[draws].mean(1)
    np.testing.assert_allclose(report['bootstrap_95'], np.quantile(expected, [.025, .975]))
    assert report['prob_positive'] == (expected > 0).mean()
    assert report['bootstrap_draws'] == 2000
    assert set(report['by_k_bootstrap_95']) == {'5', '10', '15', '20'}


def test_summary_pairing_gates_and_completion_identity(tmp_path):
    import pandas as pd
    for arm, score in [('A0', .5), ('R10', .51), ('Z10', .52), ('Z32', .54)]:
        out = tmp_path / arm
        out.mkdir()
        (out / 'complete.json').write_text(json.dumps({'identity': 'fixed', 'status': 'complete'}))
        (out / 'result.json').write_text(json.dumps({'selected_step': 0}))
        pd.DataFrame({'question_id': [str(i) for i in range(8)], 'player_id': ['p'] * 4 + ['q'] * 4,
                      'k': [5, 10, 15, 20] * 2, 'rank_id': [0] * 8, 'score': [score] * 8}).to_parquet(
                          out / 'dev_predictions.parquet')
    report = b.summarize(tmp_path, 'fixed', 2027)
    assert len(report['differences']) == 6
    assert report['Z32_candidate_gate'] and report['Z32_dimension_gate']
    assert report['differences']['Z32-A0']['equal_k'] == pytest.approx(.04)
    with pytest.raises(ValueError, match='mismatched'):
        b.summarize(tmp_path, 'changed', 2027)
    (tmp_path / 'Z32' / 'complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        b.summarize(tmp_path, 'fixed', 2027)
