import copy

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.data.cv import make_policy_residual_games, make_rank_questions
from go_aicup.tasks.strength.policy_residual.data import Cache, question_rows
from go_aicup.tasks.strength.policy_residual.model import (
    ARMS, Head, Residual, candidate_inputs, policy_image, probabilities, question_mean, statistics,
)
from go_aicup.tasks.strength.policy_residual.run import finalize_cache
from go_aicup.tasks.strength.policy_residual.train import backward_questions, fit_head, game_features


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def policies(n=8):
    torch.manual_seed(17)
    logits = torch.randn(n, 10, 362)
    logits[:, :, 50:80] = -torch.inf
    lp = logits.log_softmax(-1)
    target = torch.arange(n) % 40
    target[-1] = 361
    q = lp.gather(2, target[:, None, None].expand(-1, 10, 1)).squeeze(-1)
    return lp, target, q


@pytest.mark.parametrize('arm', ARMS)
def test_zero_residual_and_finite_gradients(arm):
    lp, target, q = policies()
    model = Residual(arm)
    torch.testing.assert_close(model(lp, target, q), q, atol=0, rtol=0)
    if arm != 'A0':
        model(lp, target, q).square().mean().backward()
        assert model.delta[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_mask_removes_other_probabilities_without_renormalising_q():
    lp, target, q = policies()
    model = Residual('F-mask')
    torch.nn.init.normal_(model.delta[-1].weight, std=.02)
    perturbed = lp.clone()
    perturbed[:, :, 100:200] += 5
    torch.testing.assert_close(model(lp, target, q), model(perturbed, target, q), atol=0, rtol=0)
    p, safe, legal = probabilities(lp)
    x = candidate_inputs(p, safe, legal, target, True, q)
    keep = torch.ones(len(lp), 362, dtype=torch.bool)
    keep[torch.arange(len(lp)), target] = False
    assert x[..., :20][keep].eq(0).all()
    assert x[..., -1].sum() == len(lp)
    torch.testing.assert_close(x[torch.arange(len(lp)), target, 10:20], q)


def test_policy_image_uses_output_maps_and_moves_marker_with_d4():
    lp, target, q = policies()
    p, safe, legal = probabilities(lp)
    original = policy_image(p, safe, target)
    rotated = policy_image(p, safe, target, 1)
    assert original.shape == (len(lp), 30, 19, 19)
    torch.testing.assert_close(rotated, torch.rot90(original, 1, (-2, -1)))
    assert original[-1, 20:].eq(0).all()
    assert original[0, 20:, 0, 0].eq(1).all()
    stats = statistics(p, safe, legal, q)
    assert stats.shape == (len(lp), 70) and torch.isfinite(stats).all()


def small_cache():
    n, per_game = 18, 6
    lp, target, q = policies(n)
    games = pd.DataFrame({'game_index': [4, 9, 12], 'length': [per_game] * 3})
    arrays = {'logp': lp.half(), 'q': q, 'target': target.short(),
              'side': torch.tensor([0, 1, 1, 0, 1, 0] * 3, dtype=torch.uint8),
              'done': torch.ones(3, dtype=torch.bool)}
    bounds = np.arange(4) * per_game
    finalize_cache(games, bounds, arrays)
    return Cache(games, bounds, arrays, torch.device('cpu'))


@pytest.mark.parametrize('arm', ['S', 'F', 'F-mask', 'C-conv'])
def test_chunked_backward_matches_direct_question_gradient(arm):
    cache = small_cache()
    model, head = Residual(arm), Head()
    torch.nn.init.normal_(model.delta[-1].weight, std=.002)
    direct_model, direct_head = copy.deepcopy(model), copy.deepcopy(head)
    rows, y = [np.array([0, 1]), np.array([1, 2, 0])], torch.tensor([2, 5])
    backward_questions(model, head, cache, rows, y, 100, 4)
    lp, target, q, slot = next(cache.chunks(np.arange(3), 100))
    evidence = direct_model(lp, target, q)
    c = torch.zeros(6, 10).index_add(0, slot, evidence) / cache.counts(np.arange(3))[:, None]
    logits = direct_head(question_mean(c.reshape(3, 20), rows))
    loss = torch.nn.functional.cross_entropy(logits, y) + direct_head.linear.weight.square().sum() / 200
    loss.backward()
    for a, b in zip(list(model.parameters()) + list(head.parameters()),
                    list(direct_model.parameters()) + list(direct_head.parameters())):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-4)


def test_exact_side_and_question_means_and_precision_guard():
    cache = small_cache()
    c = game_features(Residual('A0'), cache, np.arange(3), 4)
    for g in range(3):
        for side in (0, 1):
            span = slice(cache.bounds[g], cache.bounds[g + 1])
            want = cache.arrays['q'][span][cache.arrays['side'][span].eq(side)].mean(0)
            torch.testing.assert_close(c[g, side * 10:(side + 1) * 10], want)
    cache.arrays['q'] += 1
    with pytest.raises(ValueError, match='precision'):
        finalize_cache(cache.games, cache.bounds, cache.arrays)


def population():
    records, halves, dev = [], [], []
    for r in range(10):
        for half in ('A', 'B'):
            for j in range(2):
                pid = f'{r}-{half}-{j}'
                halves.append({'player_id': pid, 'half': half})
                for color in (0, 1):
                    for i in range(20):
                        records.append({'game_index': len(records), 'player_id': pid, 'rank_id': r,
                                        'player_color': color, 'handicap': i % 2, 'length': 101 + i})
        for i in range(20):
            row = {'game_index': len(records), 'player_id': f'dev-{r}', 'rank_id': r,
                   'player_color': 0, 'handicap': 0, 'length': 100}
            records.append(row), dev.append(row)
    return pd.DataFrame(records), pd.DataFrame(halves), pd.DataFrame(dev)


def test_sampling_preserves_identities_and_all_k_eligibility():
    meta, halves, dev = population()
    games = make_policy_residual_games(meta, halves, dev, train_per_half_rank=1, monitor_per_half_rank=1)
    assert games.groupby('player_id').role.nunique().eq(1).all()
    assert games.groupby('player_id').size().eq(20).all()
    pd.testing.assert_frame_equal(games, make_policy_residual_games(
        meta, halves, dev, train_per_half_rank=1, monitor_per_half_rank=1))
    for role in ('train', 'monitor', 'dev'):
        part = games[games.role.eq(role)].copy()
        part['color'] = part.player_color.map({0: 'B', 1: 'W'})
        q = make_rank_questions(part.set_index('game_index'), part.player_id.unique(), 80)
        rows = question_rows(games, q)
        assert len(q) == 80
        for item, chosen in zip(q.itertuples(), rows):
            assert len(set(chosen)) == item.k
            assert games.iloc[chosen].role.eq(role).all()
    halves.loc[0, 'player_id'] = dev.player_id.iloc[0]
    with pytest.raises(ValueError, match='overlap'):
        make_policy_residual_games(meta, halves, dev, train_per_half_rank=1, monitor_per_half_rank=1)


def test_shared_head_matches_sklearn_and_initial_state_can_reload():
    rng = np.random.default_rng(10)
    c = rng.normal(size=(100, 20)).astype('float32')
    questions = pd.DataFrame({'rank_id': np.repeat(np.arange(10), 10)})
    rows = [np.array([i]) for i in range(100)]
    initial = fit_head(c, questions, rows)
    a, b = Head(), Head()
    a.load_state_dict(initial), b.load_state_dict(initial)
    torch.testing.assert_close(a(torch.tensor(c)), b(torch.tensor(c)), atol=0, rtol=0)


@pytest.mark.parametrize('arm', ['A0', 'S', 'F'])
def test_frozen_training_diagnostic_matches_loss_and_does_not_update(tmp_path, arm):
    from go_aicup.tasks.strength.policy_residual.evaluate_train import evaluate_checkpoint
    from go_aicup.tasks.strength.policy_residual.train import state
    cache = small_cache()
    model, head = Residual(arm), Head()
    if arm != 'A0':
        torch.nn.init.normal_(model.delta[-1].weight, std=.002)
    checkpoint = tmp_path / 'model.pt'
    torch.save(state(model, head, 'fixed-source', 100), checkpoint)
    before = checkpoint.read_bytes()
    rows = [np.array([0]), np.array([0, 1]), np.array([0, 1, 2]), np.array([1, 2])]
    questions = pd.DataFrame({'question_id': list('abcd'), 'player_id': ['p'] * 4,
                              'rank_id': [2, 5, 3, 0], 'k': [5, 10, 15, 20]})
    report, features, predictions = evaluate_checkpoint(
        checkpoint, arm, cache, questions, rows, 'fixed-source', 4)
    direct = game_features(model, cache, np.arange(3), 4)
    expected = torch.nn.functional.cross_entropy(
        head(question_mean(direct, rows)), torch.tensor(questions.rank_id.to_numpy()))
    penalty = head.linear.weight.square().sum() / (2 * len(questions))
    assert report['step'] == 100
    assert report['ce'] == pytest.approx(float(expected.detach()))
    assert report['objective'] == pytest.approx(float((expected + penalty).detach()))
    np.testing.assert_allclose(features, direct.detach().numpy())
    assert len(predictions) == 4 and set(report['by_k']) == {'5', '10', '15', '20'}
    assert checkpoint.read_bytes() == before
    with pytest.raises(ValueError, match='identity'):
        evaluate_checkpoint(checkpoint, arm, cache, questions, rows, 'wrong-source', 4)


def test_training_diagnostic_subset_keeps_equal_k_and_uses_fixed_mask():
    from go_aicup.tasks.strength.policy_residual.summarize_train import subset_report
    frame = pd.DataFrame({'k': [5, 5, 10, 15, 20], 'rank_id': [0] * 5,
                          'player_id': list('aabcd'), 'score': [1., 0., .5, .25, .75]})
    for r in range(10):
        frame[f'p{r}'] = .4 if r == 0 else .6 / 9
    all_rows = subset_report(frame, np.ones(5, dtype=bool), .02)
    assert all_rows['equal_k'] == pytest.approx(.5)
    assert all_rows['ce'] == pytest.approx(-np.log(.4))
    assert all_rows['objective'] == pytest.approx(-np.log(.4) + .02)
    selected = subset_report(frame, np.array([True, False, True, True, True]), .02)
    assert selected['equal_k'] == pytest.approx(.625)
    assert selected['questions'] == 4
    with pytest.raises(ValueError, match='misses a k'):
        subset_report(frame, np.array([True, True, False, True, True]), .02)
