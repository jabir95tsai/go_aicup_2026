"""Synthetic regression checks for board/time alignment and learned pooling."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from go_aicup.data.board_cache import pack_planes, unpack_planes
from go_aicup.features.sgf_planes import sgf_to_planes
from go_aicup.tasks.strength.board_attention.train import (
    Games, MoveWeights, PlayerSampler, atomic_save, before_move, forward, paired_ce, pool, unpack,
)


def test_before_move_capture_mover_channels_and_game_boundary():
    sgf = '(;SZ[19];B[ba];W[aa];B[pp];W[qq];B[ab];W[pd];B[aa])'
    after = sgf_to_planes(sgf).astype(np.uint8)
    colors = np.arange(len(after)) % 2
    before = before_move(pack_planes(after), colors)
    expected = np.concatenate((np.zeros_like(after[:1]), after[:-1]))
    expected[colors == 1] = expected[colors == 1, ::-1]
    np.testing.assert_array_equal(unpack_planes(before), expected)
    np.testing.assert_array_equal(unpack(torch.from_numpy(before)).numpy(), expected)
    assert after[4, 1, 0, 0] == 0  # white corner stone was captured
    assert before_move(pack_planes(after), colors)[0].sum() == 0
    assert before_move(pack_planes(after[:2]), colors[:2])[0].sum() == 0


def test_segment_softmax_mean_and_equal_game_side_ce():
    torch.manual_seed(2)
    slot = torch.tensor([0, 0, 0, 1, 2, 3, 3])
    lp = torch.randn(7, 10)
    values, weights = pool(torch.zeros(7), lp, slot, 4)
    for s in range(4):
        torch.testing.assert_close(weights[slot == s].sum(), torch.tensor(1.0))
        torch.testing.assert_close(values.reshape(4, 10)[s], lp[slot == s].mean(0))
    target = torch.tensor([1, 8])
    expected = torch.stack([F.cross_entropy(values[g, s][None], target[g:g+1])
                            for g in range(2) for s in range(2)]).mean()
    torch.testing.assert_close(paired_ce(values, target), expected)
    scores = torch.tensor([10000., 10001., 9999., -10000., 50000., -50000., -50001.], requires_grad=True)
    values, weights = pool(scores, lp, slot, 4)
    loss = paired_ce(values, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0


def test_model_uniform_start_and_gradients_reach_conv_after_update():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = MoveWeights()
    assert sum(p.numel() for p in model.parameters()) == 7955
    packed = torch.randint(0, 256, (12, 2, 46), dtype=torch.uint8)
    lp = torch.randn(12, 10)
    ply = torch.arange(12).float()
    slot = torch.arange(12) % 4
    target = torch.tensor([2, 6])
    torch.testing.assert_close(model(packed, lp, ply), torch.zeros(12))
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        values, _ = pool(model(packed, lp, ply), lp, slot, 4)
        paired_ce(values, target).backward()
        opt.step()
    assert model.conv[0].weight.grad.abs().sum() > 0
    assert torch.isfinite(model.conv[0].weight.grad).all()


def test_player_sampler_rank_balance_unique_players_and_resume():
    table = pd.DataFrame([{'rank_id':r, 'player_id':f'{r}_{p}', 'set':s}
                          for r in range(10) for p in range(5)
                          for s in ('train', 'train', 'hold', 'U')])
    sampler = PlayerSampler(table)
    rng = np.random.default_rng(2027)
    rows = sampler.sample(3, rng)
    selected = table.loc[rows]
    assert (selected['set'] == 'train').all()
    assert (selected.groupby('rank_id').size() == 3).all()
    assert (selected.groupby('rank_id').player_id.nunique() == 3).all()
    state = rng.bit_generator.state
    expected = sampler.sample(3, rng)
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    np.testing.assert_array_equal(sampler.sample(3, restored), expected)
    with pytest.raises(ValueError):
        sampler.sample(6, rng)


def test_checkpoint_reloads_model_and_normalisation(tmp_path):
    model = MoveWeights(torch.arange(13).float(), torch.ones(13) * 2)
    path = tmp_path / 'model.pt'
    atomic_save({'model':model.state_dict(), 'step':100}, path)
    loaded = torch.load(path, weights_only=False)
    restored = MoveWeights()
    restored.load_state_dict(loaded['model'])
    assert loaded['step'] == 100
    torch.testing.assert_close(restored.mean, model.mean)
    assert not path.with_suffix('.tmp').exists()


def test_ragged_batch_reorders_without_crossing_games():
    table = pd.DataFrame({'rank_id':[0, 1], 'set':['train', 'hold']})
    data = Games(table, torch.zeros(7, 2, 46, dtype=torch.uint8),
                 torch.arange(70).reshape(7, 10).half(),
                 torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8),
                 torch.tensor([0, 1, 2, 0, 1, 2, 3], dtype=torch.int16), np.array([0, 3, 7]))
    _, lp, ply, slots, targets = data.batch([1, 0])
    torch.testing.assert_close(lp[:, 0], torch.tensor([30, 40, 50, 60, 0, 10, 20]).float())
    assert slots.tolist() == [1, 0, 1, 0, 2, 3, 2]
    assert targets.tolist() == [1, 0]
    assert ply.tolist() == [0, 1, 2, 3, 0, 1, 2]


def test_b2_identical_features_give_identical_question_scores(monkeypatch):
    import go_aicup.tasks.strength.humansl as hs
    from go_aicup.data.cv import make_rank_questions
    from go_aicup.tasks.strength.board_attention.evaluate import fit_seed
    monkeypatch.setattr(hs, 'SYNTHETIC', 300)
    fit = pd.DataFrame([{'game_index':r*20+i, 'rank_id':r, 'player_id':f'fit_{r}', 'color':'B'}
                        for r in range(10) for i in range(20)])
    valid = fit.assign(player_id=fit.player_id.str.replace('fit_', 'u_'))
    q = make_rank_questions(valid, valid.player_id.unique(), 40, 2027)
    x = np.random.default_rng(7).normal(size=(200, 22))
    pair = fit_seed(2027, fit, valid, {'mean':(x,x), 'weighted':(x.copy(),x.copy())}, {'synthetic':q})
    np.testing.assert_array_equal(pair['mean']['synthetic'],pair['weighted']['synthetic'])
    assert pair['mean']['synthetic'].shape == (40,)
    assert ((pair['mean']['synthetic'] >= 0) & (pair['mean']['synthetic'] <= 1)).all()
