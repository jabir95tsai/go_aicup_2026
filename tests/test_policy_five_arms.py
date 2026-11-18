import json

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.policy_residual import five_arms as fa
from go_aicup.tasks.strength.policy_residual.data import Cache, question_rows
from go_aicup.tasks.strength.policy_residual.run import finalize_cache
from go_aicup.tasks.strength.policy_residual.train import fit_head


def _synthetic(per_role_games=20):
    rng = np.random.default_rng(0)
    rows, questions, qid = [], {'train': [], 'monitor': []}, 0
    index = 0
    for role in ('train', 'monitor'):
        for rank in range(10):
            player = f'{role}{rank}'
            ids = list(range(index, index + per_role_games))
            index += per_role_games
            for g in ids:
                rows.append(dict(game_index=g, player_id=player, rank_id=rank, player_color=rank % 2,
                                 role=role, length=int(rng.integers(4, 8))))
            for k in (5, 10, 15, 20):
                questions[role].append(dict(question_id=qid, player_id=player, rank_id=rank, k=k,
                                            games=ids[:k]))
                qid += 1
    games = pd.DataFrame(rows)
    bounds = np.r_[0, np.cumsum(games.length)].astype(np.int64)
    n = int(bounds[-1])
    logits = torch.randn(n, 10, 362)
    legal = torch.rand(n, 362) > .3
    legal[:, -1] = True
    target = torch.multinomial(legal.float(), 1).squeeze(1)
    target[::7] = 361  # some passes
    logits = logits + torch.arange(10)[None, :, None] * .05 * (torch.arange(362) % 3)[None, None]
    lp = logits.masked_fill(~legal[:, None], -torch.inf).log_softmax(-1).half().float()
    q = lp.gather(2, target[:, None, None].expand(-1, 10, 1)).squeeze(-1)
    side = torch.tensor(np.concatenate([np.arange(l) % 2 for l in games.length]), dtype=torch.uint8)
    arrays = {'logp': lp.half(), 'q': q.float(), 'target': target.short(), 'side': side,
              'done': torch.ones(len(games), dtype=torch.bool)}
    finalize_cache(games, bounds, arrays)
    return Cache(games, bounds, arrays, torch.device('cpu')), {r: pd.DataFrame(v) for r, v in questions.items()}


def test_cnn_shapes_and_parameter_counts():
    r10, z32 = fa.build('CNN-R10', 1), fa.build('CNN-Z32', 1)
    conv = sum(p.numel() for p in r10.conv.parameters())
    assert conv == 49_664
    assert sum(p.numel() for p in r10.parameters()) == 49_664 + 4_160 + 650
    assert sum(p.numel() for p in z32.parameters()) == 49_664 + 4_160 + 2_080
    # Same seed -> identical shared CNN / hidden bodies across the two CNN arms.
    for a, b in zip([*r10.conv.parameters(), *r10.hidden.parameters()],
                    [*z32.conv.parameters(), *z32.hidden.parameters()]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    s_r, s_z = fa.build('R10', 1), fa.build('Z32', 1)
    for name in ('phi', 'scorer'):
        for a, b in zip(getattr(s_r, name).parameters(), getattr(s_z, name).parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    n = 6
    lp = torch.randn(n, 10, 362).log_softmax(-1).half()
    target = torch.tensor([0, 5, 361, 100, 360, 361])
    q = lp.float().gather(2, target[:, None, None].expand(-1, 10, 1)).squeeze(-1)
    torch.testing.assert_close(r10(lp, target, q, 3), q, atol=0, rtol=0)
    out = z32(lp, target, q, 5)
    assert out.shape == (n, 42)
    torch.testing.assert_close(out[:, :10], q, atol=0, rtol=0)


def test_policy_image_pass_marker_empty():
    p = torch.rand(2, 10, 362)
    image = fa.policy_image(p, p.log(), torch.tensor([361, 3]))
    assert image.shape == (2, 30, 19, 19)
    assert image[0, 20:].sum() == 0
    assert image[1, 20:].sum() == 10


@pytest.mark.parametrize('arm', fa.CASES)
def test_run_arm_smoke_cpu(tmp_path, arm):
    torch.use_deterministic_algorithms(True)
    cache, questions = _synthetic()
    fa.validate_questions(cache.games, questions)
    rows = question_rows(cache.games, questions['train'])
    initial = fit_head(cache.arrays['c'].numpy(), questions['train'], rows)
    config = dict(seed=3, epochs=2, questions_per_k=1, lr=.001, weight_decay=1e-4, evaluate_every=5,
                  position_chunk=37)
    result = fa.run_arm(arm, cache, questions, initial, config, 'id', tmp_path)
    assert result['completed_updates'] == 20
    curve = json.loads((tmp_path / arm / 'learning_curve.json').read_text())
    assert [e['step'] for e in curve] == [0, 5, 10, 15, 20]
    # run_arm already asserts step-0 predictions equal the shared A0 head on raw C for every arm.
    if arm != 'A0':
        assert curve[-1]['train']['ce'] != curve[0]['train']['ce']


def test_resume_matches_uninterrupted(tmp_path):
    torch.use_deterministic_algorithms(True)
    cache, questions = _synthetic()
    rows = question_rows(cache.games, questions['train'])
    initial = fit_head(cache.arrays['c'].numpy(), questions['train'], rows)
    base = dict(seed=3, questions_per_k=1, lr=.001, weight_decay=1e-4, evaluate_every=5, position_chunk=37)
    full = fa.run_arm('CNN-Z32', cache, questions, initial, dict(base, epochs=2), 'id', tmp_path / 'a')
    fa.run_arm('CNN-Z32', cache, questions, initial, dict(base, epochs=1), 'id', tmp_path / 'b')
    (tmp_path / 'b' / 'CNN-Z32' / 'complete.json').unlink()
    resumed = fa.run_arm('CNN-Z32', cache, questions, initial, dict(base, epochs=2), 'id', tmp_path / 'b')
    assert resumed['roles']['monitor'] == full['roles']['monitor']
    assert resumed['selected_step'] == full['selected_step']
