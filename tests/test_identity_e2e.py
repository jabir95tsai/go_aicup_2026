"""CPU checks of the end-to-end identity fine-tuning pieces (train_e2e)."""
import numpy as np
import torch

from go_aicup.tasks.identity.seqagg import Head
from go_aicup.tasks.identity.train_e2e import joint_games, mean_games


def test_joint_games_equals_head_per_game():
    torch.manual_seed(0)
    head = Head(attention=False).eval()
    n, m, s, d = 3, 2, 5, 256
    moves = torch.randn(n * m * s, d)
    plies = torch.randint(0, 400, (n * m * s,))
    with torch.no_grad():
        games = joint_games(head, moves, plies, (n, m, s))
        for g in range(n * m):
            x = moves[g * s:(g + 1) * s][None]
            one = head(x, plies[g * s:(g + 1) * s][None], torch.ones(1, s, dtype=torch.bool))[0]
            assert torch.allclose(games.reshape(n * m, d)[g], one, atol=1e-5)


def test_mean_games_matches_reshape_mean():
    moves = torch.arange(2 * 3 * 4 * 2, dtype=torch.float32).reshape(-1, 2)
    out = mean_games(moves, (2, 3, 4))
    assert out.shape == (2, 3, 2)
    assert torch.allclose(out[1, 2], moves[20:24].mean(0))


def test_joint_head_gradient_reaches_moves():
    head = Head(attention=False)
    moves = torch.randn(2 * 2 * 3, 256, requires_grad=True)
    games = joint_games(head, moves, torch.zeros(12, dtype=torch.long), (2, 2, 3))
    games.sum().backward()
    assert moves.grad is not None and float(moves.grad.abs().sum()) > 0
    assert np.isfinite(moves.grad.numpy()).all()


def test_restore_encoder_adam_preserves_step_and_new_lr():
    import copy
    from go_aicup.tasks.identity.train_e2e import restore_encoder_optimizer
    from go_aicup.tasks.strength.shared.training import scheduler_for
    model = torch.nn.Linear(3, 2)
    old = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(2, 3)).square().mean().backward()
    old.step()
    saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(old.state_dict()), "step": 1}
    head = torch.nn.Linear(2, 2)
    new = torch.optim.AdamW([*model.parameters(), *head.parameters()], lr=3e-5)
    assert restore_encoder_optimizer(new, model, saved) == 1
    assert new.param_groups[0]["lr"] == 3e-5
    for p in model.parameters():
        assert torch.equal(new.state[p]["exp_avg"], old.state[p]["exp_avg"])
        assert torch.equal(new.state[p]["exp_avg_sq"], old.state[p]["exp_avg_sq"])
        assert new.state[p]["step"] == 1
    assert all(p not in new.state for p in head.parameters())
    scheduler_for(new, 4800, 0.05, 0.0)
    assert abs(new.param_groups[0]["lr"] - 3e-5 / 240) < 1e-12
    new.zero_grad(set_to_none=True)
    head(model(torch.ones(2, 3))).sum().backward()
    new.step()
    assert all(new.state[p]["step"] == 2 for p in model.parameters())
    assert all(new.state[p]["step"] == 1 for p in head.parameters())


def test_restore_rejects_mismatched_weights():
    import copy
    import pytest
    from go_aicup.tasks.identity.train_e2e import restore_encoder_optimizer
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": opt.state_dict(), "step": 0}
    saved["model"]["weight"].add_(1)
    with pytest.raises(ValueError, match="source weights"):
        restore_encoder_optimizer(opt, model, saved)


def test_frozen_bn_keeps_buffers_but_trains_affine():
    from go_aicup.tasks.identity.train_e2e import training_mode
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 1))
    initial = {k: v.clone() for k, v in model.named_buffers()}
    model.eval()
    training_mode(model, True)
    assert model.training and model[1].training and not model[0].training
    model(torch.randn(4, 3)).square().mean().backward()
    assert model[0].weight.grad is not None
    assert all(torch.equal(v, initial[k]) for k, v in model.named_buffers())
    training_mode(model, False)
    model(torch.randn(4, 3))
    assert model[0].num_batches_tracked == 1


def test_bench_monitor_uses_fixed_fusion_and_buckets():
    import pandas as pd
    from go_aicup.tasks.identity.e2e_probe import BenchMonitor
    monitor = BenchMonitor.__new__(BenchMonitor)
    monitor.games = np.arange(9)
    monitor.w = 2.0
    candidates = np.array(["a", "b", "c", "d", "e"])
    q = pd.DataFrame({"games": [np.array([i]) for i in range(5, 9)],
                      "player_id": candidates[:4], "k": [1, 2, 3, 5]})
    other = np.zeros((4, 5))
    monitor.draws = [(candidates, [np.array([i]) for i in range(5)], q, other)]
    result = monitor.score_vectors({"B": np.concatenate([np.eye(5), np.eye(5)[:4]])})
    assert result["B_fused_all"] == 1.0
    assert all(result[f"B_fused_{k}"] == 1.0 for k in (1, 2, 3, 5))
    assert np.array_equal(monitor.scores["B_fused"], 32 * monitor.scores["B_style"])


def test_lr_selection_stops_without_gain_and_uses_only_dev():
    import pytest
    from scripts.identity.pid2_e2e_followup import select_arm
    def report(name, delta, first=100):
        return {'first_draw': first, 'base': 'base', 'tables': {'pool 0 | fused': [
            {'arm': name, 'draws': 20, 'd_all': delta}]}}
    reports = {a: report(a, d) for a, d in [('lr3e5', -0.01), ('lr1e5', 0.0)]}
    assert select_arm(reports)[0] is None
    reports['lr3e5'] = report('lr3e5', 0.01)
    assert select_arm(reports)[0] == 'lr3e5'
    reports['lr1e5'] = report('lr1e5', 0.01)
    assert select_arm(reports)[0] == 'lr1e5'
    reports['lr1e5']['first_draw'] = 600
    with pytest.raises(ValueError, match='development'):
        select_arm(reports)
