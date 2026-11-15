"""GE2E insurance encoder: loss, stateless batches, trunk loading."""
import numpy as np
import pandas as pd
import torch

from go_aicup.data.board_cache import write_store_cache
from go_aicup.tasks.identity.ge2e import MIN_PLY, GE2EBatches, IdentityEncoder, ge2e_loss
from go_aicup.tasks.strength.v2.model import StrengthResNet


def test_ge2e_loss_prefers_separated_players():
    torch.manual_seed(0)
    centers = torch.nn.functional.normalize(torch.randn(4, 1, 16), dim=-1)
    tight = centers + 0.01 * torch.randn(4, 5, 16)
    loose = torch.randn(4, 5, 16)
    w, b = torch.tensor(10.0), torch.tensor(-5.0)
    tight_loss, tight_acc = ge2e_loss(tight, w, b)
    loose_loss, _ = ge2e_loss(loose, w, b)
    assert tight_loss < loose_loss and tight_acc == 1.0
    games = loose.clone().requires_grad_(True)
    ge2e_loss(games, w, b)[0].backward()
    assert torch.isfinite(games.grad).all()


def test_batches_are_stateless_black_and_past_the_opening(tmp_path):
    rng = np.random.default_rng(0)
    games = [rng.choice(361, size=60, replace=False) for _ in range(6)]
    moves = np.concatenate(games).astype(np.int16)
    colors = np.concatenate([np.arange(60) % 2 for _ in games]).astype(np.int8)
    meta = pd.DataFrame({"offset": np.arange(6) * 60, "length": 60, "color": "B"})
    write_store_cache(moves, colors, meta, tmp_path / "cache")
    player_games = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    batches = GE2EBatches(tmp_path / "cache", player_games, np.full(6, 60), players=2, games=2, moves=3,
                          steps=5, seed=1, rank=0)
    game_index, plies = batches.sample(3)
    assert len(plies) == 12 and (plies >= MIN_PLY).all() and (plies % 2 == 0).all()
    again = GE2EBatches(tmp_path / "cache", player_games, np.full(6, 60), players=2, games=2, moves=3,
                        steps=5, seed=1, rank=0, start_step=3)
    np.testing.assert_array_equal(again[0].numpy(), batches[3].numpy())
    assert batches[0].shape == (12, 18, 19, 19)
    assert (batches[0][:, 16] == 1).all()          # colour plane: black to move


def test_encoder_loads_v2_trunk():
    source = StrengthResNet(channels=8, blocks=2)
    encoder = IdentityEncoder(channels=8, blocks=2, embedding=4)
    encoder.load_trunk(source.state_dict())
    torch.testing.assert_close(encoder.stem[0].weight, source.stem[0].weight)
    assert encoder(torch.zeros(3, 18, 19, 19)).shape == (3, 4)


def test_extra_players_fill_a_fixed_share_of_every_group(tmp_path):
    rng = np.random.default_rng(0)
    games = [rng.choice(361, size=60, replace=False) for _ in range(12)]
    moves = np.concatenate(games).astype(np.int16)
    colors = np.concatenate([np.arange(60) % 2 for _ in games]).astype(np.int8)
    meta = pd.DataFrame({"offset": np.arange(12) * 60, "length": 60, "color": "B"})
    write_store_cache(moves, colors, meta, tmp_path / "cache")
    ordinary = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    extra = [np.array([6, 7]), np.array([8, 9]), np.array([10, 11])]
    batches = GE2EBatches(tmp_path / "cache", ordinary, np.full(12, 60), players=3, games=2, moves=2,
                          steps=20, seed=1, rank=0, extra_games=extra, extra_per_group=1)
    for step in range(20):
        game_index, _ = batches.sample(step)
        owners = game_index.reshape(3, 4)
        assert (owners[:2] < 6).all() and (owners[2] >= 6).all()
        assert all(len(set(row // 2)) == 1 for row in owners)        # one player per block
    plain = GE2EBatches(tmp_path / "cache", ordinary, np.full(12, 60), players=2, games=2, moves=2,
                        steps=5, seed=1, rank=0)
    unchanged = GE2EBatches(tmp_path / "cache", ordinary, np.full(12, 60), players=2, games=2, moves=2,
                            steps=5, seed=1, rank=0, extra_games=extra, extra_per_group=0)
    np.testing.assert_array_equal(plain.sample(2)[0], unchanged.sample(2)[0])


def test_stratified_subset_is_nested_and_balanced():
    from go_aicup.tasks.identity.train_ge2e import stratified_subset
    rng = np.random.default_rng(0)
    players = np.array([f"p{i:04d}" for i in range(900)])
    ranks = np.repeat(np.array(["A", "B", "C"]), 300)
    counts = rng.integers(8, 200, size=900)
    half = stratified_subset(players, ranks, counts, 0.5, 7)
    most = stratified_subset(players, ranks, counts, 0.82, 7)
    assert (most | ~half).all() and half.sum() == 450 and most.sum() == 246 * 3
    assert all(most[ranks == r].sum() == 246 for r in "ABC")
    np.testing.assert_array_equal(most, stratified_subset(players, ranks, counts, 0.82, 7))


def test_fixed_gallery_draws_never_ask_a_gallery_game():
    from go_aicup.data import cv
    rng = np.random.default_rng(0)
    rows = [{"pool": p, "player_id": f"{p}-{i:03d}", "game": 100000 * p + 100 * i + g, "order": g}
            for p in (0, 1) for i in range(40) for g in range(40)]
    bank = pd.DataFrame(rows)
    candidates, gallery, rest = cv.pid2_fixed_gallery(bank, 1, 0)
    shown = set(np.concatenate(gallery).tolist())
    assert len(candidates) == 40 and all(len(g) == cv.PID2_GALLERY for g in gallery)
    assert not shown & set(np.concatenate(rest).tolist())
    asked = set()
    for draw in range(5):
        c2, g2, q = cv.draw_pid2_fixed(bank, 1, 0, draw)
        assert all((a == b).all() for a, b in zip(gallery, g2)) and (c2 == candidates).all()
        games = np.concatenate(q["games"].to_list())
        assert not shown & set(games.tolist())
        owner = {c: set(r.tolist()) for c, r in zip(candidates, rest)}
        assert all(set(g.tolist()) <= owner[p] for p, g in zip(q["player_id"], q["games"]))
        asked.add(tuple(sorted(games.tolist())))
    assert len(asked) == 5 and sorted(q["k"].value_counts().to_dict().items()) == [(1, 100), (2, 100), (3, 100), (5, 100)]
    other = cv.pid2_fixed_gallery(bank, 1, 1)[1]
    assert any((a != b).any() for a, b in zip(gallery, other))


def _toy_cache(tmp_path, n_games=24, length=60):
    rng = np.random.default_rng(0)
    games = [rng.choice(361, size=length, replace=False) for _ in range(n_games)]
    moves = np.concatenate(games).astype(np.int16)
    colors = np.concatenate([np.arange(length) % 2 for _ in games]).astype(np.int8)
    meta = pd.DataFrame({"offset": np.arange(n_games) * length, "length": length, "color": "B"})
    write_store_cache(moves, colors, meta, tmp_path / "cache")
    return games


def test_rank_focus_draws_half_a_group_from_the_centre_band(tmp_path):
    _toy_cache(tmp_path)
    player_games = [np.array([2 * i, 2 * i + 1]) for i in range(12)]
    ranks = np.repeat(np.arange(6), 2)                       # two players per rank, ranks 0..5
    kwargs = dict(players=4, games=2, moves=3, steps=40, seed=3, rank=0, min_ply=0)
    plain = GE2EBatches(tmp_path / "cache", player_games, np.full(24, 60), **kwargs)
    focus = GE2EBatches(tmp_path / "cache", player_games, np.full(24, 60), player_rank=ranks,
                        rank_focus={"share": 0.5, "radius": 1}, **kwargs)
    for step in range(40):
        rng = np.random.default_rng([3, 0, step])
        centre = ranks[np.random.default_rng([3, 0, step]).integers(12)]
        chosen = focus.focused_players(rng)
        assert len(set(chosen.tolist())) == 4
        assert (np.abs(ranks[chosen[:2]] - centre) <= 1).all()   # edge ranks keep valid ranks only
    exposure = focus.rank_exposure(world=2)
    assert sum(exposure["slots"]) == 2 * 40 * 4 and sum(exposure["centre"]) == 2 * 40
    # the default sampler is untouched: same players as a direct draw
    game_index, _ = plain.sample(5)
    direct = np.random.default_rng([3, 0, 5]).choice(12, size=4, replace=False)
    assert [int(g) // 2 for g in game_index[::6]] == direct.tolist()
    np.testing.assert_array_equal(focus[7].numpy(), GE2EBatches(
        tmp_path / "cache", player_games, np.full(24, 60), player_rank=ranks,
        rank_focus={"share": 0.5, "radius": 1}, **{**kwargs, "start_step": 7})[0].numpy())


def test_pick_plane_marks_the_played_point_and_follows_d4(tmp_path):
    games = _toy_cache(tmp_path)
    player_games = [np.array([2 * i, 2 * i + 1]) for i in range(12)]
    kwargs = dict(players=4, games=2, moves=3, steps=5, seed=1, rank=0, min_ply=0,
                  player_color=np.zeros(24, np.int8))
    plain = GE2EBatches(tmp_path / "cache", player_games, np.full(24, 60), pick=True, **kwargs)
    game_index, plies = plain.sample(2)
    x = plain[2].numpy()
    assert x.shape == (24, 19, 19, 19)
    for row, (g, p) in enumerate(zip(game_index, plies)):
        point = int(games[g][p])
        assert x[row, 18].sum() == 1 and x[row, 18].reshape(-1)[point] == 1
        column = x[row, :16].reshape(16, -1)[:, point]            # the newest board holds the move
        assert column[0] == 1 and column[1:].sum() == 0
    turned = GE2EBatches(tmp_path / "cache", player_games, np.full(24, 60), pick=True, player_d4=True, **kwargs)
    y = turned[2].numpy()
    np.testing.assert_array_equal(y[:, :18], GE2EBatches(tmp_path / "cache", player_games, np.full(24, 60),
                                                         player_d4=True, **kwargs)[2].numpy())
    # the mark turns with the board: it still sits on the stone just played
    assert (y[:, 18].reshape(24, -1).sum(axis=1) == 1).all()
    assert ((y[:, 0] * y[:, 18]).reshape(24, -1).sum(axis=1) == 1).all()
    assert ((y[:, 2:16].sum(axis=1) * y[:, 18]).reshape(24, -1).sum(axis=1) == 0).all()
    assert not np.array_equal(x[:, 18], y[:, 18])


def test_pick_encoder_reads_the_marked_point():
    torch.manual_seed(0)
    model = IdentityEncoder(channels=8, blocks=1, embedding=4, pick=True).eval()
    x = torch.zeros(2, 19, 19, 19)
    x[:, :16] = (torch.rand(1, 16, 19, 19) < 0.3).float()    # the same board twice, two marked points
    x[:, 16] = 1
    x[0, 18, 3, 3] = x[1, 18, 10, 12] = 1
    a = model(x)
    assert a.shape == (2, 4) and not torch.allclose(a[0], a[1])
    plain = IdentityEncoder(channels=8, blocks=1, embedding=4)
    assert plain(x[:, :18]).shape == (2, 4) and plain.project[0].in_features == 16
    assert model.project[0].in_features == 24
