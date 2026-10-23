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
