"""Shared synthetic fixtures."""
import csv

import pytest
import torch

from go_aicup.data.board_cache import RANKS, convert_training
from go_aicup.tasks.strength.v2.config import read_config
from go_aicup.tasks.strength.v2.prepare import prepare


@pytest.fixture(scope="session")
def v2_train_data(tmp_path_factory):
    """Per rank 10 players x 40 black games: 5 folds, k=20 in both valid sets."""
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("v2_train")
    raw, cache = root / "raw", root / "cache"
    raw.mkdir()
    for rank_id, rank in enumerate(RANKS):
        with (raw / f"train_{rank}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("player_id", "game_id", "rank", "color", "sgf_content"))
            writer.writeheader()
            for player in range(10):
                for game in range(40):
                    x = chr(ord("b") + rank_id)
                    moves = [f"B[{x}{row}];W[{chr(ord('j') + player)}{white}]"
                             for row, white in zip("bcde", "srqp")][:2 + game % 3]
                    writer.writerow(dict(player_id=f"p{rank}_{player}", game_id=f"g{player}_{game}",
                                         rank=rank, color="B", sgf_content="(;SZ[19];" + ";".join(moves) + ")"))
    convert_training(raw, cache, workers=1, games_per_shard=400, ranks=RANKS, expected_games_per_rank=400)
    config = read_config()
    config.update(device="cpu", precision="fp32")
    config["paths"].update(cache_dir=str(cache), dataset_dir=str(root / "processed"),
                           output_dir=str(root / "run"))
    config["model"].update(channels=2, blocks=1, hidden_size=4)
    config["split"]["seen_holdout_fraction"] = 0.5
    config["questions"] = {"validation": 40}
    config["monitoring"]["games_per_rank"] = 5
    config["loader"].update(num_workers=0, persistent_workers=False, pin_memory=False)
    config["training"].update(global_batch_size=512, evaluate_every_steps=4, sync_batch_norm=False)
    config["evaluation"]["global_batch_size"] = 100
    prepare(config)
    return config
