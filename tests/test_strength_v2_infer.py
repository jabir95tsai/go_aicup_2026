"""V2 per-game aggregation, question pooling and whole-game DDP inference."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.data.board_cache import convert_training
from go_aicup.metrics import NEAR
from go_aicup.tasks.strength.shared.dataset import PackedBoardDataset
from go_aicup.tasks.strength.v2.aggregate import (
    build_category, decode_questions, game_predictions, monitor_metrics,
    official_scores, question_probabilities)
from go_aicup.tasks.strength.v2.config import read_config
from go_aicup.tasks.strength.v2.infer import infer_games
from go_aicup.tasks.strength.v2.model import StrengthResNet
from go_aicup.tasks.strength.v2.prepare import prepare


def _category(confidence, moves, ranks=None, log_sum=None):
    confidence = np.asarray(confidence, dtype=float)
    count = len(confidence)
    return build_category(np.arange(count) * 3, np.zeros(count) if ranks is None else ranks,
                          moves, confidence, np.zeros_like(confidence) if log_sum is None else log_sum)


def test_game_argmax_ties_scores_and_validation():
    confidence = np.zeros((3, 10))
    confidence[0, [2, 5]] = 3.0          # tie -> smaller rank
    confidence[1, 9] = 1.0
    confidence[2] = np.arange(10)
    np.testing.assert_array_equal(game_predictions(confidence), [2, 9, 9])
    category = _category(confidence, [3, 1, 4], ranks=[3, 8, 6])
    # Sum and per-game mean have the same argmax.
    np.testing.assert_array_equal(category["predicted_rank_id"],
                                  game_predictions(confidence / np.array([3, 1, 4])[:, None]))
    np.testing.assert_allclose(official_scores(category["predicted_rank_id"], category["rank_id"]),
                               [NEAR, NEAR, 0.0])
    with pytest.raises(ValueError, match="sorted"):
        build_category([3, 1], [0, 0], [1, 1], np.zeros((2, 10)), np.zeros((2, 10)))
    with pytest.raises(ValueError, match="at least one"):
        _category(np.zeros((1, 10)), [0])
    with pytest.raises(ValueError, match="finite"):
        _category(np.full((1, 10), np.nan), [1])


def test_monitor_loss_is_move_weighted_from_log_sums():
    log_p = np.log(np.full(10, 0.1))
    log_sum = np.vstack([log_p * 2, np.log(np.r_[0.91, np.full(9, 0.01)]) * 1])
    category = _category(np.exp(log_sum), [2, 1], ranks=[4, 0], log_sum=log_sum)
    metrics = monitor_metrics(category)
    assert metrics["loss"] == pytest.approx((2 * math.log(10) - math.log(0.91)) / 3)
    assert metrics["moves"] == 3 and metrics["games"] == 2


def test_questions_weight_games_equally_and_decode_edges():
    # A long confident 0 game and a short confident 9 game: equal weight.
    confidence = np.zeros((2, 10))
    confidence[0, 0] = 100.0
    confidence[1, 9] = 1.0
    category = _category(confidence, [100, 1])
    proba = question_probabilities(category, [np.array([0, 3])])
    np.testing.assert_allclose(proba, [[0.5] + [0] * 8 + [0.5]])
    np.testing.assert_allclose(proba.sum(1), 1.0)
    # score_rank favours ranks with mass on both neighbours; argmax does not.
    middle = np.array([[0.0, 0.3, 0.25, 0.3, 0.15, 0, 0, 0, 0, 0]])
    assert decode_questions(middle, "argmax")[0] == 1
    assert decode_questions(middle, "score_rank")[0] == 2
    # Edge rank D has one neighbour: 0.38 + 0.36/e < 0.36 + (0.38 + 0.26)/e.
    edge = np.array([[0.38, 0.36, 0.26, 0, 0, 0, 0, 0, 0, 0]])
    assert decode_questions(edge, "argmax")[0] == 0
    assert decode_questions(edge, "score_rank")[0] == 1
    top = np.zeros((1, 10))
    top[0, 9] = 1.0
    assert decode_questions(top, "score_rank")[0] == 9
    with pytest.raises(ValueError, match="without a saved result"):
        question_probabilities(category, [np.array([0, 1])])


def test_log_probability_pooling_is_normalized_geometric_mean():
    p = np.array([[0.5, 0.5] + [0] * 8, [0.9, 0.1] + [0] * 8]) + 1e-12
    p /= p.sum(1, keepdims=True)
    category = _category(p * [[2], [3]], [2, 3], log_sum=np.log(p) * [[2], [3]])
    pooled = question_probabilities(category, [np.array([0, 3])], "mean_log_probability")
    expected = np.sqrt(p[0] * p[1])
    np.testing.assert_allclose(pooled[0], expected / expected.sum(), rtol=1e-9)
    with pytest.raises(ValueError, match="unknown aggregation"):
        question_probabilities(category, [np.array([0])], "vote")


@pytest.fixture(scope="module")
def v2_data(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("v2_infer")
    raw, cache = root / "raw", root / "cache"
    raw.mkdir()
    ranks = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
    for rank_id, rank in enumerate(ranks):
        with (raw / f"train_{rank}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("player_id", "game_id", "rank", "color", "sgf_content"))
            writer.writeheader()
            for game in range(40):
                x = chr(ord("b") + rank_id)
                # 2..4 labelled-player moves, so games differ in length.
                moves = [f"B[{x}{row}];W[s{white}]" for row, white in zip("bcde", "srqp")][:2 + game % 3]
                writer.writerow(dict(player_id=f"p{rank}", game_id=f"g{game}", rank=rank,
                                     color="B", sgf_content="(;SZ[19];" + ";".join(moves) + ")"))
    convert_training(raw, cache, workers=1, games_per_shard=40, ranks=ranks, expected_games_per_rank=40)
    config = read_config()
    config.update(device="cpu", precision="fp32")
    config["paths"].update(cache_dir=str(cache), dataset_dir=str(root / "processed"),
                           output_dir=str(root / "run"))
    config["model"].update(channels=2, blocks=1, hidden_size=4)
    # Inference is split-agnostic; one player per rank cannot fill 5 folds.
    config["split"] = {"expected_games_per_rank": 40, "train_games_per_rank": 20}
    config["questions"] = {"validation": 40}
    config["monitoring"]["games_per_rank"] = 5
    config["loader"].update(num_workers=0, persistent_workers=False, pin_memory=False)
    config["training"]["sync_batch_norm"] = False
    config["evaluation"]["global_batch_size"] = 7  # Games straddle batches.
    prepare(config)
    return config


def _clone(config):
    return json.loads(json.dumps(config))


def _trained_like_model(config):
    torch.manual_seed(2027)
    model = StrengthResNet(**config["model"])
    with torch.no_grad():  # Non-trivial BN statistics and logits.
        for name, value in model.named_buffers():
            if name.endswith("running_mean"):
                value.uniform_(-0.5, 0.5)
            elif name.endswith("running_var"):
                value.uniform_(0.5, 2.0)
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.3)
    return model.train()


def test_whole_game_inference_matches_direct_forward(v2_data):
    config = _clone(v2_data)
    model = _trained_like_model(config)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    category = infer_games(model, config, "monitor", torch.device("cpu"))
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    directory = Path(config["paths"]["dataset_dir"])
    games = np.load(directory / "monitor_game.npy")
    np.testing.assert_array_equal(category["game_index"], np.load(directory / "monitor_games.npy"))
    np.testing.assert_array_equal(category["num_moves"], np.bincount(games)[category["game_index"]])
    meta = pd.read_parquet(directory / "strength_v1_games.parquet").set_index("game_index")
    np.testing.assert_array_equal(category["rank_id"], meta.loc[category["game_index"], "rank_id"])
    dataset = PackedBoardDataset(directory / "monitor_game.npy", directory / "monitor_ply.npy",
                                 Path(config["paths"]["cache_dir"]))
    model.eval()
    with torch.no_grad():
        planes = torch.from_numpy(np.stack([dataset[i]["planes"] for i in range(len(dataset))])).float()
        log_p = torch.log_softmax(model(planes), -1).double().numpy()
    dataset.close()
    rows = np.searchsorted(category["game_index"], games)
    expected_log = np.zeros((len(category), 10))
    np.add.at(expected_log, rows, log_p)
    expected = np.zeros((len(category), 10))
    np.add.at(expected, rows, np.exp(log_p))
    np.testing.assert_allclose(category["log_probability_sum"], expected_log, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(category["confidence_sum"], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(category["confidence_sum"].sum(1), category["num_moves"], rtol=1e-6)
    np.testing.assert_array_equal(category["predicted_rank_id"], expected.argmax(1))
    labels = category["rank_id"].astype(int)
    assert monitor_metrics(category)["loss"] == pytest.approx(
        -expected_log[np.arange(len(labels)), labels].sum() / len(games), rel=1e-6)


def _distributed_worker(rank, config, state, rendezvous, output):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = StrengthResNet(**config["model"])
        model.load_state_dict(state)
        category = infer_games(model, config, "valid", torch.device("cpu"), rank, 2)
        np.save(f"{output}.{rank}.npy", category, allow_pickle=False)
    finally:
        dist.destroy_process_group()


def test_two_process_inference_matches_serial_with_empty_tail_rank(v2_data, tmp_path):
    config = _clone(v2_data)
    source = Path(config["paths"]["dataset_dir"])
    config["paths"]["dataset_dir"] = str(tmp_path)
    games, plies = (np.load(source / f"valid_{kind}.npy") for kind in ("game", "ply"))
    # Batch 4 over two processes; odd length leaves rank one an empty tail.
    keep = len(games) - (len(games) - 1) % 4
    np.save(tmp_path / "valid_game.npy", games[:keep])
    np.save(tmp_path / "valid_ply.npy", plies[:keep])
    assert keep % 4 == 1
    config["evaluation"]["global_batch_size"] = 4
    model = _trained_like_model(config)
    serial = infer_games(model, config, "valid", torch.device("cpu"))
    output = str(tmp_path / "parallel")
    torch.multiprocessing.spawn(_distributed_worker,
        args=(config, model.state_dict(), (tmp_path / "gloo_init").as_uri(), output),
        nprocs=2, join=True)
    for rank in range(2):
        parallel = np.load(f"{output}.{rank}.npy", allow_pickle=False)
        for field in ("game_index", "rank_id", "num_moves", "predicted_rank_id"):
            np.testing.assert_array_equal(parallel[field], serial[field])
        for field in ("confidence_sum", "log_probability_sum"):
            np.testing.assert_allclose(parallel[field], serial[field], rtol=1e-6, atol=1e-6)


def test_permutation_matches_legacy_memmap_shuffle(tmp_path):
    from go_aicup.data.board_cache import _close_mmap
    from go_aicup.tasks.strength.shared.dataset import create_permutation
    length, seed, epoch = 2_345_678, 2027, 0
    legacy = np.lib.format.open_memmap(tmp_path / "legacy.npy", mode="w+", dtype=np.uint32, shape=(length,))
    for start in range(0, length, 1_000_000):
        end = min(start + 1_000_000, length)
        legacy[start:end] = np.arange(start, end, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch]))).shuffle(legacy)
    expected = np.array(legacy)
    _close_mmap(legacy)
    path = create_permutation(tmp_path / "permutations", length, seed, epoch)
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)
    assert create_permutation(tmp_path / "permutations", length, seed, epoch) == path
