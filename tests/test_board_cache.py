"""Synthetic-only checks: no official CSV access or full-data preprocessing."""

import csv
import json
import pickle

import numpy as np
import pandas as pd
import pytest

from go_aicup.data.board_cache import (BoardCache, convert_training, pack_planes,
                                      unpack_planes, verify_cache)
from go_aicup.features.sgf_planes import sgf_to_planes
from go_aicup.tasks.strength.dataset import PackedBoardDataset


GAMES = [
    "(;SZ[19];B[ba];W[aa];B[pp];W[qq];B[ab];W[pd];B[aa])",
    "(;SZ[19];B[pd];W[dp];B[pp];W[dd])",
    "(;SZ[19];B[ss];W[aa];B[dd])",
]


def write_csv(directory, rank="D", games=GAMES):
    directory.mkdir(exist_ok=True)
    with (directory / f"train_{rank}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["player_id", "game_id", "rank", "color", "sgf_content"])
        writer.writeheader()
        for index, sgf in enumerate(games):
            writer.writerow(dict(player_id=f"{rank}_p{index}", game_id=f"g{index}",
                                 rank=rank, color="W" if index == 1 else "B", sgf_content=sgf))


@pytest.fixture(scope="module")
def synthetic_cache(tmp_path_factory):
    root = tmp_path_factory.mktemp("packed_boards")
    raw, output = root / "raw", root / "cache"
    write_csv(raw)
    write_csv(raw, rank="C")
    convert_training(raw, output, workers=2, games_per_shard=2, ranks=("D", "C"),
                     expected_games_per_rank=3)
    return raw, output


def test_codec_binary_layout_and_batch_roundtrip():
    planes = np.random.default_rng(2026).integers(0, 2, (4, 8, 2, 19, 19), dtype=np.uint8)
    packed = pack_planes(planes)
    assert packed.shape == (4, 8, 2, 46)
    assert packed.dtype == np.uint8
    assert not (packed[..., -1] & 254).any()
    np.testing.assert_array_equal(unpack_planes(packed), planes)
    single = np.zeros((2, 19, 19), np.uint8)
    single[0, 0, 0] = single[0, 0, 7] = single[1, 18, 18] = 1
    packed = pack_planes(single)
    assert packed[0, 0] == 129  # little bit order
    assert packed[1, -1] == 1  # final point, then seven zero padding bits
    assert pack_planes(np.empty((0, 2, 19, 19), np.uint8)).shape == (0, 2, 46)
    assert unpack_planes(np.empty((0, 2, 46), np.uint8)).shape == (0, 2, 19, 19)


def test_codec_rejects_wrong_values_shapes_and_ignores_padding():
    with pytest.raises(ValueError):
        pack_planes(np.full((1, 2, 19, 19), 0.5))
    with pytest.raises(ValueError):
        pack_planes(np.zeros((1, 19, 19)))
    with pytest.raises(ValueError):
        unpack_planes(np.zeros((2, 46), np.float32))
    corrupt = np.zeros((2, 46), np.uint8)
    corrupt[0, -1] = 2
    np.testing.assert_array_equal(unpack_planes(corrupt), np.zeros((2, 19, 19), np.uint8))


def test_sharded_game_decompression_colors_and_lru(synthetic_cache):
    _, output = synthetic_cache
    cache = BoardCache(output, max_open_shards=1)
    assert len(cache) == 6
    assert cache.manifest["total_ply"] == 2 * sum([7, 4, 3])
    assert len(cache.manifest["shards"]) == 4  # rank boundaries retained
    for index in range(6):
        expected = sgf_to_planes(GAMES[index % 3])
        decoded = cache.game(index)
        assert decoded.dtype == np.uint8
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(cache.game(index, [len(expected) - 1, 0]), expected[[-1, 0]])
        assert len(cache._open) <= 1
    np.testing.assert_array_equal(cache.colors(1), [0, 1, 0, 1])
    # Owned results survive eviction and closing the mapped file.
    copy = cache.packed(0)
    cache.game(5)
    cache.close()
    np.testing.assert_array_equal(unpack_planes(copy), sgf_to_planes(GAMES[0]))
    verified = verify_cache(output)
    assert verified["payload_bytes"] == 28 * 92
    assert verified["sampled_games"] == 4


def test_history_batch_order_boundaries_and_actual_color(synthetic_cache):
    _, output = synthetic_cache
    cache = BoardCache(output, max_open_shards=1)
    games, plies = [5, 1, 0, 3, 2], [0, 1, 6, 4, 2]
    histories, colors = cache.history_batch(games, plies)
    assert histories.shape == (5, 8, 2, 19, 19)
    for i, (game, ply) in enumerate(zip(games, plies)):
        planes = sgf_to_planes(GAMES[game % 3])
        for j in range(8):
            np.testing.assert_array_equal(histories[i, j], planes[ply - j] if ply >= j else np.zeros((2, 19, 19)))
    inputs = cache.training_batch(games, plies)
    assert inputs.shape == (5, 18, 19, 19)
    np.testing.assert_array_equal(inputs[:, :16], histories.reshape(5, 16, 19, 19))
    np.testing.assert_array_equal(inputs[:, 16, 0, 0], colors == 0)
    np.testing.assert_array_equal(inputs[:, 17, 0, 0], colors == 1)
    assert cache.training_batch([], []).shape == (0, 18, 19, 19)
    for game, ply in [(-1, 0), (6, 0), (0, -1), (1, 4)]:
        with pytest.raises(IndexError):
            cache.history(game, ply)
    with pytest.raises(ValueError):
        cache.history_batch([0.5], [0])
    cache.close()


def test_dataset_selected_player_and_pickle(synthetic_cache, tmp_path):
    _, output = synthetic_cache
    game_file, ply_file = tmp_path / "games.npy", tmp_path / "plies.npy"
    # Select both white and black player plies.
    np.save(game_file, np.array([1, 2, 0, 5], np.uint32))
    np.save(ply_file, np.array([1, 2, 4, 0], np.uint16))
    dataset = PackedBoardDataset(game_file, ply_file, output, max_open_shards=1)
    batch = dataset.__getitems__([3, 0, 2, 1])
    assert [row["game_index"] for row in batch] == [5, 1, 0, 2]
    assert batch[1]["planes"][17].all()
    restored = pickle.loads(pickle.dumps(dataset))
    np.testing.assert_array_equal(restored[0]["planes"], dataset[0]["planes"])
    with pytest.raises(ValueError, match="opponent"):
        PackedBoardDataset([1], [2], output)[0]
    with pytest.raises(IndexError):
        dataset[-1]
    dataset.close()
    restored.close()


def test_torch_dataloader_spawn_and_batch_protocol(synthetic_cache, tmp_path):
    torch = pytest.importorskip("torch")
    _, output = synthetic_cache
    dataset = PackedBoardDataset(np.array([1, 2, 0, 5], np.uint32),
                                 np.array([1, 2, 4, 0], np.uint16), output)
    serial = list(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0))
    parallel = list(torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2,
                                              multiprocessing_context="spawn"))
    for reference, actual in zip(serial, parallel):
        assert actual["planes"].dtype == torch.uint8
        assert actual["planes"].shape == (2, 18, 19, 19)
        for key in reference:
            torch.testing.assert_close(reference[key], actual[key])
    dataset.close()


def test_resume_validates_and_rebuilds_corrupted_shard(synthetic_cache):
    raw, output = synthetic_cache
    shard = output / "shard_00000.npy"
    untouched = output / "shard_00003.npy"
    original_time = untouched.stat().st_mtime_ns
    # Modify payload, preserving file length; hash validation must detect it.
    with shard.open("r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        verify_cache(output)
    convert_training(raw, output, workers=1, games_per_shard=2, ranks=("D", "C"),
                     expected_games_per_rank=3)
    assert untouched.stat().st_mtime_ns == original_time
    assert verify_cache(output)["num_games"] == 6


def test_source_or_config_changes_invalidate_resume(synthetic_cache):
    raw, output = synthetic_cache
    with pytest.raises(ValueError, match="changed"):
        convert_training(raw, output, games_per_shard=3, ranks=("D", "C"), expected_games_per_rank=3)
    source = raw / "train_D.csv"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="changed"):
            convert_training(raw, output, games_per_shard=2, ranks=("D", "C"), expected_games_per_rank=3)
    finally:
        source.write_bytes(original)


@pytest.mark.parametrize("games, expected, reason", [
    (["(;SZ[19]AB[aa];B[bb])"], 1, "invalid SGF"),
    (["(;SZ[19];B[zz])"], 1, "invalid SGF"),
    (GAMES, 4, "expected 4"),
])
def test_bad_games_and_counts_never_publish_manifest(tmp_path, games, expected, reason):
    raw, output = tmp_path / "raw", tmp_path / "cache"
    write_csv(raw, games=games)
    with pytest.raises(ValueError, match=reason):
        convert_training(raw, output, ranks=("D",), expected_games_per_rank=expected)
    assert not (output / "manifest.json").exists()


def test_incomplete_shard_is_unreadable(synthetic_cache, tmp_path):
    _, output = synthetic_cache
    # Change the marker temporarily and restore it even if the assertion fails.
    marker = output / "shard_00000.done.json"
    original = marker.read_text(encoding="utf-8")
    broken = json.loads(original)
    broken["complete"] = False
    try:
        marker.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match="marker"):
            BoardCache(output)
    finally:
        marker.write_text(original, encoding="utf-8")


def test_cross_rank_game_id_is_not_deduplicated(synthetic_cache):
    _, output = synthetic_cache
    meta = pd.read_parquet(output / "games.parquet")
    assert meta["game_id"].nunique() == 3
    assert not meta.duplicated(["rank", "game_id"]).any()
    assert len(meta) == 6
