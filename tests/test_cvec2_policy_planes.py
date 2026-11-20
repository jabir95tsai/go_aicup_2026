import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features import humansl_torch
from go_aicup.tasks.strength.cvec2_policy import planes
from go_aicup.tasks.strength.policy_large import cache

pytestmark = pytest.mark.skipif(not shutil.which("zstd"), reason="requires the zstd command")


def fixture_planes(lengths):
    n = sum(lengths)
    return {
        "offsets": np.r_[0, np.cumsum(lengths)].astype(np.int64),
        "bits": np.full((n, 993), 4, np.uint8),
        "globals": np.arange(n * 19, dtype=np.float32).reshape(n, 19),
        "target": np.arange(n, dtype=np.int16),
        "side": np.concatenate([np.arange(length, dtype=np.uint8) % 2 for length in lengths]),
    }


def test_plane_reader_crosses_shards_and_checks_checksum(tmp_path):
    sources = {"test": "whole games"}
    identity = object_hash(sources)
    directory = tmp_path / identity
    directory.mkdir()
    games = pd.DataFrame({"game_index": [10, 21, 15, 39], "length": [2, 3, 2, 3],
                          "handicap": [1, 0, 1, 0], "player_color": [1, 1, 0, 0]})
    tables = {}
    shards = []
    for group, idx in (("fit", 0), ("e2", 3)):
        file = directory / f"{group}.parquet"
        rows = games.iloc[idx:idx + 3]
        rows.to_parquet(file, index=False)
        tables[file.name] = file_hash(file)
        for start in range(0, len(rows), 2):
            end = min(start + 2, len(rows))
            lengths = rows.length.iloc[start:end].tolist()
            shard_path = directory / f"{group}-{start}.npz.zst"
            info = planes._write_shard(shard_path, fixture_planes(lengths), lengths)
            record = {"file": shard_path.name, "group": group, "start": start, "end": end, "positions": sum(lengths)}
            write_json(shard_path.with_suffix(".json"), record | info)
            assert planes._cached_shard(shard_path, record) == (record | info)
            shards.append(record | info)
    # The reader must ignore non-teacher shards, including their paths.
    shards.insert(1, {"group": "public", "file": "absent.npz.zst"})
    manifest = directory / "manifest.json"
    write_json(manifest, {"identity": identity, "sources": sources, "tables": tables, "shards": shards})
    write_json(directory / "complete.json", {"identity": identity, "sha256": file_hash(manifest)})
    reader = planes.PlaneReader(directory, games)
    batches = list(reader.batches(0, 4, 3))
    assert [ids for ids, _ in batches] == [[0, 1, 2], [3]]
    assert [[len(part[0]) for part in parts] for _, parts in batches] == [[2, 3, 2], [3]]
    assert [ids for ids, _ in reader.batches(1, 4, 2)] == [[1, 2], [3]]
    with pytest.raises(ValueError, match="teacher game order"):
        planes.PlaneReader(directory, games.iloc[::-1])
    (directory / "fit-2.npz.zst").write_bytes(b"corrupt")
    assert planes._cached_shard(directory / "fit-2.npz.zst", shards[2]) is None
    with pytest.raises(ValueError, match="checksum"):
        list(reader.batches(0, 4, 2))


@pytest.mark.skipif(not humansl_torch.KATAGO_PY.is_dir(), reason="requires project KataGo Python source")
def test_board_encoder_compression_keeps_color_komi_and_position_order(tmp_path):
    moves = np.array([0, 360, 1, 359], np.int16)
    colors = np.array([0, 1, 0, 1], np.int8)
    pd.DataFrame({"game_index": [12, 91], "length": [4, 4], "shard_id": [0, 0], "local_offset": [0, 4]}).to_parquet(
        tmp_path / "games.parquet", index=False,
    )
    np.save(tmp_path / "shard_00000_moves.npy", np.tile(moves, 2))
    np.save(tmp_path / "shard_00000_colors.npy", np.tile(colors, 2))
    cache.init_encoder({"version": 15}, tmp_path)
    hd0, hd1 = cache.encode((12, 0, 0, 4)), cache.encode((91, 1, 1, 4))
    arrays = {name: np.concatenate(pair) for name, pair in zip(
        ("bits", "globals", "target", "side"), zip(hd0, hd1, strict=True), strict=True,
    )}
    arrays["offsets"] = np.array([0, 4, 8], np.int64)
    path = tmp_path / "planes.npz.zst"
    digest = planes._write_shard(path, arrays, [4, 4])["sha256"]
    restored = planes._decode(path, [4, 4], digest)
    np.testing.assert_array_equal(restored["target"], np.tile(moves, 2))
    np.testing.assert_array_equal(restored["side"], np.r_[colors, 1 - colors])
    np.testing.assert_allclose(restored["globals"][[0, 4], 5], np.array([-6.5, -0.5]) / 20)
    np.testing.assert_array_equal(restored["bits"][:4], restored["bits"][4:])
    assert restored["bits"].shape == (8, 993)


@pytest.mark.skipif(not humansl_torch.KATAGO_PY.is_dir(), reason="requires project KataGo Python source")
def test_mover_stones_preserve_premove_orientation():
    moves = np.array([71, 319, 53, 258], np.int16)
    colors = np.array([0, 1, 0, 1], np.int8)
    humansl_torch._init_worker({"version": 15}, 6.5)
    planes_, _, targets = humansl_torch.encode_game((moves, colors))
    packed = cache.pack_mover_stones(torch.from_numpy(planes_))
    assert packed.shape == (4, 91) and packed.dtype == torch.uint8
    board = np.unpackbits(packed.numpy(), axis=1)[:, :722].reshape(4, 2, 361)
    wanted = np.zeros((4, 2, 361), np.uint8)
    wanted[1, 1, moves[0]] = 1
    wanted[2, 0, moves[0]] = 1
    wanted[2, 1, moves[1]] = 1
    wanted[3, 0, moves[1]] = 1
    wanted[3, 1, moves[[0, 2]]] = 1
    np.testing.assert_array_equal(board, wanted)
    np.testing.assert_array_equal(targets, moves)
