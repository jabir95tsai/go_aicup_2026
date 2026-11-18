import json

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import BoardCache, write_store_cache
from go_aicup.tasks.identity import external
from go_aicup.tasks.identity.ge2e import GE2EBatches
from scripts.identity.ogs_stream_filter import META


def _game(rng, length):
    return rng.choice(361, size=length, replace=False).astype(np.int16)


def _yugen(tmp_path, rng):
    games = [_game(rng, 80 + i) for i in range(6)]
    moves = np.concatenate(games)
    colors = np.concatenate([np.arange(len(g)) % 2 for g in games]).astype(np.int8)
    lengths = np.array([len(g) for g in games])
    meta = pd.DataFrame({"offset": np.r_[0, np.cumsum(lengths)[:-1]], "length": lengths,
                         "color": ["B", "W"] * 3, "rank_id": np.arange(6) % 10})
    directory = write_store_cache(moves, colors, meta, tmp_path / "yugen", games_per_shard=4)
    games_parquet = pd.read_parquet(directory / "games.parquet")
    # the real Yugen games.parquet has these extra columns; add them so the merge sees the same schema
    games_parquet = games_parquet.assign(rank="D", game_id="g", player_id="y", color="B", source_row=np.int32(0))
    games_parquet.to_parquet(directory / "games.parquet", index=False)
    manifest = json.loads((directory / "manifest.json").read_text())
    from go_aicup.artifacts import file_hash
    manifest["games"]["bytes"] = (directory / "games.parquet").stat().st_size
    manifest["games"]["sha256"] = file_hash(directory / "games.parquet")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory


def _filtered(tmp_path, rng):
    out = tmp_path / "filtered"
    out.mkdir()
    rows, moves, offset = [], [], 0
    pairs = [(10, 11)] * 5 + [(10, 12)] * 3 + [(13, 99)] * 4       # 99 is a bot (no folder)
    for game, (b, w) in enumerate(pairs):
        pts = _game(rng, 70 + game)
        if game == 0:
            pts[65] = 361                                           # pass after ply 60: game cut there
        rows.append((1000 + game, b, w, 2024, 0, 1, 0, 20.0, 21.0, len(pts), offset))
        moves.append(pts)
        offset += len(pts)
    np.save(out / "moves_00000.npy", np.concatenate(moves))
    np.save(out / "meta_00000.npy", np.array(rows, dtype=META))
    np.save(out / "owners.npy", np.array([10, 11, 12, 13]))
    return out, np.concatenate(moves), rows


def test_merged_cache_keeps_yugen_rows_and_adds_external(tmp_path):
    rng = np.random.default_rng(0)
    yugen = _yugen(tmp_path, rng)
    filtered, moves, rows = _filtered(tmp_path, rng)
    report = external.build(filtered, tmp_path / "merged", yugen=yugen, workers=1, min_games=3, max_players=None,
                            games_per_player=10, seed=0, human_opponent=True)
    # player 10: 8 games, 11: 5, 12: 3, 13: only bot games -> excluded
    assert report["selected_players"] == 3 and report["external_rows"] == 16
    merged, base = BoardCache(tmp_path / "merged"), BoardCache(yugen)
    assert len(merged) == len(base) + 16
    for g in range(len(base)):
        assert np.array_equal(merged.packed(g), base.packed(g))
    players = pd.read_parquet(tmp_path / "merged" / "external_players.parquet")
    assert players["game_index"].min() == len(base)
    meta = pd.read_parquet(tmp_path / "merged" / "games.parquet")
    # first external row: player 10's first game (ogs game 1000), cut at the pass (65 plies), black
    first = meta.iloc[len(base)]
    assert first["player_id"] == "ogs_10" and first["game_id"] == "ogs_1000" and first["length"] == 65
    assert first["player_color"] == 0
    assert merged.game(len(base)).shape[0] == 65
    stored = np.load(tmp_path / "merged" / f"shard_{merged.shard_ids[len(base)]:05d}_moves.npy")
    np.testing.assert_array_equal(stored[:65], moves[:65])
    # player 11 appears as white in the shared games
    assert set(meta.loc[meta["player_id"] == "ogs_11", "player_color"]) == {1}


def test_player_probabilities_draw_the_requested_share(tmp_path):
    lengths = np.full(40, 100)
    games = [np.array([i]) for i in range(40)]
    p = np.r_[np.full(10, 0.5 / 10), np.full(30, 0.5 / 30)]
    batches = GE2EBatches(tmp_path, games, lengths, players=4, games=1, moves=2, steps=400, seed=0, rank=0,
                          player_color=np.zeros(40, np.int8), min_ply=0, player_p=p)
    drawn = np.concatenate([batches.sample(s)[0][::2] for s in range(400)])
    share = np.mean(drawn < 10)
    assert 0.45 < share < 0.55
    uniform = GE2EBatches(tmp_path, games, lengths, players=4, games=1, moves=2, steps=10, seed=0, rank=0,
                          player_color=np.zeros(40, np.int8), min_ply=0)
    again = GE2EBatches(tmp_path, games, lengths, players=4, games=1, moves=2, steps=10, seed=0, rank=0,
                        player_color=np.zeros(40, np.int8), min_ply=0, player_p=None)
    assert all(np.array_equal(uniform.sample(s)[0], again.sample(s)[0]) for s in range(10))
