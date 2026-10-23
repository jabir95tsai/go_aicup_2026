"""Release path: lenient SGF, store -> board cache, and pipeline helpers."""
import numpy as np
import pandas as pd

from go_aicup.data import sgf
from go_aicup.data.board_cache import BoardCache, unpack_planes, write_store_cache
from go_aicup.features.sgf_planes import moves_to_planes
from go_aicup.tasks.identity import data
from go_aicup.tasks.identity.predict import PAD, store_prefix
from go_aicup.tasks.strength.v2.aggregate import CATEGORY_DTYPE
from go_aicup.tasks.strength.v2.predict import c2_probabilities


def test_lenient_parser_drops_passes_and_flags_setup():
    game = sgf.parse_lenient("(;GM[1]FF[4]SZ[19]KM[6.5];B[pd];W[];B[tt];W[dp])")
    assert game.moves.tolist() == [sgf.point_to_index("pd"), sgf.point_to_index("dp")]
    assert game.colors.tolist() == [sgf.BLACK, sgf.WHITE]
    assert game.props["_passes"] == ["2"] and game.ok
    setup = sgf.parse_lenient("(;SZ[19]HA[2]AB[dd][pp];W[dp])")
    assert not setup.ok and setup.moves.tolist() == [sgf.point_to_index("dp")]
    strict = "(;SZ[19];B[qd];W[dp];B[pq])"
    lenient = sgf.parse_lenient(strict)
    assert np.array_equal(lenient.moves, sgf.parse(strict).moves)
    assert np.array_equal(lenient.colors, sgf.parse(strict).colors)


def _store(games, colors_of_player):
    moves, colors, lengths = [], [], []
    for points in games:
        moves.extend(points)
        colors.extend(np.arange(len(points)) % 2)
        lengths.append(len(points))
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    meta = pd.DataFrame({"offset": offsets, "length": lengths, "color": colors_of_player})
    return np.array(moves, np.int16), np.array(colors, np.int8), meta


def test_store_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    games = [rng.choice(361, size=n, replace=False) for n in (7, 0, 12, 5)]
    moves, colors, meta = _store(games, ["B", "W", "W", "B"])
    write_store_cache(moves, colors, meta, tmp_path / "cache", games_per_shard=2)
    cache = BoardCache(tmp_path / "cache")
    table = pd.read_parquet(tmp_path / "cache" / "games.parquet")
    assert len(cache) == 3 and table["store_row"].tolist() == [0, 2, 3]
    assert cache.player_colors.tolist() == [0, 1, 0]
    for index, row in enumerate(table["store_row"]):
        expected = moves_to_planes(games[row].astype(np.int16), (np.arange(len(games[row])) % 2).astype(np.int8))
        np.testing.assert_array_equal(cache.game(index), expected.astype(np.uint8))
        np.testing.assert_array_equal(unpack_planes(cache.packed(index)), cache.game(index))


def test_store_prefix_short_games_and_d4():
    rng = np.random.default_rng(1)
    games = [rng.choice(361, size=n, replace=False) for n in (80, 20, 10)]
    moves, colors, meta = _store(games, ["B"] * 3)
    prefix, usable = store_prefix(moves, colors, meta, "d4", min_plies=16)
    assert usable.tolist() == [True, True, False]
    assert (prefix[1, 20:] == PAD).all() and (prefix[0] != PAD).all()
    np.testing.assert_array_equal(prefix[0], data.canonical_d4(games[0][None, :60].astype(np.int16))[0])
    rotated = data.d4_tables()[5][games[1]]
    again, _ = store_prefix(*_store([rotated], ["B"])[:2], _store([rotated], ["B"])[2], "d4", 16)
    np.testing.assert_array_equal(again[0], prefix[1])


def test_c2_probabilities_uses_nearest_k_and_fallback():
    category = np.zeros(3, dtype=CATEGORY_DTYPE)
    category["game_index"] = [0, 1, 2]
    category["num_moves"] = [2, 4, 1]
    category["log_probability_sum"] = category["num_moves"][:, None] * np.log(np.full((3, 10), 0.1))
    category["log_probability_sum"][0, 3] = 2 * np.log(0.5)
    questions = pd.DataFrame({"num_games": [5, 7, 20], "games": [np.array([0, 1]), np.array([2]), np.array([], int)]})
    temps = {5: 1.0, 10: 2.0, 15: 3.0, 20: 4.0}
    p = c2_probabilities(category, questions, temps)
    logits = (np.log(np.full(10, 0.1)) + np.where(np.arange(10) == 3, np.log(0.5), np.log(0.1))) / 2
    expected = np.exp(logits) / np.exp(logits).sum()
    np.testing.assert_allclose(p[0], expected)
    np.testing.assert_allclose(p[1], np.full(10, 0.1))       # k=7 -> nearest 5; uniform stays uniform
    np.testing.assert_allclose(p[2], p[:2].mean(axis=0))     # no usable game -> mean of the others
