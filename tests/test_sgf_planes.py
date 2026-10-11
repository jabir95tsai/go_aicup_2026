"""Behavioral checks for the SGF-to-training-plane converter."""

import csv

import numpy as np
import pytest

from go_aicup.data import sgf as sgflib
from go_aicup.features.sgf_planes import moves_to_planes, sgf_to_planes
from go_aicup.paths import DATA


def test_coordinates_post_move_planes_and_color_channels():
    planes = sgf_to_planes("(;SZ[19];B[pd];W[dp];B[qp])")
    assert planes.shape == (3, 2, 19, 19)
    assert planes.dtype == np.float32
    assert planes.flags.c_contiguous
    assert np.isin(planes, [0, 1]).all()
    assert planes[0, 0, 3, 15] == 1  # pd = column 15, row 3
    assert planes[0].sum() == 1
    assert planes[1, 1, 15, 3] == 1  # dp = column 3, row 15
    assert planes[2, 0, 15, 16] == 1
    np.testing.assert_array_equal(planes.sum(axis=(2, 3)), [[1, 0], [1, 1], [2, 1]])
    assert not (planes[:, 0] * planes[:, 1]).any()


def test_post_move_boards_and_no_future_information():
    nodes = [f"{('B' if i % 2 == 0 else 'W')}[{chr(97 + i)}d]" for i in range(10)]
    planes = sgf_to_planes("(;SZ[19];" + ";".join(nodes) + ")")
    for i in range(10):
        assert planes[i].sum() == i + 1
        for j in range(10):
            assert planes[i, j % 2, 3, j] == (j <= i)
    # Extending a game cannot alter any earlier training input.
    prefix = sgf_to_planes("(;SZ[19];" + ";".join(nodes[:4]) + ")")
    np.testing.assert_array_equal(planes[:4], prefix)


def test_capture_preserves_historical_stones_and_allows_refill():
    planes = sgf_to_planes("(;SZ[19];B[ba];W[aa];B[pp];W[qq];B[ab];W[pd];B[aa])")
    assert planes[4, 1, 0, 0] == 0  # aa was captured after B ab
    assert planes[3, 1, 0, 0] == 1  # previous board still contains W aa
    assert planes[1, 1, 0, 0] == 1  # old output must not share mutable board data
    assert planes[6, 0, 0, 0] == 1  # empty point may be filled again
    assert planes[6, 0].sum() == 4
    assert planes[6, 1].sum() == 2


@pytest.mark.parametrize("sgf, final_black", [
    ("(;SZ[19];W[aa];W[ba];B[ab];B[bb];B[ca])", 3),
    ("(;SZ[19];B[bb];W[cb];W[bc];B[ca];B[db];B[ac];B[bd];B[cc])", 6),
])
def test_multi_stone_and_separate_group_captures(sgf, final_black):
    planes = sgf_to_planes(sgf)
    assert planes[-1, 0].sum() == final_black
    assert planes[-1, 1].sum() == 0


def test_passes_preserve_board_and_actual_colors_are_used():
    planes = sgf_to_planes("(;SZ[19];W[pd];W[];B[tt];W[dp])")
    assert planes.shape[0] == 4
    assert planes[0, 0].sum() == 0
    assert planes[0, 1, 3, 15] == 1  # white first, not inferred from parity
    np.testing.assert_array_equal(planes[1], planes[0])
    np.testing.assert_array_equal(planes[2], planes[0])
    assert planes[3, 1].sum() == 2


def test_move_array_entrypoint_matches_sgf_without_mutating_inputs():
    game = sgflib.parse("(;SZ[19];B[ba];W[aa];B[pp];W[qq];B[ab])")
    # Non-contiguous arrays, as might be passed by an upstream dataset.
    moves = np.repeat(game.moves, 2)[::2]
    colors = np.repeat(game.colors, 2)[::2]
    before_moves, before_colors = moves.copy(), colors.copy()
    np.testing.assert_array_equal(
        moves_to_planes(moves, colors),
        sgf_to_planes("(;SZ[19];B[ba];W[aa];B[pp];W[qq];B[ab])"),
    )
    np.testing.assert_array_equal(moves, before_moves)
    np.testing.assert_array_equal(colors, before_colors)
    assert moves_to_planes([], []).shape == (0, 2, 19, 19)


def test_root_move_and_default_board_size():
    assert sgf_to_planes("(;W[pd])")[0, 1, 3, 15] == 1
    assert sgf_to_planes("(;FF[4]GM[1]SZ[19];B[pd])").shape[0] == 1


@pytest.mark.parametrize("content", [
    "", None, "(;SZ[19])", "(;SZ[9];B[aa])", "(;SZ[bad];B[aa])",
    "(;SZ[19][19];B[aa])", "(;SZ[19];B[zz])", "(;SZ[19];B[aa]",
    ";SZ[19];B[aa]", "(;SZ[19];B[aa](;W[bb])(;W[cc]))",
    "(;SZ[19];B[aa])(;SZ[19];W[bb])", "(;SZ[19];B[aa]W[bb])",
    "(;SZ[19];B[aa][bb])", "(;SZ[19]AB[aa];B[bb])",
    "(;SZ[19]AW[aa];B[bb])", "(;SZ[19];B[aa];AE[aa];W[bb])",
])
def test_invalid_or_unsupported_sgf_is_rejected(content):
    with pytest.raises(ValueError):
        sgf_to_planes(content)


@pytest.mark.parametrize("moves, colors", [
    ([[0]], [0]), ([0], []), ([0.5], [0]), ([0], [0.5]),
    ([-1], [0]), ([362], [0]), ([65536], [0]), ([0], [-1]),
    ([0], [2]), ([0], [256]), (["0"], [0]), ([True], [0]),
])
def test_invalid_move_arrays_are_rejected_before_narrowing(moves, colors):
    with pytest.raises(ValueError):
        moves_to_planes(moves, colors)


@pytest.mark.parametrize("content, reason", [
    ("(;SZ[19];B[aa];W[aa])", "occupied point"),
    ("(;SZ[19];B[ba];B[ab];W[aa])", "suicide"),
])
def test_invalid_board_moves_are_rejected(content, reason):
    with pytest.raises(ValueError, match=reason):
        sgf_to_planes(content)


KO_PREFIX = "(;SZ[19];B[ba];W[bb];B[ab];W[ca];B[bc];W[db];B[qq];W[cc];B[cb]"


def test_immediate_ko_is_rejected_but_recapture_after_passes_is_allowed():
    with pytest.raises(ValueError, match="move 10 .*immediate ko recapture"):
        sgf_to_planes(KO_PREFIX + ";W[bb])")
    planes = sgf_to_planes(KO_PREFIX + ";W[];B[];W[bb])")
    assert planes[-1, 1, 1, 1] == 1
    assert planes[-1, 0, 1, 2] == 0


def main():
    """Print the final post-move tensor for the first D-rank training game."""
    import torch

    training_path = DATA / "training" / "train_D.csv"
    if not training_path.is_file():
        training_path = DATA / "raw" / "training" / "train_D.csv"
    with training_path.open(newline="", encoding="utf-8-sig") as handle:
        first_game = next(csv.DictReader(handle), None)
    if first_game is None:
        raise ValueError(f"No games found in {training_path}")

    planes = sgf_to_planes(first_game["sgf_content"])
    final_tensor = torch.from_numpy(planes[-1])
    print(f"Source: {training_path}")
    print(f"Game: {first_game['game_id']}, moves: {len(planes)}")
    print(f"Final tensor: shape={tuple(final_tensor.shape)}, dtype={final_tensor.dtype}")
    torch.set_printoptions(profile="full", linewidth=120)
    print(final_tensor)


if __name__ == "__main__":
    main()
