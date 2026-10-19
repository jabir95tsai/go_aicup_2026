"""Convert one game to in-memory, post-move black / white board planes.

For zero-based ply i, channels 0 / 1 contain black / white stones
in the board after ply i, including captures.
Coordinates are [row, column], matching go_aicup.data.sgf.

No feature arrays are written to disk. Example:
    planes = sgf_to_planes("(;SZ[19];B[pd];W[dp])")
    # planes.shape == (2, 2, 19, 19), dtype == numpy.float32
    # tensor = torch.from_numpy(planes)

For a processed GameStore, avoid parsing SGF again:
    moves, colors = store.game(game_index)
    planes = moves_to_planes(moves, colors)

These planes deliberately use fixed black / white channels, as specified by
this representation, rather than the labelled-player perspective.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from go_aicup.data import sgf as sgflib
from go_aicup.features.game_stats import NB, NB_CNT, _flood

BOARD_SIZE = sgflib.BOARD_SIZE
N_POINTS = sgflib.N_POINTS
N_CHANNELS = 2


@njit(cache=True)
def _replay_planes(moves, colors, out, validate_moves):
    """Fill output; return (zero-based error ply, error code), or (-1, 0)."""
    board = np.zeros(N_POINTS, np.int8)
    snapshots = np.empty((2 if validate_moves else 0, N_POINTS), np.int8)
    mark = np.zeros(N_POINTS, np.int32)
    lmark = np.zeros(N_POINTS, np.int32)
    stamp = np.zeros(1, np.int32)
    stack = np.zeros(N_POINTS, np.int16)
    members = np.zeros(N_POINTS, np.int16)

    for i in range(len(moves)):
        p = moves[i]
        color = colors[i] + 1  # parser: 0/1; board: 0 empty, 1 black, 2 white
        opp = 3 - color
        if validate_moves and board[p] != 0:
            return i, 1
        board[p] = color
        for j in range(NB_CNT[p]):
            q = NB[p, j]
            if board[q] == opp:
                n, libs = _flood(
                    board, q, mark, lmark, stamp, stack, members, NB, NB_CNT
                )
                if libs == 0:
                    for m in range(n):
                        board[members[m]] = 0
        if validate_moves:
            _, libs = _flood(
                board, p, mark, lmark, stamp, stack, members, NB, NB_CNT
            )
            if libs == 0:
                return i, 2
            # Simple ko: a stone move must not recreate the board two plies ago.
            if i >= 2:
                repeats = True
                for q in range(N_POINTS):
                    if board[q] != snapshots[i % 2, q]:
                        repeats = False
                        break
                if repeats:
                    return i, 3
            snapshots[i % 2] = board

        for q in range(N_POINTS):
            out[i, 0, q] = 1.0 if board[q] == 1 else 0.0
            out[i, 1, q] = 1.0 if board[q] == 2 else 0.0

    return -1, 0


def moves_to_planes(moves: np.ndarray, colors: np.ndarray, *, validate_moves: bool = False) -> np.ndarray:
    """Return a contiguous (T, 2, 19, 19) float32 array from parsed moves.

    Channels 0 / 1 mark black / white stones after each move, respectively.

    moves: one-dimensional integer point indices 0..360.
    colors: same-length integer array, 0 for black and 1 for white.

    Replay starts on an empty 19x19 board. Empty arrays return shape
    (0, 2, 19, 19). Inputs are never modified. Set validate_moves=True for
    an explicit occupied-point, suicide and immediate-ko audit. Ordinary
    conversion only replays moves and captures.
    """
    moves = np.asarray(moves)
    colors = np.asarray(colors)
    if moves.ndim != 1 or colors.ndim != 1 or moves.shape != colors.shape:
        raise ValueError("moves and colors must be one-dimensional and equal-length")
    for name, values, upper in (("moves", moves, N_POINTS - 1), ("colors", colors, 1)):
        if values.size and values.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain integers")
        if np.any(values < 0) or np.any(values > upper):
            raise ValueError(f"{name} must be in 0..{upper}")

    # Validate before narrowing, so oversized indices cannot wrap into valid ones.
    moves = np.ascontiguousarray(moves, dtype=np.int16)
    colors = np.ascontiguousarray(colors, dtype=np.int8)
    out = np.empty((len(moves), N_CHANNELS, N_POINTS), dtype=np.float32)
    if len(moves):
        error_ply, code = _replay_planes(moves, colors, out, validate_moves)
        if code:
            reasons = {1: "occupied point", 2: "suicide", 3: "immediate ko recapture"}
            point = sgflib.index_to_point(int(moves[error_ply]))
            raise ValueError(f"move {error_ply + 1} [{point}]: {reasons[code]}")
    return out.reshape(len(moves), N_CHANNELS, BOARD_SIZE, BOARD_SIZE)


def parse_plane_game(sgf_content: str) -> sgflib.Game:
    """Parse once for both the in-memory converter and disk-cache builder."""
    game = sgflib.parse(sgf_content)
    if not game.ok:
        raise ValueError("invalid SGF: " + "; ".join(game.errors))
    return game


def sgf_to_planes(sgf_content: str, *, validate_moves: bool = False) -> np.ndarray:
    """Return post-move black/white float32 planes shaped (T, 2, 19, 19).

    Uses the competition's linear SZ[19] format. No disk I/O.
    """
    game = parse_plane_game(sgf_content)
    return moves_to_planes(game.moves, game.colors, validate_moves=validate_moves)
