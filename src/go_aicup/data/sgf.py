"""SGF parsing — the single source of truth for turning `sgf_content` into moves.

The competition format is `(;SZ[19];B[qd];W[dp];...)`: one linear game,
on-board moves only, no setup stones or metadata. Invalid input is reported
in Game.errors; properties remain available for source-data audits.

Coordinates: SGF encodes a point as [column][row] with 'a'..'s' for 0..18.
We flatten to `row * 19 + col`, so 0..360.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

BOARD_SIZE = 19
N_POINTS = BOARD_SIZE * BOARD_SIZE  # 361
INVALID = -1

BLACK, WHITE = 0, 1

# Retain property names even in unsupported records for the audit report.
_PROP_RE = re.compile(r"([A-Z]{1,2})((?:\[[^\]]*\])+)")
_VAL_RE = re.compile(r"\[([^\]]*)\]")
_GAME_RE = re.compile(r"\(;SZ\[19\](?:;[BW]\[[a-s]{2}\])+\)")


@dataclass
class Game:
    """One parsed record."""

    props: dict[str, list[str]] = field(default_factory=dict)
    moves: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    colors: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    errors: list[str] = field(default_factory=list)

    @property
    def n_moves(self) -> int:
        return int(self.moves.shape[0])

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def board_size(self) -> int | None:
        sz = self.props.get("SZ")
        if not sz:
            return None
        try:
            return int(sz[0])
        except ValueError:
            return None


def point_to_index(value: str) -> int:
    """'qd' -> row*19+col; return INVALID for points outside the board."""
    if len(value) != 2:
        return INVALID
    col = ord(value[0]) - 97
    row = ord(value[1]) - 97
    if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
        return INVALID
    return row * BOARD_SIZE + col


def index_to_point(index: int) -> str:
    """Inverse of `point_to_index`, for round-trip verification."""
    if not (0 <= index < N_POINTS):
        raise ValueError(f"index out of range: {index}")
    row, col = divmod(index, BOARD_SIZE)
    return chr(97 + col) + chr(97 + row)


def parse(sgf: str) -> Game:
    """Parse one `sgf_content` string. Never raises."""
    game = Game()

    if not isinstance(sgf, str) or not sgf.strip():
        game.errors.append("empty")
        return game

    sgf = sgf.strip()
    for match in _PROP_RE.finditer(sgf):
        key, raw = match.group(1), match.group(2)
        if key not in ("B", "W"):
            game.props.setdefault(key, []).extend(_VAL_RE.findall(raw))
    if _GAME_RE.fullmatch(sgf) is None:
        game.errors.append("expected (;SZ[19];B[point];W[point];...) with on-board moves only")
        return game
    nodes = sgf[1:-1].split(";")[2:]
    game.moves = np.asarray([point_to_index(node[2:4]) for node in nodes], dtype=np.int16)
    game.colors = np.asarray([BLACK if node[0] == "B" else WHITE for node in nodes], dtype=np.int8)
    return game
