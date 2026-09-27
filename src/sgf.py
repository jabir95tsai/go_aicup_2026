"""SGF parsing — the single source of truth for turning `sgf_content` into moves.

These records are minimal, machine-generated SGF (`(;SZ[19];B[qd];W[dp];...)`)
with names, ranks, results and comments stripped by the organisers. This is
therefore a tolerant scanner, not a full SGF-4 parser: anything it cannot read
is *reported* on the returned object, never raised. Callers that care about
data quality should check `Game.ok` / `Game.errors` rather than assuming
success.

Coordinates: SGF encodes a point as [column][row] with 'a'..'s' for 0..18.
We flatten to `row * 19 + col`, so 0..360, with PASS = 361.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

BOARD_SIZE = 19
N_POINTS = BOARD_SIZE * BOARD_SIZE  # 361
PASS = N_POINTS  # 361
INVALID = -1

BLACK, WHITE = 0, 1

# One property: a 1-2 letter uppercase key followed by one or more [..] values.
# `[^\]]` is safe here because these files carry no escaped brackets (no
# comments survived anonymisation); `coverage` below is what catches the
# day we are wrong about that.
_PROP_RE = re.compile(r"([A-Z]{1,2})((?:\[[^\]]*\])+)")
_VAL_RE = re.compile(r"\[([^\]]*)\]")


@dataclass
class Game:
    """One parsed record."""

    props: dict[str, list[str]] = field(default_factory=dict)
    moves: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int16))
    colors: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))
    setup_black: list[int] = field(default_factory=list)
    setup_white: list[int] = field(default_factory=list)
    coverage: float = 0.0  # fraction of the string the scanner accounted for
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
    """'qd' -> row*19+col. Empty string (or the legacy 'tt') is a pass."""
    if value == "" or value == "tt":
        return PASS
    if len(value) != 2:
        return INVALID
    col = ord(value[0]) - 97
    row = ord(value[1]) - 97
    if not (0 <= col < BOARD_SIZE and 0 <= row < BOARD_SIZE):
        return INVALID
    return row * BOARD_SIZE + col


def index_to_point(index: int) -> str:
    """Inverse of `point_to_index`, for round-trip verification."""
    if index == PASS:
        return ""
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

    moves: list[int] = []
    colors: list[int] = []
    consumed = 0

    for match in _PROP_RE.finditer(sgf):
        key, raw = match.group(1), match.group(2)
        consumed += match.end() - match.start()
        values = _VAL_RE.findall(raw)

        if key in ("B", "W"):
            colour = BLACK if key == "B" else WHITE
            for value in values:
                index = point_to_index(value)
                if index == INVALID:
                    game.errors.append(f"bad move {key}[{value}]")
                    continue
                moves.append(index)
                colors.append(colour)
        elif key in ("AB", "AW"):
            target = game.setup_black if key == "AB" else game.setup_white
            for value in values:
                index = point_to_index(value)
                if index == INVALID or index == PASS:
                    game.errors.append(f"bad setup {key}[{value}]")
                    continue
                target.append(index)
            game.props.setdefault(key, []).extend(values)
        else:
            game.props.setdefault(key, []).extend(values)

    game.moves = np.asarray(moves, dtype=np.int16)
    game.colors = np.asarray(colors, dtype=np.int8)

    # Structural punctuation the property regex legitimately skips.
    structural = sum(sgf.count(character) for character in "();\n\r \t")
    game.coverage = min(1.0, (consumed + structural) / len(sgf))
    if game.coverage < 0.999:
        game.errors.append(f"low coverage {game.coverage:.4f}")
    if not moves:
        game.errors.append("no moves")

    size = game.board_size
    if size is not None and size != BOARD_SIZE:
        game.errors.append(f"board size {size}")

    return game


def move_tokens(game: Game) -> np.ndarray:
    """Pack (colour, point) into a single token id in 0..723.

    Token layout: `colour * 362 + point`, so black points are 0..361 and white
    points are 362..723. This is the vocabulary the sequence model consumes.
    """
    return (game.colors.astype(np.int16) * (PASS + 1) + game.moves).astype(np.int16)
