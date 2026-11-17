"""Local opening states for competition 1 (multi-level back-off, 2026-11-17).

The exact-prefix opening scorer (opening.prefix_backoff) loses all information once a
query game leaves every gallery line, even when the corner that the move is played in
repeats a habit exactly. Two coarser states per own move, both invariant to any board
symmetry of the whole game:

L1 corner   the stones (own / opponent, after captures) in the 9x9 quadrant the move is
            played in, mapped to one canonical corner (row/column flips, then the
            smaller of the pattern and its transpose); moves on the centre lines have
            no L1 state.
L2 reply    the 7x7 window (own / opponent / empty / off-board) around the opponent's
            previous move, canonical over the 8 symmetries, plus where the move lies in
            that window ("tenuki" when outside).

Each state comes as a context key (state before the move) and an outcome key (state
plus the move, with the move folded over any symmetry the state keeps), so that
P(move | state) can be estimated by counting.
"""
from __future__ import annotations

import hashlib

import numpy as np

SIZE = 19
HALF = 9
WIN = 3                      # window half-width -> 7x7
TENUKI = (99, 99)
NONE = np.int64(0)           # "no state" key


def _neighbors(r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r < SIZE - 1:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c < SIZE - 1:
        yield r, c + 1


def _group(board, r, c):
    color = board[r, c]
    stack, seen, libs = [(r, c)], {(r, c)}, False
    while stack:
        y, x = stack.pop()
        for ny, nx in _neighbors(y, x):
            v = board[ny, nx]
            if v == 0:
                libs = True
            elif v == color and (ny, nx) not in seen:
                seen.add((ny, nx))
                stack.append((ny, nx))
    return seen, libs


def play(board: np.ndarray, point: int, color: int) -> None:
    """Place `color` (1 black, 2 white) at `point` and remove captured opponent groups."""
    r, c = divmod(int(point), SIZE)
    board[r, c] = color
    other = 3 - color
    for ny, nx in _neighbors(r, c):
        if board[ny, nx] == other:
            stones, libs = _group(board, ny, nx)
            if not libs:
                for y, x in stones:
                    board[y, x] = 0


def _key(*parts) -> np.int64:
    """Deterministic across processes (Python's hash() of bytes is salted per process)."""
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    h = int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF
    return np.int64(h or 1)


def _relabel(board: np.ndarray, own: int) -> np.ndarray:
    """0 empty, 1 own, 2 opponent."""
    out = np.zeros_like(board)
    out[board == own] = 1
    out[board == 3 - own] = 2
    return out


def corner_keys(rel: np.ndarray, point: int) -> tuple[np.int64, np.int64]:
    """(context, outcome) of L1 for a move at `point` on the relabelled board."""
    r, c = divmod(int(point), SIZE)
    if r == HALF or c == HALF:
        return NONE, NONE
    rows = slice(0, HALF) if r < HALF else slice(HALF + 1, SIZE)
    cols = slice(0, HALF) if c < HALF else slice(HALF + 1, SIZE)
    a = rel[rows, cols]
    rr = r if r < HALF else SIZE - 1 - r
    cc = c if c < HALF else SIZE - 1 - c
    if r >= HALF:
        a = a[::-1]
    if c >= HALF:
        a = a[:, ::-1]
    t = a.T
    ab, tb = a.tobytes(), np.ascontiguousarray(t).tobytes()
    if tb < ab:
        ab, rr, cc = tb, cc, rr
    elif tb == ab:
        rr, cc = min(rr, cc), max(rr, cc)
    return _key("c", ab), _key("c", ab, rr, cc)


def _images(a: np.ndarray):
    for k in range(4):
        b = np.rot90(a, k)
        yield b
        yield b[:, ::-1]


def reply_keys(rel: np.ndarray, prev: int | None, point: int) -> tuple[np.int64, np.int64]:
    """(context, outcome) of L2: window around the opponent's previous move `prev`."""
    if prev is None:
        return NONE, NONE
    pr, pc = divmod(int(prev), SIZE)
    r, c = divmod(int(point), SIZE)
    pad = np.full((SIZE + 2 * WIN, SIZE + 2 * WIN), 3, rel.dtype)
    pad[WIN:WIN + SIZE, WIN:WIN + SIZE] = rel
    win = pad[pr:pr + 2 * WIN + 1, pc:pc + 2 * WIN + 1]
    dr, dc = r - pr, c - pc
    inside = abs(dr) <= WIN and abs(dc) <= WIN
    mark = np.zeros_like(win)
    if inside:
        mark[WIN + dr, WIN + dc] = 1
    best, offs = None, []
    for w_img, m_img in zip(_images(win), _images(mark)):
        b = np.ascontiguousarray(w_img).tobytes()
        if inside:
            i, j = np.unravel_index(int(np.argmax(m_img)), m_img.shape)
            off = (int(i) - WIN, int(j) - WIN)
        else:
            off = TENUKI
        if best is None or b < best:
            best, offs = b, [off]
        elif b == best:
            offs.append(off)
    return _key("r", best), _key("r", best, min(offs))


def game_keys(moves: np.ndarray, own: int, n_own: int) -> np.ndarray:
    """int64 [n_own, 4]: (L1 context, L1 outcome, L2 context, L2 outcome) for the player's
    first `n_own` moves; `own` 0 black / 1 white is the player's colour, moves[ply] the
    point of ply (black plays even plies)."""
    board = np.zeros((SIZE, SIZE), np.int8)
    out = np.zeros((n_own, 4), np.int64)
    last = own + 2 * (n_own - 1)
    for ply in range(last + 1):
        color = 1 + ply % 2
        if ply % 2 == own:
            i = (ply - own) // 2
            rel = _relabel(board, 1 + own)
            out[i, :2] = corner_keys(rel, moves[ply])
            out[i, 2:] = reply_keys(rel, moves[ply - 1] if ply > 0 else None, moves[ply])
        play(board, moves[ply], color)
    return out
