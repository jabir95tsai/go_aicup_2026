"""Board-replay move features for the player-likelihood term (2026-11-21, PID_LIKELIHOOD_RICH_PLAN.md).

For each listed own-move position, the game is replayed (captures resolved) up to the board before
that move, every group's liberties are counted, and each of the 32 candidate points gets eight 0/1
features (point = row * 19 + col; 361 = pass gets zeros):

  0 contact      a 4-neighbour is an opponent stone
  1 solid        a 4-neighbour is an own stone
  2 diagonal     no own 4-neighbour, a diagonal neighbour is own (kosumi-like)
  3 jump         none of the above own relations, own stone two points away in a line, middle empty
  4 knight       none of the above own relations, own stone a knight's move away
  5 capture      an adjacent opponent group has 1 liberty
  6 atari        an adjacent opponent group has 2 liberties
  7 rescue       an adjacent own group has 1 liberty

The colour to move alternates from black (all bench and official added-set games are HD <= 1).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

N_RICH = 8


@njit(cache=True)
def _neighbours():
    nb = np.full((361, 4), -1, np.int64)
    for p in range(361):
        r, c = p // 19, p % 19
        k = 0
        if r > 0:
            nb[p, k] = p - 19
            k += 1
        if r < 18:
            nb[p, k] = p + 19
            k += 1
        if c > 0:
            nb[p, k] = p - 1
            k += 1
        if c < 18:
            nb[p, k] = p + 1
            k += 1
    return nb


@njit(cache=True)
def _group(board, start, nb, mark, stamp, stack, members):
    """flood fill of the group at `start`: (n_members, n_liberties); members[:n] filled."""
    colour = board[start]
    n = 0
    libs = 0
    top = 0
    stack[top] = start
    top += 1
    mark[start] = stamp
    while top > 0:
        top -= 1
        p = stack[top]
        members[n] = p
        n += 1
        for k in range(4):
            q = nb[p, k]
            if q < 0 or mark[q] == stamp:
                continue
            if board[q] == colour:
                mark[q] = stamp
                stack[top] = q
                top += 1
            elif board[q] == 0:
                mark[q] = stamp            # count each liberty once
                libs += 1
    return n, libs


@njit(cache=True)
def _play(board, p, colour, nb, mark, stamp, stack, members):
    """place `colour` (1 black, 2 white) at p and remove captured opponent groups; returns stamp."""
    board[p] = colour
    other = 3 - colour
    for k in range(4):
        q = nb[p, k]
        if q >= 0 and board[q] == other:
            stamp += 1
            n, libs = _group(board, q, nb, mark, stamp, stack, members)
            if libs == 0:
                for i in range(n):
                    board[members[i]] = 0
    return stamp


@njit(cache=True)
def _features_at(board, own, nb, mark, stamp, stack, members, gid, glibs, points, out):
    """liberties of every group on the current board, then the eight features of each point."""
    for p in range(361):
        gid[p] = -1
    g = 0
    for p in range(361):
        if board[p] != 0 and gid[p] < 0:
            stamp += 1
            n, libs = _group(board, p, nb, mark, stamp, stack, members)
            for i in range(n):
                gid[members[i]] = g
            glibs[g] = libs
            g += 1
    opp = 3 - own
    for j in range(points.shape[0]):
        q = points[j]
        if q < 0 or q >= 361:
            continue
        r, c = q // 19, q % 19
        solid = 0
        for k in range(4):
            t = nb[q, k]
            if t < 0:
                continue
            if board[t] == opp:
                out[j, 0] = 1
                if glibs[gid[t]] == 1:
                    out[j, 5] = 1
                elif glibs[gid[t]] == 2:
                    out[j, 6] = 1
            elif board[t] == own:
                solid = 1
                if glibs[gid[t]] == 1:
                    out[j, 7] = 1
        out[j, 1] = solid
        if solid:
            continue
        diag = 0
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 19 and 0 <= cc < 19 and board[rr * 19 + cc] == own:
                    diag = 1
        if diag:
            out[j, 2] = 1
            continue
        jump = 0
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 19 and 0 <= cc < 19 and board[rr * 19 + cc] == own \
                    and board[(r + dr // 2) * 19 + (c + dc // 2)] == 0:
                jump = 1
        if jump:
            out[j, 3] = 1
            continue
        for dr, dc in ((-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (1, -2), (-1, 2), (1, 2)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 19 and 0 <= cc < 19 and board[rr * 19 + cc] == own:
                out[j, 4] = 1
    return stamp


@njit(parallel=True, cache=True)
def rich_features(moves, move_off, pos_ply, pos_off, points, out):
    """moves: concatenated int64 moves of all games (move_off [G+1]); per game g the positions
    pos_off[g]:pos_off[g+1] have plies pos_ply (ascending) and candidate points points[pos] [P, 32];
    out uint8 [P, 32, 8] (zeroed by the caller)."""
    nb = _neighbours()
    for g in prange(len(move_off) - 1):
        board = np.zeros(361, np.int8)
        mark = np.zeros(361, np.int64)
        stack = np.empty(361, np.int64)
        members = np.empty(361, np.int64)
        gid = np.empty(361, np.int64)
        glibs = np.empty(361, np.int64)
        stamp = 0
        m0 = move_off[g]
        ply = 0
        for i in range(pos_off[g], pos_off[g + 1]):
            target = pos_ply[i]
            while ply < target:
                mv = moves[m0 + ply]
                if mv < 361:
                    stamp = _play(board, mv, 1 + (ply % 2), nb, mark, stamp, stack, members)
                ply += 1
            stamp = _features_at(board, 1 + (target % 2), nb, mark, stamp, stack, members, gid, glibs,
                                 points[i], out[i])


@njit(parallel=True, cache=True)
def boards_before(moves, move_off, pos_ply, pos_off, out):
    """the board (0 empty, 1 black, 2 white) before each listed ply; out int8 [P, 361]."""
    nb = _neighbours()
    for g in prange(len(move_off) - 1):
        board = np.zeros(361, np.int8)
        mark = np.zeros(361, np.int64)
        stack = np.empty(361, np.int64)
        members = np.empty(361, np.int64)
        stamp = 0
        m0 = move_off[g]
        ply = 0
        for i in range(pos_off[g], pos_off[g + 1]):
            while ply < pos_ply[i]:
                mv = moves[m0 + ply]
                if mv < 361:
                    stamp = _play(board, mv, 1 + (ply % 2), nb, mark, stamp, stack, members)
                ply += 1
            out[i] = board


def compute(game_moves: list[np.ndarray], game_plies: list[np.ndarray], points: np.ndarray) -> np.ndarray:
    """game_moves[g] int moves, game_plies[g] ascending plies; points [sum plies, 32] -> uint8 [P, 32, 8]."""
    moves = np.concatenate(game_moves).astype(np.int64)
    move_off = np.concatenate([[0], np.cumsum([len(m) for m in game_moves])]).astype(np.int64)
    pos_ply = np.concatenate(game_plies).astype(np.int64)
    pos_off = np.concatenate([[0], np.cumsum([len(p) for p in game_plies])]).astype(np.int64)
    assert len(pos_ply) == len(points)
    out = np.zeros((len(points), points.shape[1], N_RICH), np.uint8)
    rich_features(moves, move_off, pos_ply, pos_off, points.astype(np.int64), out)
    return out


def replay_boards(game_moves, game_plies) -> np.ndarray:
    moves = np.concatenate(game_moves).astype(np.int64)
    move_off = np.concatenate([[0], np.cumsum([len(m) for m in game_moves])]).astype(np.int64)
    pos_ply = np.concatenate(game_plies).astype(np.int64)
    pos_off = np.concatenate([[0], np.cumsum([len(p) for p in game_plies])]).astype(np.int64)
    out = np.zeros((len(pos_ply), 361), np.int8)
    boards_before(moves, move_off, pos_ply, pos_off, out)
    return out
