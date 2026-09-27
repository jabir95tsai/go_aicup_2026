"""Per-game move statistics from a full board replay (captures, ataris, liberties...).

A numba board engine replays every game of a store once and accumulates, for
each side (black / white) separately, counts such as captures, self-ataris,
contact moves and line usage. The result is re-keyed to the labelled player's
perspective (`me_*` = the labelled player, `op_*` = the opponent), so it
follows the colour normalisation rule of CLAUDE.md section 4.

These are descriptive statistics for EDA; they carry no ids and are derived
from the moves alone, so they are legal features for both competitions.

Usage:
    python src/game_stats.py            # train store -> data/processed/eda/game_stats_train.parquet
    python src/game_stats.py --store pool_public
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from data import GameStore  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "processed" / "eda"

SIZE = 19
N_POINTS = SIZE * SIZE
PASS = N_POINTS
EARLY_PLIES = 60  # "opening" window for the low-line statistic


def _neighbours() -> tuple[np.ndarray, np.ndarray]:
    nb = np.full((N_POINTS, 4), -1, dtype=np.int16)
    cnt = np.zeros(N_POINTS, dtype=np.int8)
    for p in range(N_POINTS):
        r, c = divmod(p, SIZE)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < SIZE and 0 <= cc < SIZE:
                nb[p, cnt[p]] = rr * SIZE + cc
                cnt[p] += 1
    return nb, cnt


NB, NB_CNT = _neighbours()

# per-side accumulator layout (sums; ratios are formed in pandas)
SIDE_FIELDS = [
    "n_moves",        # moves played by this side
    "cap_events",     # moves that captured >= 1 stone
    "cap_stones",     # stones removed by this side's moves
    "atari_given",    # moves leaving an adjacent enemy group on 1 liberty
    "self_atari",     # moves leaving the own group on 1 liberty (no capture)
    "contact",        # moves touching an enemy stone
    "own_adj",        # moves touching an own stone
    "d_opp_sum",      # euclidean distance to the opponent's previous move
    "d_opp_n",
    "local_resp",     # chebyshev <= 2 from the opponent's previous move
    "tenuki",         # chebyshev >= 6 from the opponent's previous move
    "d_own_sum",      # euclidean distance to this side's previous move
    "d_own_n",
    "line1", "line2", "line3", "line4", "line5p",
    "early_low",      # plies < EARLY_PLIES on line 1-2
    "early_n",
    "libs_sum",       # liberties of the placed stone's group (capped at 10)
    "low_libs",       # placed group has <= 2 liberties
    "refill",         # point was occupied earlier in the game (i.e. after a capture)
    "cap_single",     # capture events of exactly one stone (ko-like)
    "stones_end",     # own stones on the board at the end
    "cap_big",        # capture events of >= 5 stones
    "escape",         # move extends an own group that was in atari
]
NF = len(SIDE_FIELDS)
F = {name: i for i, name in enumerate(SIDE_FIELDS)}


@njit(cache=True)
def _flood(board, start, mark, lmark, stamp, stack, members, nb, nbc):
    """Group containing `start`: returns (n_members, n_liberties); fills `members`."""
    color = board[start]
    stamp[0] += 1
    s = stamp[0]
    mark[start] = s
    stack[0] = start
    top = 1
    n = 0
    libs = 0
    while top > 0:
        top -= 1
        p = stack[top]
        members[n] = p
        n += 1
        for j in range(nbc[p]):
            q = nb[p, j]
            v = board[q]
            if v == 0:
                if lmark[q] != s:
                    lmark[q] = s
                    libs += 1
            elif v == color and mark[q] != s:
                mark[q] = s
                stack[top] = q
                top += 1
    return n, libs


@njit(parallel=True, cache=True)
def _replay(moves, offsets, lengths, nb, nbc, early_plies, out):
    n_games = offsets.shape[0]
    for g in prange(n_games):
        board = np.zeros(N_POINTS, np.int8)
        ever = np.zeros(N_POINTS, np.bool_)
        mark = np.zeros(N_POINTS, np.int32)
        lmark = np.zeros(N_POINTS, np.int32)
        stamp = np.zeros(1, np.int32)
        stack = np.zeros(N_POINTS, np.int16)
        members = np.zeros(N_POINTS, np.int16)
        off = offsets[g]
        for i in range(lengths[g]):
            p = moves[off + i]
            side = i & 1
            if p >= N_POINTS:  # pass (none in the train store, kept for safety)
                out[g, side, 0] += 1.0
                continue
            color = side + 1
            opp = 3 - color
            f = out[g, side]
            f[0] += 1.0
            r = p // SIZE
            c = p % SIZE
            line = min(min(r, c), min(SIZE - 1 - r, SIZE - 1 - c))
            f[13 + min(line, 4)] += 1.0
            if i < early_plies:
                f[19] += 1.0
                if line <= 1:
                    f[18] += 1.0
            if ever[p]:
                f[22] += 1.0
            # neighbourhood before the move
            contact = False
            own_adj = False
            escape = False
            for j in range(nbc[p]):
                q = nb[p, j]
                v = board[q]
                if v == opp:
                    contact = True
                elif v == color:
                    own_adj = True
                    if not escape:
                        _, libs = _flood(board, q, mark, lmark, stamp, stack, members, nb, nbc)
                        if libs == 1:
                            escape = True
            if contact:
                f[5] += 1.0
            if own_adj:
                f[6] += 1.0
            if escape:
                f[26] += 1.0
            # distances to the previous moves
            if i >= 1:
                q = moves[off + i - 1]
                if q < N_POINTS:
                    dr = abs(r - q // SIZE)
                    dc = abs(c - q % SIZE)
                    f[7] += np.sqrt(dr * dr + dc * dc)
                    f[8] += 1.0
                    cheb = max(dr, dc)
                    if cheb <= 2:
                        f[9] += 1.0
                    if cheb >= 6:
                        f[10] += 1.0
            if i >= 2:
                q = moves[off + i - 2]
                if q < N_POINTS:
                    dr = abs(r - q // SIZE)
                    dc = abs(c - q % SIZE)
                    f[11] += np.sqrt(dr * dr + dc * dc)
                    f[12] += 1.0
            # play, then resolve captures
            board[p] = color
            ever[p] = True
            captured = 0
            for j in range(nbc[p]):
                q = nb[p, j]
                if board[q] == opp:
                    n, libs = _flood(board, q, mark, lmark, stamp, stack, members, nb, nbc)
                    if libs == 0:
                        for m in range(n):
                            board[members[m]] = 0
                        captured += n
            if captured > 0:
                f[1] += 1.0
                f[2] += captured
                if captured == 1:
                    f[23] += 1.0
                if captured >= 5:
                    f[25] += 1.0
            atari = False
            for j in range(nbc[p]):
                q = nb[p, j]
                if board[q] == opp:
                    _, libs = _flood(board, q, mark, lmark, stamp, stack, members, nb, nbc)
                    if libs == 1:
                        atari = True
                        break
            if atari:
                f[3] += 1.0
            _, libs = _flood(board, p, mark, lmark, stamp, stack, members, nb, nbc)
            f[20] += min(libs, 10)
            if libs <= 2:
                f[21] += 1.0
            if libs == 1 and captured == 0:
                f[4] += 1.0
        n_black = 0
        n_white = 0
        for p in range(N_POINTS):
            if board[p] == 1:
                n_black += 1
            elif board[p] == 2:
                n_white += 1
        out[g, 0, 24] = n_black
        out[g, 1, 24] = n_white


def replay(store: GameStore) -> np.ndarray:
    """(n_games, 2, NF) float32 sums, side 0 = black, side 1 = white."""
    moves = np.ascontiguousarray(store.moves)
    offsets = store.meta["offset"].to_numpy(np.int64)
    lengths = store.lengths.astype(np.int64)
    # every game in the train store is black-first and alternating (RESULTS A-002)
    first = store.meta["first_color"].to_numpy()
    assert (first == 0).all(), "engine assumes black moves first"
    out = np.zeros((len(store), 2, NF), dtype=np.float32)
    _replay(moves, offsets, lengths, NB, NB_CNT, EARLY_PLIES, out)
    return out


def to_frame(store: GameStore, sums: np.ndarray) -> pd.DataFrame:
    """Labelled-player perspective: me_* / op_* ratios + game-level columns."""
    pc = store.player_color.astype(np.int64)
    assert (pc >= 0).all(), "store has no colour column"
    rows = np.arange(len(store))
    me = sums[rows, pc]
    op = sums[rows, 1 - pc]
    length = store.lengths.astype(np.float32)
    frame = {"length": length,
             # the side that moved last: in a resigned game this is usually the winner
             "me_last": ((length - 1) % 2 == pc).astype(np.float32)}
    for tag, s in (("me", me), ("op", op)):
        n = np.maximum(s[:, F["n_moves"]], 1)
        frame[f"{tag}_cap_rate"] = s[:, F["cap_events"]] / n
        frame[f"{tag}_cap_stones"] = s[:, F["cap_stones"]] / n
        frame[f"{tag}_cap_single"] = s[:, F["cap_single"]] / n
        frame[f"{tag}_cap_big"] = s[:, F["cap_big"]]
        frame[f"{tag}_atari"] = s[:, F["atari_given"]] / n
        frame[f"{tag}_self_atari"] = s[:, F["self_atari"]] / n
        frame[f"{tag}_escape"] = s[:, F["escape"]] / n
        frame[f"{tag}_contact"] = s[:, F["contact"]] / n
        frame[f"{tag}_own_adj"] = s[:, F["own_adj"]] / n
        frame[f"{tag}_d_opp"] = s[:, F["d_opp_sum"]] / np.maximum(s[:, F["d_opp_n"]], 1)
        frame[f"{tag}_local"] = s[:, F["local_resp"]] / np.maximum(s[:, F["d_opp_n"]], 1)
        frame[f"{tag}_tenuki"] = s[:, F["tenuki"]] / np.maximum(s[:, F["d_opp_n"]], 1)
        frame[f"{tag}_d_own"] = s[:, F["d_own_sum"]] / np.maximum(s[:, F["d_own_n"]], 1)
        for name in ("line1", "line2", "line3", "line4", "line5p"):
            frame[f"{tag}_{name}"] = s[:, F[name]] / n
        frame[f"{tag}_early_low"] = s[:, F["early_low"]] / np.maximum(s[:, F["early_n"]], 1)
        frame[f"{tag}_libs"] = s[:, F["libs_sum"]] / n
        frame[f"{tag}_low_libs"] = s[:, F["low_libs"]] / n
        frame[f"{tag}_refill"] = s[:, F["refill"]] / n
    stones_me = me[:, F["stones_end"]]
    stones_op = op[:, F["stones_end"]]
    frame["stone_share_end"] = stones_me / np.maximum(stones_me + stones_op, 1)
    frame["me_lost_stones"] = op[:, F["cap_stones"]] / np.maximum(me[:, F["n_moves"]], 1)
    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in frame.items()})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="train")
    args = parser.parse_args()
    store = GameStore(args.store)
    t0 = time.time()
    sums = replay(store)
    t1 = time.time()
    frame = to_frame(store, sums)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"game_sums_{args.store}.npy", sums)
    frame.to_parquet(OUT_DIR / f"game_stats_{args.store}.parquet", index=False)
    print(f"{args.store}: {len(store):,} games, {int(store.lengths.sum()):,} moves, "
          f"replay {t1 - t0:.1f}s ({store.lengths.sum() / (t1 - t0) / 1e6:.1f}M moves/s)")
    print(frame.describe().T.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
