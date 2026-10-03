"""Board-replay engine used by the EDA statistics (go_aicup.features.game_stats)."""

from __future__ import annotations

import numpy as np

from go_aicup.data import sgf as sgflib
from go_aicup.features import game_stats as gs


def replay(points: list[str]) -> np.ndarray:
    moves = np.array([sgflib.point_to_index(p) for p in points], np.int16)
    out = np.zeros((1, 2, gs.NF), np.float32)
    gs._replay(moves, np.array([0]), np.array([len(moves)]), gs.NB, gs.NB_CNT,
               gs.EARLY_PLIES, out)
    return out[0]


def test_corner_capture_and_self_atari():
    # B ba, W aa (self-atari in the corner), B pp, W qq, B ab captures aa, W pd
    black, white = replay(["ba", "aa", "pp", "qq", "ab", "pd"])
    f = gs.F
    assert black[f["cap_events"]] == 1 and black[f["cap_stones"]] == 1
    assert black[f["cap_single"]] == 1
    assert white[f["self_atari"]] == 1 and black[f["self_atari"]] == 0
    assert black[f["stones_end"]] == 3 and white[f["stones_end"]] == 2
    # aa answers ba locally, qq answers pp locally, pd is far from ab
    assert white[f["local_resp"]] == 2 and white[f["tenuki"]] == 1


def test_refill_after_capture():
    # same opening as above; after W aa is captured, B fills the emptied point aa
    black, _ = replay(["ba", "aa", "pp", "qq", "ab", "pd", "aa"])
    assert black[gs.F["refill"]] == 1


def test_atari_given():
    # B dd, W ed, B fd (W ed now has ec, ee), B ec -> W ed in atari
    black, _ = replay(["dd", "ed", "fd", "qq", "ec"])
    assert black[gs.F["atari_given"]] == 1
