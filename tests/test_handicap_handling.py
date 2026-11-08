"""HD[n] handling for competition-1 stores (CLAUDE.md §1: HD>=2 labels reversed, stones missing)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import _parse_batch
from go_aicup.tasks.identity.predict import handicap_ok

import day0_audit


def test_parse_records_hd_value_or_minus_one():
    sgfs = ["(;SZ[19]HD[0];B[pd];W[dp])", "(;SZ[19]HD[3];B[qc];W[cq])", "(;SZ[19];B[pd];W[dd])"]
    for lenient in (False, True):
        hd = _parse_batch(sgfs, lenient)[-1]
        assert hd.tolist() == [0, 3, -1]


def test_handicap_ok_drops_only_hd2_plus():
    meta = pd.DataFrame({"hd": np.array([-1, 0, 1, 2, 9], np.int8)})
    assert handicap_ok(meta).tolist() == [True, True, True, False, False]
    assert handicap_ok(pd.DataFrame({"x": [1, 2]})).tolist() == [True, True]


def test_day0_hd_report_flags_moves_on_stone_points():
    # game 0: HD2, first move on pd (a stone point: stones would be on the board) -> flagged
    # game 1: HD2, white approach first (labels reversed, stones missing) -> not flagged
    pd_point, qc_point = 3 * 19 + 15, 2 * 19 + 16
    moves = np.array([pd_point, 100, qc_point, 101], np.int16)
    meta = pd.DataFrame({"offset": [0, 2], "length": [2, 2], "hd": np.array([2, 2], np.int8),
                         "question_id": ["q1", "q1"]})
    report = day0_audit.hd_report("pid_public", moves, meta)
    assert report["hd_counts"] == {"2": 2}
    assert report["hd2plus_games_with_early_move_on_stone_point"] == 1
    assert report["questions_all_hd2plus"] == 1


def test_hd_scores_prefers_candidate_with_matching_handicap_habit():
    from go_aicup.tasks.identity.predict import hd_scores
    pool_hd = np.array([0, 0, 0, 0, 9, 9, 9, 0])
    players = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    q_hd = np.array([9, 9, 0])
    s = hd_scores(pool_hd, players, np.array(["a", "b"]), q_hd, [np.array([0, 1]), np.array([2])])
    assert s.shape == (2, 2)
    assert s[0, 1] > s[0, 0]      # two HD9 games -> b
    assert s[1, 0] > s[1, 1]      # one HD0 game -> a
