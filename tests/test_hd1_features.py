"""HD1 flags of the P1+P2 submission path (humansl_predict.hd1_features)."""
from __future__ import annotations

import pandas as pd
import pytest

from go_aicup.tasks.strength.humansl_predict import HD1_COLUMNS, hd1_features


def _work(tmp_path, handicaps=(0, 1, 1, 0), raw_colors=None):
    # Two questions, two games each; store rows are listed out of cache order on purpose.
    meta = pd.DataFrame({"question_id": ["q1", "q1", "q2", "q2"], "slot": [1, 2, 1, 2],
                         "color": ["B", "B", "W", "W"]})
    work = tmp_path / "rank_public"
    (work / "cache").mkdir(parents=True)
    meta.to_parquet(work / "store_meta.parquet")
    pd.DataFrame({"store_row": [3, 0, 1, 2], "game_index": [10, 11, 12, 13]}).to_parquet(work / "cache" / "games.parquet")
    colors = raw_colors or ["B", "B", "W", "W"]
    raw = pd.DataFrame({"question_id": ["q1", "q2"], "num_games": ["2", "2"],
                        "sgf_1": [f"(;HD[{handicaps[0]}]SZ[19];B[pd])", f"(;HD[{handicaps[2]}]SZ[19];B[dd])"],
                        "sgf_2": [f"(;HD[{handicaps[1]}]SZ[19];B[qd])", f"(;HD[{handicaps[3]}]SZ[19];B[dp])"],
                        "color_1": [colors[0], colors[2]], "color_2": [colors[1], colors[3]]})
    raw.to_csv(tmp_path / "raw.csv", index=False)
    return work, tmp_path / "raw.csv"


def test_flags_follow_store_rows_and_colour(tmp_path):
    work, raw = _work(tmp_path)
    table = hd1_features(work, raw)
    assert list(table.columns) == HD1_COLUMNS
    # game 10 = store row 3 (q2 slot 2, W, HD0); 11 = q1 slot 1 (B, HD0); 12 = q1 slot 2 (B, HD1); 13 = q2 slot 1 (W, HD1)
    assert table.loc[[10, 11, 12, 13]].values.tolist() == [[0, 0], [0, 0], [1, 0], [0, 1]]


def test_rejects_handicap_games(tmp_path):
    work, raw = _work(tmp_path, handicaps=(0, 2, 1, 0))
    with pytest.raises(ValueError, match="HD>=2"):
        hd1_features(work, raw)


def test_rejects_colour_mismatch(tmp_path):
    work, raw = _work(tmp_path, raw_colors=["B", "W", "W", "W"])
    with pytest.raises(ValueError, match="colour"):
        hd1_features(work, raw)
