import numpy as np
import pandas as pd
import pytest

from scripts.strength import rank_opp_llr_mix as mix


def test_average_aligns_both_llr_matrices_and_rejects_mismatched_rows():
    base = {"qrows": np.array([1, 2]), "masked_rows": np.array([1, 2]), "seen_rows": np.array([1]),
            "masked_llr": np.array([[2, -4], [0, 0]]), "seen_llr": np.array([[0, 4]])}
    rich = base | {"masked_llr": np.array([[4, 2], [2, 4]]), "seen_llr": np.array([[4, 2]])}
    result = mix.mix_llr(base, rich, width=2)
    np.testing.assert_array_equal(result["masked_llr"], [[3, -1], [1, 2]])
    np.testing.assert_array_equal(result["seen_llr"], [[2, 3]])
    with pytest.raises(ValueError, match="query rows differ"):
        mix.mix_llr(base, rich | {"masked_rows": np.array([2, 1])}, width=2)
    with pytest.raises(ValueError, match="seen LLR matrices"):
        mix.mix_llr(base, rich | {"seen_llr": np.ones((1, 1))}, width=2)


def test_independent_questions_exclude_players_from_all_buckets_and_fit():
    def question(pid, k, start, color):
        return {"question_id": 0, "player_id": pid, "rank_id": 0, "k": k, "color": color,
                "games": np.arange(start, start + k)}

    games = pd.DataFrame([{"game_index": i, "player_id": pid, "player_color": color, "rank_id": 0,
                           "handicap": hd}
                          for pid, color, start, stop, hd in (
                              ("e1", 0, 0, 10, 2), ("e2", 1, 10, 15, 0),
                              ("fit", 1, 15, 20, 0), ("new", 1, 20, 26, 0),
                              ("too_few", 1, 26, 30, 0), ("too_few", 1, 30, 31, 2),
                          ) for i in range(start, stop)])
    sets = {"E1": pd.DataFrame([question("e1", 10, 0, "B")]),
            "E2": pd.DataFrame([question("e2", 5, 10, 1)])}
    got = mix.validation_questions(sets, games, ["fit"])
    assert got["player_id"].tolist() == ["e1", "e2", "new"]
    assert got["has_hd2plus"].tolist() == [True, False, False]
    assert len(np.unique(got.iloc[-1]["games"])) == 5
    pd.testing.assert_frame_equal(got, mix.validation_questions(sets, games.sample(frac=1), ["fit"]))


def test_summary_pairs_against_the_same_seed():
    frame = pd.DataFrame({"arm": ["opp7", "mix", "opp7", "mix"],
                          "seed": [1, 1, 2, 2], "k": 5, "score": [0.25, 0.5, 0.5, 0.5]})
    result = mix.summarize(frame, (5,))
    assert result["mean_delta"]["mix"] == {5: 0.125, "equal_k": 0.125}
    assert result["delta_by_seed"]["mix"] == {1: {5: 0.25, "equal_k": 0.25}, 2: {5: 0, "equal_k": 0}}
