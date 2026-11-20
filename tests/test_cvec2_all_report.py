import numpy as np
import pandas as pd

from go_aicup.tasks.strength.cvec2_policy import all_report


def test_paired_bootstrap_uses_the_same_players_and_keeps_k_buckets():
    questions = pd.DataFrame({"player_id": ["a", "b"] * 4,
                              "k": np.repeat([5, 10, 15, 20], 2)})
    scores = np.arange(8, dtype=float) / 10
    intervals = all_report.bootstrap(questions, np.stack([scores, scores + .1]))
    for k, limits in intervals[0].items():
        np.testing.assert_allclose(intervals[1][k], np.array(limits) + .1)


def test_candidate_gate_uses_unrounded_thresholds_and_paired_controls():
    arm = "A/CNN-R10-D"
    weighted = arm + "-weighted"
    base = {str(k): -.005 for k in [5, 10, 15, 20]} | {"equal_k": .005}
    deltas = {f"{arm} minus A": base,
              f"{arm} minus A/CNN-R10-C": {"equal_k": 1e-10},
              f"{arm} minus {all_report.C_NORMAL}": {"equal_k": 1e-10},
              f"{weighted} minus {all_report.C_NORMAL}": {"equal_k": 1e-10},
              f"{weighted} minus A": base,
              f"{weighted} minus {arm}": base}
    assert not all_report.candidate_gate(arm, deltas, None)
    assert not all_report.candidate_gate(weighted, deltas, {"own": {"p05": .5}, "opp": {"p05": .5}})
    failures = all_report.candidate_gate(weighted, deltas, {"own": {"p05": .499999}, "opp": {"p05": .5}})
    assert len(failures) == 1 and failures[0].startswith("own")
