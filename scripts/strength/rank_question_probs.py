"""Save the current base's per-question 10-rank probabilities (C v2 base, 751959-cvec2.csv).

Refits exactly what `rank_cvec_v2 export` fits (seed 2027), checks the score_rank decode reproduces
submissions/rank/751959-cvec2.csv, and writes outputs/rank/humansl/question_probs/cvec2.parquet
(question_id, num_games, p_0..p_9) for the Public / Private test questions. CPU only.

    python -m scripts.strength.rank_question_probs
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_cvec_v2 as cv2
import scripts.strength.rank_more_players as more

OUT = ROOT / "outputs" / "rank" / "humansl" / "question_probs"
REFERENCE = ROOT / "submissions" / "rank" / "751959-cvec2.csv"


def main() -> int:
    import pickle
    games, fit, eval_games, xs = cv2.data()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    xf, xe = xs["v2"]
    model = more.PriorMatched(prior).fit(stacked, pd.DataFrame(), np.vstack([xf[hd01], xe[u]]))
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    frames = []
    for part in ("public", "private"):
        _, x2, index = cv2.test_x(part, columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        p = model.probabilities(x2, index, questions)
        frame = pd.DataFrame(p, columns=[f"p_{r}" for r in range(10)])
        frame.insert(0, "question_id", questions["question_id"].to_numpy())
        if "num_games" in questions:
            frame.insert(1, "num_games", questions["num_games"].to_numpy())
        frame["pred_rank"] = [RANKS[r] for r in score_rank(p)]
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    ref = pd.read_csv(REFERENCE, dtype=str).set_index("question_id")["pred_rank"]
    diff = int((out.set_index("question_id")["pred_rank"] != ref.loc[out["question_id"]].to_numpy()).sum())
    if diff:
        raise ValueError(f"refit differs from {REFERENCE.name} on {diff} questions")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "cvec2.parquet")
    print(f"saved {len(out)} questions; reproduces {REFERENCE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
