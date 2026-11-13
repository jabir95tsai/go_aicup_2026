"""C1 follow-up: choose B2 regularisation inside the fitting set (RESULTS A-052), CPU only.

Fitting games as in A-051 (Valid-S S-fit + S-extra, 38,589). Two-fold cross-
validation by player inside them: B2 fitted on one fold, scored on synthetic
questions of the other fold's players (per k, 1,000 questions: a random eligible
player, k of their games). PCA is fitted on the training fold only. Grid:
C in {0.003, 0.01, 0.03, 0.1, 0.3, 1}; for views with embeddings, PCA
components in {16, 32, 64, 128}. S-select is not used for the choice.
Then each view's chosen configuration is fitted on all fitting games and scored
once on the 1,000 S-select questions.
Pre-registered gate: AH_Emb (its chosen config) - AH (its chosen config) >= +0.01.

    python -m scripts.strength.emb_regularization
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from scripts.strength.emb_stage1 import table
from scripts.strength.humansl_pilot import a024_features
from scripts.strength.humansl_questions import by_k
from go_aicup.features.humansl_torch import run_embeddings
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs"
CS, DIMS, HELD_OUT_PER_K, MIN_GAIN = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0), (16, 32, 64, 128), 1000, 0.01


def synthetic_questions(games: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    by_player = games.groupby(games["player_id"].astype(str))
    rows = []
    for k in K_BUCKETS:
        eligible = [(rank, idx) for (_, idx), rank in
                    zip(by_player["game_index"].apply(np.asarray).items(), by_player["rank_id"].first()) if len(idx) >= k]
        for p in rng.integers(0, len(eligible), HELD_OUT_PER_K):
            rank, idx = eligible[p]
            rows.append({"games": rng.choice(idx, k, replace=False), "k": k, "rank_id": int(rank)})
    return pd.DataFrame(rows)


def features(frame, embedding, human, pca, view):
    """embedding = (array [games, 3072], pd.Series game_index -> row); arrays are memmapped by joblib."""
    array, rows = embedding
    parts = {"A": lambda: a024_features(frame["game_index"]), "H": lambda: frame[human].to_numpy(),
             "Emb": lambda: pca.transform(array[rows.loc[frame["game_index"]].to_numpy()])}
    return np.hstack([parts[p]() for p in view])


def evaluate(train, test, questions, embedding, human, view, dims, c) -> float:
    array, rows = embedding
    pca = PCA(dims, random_state=SEED).fit(array[rows.loc[train["game_index"]].to_numpy()]) if "Emb" in view else None
    model = RankModel("B2", C=c).fit(train, pd.DataFrame(), features(train, embedding, human, pca, view))
    pred = score_rank(model.probabilities(features(test, embedding, human, pca, view),
                                          test["game_index"].to_numpy(), questions))
    return by_k(official_scores(pred, questions["rank_id"].to_numpy()), questions["k"].to_numpy())["equal_k"]


def main() -> int:
    fit = pd.concat([table(OUT / "humansl_q1000", ["S-fit"]), table(OUT / "humansl_vs_rest", ["S-fit", "S-extra"])],
                    ignore_index=True)
    test = table(OUT / "humansl_qsel", ["S-select"])
    s_questions = pd.read_parquet(OUT / "humansl_qsel" / "questions.parquet")
    frame = run_embeddings(OUT / "emb_stage1")
    embedding = (frame.drop(columns="game_index").to_numpy(np.float32),
                 pd.Series(np.arange(len(frame)), index=frame["game_index"].to_numpy()))
    del frame
    human = [c for c in fit.columns if c.startswith(("own_", "opp_"))]
    players = np.sort(fit["player_id"].astype(str).unique())
    half = set(np.random.default_rng(SEED + 3).permutation(players)[:len(players) // 2])
    fold = fit["player_id"].astype(str).isin(half).to_numpy()
    folds = []
    for side in (True, False):
        train, held = fit[fold != side].reset_index(drop=True), fit[fold == side].reset_index(drop=True)
        folds.append((train, held, synthetic_questions(held, SEED + int(side))))
    views = {"AH": (("A", "H"), (None,)), "AH_Emb": (("A", "H", "Emb"), DIMS)}
    grid = [(name, dims, c) for name, (_, dim_list) in views.items() for dims, c in itertools.product(dim_list, CS)]
    scores = Parallel(n_jobs=12, max_nbytes="1M")(delayed(evaluate)(train, held, q, embedding, human, views[name][0], dims, c)
                                 for name, dims, c in grid for train, held, q in folds)
    cv = {}
    for i, (name, dims, c) in enumerate(grid):
        cv.setdefault(name, {})[f"dims={dims},C={c}"] = round(float(np.mean(scores[2 * i:2 * i + 2])), 4)
    result = {"cv_equal_k": cv, "chosen": {}, "S-select": {}}
    for name, (view, _) in views.items():
        best = max(cv[name], key=cv[name].get)
        dims = None if best.startswith("dims=None") else int(best.split(",")[0].split("=")[1])
        c = float(best.split("C=")[1])
        result["chosen"][name] = {"dims": dims, "C": c, "cv": cv[name][best]}
        pca = PCA(dims, random_state=SEED).fit(embedding[0][embedding[1].loc[fit["game_index"]].to_numpy()]) if dims else None
        model = RankModel("B2", C=c).fit(fit, pd.DataFrame(), features(fit, embedding, human, pca, view))
        pred = score_rank(model.probabilities(features(test, embedding, human, pca, view),
                                              test["game_index"].to_numpy(), s_questions))
        result["S-select"][name] = by_k(official_scores(pred, s_questions["rank_id"].to_numpy()), s_questions["k"].to_numpy())
        print(name, result["chosen"][name], result["S-select"][name], flush=True)
    gain = result["S-select"]["AH_Emb"]["equal_k"] - result["S-select"]["AH"]["equal_k"]
    result["gate"] = {"gain_AH_Emb_minus_AH": round(gain, 4), "min_gain": MIN_GAIN, "pass": bool(gain >= MIN_GAIN)}
    print(json.dumps(result["gate"]))
    (OUT / "emb_regularization.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
