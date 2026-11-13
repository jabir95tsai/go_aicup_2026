"""S1: does an out-of-player discriminative feature add to H+C? (RESULTS 2026-11-12)

Every earlier stack of a discriminative model on H+C (A-024 in AH+C, SE3+H+C, the noA
ablation) fitted B2 on players the discriminative model had been trained on, so its features
were in-sample on the fit side and out-of-sample on unseen players. Valid-U players were never
seen by A-024, so inside Valid-U its features are out-of-player for fit and evaluation alike.

Design: Valid-U players split into 5 rank-stratified folds (seed 2027). Per fold, B2
(PriorMatched, prior = eligible-player share of the fitting folds) is fitted on the HD<=1
games of the other four folds and scores the E1 / E2 questions of the held-out players.
Arms differ only in the feature columns ([A 10 | H 30 | C 20 | HD1 flags 2], P3 komi):
  hc     H + C + flags (52)                     reference
  hca    + A-024 mean log-prob (62)
  hca2   + first two principal components of A (54; PCA fitted on the fitting folds)
  a      A + flags (12)                         how strong A is alone here
Gate (pre-registered): best of hca / hca2 minus hc, five-seed mean Equal-k >= +0.005 on E1 or
E2, the other set >= 0, and every E2 k-bucket >= -0.005. The fit pool is ~1,150 players, so
extra columns are penalised more than in the production pool; hca2 limits that.

    python -m scripts.strength.rank_clean_stack [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_b2_bagging as bag
import scripts.strength.rank_more_players as mp

OUT = ROOT / "outputs" / "rank" / "humansl" / "clean_stack"
FOLDS = 5
ARMS = ("hc", "hca", "hca2", "a")
SETS = ("E1", "E2")


def player_folds(games: pd.DataFrame) -> pd.Series:
    """player_id (str) -> fold, stratified by rank."""
    players = games.groupby(games["player_id"].astype(str))["rank_id"].first().sort_index()
    rng = np.random.default_rng(hf.SEED)
    fold = pd.Series(0, index=players.index)
    for _, ids in players.groupby(players):
        order = rng.permutation(len(ids))
        fold.loc[ids.index[order]] = np.arange(len(ids)) % FOLDS
    return fold


def arm_features(x: np.ndarray, fit_rows: np.ndarray) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    a, rest = x[:, :10], x[:, 10:]
    scaler = StandardScaler().fit(a[fit_rows])
    pca = PCA(2, random_state=hf.SEED).fit(scaler.transform(a[fit_rows]))
    return {"hc": rest, "hca": x, "hca2": np.hstack([pca.transform(scaler.transform(a)), rest]),
            "a": np.hstack([a, x[:, 60:]])}


def run_fold(seed: int, fold: int, eval_games, x, game_fold, sets, set_fold) -> dict:
    """Question probabilities of the held-out fold: {set: {arm: [q, 10]}}."""
    hs.SEED = seed
    fit_rows = np.flatnonzero((game_fold != fold) & (eval_games["handicap"] <= 1).to_numpy())
    fit = eval_games.iloc[fit_rows].reset_index(drop=True)
    prior = mp.player_prior(fit)
    features = arm_features(x, fit_rows)
    out = {name: {} for name in sets}
    for arm in ARMS:
        model = mp.PriorMatched(prior).fit(fit, pd.DataFrame(), features[arm][fit_rows])
        for name, q in sets.items():
            held = q[set_fold[name] == fold]
            out[name][arm] = model.probabilities(features[arm], eval_games["game_index"].to_numpy(), held)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    from joblib import Parallel, delayed

    games, _, eval_games, _, x = bag.data()
    assert x.shape[1] == 62, x.shape
    fold = player_folds(eval_games)
    game_fold = fold.loc[eval_games["player_id"].astype(str)].to_numpy()
    sets = {k: v.reset_index(drop=True) for k, v in hf.question_sets(games, rebuild=False).items() if k in SETS}
    set_fold = {name: fold.loc[q["player_id"].astype(str)].to_numpy() for name, q in sets.items()}
    result = {"players": int(len(fold)), "players_per_fold": np.bincount(fold, minlength=FOLDS).tolist(),
              "fit_games_hd01": int((eval_games["handicap"] <= 1).sum()),
              "questions": {n: len(q) for n, q in sets.items()}, "seeds": {}}
    print(json.dumps(result), flush=True)

    tasks = [(seed, f) for seed in hf.SEEDS for f in range(FOLDS)]
    parts = Parallel(n_jobs=args.jobs)(delayed(run_fold)(s, f, eval_games, x, game_fold, sets, set_fold)
                                       for s, f in tasks)
    for seed in hf.SEEDS:
        entry = {}
        for name, q in sets.items():
            k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
            scores = {}
            for arm in ARMS:
                p = np.empty((len(q), 10))
                for (s, f), part in zip(tasks, parts):
                    if s == seed:
                        p[set_fold[name] == f] = part[name][arm]
                scores[arm] = official_scores(score_rank(p), truth)
            cell = {arm: hf.by_k(s, k) for arm, s in scores.items()}
            for arm in ARMS[1:]:
                cell[f"{arm}-hc"] = {"by_k": {str(b): round(float(scores[arm][k == b].mean()
                                                                  - scores["hc"][k == b].mean()), 4) for b in K_BUCKETS},
                                     "equal_k": round(hf.equal_k(scores[arm], k) - hf.equal_k(scores["hc"], k), 4)}
                if seed == hf.SEED:
                    cell[f"{arm}-hc"]["bootstrap_95"] = hf.bootstrap(scores["hc"], scores[arm], q)
            entry[name] = cell
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {a: e[a]["equal_k"] for a in e} for n, e in entry.items()}), flush=True)

    diffs = [f"{a}-hc" for a in ARMS[1:]]
    result["mean"] = {n: {a: round(float(np.mean([result["seeds"][str(s)][n][a]["equal_k"] for s in hf.SEEDS])), 4)
                          for a in list(ARMS) + diffs} for n in sets}
    result["mean_by_k"] = {n: {d: {str(b): round(float(np.mean([result["seeds"][str(s)][n][d]["by_k"][str(b)]
                                                                 for s in hf.SEEDS])), 4) for b in K_BUCKETS}
                               for d in diffs} for n in sets}
    gate = {}
    for arm in ("hca", "hca2"):
        e1, e2 = result["mean"]["E1"][f"{arm}-hc"], result["mean"]["E2"][f"{arm}-hc"]
        worst = min(result["mean_by_k"]["E2"][f"{arm}-hc"].values())
        gate[arm] = {"E1": e1, "E2": e2, "min_E2_k": worst,
                     "pass": bool(max(e1, e2) >= 0.005 and min(e1, e2) >= 0 and worst >= -0.005)}
    result["gate"] = gate
    print(json.dumps({"mean": result["mean"], "mean_by_k": result["mean_by_k"], "gate": gate}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
