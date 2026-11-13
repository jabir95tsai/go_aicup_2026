"""E1: hand-crafted per-game stats + LightGBM (competition 2).

Same features, questions and player-disjoint split as the A-005/A-006 linear
probe (scripts/data/eda_report.py section_probe); one variable changes per variant.
  eval = rank_questions f4 + holdout (14,000 questions, players never trained on)
Variants:
  lr         LR on f0-f3 question means (reproduces A-006 0.412)
  gbdt       LightGBM on question means, fit f0-f2, early stop f3      (vs lr: model)
  gbdt_std   gbdt + per-question std features                          (vs gbdt: features)
  gbdt_game  LightGBM on single games of fold 0-2 players, early stop
             fold 3, question = mean log-prob over its k games         (vs gbdt: training unit)

  python -m go_aicup.tasks.strength.gbdt_e1
"""

from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from go_aicup.data import cv
from go_aicup.metrics import format_report, rank_report, score_rank
from go_aicup.paths import ROOT

STATS = ROOT / "data" / "processed" / "eda" / "game_stats_train.parquet"
META = ROOT / "data" / "processed" / "train_meta.parquet"
OUT = ROOT / "outputs" / "rank_gbdt_e1"
SEED = 2026
PARAMS = dict(objective="multiclass", num_class=10, learning_rate=0.03, num_leaves=31,
              min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, seed=SEED, verbose=-1, num_threads=0)


def load_questions() -> pd.DataFrame:
    parts = []
    for name in ["f0", "f1", "f2", "f3", "f4", "holdout"]:
        q = cv.load(f"rank_questions_{name}")
        q["fold"] = name
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def question_features(x: np.ndarray, questions: pd.DataFrame, with_std: bool) -> np.ndarray:
    blocks = []
    for games in questions["games"]:
        g = x[np.asarray(games)]
        blocks.append(np.concatenate([g.mean(0), g.std(0)]) if with_std else g.mean(0))
    # k is a legal input (num_games is given in the test file)
    return np.column_stack([np.stack(blocks), np.log(questions["k"].to_numpy())])


def fit_lgb(x, y, x_stop, y_stop) -> lgb.Booster:
    return lgb.train(PARAMS, lgb.Dataset(x, y), num_boost_round=5000,
                     valid_sets=[lgb.Dataset(x_stop, y_stop)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def mean_logprob(game_proba: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Equal-weight mean of per-game log-probs within each question, renormalised."""
    logp = np.log(np.clip(game_proba, 1e-9, 1.0))
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    mean = np.add.reduceat(logp, starts, axis=0) / sizes[:, None]
    p = np.exp(mean - mean.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def main() -> int:
    t0 = time.time()
    stats = pd.read_parquet(STATS)
    # same column choice as the A-006 probe: me_* + op_* + length
    cols = [c for c in stats.columns if c.startswith(("me_", "op_"))] + ["length"]
    x = stats[cols].to_numpy(np.float32)
    q = load_questions()
    y, k, fold = q["rank_id"].to_numpy(), q["k"].to_numpy(), q["fold"].to_numpy()
    ev = np.isin(fold, ["f4", "holdout"])
    fit, stop = np.isin(fold, ["f0", "f1", "f2"]), fold == "f3"
    # the split must stay player-disjoint, otherwise the score measures memorised players
    pid = q["player_id"].to_numpy()
    assert not set(pid[ev]) & set(pid[~ev]) and not set(pid[fit]) & set(pid[stop])

    OUT.mkdir(parents=True, exist_ok=True)
    reports, summary = [], {}

    def record(name: str, proba: np.ndarray, info: dict) -> None:
        np.save(OUT / f"{name}_proba.npy", proba)
        for decode, pred in [("score_rank", score_rank(proba)), ("argmax", proba.argmax(1))]:
            rep = rank_report(pred, y[ev], k[ev])
            rep.insert(0, "decode", decode)
            rep.insert(0, "variant", name)
            reports.append(rep.reset_index())
        rep = rank_report(score_rank(proba), y[ev], k[ev])
        summary[name] = {**info, "score": rep["score"].to_dict()}
        print(name, format_report(rep), flush=True)

    for name in ["lr", "gbdt", "gbdt_std"]:
        f = question_features(x, q, with_std=name == "gbdt_std")
        if name == "lr":
            train = fit | stop
            scaler = StandardScaler().fit(f[train])
            model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(f[train]), y[train])
            record(name, model.predict_proba(scaler.transform(f[ev])), {})
        else:
            booster = fit_lgb(f[fit], y[fit], f[stop], y[stop])
            booster.save_model(str(OUT / f"{name}.txt"))
            record(name, booster.predict(f[ev], num_iteration=booster.best_iteration),
                   {"best_iteration": booster.best_iteration})

    folds = cv.load("rank_folds").set_index("player_id")
    gpid = pd.read_parquet(META, columns=["player_id"])["player_id"].astype(str)
    folds.index = folds.index.astype(str)
    gfold = gpid.map(folds["fold"]).to_numpy()
    grank = gpid.map(folds["rank_id"]).to_numpy(np.int64)
    gfit, gstop = np.isin(gfold, [0, 1, 2]), gfold == 3
    assert not set(gpid[gfit | gstop]) & set(pid[ev])
    booster = fit_lgb(x[gfit], grank[gfit], x[gstop], grank[gstop])
    booster.save_model(str(OUT / "gbdt_game.txt"))
    ev_games = q.loc[ev, "games"].to_list()
    game_proba = booster.predict(x[np.concatenate(ev_games)], num_iteration=booster.best_iteration)
    record("gbdt_game", mean_logprob(game_proba, np.array([len(g) for g in ev_games])),
           {"best_iteration": booster.best_iteration, "train_games": int(gfit.sum())})

    pd.concat(reports, ignore_index=True).to_csv(OUT / "report.csv", index=False)
    summary["seconds"] = round(time.time() - t0, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done in {summary['seconds']}s -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
