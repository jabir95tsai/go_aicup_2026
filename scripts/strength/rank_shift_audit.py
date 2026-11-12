"""Test-population audit for the rank base (RESULTS 2026-11-11, Fable plan D1/D2).

Two questions, answered with existing per-game H/C features (CPU only):

prior     How much is the true test rank share worth?  B2 arms that differ only in the
          synthetic-question prior (base = players with >= k games in Valid-S; allplayers =
          share of all 8,007 train players, A 19.8% / 1D 5.4%; mix = half/half), scored on
          Valid-U E2 and reweighted per (k, rank) cell under three test scenarios
          (balanced, players with >= k games, all players).  Five seeds.
          The same base models give the activity table: E2 score and signed rank error by
          the player's total number of training games (is the base biased on light players?).
domain    Do the test questions look like Valid-U questions?  Per k: AUC of a test-vs-E2
          classifier on the 52 question-mean features (plain, and with E2 reweighted to the
          test's predicted-rank mix), the largest standardized mean differences, and the
          mean P(light player) of the test under a light(<=50 games)-vs-heavy(>=150)
          classifier fitted on E2.

    python -m scripts.strength.rank_shift_audit prior
    python -m scripts.strength.rank_shift_audit domain
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength import humansl_predict as hp

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_more_players as mp
from scripts.strength.rank_uniform_prior import cell_means

OUT = ROOT / "outputs" / "rank" / "humansl" / "shift_audit"
FOLDS = PROCESSED / "cv" / "strength_folds_seed2027.parquet"
ACTIVITY_BINS = [0, 30, 60, 120, 300, 100000]
LIGHT, HEAVY = 50, 150


def train_players() -> pd.DataFrame:
    return pd.read_parquet(FOLDS).set_index("player_id")


def setup():
    games, fit, eval_games, x_fit, x_eval, _ = mp.data(with_ex=False)
    q = hf.question_sets(games, rebuild=False)["E2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    return fit, eval_games, x_fit, x_eval, q, hd01, mp.player_prior(fit[valid_s])


def prior() -> None:
    fit, eval_games, x_fit, x_eval, q, hd01, fit_prior = setup()
    players = train_players()
    all_share = np.bincount(players["rank_id"], minlength=10) / len(players)
    all_players = {k: all_share for k in K_BUCKETS}
    u = (eval_games["handicap"] <= 1).to_numpy()
    arms = {"base": fit_prior, "allplayers": all_players,
            "mix": {k: 0.5 * fit_prior[k] + 0.5 * all_share for k in K_BUCKETS}}
    scenarios = {"balanced": {k: np.full(10, 0.1) for k in K_BUCKETS},
                 "players_ge_k": mp.player_prior(pd.concat([fit[hd01 & (fit["role"] == "S").to_numpy()],
                                                            eval_games[u]], ignore_index=True)),
                 "all_players": all_players}
    truth = q["rank_id"].to_numpy()
    cells, score_sum, error_sum = {}, np.zeros(len(q)), np.zeros(len(q))
    for seed in hf.SEEDS:
        hs.SEED = seed
        for name, arm_prior in arms.items():
            model = mp.PriorMatched(arm_prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01])
            p = model.probabilities(x_eval, eval_games["game_index"].to_numpy(), q)
            scores = hf.official_scores(score_rank(p), truth)
            cells[(seed, name)] = cell_means(scores, q)
            if name == "base":
                score_sum += scores
                error_sum += p @ np.arange(10) - truth
        print(seed, "done", flush=True)

    def weighted(cell, scenario):
        per_k = {str(k): float(cell[i] @ scenarios[scenario][k]) for i, k in enumerate(K_BUCKETS)}
        return per_k | {"equal_k": float(np.mean(list(per_k.values())))}

    result = {"arm_prior": {a: {str(k): np.round(v[k], 4).tolist() for k in K_BUCKETS} for a, v in arms.items()},
              "scenarios": {}}
    for scenario in scenarios:
        entry = {}
        for name in arms:
            runs = [weighted(cells[(s, name)], scenario) for s in hf.SEEDS]
            entry[name] = {key: round(float(np.mean([r[key] for r in runs])), 4) for key in runs[0]}
            if name != "base":
                diffs = [r["equal_k"] - weighted(cells[(s, "base")], scenario)["equal_k"]
                         for r, s in zip(runs, hf.SEEDS)]
                entry[name]["diff_vs_base"] = round(float(np.mean(diffs)), 4)
                entry[name]["diff_per_seed"] = [round(d, 4) for d in diffs]
        result["scenarios"][scenario] = entry
    result["per_rank"] = {name: np.mean([cells[(s, name)].mean(axis=0) for s in hf.SEEDS], axis=0).round(4).tolist()
                          for name in arms}

    # activity: base arm, mean over seeds, by the player's total training games
    table = q[["player_id", "rank_id", "k"]].copy()
    table["score"] = score_sum / len(hf.SEEDS)
    table["signed_error"] = error_sum / len(hf.SEEDS)
    table["n_games"] = players["n_games"].reindex(table["player_id"].astype(str)).to_numpy()
    table["bin"] = pd.cut(table["n_games"], ACTIVITY_BINS).astype(str)
    # remove the (k, rank) cell mean so bins with different rank mixes are comparable
    cell = table.groupby(["k", "rank_id"])[["score", "signed_error"]].transform("mean")
    table["score_adj"] = table["score"] - cell["score"]
    table["error_adj"] = table["signed_error"] - cell["signed_error"]
    by_bin = table.groupby("bin").agg(questions=("score", "size"), players=("player_id", "nunique"),
                                      score=("score", "mean"), score_adj=("score_adj", "mean"),
                                      signed_error=("signed_error", "mean"), error_adj=("error_adj", "mean"))
    result["activity"] = {"bins": by_bin.round(4).reset_index().to_dict("records"),
                          "light_le_%d" % LIGHT: table.loc[table.n_games <= LIGHT, ["score_adj", "error_adj"]]
                          .mean().round(4).to_dict(),
                          "heavy_ge_%d" % HEAVY: table.loc[table.n_games >= HEAVY, ["score_adj", "error_adj"]]
                          .mean().round(4).to_dict(),
                          "by_k_light": table[table.n_games <= LIGHT].groupby("k")[["score_adj", "error_adj"]]
                          .mean().round(4).reset_index().to_dict("records"),
                          "corr_log_games": {c: round(float(np.corrcoef(np.log(table["n_games"]), table[c])[0, 1]), 4)
                                             for c in ("score_adj", "error_adj")}}
    print(json.dumps({s: {a: (e["equal_k"], e.get("diff_vs_base")) for a, e in v.items()}
                      for s, v in result["scenarios"].items()}, indent=1))
    print(by_bin.round(4).to_string())
    print(json.dumps({k: v for k, v in result["activity"].items() if k != "bins"}, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prior.json").write_text(json.dumps(result, indent=1))


def weighted_auc(features, label, groups, weight) -> float:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    oof = cross_val_predict(model, features, label, groups=groups, cv=GroupKFold(5), method="predict_proba",
                            params={"logisticregression__sample_weight": weight})[:, 1]
    return float(roc_auc_score(label, oof, sample_weight=weight))


def domain() -> None:
    fit, eval_games, x_fit, x_eval, q, hd01, fit_prior = setup()
    players = train_players()
    hs.SEED = hf.SEED
    model = mp.PriorMatched(fit_prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01])
    index = eval_games["game_index"].to_numpy()
    e2_mean = hs.question_mean(x_eval, index, q)
    e2_pred = score_rank(model.probabilities(x_eval, index, q))
    e2_games = players["n_games"].reindex(q["player_id"].astype(str)).to_numpy()
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    names = columns[10:]
    test = {}
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        x = table.to_numpy()[:, 10:]
        k = (questions["k"] if "k" in questions else questions["num_games"].map(hs.k_bucket)).to_numpy()
        test[part] = {"mean": hs.question_mean(x, table.index.to_numpy(), questions), "k": k,
                      "pred": score_rank(model.probabilities(x, table.index.to_numpy(), questions))}
    result = {"e2_light_share": round(float((e2_games <= LIGHT).mean()), 4),
              "e2_heavy_share": round(float((e2_games >= HEAVY).mean()), 4), "per_k": {}}
    for bucket in K_BUCKETS:
        e = (q["k"] == bucket).to_numpy()
        entry = {}
        # light-vs-heavy classifier on E2 (player-grouped out-of-fold)
        light, heavy = e & (e2_games <= LIGHT), e & (e2_games >= HEAVY)
        rows = light | heavy
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        oof = cross_val_predict(pipe, e2_mean[rows], light[rows].astype(int), cv=GroupKFold(5),
                                groups=q.loc[rows, "player_id"].astype(str).to_numpy(), method="predict_proba")[:, 1]
        pipe.fit(e2_mean[rows], light[rows].astype(int))
        entry["light_vs_heavy"] = {"light": int(light.sum()), "heavy": int(heavy.sum()),
                                   "auc_oof": round(float(roc_auc_score(light[rows], oof)), 4),
                                   "p_light_e2_light": round(float(oof[light[rows]].mean()), 4),
                                   "p_light_e2_heavy": round(float(oof[heavy[rows]].mean()), 4)}
        for part, t in test.items():
            m = t["k"] == bucket
            feats, pred = t["mean"][m], t["pred"][m]
            share_t = np.bincount(pred, minlength=10) / len(pred)
            share_e = np.bincount(e2_pred[e], minlength=10) / e.sum()
            w_e = share_t[e2_pred[e]] / np.maximum(share_e[e2_pred[e]], 1e-9)
            x_all = np.vstack([e2_mean[e], feats])
            y_all = np.r_[np.zeros(e.sum(), int), np.ones(len(feats), int)]
            groups = np.r_[q.loc[e, "player_id"].astype(str).to_numpy(), [f"{part}{i}" for i in range(len(feats))]]
            ones = np.ones(len(y_all))
            w_all = np.r_[w_e * e.sum() / w_e.sum(), np.ones(len(feats))]
            mu_e = np.average(e2_mean[e], axis=0, weights=w_e)
            sd_e = np.sqrt(np.average((e2_mean[e] - mu_e) ** 2, axis=0, weights=w_e))
            smd = (feats.mean(axis=0) - mu_e) / sd_e
            top = np.argsort(-np.abs(smd))[:8]
            p_light = pipe.predict_proba(feats)[:, 1]
            # E2 reference for P(light) at the test's predicted-rank mix (in-sample for E2 rows)
            ref = pipe.predict_proba(e2_mean[e])[:, 1]
            entry[part] = {"questions": int(m.sum()),
                           "auc": round(weighted_auc(x_all, y_all, groups, ones), 4),
                           "auc_rank_matched": round(weighted_auc(x_all, y_all, groups, w_all), 4),
                           "top_smd": {names[i]: round(float(smd[i]), 3) for i in top},
                           "mean_abs_smd": round(float(np.abs(smd).mean()), 4),
                           "p_light_test": round(float(p_light.mean()), 4),
                           "p_light_e2_rank_matched": round(float(np.average(ref, weights=w_e)), 4)}
        result["per_k"][str(bucket)] = entry
        print(bucket, json.dumps(entry, indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "domain.json").write_text(json.dumps(result, indent=1))


def main() -> int:
    {"prior": prior, "domain": domain}[sys.argv[1]]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
