"""Uniform rank prior for the B2 synthetic questions (RESULTS 2026-11-11).

The base (noA-t20-allp-prior) draws synthetic ranks from the player share per rank
(1D 433 players, A 1,582), and its test predictions under-call 1D (23-26 of 400 per part).
If the test questions are rank-balanced (the training set is 100,000 games per rank), a
uniform prior is the matching one; if the test follows the player share, the base is.

local  Valid-S + train20 (HD<=1) fit, Valid-U E2 (50 questions per k x rank), five seeds.
       Arms: player prior (base), uniform, half/half mix. Per (k, rank) mean scores are
       reweighted under two test scenarios: balanced (every rank 1/10) and player share
       (the production prior: Valid-S + Valid-U players with >= k HD<=1 games).
test   production fit (Valid-S + train20 + Valid-U, HD<=1, seed 2027) per arm; test-set
       predicted rank counts and the number of changed questions vs the base. No file written.

    python -m scripts.strength.rank_uniform_prior local
    python -m scripts.strength.rank_uniform_prior test
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_more_players as mp

OUT = ROOT / "outputs" / "rank" / "humansl" / "uniform_prior"
BASE = ROOT / "submissions" / "rank" / "ahc-p3-noA-t20-allp-prior_736589_rank.csv"
UNIFORM = {k: np.full(10, 0.1) for k in K_BUCKETS}


def arms(player: dict) -> dict:
    return {"player": player, "uniform": UNIFORM,
            "mix": {k: 0.5 * player[k] + 0.5 * UNIFORM[k] for k in K_BUCKETS}}


def cell_means(scores, q) -> np.ndarray:
    """[k, rank] mean official score."""
    out = np.zeros((len(K_BUCKETS), 10))
    for i, k in enumerate(K_BUCKETS):
        for r in range(10):
            out[i, r] = scores[(q["k"] == k).to_numpy() & (q["rank_id"] == r).to_numpy()].mean()
    return out


def local() -> None:
    games, fit, eval_games, x_fit, x_eval, _ = mp.data(with_ex=False)
    q = hf.question_sets(games, rebuild=False)["E2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    fit_prior = mp.player_prior(fit[valid_s])
    # test scenario "player share": the production prior (Valid-S + Valid-U players)
    scenario_player = mp.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    scenarios = {"balanced": {k: np.full(10, 0.1) for k in K_BUCKETS}, "player_share": scenario_player}
    cells = {}
    for seed in hf.SEEDS:
        hs.SEED = seed
        for name, prior in arms(fit_prior).items():
            model = mp.PriorMatched(prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01])
            cells[(seed, name)] = cell_means(hf.predict(model, eval_games, x_eval, q), q)
        print(seed, "done", flush=True)

    def weighted(cell, scenario):
        per_k = {str(k): float(cell[i] @ scenarios[scenario][k]) for i, k in enumerate(K_BUCKETS)}
        return per_k | {"equal_k": float(np.mean(list(per_k.values())))}

    result = {"fit_prior": {str(k): v.round(4).tolist() for k, v in fit_prior.items()},
              "scenario_player_share": {str(k): v.round(4).tolist() for k, v in scenario_player.items()},
              "questions": int(len(q)), "scenarios": {}}
    for scenario in scenarios:
        entry = {}
        for name in ("player", "uniform", "mix"):
            runs = [weighted(cells[(s, name)], scenario) for s in hf.SEEDS]
            mean = {key: round(float(np.mean([r[key] for r in runs])), 4) for key in runs[0]}
            entry[name] = mean
            if name != "player":
                diffs = [r["equal_k"] - weighted(cells[(s, "player")], scenario)["equal_k"]
                         for r, s in zip(runs, hf.SEEDS)]
                entry[name]["diff_vs_player"] = round(float(np.mean(diffs)), 4)
                entry[name]["diff_per_seed"] = [round(d, 4) for d in diffs]
        result["scenarios"][scenario] = entry
    # per rank (k-averaged, five seeds): where the change moves score
    result["per_rank"] = {name: np.mean([cells[(s, name)].mean(axis=0) for s in hf.SEEDS], axis=0).round(4).tolist()
                          for name in ("player", "uniform", "mix")}
    print(json.dumps({s: {a: (e.get("equal_k"), e.get("diff_vs_player")) for a, e in v.items()}
                      for s, v in result["scenarios"].items()}, indent=1))
    print(json.dumps(result["per_rank"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local.json").write_text(json.dumps(result, indent=1))


def test() -> None:
    games, fit, eval_games, x_fit, x_eval, _ = mp.data(with_ex=False)
    u = (eval_games["handicap"] <= 1).to_numpy()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    base = hd01 & (fit["role"] != "X").to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()

    def stack(mask):
        return pd.concat([fit[mask], eval_games[u]], ignore_index=True), np.vstack([x_fit[mask], x_eval[u]])

    hs.SEED = hf.SEED
    allp_prior = mp.player_prior(stack(valid_s)[0])
    g, x = stack(base)
    models = {name: mp.PriorMatched(prior).fit(g, pd.DataFrame(), x) for name, prior in arms(allp_prior).items()}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {name: [] for name in models}
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for name, model in models.items():
            p = model.probabilities(table.to_numpy()[:, 10:], table.index.to_numpy(), questions)
            rows[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                            "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    reference = pd.read_csv(BASE, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {}
    for name, frames in rows.items():
        pred = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = pred != reference
        if name == "player" and changed.any():
            raise ValueError("player arm does not reproduce the base")
        report[name] = {"changed_public": int(changed[changed.index.str.startswith("pub")].sum()),
                        "changed_private": int(changed[changed.index.str.startswith("priv")].sum())}
        for tag in ("pub", "priv"):
            part = pred[pred.index.str.startswith(tag)]
            report[name][f"counts_{tag}"] = part.value_counts().reindex(RANKS).fillna(0).astype(int).tolist()
        pd.DataFrame({"question_id": pred.index, "pred_rank": pred.to_numpy()}).to_parquet(OUT / f"test_{name}.parquet")
    print(json.dumps(report, indent=1))
    (OUT / "test.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"local": local, "test": test}[sys.argv[1]]()
