"""Strong-engine features at 16 / 32 visits on the A-048 pilot games (RESULTS 2026-11-11), CPU only.

A-048 used 1 visit and AH (A-024 + HumanSL) as the reference; the current base is noA
(H at P3 komi | C | HD1 flags, 52 columns, `rank_train20_hc.data`). Same split as A-048:
B2 fitted on the q1000 S-fit games, scored on the qsel S-select questions; HD<=1 only
(questions with an HD>=2 game are dropped). Engine features are A-048's 14 per game
(own / opp move quality), komi 6.5 for every game.

  HC       base features
  HC_E1    + engine, 1 visit (A-048 run)
  HC_E16   + engine, 16 visits
  HC_E32   + engine, 32 visits

Five B2 sampling seeds; player bootstrap (2,000) of the best arm minus HC on seed 2027.
Pre-registered gate: best HC_E - HC >= +0.01 Equal-k (five-seed mean) and every k
>= -0.005 -> extend the engine features to the full fit pool and the test questions.

    python -m scripts.strength.engine_visits_pilot
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.features.katago_engine import run_features as engine_features
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import RankModel

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_train20_hc as t20

RUNS = ROOT / "outputs" / "rank" / "humansl"
ENGINE = {"E1": RUNS / "engine_pilot", "E16": RUNS / "engine_pilot_v16", "E32": RUNS / "engine_pilot_v32"}
OUT = RUNS / "engine_visits_pilot"
MIN_GAIN, MIN_K = 0.01, -0.005


def pilot_tables():
    fit_games = pd.read_parquet(RUNS / "humansl_q1000" / "games.parquet")
    fit_games = fit_games[fit_games["role"] == "S-fit"].reset_index(drop=True)
    test_games = pd.read_parquet(RUNS / "humansl_qsel" / "games.parquet")
    questions = pd.read_parquet(RUNS / "humansl_qsel" / "questions.parquet")
    return fit_games, test_games, questions


def main() -> int:
    _, fit, _, x_fit, _, _ = t20.data()
    s = (fit["role"] == "S").to_numpy()
    base = pd.DataFrame(x_fit[s], index=fit.loc[s, "game_index"].to_numpy())
    hd = pd.Series(fit.loc[s, "handicap"].to_numpy(), index=base.index)
    engines = {name: engine_features(run).set_index("game_index") for name, run in ENGINE.items()}
    fit_games, test_games, questions = pilot_tables()

    def table(games):
        gi = games["game_index"].to_numpy()
        missing = {"base": int((~np.isin(gi, base.index)).sum())}
        missing |= {n: int((~np.isin(gi, e.index)).sum()) for n, e in engines.items()}
        return gi, missing

    fit_gi, fit_missing = table(fit_games)
    test_gi, test_missing = table(test_games)
    # HD<=1 games with every feature set present
    have = lambda gi: np.isin(gi, base.index) & np.all([np.isin(gi, e.index) for e in engines.values()], axis=0)
    fit_ok = have(fit_gi) & (hd.reindex(fit_gi).to_numpy() <= 1)
    test_ok_games = set(test_gi[have(test_gi) & (hd.reindex(test_gi).to_numpy() <= 1)])
    q_ok = questions["games"].apply(lambda g: all(int(i) in test_ok_games for i in g)).to_numpy()
    fit_games, questions = fit_games[fit_ok].reset_index(drop=True), questions[q_ok].reset_index(drop=True)
    used = np.unique(np.concatenate(questions["games"].to_numpy()))
    test_games = pd.DataFrame({"game_index": used})
    coverage = {"fit_missing": fit_missing, "test_missing": test_missing, "fit_games": int(len(fit_games)),
                "fit_players": int(fit_games["player_id"].nunique()), "questions": int(len(questions)),
                "questions_per_k": questions["k"].value_counts().sort_index().astype(int).to_dict()}
    print(json.dumps(coverage), flush=True)

    def x(gi, arm):
        parts = [base.loc[gi].to_numpy()]
        if arm != "HC":
            parts.append(engines[arm[3:]].loc[gi].to_numpy())
        return np.hstack(parts)

    arms = ["HC", "HC_E1", "HC_E16", "HC_E32"]
    k = questions["k"].to_numpy()
    result = {"coverage": coverage, "engine_columns": list(engines["E1"].columns), "seeds": {}}
    scores = {}
    for seed in hf.SEEDS:
        hs.SEED = seed
        entry = {}
        for arm in arms:
            model = RankModel("B2").fit(fit_games, pd.DataFrame(), x(fit_games["game_index"].to_numpy(), arm))
            scores[(seed, arm)] = hf.predict(model, test_games, x(test_games["game_index"].to_numpy(), arm), questions)
            entry[arm] = hf.by_k(scores[(seed, arm)], k)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({a: e["equal_k"] for a, e in entry.items()}), flush=True)

    summary = {}
    for arm in arms[1:]:
        diffs = {str(b): float(np.mean([scores[(s, arm)][k == b].mean() - scores[(s, "HC")][k == b].mean()
                                       for s in hf.SEEDS])) for b in K_BUCKETS}
        gain = float(np.mean([hf.equal_k(scores[(s, arm)], k) - hf.equal_k(scores[(s, "HC")], k) for s in hf.SEEDS]))
        summary[arm] = {"diff_equal_k": round(gain, 4), "diff_by_k": {b: round(v, 4) for b, v in diffs.items()},
                        "bootstrap_95_seed2027": hf.bootstrap(scores[(hf.SEED, "HC")], scores[(hf.SEED, arm)],
                                                              questions),
                        "pass": bool(gain >= MIN_GAIN and min(diffs.values()) >= MIN_K)}
    result["mean_equal_k"] = {a: round(float(np.mean([result["seeds"][str(s)][a]["equal_k"] for s in hf.SEEDS])), 4)
                              for a in arms}
    result["vs_HC"] = summary
    result["gate"] = {"min_gain": MIN_GAIN, "min_k": MIN_K, "pass": any(v["pass"] for v in summary.values())}
    print(json.dumps({"mean_equal_k": result["mean_equal_k"], "vs_HC": summary}, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
