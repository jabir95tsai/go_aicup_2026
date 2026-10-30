"""HumanSL question aggregation choice (RESULTS A-041), CPU only.

Methods B0 / B1 / B2 are tasks/strength/humansl.RankModel (see its docstring).
Fitting data = S-fit games and questions of outputs/rank/humansl/humansl_q1000; selection =
S-select questions of outputs/rank/humansl/humansl_qsel; Valid-U is scored once, for the
selected method only (plus B0, which A-040 already reported, as a
reproduction check).

Pre-registered: the best S-select Equal-k replaces B0 if it beats B0 on S-select
by >= 0.005; otherwise B0 stays. Learning curve: the chosen method fitted on half
of the S-fit players vs all of them, scored on S-select; >= 0.01 -> more fitting
data is worth extracting.

    python scripts/humansl_aggregation.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from humansl_pilot import a024_features
from humansl_questions import by_k, equal_k, reference_scores
from go_aicup.features.humansl import run_features
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import METHODS, SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

MIN_GAIN, CURVE_GAIN, BOOTSTRAP = 0.005, 0.01, 2000


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    questions = pd.read_parquet(run / "questions.parquet")
    games = pd.read_parquet(run / "games.parquet").merge(run_features(run), on="game_index", validate="one_to_one")
    human = [c for c in games.columns if c.startswith(("own_", "opp_"))]
    x = np.hstack([a024_features(games["game_index"]), games[human].to_numpy()])
    return games, questions, x


def part(data, role: str):
    games, questions, x = data
    rows = (games["role"] == role).to_numpy()
    return games[rows].reset_index(drop=True), questions[questions["role"] == role].reset_index(drop=True), x[rows]


def subset(data, players: set):
    games, questions, x = data
    keep = games["player_id"].astype(str).isin(players).to_numpy()
    return (games[keep].reset_index(drop=True),
            questions[questions["player_id"].astype(str).isin(players)].reset_index(drop=True), x[keep])


def score(method: str, fit, test) -> np.ndarray:
    model = RankModel(method).fit(*fit)
    games, questions, x = test
    pred = score_rank(model.probabilities(x, games["game_index"].to_numpy(), questions))
    return official_scores(pred, questions["rank_id"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fit-run", type=Path, default=ROOT / "outputs" / "rank" / "humansl" / "humansl_q1000")
    parser.add_argument("--select-run", type=Path, default=ROOT / "outputs" / "rank" / "humansl" / "humansl_qsel")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "rank" / "humansl" / "humansl_aggregation.json")
    args = parser.parse_args()
    data = load(args.fit_run)
    fit, u = part(data, "S-fit"), part(data, "U")
    select = load(args.select_run)
    s_k = select[1]["k"].to_numpy()
    result = {"S-select": {method: by_k(score(method, fit, select), s_k) for method in METHODS}}
    s_scores = {m: v["equal_k"] for m, v in result["S-select"].items()}
    best = max(s_scores, key=s_scores.get)
    chosen = best if s_scores[best] - s_scores["B0"] >= MIN_GAIN else "B0"
    players = np.sort(fit[0]["player_id"].astype(str).unique())
    half = set(np.random.default_rng(SEED + 2).permutation(players)[:len(players) // 2])
    curve = {"half": by_k(score(chosen, subset(fit, half), select), s_k)["equal_k"], "all": s_scores[chosen]}
    curve["gain"] = round(curve["all"] - curve["half"], 4)
    curve["more_data_worth_it"] = bool(curve["gain"] >= CURVE_GAIN)
    k_u = u[1]["k"].to_numpy()
    u_scores = score(chosen, fit, u)
    reference, _ = reference_scores(u[1])
    groups = pd.Series(np.arange(len(u[1]))).groupby(u[1]["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(u_scores[rows], k_u[rows]) - equal_k(reference[rows], k_u[rows]))
    result.update({
        "chosen": chosen, "best_on_S_select": best, "learning_curve": curve,
        "B0_U_reproduction": by_k(score("B0", fit, u), k_u)["equal_k"],   # A-040 AH: 0.6232
        "U": {"chosen": by_k(u_scores, k_u), "A024_C2": by_k(reference, k_u),
              "gain": round(equal_k(u_scores, k_u) - equal_k(reference, k_u), 4),
              "bootstrap_95": [round(float(v), 4) for v in np.quantile(draws, [0.025, 0.975])]}})
    print(json.dumps(result, indent=1, default=str))
    args.out.write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
