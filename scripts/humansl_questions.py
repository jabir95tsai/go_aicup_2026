"""HumanSL question-level gate (RESULTS A-040), CPU only; features from features/humansl.py.

select  : 250 Valid-U questions per k (evaluation) and 250 per k from the S-fit
          half of Valid-S (fitting; A-025 player draw), seed 2027; writes the
          union of their games for HumanSL extraction.
analyze : per-game multinomial logistic regression (StandardScaler, C=1) on
          S-fit games -> per-game log-posterior; a question averages its games
          (1/k each), multiplies by a per-k temperature fitted on S-fit
          questions, softmax, metrics.score_rank. S-fit log-posteriors used for
          the temperature fit are out-of-fold (two folds by player); the U model
          is refit on all S-fit games. Candidates: H (HumanSL own + opponent),
          AH (+ A-024 per-game mean log-prob), A_LR (A-024 features through the
          same pipeline, pipeline control). Reference: A-024 + C2 exactly as
          A-025 (T_k fitted on all S-fit questions), on the same U questions.
Pre-registered gate: AH Equal-k on the 1,000 U questions >= reference + 0.005
and the 95% interval of a player-grouped bootstrap of the paired difference
has a lower bound > 0.

    python scripts/humansl_questions.py select
    python scripts/humansl_questions.py analyze --run outputs/humansl_q1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from humansl_pilot import a024_features, load_features
from v2_candidate_gate import c2_scores
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.v2.aggregate import fit_temperature, official_scores, tempered_probabilities
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASET = PROCESSED / "strength_v2" / "seed2027"
A024 = ROOT / "outputs" / "v2_strength"
SEED, PER_K, BOOTSTRAP, MIN_GAIN = 2027, 250, 2000, 0.005


def fit_players(seen: pd.DataFrame) -> set:
    players = np.sort(seen["player_id"].astype(str).unique())
    return set(np.random.default_rng(SEED).permutation(players)[:len(players) // 2])


def select(out: Path) -> None:
    unseen = pd.read_parquet(DATASET / "strength_v2_valid_unseen_questions.parquet")
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    s_fit = seen[seen["player_id"].astype(str).isin(fit_players(seen))]
    rng = np.random.default_rng(SEED)
    parts = []
    for role, pool in (("U", unseen), ("S-fit", s_fit)):
        for k in K_BUCKETS:
            group = pool[pool["k"] == k]
            parts.append(group.iloc[np.sort(rng.permutation(len(group))[:PER_K])].assign(role=role))
    questions = pd.concat(parts).reset_index(drop=True)
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet").set_index("game_index")
    index = np.unique(np.concatenate(list(questions["games"])))
    table = games.loc[index, ["rank_id", "player_id", "player_color", "length", "split"]].reset_index()
    table["role"] = np.where(table["split"] == "valid_unseen", "U", "S-fit")
    out.mkdir(parents=True, exist_ok=True)
    questions.to_parquet(out / "questions.parquet")
    table.drop(columns="split").to_parquet(out / "games.parquet")
    print(questions.groupby(["role", "k"]).size().unstack(), "\ngames", table.groupby("role").size().to_dict(),
          "positions", int(table["length"].sum()))


def log_posterior(train_x, train_y, test_x) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(train_x), train_y)
    return model.predict_log_proba(scaler.transform(test_x))


def question_scores(per_game: pd.DataFrame, questions: pd.DataFrame, temps: dict) -> np.ndarray:
    logits = np.stack([per_game.loc[list(g)].to_numpy().mean(axis=0) for g in questions["games"]])
    probabilities = np.empty_like(logits)
    for k in K_BUCKETS:
        mask = questions["k"].to_numpy() == k
        probabilities[mask] = tempered_probabilities(logits[mask], temps[k])
    return official_scores(score_rank(probabilities), questions["rank_id"].to_numpy())


def candidate(x: np.ndarray, games: pd.DataFrame, questions: pd.DataFrame) -> tuple[np.ndarray, dict]:
    fit = (games["role"] == "S-fit").to_numpy()
    y = games["rank_id"].to_numpy()
    index = games["game_index"].to_numpy()
    # Out-of-fold S-fit log-posteriors (two folds by player) for the temperature fit.
    players = np.sort(games.loc[fit, "player_id"].astype(str).unique())
    half = set(np.random.default_rng(SEED + 1).permutation(players)[:len(players) // 2])
    fold = games["player_id"].astype(str).isin(half).to_numpy()
    oof = np.zeros((fit.sum(), 10))
    fit_rows = np.flatnonzero(fit)
    for side in (True, False):
        train, test = fit & (fold != side), fit & (fold == side)
        oof[np.isin(fit_rows, np.flatnonzero(test))] = log_posterior(x[train], y[train], x[test])
    s_games = pd.DataFrame(oof, index=index[fit])
    s_questions = questions[questions["role"] == "S-fit"].reset_index(drop=True)
    s_logits = np.stack([s_games.loc[list(g)].to_numpy().mean(axis=0) for g in s_questions["games"]])
    temps, edges = {}, {}
    for k in K_BUCKETS:
        mask = s_questions["k"].to_numpy() == k
        temps[k], edges[k] = fit_temperature(s_logits[mask], s_questions["rank_id"].to_numpy()[mask])
    u_games = pd.DataFrame(log_posterior(x[fit], y[fit], x[~fit]), index=index[~fit])
    u_questions = questions[questions["role"] == "U"].reset_index(drop=True)
    return question_scores(u_games, u_questions, temps), {"temperatures": temps, "grid_edge": edges}


def equal_k(scores, k) -> float:
    return float(np.mean([scores[k == b].mean() for b in K_BUCKETS]))


def by_k(scores, k) -> dict:
    return {str(b): round(float(scores[k == b].mean()), 4) for b in K_BUCKETS} | {"equal_k": round(equal_k(scores, k), 4)}


def reference_scores(u_questions: pd.DataFrame) -> tuple[np.ndarray, dict]:
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    s_fit = seen[seen["player_id"].astype(str).isin(fit_players(seen))].reset_index(drop=True)
    categories = {name: np.load(A024 / file, allow_pickle=False) for name, file in
                  (("U", "validation_unseen_category.npy"), ("S", "validation_seen_category.npy"))}
    scores, fit = c2_scores(categories, {"S-fit": s_fit, "S-select": s_fit, "U": u_questions})
    return scores["U"], fit


def analyze(run: Path) -> None:
    questions = pd.read_parquet(run / "questions.parquet")
    games = pd.read_parquet(run / "games.parquet").merge(load_features(run), on="game_index", validate="one_to_one")
    if len(games) != len(pd.read_parquet(run / "games.parquet")):
        raise ValueError("HumanSL features missing for some games")
    human = games[[c for c in games.columns if c.startswith(("own_", "opp_"))]].to_numpy()
    a = a024_features(games["game_index"])
    u_questions = questions[questions["role"] == "U"].reset_index(drop=True)
    k = u_questions["k"].to_numpy()
    scores, result = {}, {"questions": questions.groupby("role").size().to_dict(),
                          "games": games.groupby("role").size().to_dict()}
    scores["A024_C2"], fit = reference_scores(u_questions)
    result["A024_C2"] = {"U": by_k(scores["A024_C2"], k), **fit}
    for name, x in (("A_LR", a), ("H", human), ("AH", np.hstack([a, human]))):
        scores[name], fit = candidate(x, games, questions)
        result[name] = {"U": by_k(scores[name], k), **fit}
    groups = pd.Series(np.arange(len(u_questions))).groupby(u_questions["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = {name: [] for name in ("AH", "H")}
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for name in draws:
            draws[name].append(equal_k(scores[name][rows], k[rows]) - equal_k(scores["A024_C2"][rows], k[rows]))
    result["comparisons"] = {f"{name} - A024_C2": {
        "gain": round(equal_k(scores[name], k) - equal_k(scores["A024_C2"], k), 4),
        "bootstrap_95": [round(float(v), 4) for v in np.quantile(draws[name], [0.025, 0.975])]} for name in draws}
    gate = result["comparisons"]["AH - A024_C2"]
    result["gate"] = {**gate, "players": len(groups), "min_gain": MIN_GAIN,
                      "pass": bool(gate["gain"] >= MIN_GAIN and gate["bootstrap_95"][0] > 0)}
    print(json.dumps(result, indent=1, default=str))
    (run / "result.json").write_text(json.dumps(result, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "analyze"))
    parser.add_argument("--run", type=Path, default=ROOT / "outputs" / "humansl_q1000")
    args = parser.parse_args()
    select(args.run) if args.command == "select" else analyze(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
