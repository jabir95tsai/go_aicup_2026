"""HumanSL pilot (RESULTS A-039): does KataGo HumanSL add rank signal to A-024? CPU only.

select  : 200 games per rank from Valid-U (evaluation) and 200 per rank from the
          S-fit half of Valid-S (fitting; same player draw as A-025), seed 2027.
analyze : per game, mean HumanSL log-likelihood of the labelled player's moves
          for each profile (and of the opponent's moves). Multinomial logistic
          regressions fitted on S-fit only, decoded with metrics.score_rank,
          scored per game on Valid-U:
            A  A-024 per-game mean log-prob (10 features)
            H  HumanSL features only
            AH both
Pre-registered gate: AH - A single-game official score on Valid-U >= +0.01 and
the 95% interval of a player-grouped bootstrap of the paired difference has a
lower bound > 0 -> continue to question-level (k buckets) evaluation.

    python scripts/humansl_pilot.py select
    python scripts/humansl_pilot.py analyze --run outputs/humansl_pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
A024 = ROOT / "outputs" / "v2_strength"
SEED, PER_RANK, BOOTSTRAP, MIN_GAIN = 2027, 200, 2000, 0.01


def select(out: Path) -> None:
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet")
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    players = np.sort(seen["player_id"].astype(str).unique())
    fit_players = set(np.random.default_rng(SEED).permutation(players)[:len(players) // 2])
    rng = np.random.default_rng(SEED)
    pools = {"U": games[games["split"] == "valid_unseen"],
             "S-fit": games[(games["split"] == "valid_seen") & games["player_id"].astype(str).isin(fit_players)]}
    parts = []
    for role, pool in pools.items():
        for _, group in pool.groupby("rank_id"):
            parts.append(group.iloc[rng.choice(len(group), PER_RANK, replace=False)].assign(role=role))
    table = pd.concat(parts).sort_values("game_index").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    table[["game_index", "role", "rank_id", "player_id", "player_color", "length"]].to_parquet(out / "games.parquet")
    print(table.groupby(["role", "rank_id"]).size().unstack())


def load_features(run: Path) -> pd.DataFrame:
    rows = []
    for shard in sorted(p for p in run.glob("shard_*") if p.is_dir()):
        table = pd.read_parquet(shard / "games.parquet")
        logp, offsets, done = (np.load(shard / f"{name}.npy") for name in ("logp", "offsets", "done"))
        profiles = json.loads((shard / "meta.json").read_text())["profiles"]
        for i, game in enumerate(table.itertuples()):
            if not done[i].all():
                continue
            values = logp[offsets[i]:offsets[i + 1]]
            mine = (np.arange(len(values)) % 2) == game.player_color   # black first, alternating (A-002)
            rows.append({"game_index": game.game_index,
                         **{f"own_{p}": float(values[mine, j].mean()) for j, p in enumerate(profiles)},
                         **{f"opp_{p}": float(values[~mine, j].mean()) for j, p in enumerate(profiles)}})
    return pd.DataFrame(rows)


def a024_features(game_index) -> np.ndarray:
    parts = []
    for name in ("validation_unseen_category.npy", "validation_seen_category.npy"):
        category = np.load(A024 / name, allow_pickle=False)
        parts.append(pd.DataFrame(category["log_probability_sum"] / category["num_moves"][:, None],
                                  index=category["game_index"].astype(np.int64)))
    table = pd.concat(parts)
    return table.loc[np.asarray(game_index, dtype=np.int64)].to_numpy()


def fit_predict(train_x, train_y, test_x) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(train_x), train_y)
    return score_rank(model.predict_proba(scaler.transform(test_x)))


def analyze(run: Path) -> None:
    games = pd.read_parquet(run / "games.parquet")
    table = games.merge(load_features(run), on="game_index", validate="one_to_one")
    human = [c for c in table.columns if c.startswith(("own_", "opp_"))]
    own = [c for c in human if c.startswith("own_")]
    a = a024_features(table["game_index"])
    feature_sets = {"A": a, "H_own": table[own].to_numpy(), "H": table[human].to_numpy(),
                    "AH": np.hstack([a, table[human].to_numpy()])}
    fit, test = (table["role"] == "S-fit").to_numpy(), (table["role"] == "U").to_numpy()
    y = table["rank_id"].to_numpy()
    scores = {name: official_scores(fit_predict(x[fit], y[fit], x[test]), y[test]) for name, x in feature_sets.items()}
    players = table.loc[test, "player_id"].astype(str).to_numpy()
    groups = pd.Series(np.arange(test.sum())).groupby(players).apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    diff = scores["AH"] - scores["A"]
    draws = [diff[np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])].mean()
             for _ in range(BOOTSTRAP)]
    low, high = np.quantile(draws, [0.025, 0.975])
    per_rank = {name: pd.Series(value).groupby(y[test]).mean().round(3).tolist() for name, value in scores.items()}
    result = {"games": {"S-fit": int(fit.sum()), "U": int(test.sum()), "U_players": len(groups)},
              "single_game_score_U": {name: round(float(value.mean()), 4) for name, value in scores.items()},
              "per_rank_U": per_rank,
              "gate": {"gain_AH_minus_A": round(float(diff.mean()), 4),
                       "bootstrap_95": [round(float(low), 4), round(float(high), 4)],
                       "min_gain": MIN_GAIN, "pass": bool(diff.mean() >= MIN_GAIN and low > 0)}}
    print(json.dumps(result, indent=1))
    (run / "result.json").write_text(json.dumps(result, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "analyze"))
    parser.add_argument("--run", type=Path, default=ROOT / "outputs" / "humansl_pilot")
    args = parser.parse_args()
    select(args.run) if args.command == "select" else analyze(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
