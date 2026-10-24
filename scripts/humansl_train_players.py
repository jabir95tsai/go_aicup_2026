"""HumanSL fitting data from train-split players (RESULTS A-047), CPU only.

select  : up to 20 random train-split games per player (seed 2027), all 5,767
          players -> outputs/humansl_train20. HumanSL-only features: A-024 was
          trained on these games, so its features would be biased (A-045).
analyze : B2 on HumanSL mean features (F0), scored on the 1,000 S-select questions:
            (a) H, fit = Valid-S S-fit + S-extra (A-046 F0 setting, 0.7025)
            (b) H, fit = (a) + train games of players outside the S-select half
                (Valid-S players are training players; the S-select half's
                train games would leak into S-select)
          reference: AH on the (a) games (A-044 (i), 0.7101).
Pre-registered: data gain = (b) - (a). If (b) >= AH + 0.005 on S-select, the
production model switches to H fitted on all Valid-S + all train games, and that
model is scored once on Valid-U against AH on all Valid-S (A-044, 0.6833) with a
player-grouped bootstrap; otherwise AH stays and Valid-U is not used.

    python scripts/humansl_train_players.py select
    python scripts/humansl_train_players.py analyze
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from humansl_aggregation import load, part
from humansl_questions import by_k, equal_k, fit_players
from go_aicup.features.humansl import run_features
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"
RUN = OUT / "humansl_train20"
PER_PLAYER, MIN_GAIN, BOOTSTRAP = 20, 0.005, 2000


def select() -> None:
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet")
    train = games[games["split"] == "train"]
    rng = np.random.default_rng(SEED)
    picks = [group.iloc[np.sort(rng.permutation(len(group))[:PER_PLAYER])]
             for _, group in train.groupby("player_id", sort=True)]
    table = pd.concat(picks).sort_values("game_index").assign(role="train")
    RUN.mkdir(parents=True, exist_ok=True)
    table[["game_index", "rank_id", "player_id", "player_color", "length", "role"]].reset_index(drop=True) \
        .to_parquet(RUN / "games.parquet")
    print("games", len(table), "players", table["player_id"].nunique(), "positions", int(table["length"].sum()))


def human_games(run, roles=None) -> pd.DataFrame:
    games = pd.read_parquet(run / "games.parquet")
    if roles is not None:
        games = games[games["role"].isin(roles)]
    table = games.merge(run_features(run), on="game_index", validate="one_to_one")
    if len(table) != len(games):
        raise ValueError(f"{run}: features missing")
    return table


def fit_score(fit: pd.DataFrame, test: pd.DataFrame, questions: pd.DataFrame) -> np.ndarray:
    columns = [c for c in fit.columns if c.startswith(("own_", "opp_"))]
    model = RankModel("B2").fit(fit, pd.DataFrame(), fit[columns].to_numpy())
    pred = score_rank(model.probabilities(test[columns].to_numpy(), test["game_index"].to_numpy(), questions))
    return official_scores(pred, questions["rank_id"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "analyze"))
    args = parser.parse_args()
    if args.command == "select":
        select()
        return 0
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    select_half = set(seen["player_id"].astype(str)) - fit_players(seen)
    questions = pd.read_parquet(OUT / "humansl_qsel" / "questions.parquet")
    if not set(questions["player_id"].astype(str)) <= select_half:
        raise ValueError("S-select questions outside the S-select half")
    k = questions["k"].to_numpy()
    test = human_games(OUT / "humansl_qsel", ["S-select"])
    base = pd.concat([human_games(OUT / "humansl_q1000", ["S-fit"]),
                      human_games(OUT / "humansl_vs_rest", ["S-fit", "S-extra"])], ignore_index=True)
    train = human_games(RUN)
    train_out = train[~train["player_id"].astype(str).isin(select_half)]
    scores = {"a_H_valid_s": fit_score(base, test, questions),
              "b_H_plus_train": fit_score(pd.concat([base, train_out], ignore_index=True), test, questions)}
    ah = 0.7101   # A-044 (i): AH on the same Valid-S fitting games
    result = {"train_games": len(train), "train_players": int(train["player_id"].nunique()),
              "train_games_used_for_S_select": len(train_out),
              "S-select": {name: by_k(value, k) for name, value in scores.items()},
              "AH_reference": ah}
    eq = {name: value["equal_k"] for name, value in result["S-select"].items()}
    result["data_gain"] = round(eq["b_H_plus_train"] - eq["a_H_valid_s"], 4)
    result["switch_to_H"] = bool(eq["b_H_plus_train"] >= ah + MIN_GAIN)
    if result["switch_to_H"]:
        data = load(OUT / "humansl_q1000")
        u = part(data, "U")
        u_games = human_games(OUT / "humansl_q1000", ["U"])
        everything = pd.concat([human_games(OUT / "humansl_q1000", ["S-fit"]), test,
                                human_games(OUT / "humansl_vs_rest"), train], ignore_index=True)
        u_h = fit_score(everything, u_games, u[1])
        from humansl_aggregation import score as ah_score
        from humansl_more_data import combine, rest_data
        new_games, new_x = rest_data()
        select_set = load(OUT / "humansl_qsel")
        u_ah = ah_score("B2", combine(part(data, "S-fit"), (select_set[0], select_set[2]), (new_games, new_x)), u)
        u_k = u[1]["k"].to_numpy()
        groups = pd.Series(np.arange(len(u[1]))).groupby(u[1]["player_id"].astype(str).to_numpy()) \
            .apply(np.asarray).to_list()
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(BOOTSTRAP):
            rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            draws.append(equal_k(u_h[rows], u_k[rows]) - equal_k(u_ah[rows], u_k[rows]))
        result["U"] = {"H_all": by_k(u_h, u_k), "AH_all_valid_s": by_k(u_ah, u_k),
                       "gain": round(equal_k(u_h, u_k) - equal_k(u_ah, u_k), 4),
                       "bootstrap_95": [round(float(v), 4) for v in np.quantile(draws, [0.025, 0.975])],
                       "fit_games": len(everything), "fit_players": int(everything["player_id"].nunique())}
    print(json.dumps(result, indent=1, default=str))
    (RUN / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
