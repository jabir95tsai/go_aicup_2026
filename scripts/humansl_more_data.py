"""More HumanSL fitting data from the rest of Valid-S (RESULTS A-044), CPU only.

select  : every Valid-S game not yet extracted (pilot, q1000, qsel), tagged
          S-fit / S-select by the A-025 player draw, or S-extra for players in
          no Valid-S question (fewer than 5 held-out games; in neither half)
          -> outputs/rank/humansl/humansl_vs_rest.
analyze : (i)  B2 fitted on old S-fit + new S-fit and S-extra games (players
               disjoint from S-select) vs old S-fit only,
               both scored on the 1,000 S-select questions (A-041);
          (ii) B2 fitted on every extracted Valid-S game, scored once on the
               1,000 Valid-U questions against B2 on S-fit only (A-041, 0.6506)
               and A-024 + C2, player-grouped bootstrap.
Pre-registered: the all-Valid-S model becomes the production model if (i) gains
> 0 on S-select; (ii) is reported either way.

    python scripts/humansl_more_data.py select
    python scripts/humansl_more_data.py analyze
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from humansl_aggregation import load, part, score
from humansl_pilot import a024_features
from humansl_questions import by_k, equal_k, fit_players, reference_scores
from go_aicup.features.humansl import run_features
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import SEED

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"
REST = OUT / "rank" / "humansl" / "humansl_vs_rest"
BOOTSTRAP = 2000


def select() -> None:
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet")
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    done = set()
    for run in ("humansl_pilot", "humansl_q1000", "humansl_qsel"):
        for shard in (OUT / "rank" / "humansl" / run).glob("shard_*"):
            if shard.is_dir():
                done |= set(pd.read_parquet(shard / "games.parquet")["game_index"].astype(int))
    rest = games[(games["split"] == "valid_seen") & ~games["game_index"].isin(done)]
    half, in_questions = fit_players(seen), set(seen["player_id"].astype(str))
    player = rest["player_id"].astype(str)
    rest = rest.assign(role=np.where(player.isin(half), "S-fit",
                                     np.where(player.isin(in_questions), "S-select", "S-extra")))
    REST.mkdir(parents=True, exist_ok=True)
    rest[["game_index", "rank_id", "player_id", "player_color", "length", "role"]].reset_index(drop=True) \
        .to_parquet(REST / "games.parquet")
    print(rest.groupby("role").agg(games=("game_index", "size"), players=("player_id", "nunique")).to_dict(),
          "positions", int(rest["length"].sum()))


def rest_data():
    games = pd.read_parquet(REST / "games.parquet").merge(run_features(REST), on="game_index", validate="one_to_one")
    human = [c for c in games.columns if c.startswith(("own_", "opp_"))]
    return games, np.hstack([a024_features(games["game_index"]), games[human].to_numpy()])


def combine(*parts):
    games = pd.concat([p[0] for p in parts], ignore_index=True)
    return games, pd.DataFrame(columns=["games", "k", "rank_id", "player_id"]), np.vstack([p[-1] for p in parts])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "analyze"))
    args = parser.parse_args()
    if args.command == "select":
        select()
        return 0
    data = load(OUT / "rank" / "humansl" / "humansl_q1000")
    old_fit, u = part(data, "S-fit"), part(data, "U")
    select_set = load(OUT / "rank" / "humansl" / "humansl_qsel")
    new_games, new_x = rest_data()
    new_fit = new_games["role"].isin(["S-fit", "S-extra"]).to_numpy()
    s_k, u_k = select_set[1]["k"].to_numpy(), u[1]["k"].to_numpy()
    result = {"new_games": new_games.groupby("role").size().to_dict(),
              "new_players": new_games.groupby("role")["player_id"].nunique().to_dict()}
    # (i) B2 needs no questions, so fitting sets are game tables only.
    old = combine(old_fit)
    if set(new_games.loc[new_fit, "player_id"].astype(str)) & set(select_set[1]["player_id"].astype(str)):
        raise ValueError("new fitting games share players with the S-select questions")
    more = combine(old_fit, (new_games[new_fit], new_x[new_fit]))
    s_old, s_more = score("B2", old, select_set), score("B2", more, select_set)
    result["i_S_select"] = {"old_S_fit": by_k(s_old, s_k), "plus_new_S_fit": by_k(s_more, s_k),
                            "gain": round(equal_k(s_more, s_k) - equal_k(s_old, s_k), 4)}
    result["production_all_valid_s"] = bool(result["i_S_select"]["gain"] > 0)
    # (ii) every extracted Valid-S game.
    all_s = combine(old_fit, (select_set[0], select_set[2]), (new_games, new_x))
    u_all, u_old = score("B2", all_s, u), score("B2", old, u)
    reference, _ = reference_scores(u[1])
    groups = pd.Series(np.arange(len(u[1]))).groupby(u[1]["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = {"vs_B2_S_fit": [], "vs_A024_C2": []}
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        base = equal_k(u_all[rows], u_k[rows])
        draws["vs_B2_S_fit"].append(base - equal_k(u_old[rows], u_k[rows]))
        draws["vs_A024_C2"].append(base - equal_k(reference[rows], u_k[rows]))
    baselines = {"vs_B2_S_fit": u_old, "vs_A024_C2": reference}
    result["ii_U"] = {"all_valid_s": by_k(u_all, u_k), "S_fit_only": by_k(u_old, u_k), "A024_C2": by_k(reference, u_k),
                      "fit_games": len(all_s[0]), "fit_players": int(all_s[0]["player_id"].nunique())}
    for name, values in draws.items():
        result["ii_U"][name] = {"gain": round(equal_k(u_all, u_k) - equal_k(baselines[name], u_k), 4),
                                "bootstrap_95": [round(float(q), 4) for q in np.quantile(values, [0.025, 0.975])]}
    print(json.dumps(result, indent=1, default=str))
    (REST / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
