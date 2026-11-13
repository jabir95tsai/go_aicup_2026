"""C4: HumanSL fine-tuned as a Yugen-rank-conditioned policy (RESULTS A-057).

select : train-split players (fold 1-4, seed 2027 split) are split per rank
         (seed 2027) into F (fine-tuning) and B (B2 fitting); 10% of B players
         are Bm (training monitor, never used for B2).
           outputs/humansl_finetune/games.parquet          every train-split game of F (role learn)
           outputs/humansl_finetune/monitor/games.parquet  300 train-split games per rank of Bm (role monitor)
           outputs/humansl_finetune/players.parquet        player_id, rank_id, half (F / B / Bm)
         Valid-U players are in neither half.

    python -m scripts.strength.humansl_finetune select
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from go_aicup.paths import PROCESSED, ROOT

DATASET = PROCESSED / "strength_v2" / "seed2027"
RUN = ROOT / "outputs" / "humansl_finetune"
SEED, MONITOR_SHARE, MONITOR_PER_RANK = 2027, 0.1, 300
COLUMNS = ["game_index", "rank_id", "player_id", "player_color", "length"]


def select() -> None:
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet")
    train = games[games["split"] == "train"]
    rng = np.random.default_rng(SEED)
    halves = []
    for rank, group in train.groupby("rank_id", sort=True):
        players = rng.permutation(np.sort(group["player_id"].unique()))
        f, b = players[:len(players) // 2], players[len(players) // 2:]
        bm = b[:int(round(MONITOR_SHARE * len(b)))]
        halves.append(pd.DataFrame({"player_id": players, "rank_id": rank,
                                    "half": ["F" if p in set(f) else "Bm" if p in set(bm) else "B" for p in players]}))
    players = pd.concat(halves, ignore_index=True)
    if players["player_id"].duplicated().any():
        raise ValueError("a player in two ranks")
    unseen = set(games.loc[games["split"] == "valid_unseen", "player_id"])
    if unseen & set(players["player_id"]):
        raise ValueError("Valid-U players in the split")
    half = train["player_id"].map(players.set_index("player_id")["half"])
    learn = train[half == "F"][COLUMNS].assign(role="learn")
    monitor = pd.concat([group.iloc[np.sort(rng.permutation(len(group))[:MONITOR_PER_RANK])]
                         for _, group in train[half == "Bm"].groupby("rank_id", sort=True)])[COLUMNS].assign(role="monitor")
    (RUN / "monitor").mkdir(parents=True, exist_ok=True)
    players.to_parquet(RUN / "players.parquet")
    learn.sort_values("game_index").reset_index(drop=True).to_parquet(RUN / "games.parquet")
    monitor.sort_values("game_index").reset_index(drop=True).to_parquet(RUN / "monitor" / "games.parquet")
    print(players.groupby(["rank_id", "half"]).size().unstack().to_string())
    print("learn games", len(learn), "positions", int(learn["length"].sum()))
    print("monitor games", len(monitor), "players", monitor["player_id"].nunique(), "positions", int(monitor["length"].sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select",))
    {"select": select}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
