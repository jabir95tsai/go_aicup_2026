"""AH+C fitted on every labelled player (RESULTS A-058), CPU except the extraction jobs.

select  : per Valid-U player up to 20 games (already-extracted q1000 U games first,
          then random, seed 2027) -> outputs/humansl_allu/fit_games.parquet; the
          games still lacking HumanSL / C features -> outputs/humansl_allu/games.parquet
          (extract with slurm/humansl.slurm and slurm/humansl_rankvec.slurm).
analyze : Valid-U players split per rank into U1 / U2 (seed 2027). Cross-fit: B2 on
          Valid-S + U1 games scored on the q1000 U questions of U2 players, and
          Valid-S + U2 on those of U1; together the 1,000 Valid-U questions. Reference:
          the production AH+C (Valid-S only, A-055 0.6934) on the same questions.
          Reported, no gate (D-003): gain and player bootstrap. Then the all-player
          model (Valid-S + every Valid-U fit game) -> outputs/humansl_allplayers_model/model.pkl.

    python -m scripts.strength.humansl_allplayers select
    python -m scripts.strength.humansl_allplayers analyze
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.strength.humansl_questions import by_k, equal_k
from go_aicup.features.humansl import run_features
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.humansl import SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"
RUN, MODEL = OUT / "humansl_allu", OUT / "humansl_allplayers_model" / "model.pkl"
Q1000 = OUT / "humansl_q1000"
PER_PLAYER, BOOTSTRAP = 20, 2000
COLUMNS = ["game_index", "rank_id", "player_id", "player_color", "length"]


def select() -> None:
    games = pd.read_parquet(DATASET / "strength_v2_games.parquet")
    unseen = games[games["split"] == "valid_unseen"]
    extracted = set(pd.read_parquet(Q1000 / "games.parquet").query("role == 'U'")["game_index"])
    rng = np.random.default_rng(SEED)
    picks = []
    for _, group in unseen.groupby("player_id", sort=True):
        done = group[group["game_index"].isin(extracted)]
        rest = group[~group["game_index"].isin(extracted)]
        chosen = done.iloc[rng.permutation(len(done))[:PER_PLAYER]]
        need = PER_PLAYER - len(chosen)
        picks.append(pd.concat([chosen, rest.iloc[rng.permutation(len(rest))[:max(need, 0)]]]))
    fit = pd.concat(picks)[COLUMNS].sort_values("game_index").reset_index(drop=True).assign(role="U-fit")
    new = fit[~fit["game_index"].isin(extracted)].reset_index(drop=True)
    RUN.mkdir(parents=True, exist_ok=True)
    fit.to_parquet(RUN / "fit_games.parquet")
    new.to_parquet(RUN / "games.parquet")
    print("fit games", len(fit), "players", fit["player_id"].nunique(), "| to extract", len(new),
          "positions", int(new["length"].sum()))


def u_halves(players: pd.DataFrame) -> set:
    """U1 player ids: half of each rank's Valid-U players."""
    rng = np.random.default_rng(SEED)
    half = []
    for _, group in players.groupby("rank_id", sort=True):
        ids = rng.permutation(np.sort(group["player_id"].astype(str).unique()))
        half += list(ids[:len(ids) // 2])
    return set(half)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "analyze"))
    if parser.parse_args().command == "select":
        select()
        return 0
    a024 = pd.concat([hp.a024_per_game(np.load(hp.A024 / n, allow_pickle=False))
                      for n in ("validation_unseen_category.npy", "validation_seen_category.npy")])
    human = pd.concat([run_features(r) for r in (*hp.FIT_RUNS, RUN)], ignore_index=True).drop_duplicates("game_index")
    table = hp.features(a024, human, hp.rankvec_features((*hp.RANKVEC_RUNS, RUN / "rankvec")))
    s_games = pd.concat([g[g["role"].isin(hp.FIT_ROLES)] for g in (pd.read_parquet(r / "games.parquet")
                                                                   for r in hp.FIT_RUNS)], ignore_index=True)
    u_games = pd.read_parquet(RUN / "fit_games.parquet")
    for frame, name in ((s_games, "Valid-S"), (u_games, "Valid-U fit")):
        missing = set(frame["game_index"]) - set(table.index)
        if missing:
            raise ValueError(f"{name}: {len(missing)} games without features")
    questions = pd.read_parquet(Q1000 / "questions.parquet").query("role == 'U'").reset_index(drop=True)
    u1 = u_halves(u_games)
    in_u1 = questions["player_id"].astype(str).isin(u1).to_numpy()
    reference, _ = RankModel.load(hp.RANKVEC_MODEL)

    def score(model, rows):
        q = questions[rows]
        pred = score_rank(model.probabilities(table.to_numpy(), table.index.to_numpy(), q))
        return official_scores(pred, q["rank_id"].to_numpy())

    def fitted(extra: pd.DataFrame) -> RankModel:
        games = pd.concat([s_games, extra], ignore_index=True)
        return RankModel("B2").fit(games, pd.DataFrame(), table.loc[games["game_index"]].to_numpy())

    k = questions["k"].to_numpy()
    cross, ref = np.zeros(len(questions)), np.zeros(len(questions))
    for held in (True, False):   # held: score the U1 questions with the model that added U2
        rows = in_u1 == held
        extra = u_games[u_games["player_id"].astype(str).isin(u1) != held]
        cross[rows] = score(fitted(extra), rows)
        ref[rows] = score(reference, rows)
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        r = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(cross[r], k[r]) - equal_k(ref[r], k[r]))
    final = fitted(u_games)
    columns = list(table.columns)
    final.save(MODEL, method="B2", columns=columns, fit_runs=[str(r) for r in (*hp.FIT_RUNS, RUN)],
               fit_games=len(s_games) + len(u_games), fit_questions=0, a024_run=str(hp.A024),
               rankvec_runs=[str(r) for r in (*hp.RANKVEC_RUNS, RUN / "rankvec")],
               rankvec_vectors=str(ROOT / "outputs" / "humansl_rankvec" / "vectors.pt"))
    result = {"valid_s_games": len(s_games), "valid_u_fit_games": len(u_games),
              "players": int(pd.concat([s_games, u_games])["player_id"].nunique()),
              "u1_players": len(u1), "questions_u1": int(in_u1.sum()),
              "Valid-U": {"production_AH_C": by_k(ref, k), "cross_fit_plus_half_U": by_k(cross, k)},
              "gain": round(equal_k(cross, k) - equal_k(ref, k), 4),
              "bootstrap_95": [round(float(v), 4) for v in np.quantile(draws, [0.025, 0.975])],
              "model": str(MODEL)}
    print(json.dumps(result, indent=1))
    (RUN / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
