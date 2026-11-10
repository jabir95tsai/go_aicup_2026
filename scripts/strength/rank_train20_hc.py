"""More fitting games per player for noA (RESULTS 2026-11-09), H+C on A-047's train20 sample.

noA (P1+P2+P3 without the A-024 columns, Public 0.7271) needs only H and C, which carry no
in-sample bias on training-split games. A-047 already ran H (komi 6.5) on 106,057 train20
games (20 per train-split player, 5,767 players, 5,000 of them Valid-S players). Here:
  - HD0 games: C at komi 6.5 (H is in data/processed/strength_train_hc/standard)
  - HD1 games: H and C at komi 0.5 (P3)
  - left out: HD>=2 (P1) and the 15,000 games the C vectors were learned on (C would be
    optimistic there)

    python -m scripts.strength.rank_train20_hc tables             # game tables for the GPU job
    sbatch slurm/rank_train20_hc.slurm                   # 8 GPUs: H+C of HD1, C of HD0
    python -m scripts.strength.rank_train20_hc evaluate           # E2 / E1, five seeds, noA vs noA+train20
    python -m scripts.strength.rank_train20_hc export --job ID    # LB probe files

Arms (B2 on HD<=1 games, x = [H 30 | C 20 | HD1 flags 2], as noA):
  all    noA: Valid-S games only (the reference of humansl_hd_filter.compare)
  t20    noA: Valid-S + train20 games
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.features import humansl as hfeat
from go_aicup.features import humansl_rankvec as cfeat
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.humansl import RankModel

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_b2_bagging as bag

OUT = ROOT / "outputs" / "rank" / "humansl" / "train20_hc"
TRAIN20 = ROOT / "outputs" / "rank" / "humansl" / "humansl_train20" / "games.parquet"
LEARNED = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec" / "games.parquet"
NOA = bag.NOA


def metadata() -> pd.DataFrame:
    meta = pd.read_parquet(hf.CANONICAL, columns=["game_index", "rank", "game_id", "rank_id", "player_id",
                                                   "player_color"])
    return meta.merge(pd.read_csv(hf.HANDICAP, usecols=["rank", "game_id", "handicap"]), on=["rank", "game_id"],
                      validate="one_to_one")


def train20_games() -> pd.DataFrame:
    """train20 games used here: HD<=1, not used to learn the C vectors."""
    t20 = pd.read_parquet(TRAIN20, columns=["game_index"])
    learned = set(pd.read_parquet(LEARNED, columns=["game_index"])["game_index"])
    games = metadata().set_index("game_index").loc[t20["game_index"]].reset_index()
    games = games[(games["handicap"] <= 1) & ~games["game_index"].isin(learned)].reset_index(drop=True)
    return games.assign(role="T")


def tables() -> None:
    games = train20_games()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for hd in (0, 1):
        table = games.loc[games["handicap"] == hd, ["game_index", "player_color", "role"]].reset_index(drop=True)
        table.to_parquet(OUT / f"train20_hd{hd}.parquet")
        report[f"hd{hd}"] = len(table)
    report["players"] = int(games["player_id"].nunique())
    (OUT / "tables.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def train20_x(games: pd.DataFrame) -> np.ndarray:
    """[H 30 | C 20 | HD1 flags 2] per train20 game; rows lacking a feature are NaN."""
    index = np.load(hf.STANDARD / "game_index.npy")
    standard = pd.DataFrame(np.load(hf.STANDARD / "features.npy")[:, :30], index=index)
    h1 = hfeat.run_features(OUT / "h_hd1").set_index("game_index")
    h1 = h1[[c for c in h1.columns if c.startswith(("own_", "opp_"))]]
    c0 = cfeat.run_features(OUT / "c_hd0").set_index("game_index")
    c1 = cfeat.run_features(OUT / "c_hd1").set_index("game_index")
    x = np.full((len(games), 50), np.nan)
    gi, hd = games["game_index"].to_numpy(), games["handicap"].to_numpy()
    for value, h_source, c_source in ((0, standard, c0), (1, h1, c1)):
        rows = np.flatnonzero(hd == value)
        have_h, have_c = np.isin(gi[rows], h_source.index), np.isin(gi[rows], c_source.index)
        x[rows[have_h], :30] = h_source.loc[gi[rows[have_h]]].to_numpy()
        x[rows[have_c], 30:] = c_source.loc[gi[rows[have_c]]].to_numpy()
    return np.hstack([x, hf.hd1_columns(games)])


def data():
    """(games, fit = Valid-S + train20, eval = Valid-U, x_fit, x_eval); x is noA."""
    games, fit, eval_games, x_fit, x_eval = bag.data()
    t20 = train20_games()
    x_t20 = train20_x(t20)
    complete = np.isfinite(x_t20).all(axis=1)
    coverage = {"train20_games": int(len(t20)), "complete": int(complete.sum())}
    if complete.mean() < 0.99:
        raise ValueError(f"train20 features incomplete: {coverage}")
    t20, x_t20 = t20[complete].reset_index(drop=True), x_t20[complete]
    fit = pd.concat([fit, t20[fit.columns]], ignore_index=True)
    return games, fit, eval_games, np.vstack([x_fit[:, 10:], x_t20]), x_eval[:, 10:], coverage


def evaluate() -> None:
    games, fit, eval_games, x_fit, x_eval, coverage = data()
    sets = hf.question_sets(games, rebuild=False)
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    eligible = {}
    for arm, mask in (("all", valid_s), ("t20", hd01)):
        n = fit[mask].groupby("player_id").size()
        eligible[arm] = {str(k): int((n >= k).sum()) for k in (5, 10, 15, 20)}
    result = {"coverage": coverage, "fit_games": {"all": int(valid_s.sum()), "t20": int(hd01.sum())},
              "eligible_players": eligible, "seeds": {}}
    print(json.dumps(result), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hf.OUT = OUT     # compare writes its json into hf.OUT
    hf.compare(fit, eval_games, sets, {"all": (valid_s, x_fit, x_eval), "t20": (hd01, x_fit, x_eval)}, "t20",
               result, "result.json")


def export(job: str) -> None:
    """noA-t20 (Valid-S + train20) and noA-t20-allp (+ Valid-U), seed 2027; diffs vs noA."""
    games, fit, eval_games, x_fit, x_eval, _ = data()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    hs.SEED = hf.SEED
    models = {"noA": RankModel("B2").fit(fit[valid_s].reset_index(drop=True), pd.DataFrame(), x_fit[valid_s]),
              "noA-t20": RankModel("B2").fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01]),
              "noA-t20-allp": RankModel("B2").fit(
                  pd.concat([fit[hd01], eval_games[u]], ignore_index=True), pd.DataFrame(),
                  np.vstack([x_fit[hd01], x_eval[u]]))}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(hf.ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {name: [] for name in models}
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for name, model in models.items():
            p = model.probabilities(table.to_numpy()[:, 10:], table.index.to_numpy(), questions)
            rows[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                            "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    reference = pd.read_csv(NOA, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {}
    for name, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = result != reference
        report[name] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                        "private": int(changed[changed.index.str.startswith("priv")].sum())}
        if name == "noA":
            if changed.any():
                raise ValueError("noA refit does not reproduce ahc-p3-noA_725504")
            continue
        output = ROOT / "submissions" / "rank" / f"ahc-p3-{name}_{job}_rank.csv"
        result.reset_index().to_csv(output, index=False, lineterminator="\r\n")
        print("wrote", output, flush=True)
    print(json.dumps({"changed_vs_noA": report}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tables", "evaluate", "export"])
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    {"tables": tables, "evaluate": evaluate}.get(args.command, lambda: export(args.job))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
