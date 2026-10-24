"""Competition-2 HumanSL submission path (RESULTS A-041, A-042).

    fit        RankModel on every extracted Valid-S game (A-044: q1000 S-fit,
               qsel S-select, vs_rest) -> outputs/humansl_model/model.pkl. CPU.
    aggregate  for a release work directory already processed by
               v2/predict.py (prepare, infer, aggregate) and by HumanSL
               extraction into <work>/humansl/shard_*: per-game A-024 mean
               log-prob + HumanSL features -> RankModel -> score_rank ->
               humansl_predictions.parquet and a provisional submission CSV.
               Games without HumanSL features are left out of their question;
               a question with none falls back to the V2 C2 prediction.

    python -m go_aicup.tasks.strength.humansl_predict fit --method B0
    python -m go_aicup.tasks.strength.humansl_predict aggregate --work outputs/rank_submit/rank_public \\
        --submission outputs/rank_submit/rank_public/humansl_submission.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.features.humansl import run_features
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.predict import question_table

MODEL = ROOT / "outputs" / "humansl_model" / "model.pkl"
FIT_RUNS = (ROOT / "outputs" / "humansl_q1000", ROOT / "outputs" / "humansl_qsel", ROOT / "outputs" / "humansl_vs_rest")
FIT_ROLES = ("S-fit", "S-select", "S-extra")   # Valid-U is never fitted
A024 = ROOT / "outputs" / "v2_strength"
DATASET = PROCESSED / "strength_v2" / "seed2027"


def a024_per_game(category: np.ndarray) -> pd.DataFrame:
    values = category["log_probability_sum"] / category["num_moves"][:, None]
    return pd.DataFrame(values, index=category["game_index"].astype(np.int64),
                        columns=[f"a024_{r}" for r in RANKS])


def features(a024: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """A-024 columns then HumanSL own_/opp_ columns, games present in both."""
    human = human.set_index("game_index")
    columns = [c for c in human.columns if c.startswith(("own_", "opp_"))]
    return a024.join(human[columns], how="inner")


def fit(method: str, out: Path) -> None:
    a024 = pd.concat([a024_per_game(np.load(A024 / name, allow_pickle=False))
                      for name in ("validation_unseen_category.npy", "validation_seen_category.npy")])
    games, questions, tables = [], [], []
    for run in FIT_RUNS:
        g = pd.read_parquet(run / "games.parquet")
        games.append(g[g["role"].isin(FIT_ROLES)])
        if (run / "questions.parquet").exists():   # B0/B1 temperatures need questions; B2 does not
            q = pd.read_parquet(run / "questions.parquet")
            questions.append(q[q["role"].isin(FIT_ROLES)])
        tables.append(run_features(run))
    games = pd.concat(games, ignore_index=True)
    questions = pd.concat(questions, ignore_index=True)
    table = features(a024, pd.concat(tables, ignore_index=True).drop_duplicates("game_index")).loc[games["game_index"]]
    if len(table) != len(games):
        raise ValueError("fitting games without features")
    model = RankModel(method).fit(games, questions, table.to_numpy())
    model.save(out, method=method, columns=list(table.columns), fit_runs=[str(r) for r in FIT_RUNS],
               fit_games=len(games), fit_questions=len(questions), a024_run=str(A024))
    print(f"fit: {method} on {len(games):,} games / {len(questions):,} questions -> {out}")


def aggregate(work: Path, model_path: Path, submission: Path | None) -> pd.DataFrame:
    model, info = RankModel.load(model_path)
    table = features(a024_per_game(np.load(work / "category.npy", allow_pickle=False)),
                     run_features(work / "humansl"))
    if list(table.columns) != info["columns"]:
        raise ValueError("feature columns differ from the fitted model")
    questions = question_table(work)
    available = set(table.index)
    questions["games"] = [np.array([g for g in games if g in available], dtype=np.int64)
                          for games in questions["games"]]
    usable = questions["games"].map(len).to_numpy() > 0
    v2 = pd.read_parquet(work / "predictions.parquet").set_index("question_id")
    probabilities = v2.loc[questions["question_id"], [f"p_{r}" for r in RANKS]].to_numpy()
    probabilities[usable] = model.probabilities(table.to_numpy(), table.index.to_numpy(), questions[usable])
    rank_id = score_rank(probabilities)
    result = questions.drop(columns=["games"]).assign(
        humansl_games=questions["games"].map(len), fallback_v2=~usable, rank_id=rank_id,
        rank=[RANKS[r] for r in rank_id])
    for r, rank in enumerate(RANKS):
        result[f"p_{rank}"] = probabilities[:, r]
    result.to_parquet(work / "humansl_predictions.parquet", index=False)
    print(f"aggregate: {len(result)} questions, fallback_v2={int((~usable).sum())}, "
          f"missing-some-games={int((result['humansl_games'] < result['num_games']).sum())}, "
          f"predicted ranks={result['rank'].value_counts().reindex(RANKS, fill_value=0).to_dict()}")
    if submission is not None:
        submission.parent.mkdir(parents=True, exist_ok=True)
        # Provisional format; replace once the official sample_submission is out.
        result[["question_id", "rank"]].to_csv(submission, index=False)
        print(f"submission (provisional format): {submission}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--method", required=True, choices=("B0", "B1", "B2"))
    f.add_argument("--out", type=Path, default=MODEL)
    a = sub.add_parser("aggregate")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--model", type=Path, default=MODEL)
    a.add_argument("--submission", type=Path)
    args = parser.parse_args(argv)
    if args.command == "fit":
        fit(args.method, args.out)
    else:
        aggregate(args.work, args.model, args.submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
