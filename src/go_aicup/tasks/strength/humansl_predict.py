"""Competition-2 HumanSL submission path (RESULTS A-041, A-042, A-055).

    fit        RankModel on every extracted Valid-S game (A-044: q1000 S-fit,
               qsel S-select, vs_rest) -> outputs/rank/humansl/humansl_model/model.pkl (AH). CPU.
               --rankvec adds the C2 rank-vector features (A-055, columns c_*)
               -> outputs/rank/humansl/humansl_rankvec_model/model.pkl (AH+C, the primary).
    aggregate  for a release work directory already processed by
               v2/predict.py (prepare, infer, aggregate) and by HumanSL
               extraction into <work>/humansl/shard_*: per-game A-024 mean
               log-prob + HumanSL features -> RankModel -> score_rank ->
               humansl_predictions.parquet and a provisional submission CSV.
               Games without HumanSL features (or C features, for an AH+C
               model: <work>/rankvec/shard_*) are left out of their question;
               a question with none falls back to --fallback (default the V2 C2
               prediction; the AH+C path falls back to the AH predictions).

    python -m go_aicup.tasks.strength.humansl_predict fit --method B2 [--rankvec]
    python -m go_aicup.tasks.strength.humansl_predict aggregate --work outputs/rank/submit/rank_public \\
        --submission outputs/rank/submit/rank_public/humansl_submission.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.features import humansl_rankvec
from go_aicup.features.humansl import run_features, shard_features
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.predict import question_table

MODEL = ROOT / "outputs" / "rank" / "humansl" / "humansl_model" / "model.pkl"
RANKVEC_MODEL = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec_model" / "model.pkl"
RANKVEC_RUNS = (ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec_feat", ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec_feat_vu")
FIT_RUNS = (ROOT / "outputs" / "rank" / "humansl" / "humansl_q1000", ROOT / "outputs" / "rank" / "humansl" / "humansl_qsel", ROOT / "outputs" / "rank" / "humansl" / "humansl_vs_rest")
FIT_ROLES = ("S-fit", "S-select", "S-extra")   # Valid-U is never fitted
A024 = ROOT / "outputs" / "rank" / "v2" / "v2_strength"
DATASET = PROCESSED / "strength_v2" / "seed2027"


def a024_per_game(category: np.ndarray) -> pd.DataFrame:
    values = category["log_probability_sum"] / category["num_moves"][:, None]
    return pd.DataFrame(values, index=category["game_index"].astype(np.int64),
                        columns=[f"a024_{r}" for r in RANKS])


def features(a024: pd.DataFrame, human: pd.DataFrame, rankvec: pd.DataFrame | None = None) -> pd.DataFrame:
    """A-024 columns, HumanSL own_/opp_ columns, then C2 c_ columns if given; games present in all."""
    human = human.set_index("game_index")
    columns = [c for c in human.columns if c.startswith(("own_", "opp_"))]
    table = a024.join(human[columns], how="inner")
    if rankvec is not None:
        table = table.join(rankvec.set_index("game_index").add_prefix("c_"), how="inner")
    return table


def rankvec_features(runs) -> pd.DataFrame:
    """C2 features of finished shards (meta.json written last) of each run directory."""
    parts = [humansl_rankvec.run_features(run) for run in runs if any(run.glob("shard_*/meta.json"))]
    if not parts:
        return pd.DataFrame({"game_index": []})
    return pd.concat(parts, ignore_index=True).drop_duplicates("game_index")


def fit(method: str, out: Path, rankvec: bool = False) -> None:
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
    human = pd.concat(tables, ignore_index=True).drop_duplicates("game_index")
    table = features(a024, human, rankvec_features(RANKVEC_RUNS) if rankvec else None)
    missing = set(games["game_index"]) - set(table.index)
    if missing:
        raise ValueError(f"{len(missing)} fitting games without features")
    table = table.loc[games["game_index"]]
    model = RankModel(method).fit(games, questions, table.to_numpy())
    extra = {"rankvec_runs": [str(r) for r in RANKVEC_RUNS],
             "rankvec_vectors": str(ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec" / "vectors.pt")} if rankvec else {}
    model.save(out, method=method, columns=list(table.columns), fit_runs=[str(r) for r in FIT_RUNS],
               fit_games=len(games), fit_questions=len(questions), a024_run=str(A024), **extra)
    print(f"fit: {method} on {len(games):,} games / {len(questions):,} questions -> {out}")


def release_features(run: Path) -> pd.DataFrame:
    """HumanSL features of finished shards only; a failed shard (no meta.json) is skipped."""
    shards = sorted(p for p in run.glob("shard_*") if (p / "meta.json").exists() and (p / "done.npy").exists())
    skipped = sorted(p.name for p in run.glob("shard_*") if p.is_dir() and p not in shards)
    if skipped:
        print(f"aggregate: skipping unfinished HumanSL shards {skipped}")
    parts = [shard_features(p) for p in shards]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"game_index": []})


def aggregate(work: Path, model_path: Path, submission: Path | None,
              predictions: str = "humansl_predictions.parquet", fallback: str = "predictions.parquet") -> pd.DataFrame:
    """fallback: predictions file inside work (p_<rank> columns) for questions without usable games."""
    model, info = RankModel.load(model_path)
    with_c = any(c.startswith("c_") for c in info["columns"])
    table = features(a024_per_game(np.load(work / "category.npy", allow_pickle=False)),
                     release_features(work / "humansl"),
                     rankvec_features([work / "rankvec"]) if with_c else None)
    if len(table) and list(table.columns) != info["columns"]:
        raise ValueError("feature columns differ from the fitted model")
    questions = question_table(work)
    available = set(table.index)
    questions["games"] = [np.array([g for g in games if g in available], dtype=np.int64)
                          for games in questions["games"]]
    usable = questions["games"].map(len).to_numpy() > 0
    backup = pd.read_parquet(work / fallback).set_index("question_id").loc[questions["question_id"]]
    probabilities = backup[[f"p_{r}" for r in RANKS]].to_numpy()
    to_v2 = ~usable if fallback == "predictions.parquet" else ~usable & backup["fallback_v2"].to_numpy()
    if usable.any():
        probabilities[usable] = model.probabilities(table.to_numpy(), table.index.to_numpy(), questions[usable])
    rank_id = score_rank(probabilities)
    result = questions.drop(columns=["games"]).assign(
        humansl_games=questions["games"].map(len), fallback_v2=to_v2, fallback_other=~usable & ~to_v2, rank_id=rank_id,
        rank=[RANKS[r] for r in rank_id])
    for r, rank in enumerate(RANKS):
        result[f"p_{rank}"] = probabilities[:, r]
    result.to_parquet(work / predictions, index=False)
    print(f"aggregate ({model_path.parent.name}): {len(result)} questions, fallback_v2={int(to_v2.sum())}, "
          f"fallback_{fallback.removesuffix('.parquet')}={int((~usable & ~to_v2).sum())}, "
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
    f.add_argument("--rankvec", action="store_true", help="AH+C (A-055); default out becomes the rankvec model")
    f.add_argument("--out", type=Path)
    a = sub.add_parser("aggregate")
    a.add_argument("--work", type=Path, required=True)
    a.add_argument("--model", type=Path, default=MODEL)
    a.add_argument("--submission", type=Path)
    a.add_argument("--predictions", default="humansl_predictions.parquet", help="file name inside --work")
    a.add_argument("--fallback", default="predictions.parquet", help="file inside --work for questions without games")
    args = parser.parse_args(argv)
    if args.command == "fit":
        fit(args.method, args.out or (RANKVEC_MODEL if args.rankvec else MODEL), args.rankvec)
    else:
        aggregate(args.work, args.model, args.submission, args.predictions, args.fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
