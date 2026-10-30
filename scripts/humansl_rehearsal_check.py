"""HumanSL rehearsal gate (RESULTS A-041 / A-042), CPU only.

The fake release (scripts/make_fake_release.py) copies Valid-U questions, whose
games were already extracted through the training cache (humansl_pilot,
humansl_q1000). Pre-registered:
  (a) per-game HumanSL features, release path vs training path: p99 of the per-game
      max |diff| <= 1e-2 (was max <= 1e-2 until A-044; A-042 recorded FAIL under it);
  (b) question classes: release path (humansl_predict aggregate) vs the same
      RankModel on training-path features and the stored A-024 Valid-U values;
      any differing class must be a near tie (top-two expected official score
      gap < 0.01 under the reference probabilities).

    python scripts/humansl_rehearsal_check.py [--root outputs/release/rehearsal]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.features.humansl import run_features
from go_aicup.metrics import NEAR, rank_report, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.humansl_predict import MODEL, a024_per_game, aggregate, features

DATASET = PROCESSED / "strength_v2" / "seed2027"
FEATURE_TOL, TIE = 1e-2, 0.01   # p99 tolerance, near-tie gap


def training_features() -> pd.DataFrame:
    a024 = a024_per_game(np.load(ROOT / "outputs" / "rank" / "v2" / "v2_strength" / "validation_unseen_category.npy",
                                 allow_pickle=False))
    human = pd.concat([run_features(ROOT / "outputs" / "rank" / "humansl" / run) for run in ("humansl_pilot", "humansl_q1000")])
    return features(a024, human.drop_duplicates("game_index", keep="last"))


def expected_gap(probabilities: np.ndarray) -> np.ndarray:
    padded = np.pad(probabilities, ((0, 0), (1, 1)))
    expected = probabilities + NEAR * (padded[:, :-2] + padded[:, 2:])
    top = np.sort(expected, axis=1)
    return top[:, -1] - top[:, -2]


def check(root: Path, name: str, reference: pd.DataFrame, model: RankModel) -> bool:
    work = root / "strength" / name
    answers = pd.read_csv(root / "answers" / f"{name}.csv")
    unseen = pd.read_parquet(DATASET / "strength_v2_valid_unseen_questions.parquet")
    source_games = dict(zip(unseen["question_id"], unseen["games"].map(np.asarray)))
    _, _, meta = load_store(name, root / "store")
    source_q = dict(zip(answers["question_id"].astype(str), answers["source_question_id"]))
    store_source = np.array([source_games[source_q[str(q)]][slot - 1] for q, slot in zip(meta["question_id"], meta["slot"])])
    cache = pd.read_parquet(work / "cache" / "games.parquet")
    to_source = dict(zip(cache["game_index"], store_source[cache["store_row"].to_numpy()]))

    release = features(a024_per_game(np.load(work / "category.npy", allow_pickle=False)), run_features(work / "humansl"))
    human_cols = [c for c in release.columns if c.startswith(("own_", "opp_"))]
    ref = reference.loc[[to_source[g] for g in release.index]]
    per_game = np.abs(release[human_cols].to_numpy() - ref[human_cols].to_numpy()).max(axis=1)
    feature_diff, feature_p99 = float(per_game.max()), float(np.quantile(per_game, 0.99))
    a024_diff = float(np.abs(release.drop(columns=human_cols).to_numpy() - ref.drop(columns=human_cols).to_numpy()).max())
    ok_a = feature_p99 <= FEATURE_TOL and len(release) == len(cache)
    print(f"  (a) {name}: {len(release)}/{len(cache)} games with features, HumanSL p99 {feature_p99:.2e} "
          f"max {feature_diff:.2e}, "
          f"A-024 max|diff| {a024_diff:.2e} -> {'PASS' if ok_a else 'FAIL'}")

    pred = aggregate(work, MODEL, work / "humansl_submission.csv")
    pred["question_id"] = pred["question_id"].astype(int)
    merged = answers.merge(pred, on="question_id", suffixes=("", "_pred"), validate="one_to_one")
    ref_questions = pd.DataFrame({"games": [source_games[q] for q in merged["source_question_id"]],
                                  "num_games": merged["num_games"].to_numpy()})
    ref_prob = model.probabilities(reference.to_numpy(), reference.index.to_numpy(), ref_questions)
    ref_class = score_rank(ref_prob)
    differ = ref_class != merged["rank_id_pred"].to_numpy()
    gaps = expected_gap(ref_prob)[differ]
    ok_b = bool((gaps < TIE).all())
    print(f"  (b) {name}: classes identical {int((~differ).sum())}/{len(merged)}, differing gaps "
          f"{np.round(gaps, 4).tolist()} -> {'PASS' if ok_b else 'FAIL'}")
    report = rank_report(merged["rank_id_pred"], merged["rank_id"], merged["k"])
    print("      score by k:", report["score"].round(4).to_dict())
    return ok_a and ok_b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "release" / "rehearsal")
    args = parser.parse_args()
    model, info = RankModel.load(MODEL)
    print(f"model: {info['method']} on {info['fit_games']} games")
    reference = training_features()
    ok = all([check(args.root, name, reference, model) for name in ("rank_public", "rank_private")])
    print(json.dumps({"humansl_rehearsal": "PASS" if ok else "FAIL"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
