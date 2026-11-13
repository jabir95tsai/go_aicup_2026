"""T1 rehearsal gate: the release pipeline must reproduce the validated path.

Pre-registered (EXECUTION_PLAN T1):
  (1) every game's moves / colours in the release stores equal the training cache;
  (2) question predictions are identical (competition 1 top-5, competition 2 class)
      and competition-2 question probabilities differ by at most 2e-3 (was 1e-4;
      bf16 results depend on batch composition, RESULTS A-036 / A-037).

References: competition 2 = C2 on the stored A-024 Valid-U per-game sums
(validation_unseen_category.npy); competition 1 = the A-029 pp+slot + L6
scorer run directly on the training prefix arrays (split 1 / 2).
Scores against the answer keys are printed per k as a sanity check.

    python -m scripts.release.rehearsal_check [--root outputs/rehearsal] [--skip-strength]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.data.preprocess import load_store
from go_aicup.metrics import K_BUCKETS, pid_report, rank_report
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import assign, data
from go_aicup.tasks.identity.predict import load_params, opening_scores, train_prefix
from go_aicup.tasks.strength.v2.aggregate import decode_questions, question_logits, tempered_probabilities
from go_aicup.tasks.strength.v2.predict import DEFAULT_CALIBRATION, temperatures

CACHE = PROCESSED / "boards" / "strength_v1"
V2_DIR = PROCESSED / "strength_v2" / "seed2027"
PROB_TOL = 2e-3
PID_SPLITS = {"public": 1, "private": 2}


class TrainMoves:
    def __init__(self):
        self.games = pd.read_parquet(CACHE / "games.parquet")
        self._shards = {}

    def game(self, index: int):
        row = self.games.iloc[index]
        if row.shard_id not in self._shards:
            prefix = CACHE / f"shard_{int(row.shard_id):05d}"
            self._shards[row.shard_id] = (np.load(f"{prefix}_moves.npy", mmap_mode="r"),
                                          np.load(f"{prefix}_colors.npy", mmap_mode="r"))
        moves, colors = self._shards[row.shard_id]
        span = slice(int(row.local_offset), int(row.local_offset + row.length))
        return moves[span], colors[span]


def check_moves(store_dir: Path, name: str, sources: np.ndarray, train: TrainMoves) -> int:
    moves, colors, meta = load_store(name, store_dir)
    bad = 0
    for i, source in enumerate(sources):
        m, c = train.game(int(source))
        span = slice(int(meta["offset"].iloc[i]), int(meta["offset"].iloc[i] + meta["length"].iloc[i]))
        bad += not (np.array_equal(moves[span], m) and np.array_equal(colors[span], c))
    print(f"  (1) {name}: {len(sources) - bad}/{len(sources)} games identical to the training cache")
    return bad


def question_sources(store_dir: Path, name: str, answers: pd.DataFrame, source_games: dict) -> np.ndarray:
    _, _, meta = load_store(name, store_dir)
    source_q = dict(zip(answers["question_id"].astype(str), answers["source_question_id"]))
    return np.array([source_games[source_q[str(q)]][slot - 1] for q, slot in zip(meta["question_id"], meta["slot"])])


def strength(root: Path, train: TrainMoves) -> bool:
    ok = True
    unseen = pd.read_parquet(V2_DIR / "strength_v2_valid_unseen_questions.parquet")
    games = dict(zip(unseen["question_id"], unseen["games"].map(np.asarray)))
    category = np.load(ROOT / "outputs" / "v2_strength" / "validation_unseen_category.npy")
    temps = temperatures(DEFAULT_CALIBRATION)
    for part in ("public", "private"):
        name = f"rank_{part}"
        answers = pd.read_csv(root / "answers" / f"{name}.csv")
        ok &= check_moves(root / "store", name, question_sources(root / "store", name, answers, games), train) == 0
        path = root / "strength" / name / "predictions.parquet"
        if not path.exists():
            print(f"  (2) {name}: {path} missing (run the inference step)")
            ok = False
            continue
        pred = pd.read_parquet(path)
        pred["question_id"] = pred["question_id"].astype(int)
        merged = answers.merge(pred, on="question_id", validate="one_to_one")
        logits = question_logits(category, [games[q] for q in merged["source_question_id"]], "mean_log_probability")
        reference = np.empty_like(logits)
        for k in K_BUCKETS:
            mask = merged["k"].to_numpy() == k
            reference[mask] = tempered_probabilities(logits[mask], temps[k])
        mine = merged[[f"p_{r}" for r in RANKS]].to_numpy()
        diff = float(np.abs(mine - reference).max())
        same = int((decode_questions(reference, "score_rank") == merged["rank_id_y"].to_numpy()).sum())
        passed = diff <= PROB_TOL and same == len(merged)
        ok &= passed
        print(f"  (2) {name}: max|p - p_ref| = {diff:.2e}, classes identical {same}/{len(merged)} "
              f"-> {'PASS' if passed else 'FAIL'}")
        report = rank_report(merged["rank_id_y"], merged["rank_id_x"], merged["k"])
        print("      score by k:", report["score"].round(4).to_dict())
    return ok


def identity(root: Path, train: TrainMoves) -> bool:
    ok = True
    meta, prefix = train_prefix("d4")
    gallery, questions = data.pid_splits(meta)
    params = load_params("d4")
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    for part, split in PID_SPLITS.items():
        answers = pd.read_csv(root / "answers" / f"pid_{part}.csv")
        q_split = questions[questions["split"] == split]
        source_games = dict(zip(q_split["question_id"], q_split["games"]))
        ok &= check_moves(root / "store", f"pid_{part}", question_sources(
            root / "store", f"pid_{part}", answers, source_games), train) == 0
        _, _, pool_meta = load_store(f"pool_{part}", root / "store")
        ok &= check_moves(root / "store", f"pool_{part}",
                          pool_meta["game_id"].str[1:].astype(int).to_numpy(), train) == 0

        candidates, gal, q = data.split_view(gallery, questions, split)
        population = np.flatnonzero(black & ~np.isin(pid, candidates))
        scores = opening_scores(prefix, gal, list(q["games"]), population, params)
        log_q = assign.sinkhorn(assign.row_posterior(scores, params["tau"]))
        reference = pd.DataFrame(candidates[assign.hungarian_top5(log_q)],
                                 columns=[f"top{j}" for j in range(1, 6)])
        reference["source_question_id"] = q["question_id"].to_numpy()
        pred = pd.read_parquet(root / "identity" / part / "predictions.parquet")
        pred["question_id"] = pred["question_id"].astype(int)
        merged = answers.merge(pred, on="question_id").merge(reference, on="source_question_id",
                                                             suffixes=("", "_ref"))
        cols = [f"top{j}" for j in range(1, 6)]
        same = int((merged[cols].to_numpy() == merged[[c + "_ref" for c in cols]].to_numpy()).all(axis=1).sum())
        passed = same == len(merged) == 400
        ok &= passed
        print(f"  (2) pid_{part}: top-5 identical {same}/{len(merged)} -> {'PASS' if passed else 'FAIL'}")
        report = pid_report(merged[cols].to_numpy(), merged["player_id"].astype(str).to_numpy(), merged["k"])
        print("      score by k:", report["score"].round(4).to_dict())
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs" / "rehearsal")
    parser.add_argument("--skip-strength", action="store_true")
    parser.add_argument("--skip-identity", action="store_true")
    args = parser.parse_args()
    train = TrainMoves()
    results = {}
    if not args.skip_identity:
        print("competition 1")
        results["identity"] = identity(args.root, train)
    if not args.skip_strength:
        print("competition 2")
        results["strength"] = strength(args.root, train)
    print(json.dumps({k: "PASS" if v else "FAIL" for k, v in results.items()}))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
