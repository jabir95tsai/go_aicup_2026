"""CPU evaluation of saved, index-aligned V1 scores; no model inference."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.data.board_cache import _close_mmap
from go_aicup.metrics import K_BUCKETS, NEAR, rank_report
from go_aicup.tasks.strength.runtime import file_hash, load_dataset_manifest, check_index_hashes

CHUNK = 1_000_000


def decode_scores(scores, boundaries) -> np.ndarray:
    scores, boundaries = np.asarray(scores, dtype=np.float64), np.asarray(boundaries, dtype=np.float64)
    if boundaries.shape != (9,) or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) < 0):
        raise ValueError("nine finite nondecreasing cutpoints required")
    if not np.isfinite(scores).all():
        raise ValueError("question scores must be finite")
    return np.searchsorted(boundaries, scores, side="right").astype(np.int64)


def fit_cutpoints(scores, rank_ids, *, scratch_dir=None, stop_check=None) -> dict:
    """Exact ordered-interval DP on ALL training plies, each weighted equally.

    Vectorized prefix maxima preserve first-predecessor ties. Backpointers
    use temporary uint32 memmaps rather than nine in-memory int64 vectors.
    Equal scores are indivisible; repeated boundaries/empty ranks are legal.
    """
    scores, labels = np.asarray(scores), np.asarray(rank_ids)
    if scores.ndim != 1 or not len(scores) or scores.shape != labels.shape:
        raise ValueError("nonempty equal-length score/label vectors required")
    for start in range(0, len(scores), CHUNK):
        values, ranks = scores[start:start + CHUNK], labels[start:start + CHUNK]
        if not np.isfinite(values).all() or np.any((ranks < 0) | (ranks > 9)) or ranks.dtype.kind not in "iu":
            raise ValueError("invalid training scores or ranks")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int8, copy=False)
    del order
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    unique = sorted_scores[starts].astype(np.float64)
    del sorted_scores
    size = len(unique) + 1
    if size > np.iinfo(np.uint32).max:
        raise ValueError("too many unique scores for uint32 DP")

    def prefix_for(prediction):
        reward = (sorted_labels == prediction).astype(np.float64)
        reward += (np.abs(sorted_labels - prediction) == 1) * NEAR
        prefix = np.empty(size, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(np.add.reduceat(reward, starts) / len(scores), out=prefix[1:])
        return prefix

    previous = prefix_for(0)
    with tempfile.TemporaryDirectory(prefix="cutpoints_", dir=scratch_dir) as temporary:
        pointers = np.lib.format.open_memmap(Path(temporary) / "predecessors.npy", mode="w+",
                                             dtype=np.uint32, shape=(9, size))
        try:
            for prediction in range(1, 10):
                if stop_check is not None and stop_check():
                    raise InterruptedError("cutpoint fitting interrupted")
                prefix = prefix_for(prediction)
                candidates = previous - prefix
                maxima = np.maximum.accumulate(candidates)
                records = np.r_[True, candidates[1:] > maxima[:-1]]
                predecessors = np.where(records, np.arange(size, dtype=np.uint32), np.uint32(0))
                np.maximum.accumulate(predecessors, out=pointers[prediction - 1])
                previous = prefix + maxima
            ends, end = [0] * 9, len(unique)
            for prediction in range(9, 0, -1):
                end = int(pointers[prediction - 1, end])
                ends[prediction - 1] = end
        finally:
            _close_mmap(pointers)
    above_max = float(np.nextafter(unique[-1], np.inf))
    if not np.isfinite(above_max):
        raise ValueError("finite upper cutpoint cannot be represented")
    boundaries = [float(unique[index]) if index < len(unique) else above_max for index in ends]
    return {"boundaries": boundaries, "training_ply_score": float(previous[-1]),
            "unique_scores": len(unique), "num_training_plies": len(scores),
            "scope": "all training positions; equal weight per position; no validation labels",
            "boundary_rule": "searchsorted(side=right)"}


def question_scores(questions: pd.DataFrame, games: np.ndarray) -> np.ndarray:
    result = []
    for row in questions.itertuples(index=False):
        indices = np.asarray(row.games, dtype=np.int64)
        if len(indices) != row.k or len(np.unique(indices)) != row.k:
            raise ValueError("question repeats games or disagrees with k")
        values = games[indices]
        if not np.isfinite(values).all():
            raise ValueError(f"missing game predictions for question {row.question_id}")
        result.append(float(values.mean()))
    return np.asarray(result)


def score_report(scores, labels, boundaries) -> dict:
    """Chunked metrics, avoiding full-size prediction/error arrays."""
    totals = np.zeros(3, dtype=np.float64)
    for start in range(0, len(scores), CHUNK):
        error = np.abs(decode_scores(scores[start:start + CHUNK], boundaries) - labels[start:start + CHUNK])
        totals += [np.where(error == 0, 1., np.where(error == 1, NEAR, 0.)).sum(),
                   (error == 0).sum(), error.sum()]
    return dict(zip(("score", "acc", "mae"), (totals / len(scores)).tolist()))


def aggregate_games(scores, indices, num_games):
    sums = np.zeros(num_games, dtype=np.float64)
    counts = np.zeros(num_games, dtype=np.int64)
    for start in range(0, len(scores), CHUNK):
        games = indices[start:start + CHUNK]
        values = scores[start:start + CHUNK]
        if not np.isfinite(values).all() or np.any((games < 0) | (games >= num_games)):
            raise ValueError("nonfinite scores or invalid game indices")
        sums += np.bincount(games.astype(np.int64), weights=values, minlength=num_games)
        counts += np.bincount(games.astype(np.int64), minlength=num_games)
    means = np.full(num_games, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means, counts


def load_saved_run(run: Path, dataset_dir: Path | None = None, cache_dir: Path | None = None):
    """Validate artifact hashes, model identity and external processed inputs."""
    run = Path(run)
    payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "strength_v1_compact_1":
        raise ValueError("expected a completed compact V1 model.pt")
    config = payload["config"]
    if dataset_dir is not None:
        config["paths"]["dataset_dir"] = str(dataset_dir.resolve())
    if cache_dir is not None:
        config["paths"]["cache_dir"] = str(cache_dir.resolve())
    manifest = load_dataset_manifest(config)
    if manifest["dataset_id"] != payload["dataset_id"] or config["seed"] != payload["seed"]:
        raise ValueError("score dataset/seed mismatch")
    check_index_hashes(config, manifest)
    for name, digest in payload["artifact_sha256"].items():
        if name not in ("losses.npy", "train_scores.npy", "valid_scores.npy") or file_hash(run / name) != digest:
            raise ValueError(f"artifact integrity mismatch: {name}")
    return payload, config, manifest


def evaluate_saved(run: Path, *, payload=None, dataset_dir=None, cache_dir=None,
                   fit=False, stop_check=None):
    """Return reports from saved scores. Training passes its unpublished payload."""
    run = Path(run)
    if payload is None:
        payload, config, manifest = load_saved_run(run, dataset_dir, cache_dir)
    else:
        config = payload["config"]
        manifest = load_dataset_manifest(config)
    directory = Path(config["paths"]["dataset_dir"])
    meta = pd.read_parquet(directory / "strength_v1_games.parquet")
    if not np.array_equal(meta["game_index"], np.arange(len(meta))):
        raise ValueError("game_index must match metadata row order")
    all_labels = meta["rank_id"].to_numpy(dtype=np.int8)
    results = {"seed": payload["seed"], "dataset_id": manifest["dataset_id"],
               "player_overlap": manifest["player_overlap"]}
    boundaries = None if fit else payload["cutpoints"]["boundaries"]
    for prefix in ("train", "valid"):
        scores = np.load(run / f"{prefix}_scores.npy", mmap_mode="r", allow_pickle=False)
        indices = np.load(directory / f"{prefix}_game.npy", mmap_mode="r", allow_pickle=False)
        plies = np.load(directory / f"{prefix}_ply.npy", mmap_mode="r", allow_pickle=False)
        try:
            expected = manifest["sets"][prefix]
            if scores.dtype != np.float32 or scores.shape != (expected["num_samples"],) or indices.shape != scores.shape or plies.shape != scores.shape:
                raise ValueError(f"{prefix}: invalid score/index shape or dtype")
            games, counts = aggregate_games(scores, indices, len(meta))
            selected = counts > 0
            if selected.sum() != expected["num_games"] or not np.array_equal(selected, meta["split"].to_numpy() == prefix):
                raise ValueError(f"{prefix}: incorrect game coverage")
            if not np.array_equal(counts[selected], meta.loc[selected, "player_ply_count"].to_numpy()):
                raise ValueError(f"{prefix}: incorrect per-game ply coverage")
            labels = all_labels[indices]
            if prefix == "train" and fit:
                payload["cutpoints"] = fit_cutpoints(scores, labels, scratch_dir=run, stop_check=stop_check)
                boundaries = payload["cutpoints"]["boundaries"]
            results[prefix] = {"plies": score_report(scores, labels, boundaries),
                               "games": score_report(games[selected], all_labels[selected], boundaries),
                               "num_samples": len(scores), "num_games": int(selected.sum())}
            if prefix == "valid":
                questions = pd.read_parquet(directory / "strength_v1_validation_questions.parquet")
                for row in questions.itertuples(index=False):
                    chosen = meta.iloc[np.asarray(row.games, dtype=np.int64)]
                    if not (chosen["split"] == "valid").all() or not (chosen["rank_id"] == row.rank_id).all() or chosen["player_id"].nunique() != 1 or chosen["color"].nunique() != 1:
                        raise ValueError("invalid validation question membership")
                values = question_scores(questions, games)
                predictions = decode_scores(values, boundaries)
                report = rank_report(predictions, questions["rank_id"].to_numpy(), questions["k"].to_numpy())
                results["questions"] = {"by_k": report.to_dict(orient="index"),
                    "equal_k": {key: float(report.loc[[str(k) for k in K_BUCKETS], key].mean()) for key in ("score", "acc", "mae")}}
        finally:
            for array in (scores, indices, plies):
                _close_mmap(array)
    results["cutpoints"] = payload["cutpoints"]
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, help="Override processed location after moving artifacts")
    parser.add_argument("--cache-dir", type=Path, help="Override shared board cache location")
    args = parser.parse_args(argv)
    result = evaluate_saved(args.run_dir, dataset_dir=args.dataset_dir, cache_dir=args.cache_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
