"""Fixed-weight V1 inference, train-only ordered cutpoints and k-bucket scores.

All positions -> equal-weight game means -> equal-weight same-player/color
question means. Calibration uses only the fixed training questions. No
validation labels enter the cutpoint optimizer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from go_aicup.data.board_cache import _close_mmap
from go_aicup.metrics import K_BUCKETS, NEAR, rank_report
from go_aicup.models.strength_resnet import StrengthResNet
from go_aicup.tasks.strength.dataset import GlobalBatchSampler, PackedBoardDataset, make_loader
from go_aicup.tasks.strength.losses import distributed, rank_statistics, ranking_loss_from_means
from go_aicup.tasks.strength.runtime import (DEFAULT_CONFIG, autocast_context, barrier,
    check_index_hashes, file_hash, load_dataset_manifest, read_config,
    setup_distributed, write_json)


class EvaluationInterrupted(RuntimeError):
    """All processes stopped at a shared inference batch boundary."""


def decode_scores(scores, boundaries) -> np.ndarray:
    scores, boundaries = np.asarray(scores, dtype=np.float64), np.asarray(boundaries, dtype=np.float64)
    if boundaries.shape != (9,) or not np.isfinite(boundaries).all() or np.any(np.diff(boundaries) < 0):
        raise ValueError("nine finite nondecreasing cutpoints required")
    if not np.isfinite(scores).all():
        raise ValueError("question scores must be finite")
    return np.searchsorted(boundaries, scores, side="right").astype(np.int64)


def fit_cutpoints(scores, rank_ids, k) -> dict:
    """Exact ordered-interval DP, weighting each of the four k buckets 1/4.

    Equal scores are indivisible. Empty intervals/repeated boundaries are
    legal. Ties use the first predecessor for reproducible backtracking.
    Complexity O(10 * number_of_distinct_scores), excluding sorting.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rank_ids, k = np.asarray(rank_ids, dtype=np.int64), np.asarray(k, dtype=np.int64)
    if scores.ndim != 1 or not len(scores) or scores.shape != rank_ids.shape or scores.shape != k.shape:
        raise ValueError("calibration inputs must be nonempty equal-length vectors")
    if not np.isfinite(scores).all() or np.any((rank_ids < 0) | (rank_ids > 9)):
        raise ValueError("invalid calibration scores or ranks")
    if set(k.tolist()) != set(K_BUCKETS):
        raise ValueError("calibration requires all four k buckets")
    weights = np.empty(len(scores), dtype=np.float64)
    for bucket in K_BUCKETS:
        selected = k == bucket
        weights[selected] = 0.25 / int(selected.sum())
    unique, inverse = np.unique(scores, return_inverse=True)
    rewards = np.zeros((10, len(unique)), dtype=np.float64)
    for prediction in range(10):
        error = np.abs(rank_ids - prediction)
        reward = np.where(error == 0, 1.0, np.where(error == 1, NEAR, 0.0)) * weights
        rewards[prediction] = np.bincount(inverse, weights=reward, minlength=len(unique))
    prefix = np.pad(np.cumsum(rewards, axis=1), ((0, 0), (1, 0)))
    previous = prefix[0].copy()
    predecessors = np.zeros((10, len(unique) + 1), dtype=np.int64)
    for prediction in range(1, 10):
        candidates = previous - prefix[prediction]
        best, where = -np.inf, 0
        current = np.empty_like(previous)
        for end, candidate in enumerate(candidates):
            if candidate > best:
                best, where = float(candidate), end
            predecessors[prediction, end] = where
            current[end] = prefix[prediction, end] + best
        previous = current
    ends = [0] * 9
    end = len(unique)
    for prediction in range(9, 0, -1):
        end = int(predecessors[prediction, end])
        ends[prediction - 1] = end
    above_max = float(np.nextafter(unique[-1], np.inf))
    if not np.isfinite(above_max):
        raise ValueError("calibration score magnitude cannot be represented by finite cutpoints")
    boundaries = [float(unique[index]) if index < len(unique) else above_max for index in ends]
    predictions = decode_scores(scores, boundaries)
    score = sum(float((np.where(np.abs(predictions[k == bucket] - rank_ids[k == bucket]) == 0, 1.0,
                    np.where(np.abs(predictions[k == bucket] - rank_ids[k == bucket]) == 1, NEAR, 0.0))).mean())
                for bucket in K_BUCKETS) / 4
    return {"boundaries": boundaries, "training_equal_k_score": score,
            "dp_reward": float(previous[-1]), "unique_scores": len(unique),
            "empty_predicted_ranks": sorted(set(range(10)) - set(predictions.tolist())),
            "boundary_rule": "searchsorted(side=right); score equal to cutpoint enters higher interval"}


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


def distribution(scores, labels) -> dict:
    scores, labels = np.asarray(scores), np.asarray(labels)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("empty/nonfinite score distribution")
    low, high = float(scores.min()), float(scores.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    histograms, records = [], []
    for rank in range(10):
        values = scores[labels == rank]
        if not len(values):
            raise ValueError(f"distribution missing rank {rank}")
        hist, edges = np.histogram(values, bins=64, range=(low, high))
        histograms.append(hist / len(values))
        records.append({"rank_id": rank, "n": len(values), "mean": float(values.mean(dtype=np.float64)),
                        "std": float(values.std(dtype=np.float64)),
                        "quantiles_05_25_50_75_95": np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist(),
                        "histogram": hist.tolist()})
    return {"ranks": records, "histogram_edges": edges.tolist(),
            "adjacent_histogram_overlap": [float(np.minimum(histograms[r], histograms[r + 1]).sum())
                                            for r in range(9)]}


@torch.no_grad()
def infer_positions(model, config: dict, manifest: dict, prefix: str, output: Path,
                    device, rank: int, world: int, stop_check=None):
    directory = Path(config["paths"]["dataset_dir"])
    meta = pd.read_parquet(Path(config["paths"]["cache_dir"]) / "games.parquet")
    dataset = PackedBoardDataset(directory / f"{prefix}_game.npy", directory / f"{prefix}_ply.npy",
                                Path(config["paths"]["cache_dir"]),
                                max_open_shards=config["loader"]["max_open_shards"], seed=config["seed"])
    sampler = GlobalBatchSampler(len(dataset), config["evaluation"]["global_batch_size"], rank=rank, world_size=world)
    options = {key: config["loader"][key] for key in
               ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")}
    loader = make_loader(dataset, sampler, **options)
    score_path = output / f"{prefix}_ply_scores.npy"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(len(dataset),))
        array[:] = np.nan
        array.flush()
        _close_mmap(array)
    barrier()
    score_store = np.load(score_path, mmap_mode="r+", allow_pickle=False)
    game_sums = np.zeros(len(meta), dtype=np.float64)
    game_counts = np.zeros(len(meta), dtype=np.float64)
    rank_sums = torch.zeros(10, dtype=torch.float64, device=device)
    rank_counts = torch.zeros(10, dtype=torch.float64, device=device)
    batch_losses, missing_batches = [], []
    started = time.monotonic()
    model.eval()
    try:
        for step, batch in enumerate(loader):
            stopping = torch.tensor(int(stop_check is not None and stop_check()),
                                    device=device, dtype=torch.int32)
            if distributed():
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                raise EvaluationInterrupted("inference interrupted; partial outputs have no complete.json")
            valid = batch["valid"].bool()
            labels = batch["rank_id"][valid].to(device, non_blocking=True)
            if valid.any():
                x = batch["planes"][valid].to(device, dtype=torch.float32, non_blocking=True)
                if config["training"]["channels_last"]:
                    x = x.contiguous(memory_format=torch.channels_last)
                with autocast_context(config, device):
                    scores = model(x).float()
            else:
                scores = torch.empty(0, device=device, dtype=torch.float32)
            sums, counts = rank_statistics(scores, labels, differentiable=False)
            rank_sums += sums.double()
            rank_counts += counts.double()
            if (counts > 0).all():
                batch_losses.append({"batch": step, "n": int(counts.sum().item()),
                                     "loss": float(ranking_loss_from_means(sums / counts).item())})
            else:
                missing_batches.append({"batch": step, "n": int(counts.sum().item()),
                                        "missing_ranks": torch.where(counts == 0)[0].tolist()})
            values = scores.cpu().numpy()
            indices = batch["sample_index"][valid].numpy()
            games = batch["game_index"][valid].numpy()
            score_store[indices] = values
            np.add.at(game_sums, games, values)
            np.add.at(game_counts, games, 1)
            if rank == 0 and (step + 1) % config["training"]["log_every_steps"] == 0:
                print(f"eval {prefix}: batch {step + 1}/{len(sampler)}", flush=True)
        score_store.flush()
    finally:
        dataset.close()
        _close_mmap(score_store)
    if distributed():
        for array in (game_sums, game_counts):
            tensor = torch.from_numpy(array).to(device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            array[:] = tensor.cpu().numpy()
    if int(game_counts.sum()) != manifest["sets"][prefix]["num_samples"]:
        raise ValueError("inference did not cover each sample exactly once")
    game_scores = np.full(len(meta), np.nan, dtype=np.float64)
    np.divide(game_sums, game_counts, out=game_scores, where=game_counts > 0)
    barrier()
    diagnostics = None
    if rank == 0:
        selected = game_counts > 0
        np.save(output / f"{prefix}_game_scores.npy", game_scores)
        meta.loc[selected, ["game_index", "rank_id", "player_id", "color"]].assign(
            score=game_scores[selected], player_plies=game_counts[selected].astype(np.int64)
        ).to_parquet(output / f"{prefix}_game_scores.parquet", index=False)
        ply_scores = np.load(score_path, mmap_mode="r", allow_pickle=False)
        game_indices = np.load(directory / f"{prefix}_game.npy", mmap_mode="r", allow_pickle=False)
        try:
            ply_labels = meta["rank_id"].to_numpy()[game_indices]
            ply_distribution = distribution(ply_scores, ply_labels)
        finally:
            _close_mmap(ply_scores)
            _close_mmap(game_indices)
        means = (rank_sums / rank_counts).float()
        diagnostics = {"num_samples": int(game_counts.sum()), "num_games": int(selected.sum()),
                       "seconds": time.monotonic() - started,
                       "batch_loss_unweighted": float(np.mean([row["loss"] for row in batch_losses])) if batch_losses else None,
                       "batch_loss_sample_weighted": float(np.average([row["loss"] for row in batch_losses],
                                                          weights=[row["n"] for row in batch_losses])) if batch_losses else None,
                       "batch_losses": batch_losses, "missing_rank_batches": missing_batches,
                       "whole_dataset_mean_loss": float(ranking_loss_from_means(means).item()),
                       "rank_means": means.cpu().tolist(), "rank_counts": rank_counts.cpu().tolist(),
                       "ordered_means": bool((means[1:] > means[:-1]).all().item()),
                       "ply_distribution": ply_distribution,
                       "game_distribution": distribution(game_scores[selected], meta["rank_id"].to_numpy()[selected])}
        if not config["evaluation"]["save_ply_scores"]:
            score_path.unlink()
    return game_scores, diagnostics


def evaluate(model, config: dict, manifest: dict, checkpoint: Path, output: Path,
             device, rank: int = 0, world: int = 1, stop_check=None) -> dict | None:
    if rank == 0 and output.exists() and any(output.iterdir()):
        raise ValueError(f"evaluation output must be new/empty: {output}")
    barrier()
    training_games, training_info = infer_positions(model, config, manifest, "calibration", output, device, rank, world, stop_check)
    validation_games, validation_info = infer_positions(model, config, manifest, "valid", output, device, rank, world, stop_check)
    result = None
    if rank == 0:
        cv_dir = Path(config["paths"]["cv_dir"])
        train_questions = pd.read_parquet(cv_dir / "strength_v1_train_calibration_questions.parquet")
        valid_questions = pd.read_parquet(cv_dir / "strength_v1_validation_questions.parquet")
        train_scores = question_scores(train_questions, training_games)
        valid_scores = question_scores(valid_questions, validation_games)
        cutpoints = fit_cutpoints(train_scores, train_questions["rank_id"], train_questions["k"])
        cutpoints.update(checkpoint_sha256=file_hash(checkpoint), dataset_id=manifest["dataset_id"],
                         training_questions_sha256=manifest["cv_files"]["strength_v1_train_calibration_questions.parquet"]["sha256"])
        write_json(output / "cutpoints.json", cutpoints)
        reports = {}
        for name, questions, scores in (("training", train_questions, train_scores),
                                        ("validation", valid_questions, valid_scores)):
            predictions = decode_scores(scores, cutpoints["boundaries"])
            report = rank_report(predictions, questions["rank_id"].to_numpy(), questions["k"].to_numpy())
            equal_k = {key: float(report.loc[[str(k) for k in K_BUCKETS], key].mean())
                       for key in ("score", "acc", "mae")}
            confusion = {}
            for k in K_BUCKETS:
                mask = questions["k"].to_numpy() == k
                matrix = np.zeros((10, 10), dtype=np.int64)
                np.add.at(matrix, (questions["rank_id"].to_numpy()[mask], predictions[mask]), 1)
                confusion[str(k)] = matrix.tolist()
            reports[name] = {"by_k": report.to_dict(orient="index"), "equal_k": equal_k,
                             "confusion_true_rows_pred_columns": confusion,
                             "question_distribution": distribution(scores, questions["rank_id"].to_numpy())}
            questions.assign(score=scores, prediction=predictions).to_parquet(
                output / f"{name}_question_predictions.parquet", index=False)
        result = {"checkpoint": str(checkpoint), "checkpoint_sha256": cutpoints["checkpoint_sha256"],
                  "dataset_id": manifest["dataset_id"], "player_overlap": manifest["player_overlap"],
                  "calibration_scope": "only unique training games used by fixed calibration questions; frozen model/BN, no D4",
                  "cutpoints": cutpoints, "training_calibration": training_info,
                  "validation": validation_info, "questions": reports}
        write_json(output / "metrics.json", result)
        write_json(output / "complete.json", {"checkpoint_sha256": cutpoints["checkpoint_sha256"],
                                               "metrics_sha256": file_hash(output / "metrics.json")})
        print(json.dumps(reports["validation"]["equal_k"], indent=2), flush=True)
    barrier()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    manifest = load_dataset_manifest(config)
    check_index_hashes(config, manifest)
    from go_aicup.tasks.strength.train_rank_v1 import configure_torch, resolve_checkpoint
    configure_torch(config)
    checkpoint_path = resolve_checkpoint(args.checkpoint.resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["dataset_id"] != manifest["dataset_id"] or checkpoint["config"]["model"] != config["model"]:
        raise ValueError("checkpoint dataset/model does not match evaluation config")
    device, rank, world = setup_distributed(config)
    try:
        model = StrengthResNet(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        if config["training"]["channels_last"]:
            model.to(memory_format=torch.channels_last)
        evaluate(model, config, manifest, checkpoint_path, args.output_dir.resolve(), device, rank, world)
    finally:
        if distributed():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
