"""Plot V1 training loss and completed validation evaluation (CPU only).

Example:
    python scripts/plot_rank_v1.py --run-dir outputs/rank/v1/rank_cnn_v1_seed2026
    python scripts/plot_rank_v1.py --run-dir outputs/my_run --loss-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
K_BUCKETS = (5, 10, 15, 20)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # A resumed session may replay updates after its most recent checkpoint.
    # Discard the abandoned branch when its step numbers are replayed.
    rows: dict[int, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                warnings.warn(f"Ignoring incomplete final JSONL line: {path}:{number}")
                continue
            raise ValueError(f"Invalid JSONL: {path}:{number}") from None
        if record.get("event") != "update":
            continue
        step, loss = int(record["optimizer_steps"]), float(record["loss"])
        if step < 1 or not np.isfinite(loss):
            raise ValueError(f"Invalid training update: {path}:{number}")
        if rows and step <= next(reversed(rows)):
            rows = {key: value for key, value in rows.items() if key < step}
        rows[step] = loss
    if not rows:
        raise ValueError(f"No training update events in {path}")
    return np.asarray(list(rows)), np.asarray(list(rows.values()))


def load_metrics(directory: Path) -> dict:
    complete = read_json(directory / "complete.json")
    metrics_path = directory / "metrics.json"
    if hashlib.sha256(metrics_path.read_bytes()).hexdigest() != complete["metrics_sha256"]:
        raise ValueError(f"Evaluation metrics hash mismatch: {directory}")
    metrics = read_json(metrics_path)
    if complete["checkpoint_sha256"] != metrics["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint hashes disagree")
    return metrics


def load_evaluation(directory: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    metrics = load_metrics(directory)
    cutpoints = read_json(directory / "cutpoints.json")
    if metrics["checkpoint_sha256"] != cutpoints["checkpoint_sha256"]:
        raise ValueError("Evaluation checkpoint hashes disagree")
    boundaries = np.asarray(cutpoints["boundaries"], dtype=float)
    if (boundaries.shape != (9,) or not np.isfinite(boundaries).all()
            or np.any(np.diff(boundaries) < 0)
            or not np.array_equal(boundaries, metrics["cutpoints"]["boundaries"])):
        raise ValueError("Expected nine finite, nondecreasing, matching cutpoints")
    table = pd.read_parquet(directory / "validation_question_predictions.parquet",
                            columns=["rank_id", "k", "score", "prediction"])
    if table.empty or table.isna().any().any() or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Validation questions are empty or contain nonfinite values")
    if (not table["rank_id"].isin(range(10)).all()
            or not table["prediction"].isin(range(10)).all()
            or set(table["k"]) != set(K_BUCKETS)):
        raise ValueError("Invalid rank IDs, predictions or k buckets")
    decoded = np.searchsorted(boundaries, table["score"].to_numpy(), side="right")
    if not np.array_equal(decoded, table["prediction"].to_numpy()):
        raise ValueError("Saved predictions disagree with saved cutpoints")
    for k in K_BUCKETS:
        if set(table.loc[table["k"] == k, "rank_id"]) != set(range(10)):
            raise ValueError(f"Validation k={k} does not contain all ten ranks")
    return table, boundaries, metrics



def load_validation_losses(run: Path, selected: Path | None) -> tuple[np.ndarray, np.ndarray]:
    directories = {path.parent.resolve() for path in (run / "evaluation").glob("epoch_*/complete.json")}
    if selected is not None:
        directories.add(selected.resolve())
    rows = {}
    for directory in sorted(directories):
        metrics = load_metrics(directory)
        # Trainer filenames retain exact global steps even after checkpoints are pruned.
        match = re.match(r"update_(\d+)_epoch_", Path(metrics["checkpoint"]).name)
        if match is None:
            raise ValueError(f"Cannot recover optimizer step from checkpoint: {metrics['checkpoint']}")
        step = int(match.group(1))
        loss = metrics["validation"].get("batch_loss_sample_weighted")
        if loss is None:
            warnings.warn(f"Skipping unavailable validation batch loss: {directory}. "
                          "Evaluation batches may not contain all ten ranks.")
            continue
        if not np.isfinite(loss):
            raise ValueError(f"Nonfinite validation batch loss: {directory}")
        rows[step] = float(loss)
    steps = np.asarray(sorted(rows), dtype=int)
    return steps, np.asarray([rows[step] for step in steps])


def load_game_scores(directory: Path, metrics: dict) -> pd.DataFrame:
    table = pd.read_parquet(directory / "valid_game_scores.parquet",
                            columns=["game_index", "rank_id", "score", "player_plies"])
    if table.empty or table.isna().any().any() or not np.isfinite(table.to_numpy()).all():
        raise ValueError("Validation game scores are empty or nonfinite")
    if (set(table["rank_id"]) != set(range(10)) or table["game_index"].duplicated().any()
            or (table["player_plies"] <= 0).any()):
        raise ValueError("Invalid game ranks, duplicate games or empty games")
    info = metrics["validation"]
    if len(table) != info["num_games"] or table["player_plies"].sum() != info["num_samples"]:
        raise ValueError("Game scores disagree with completed validation coverage")
    return table


def save_figure(fig, output: Path, filename: str) -> None:
    path = output / filename
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def plot_loss(steps, losses, output: Path, window: int, validation_steps, validation_losses) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.plot(steps, losses, color="#4c78a8", alpha=0.45, linewidth=0.8, label="Logged batch loss")
    if window > 1 and len(losses) >= window:
        smooth = pd.Series(losses).rolling(window, min_periods=window).mean().to_numpy()
        ax.plot(steps, smooth, color="#153f68", linewidth=1.6,
                label=f"Trailing mean ({window} logged updates)")
    if len(validation_steps):
        ax.plot(validation_steps, validation_losses, "o-", color="#e45756", linewidth=1.8,
                label="Validation batch loss (sample-weighted)")
    title = "Strength V1: training and validation loss" if len(validation_steps) else "Strength V1: training loss"
    ax.set(title=title, xlabel="Optimizer step (global, across resumes)",
           ylabel="Global rank-mean ranking loss")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, output, "loss_vs_step.png")


def plot_confusion(table: pd.DataFrame, output: Path, label: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 13), layout="constrained")
    for ax, k in zip(axes.flat, K_BUCKETS):
        part = table.loc[table["k"] == k]
        matrix = np.zeros((10, 10), dtype=np.int64)
        np.add.at(matrix, (part["rank_id"].to_numpy(dtype=int),
                           part["prediction"].to_numpy(dtype=int)), 1)
        fractions = matrix / matrix.sum(axis=1, keepdims=True)
        image = ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
        for row in range(10):
            for col in range(10):
                ax.text(col, row, f"{fractions[row, col]:.0%}\n({matrix[row, col]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if fractions[row, col] > 0.55 else "#222222")
        error = np.abs(part["rank_id"].to_numpy() - part["prediction"].to_numpy())
        score = np.where(error == 0, 1.0, np.where(error == 1, np.exp(-1), 0.0)).mean()
        ax.set(xticks=range(10), yticks=range(10), xticklabels=RANKS, yticklabels=RANKS,
               xlabel="Predicted rank", ylabel="True rank",
               title=f"k={k} | n={len(part):,} | official score={score:.4f}")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Fraction within true rank", shrink=0.7)
    fig.suptitle(f"Validation question confusion matrices | {label}\nCells: row percentage (question count)")
    save_figure(fig, output, "confusion_matrix.png")


def plot_distributions(table: pd.DataFrame, boundaries, output: Path, label: str, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    low = min(table["score"].min(), boundaries.min())
    high = max(table["score"].max(), boundaries.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, bins + 1)
    for rank, color in enumerate(plt.get_cmap("tab10").colors):
        values = table.loc[table["rank_id"] == rank, "score"].to_numpy()
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=1.6,
                color=color, label=f"{RANKS[rank]} (n={len(values):,})")
    for index, boundary in enumerate(boundaries, 1):
        ax.axvline(boundary, color="#333333", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.text(boundary, 0.98 - ((index - 1) % 3) * 0.075, f"t{index}",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    ax.set(title=f"Validation per-game score distributions | {label}\n"
                 "One score per game: mean over all positions of the designated player\n"
                 "Dashed lines: saved training cutpoints",
           xlabel="Game scalar score (mean over designated player's positions)",
           ylabel="Density within true rank")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), title="True rank")
    values_text = " | ".join(f"t{i}={value:.5g}" for i, value in enumerate(boundaries, 1))
    fig.supxlabel(values_text, fontsize=9)
    save_figure(fig, output, "score_distributions_cutpoints.png")


def plot_compact(args):
    from go_aicup.tasks.strength.v1.infer import LOSS_DTYPE
    losses = np.load(args.run_dir / "losses.npy", allow_pickle=False)
    if (losses.dtype != LOSS_DTYPE or losses.ndim != 1 or not len(losses)
            or (losses["step"] < 0).any() or (np.diff(losses["step"]) <= 0).any()
            or not np.isfinite(losses["train_loss"]).all()
            or not np.isfinite(losses["valid_loss"]).all()):
        raise ValueError("Invalid or empty full-dataset losses.npy")
    if args.evaluation_dir:
        raise ValueError("--evaluation-dir applies only to legacy runs")
    payload = config = manifest = None
    if not args.loss_only:
        from go_aicup.tasks.strength.v1.evaluate import load_saved_run
        payload, config, manifest = load_saved_run(args.run_dir, args.dataset_dir, args.cache_dir)
    output = args.output_dir or args.run_dir / "plots"
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.plot(losses["step"], losses["train_loss"], "o-", label="Training batch loss (train mode)")
    ax.plot(losses["step"], losses["valid_loss"], "o-", label="Validation loss (fixed 100,000 positions; eval mode)")
    ax.set(title=f"Strength V1: ranking losses | {args.run_dir.name}", xlabel="Optimizer step",
           ylabel="Ranking loss")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, output, "loss_vs_step.png")
    if args.loss_only:
        return
    from go_aicup.data.board_cache import _close_mmap
    from go_aicup.tasks.strength.v1.evaluate import aggregate_games, question_scores, decode_scores
    directory = Path(config["paths"]["dataset_dir"])
    meta = pd.read_parquet(directory / "strength_v1_games.parquet")
    scores = np.load(args.run_dir / "valid_scores.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(directory / "valid_game.npy", mmap_mode="r", allow_pickle=False)
    try:
        if scores.dtype != np.float32 or scores.shape != indices.shape or scores.shape != (manifest["sets"]["valid"]["num_samples"],):
            raise ValueError("Validation scores disagree with indices/manifest")
        means, counts = aggregate_games(scores, indices, len(meta))
    finally:
        _close_mmap(scores)
        _close_mmap(indices)
    selected = counts > 0
    if not np.array_equal(selected, meta["split"].to_numpy() == "valid") or not np.array_equal(counts[selected], meta.loc[selected, "player_ply_count"].to_numpy()):
        raise ValueError("Validation game coverage mismatch")
    boundaries = np.asarray(payload["cutpoints"]["boundaries"])
    questions = pd.read_parquet(directory / "strength_v1_validation_questions.parquet")
    values = question_scores(questions, means)
    table = questions.assign(score=values, prediction=decode_scores(values, boundaries))
    games = meta.loc[selected, ["game_index", "rank_id"]].assign(score=means[selected], player_plies=counts[selected])
    label = f"seed {payload['seed']} | step {payload['optimizer_steps']}"
    if not args.skip_confusion:
        plot_confusion(table, output, label)
    plot_distributions(games, boundaries, output, label, args.bins)
    print("Cutpoint scope: " + payload["cutpoints"]["scope"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run containing train.jsonl")
    parser.add_argument("--evaluation-dir", type=Path,
                        help="Specific completed evaluation; default: latest completed epoch in run")
    parser.add_argument("--output-dir", type=Path, help="Default: selected evaluation/plots, or run/plots for loss-only")
    parser.add_argument("--dataset-dir", type=Path, help="Compact run: relocated processed directory")
    parser.add_argument("--cache-dir", type=Path, help="Compact run: relocated board cache")
    parser.add_argument("--skip-confusion", action="store_true", help="Produce only loss and per-game distributions")
    parser.add_argument("--loss-only", action="store_true", help="Plot loss without requiring evaluation")
    parser.add_argument("--smooth-window", type=int, default=50, help="Trailing logged updates; 1 disables smoothing")
    parser.add_argument("--bins", type=int, default=50, help="Common histogram bins")
    args = parser.parse_args()
    if args.smooth_window < 1 or args.bins < 2:
        parser.error("--smooth-window must be >=1 and --bins must be >=2")
    if args.loss_only and args.evaluation_dir:
        parser.error("--loss-only cannot be combined with --evaluation-dir")
    try:
        if (args.run_dir / "losses.npy").exists():
            plot_compact(args)
            return
        steps, losses = load_loss(args.run_dir / "train.jsonl")
        evaluation = args.evaluation_dir
        if not args.loss_only:
            if evaluation is None:
                candidates = sorted((args.run_dir / "evaluation").glob("epoch_*/complete.json"))
                if not candidates:
                    raise ValueError("No completed evaluation found; use --evaluation-dir or --loss-only")
                evaluation = candidates[-1].parent
            table, boundaries, metrics = load_evaluation(evaluation)
            games = load_game_scores(evaluation, metrics)
        validation_steps, validation_losses = load_validation_losses(args.run_dir, evaluation)
        output = args.output_dir or ((args.run_dir if args.loss_only else evaluation) / "plots")
        output.mkdir(parents=True, exist_ok=True)
        plot_loss(steps, losses, output, args.smooth_window, validation_steps, validation_losses)
        if not args.loss_only:
            label = f"{evaluation.name} | checkpoint {metrics['checkpoint_sha256'][:12]}"
            if not args.skip_confusion:
                plot_confusion(table, output, label)
            plot_distributions(games, boundaries, output, label, args.bins)
            print("Validation split: " + json.dumps(metrics.get("player_overlap", {})))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
