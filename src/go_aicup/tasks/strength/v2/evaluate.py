"""CPU evaluation of a finished V2 run; no model forward, no SGF parsing.

Reads model.pt (config), losses.npy and the two per-game category files,
plus the run's dataset tables and fixed questions. Prints single-game and
k=5/10/15/20 question reports for Valid-U and Valid-S, the memorisation gap
G = Equal-k(S) - Equal-k(U), and saves PNG figures to <run-dir>/plot/.

Official question rule: mean per-game probability (each game 1/k), decoded
with metrics.score_rank. argmax and mean log-probability pooling are reported
as clearly labelled diagnostics; nothing is selected on validation scores.

    python -m go_aicup.tasks.strength.v2.evaluate --run-dir outputs/v2_strength
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, rank_report
from go_aicup.tasks.strength.v2.aggregate import (decode_questions, game_probabilities,
                                                  official_scores, question_probabilities)

SETS = {"unseen": "validation_unseen_category.npy", "seen": "validation_seen_category.npy"}
LABELS = {"unseen": "Valid-U (unseen players)", "seen": "Valid-S (seen players)"}
OFFICIAL = ("mean_probability", "score_rank")
AGGREGATIONS = ("mean_probability", "mean_log_probability")
DECODERS = ("score_rank", "argmax")
BASELINE, CANDIDATE = 0.4631, 0.47
GAP_LIMITS = (0.03, 0.08)
COLORS = {"unseen": "#2a78d6", "seen": "#eb6834", "train": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_run(run_dir: Path, dataset_dir: Path | None = None) -> dict:
    import torch
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "strength_v2_1":
        raise ValueError("not a Strength V2 run directory")
    directory = Path(dataset_dir or payload["config"]["paths"]["dataset_dir"])
    run = {"payload": payload, "losses": np.load(run_dir / "losses.npy", allow_pickle=False),
           "games": pd.read_parquet(directory / "strength_v2_games.parquet").set_index("game_index")}
    for name, file in SETS.items():
        run[name] = np.load(run_dir / file, allow_pickle=False)
        run[f"{name}_questions"] = pd.read_parquet(
            directory / f"strength_v2_valid_{name}_questions.parquet")
    return run


def game_report(category: np.ndarray, games: pd.DataFrame) -> dict:
    true = category["rank_id"].astype(int)
    pred = category["predicted_rank_id"].astype(int)
    error = np.abs(pred - true)
    scores = official_scores(pred, true)
    per_rank = pd.DataFrame({"rank": RANKS, "games": np.bincount(true, minlength=10),
                             "score": [scores[true == r].mean() if (true == r).any() else np.nan for r in range(10)],
                             "accuracy": [(pred[true == r] == r).mean() if (true == r).any() else np.nan
                                          for r in range(10)],
                             "predicted": np.bincount(pred, minlength=10)})
    colors = games.loc[category["game_index"], "color"].astype(str).to_numpy()
    by_color = pd.DataFrame([{"color": color, "games": int((colors == color).sum()),
                              "score": float(scores[colors == color].mean())}
                             for color in ("B", "W") if (colors == color).any()])
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (true, pred), 1)
    return {"games": len(category), "score": float(scores.mean()), "accuracy": float((error == 0).mean()),
            "off_by_one": float((error == 1).mean()), "off_by_two_plus": float((error >= 2).mean()),
            "per_rank": per_rank, "by_color": by_color, "confusion": confusion}


def question_report(category: np.ndarray, questions: pd.DataFrame) -> dict:
    """Per (aggregation, decoder): k-bucket table and Equal-k score."""
    true, k = questions["rank_id"].to_numpy(), questions["k"].to_numpy()
    if sorted(set(k)) != list(K_BUCKETS):
        raise ValueError("every k bucket needs questions for an Equal-k score")
    reports = {}
    for aggregation in AGGREGATIONS:
        probabilities = question_probabilities(category, questions["games"], aggregation)
        for decoder in DECODERS:
            pred = decode_questions(probabilities, decoder)
            table = rank_report(pred, true, k)[["n", "score", "acc", "within1"]]
            confusion = {bucket: np.zeros((10, 10), dtype=np.int64) for bucket in K_BUCKETS}
            for bucket in K_BUCKETS:
                np.add.at(confusion[bucket], (true[k == bucket], pred[k == bucket]), 1)
            reports[aggregation, decoder] = {
                "table": table, "confusion": confusion,
                "equal_k": float(table.loc[[str(b) for b in K_BUCKETS], "score"].mean())}
    return reports


def verdict(unseen: float, seen: float) -> dict:
    gap = seen - unseen
    gap_rule = ("small (<= 0.03): keep split, report U" if gap <= GAP_LIMITS[0] else
                "moderate (<= 0.08): judge everything on U" if gap <= GAP_LIMITS[1] else
                "large (> 0.08): add anti-memorisation measures")
    return {"equal_k_unseen": unseen, "equal_k_seen": seen, "gap": gap, "gap_rule": gap_rule,
            "passes_baseline": unseen >= BASELINE, "submission_candidate": unseen >= CANDIDATE}


def evaluate(run_dir: Path, dataset_dir: Path | None = None, *, plots: bool = True) -> dict:
    run = load_run(run_dir, dataset_dir)
    result = {}
    for name in SETS:
        result[name] = {"game": game_report(run[name], run["games"]),
                        "questions": question_report(run[name], run[f"{name}_questions"])}
    official = [result[name]["questions"][OFFICIAL]["equal_k"] for name in ("unseen", "seen")]
    result["verdict"] = verdict(*official)
    print_report(run, result)
    if plots:
        save_plots(run, result, run_dir / "plot")
    return result


def print_report(run: dict, result: dict) -> None:
    payload = run["payload"]
    print(f"Strength V2 seed={payload['seed']} steps={payload['optimizer_steps']} "
          f"overlap={json.dumps(payload['player_overlap'])}")
    for name in SETS:
        game = result[name]["game"]
        print(f"\n== {LABELS[name]}: single game (confidence-sum argmax; diagnostic) ==")
        print(f"games={game['games']} score={game['score']:.4f} acc={game['accuracy']:.4f} "
              f"off1={game['off_by_one']:.4f} off2+={game['off_by_two_plus']:.4f}")
        print(game["per_rank"].to_string(index=False, float_format="%.4f"))
        print(game["by_color"].to_string(index=False, float_format="%.4f"))
        for (aggregation, decoder), report in result[name]["questions"].items():
            tag = "OFFICIAL" if (aggregation, decoder) == OFFICIAL else "diagnostic"
            print(f"\n-- {LABELS[name]} questions: {aggregation} + {decoder} [{tag}] "
                  f"Equal-k={report['equal_k']:.4f}")
            print(report["table"].to_string(float_format="%.4f"))
    print("\n== verdict (official rule) ==")
    print(json.dumps(result["verdict"], indent=1))


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _style(axis):
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
    axis.tick_params(colors=MUTED)


def _heatmap(axis, matrix, title, fmt):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=None if fmt == "d" else 1)
    axis.set_xticks(range(10), RANKS)
    axis.set_yticks(range(10), RANKS)
    axis.set_xlabel("predicted rank")
    axis.set_ylabel("true rank")
    axis.set_title(title, fontsize=10)
    # Ink follows the drawn colour: counts scale to their maximum, fractions to vmax=1.
    limit = (np.nanmax(matrix) if np.isfinite(matrix).any() else 1) if fmt == "d" else 1
    for (row, column), value in np.ndenumerate(matrix):
        text = "N/A" if np.isnan(value) else (f"{int(value)}" if fmt == "d" else f"{value:.2f}")
        axis.text(column, row, text, ha="center", va="center", fontsize=6,
                  color="white" if np.isfinite(value) and value > 0.55 * limit else INK)
    return image


def _row_normalised(confusion):
    totals = confusion.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, confusion / totals, np.nan)


def save_plots(run: dict, result: dict, directory: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    seed, saved = run["payload"]["seed"], []

    def publish(figure, name):
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
        plt.close(figure)
        saved.append(path)

    losses = run["losses"]
    sets = run["payload"]["monitor_sets"]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    top.plot(losses["step"], losses["train_loss"], color=COLORS["train"], linewidth=2, marker="o",
             markersize=4, label="training: latest batch (D4, train mode)")
    for name in ("unseen", "seen"):
        info = sets[f"monitor_{name}"]
        top.plot(losses["step"], losses[f"{name}_loss"], color=COLORS[name], linewidth=2, marker="o",
                 markersize=4, label=f"{LABELS[name]} monitor: {info['num_games']:,} games, "
                                     f"{info['num_samples']:,} moves")
        bottom.plot(losses["step"], losses[f"{name}_game_score"], color=COLORS[name], linewidth=2,
                    marker="o", markersize=4, label=f"{LABELS[name]} monitor")
    top.set_ylabel("cross entropy (per move)")
    top.set_title(f"V2 seed {seed}: loss by optimizer step (raw points, final step {losses['step'][-1]})",
                  fontsize=10)
    bottom.set_ylabel("single-game official score")
    bottom.set_ylim(0, 1)
    bottom.set_xlabel("optimizer step")
    bottom.set_title("monitor single-game score (confidence-sum argmax, game-equal; not a question score)",
                     fontsize=10)
    for axis in (top, bottom):
        _style(axis)
        axis.legend(fontsize=8, frameon=False)
    publish(figure, "training_curves")

    for name in SETS:
        game = result[name]["game"]
        figure, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        _heatmap(axes[0], game["confusion"].astype(float), f"{LABELS[name]}: game counts", "d")
        _heatmap(axes[1], _row_normalised(game["confusion"]), f"{LABELS[name]}: row-normalised", "f")
        figure.suptitle(f"V2 seed {seed}: single-game confusion, {game['games']:,} games", fontsize=11)
        publish(figure, f"game_confusion_{name}")

        category = run[name]
        probabilities = game_probabilities(category)
        true = category["rank_id"].astype(int)
        mean = np.vstack([probabilities[true == r].mean(axis=0) if (true == r).any() else np.full(10, np.nan)
                          for r in range(10)])
        figure, axis = plt.subplots(figsize=(6.4, 5.6))
        _heatmap(axis, mean, f"{LABELS[name]}: mean per-game probability by true rank\n"
                             "(each game equal weight; not a calibration claim)", "f")
        axis.set_xlabel("predicted class")
        publish(figure, f"game_confidence_{name}")

    figure, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    width, x = 0.38, np.arange(10)
    for offset, name in ((-width / 2, "unseen"), (width / 2, "seen")):
        table = result[name]["game"]["per_rank"]
        for axis, column in zip(axes, ("score", "accuracy")):
            axis.bar(x + offset, table[column], width - 0.03, color=COLORS[name], label=LABELS[name])
    for axis, column in zip(axes, ("single-game official score", "single-game accuracy")):
        axis.set_ylabel(column)
        axis.set_ylim(0, 1)
        _style(axis)
    counts = result["unseen"]["game"]["per_rank"]["games"]
    axes[1].set_xticks(x, [f"{rank}\nU n={n:,}" for rank, n in zip(RANKS, counts)], fontsize=7)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_title(f"V2 seed {seed}: single-game results by true rank", fontsize=10)
    publish(figure, "game_by_rank")

    groups = [*map(str, K_BUCKETS), "Equal-k"]
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True)
    width, x = 0.2, np.arange(len(groups))
    series = [(name, decoder) for name in ("unseen", "seen") for decoder in DECODERS]
    for number, (name, decoder) in enumerate(series):
        report = result[name]["questions"]["mean_probability", decoder]
        table = report["table"]
        scores = [*table.loc[[str(b) for b in K_BUCKETS], "score"], report["equal_k"]]
        accuracy = [*table.loc[[str(b) for b in K_BUCKETS], "acc"], np.nan]
        style = dict(color=COLORS[name], hatch=None if decoder == "score_rank" else "////",
                     edgecolor="#fcfcfb", alpha=1.0 if decoder == "score_rank" else 0.55)
        label = f"{LABELS[name]}, {decoder}" + (" (official)" if decoder == "score_rank" else " (diagnostic)")
        offset = (number - 1.5) * width
        axes[0].bar(x + offset, scores, width - 0.02, label=label, **style)
        axes[1].bar(x + offset, accuracy, width - 0.02, **style)
    axes[0].axhline(BASELINE, color=MUTED, linewidth=1, linestyle="--")
    axes[0].text(len(groups) - 0.5, BASELINE + 0.01, f"official baseline {BASELINE}", ha="right",
                 fontsize=7, color=MUTED)
    n = result["unseen"]["questions"][OFFICIAL]["table"]["n"]
    axes[1].set_xticks(x, [f"k={g}\nn={n[g]:,}" if g != "Equal-k" else "Equal-k\n(4 buckets)" for g in groups],
                       fontsize=8)
    axes[0].set_ylabel("question official score")
    axes[1].set_ylabel("question accuracy")
    for axis in axes:
        axis.set_ylim(0, 1)
        _style(axis)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[0].set_title(f"V2 seed {seed}: questions (mean per-game probability, each game 1/k)", fontsize=10)
    publish(figure, "question_by_k")

    for decoder in DECODERS:
        report = result["unseen"]["questions"]["mean_probability", decoder]
        figure, axes = plt.subplots(2, 2, figsize=(11, 10))
        for axis, bucket in zip(axes.ravel(), K_BUCKETS):
            _heatmap(axis, _row_normalised(report["confusion"][bucket]), f"k={bucket}", "f")
        tag = "official" if decoder == "score_rank" else "diagnostic"
        figure.suptitle(f"V2 seed {seed}: {LABELS['unseen']} question confusion, {decoder} ({tag}), "
                        "row-normalised", fontsize=11)
        publish(figure, f"question_confusion_{decoder}")

    figure, axis = plt.subplots(figsize=(9, 4.2))
    series = [(name, aggregation) for name in ("unseen", "seen") for aggregation in AGGREGATIONS]
    for number, (name, aggregation) in enumerate(series):
        report = result[name]["questions"][aggregation, "score_rank"]
        scores = [*report["table"].loc[[str(b) for b in K_BUCKETS], "score"], report["equal_k"]]
        official = aggregation == "mean_probability"
        axis.bar(np.arange(len(groups)) + (number - 1.5) * 0.2, scores, 0.18, color=COLORS[name],
                 hatch=None if official else "\\\\\\\\", edgecolor="#fcfcfb", alpha=1.0 if official else 0.55,
                 label=f"{LABELS[name]}, {aggregation}" + (" (official)" if official else " (diagnostic)"))
    axis.set_xticks(np.arange(len(groups)), groups)
    axis.set_ylim(0, 1)
    axis.set_ylabel("question official score (score_rank)")
    axis.set_title(f"V2 seed {seed}: aggregation comparison, probability mean vs log-probability mean",
                   fontsize=10)
    _style(axis)
    axis.legend(fontsize=7, frameon=False, ncol=2)
    publish(figure, "question_aggregation")
    return saved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, help="override the dataset directory saved in model.pt")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    evaluate(args.run_dir, args.dataset_dir, plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
