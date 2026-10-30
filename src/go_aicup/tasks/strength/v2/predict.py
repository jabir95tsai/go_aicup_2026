"""Competition-2 test pipeline: release GameStore -> V2 inference -> C2 -> score_rank.

Steps (each resumable; outputs under <work-dir>/<name>/):

    prepare    store `<name>` -> board cache (cache/) + sample index of every
               judged-player move (test_game.npy, test_ply.npy). CPU.
    infer      frozen model.pt over every sample, bf16 as in training ->
               category.npy (per-game confidence / log-prob sums). GPU.
    aggregate  official rule (RESULTS D-002): per-game mean log-prob, questions
               average games 1/k, softmax(T_k * .) with T_k from the A-025
               calibration, decoded with metrics.score_rank -> predictions.parquet
               and a provisional submission CSV (format fixed on 11/04).

Questions whose games all failed to parse get the mean probability of the
other questions (count printed). `num_games` outside {5,10,15,20} uses the
nearest calibrated k. No label column of the release file is ever read.

    python -m go_aicup.tasks.strength.v2.predict --store-dir data/processed --name rank_public \\
        --work-dir outputs/rank/submit --run-dir outputs/rank/v2/v2_strength
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS, write_store_cache
from go_aicup.data.preprocess import load_store
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import N_RANKS, decode_questions, tempered_probabilities

DEFAULT_RUN = ROOT / "outputs" / "rank" / "v2" / "v2_strength"
DEFAULT_CALIBRATION = ROOT / "outputs" / "rank" / "v2" / "v2_agg_calibration" / "result.json"
PREFIX = "test"
STEPS = ("prepare", "infer", "aggregate")


def prepare(store_dir: Path, name: str, work: Path) -> None:
    if (work / f"{PREFIX}_ply.npy").exists():
        print(f"prepare: {work} exists, skipping")
        return
    moves, colors, meta = load_store(name, store_dir, mmap=False)
    work.mkdir(parents=True, exist_ok=True)
    meta.drop(columns=[c for c in ("rank", "rank_id") if c in meta]).to_parquet(work / "store_meta.parquet")
    write_store_cache(moves, colors, meta.drop(columns=[c for c in ("rank_id",) if c in meta]), work / "cache")
    games = pd.read_parquet(work / "cache" / "games.parquet")
    sample_game, sample_ply = [], []
    for row in games.itertuples():
        start = int(meta["offset"].iloc[row.store_row])
        own = np.flatnonzero(colors[start:start + row.length] == row.player_color)
        sample_game.append(np.full(len(own), row.game_index, np.int64))
        sample_ply.append(own.astype(np.int64))
    np.save(work / f"{PREFIX}_game.npy", np.concatenate(sample_game))
    np.save(work / f"{PREFIX}_ply.npy", np.concatenate(sample_ply))
    skipped = len(meta) - len(games)
    print(f"prepare: {len(games):,} games ({skipped} without moves), "
          f"{sum(len(s) for s in sample_ply):,} judged-player moves")


def infer(work: Path, run_dir: Path, batch: int, workers: int, device_name: str) -> None:
    import torch
    from go_aicup.tasks.strength.shared.training import configure_torch, seed_process
    from go_aicup.tasks.strength.v2.infer import infer_games
    from go_aicup.tasks.strength.v2.model import StrengthResNet

    out = work / "category.npy"
    if out.exists():
        print(f"infer: {out} exists, skipping")
        return
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "strength_v2_1":
        raise ValueError("not a Strength V2 model.pt")
    config = json.loads(json.dumps(payload["config"]))
    config["paths"].update(dataset_dir=str(work), cache_dir=str(work / "cache"))
    config["evaluation"]["global_batch_size"] = batch
    config["loader"]["num_workers"] = workers
    if device_name == "cpu":
        config.update(device="cpu", precision="fp32")
        config["loader"]["pin_memory"] = False
    configure_torch(config)
    seed_process(config["seed"])
    device = torch.device("cuda", 0) if config["device"] == "cuda" else torch.device("cpu")
    model = StrengthResNet(**config["model"])
    model.load_state_dict(payload["model"])
    model.to(device)
    category = infer_games(model, config, PREFIX, device)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, category, allow_pickle=False)
    tmp.replace(out)
    print(f"infer: {len(category):,} games, {int(category['num_moves'].sum()):,} moves "
          f"({config['device']}, {config['precision']})")


def temperatures(calibration: Path) -> dict[int, float]:
    result = json.loads(calibration.read_text())["candidates"]["C2"]
    if result["method"] != "mean_log_probability":
        raise ValueError("calibration C2 must be mean_log_probability")
    return {int(k): float(t) for k, t in result["temperatures"].items()}


def question_table(work: Path) -> pd.DataFrame:
    """Per question: declared k and the category rows of its usable games."""
    meta = pd.read_parquet(work / "store_meta.parquet")
    games = pd.read_parquet(work / "cache" / "games.parquet")
    meta["cache_game"] = -1
    meta.loc[games["store_row"].to_numpy(), "cache_game"] = games["game_index"].to_numpy()
    rows = []
    for question_id, group in meta.groupby("question_id", sort=False):
        usable = group["cache_game"].to_numpy()
        rows.append({"question_id": question_id, "num_games": int(group["num_games"].iloc[0]),
                     "games_in_file": len(group), "color": ",".join(sorted(set(group["color"].astype(str)))),
                     "games": np.sort(usable[usable >= 0])})
    return pd.DataFrame(rows)


def c2_probabilities(category: np.ndarray, questions: pd.DataFrame, temps: dict[int, float]) -> np.ndarray:
    per_game = category["log_probability_sum"] / category["num_moves"][:, None]
    position = {int(g): i for i, g in enumerate(category["game_index"])}
    probabilities = np.full((len(questions), N_RANKS), np.nan)
    for number, question in enumerate(questions.itertuples()):
        if not len(question.games):
            continue
        logits = per_game[[position[int(g)] for g in question.games]].mean(axis=0)
        k = min(K_BUCKETS, key=lambda bucket: (abs(bucket - question.num_games), bucket))
        probabilities[number] = tempered_probabilities(logits[None], temps[k])[0]
    missing = np.isnan(probabilities[:, 0])
    if missing.any():
        if missing.all():
            raise ValueError("no question has a usable game")
        probabilities[missing] = probabilities[~missing].mean(axis=0)
    return probabilities


def aggregate(work: Path, calibration: Path, submission: Path | None) -> pd.DataFrame:
    category = np.load(work / "category.npy", allow_pickle=False)
    questions = question_table(work)
    probabilities = c2_probabilities(category, questions, temperatures(calibration))
    rank_id = decode_questions(probabilities, "score_rank")
    result = questions.drop(columns=["games"]).assign(
        used_games=[len(g) for g in questions["games"]], rank_id=rank_id,
        rank=[RANKS[r] for r in rank_id])
    for r, rank in enumerate(RANKS):
        result[f"p_{rank}"] = probabilities[:, r]
    result.to_parquet(work / "predictions.parquet", index=False)
    fallback = int((result["used_games"] == 0).sum())
    short = int((result["used_games"] < result["num_games"]).sum())
    print(f"aggregate: {len(result)} questions, fallback={fallback}, missing-some-games={short}, "
          f"predicted ranks={result['rank'].value_counts().reindex(RANKS, fill_value=0).to_dict()}")
    if submission is not None:
        submission.parent.mkdir(parents=True, exist_ok=True)
        # Provisional format; replace once the official sample_submission is out.
        result[["question_id", "rank"]].to_csv(submission, index=False)
        print(f"submission (provisional format): {submission}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, help="store name, e.g. rank_public")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--steps", default=",".join(STEPS))
    parser.add_argument("--batch", type=int, default=8192, help="inference batch (training used 8192 per GPU)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    steps = args.steps.split(",")
    if not set(steps) <= set(STEPS):
        raise ValueError(f"steps must be from {STEPS}")
    work = args.work_dir / args.name
    if "prepare" in steps:
        prepare(args.store_dir, args.name, work)
    if "infer" in steps:
        infer(work, args.run_dir, args.batch, args.workers, args.device)
    if "aggregate" in steps:
        aggregate(work, args.calibration, args.submission)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
