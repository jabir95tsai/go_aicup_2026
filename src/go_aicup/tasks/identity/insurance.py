"""Competition-1 GE2E insurance vectors for release stores (RESULTS A-035, A-053).

For a GameStore (candidate pool or question file): build a board cache exactly
as competition 2 does (board_cache.write_store_cache), then embed every game
with the A-035 encoder over its black moves at ply >= MIN_PLY (as
train_ge2e.embed_games; release games use the stored colours, so a dropped
pass cannot misplace a move), bf16 as in training. One GPU.

Output <work>/<store>/ge2e.npz: store_row, vector (mean move embedding), moves.
identity/predict.py --insurance-dir fuses them: opening + lambda * w * cosine
(scripts/identity_ge2e_eval.py; lambda and w from the run's eval_summary.json
and model.pt).

    python -m go_aicup.tasks.identity.insurance --store-dir data/processed --name pool_public \\
        --work-dir outputs/pid_submit/insurance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.data.board_cache import BoardCache, assemble_inputs, write_store_cache
from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity.ge2e import MIN_PLY, IdentityEncoder
from go_aicup.tasks.strength.shared.runtime import autocast_context

RUN = ROOT / "outputs" / "pid_ge2e_v1_short"   # A-035 choice


def load_run(run_dir: Path = RUN) -> tuple[IdentityEncoder, dict, float, float]:
    """(encoder, config, w, lambda)."""
    config = json.loads((run_dir / "config.json").read_text())
    payload = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(payload["model"])
    summary = json.loads((run_dir / "eval_summary.json").read_text())
    return model, config, float(payload["model"]["w"]), float(summary["lambda"])


@torch.no_grad()
def embed_store(store_dir: Path, name: str, work: Path, run_dir: Path = RUN, batch: int = 8192,
                device_name: str = "cuda") -> Path:
    out = work / name / "ge2e.npz"
    if out.exists():
        print(f"insurance: {out} exists, skipping")
        return out
    moves, colors, meta = load_store(name, store_dir, mmap=False)
    cache_dir = work / name / "cache"
    if not (cache_dir / "games.parquet").exists():
        write_store_cache(moves, colors, meta.drop(columns=[c for c in ("rank_id",) if c in meta]), cache_dir)
    games = pd.read_parquet(cache_dir / "games.parquet")
    offsets = meta["offset"].to_numpy()
    sample_game, sample_ply, owner = [], [], []
    for i, row in enumerate(games.itertuples()):
        game_colors = colors[offsets[row.store_row]:offsets[row.store_row] + row.length]
        plies = np.flatnonzero((game_colors == 0) & (np.arange(row.length) >= MIN_PLY))
        sample_game.append(np.full(len(plies), row.game_index)), sample_ply.append(plies)
        owner.append(np.full(len(plies), i))
    sample_game, sample_ply, owner = map(np.concatenate, (sample_game, sample_ply, owner))
    model, config, _, _ = load_run(run_dir)
    device = torch.device(device_name)
    if device.type == "cpu":
        config = {**config, "precision": "fp32"}
    model.to(device).eval()
    cache = BoardCache(cache_dir)
    sums = np.zeros((len(games), config["model"]["embedding"]), np.float64)
    for start in range(0, len(sample_game), batch):
        span = slice(start, start + batch)
        histories, side = cache.history_batch(sample_game[span], sample_ply[span])
        if np.any(side != 0):
            raise ValueError("insurance sample is not a black move")
        with autocast_context(config, device):
            vectors = model(torch.from_numpy(assemble_inputs(histories, side)).to(device, dtype=torch.float32))
        np.add.at(sums, owner[span], vectors.float().cpu().numpy().astype(np.float64))
    counts = np.bincount(owner, minlength=len(games))
    usable = counts > 0
    vector = np.zeros_like(sums, dtype=np.float32)
    vector[usable] = (sums[usable] / counts[usable, None]).astype(np.float32)
    np.savez(out, store_row=games["store_row"].to_numpy()[usable], vector=vector[usable], moves=counts[usable])
    print(f"insurance: {name}: {int(usable.sum())}/{len(meta)} games embedded, {len(sample_game):,} moves")
    return out


def fused_scores(opening: np.ndarray, work: Path, pool_name: str, question_name: str, candidates,
                 pool_players, question_rows: list[np.ndarray], run_dir: Path = RUN) -> tuple[np.ndarray, dict]:
    """opening + lambda * w * cos(question mean, candidate mean); missing vectors contribute 0."""
    _, _, w, lam = load_run(run_dir)

    def vectors(name, n):
        saved = np.load(work / name / "ge2e.npz")
        out = np.full((n, saved["vector"].shape[1]), np.nan)
        out[saved["store_row"]] = saved["vector"]
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    pool = vectors(pool_name, len(pool_players))
    questions = vectors(question_name, max(int(r.max()) for r in question_rows if len(r)) + 1)

    def pooled(rows):
        """Mean of normalised game vectors, normalised (identity_ge2e_eval.insurance_scores)."""
        x = rows[~np.isnan(rows[:, 0])]
        return x.mean(axis=0) / np.linalg.norm(x.mean(axis=0)) if len(x) else None

    cand = [pooled(pool[pool_players == c]) for c in candidates]
    quer = [pooled(questions[r]) for r in question_rows]
    insurance = np.zeros_like(opening)
    for i, q in enumerate(quer):
        if q is None:
            continue
        insurance[i] = [w * float(q @ c) if c is not None else 0.0 for c in cand]
    info = {"insurance_lambda": lam, "insurance_w": w,
            "questions_without_insurance": int(sum(q is None for q in quer)),
            "candidates_without_insurance": int(sum(c is None for c in cand))}
    return opening + lam * insurance, info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--name", required=True, nargs="+", help="store names, e.g. pool_public pid_public")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    for name in args.name:
        embed_store(args.store_dir, name, args.work_dir, args.run_dir, device_name=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
