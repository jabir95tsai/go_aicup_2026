"""Paired comparison of two style encoders on the added-set bench (cv.draw_pid2).

Each draw scores both runs' fused models (opening + lambda * w * style + mu * HD-and-colour,
parameters from each run's pid2_bench summary) on the same candidates and questions, so the
per-draw difference cancels the draw-to-draw spread. Draw seeds start at --first (the bench
itself used 0-4).

    python -m scripts.identity.pid2_paired --pool 1 --draws 30 \\
        --runs pid_ge2e_v2_both_p0_600 pid_ge2e_v2_both_p0_1200
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.metrics import PID_WEIGHTS
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import assign, opening, twocolor
from go_aicup.tasks.identity.predict import train_prefix

BENCH = ROOT / "outputs" / "pid" / "pid2_bench"
BANK = PROCESSED / "identity" / "pid2_bank.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
OUT = ROOT / "outputs" / "pid" / "pid2_paired"


def points(scores, candidates, q) -> np.ndarray:
    ranked = candidates[assign.top5(scores)]
    return ((ranked == q["player_id"].to_numpy()[:, None]) * PID_WEIGHTS).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=int, required=True)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--first", type=int, default=100)
    parser.add_argument("--runs", nargs=2, required=True)
    parser.add_argument("--fixed-gallery", type=int, default=None, metavar="SEED",
                        help="score on cv.draw_pid2_fixed draws (the gallery an encoder was adapted to)")
    args = parser.parse_args()

    meta, prefix = train_prefix("d4")
    bank = pd.read_parquet(BANK)
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy().astype(np.int64)
    color = meta["player_color"].to_numpy().astype(np.int64)
    pid = meta["player_id"].astype(str).to_numpy()
    hashes = opening.prefix_hashes(prefix)
    codes = np.minimum(handicap, 1) + 2 * color
    outside = ~np.isin(pid, bank["player_id"].unique())
    populations = (np.flatnonzero(outside & (color == 0)), np.flatnonzero(outside & (color == 1)))
    runs = []
    for name in args.runs:
        summary = json.loads((BENCH / name / "summary.json").read_text())
        saved = np.load(BENCH / name / "embeddings.npz")
        position = np.full(len(meta), -1)
        position[saved["game_index"]] = np.arange(len(saved["game_index"]))
        runs.append((summary, saved["swap"], position))
    if runs[0][0]["opening"] != runs[1][0]["opening"]:
        raise ValueError("runs chose different opening parameters; score them separately")

    rows = []
    for draw in range(args.first, args.first + args.draws):
        if args.fixed_gallery is None:
            candidates, gallery, q = cv.draw_pid2(bank, args.pool, draw)
        else:
            candidates, gallery, q = cv.draw_pid2_fixed(bank, args.pool, args.fixed_gallery, draw)
        query = list(q["games"])
        op = twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, runs[0][0]["opening"])
        cat = twocolor.category_scores(codes, gallery, query, 4)
        ks = q["k"].to_numpy()
        for i, (summary, vectors, position) in enumerate(runs):
            style = summary["w"] * twocolor.style_scores(vectors, [position[g] for g in gallery],
                                                         [position[g] for g in query])
            pts = points(op + summary["lambda"] * style + summary["mu"] * cat, candidates, q)
            rows.append({"draw": draw, "run": i, **{str(k): pts[ks == k].mean() for k in cv.PID2_K},
                         "all": pts.mean()})
        print(f"draw {draw}: {rows[-2]['all']:.4f} -> {rows[-1]['all']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "" if args.fixed_gallery is None else f"_gallery{args.fixed_gallery}"
    frame.to_csv(OUT / f"{args.runs[0]}__{args.runs[1]}_pool{args.pool}{tag}.csv", index=False)
    cols = [str(k) for k in cv.PID2_K] + ["all"]
    a = frame[frame["run"] == 0].set_index("draw")[cols]
    b = frame[frame["run"] == 1].set_index("draw")[cols]
    diff = b - a
    print(f"pool {args.pool}, {args.draws} draws; mean {args.runs[0]}:\n{a.mean().round(4).to_string()}")
    print(f"mean {args.runs[1]}:\n{b.mean().round(4).to_string()}")
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    print("paired diff (mean, se, share of draws > 0):")
    print(pd.DataFrame({"mean": diff.mean(), "se": se, "pos": (diff > 0).mean()}).round(4).to_string())


if __name__ == "__main__":
    main()
