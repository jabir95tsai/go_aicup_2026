"""Constant-rank LB probes: recover the Public rank histogram from leaderboard scores.

A file that predicts rank r for every question scores
``(n[r] + (n[r-1] + n[r+1]) / e) / N`` on the Public questions, so the scores of
constant files form a tridiagonal linear system in the Public rank counts ``n``.
Nine files plus ``sum(n) == N`` determine all ten counts.

    python -m scripts.strength.rank_const_probe write
    python -m scripts.strength.rank_const_probe solve A=0.31 1D=0.22 ...
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT as REPO_ROOT
from go_aicup.tasks.strength.submission_files import write_rank_csv

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
SAMPLE = REPO_ROOT / "data/raw/tests/sample_submission_rank.csv"
PROBE_DIR = REPO_ROOT / "submissions/rank/probe"
N_PUBLIC = 400


def write(_: argparse.Namespace) -> None:
    ids = pd.read_csv(SAMPLE)["question_id"]
    for rank in RANKS:
        frame = pd.DataFrame({"question_id": ids, "pred_rank": rank})
        write_rank_csv(frame, tag=rank.lower(), job="const", directory=PROBE_DIR)


def system(observed: dict[str, float], total: int) -> tuple[np.ndarray, np.ndarray]:
    rows, rhs = [], []
    for rank, score in observed.items():
        r = RANKS.index(rank)
        row = np.zeros(len(RANKS))
        row[r] = 1.0
        for neighbour in (r - 1, r + 1):
            if 0 <= neighbour < len(RANKS):
                row[neighbour] = 1.0 / math.e
        rows.append(row)
        rhs.append(score * total)
    rows.append(np.ones(len(RANKS)))
    rhs.append(float(total))
    return np.array(rows), np.array(rhs)


def solve(args: argparse.Namespace) -> None:
    observed = {}
    for item in args.scores:
        rank, value = item.split("=")
        if rank not in RANKS:
            raise SystemExit(f"unknown rank {rank!r}")
        observed[rank] = float(value)
    matrix, rhs = system(observed, args.total)
    counts, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    rounded = np.rint(counts).astype(int)
    report = {
        "observed": observed,
        "rank_of_system": int(np.linalg.matrix_rank(matrix)),
        "counts_real": dict(zip(RANKS, np.round(counts, 3).tolist())),
        "counts": dict(zip(RANKS, rounded.tolist())),
        "sum": int(rounded.sum()),
        "max_rounding_gap": float(np.abs(counts - rounded).max()),
        "max_score_residual": float(np.abs(matrix[:-1] @ rounded - rhs[:-1]).max() / args.total),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("write").set_defaults(func=write)
    p = sub.add_parser("solve")
    p.add_argument("scores", nargs="+", help="RANK=public_score, e.g. A=0.3123456")
    p.add_argument("--total", type=int, default=N_PUBLIC)
    p.add_argument("--output", type=lambda s: REPO_ROOT / s, default=None)
    p.set_defaults(func=solve)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
