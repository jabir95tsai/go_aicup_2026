"""Competition-1 HD handling on the rehearsal pools (RESULTS 2026-11-08).

Arms: A-053 behaviour (keep HD>=2, no HD term), drop HD>=2, and keep + HD-histogram
term at several weights; opening + GE2E insurance, no assignment. Scores use the
official top-5 exponential decay, per k. Public = synthetic split 1, private = split 2
(the weight is read off one and checked on the other).

    python scripts/identity_hd_eval.py --rehearsal outputs/rehearsal_hd_20261108
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.tasks.identity.predict import predict

ARMS = {"keep_w0": ("keep", 0.0), "drop_w0": ("drop", 0.0), "keep_w0.5": ("keep", 0.5),
        "keep_w1": ("keep", 1.0), "keep_w2": ("keep", 2.0), "keep_w4": ("keep", 4.0)}


def score(table: pd.DataFrame, answers: pd.DataFrame) -> dict:
    m = table.assign(question_id=table["question_id"].astype(str)).merge(
        answers.assign(question_id=answers["question_id"].astype(str)), on="question_id")
    top = m[[f"top{j}" for j in range(1, 6)]].to_numpy().astype(str)
    hit = top == m["player_id"].astype(str).to_numpy()[:, None]
    rank = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    s = np.where(rank >= 0, np.exp(-np.maximum(rank, 0)), 0.0)
    out = {str(k): round(float(s[m["k"].to_numpy() == k].mean()), 4) for k in sorted(m["k"].unique())}
    return out | {"mean": round(float(s.mean()), 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehearsal", type=Path, required=True)
    args = parser.parse_args()
    r = args.rehearsal
    result = {}
    for part in ("public", "private"):
        answers = pd.read_csv(r / "answers" / f"pid_{part}.csv", dtype={"question_id": str, "player_id": str})
        result[part] = {}
        for name, (mode, w) in ARMS.items():
            table, _, _, info = predict(r / "store", f"pool_{part}", f"pid_{part}", "d4", "none",
                                        insurance_dir=r / "pid" / "insurance", hd2plus=mode, hd_weight=w)
            result[part][name] = score(table, answers) | {"unscored": info["unscored_questions"]}
            print(part, name, json.dumps(result[part][name]), flush=True)
    (r / "hd_eval.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
