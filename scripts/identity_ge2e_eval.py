"""Score the GE2E insurance encoder on the synthetic pools (CPU, saved embeddings).

Per split: game vectors are L2-normalised; a candidate is the normalised mean
of its 100 gallery games, a question the normalised mean of its k games;
score = w * cosine (w = the trained GE2E scale, a log-likelihood-like scale).
The encoder only sees black moves from ply 20 on, so "insurance alone" is the
pre-registered "first 20 plies removed" condition.

Fusion: opening pp+slot (A-029, D4) + lambda * insurance. lambda is chosen on
split 0 only (official score, plain top-5); splits 1-2 are only reported.
Gate (D-002): insurance alone, mean of splits 1-2 >= 0.2845; fused score with
the opening intact not below the opening alone (plain top-5 and Hungarian).

    python scripts/identity_ge2e_eval.py --run-dir outputs/pid_ge2e_v1_short
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.metrics import pid_report
from go_aicup.tasks.identity import assign, data
from go_aicup.tasks.identity.predict import load_params, opening_scores, train_prefix

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
BASELINE = 0.2845


def normalise(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def insurance_scores(run_dir: Path, split: int, gal, q, w: float) -> np.ndarray:
    saved = np.load(run_dir / f"embeddings_split{split}.npz")
    position = {int(g): i for i, g in enumerate(saved["game_index"])}
    vectors = normalise(saved["vector"].astype(np.float64))

    def pooled(games):
        return normalise(vectors[[position[int(g)] for g in games]].mean(axis=0))

    candidates = np.stack([pooled(g) for g in gal])
    queries = np.stack([pooled(g) for g in q["games"]])
    return w * queries @ candidates.T


def report(scores, candidates, q, hungarian_tau=None) -> pd.DataFrame:
    if hungarian_tau is None:
        ranked = assign.top5(scores)
    else:
        ranked = assign.hungarian_top5(assign.sinkhorn(assign.row_posterior(scores, hungarian_tau)))
    return pid_report(candidates[ranked], q["player_id"].to_numpy(), q["k"].to_numpy())


def row(name, split, table) -> dict:
    return {"variant": name, "split": split, **{k: round(float(table.loc[k, "score"]), 4) for k in table.index}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    w = float(torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)["model"]["w"])
    meta, prefix = train_prefix("d4")
    gallery, questions = data.pid_splits(meta)
    params = load_params("d4")
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    splits = sorted(int(p.stem.removeprefix("embeddings_split")) for p in args.run_dir.glob("embeddings_split*.npz"))
    rows, cache = [], {}
    for split in splits:
        candidates, gal, q = data.split_view(gallery, questions, split)
        population = np.flatnonzero(black & ~np.isin(pid, candidates))
        cache[split] = (insurance_scores(args.run_dir, split, gal, q, w),
                        opening_scores(prefix, gal, list(q["games"]), population, params), candidates, q)
    if 0 not in cache:
        raise SystemExit("split 0 embeddings are needed to choose lambda")
    ins, opn, candidates, q = cache[0]
    curve = {lam: float(report(opn + lam * ins, candidates, q).loc["all", "score"]) for lam in LAMBDAS}
    best = max(LAMBDAS, key=lambda lam: (curve[lam], -lam))
    for split, (ins, opn, candidates, q) in cache.items():
        rows.append(row("insurance alone (ply>=20)", split, report(ins, candidates, q)))
        rows.append(row("opening alone", split, report(opn, candidates, q)))
        rows.append(row(f"opening + {best} x insurance", split, report(opn + best * ins, candidates, q)))
        rows.append(row("opening + Hungarian", split, report(opn, candidates, q, params["tau"])))
        rows.append(row(f"fused + Hungarian", split, report(opn + best * ins, candidates, q, params["tau"])))
    table = pd.DataFrame(rows)
    print(f"w={w:.3f}; lambda curve on split 0 (plain top-5, all): {json.dumps(curve)} -> lambda={best}")
    print(table.to_string(index=False))
    reported = table[table["split"].isin([1, 2])]
    summary = {"w": w, "lambda_curve_split0": curve, "lambda": best, "splits": splits}
    if len(reported):
        mean = reported.groupby("variant")[["5", "10", "15", "20", "all"]].mean()
        print("\nmean of splits 1-2\n", mean.round(4).to_string())
        ins_all = float(mean.loc["insurance alone (ply>=20)", "all"])
        summary["gate"] = {
            "insurance_alone_all": ins_all, "passes_baseline": ins_all >= BASELINE,
            "fused_minus_opening_plain": float(mean.loc[f"opening + {best} x insurance", "all"] - mean.loc["opening alone", "all"]),
            "fused_minus_opening_hungarian": float(mean.loc["fused + Hungarian", "all"] - mean.loc["opening + Hungarian", "all"])}
        print(json.dumps(summary["gate"], indent=1))
    table.to_csv(args.run_dir / "eval.csv", index=False)
    (args.run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
