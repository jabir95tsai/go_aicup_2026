"""Raw-orientation opening for an official added-set prediction directory (2026-11-15, CPU).

Reads a predict2 output directory, rebuilds its D4 opening term and checks that it and the
fused scores are reproduced, then replaces the opening term by the same model on raw
coordinates (training population, candidate galleries and questions all unrotated). Style
term, HD-and-colour term and fusion weights are the saved ones. Writes predictions.parquet
for `pid_submission merge`; the source directory and the pipeline are not changed.

    python -m scripts.identity.pid2_raw_export --src outputs/pid/day0_1112/public2_d4_9600 \\
        --out outputs/pid/day0_1112/public2_d4_9600_raw --store-dir outputs/pid/day0_1112/store \\
        --pool pool2_public --questions pid2_public
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.tasks.identity import assign, opening, twocolor
from go_aicup.tasks.identity.predict import PLIES, store_prefix, train_prefix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--questions", required=True)
    args = parser.parse_args()

    summary = json.loads((args.src / "info.json").read_text())["summary"]
    if summary.get("use_per_k", False):
        raise ValueError("per-k lambda is not supported here")
    params = summary["opening"]
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    pool_moves, pool_colors, pool_meta = load_store(args.pool, args.store_dir, mmap=False)
    q_moves, q_colors, q_meta = load_store(args.questions, args.store_dir, mmap=False)
    is_white = lambda meta: (meta["color"].astype(str).to_numpy() == "W").astype(np.int64)

    pool_players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(pool_players)))
    assert np.array_equal(candidates, np.load(args.src / "candidates.npy", allow_pickle=True))
    question_ids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[question_ids].to_numpy()

    terms = {}
    for orient in ("d4", "raw"):
        train_meta, train_pre = train_prefix(orient)
        pool_pre, pool_ok = store_prefix(pool_moves, pool_colors, pool_meta, orient, min_plies)
        q_pre, q_ok = store_prefix(q_moves, q_colors, q_meta, orient, min_plies)
        assert pool_ok.all() and q_ok.all(), "a game is too short or does not alternate"
        train_color = train_meta["player_color"].to_numpy().astype(np.int64)
        populations = (np.flatnonzero(train_color == 0), np.flatnonzero(train_color == 1))
        n_train, n_pool = len(train_meta), len(pool_meta)
        prefix = np.concatenate([train_pre[:, :PLIES], pool_pre, q_pre])
        color = np.concatenate([train_color, is_white(pool_meta), is_white(q_meta)])
        gallery = [n_train + np.flatnonzero(pool_players == c) for c in candidates]
        query = [n_train + n_pool + np.sort(rows_by_q[q]) for q in question_ids]
        terms[orient] = twocolor.opening_scores(prefix, opening.prefix_hashes(prefix), gallery, query, color,
                                                populations, params)
        del prefix

    parts = np.load(args.src / "parts.npz")
    assert np.allclose(terms["d4"], parts["opening"]), "D4 opening term not reproduced"
    rest = summary["lambda"] * parts["style"] + summary["mu"] * parts["category"]
    base = terms["d4"] + rest
    assert np.allclose(base, np.load(args.src / "scores.npy")), "fused scores not reproduced"
    old = pd.read_parquet(args.src / "predictions.parquet")
    assert list(old["question_id"]) == question_ids
    assert (old["top1"].to_numpy() == candidates[np.argmax(base, axis=1)]).all()

    scores = terms["raw"] + rest
    ranked = assign.top5(scores)
    table = pd.DataFrame({"question_id": question_ids, "num_games": ks})
    for j in range(5):
        table[f"top{j + 1}"] = candidates[ranked[:, j]]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "predictions.parquet", index=False)
    np.save(args.out / "scores.npy", scores)
    np.save(args.out / "candidates.npy", candidates)
    np.savez(args.out / "parts.npz", opening=terms["raw"], style=parts["style"], category=parts["category"])
    changed = table["top1"].to_numpy() != old["top1"].to_numpy()
    top5_same = (table[[f"top{j}" for j in range(1, 6)]].to_numpy()
                 == old[[f"top{j}" for j in range(1, 6)]].to_numpy()).all(axis=1)
    report = {"source": str(args.src), "opening_orient": "raw", "questions": len(question_ids),
              "candidates": len(candidates), "top1_changed": int(changed.sum()),
              "top1_changed_by_k": {str(k): int(changed[ks == k].sum()) for k in sorted(set(ks))},
              "top5_rows_changed": int((~top5_same).sum()),
              "distinct_top1": int(table["top1"].nunique()), "summary": summary}
    (args.out / "info.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "summary"}))


if __name__ == "__main__":
    main()
