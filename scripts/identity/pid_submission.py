"""Competition-1 submission assembly (2026-11-11 format) and the black-only floor for the added set.

    python -m scripts.identity.pid_submission black-stores --store-dir S --out-dir S_black
    python -m scripts.identity.pid_submission merge --parts pub=DIR priv=DIR pub2=DIR priv2=DIR --out FILE

`black-stores` writes pool2_* / pid2_* stores holding only the games the player played as
Black, so the black-move models (opening, GE2E) can score the added set as a floor; questions
without a black game are left out and `merge` fills them with the first five candidates.
`merge` checks the result against sample_submission_player_v2.csv: same 1,600 question ids,
five distinct candidates per row, every id carrying the question's prefix and present in the pool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import _write_store, load_store
from go_aicup.paths import ROOT

SAMPLE = ROOT / "data/raw/tests/sample_submission_player_v2.csv"
POOLS = {"pub": "pool_public", "priv": "pool_private", "pub2": "pool2_public", "priv2": "pool2_private"}
COLUMNS = [f"top_{j}" for j in range(1, 6)]


def black_stores(args) -> None:
    for name in ("pool2_public", "pool2_private", "pid2_public", "pid2_private"):
        moves, colors, meta = load_store(name, args.store_dir, mmap=False)
        keep = np.flatnonzero(meta["color"].astype(str).to_numpy() == "B")
        spans = [np.arange(o, o + n) for o, n in zip(meta["offset"].to_numpy()[keep], meta["length"].to_numpy()[keep])]
        index = np.concatenate(spans)
        _write_store(name, moves[index], colors[index], meta.iloc[keep].drop(columns="offset"), args.out_dir)
        print(f"{name}: {len(keep)}/{len(meta)} black games")


def merge(args) -> None:
    sample = pd.read_csv(SAMPLE, dtype=str)
    frames = []
    for item in args.parts:
        prefix, directory = item.split("=")
        _, _, pool = load_store(POOLS[prefix], args.store_dir)
        candidates = np.array(sorted(set(pool["player_id"].astype(str))))
        table = pd.read_parquet(Path(directory) / "predictions.parquet")
        frame = table[["question_id"]].assign(**{f"top_{j}": table[f"top{j}"] for j in range(1, 6)})
        wanted = sample.loc[sample["question_id"].str.startswith(prefix + "_q_"), "question_id"]
        missing = wanted[~wanted.isin(frame["question_id"])]
        filler = pd.DataFrame({"question_id": missing, **{c: candidates[j] for j, c in enumerate(COLUMNS)}})
        frame = pd.concat([frame, filler], ignore_index=True)
        assert frame[COLUMNS].isin(candidates).all().all(), prefix
        assert frame[COLUMNS].apply(lambda col: col.str.startswith(prefix + "_p_")).all().all(), prefix
        print(f"{prefix}: {len(table)} predicted, {len(missing)} filled")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = out.set_index("question_id").loc[sample["question_id"]].reset_index()
    assert list(out.columns) == list(sample.columns) and len(out) == len(sample) == 1600
    assert out["question_id"].is_unique and (out[COLUMNS].nunique(axis=1) == 5).all()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("black-stores")
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=black_stores)
    p = sub.add_parser("merge")
    p.add_argument("--store-dir", type=Path, required=True)
    p.add_argument("--parts", nargs=4, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=merge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
