"""Day-0 audit of the official release (EXECUTION_PLAN section 5), on GameStores.

Items covered (numbers as in the plan):
  1  SGF properties, black first, alternation, passes, short games, colours
  2  exact duplicate games (by move sequence) test vs pool / train / each other
  3  player_id overlap: train, public pool, private pool
  4  num_games distribution per question file
  5  orientation: first black move distribution, questions vs pool (raw and D4)
  6  opening concentration: per-candidate most-common first move share,
     versus the synthetic pools (split 0)
  7  one-to-one diagnostic from identity scores.npy if present
  9  official CSV headers (and sample submission files if any)
Not covered here: 8 (L10 needs a model run), 10 (judgement from 5-7).

`--train-self-check` runs items 1-3 on the training board cache instead; it
must reproduce RESULTS A-001 / A-002 (lengths 100-445, median 228, p99 323,
0 duplicates, black first, alternating, 8,007 players, none in two ranks).

    python scripts/day0_audit.py --store-dir data/processed --raw-dir data/raw
    python scripts/day0_audit.py --train-self-check
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import RELEASE_FILES, load_store
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import data

CACHE = PROCESSED / "boards" / "strength_v1"
TRAIN_HASH = PROCESSED / "audit" / "train_move_hash.npy"
SHORT = (16, 60, 100)


def move_hash(moves: np.ndarray) -> int:
    digest = hashlib.blake2b(np.ascontiguousarray(moves, dtype=np.int16).tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def store_hashes(moves, meta) -> np.ndarray:
    return np.array([move_hash(moves[o:o + n]) for o, n in zip(meta["offset"], meta["length"])], np.int64)


def train_hashes(meta: pd.DataFrame) -> np.ndarray:
    if TRAIN_HASH.exists():
        return np.load(TRAIN_HASH)
    result = np.empty(len(meta), np.int64)
    offsets, lengths = meta["local_offset"].to_numpy(), meta["length"].to_numpy()
    for shard, rows in meta.groupby("shard_id", sort=True).indices.items():
        moves = np.load(CACHE / f"shard_{shard:05d}_moves.npy")
        for row in rows:
            result[row] = move_hash(moves[offsets[row]:offsets[row] + lengths[row]])
    TRAIN_HASH.parent.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_HASH, result)
    return result


def format_report(name: str, colors, meta: pd.DataFrame) -> dict:
    lengths = meta["length"].to_numpy()
    first = np.array([colors[o] if n else -1 for o, n in zip(meta["offset"], meta["length"])])
    props = meta["props"].str.split(",").explode() if "props" in meta else pd.Series(dtype=str)
    report = {
        "games": len(meta),
        "property_rate": (props[props != ""].value_counts() / len(meta)).round(4).to_dict(),
        "parse_errors": int((meta["n_errors"] > 0).sum()),
        "passes_total": int(meta["n_passes"].sum()) if "n_passes" in meta else None,
        "games_with_pass": int((meta["n_passes"] > 0).sum()) if "n_passes" in meta else None,
        "black_first": float((first == 0).mean()),
        "alternating": float(meta["alternating"].mean()),
        "length": {"min": int(lengths.min()), "p1": float(np.percentile(lengths, 1)),
                   "median": float(np.median(lengths)), "p99": float(np.percentile(lengths, 99)),
                   "max": int(lengths.max())},
        "shorter_than": {str(s): int((lengths < s).sum()) for s in SHORT},
        "player_color": meta["color"].astype(str).value_counts().to_dict(),
    }
    if "question_id" in meta:
        per_q = meta.groupby("question_id").agg(k=("num_games", "first"), n=("slot", "size"),
                                                colors=("color", "nunique"))
        report["num_games"] = per_q["k"].value_counts().sort_index().to_dict()
        report["slots_ne_num_games"] = int((per_q["k"] != per_q["n"]).sum())
        report["mixed_color_questions"] = int((per_q["colors"] > 1).sum())
        report["questions"] = len(per_q)
    print(f"\n[1/4] {name}\n" + json.dumps(report, indent=1, ensure_ascii=False))
    return report


def first_black_move(moves, colors, meta) -> np.ndarray:
    out = np.full(len(meta), -1)
    for i, (o, n) in enumerate(zip(meta["offset"], meta["length"])):
        black = np.flatnonzero(colors[o:o + n] == 0)
        if len(black):
            out[i] = moves[o + black[0]]
    return out


def tv(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.bincount(a[a >= 0], minlength=361) / max(1, (a >= 0).sum())
    pb = np.bincount(b[b >= 0], minlength=361) / max(1, (b >= 0).sum())
    return float(0.5 * np.abs(pa - pb).sum())


def orbit(points: np.ndarray) -> np.ndarray:
    """D4 canonical point (smallest of the 8 images)."""
    tables = data.d4_tables()
    return np.where(points >= 0, tables[:, np.maximum(points, 0)].min(axis=0), -1)


def concentration(first: np.ndarray, players: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"p": players, "m": first})
    return frame.groupby("p")["m"].agg(lambda m: m.value_counts(normalize=True).iloc[0])


def quantiles(series) -> dict:
    return {q: round(float(np.quantile(series, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


# Q&A Q13 star points of the handicap stones (row * 19 + col), as audit_handicap_semantics.py
_STAR = {"pd": 3 * 19 + 15, "dp": 15 * 19 + 3, "dd": 3 * 19 + 3, "pp": 15 * 19 + 15, "jj": 9 * 19 + 9,
         "dj": 9 * 19 + 3, "pj": 9 * 19 + 15, "jd": 3 * 19 + 9, "jp": 15 * 19 + 9}
_STONES = {2: "pd dp", 3: "pd dp dd", 4: "pd dp dd pp", 5: "pd dp dd pp jj", 6: "pd dp dd pp dj pj",
           7: "pd dp dd pp dj pj jj", 8: "pd dp dd pp dj pj jd jp", 9: "pd dp dd pp dj pj jd jp jj"}


def hd_report(name: str, moves, meta: pd.DataFrame) -> dict:
    """HD distribution; for HD>=2, how many games play on a Q13 stone point in the first
    10 plies (0 = stones missing from the SGF and labels reversed, as in training)."""
    if "hd" not in meta:
        return {"hd_column": False}
    hd = meta["hd"].to_numpy()
    report = {"hd_counts": {str(k): int(v) for k, v in pd.Series(hd).value_counts().sort_index().items()}}
    on_stone = 0
    big = np.flatnonzero(hd >= 2)
    for i in big:
        o, n = int(meta["offset"].iloc[i]), int(meta["length"].iloc[i])
        points = {_STAR[p] for p in _STONES[min(int(hd[i]), 9)].split()}
        on_stone += bool(set(np.asarray(moves[o:o + min(n, 10)]).tolist()) & points)
    report["hd2plus_games_with_early_move_on_stone_point"] = int(on_stone)
    report["hd2plus_games"] = int(len(big))
    if "question_id" in meta:
        q = meta.assign(big=hd >= 2).groupby("question_id")["big"].agg(["sum", "size"])
        report["questions_with_hd2plus"] = int((q["sum"] > 0).sum())
        report["questions_all_hd2plus"] = int((q["sum"] == q["size"]).sum())
    print(f"\n[1b] {name} handicap\n" + json.dumps(report, indent=1))
    return report


def audit_release(store_dir: Path, raw_dirs: list[Path], identity_dir: Path | None) -> dict:
    stores = {}
    for name in RELEASE_FILES:
        if (store_dir / f"{name}_meta.parquet").exists():
            stores[name] = load_store(name, store_dir)
    if not stores:
        raise SystemExit(f"no release stores under {store_dir}")
    out = {"format": {name: format_report(name, s[1], s[2]) for name, s in stores.items()},
           "handicap": {name: hd_report(name, s[0], s[2]) for name, s in stores.items()}}
    sizes = {}
    for part in ("public", "private"):
        if f"pool_{part}" in stores and f"pid_{part}" in stores:
            questions = stores[f"pid_{part}"][2]["question_id"].nunique()
            players = stores[f"pool_{part}"][2]["player_id"].nunique()
            sizes[part] = {"questions": int(questions), "candidates": int(players),
                           "one_to_one_possible": bool(questions <= players)}
    print("\n[4b] questions vs candidates\n" + json.dumps(sizes, indent=1))
    out["questions_vs_candidates"] = sizes

    train_meta = data.load_meta()
    train_h = set(train_hashes(train_meta).tolist())
    hashes = {name: store_hashes(s[0], s[2]) for name, s in stores.items()}
    dup = {}
    for name, h in hashes.items():
        row = {"within": int(pd.Series(h).duplicated().sum()), "in_train": int(np.isin(h, list(train_h)).sum())}
        for other, g in hashes.items():
            if other != name:
                row[f"in_{other}"] = int(np.isin(h, g).sum())
        dup[name] = row
    print("\n[2] exact duplicate games (by move sequence)\n" + json.dumps(dup, indent=1))
    out["duplicates"] = dup

    train_ids = set(train_meta["player_id"].astype(str))
    pools = {n: set(s[2]["player_id"].astype(str)) for n, s in stores.items() if n.startswith("pool")}
    overlap = {n: {"players": len(p), "in_train": len(p & train_ids),
                   **{f"in_{m}": len(p & q) for m, q in pools.items() if m != n}} for n, p in pools.items()}
    print("\n[3] player_id overlap\n" + json.dumps(overlap, indent=1))
    out["player_overlap"] = overlap

    orient = {}
    gallery, questions = data.pid_splits(train_meta)
    synthetic = gallery[gallery["split"] == 0]
    train_first = data.opening_prefix(train_meta, 60)[:, 0]
    for part in ("public", "private"):
        if f"pool_{part}" not in stores or f"pid_{part}" not in stores:
            continue
        pool_first = first_black_move(*stores[f"pool_{part}"])
        q_first = first_black_move(*stores[f"pid_{part}"])
        top = pd.Series(pool_first).value_counts(normalize=True).head(5).round(3).to_dict()
        orient[part] = {"tv_raw_query_vs_pool": round(tv(q_first, pool_first), 4),
                        "tv_d4_query_vs_pool": round(tv(orbit(q_first), orbit(pool_first)), 4),
                        "pool_top_first_moves": {str(k): v for k, v in top.items()},
                        "concentration_pool": quantiles(concentration(
                            pool_first, stores[f"pool_{part}"][2]["player_id"].astype(str).to_numpy())),
                        "concentration_synthetic_split0": quantiles(concentration(
                            train_first[synthetic["game"].to_numpy()], synthetic["player_id"].to_numpy()))}
    if orient:
        print("\n[5/6] orientation and opening concentration\n" + json.dumps(orient, indent=1))
    out["orientation"] = orient

    if identity_dir is not None:
        one = {}
        for part in ("public", "private"):
            path = identity_dir / part / "scores.npy"
            if path.exists():
                scores = np.load(path)
                best = scores.argmax(axis=1)
                one[part] = {"questions": scores.shape[0], "candidates": scores.shape[1],
                             "distinct_row_argmax": int(len(np.unique(best))),
                             "max_questions_on_one_candidate": int(np.bincount(best).max())}
        print("\n[7] one-to-one diagnostic (row argmax of opening scores)\n" + json.dumps(one, indent=1))
        out["one_to_one"] = one

    headers = {}
    for directory in raw_dirs:
        for path in sorted(Path(directory).glob("*.csv")):
            headers[path.name] = pd.read_csv(path, nrows=2, dtype=str).columns.tolist()[:8]
    print("\n[9] CSV headers (first 8 columns)\n" + json.dumps(headers, indent=1))
    out["headers"] = headers
    return out


def train_self_check() -> dict:
    meta = data.load_meta()
    lengths = meta["length"].to_numpy()
    first_white = alternating = 0
    offsets = meta["local_offset"].to_numpy()
    for shard, rows in meta.groupby("shard_id", sort=True).indices.items():
        colors = np.load(CACHE / f"shard_{shard:05d}_colors.npy")
        for row in rows:
            start, n = offsets[row], lengths[row]
            c = colors[start:start + n]
            first_white += int(c[0] == 1)
            alternating += int(np.array_equal(c, np.arange(n) % 2))
    hashes = train_hashes(meta)
    report = {"games": len(meta), "length": [int(lengths.min()), float(np.median(lengths)),
                                               float(np.percentile(lengths, 99)), int(lengths.max())],
              "first_white": first_white, "alternating": alternating,
              "duplicate_move_sequences": int(pd.Series(hashes).duplicated().sum()),
              "players": int(meta["player_id"].nunique()),
              "players_in_two_ranks": int((meta.groupby("player_id", observed=True)["rank"].nunique() > 1).sum())}
    print(json.dumps(report, indent=1))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, default=PROCESSED)
    parser.add_argument("--raw-dir", type=Path, action="append", default=None)
    parser.add_argument("--identity-dir", type=Path, help="dir with <part>/scores.npy from identity.predict")
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    parser.add_argument("--train-self-check", action="store_true")
    args = parser.parse_args()
    if args.train_self_check:
        report = train_self_check()
    else:
        report = audit_release(args.store_dir, args.raw_dir or [ROOT / "data" / "raw"], args.identity_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
