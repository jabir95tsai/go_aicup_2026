"""One-shot conversion: raw CSV (SGF strings) -> flat int16 move store.

Every downstream job reads the store this script writes; nothing else parses
SGF during training. Layout of a store `<name>` under `data/processed/`:

    <name>_moves.npy    int16[N_MOVES]  point index 0..360
    <name>_colors.npy   int8[N_MOVES]   0 = black, 1 = white (who played it)
    <name>_meta.parquet one row per game, row i <-> game i; columns include
                        `offset`, `length` (slice into the flat arrays)

Stores:
    train          the 10 training CSVs (unique key = (rank, game_id); RESULTS A-001 A3)
    pool_public    competition-1 candidate pool (when released)
    pool_private
    pid_public     competition-1 questions, melted wide -> one row per game
    pid_private
    rank_public    competition-2 questions, melted wide -> one row per game
    rank_private

The training conversion is sharded per rank file and resumable: a shard is
considered done once its meta parquet exists (written last, via rename).

Usage:
    python -m go_aicup.data.preprocess train              # convert + combine + verify
    python -m go_aicup.data.preprocess train --verify-only
    python -m go_aicup.data.preprocess release            # pools / questions present on disk
    python -m go_aicup.data.preprocess release --raw-dir outputs/rehearsal/raw --out-dir outputs/rehearsal/store
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from go_aicup.data import sgf as sgflib
from go_aicup.paths import ROOT

TRAIN_DIR = ROOT / "data" / "training"
RAW_DIRS = [ROOT / "data" / "raw", ROOT / "data", ROOT / "data" / "test"]
OUT_DIR = ROOT / "data" / "processed"
SHARD_DIR = OUT_DIR / "shards"

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
RANK_TO_ID = {rank: i for i, rank in enumerate(RANKS)}
MAX_SLOTS = 20

RELEASE_FILES = {
    "pool_public": "player_identification_candidates_public.csv",
    "pool_private": "player_identification_candidates_private.csv",
    "pid_public": "player_identification_test_public.csv",
    "pid_private": "player_identification_test_private.csv",
    "rank_public": "rank_prediction_test_public.csv",
    "rank_private": "rank_prediction_test_private.csv",
}


# --------------------------------------------------------------------------
# parsing (runs in worker processes)
# --------------------------------------------------------------------------

def sgf_hash(content: str) -> int:
    """Stable 63-bit content hash, used for exact-duplicate audits across stores."""
    digest = hashlib.blake2b(content.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def _parse_batch(contents: list[str], lenient: bool = False):
    moves, colors = [], []
    lengths = np.zeros(len(contents), dtype=np.int32)
    n_errors = np.zeros(len(contents), dtype=np.int16)
    alternating = np.zeros(len(contents), dtype=bool)
    first_color = np.full(len(contents), -1, dtype=np.int8)
    hashes = np.zeros(len(contents), dtype=np.int64)
    passes = np.zeros(len(contents), dtype=np.int16)
    handicap = np.full(len(contents), -1, dtype=np.int8)   # HD[n] value; -1 when the SGF has none
    props = []
    for i, content in enumerate(contents):
        game = sgflib.parse_lenient(content) if lenient else sgflib.parse(content)
        if not game.ok and not lenient:
            raise ValueError(f"batch game {i}: {'; '.join(game.errors)}")
        passes[i] = int(game.props.get("_passes", ["0"])[0])
        props.append(",".join(sorted(k for k in game.props if not k.startswith("_"))))
        if game.props.get("HD"):
            handicap[i] = int(game.props["HD"][0])
        moves.append(game.moves)
        colors.append(game.colors)
        lengths[i] = game.n_moves
        n_errors[i] = len(game.errors)
        if game.n_moves:
            first_color[i] = game.colors[0]
            expected = (np.arange(game.n_moves) + game.colors[0]) % 2
            alternating[i] = bool(np.array_equal(game.colors, expected))
        hashes[i] = sgf_hash(content) if isinstance(content, str) else 0
    flat_moves = np.concatenate(moves) if moves else np.empty(0, np.int16)
    flat_colors = np.concatenate(colors) if colors else np.empty(0, np.int8)
    return (flat_moves, flat_colors, lengths, n_errors, alternating, first_color, hashes, passes,
            np.asarray(props, dtype=object), handicap)


def parse_many(contents: list[str], workers: int, batch: int = 5_000, lenient: bool = False):
    """Parse a list of SGF strings in parallel, preserving order."""
    batches = [contents[i:i + batch] for i in range(0, len(contents), batch)]
    flags = [lenient] * len(batches)
    if workers <= 1:
        results = [_parse_batch(b, lenient) for b in batches]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_parse_batch, batches, flags))
    parts = list(zip(*results))
    return (
        np.concatenate(parts[0]).astype(np.int16),
        np.concatenate(parts[1]).astype(np.int8),
        np.concatenate(parts[2]),
        np.concatenate(parts[3]),
        np.concatenate(parts[4]),
        np.concatenate(parts[5]),
        np.concatenate(parts[6]),
        np.concatenate(parts[7]),
        np.concatenate(parts[8]),
        np.concatenate(parts[9]),
    )


def _to_store(frame: pd.DataFrame, contents: list[str], workers: int, lenient: bool = False):
    """Parse `contents` (aligned with `frame`) into (moves, colors, meta)."""
    moves, colors, lengths, n_errors, alternating, first_color, hashes, passes, props, handicap = parse_many(
        contents, workers, lenient=lenient
    )
    meta = frame.reset_index(drop=True).copy()
    meta["length"] = lengths.astype(np.int16)
    meta["n_errors"] = n_errors
    meta["alternating"] = alternating
    meta["first_color"] = first_color
    meta["sgf_hash"] = hashes
    meta["n_passes"] = passes
    meta["props"] = props  # SGF property names other than B/W, for the Day-0 audit
    # HD[n] (v1.1, Q&A Q13): HD>=2 games have reversed B/W labels and no handicap stones
    meta["hd"] = handicap
    return moves, colors, meta


def _write_store(name: str, moves, colors, meta: pd.DataFrame, directory: pathlib.Path):
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(meta), dtype=np.int64)
    if len(meta):
        offsets[1:] = np.cumsum(meta["length"].to_numpy(np.int64))[:-1]
    meta = meta.copy()
    meta.insert(0, "offset", offsets)
    assert int(meta["length"].sum()) == moves.shape[0] == colors.shape[0]
    np.save(directory / f"{name}_moves.npy", moves)
    np.save(directory / f"{name}_colors.npy", colors)
    tmp = directory / f"{name}_meta.parquet.tmp"
    meta.to_parquet(tmp, index=False)
    os.replace(tmp, directory / f"{name}_meta.parquet")


def load_store(name: str, directory: pathlib.Path = OUT_DIR, mmap: bool = True):
    directory, mode = pathlib.Path(directory), "r" if mmap else None
    moves = np.load(directory / f"{name}_moves.npy", mmap_mode=mode)
    colors = np.load(directory / f"{name}_colors.npy", mmap_mode=mode)
    meta = pd.read_parquet(directory / f"{name}_meta.parquet")
    return moves, colors, meta


# --------------------------------------------------------------------------
# training set
# --------------------------------------------------------------------------

def convert_train_shard(rank: str, workers: int) -> None:
    done = SHARD_DIR / f"train_{rank}_meta.parquet"
    if done.exists():
        print(f"  {rank}: shard exists, skipping")
        return
    start = time.time()
    path = TRAIN_DIR / f"train_{rank}.csv"
    frame = pd.read_csv(path, dtype={"player_id": str, "game_id": str,
                                     "rank": str, "color": str})
    assert (frame["rank"] == rank).all(), f"{path.name}: rank column mismatch"
    contents = frame.pop("sgf_content").tolist()
    frame["source_row"] = np.arange(len(frame), dtype=np.int32)
    frame["rank_id"] = np.int8(RANK_TO_ID[rank])
    moves, colors, meta = _to_store(frame, contents, workers)
    _write_store(f"train_{rank}", moves, colors, meta, SHARD_DIR)
    print(f"  {rank}: {len(meta):,} games, {moves.shape[0]:,} moves "
          f"({time.time() - start:.0f}s)")


def combine_train() -> None:
    all_moves, all_colors, metas = [], [], []
    for rank in RANKS:
        moves, colors, meta = load_store(f"train_{rank}", SHARD_DIR, mmap=False)
        all_moves.append(moves)
        all_colors.append(colors)
        metas.append(meta.drop(columns=["offset"]))
    meta = pd.concat(metas, ignore_index=True)
    moves = np.concatenate(all_moves)
    colors = np.concatenate(all_colors)

    key = meta["rank"] + "/" + meta["game_id"]
    assert key.is_unique, "(rank, game_id) must be unique"
    for column in ("player_id", "rank", "color", "game_id"):
        meta[column] = meta[column].astype("category")
    _write_store("train", moves, colors, meta, OUT_DIR)
    print(f"  combined: {len(meta):,} games, {moves.shape[0]:,} moves")


def verify_train(n: int = 1_000, seed: int = 0) -> bool:
    """Gate: re-encode sampled games from the store and compare with raw SGF."""
    moves, colors, meta = load_store("train")
    rng = np.random.default_rng(seed)
    sample = meta.iloc[np.sort(rng.choice(len(meta), size=n, replace=False))]
    mismatches = 0
    for rank, group in sample.groupby("rank", observed=True):
        raw = pd.read_csv(TRAIN_DIR / f"train_{rank}.csv",
                          usecols=["game_id", "player_id", "sgf_content"],
                          dtype=str)
        for row in group.itertuples():
            original = raw.iloc[row.source_row]
            assert original["game_id"] == row.game_id
            assert original["player_id"] == row.player_id
            span = slice(row.offset, row.offset + row.length)
            rebuilt = "".join(
                f";{'B' if c == sgflib.BLACK else 'W'}[{sgflib.index_to_point(int(m))}]"
                for m, c in zip(moves[span], colors[span])
            )
            body = original["sgf_content"].strip()
            # everything from the first move node on, e.g. ";B[pd];W[dp]..."
            first = min(i for i in (body.find(";B["), body.find(";W[")) if i >= 0)
            expected = body[first:].rstrip(")")
            if rebuilt != expected:
                mismatches += 1
    print(f"  verify: {n - mismatches}/{n} games round-trip exactly")

    lengths = meta["length"].to_numpy()
    print(f"  games={len(meta):,} moves={moves.shape[0]:,} "
          f"errors={int(meta['n_errors'].sum())} "
          f"non-alternating={int((~meta['alternating']).sum())} "
          f"first_color_white={int((meta['first_color'] == 1).sum())} "
          f"length=[{lengths.min()}, {lengths.max()}] "
          f"dup_hashes={int(meta['sgf_hash'].duplicated().sum())}")
    return mismatches == 0


# --------------------------------------------------------------------------
# released files (pools, questions)
# --------------------------------------------------------------------------

def _find_release(filename: str, raw_dirs=None) -> pathlib.Path | None:
    for directory in raw_dirs or RAW_DIRS:
        path = pathlib.Path(directory) / filename
        if path.exists():
            return path
    return None


def melt_questions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Wide question rows (sgf_1..sgf_20, color_1..color_20) -> one row per game."""
    rows, contents = [], []
    for record in frame.to_dict("records"):
        for slot in range(1, MAX_SLOTS + 1):
            content = record.get(f"sgf_{slot}")
            if not isinstance(content, str) or not content.strip():
                continue
            rows.append({
                "question_id": record["question_id"],
                "num_games": int(record["num_games"]),
                "slot": slot,
                "color": record.get(f"color_{slot}"),
            })
            contents.append(content)
    meta = pd.DataFrame(rows)
    counts = meta.groupby("question_id").size()
    declared = frame.set_index("question_id")["num_games"].astype(int)
    bad = (counts.reindex(declared.index).fillna(0) != declared)
    if bad.any():
        print(f"  ! {int(bad.sum())} questions whose non-empty slots != num_games")
    return meta, contents


def convert_release(workers: int, raw_dirs=None, out_dir: pathlib.Path = OUT_DIR,
                    names=None, overwrite: bool = False) -> dict:
    """Official pools / questions -> GameStores `<name>_*` in `out_dir`.

    Parsing is lenient (sgf.parse_lenient): an unexpected header, pass or
    setup stone is counted in the meta (`n_errors`, `n_passes`) instead of
    aborting, so a submission can still be produced. Returns {name: path}.
    """
    out_dir = pathlib.Path(out_dir)
    written = {}
    for name, filename in RELEASE_FILES.items():
        if names is not None and name not in names:
            continue
        path = _find_release(filename, raw_dirs)
        if path is None:
            print(f"  {name}: {filename} not found, skipping")
            continue
        if (out_dir / f"{name}_meta.parquet").exists() and not overwrite:
            print(f"  {name}: store exists, skipping")
            written[name] = path
            continue
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if name.startswith("pool"):
            contents = frame.pop("sgf_content").tolist()
            meta = frame
        else:
            meta, contents = melt_questions(frame)
        moves, colors, meta = _to_store(meta, contents, workers, lenient=True)
        _write_store(name, moves, colors, meta, out_dir)
        written[name] = path
        lengths = meta["length"].to_numpy()
        print(f"  {name}: {len(meta):,} games from {path.name}, "
              f"errors={int((meta['n_errors'] > 0).sum())} passes={int(meta['n_passes'].sum())} "
              f"non-alternating={int((~meta['alternating']).sum())} "
              f"length=[{lengths.min()}, {lengths.max()}]")
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["train", "release"])
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--raw-dir", type=pathlib.Path, action="append",
                        help="release: directory holding the official CSVs (repeatable)")
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    start = time.time()
    if args.target == "train":
        if not args.verify_only:
            print("[1/3] shards ...")
            for rank in RANKS:
                convert_train_shard(rank, args.workers)
            print("[2/3] combine ...")
            combine_train()
            if not args.keep_shards:
                for path in SHARD_DIR.glob("train_*"):
                    path.unlink()
        print("[3/3] verify ...")
        ok = verify_train()
        print(f"done ({time.time() - start:.0f}s)")
        return 0 if ok else 1

    convert_release(args.workers, args.raw_dir, args.out_dir, overwrite=args.overwrite)
    print(f"done ({time.time() - start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
