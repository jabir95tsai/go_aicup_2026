"""Strength V1 packed boards, streamed CSV conversion and lazy decompression.

Only an explicit CLI invocation converts data; importing this module does no
I/O. Run (in a CPU allocation on Nano4, not on the login node)::

    python -m go_aicup.data.board_cache convert --workers 8
    python -m go_aicup.data.board_cache verify

Defaults: input data/raw/training, output data/processed/boards/strength_v1.
Each game has variable T (all actual moves, including both sides and passes).
Logical shape is [N_games, T, 2, 19, 19]; storage is sharded uint8[sum(T),2,46].
Plane 0 is black and plane 1 white, row-major, little bit order, zero padding.
Every shard also retains int16 moves, int8 actual colors and game metadata.

DataLoader code uses BoardCache.game(), history() or training_batch(). The
reader never parses SGF and never decompresses the complete dataset. See
go_aicup.tasks.strength.dataset.PackedBoardDataset for a map-style adapter.
No split, training, augmentation or full uncompressed copy is created here.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import csv
import hashlib
import json
import multiprocessing
import operator
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from go_aicup.paths import DATA, PROCESSED

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
DEFAULT_INPUT = DATA / "raw" / "training"
DEFAULT_OUTPUT = PROCESSED / "boards" / "strength_v1"
FORMAT_VERSION = 1
CODEC = "packed"
BITORDER = "little"
POINTS = 361
PLANE_BYTES = 46
NUMERIC_COLUMNS = ["shard_id", "local_offset", "length", "rank_id", "player_color"]


def pack_planes(planes: np.ndarray) -> np.ndarray:
    """Pack binary [...,2,19,19] into contiguous uint8[...,2,46].

    Both uint8 and the existing float32 converter output are accepted, but
    values must be exactly 0/1. Each channel is padded independently.
    """
    planes = np.asarray(planes)
    if planes.ndim < 3 or planes.shape[-3:] != (2, 19, 19):
        raise ValueError("planes must have trailing shape (2, 19, 19)")
    if not np.all((planes == 0) | (planes == 1)):
        raise ValueError("planes must contain only 0 and 1")
    flat = planes.reshape(planes.shape[:-2] + (POINTS,)).astype(np.uint8)
    return np.ascontiguousarray(np.packbits(flat, axis=-1, bitorder=BITORDER))


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    """Decompress [...,2,46] to contiguous uint8[...,2,19,19].

    Exactly 361 bits are restored per channel. Nonzero padding is corruption,
    not extra board points. Suitable for one game, histories or an entire batch.
    """
    packed = np.asarray(packed)
    if packed.dtype != np.uint8 or packed.ndim < 2 or packed.shape[-2:] != (2, 46):
        raise ValueError("packed must be uint8 with trailing shape (2, 46)")
    if np.any(packed[..., -1] & np.uint8(254)):
        raise ValueError("nonzero padding bits in packed boards")
    flat = np.unpackbits(packed, axis=-1, count=POINTS, bitorder=BITORDER)
    return np.ascontiguousarray(flat.reshape(packed.shape[:-1] + (19, 19)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _close_mmap(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _array_check(path: Path, dtype: str, shape: list[int]) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.dtype != np.dtype(dtype) or list(array.shape) != shape:
            raise ValueError(f"wrong dtype/shape: {path}")
        if path.stat().st_size != array.offset + array.nbytes:
            raise ValueError(f"truncated or oversized array: {path}")
    finally:
        _close_mmap(array)


def _shard_check(directory: Path, summary: dict, hashes: bool = False) -> None:
    if summary.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported shard version")
    for filename, record in summary["files"].items():
        path = directory / filename
        if Path(filename).name != filename or not path.is_file():
            raise ValueError(f"missing or unsafe shard file: {filename}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"wrong file size: {path}")
        if "shape" in record:
            _array_check(path, record["dtype"], record["shape"])
        if hashes and _sha256(path) != record["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")


def _worker_init() -> None:
    # Also set in the parent before spawn, so BLAS is limited before import.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    from numba import set_num_threads
    from go_aicup.features.sgf_planes import moves_to_planes
    set_num_threads(1)
    moves_to_planes(np.array([0], np.int16), np.array([0], np.int8))


def _write_shard(task: dict) -> dict:
    """Parse a bounded CSV chunk, then replay each game into worker-owned files.

    Only SGF strings enter the process queue and only this small summary leaves
    it. Parsed moves are compact; float planes exist for just one game at a time.
    """
    from go_aicup.features.sgf_planes import moves_to_planes, parse_plane_game

    start = time.monotonic()
    directory = Path(task["directory"])
    prefix = f"shard_{task['shard_id']:05d}"
    parsed, records = [], []
    offset = 0
    for i, row in enumerate(task["rows"]):
        source_row = task["source_start"] + i
        context = f"rank={task['rank']} game_id={row.get('game_id')} source_row={source_row}"
        try:
            if row["rank"] != task["rank"] or row["color"] not in ("B", "W"):
                raise ValueError("rank mismatch or invalid player color")
            if not row["game_id"] or not row["player_id"]:
                raise ValueError("missing game_id/player_id")
            game = parse_plane_game(row["sgf_content"])
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{context}: {exc}") from exc
        parsed.append(game)
        records.append({
            "game_index": task["game_start"] + i,
            "rank": task["rank"], "rank_id": RANKS.index(task["rank"]),
            "game_id": row["game_id"], "player_id": row["player_id"],
            "color": row["color"], "player_color": int(row["color"] == "W"),
            "source_row": source_row, "shard_id": task["shard_id"],
            "local_offset": offset, "length": game.n_moves,
            "player_ply_count": int(np.count_nonzero(game.colors == int(row["color"] == "W"))),
        })
        offset += game.n_moves

    layouts = {
        f"{prefix}.npy": ("uint8", (offset, 2, PLANE_BYTES)),
        f"{prefix}_moves.npy": ("int16", (offset,)),
        f"{prefix}_colors.npy": ("int8", (offset,)),
    }
    arrays = {}
    try:
        for filename, (dtype, shape) in layouts.items():
            arrays[filename] = np.lib.format.open_memmap(
                directory / (filename + ".tmp"), mode="w+", dtype=dtype, shape=shape
            )
        for game, record in zip(parsed, records):
            span = slice(record["local_offset"], record["local_offset"] + record["length"])
            try:
                planes = moves_to_planes(game.moves, game.colors)
                packed = pack_planes(planes)
                # A per-game gate includes captures, passes, bit order and padding.
                if not np.array_equal(unpack_planes(packed), planes):
                    raise ValueError("packed board round-trip mismatch")
                arrays[f"{prefix}.npy"][span] = packed
                arrays[f"{prefix}_moves.npy"][span] = game.moves
                arrays[f"{prefix}_colors.npy"][span] = game.colors
                del planes, packed
            except ValueError as exc:
                raise ValueError(
                    f"rank={record['rank']} game_id={record['game_id']} "
                    f"source_row={record['source_row']}: {exc}"
                ) from exc
        for array in arrays.values():
            array.flush()
    finally:
        for array in arrays.values():
            _close_mmap(array)
    for filename in layouts:
        os.replace(directory / (filename + ".tmp"), directory / filename)

    meta_name = f"{prefix}_games.parquet"
    meta = pd.DataFrame(records)
    for column, dtype in {"game_index": "int64", "shard_id": "int32",
                          "source_row": "int32", "length": "int32",
                          "local_offset": "int64", "rank_id": "uint8",
                          "player_color": "uint8", "player_ply_count": "int32"}.items():
        meta[column] = meta[column].astype(dtype)
    meta.to_parquet(directory / (meta_name + ".tmp"), index=False)
    os.replace(directory / (meta_name + ".tmp"), directory / meta_name)
    files = {}
    for filename in [*layouts, meta_name]:
        path = directory / filename
        info = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if filename in layouts:
            dtype, shape = layouts[filename]
            info.update(dtype=dtype, shape=list(shape))
        files[filename] = info
    summary = {
        "format_version": FORMAT_VERSION, "build_id": task["build_id"],
        "shard_id": task["shard_id"], "rank": task["rank"],
        "source_start": task["source_start"], "game_start": task["game_start"],
        "num_games": len(records), "total_ply": offset,
        "player_ply_count": int(meta["player_ply_count"].sum()),
        "files": files, "seconds": time.monotonic() - start,
    }
    _shard_check(directory, summary)
    _json_write(directory / f"{prefix}.done.json", summary)
    return summary


def available_cpus() -> int:
    """Respect affinity and Slurm allocation; never infer GPU CPU quota."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation:
        available = min(available, int(allocation))
    return max(1, available)


def _tasks(input_directory, directory, ranks, games_per_shard, expected, build_id):
    shard_id, game_start = 0, 0
    required = {"player_id", "game_id", "rank", "color", "sgf_content"}
    for rank in ranks:
        count = 0
        with (input_directory / f"train_{rank}.csv").open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"train_{rank}.csv: missing required CSV columns")
            rows = []
            for row in reader:
                rows.append(row)
                count += 1
                if len(rows) == games_per_shard:
                    yield dict(directory=str(directory), rank=rank, rows=rows,
                               source_start=count - len(rows), shard_id=shard_id,
                               game_start=game_start, build_id=build_id)
                    game_start += len(rows)
                    shard_id += 1
                    rows = []
            if rows:
                yield dict(directory=str(directory), rank=rank, rows=rows,
                           source_start=count - len(rows), shard_id=shard_id,
                           game_start=game_start, build_id=build_id)
                game_start += len(rows)
                shard_id += 1
        if count != expected:
            raise ValueError(f"train_{rank}.csv: expected {expected:,} games, found {count:,}")


def convert_training(input_directory: Path = DEFAULT_INPUT, directory: Path = DEFAULT_OUTPUT,
                     *, workers: int = 1, games_per_shard: int = 4096,
                     ranks=RANKS, expected_games_per_rank: int = 100_000) -> Path:
    """Explicit, resumable conversion. Complete manifest is published last.

    Defaults require all ten ranks and exactly one million games. Subsets and
    smaller counts must be explicitly requested (e.g. for synthetic tests).
    Resume checks all source/converter SHA-256 fingerprints and shard hashes.
    Use a new directory for a different dataset, codec or shard configuration.
    """
    input_directory, directory = Path(input_directory).resolve(), Path(directory).resolve()
    ranks = tuple(ranks)
    if not ranks or len(set(ranks)) != len(ranks) or any(rank not in RANKS for rank in ranks):
        raise ValueError("ranks must be a nonempty unique subset of the ten ranks")
    if min(workers, games_per_shard, expected_games_per_rank) < 1:
        raise ValueError("workers, games_per_shard and expected_games_per_rank must be positive")
    if workers > available_cpus():
        raise ValueError(f"requested {workers} workers but CPU allocation allows {available_cpus()}")
    sources = []
    for rank in ranks:
        path = input_directory / f"train_{rank}.csv"
        sources.append({"rank": rank, "file": path.name,
                        "bytes": path.stat().st_size, "sha256": _sha256(path)})
    package = Path(__file__).resolve().parents[1]
    converter_files = [Path(__file__), package / "data" / "sgf.py",
                       package / "features" / "sgf_planes.py", package / "features" / "game_stats.py"]
    config = {
        "format_version": FORMAT_VERSION, "codec": CODEC, "bitorder": BITORDER,
        "plane_shape": [2, 19, 19], "packed_shape": [2, PLANE_BYTES],
        "dtype": "uint8", "channels": ["black", "white"], "padding_bits": 7,
        "input_directory": str(input_directory), "sources": sources,
        "ranks": list(ranks), "games_per_shard": games_per_shard,
        "expected_games_per_rank": expected_games_per_rank,
        "converter_sha256": {path.name: _sha256(path) for path in converter_files},
    }
    build_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["build_id"] = build_id
    directory.mkdir(parents=True, exist_ok=True)
    build_path = directory / "build.json"
    if build_path.exists():
        if json.loads(build_path.read_text(encoding="utf-8")) != config:
            raise ValueError("source/converter/config changed; choose a new output directory")
    else:
        if any(directory.iterdir()):
            raise ValueError("output directory is nonempty but has no build.json")
        _json_write(build_path, config)
    manifest_path = directory / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    summaries = []
    pending = {}

    def collect(done):
        for future in done:
            pending.pop(future)
            summary = future.result()
            summaries.append(summary)
            print(f"shard {summary['shard_id']:05d}: {summary['num_games']:,} games, "
                  f"{summary['total_ply']:,} plies, {summary['seconds']:.1f}s", flush=True)

    # spawn works on Windows and avoids forking an initialized Numba runtime.
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = "1"
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as pool:
            try:
                for task in _tasks(input_directory, directory, ranks, games_per_shard,
                                   expected_games_per_rank, build_id):
                    done_path = directory / f"shard_{task['shard_id']:05d}.done.json"
                    if done_path.exists():
                        try:
                            summary = json.loads(done_path.read_text(encoding="utf-8"))
                            for key in ("build_id", "rank", "source_start", "game_start", "shard_id"):
                                if summary[key] != task[key]:
                                    raise ValueError(f"resume mismatch: {key}")
                            if summary["num_games"] != len(task["rows"]):
                                raise ValueError("resume game count mismatch")
                            _shard_check(directory, summary, hashes=True)
                        except (OSError, ValueError, KeyError):
                            # This shard failed validation: rebuild only its owned files.
                            done_path.unlink()
                        else:
                            summaries.append(summary)
                            print(f"shard {summary['shard_id']:05d}: verified, resuming", flush=True)
                            continue
                    pending[pool.submit(_write_shard, task)] = task["shard_id"]
                    if len(pending) >= 2 * workers:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        collect(done)
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    summaries.sort(key=lambda summary: summary["game_start"])
    metadata = []
    for summary in summaries:
        _shard_check(directory, summary)
        metadata.append(pd.read_parquet(directory / f"shard_{summary['shard_id']:05d}_games.parquet"))
    games = pd.concat(metadata, ignore_index=True)
    if not np.array_equal(games["game_index"], np.arange(len(games))):
        raise ValueError("noncontiguous game index")
    if games.duplicated(["rank", "game_id"]).any():
        raise ValueError("duplicate (rank, game_id); never deduplicate by game_id alone")
    if len(games) != expected_games_per_rank * len(ranks):
        raise ValueError("incomplete game count")
    if (games.groupby("player_id")["rank"].nunique() > 1).any():
        raise ValueError("player appears in more than one rank")
    # Detect source modification during conversion before publishing completion.
    for source in sources:
        if _sha256(input_directory / source["file"]) != source["sha256"]:
            raise ValueError(f"source changed during conversion: {source['file']}")
    games_path = directory / "games.parquet"
    games.to_parquet(directory / "games.parquet.tmp", index=False)
    os.replace(directory / "games.parquet.tmp", games_path)
    manifest = dict(config, complete=True, num_games=len(games),
                    total_ply=int(games["length"].sum()),
                    player_ply_count=int(games["player_ply_count"].sum()),
                    games={"file": "games.parquet", "bytes": games_path.stat().st_size,
                           "sha256": _sha256(games_path)}, shards=summaries)
    _json_write(manifest_path, manifest)
    return manifest_path


class BoardCache:
    """Read-only, process-local memmaps with an LRU limit on open shards.

    Only numeric per-game metadata is retained, not SGF strings or ID lists.
    Corrupt/missing completion markers or wrong sizes/shapes are rejected.
    Full content integrity is checked by verify_cache() or conversion resume.
    """

    def __init__(self, directory: Path = DEFAULT_OUTPUT, *, max_open_shards: int = 8):
        self.directory = Path(directory).resolve()
        if max_open_shards < 1:
            raise ValueError("max_open_shards must be positive")
        self.max_open_shards = max_open_shards
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        if (manifest.get("complete") is not True or manifest.get("format_version") != FORMAT_VERSION
                or manifest.get("codec") != CODEC or manifest.get("bitorder") != BITORDER
                or manifest.get("packed_shape") != [2, 46] or manifest.get("dtype") != "uint8"):
            raise ValueError("incomplete or unsupported board cache")
        record = manifest["games"]
        if record["file"] != "games.parquet":
            raise ValueError("invalid games metadata path")
        meta_path = self.directory / "games.parquet"
        if meta_path.stat().st_size != record["bytes"] or _sha256(meta_path) != record["sha256"]:
            raise ValueError("games metadata checksum mismatch")
        frame = pd.read_parquet(meta_path, columns=["game_index", *NUMERIC_COLUMNS])
        if len(frame) != manifest["num_games"] or not np.array_equal(frame["game_index"], np.arange(len(frame))):
            raise ValueError("invalid game indices/count")
        self.shard_ids = frame["shard_id"].to_numpy(np.int32)
        self.offsets = frame["local_offset"].to_numpy(np.int64)
        self.lengths = frame["length"].to_numpy(np.int32)
        self.rank_ids = frame["rank_id"].to_numpy(np.uint8)
        self.player_colors = frame["player_color"].to_numpy(np.uint8)
        self._summaries = {}
        for summary in manifest["shards"]:
            shard_id = summary["shard_id"]
            if shard_id in self._summaries:
                raise ValueError("duplicate shard ID")
            done = self.directory / f"shard_{shard_id:05d}.done.json"
            if json.loads(done.read_text(encoding="utf-8")) != summary:
                raise ValueError(f"shard completion marker mismatch: {shard_id}")
            if summary["build_id"] != manifest["build_id"]:
                raise ValueError("shard belongs to a different build")
            _shard_check(self.directory, summary)
            self._summaries[shard_id] = summary
            begin, size = summary["game_start"], summary["num_games"]
            end = begin + size
            lens = self.lengths[begin:end]
            expected_offsets = np.r_[0, np.cumsum(lens, dtype=np.int64)[:-1]]
            if (len(lens) != size or np.any(lens <= 0)
                    or not np.all(self.shard_ids[begin:end] == shard_id)
                    or not np.array_equal(self.offsets[begin:end], expected_offsets)
                    or int(lens.sum()) != summary["total_ply"]):
                raise ValueError(f"invalid shard offsets/lengths: {shard_id}")
        if (sum(s["num_games"] for s in self._summaries.values()) != len(self)
                or sum(s["total_ply"] for s in self._summaries.values()) != manifest["total_ply"]
                or np.any(self.rank_ids > 9) or np.any(self.player_colors > 1)):
            raise ValueError("invalid cache metadata")
        self._pid = os.getpid()
        self._open = OrderedDict()

    def __len__(self):
        return len(self.lengths)

    def close(self):
        for arrays in self._open.values():
            for array in arrays:
                _close_mmap(array)
        self._open.clear()

    def __getstate__(self):
        # Windows DataLoader spawn must not pickle mmap handles or payloads.
        state = self.__dict__.copy()
        state["_open"] = OrderedDict()
        state["_pid"] = None
        return state

    def _arrays(self, shard_id):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if shard_id in self._open:
            self._open.move_to_end(shard_id)
            return self._open[shard_id]
        prefix = f"shard_{int(shard_id):05d}"
        arrays = tuple(np.load(self.directory / filename, mmap_mode="r", allow_pickle=False)
                       for filename in (f"{prefix}.npy", f"{prefix}_colors.npy"))
        self._open[shard_id] = arrays
        if len(self._open) > self.max_open_shards:
            _, old = self._open.popitem(last=False)
            for array in old:
                _close_mmap(array)
        return arrays

    def _game_index(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError("game index out of range")
        return index

    def packed(self, game_index: int, plies=None) -> np.ndarray:
        """Return an owned packed copy, safe even after LRU eviction."""
        index = self._game_index(game_index)
        boards, _ = self._arrays(int(self.shard_ids[index]))
        start, length = int(self.offsets[index]), int(self.lengths[index])
        game = boards[start:start + length]
        if plies is None:
            return np.array(game, copy=True)
        indices = np.asarray(plies)
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= length):
            raise IndexError("ply index out of range or not integer")
        return np.array(game[indices], copy=True)

    def game(self, game_index: int, plies=None) -> np.ndarray:
        """On-demand uint8[T,2,19,19], or just the requested plies."""
        return unpack_planes(self.packed(game_index, plies))

    def colors(self, game_index: int) -> np.ndarray:
        """Actual move colors (0 black, 1 white); no parity assumption."""
        index = self._game_index(game_index)
        _, colors = self._arrays(int(self.shard_ids[index]))
        start, length = int(self.offsets[index]), int(self.lengths[index])
        result = np.array(colors[start:start + length], copy=True)
        if np.any((result != 0) & (result != 1)):
            raise ValueError("invalid cached move color")
        return result

    def history_batch(self, game_indices, plies, *, steps: int = 8):
        """Grouped batch read -> (uint8[B,steps,2,19,19], int8[B] colors).

        Newest board comes first. Missing history is zero, never another game.
        Packed reads are grouped by shard, then restored to input batch order.
        """
        games, plies = np.asarray(game_indices), np.asarray(plies)
        if games.ndim != 1 or plies.ndim != 1 or games.shape != plies.shape:
            raise ValueError("game_indices and plies must be equal-length 1D arrays")
        if steps < 1:
            raise ValueError("steps must be positive")
        if games.size and (games.dtype.kind not in "iu" or plies.dtype.kind not in "iu"):
            raise ValueError("game_indices and plies must contain integers")
        if np.any(games < 0) or np.any(games >= len(self)):
            raise IndexError("game index out of range")
        games, plies = games.astype(np.int64), plies.astype(np.int64)
        if np.any(plies < 0) or np.any(plies >= self.lengths[games]):
            raise IndexError("ply index out of range")
        packed = np.zeros((len(games), steps, 2, PLANE_BYTES), np.uint8)
        move_colors = np.empty(len(games), np.int8)
        shard_ids = self.shard_ids[games]
        for shard_id in np.unique(shard_ids):
            selected = np.flatnonzero(shard_ids == shard_id)
            boards, colors = self._arrays(int(shard_id))
            local_ply = plies[selected, None] - np.arange(steps)
            valid = local_ply >= 0
            offsets = self.offsets[games[selected]]
            addresses = offsets[:, None] + np.maximum(local_ply, 0)
            block = np.array(boards[addresses], copy=True)
            block[~valid] = 0
            packed[selected] = block
            move_colors[selected] = colors[offsets + plies[selected]]
        if np.any((move_colors != 0) & (move_colors != 1)):
            raise ValueError("invalid cached move color")
        return unpack_planes(packed), move_colors

    def history(self, game_index: int, ply: int, *, steps: int = 8) -> np.ndarray:
        return self.history_batch([game_index], [ply], steps=steps)[0][0]

    def training_batch(self, game_indices, plies) -> np.ndarray:
        """On-demand uint8[B,18,19,19], 8 histories then actual color planes."""
        histories, colors = self.history_batch(game_indices, plies)
        inputs = np.empty((len(colors), 18, 19, 19), np.uint8)
        inputs[:, :16] = histories.reshape(len(colors), 16, 19, 19)
        inputs[:, 16] = (colors == 0)[:, None, None]
        inputs[:, 17] = (colors == 1)[:, None, None]
        return inputs


def verify_cache(directory: Path = DEFAULT_OUTPUT) -> dict:
    """Explicit full file-hash/padding check; never creates unpacked files."""
    cache = BoardCache(directory)
    try:
        for summary in cache.manifest["shards"]:
            _shard_check(cache.directory, summary, hashes=True)
            boards, colors = cache._arrays(summary["shard_id"])
            for begin in range(0, len(boards), 65536):
                if np.any(boards[begin:begin + 65536, :, -1] & np.uint8(254)):
                    raise ValueError("nonzero padding bits")
                block = colors[begin:begin + 65536]
                if np.any((block != 0) & (block != 1)):
                    raise ValueError("invalid move colors")
        return {"num_games": len(cache), "total_ply": cache.manifest["total_ply"],
                "payload_bytes": cache.manifest["total_ply"] * 92}
    finally:
        cache.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert", help="explicitly convert training CSVs to packed shards")
    convert.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    convert.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    convert.add_argument("--workers", type=int, default=min(8, available_cpus()))
    convert.add_argument("--games-per-shard", type=int, default=4096)
    convert.add_argument("--ranks", nargs="+", choices=RANKS, default=list(RANKS))
    convert.add_argument("--expected-games-per-rank", type=int, default=100_000)
    verify = commands.add_parser("verify", help="verify completed files without creating unpacked copies")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.command == "convert":
        manifest = convert_training(args.input_dir, args.output_dir, workers=args.workers,
                                    games_per_shard=args.games_per_shard, ranks=args.ranks,
                                    expected_games_per_rank=args.expected_games_per_rank)
        print(f"Complete cache manifest: {manifest}")
    else:
        print(json.dumps(verify_cache(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
