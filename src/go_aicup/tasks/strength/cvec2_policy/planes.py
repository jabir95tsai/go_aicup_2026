"""Versioned whole-game zstd input planes shared by the H9 and C10 teachers."""

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import tempfile
import time

from filelock import FileLock
import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features import humansl_torch
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_large import cache

GROUPS = ("fit", "e2", "reserve", "public", "private")
META = ["game_index", "length", "handicap", "player_color"]
PLAN_COUNTS = {
    "fit": (149615, 33363661), "e2": (15460, 3445528), "reserve": (18196, 4057717),
}
GAMES_PER_SHARD = 64
BOARD = ROOT / "data/processed/boards/strength_v1"
RAW_ZIP = ROOT / "data/raw/raw.zip"
_NET_CONFIG = _BOARD_ROOT = None


def _checked_arrays(arrays, lengths):
    offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    n = int(offsets[-1])
    for key, shape, dtype in (
        ("offsets", offsets.shape, np.int64), ("bits", (n, 993), np.uint8),
        ("globals", (n, 19), np.float32), ("target", (n,), np.int16), ("side", (n,), np.uint8),
    ):
        if arrays[key].shape != shape or arrays[key].dtype != dtype:
            raise ValueError(f"unexpected plane shard shape or dtype: {key}")
    if not np.array_equal(arrays["offsets"], offsets):
        raise ValueError("plane shard position offsets differ from the game index")
    if (arrays["target"] < 0).any() or (arrays["target"] > 361).any() or (arrays["side"] > 1).any():
        raise ValueError("invalid plane shard target or side")
    if not np.isfinite(arrays["globals"]).all() or (arrays["bits"][:, -1] & 3).any():
        raise ValueError("invalid input planes or bit padding")
    for a, b in zip(offsets[:-1], offsets[1:]):
        side = arrays["side"][a:b]
        if side.min() != 0 or side.max() != 1:
            raise ValueError("each game must contain both players' moves")
    return arrays


def _decode(path, lengths, expected_sha256=None):
    compressed = path.read_bytes()
    if expected_sha256 is not None and sha256(compressed).hexdigest() != expected_sha256:
        raise ValueError(f"plane shard checksum differs: {path}")
    result = subprocess.run(["zstd", "-q", "-d", "-c"], input=compressed, capture_output=True, check=True)
    with np.load(BytesIO(result.stdout), allow_pickle=False) as data:
        return _checked_arrays(dict(data), lengths)


def _write_shard(path, arrays, lengths):
    buffer = BytesIO()
    np.savez(buffer, **_checked_arrays(arrays, lengths))
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        subprocess.run(["zstd", "-q", "-1", "-c"], input=buffer.getbuffer(), stdout=stream, check=True)
    restored = _decode(temporary, lengths)
    for name, value in arrays.items():
        np.testing.assert_array_equal(restored[name], value)
    digest = file_hash(temporary)
    temporary.replace(path)
    return {"sha256": digest, "bytes": path.stat().st_size}


def _init_worker(config):
    global _NET_CONFIG
    _NET_CONFIG = config


def _encode_shard(task):
    global _BOARD_ROOT
    out, record, rows, board = task
    if _BOARD_ROOT != board:
        cache.init_encoder(_NET_CONFIG, board)
        _BOARD_ROOT = board
    parts = [cache.encode(row) for row in rows]
    lengths = [row[-1] for row in rows]
    arrays = {key: np.concatenate(values) for key, values in zip(
        ("bits", "globals", "target", "side"), zip(*parts, strict=True), strict=True
    )}
    arrays["offsets"] = np.r_[0, np.cumsum(lengths, dtype=np.int64)]
    path = out / record["file"]
    record = record | _write_shard(path, arrays, lengths)
    write_json(path.with_suffix(".json"), record)
    return record


def _canonical_sources(groups):
    manifest_path = BOARD / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = BOARD / "games.parquet"
    digest = file_hash(path)
    if not manifest["complete"] or manifest["games"]["sha256"] != digest:
        raise ValueError("canonical board index differs from its manifest")
    board = pd.read_parquet(path, columns=["game_index", "length", "shard_id"]).set_index("game_index")
    groups = pd.concat(groups.values(), ignore_index=True)
    if not groups.game_index.is_unique:
        raise ValueError("canonical plane groups overlap")
    selected = board.loc[groups.game_index]
    if not np.array_equal(selected.length, groups.length):
        raise ValueError("canonical board lengths differ")
    sources = {"manifest.json": file_hash(manifest_path), "games.parquet": digest}
    for sid in sorted(set(selected.shard_id)):
        for suffix in ("moves.npy", "colors.npy"):
            name = f"shard_{sid:05d}_{suffix}"
            digest = file_hash(BOARD / name)
            if manifest["shards"][sid]["files"][name]["sha256"] != digest:
                raise ValueError(f"canonical board source checksum differs: {name}")
            sources[name] = digest
    return sources


def _official(scratch, smoke):
    from go_aicup.tasks.strength.policy_large import export

    games, questions = export.prepare(scratch)
    groups, selected = {}, []
    for part in ("public", "private"):
        q = questions[questions.part.eq(part)].reset_index(drop=True)
        if smoke:
            q = q.iloc[:1].copy()
        source_ids = np.concatenate(q.games)
        rows = games.set_index("game_index", drop=False).loc[source_ids, META].reset_index(drop=True)
        if not np.array_equal(source_ids, np.arange(source_ids[0], source_ids[0] + len(rows))):
            raise ValueError(f"{part} official game order differs from question order")
        rows["source_game_index"] = rows.game_index
        rows["game_index"] -= source_ids[0]
        q["games"] = [np.asarray(ids, dtype=np.int64) - source_ids[0] for ids in q.games]
        q["dataset"] = "official"
        groups[part] = rows
        selected.append(q)
    if not smoke and sum(g.length.sum() for g in groups.values()) != 2222151:
        raise ValueError("official move count differs from the plan")
    return groups, pd.concat(selected, ignore_index=True)


def _verified_manifest(directory):
    marker = json.loads((directory / "complete.json").read_text())
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    if (marker != {"identity": manifest["identity"], "sha256": file_hash(path)}
            or object_hash(manifest["sources"]) != manifest["identity"]
            or directory.name != manifest["identity"]):
        raise ValueError("plane manifest or cache identity differs")
    return manifest


def _published(directory, manifest):
    for name, digest in manifest["tables"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"plane index checksum differs: {name}")
    for shard in manifest["shards"]:
        if file_hash(directory / shard["file"]) != shard["sha256"]:
            raise ValueError(f"plane shard checksum differs: {shard['file']}")


def _cached_shard(target, record):
    try:
        saved = json.loads(target.with_suffix(".json").read_text())
        if (all(saved[key] == value for key, value in record.items())
                and target.stat().st_size == saved["bytes"] and file_hash(target) == saved["sha256"]):
            return saved
    except (OSError, ValueError, KeyError):
        pass
    return None


def build(groups, out, workers, smoke=False):
    """Build one reusable plane cache version and return its directory."""
    from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table

    started = time.monotonic()
    if set(groups) != set(GROUPS[:3]) or workers < 1:
        raise ValueError("expected fit, e2 and reserve plane groups and at least one worker")
    groups = {name: groups[name][META].astype(np.int64).reset_index(drop=True) for name in GROUPS[:3]}
    for name, rows in groups.items():
        if (not rows.handicap.isin([0, 1]).all() or not rows.player_color.isin([0, 1]).all()
                or (rows.length < 2).any()):
            raise ValueError(f"invalid plane game metadata: {name}")
        if not smoke and (len(rows), int(rows.length.sum())) != PLAN_COUNTS[name]:
            raise ValueError(f"wrong plane cache scope: {name}")
    net_config = torch.load(humansl_torch.CKPT, map_location="cpu", weights_only=False)["config"]
    code = [Path(__file__), Path(cache.__file__), Path(cache.__file__).with_name("export.py"), Path(humansl_torch.__file__),
            ROOT / "src/go_aicup/data/sgf.py"]
    code.extend(humansl_torch.KATAGO_PY / "katago" / "game" / f"{name}.py"
                for name in ("board", "features", "gamestate"))
    sources = {
        "format": 1, "smoke": smoke, "games_per_shard": GAMES_PER_SHARD,
        "board": _canonical_sources(groups), "official_zip": file_hash(RAW_ZIP),
        "game_order": {name: sha256(rows.to_numpy(dtype="<i8").tobytes()).hexdigest()
                       for name, rows in groups.items()},
        "code": {str(path.relative_to(ROOT)): file_hash(path) for path in code},
        "input_config": net_config, "rules": "Japanese", "komi": {"0": 6.5, "1": 0.5},
        "compression": "zstd -1", "bitorder": "big",
    }
    identity = object_hash(sources)
    directory = Path(out) / identity
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / ".writer.lock")):
        if (directory / "complete.json").is_file():
            try:
                manifest = _verified_manifest(directory)
                _published(directory, manifest)
                return directory
            except (OSError, ValueError, KeyError):
                (directory / "complete.json").unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="test-boards-", dir=directory) as temporary:
            scratch = Path(temporary)
            official, questions = _official(scratch, smoke)
            groups.update(official)
            if not smoke and (sum(map(len, groups.values())), sum(g.length.sum() for g in groups.values())) != (193271, 43089057):
                raise ValueError("overall plane cache scope differs from the plan")
            tasks, shards, tables = [], [], {}
            for name, rows in groups.items():
                offsets = np.r_[0, np.cumsum(rows.length.to_numpy(np.int64))]
                table = f"{name}.parquet"
                atomic_table(directory / table, rows[META].assign(
                    dataset="official" if name in official else "canonical", part=name,
                    position_start=offsets[:-1], position_stop=offsets[1:],
                ))
                tables[table] = file_hash(directory / table)
                for start in range(0, len(rows), GAMES_PER_SHARD):
                    end = min(start + GAMES_PER_SHARD, len(rows))
                    record = {"file": f"{name}-{start:06d}.npz.zst", "group": name,
                              "start": start, "end": end, "positions": int(offsets[end] - offsets[start])}
                    saved = _cached_shard(directory / record["file"], record)
                    if saved is not None:
                        shards.append(saved)
                        continue
                    index = "source_game_index" if name in official else "game_index"
                    columns = [index, "handicap", "player_color", "length"]
                    rows_for_task = list(rows.iloc[start:end][columns].itertuples(index=False, name=None))
                    tasks.append((directory, record, rows_for_task, scratch / "boards" if name in official else BOARD))
            atomic_table(directory / "questions.parquet", questions)
            tables["questions.parquet"] = file_hash(directory / "questions.parquet")
            if tasks:
                with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                         initializer=_init_worker, initargs=(net_config,)) as pool:
                    shards.extend(pool.map(_encode_shard, tasks))
        manifest = {
            "sources": sources, "identity": identity, "tables": tables,
            "groups": {name: {"games": len(rows), "positions": int(rows.length.sum())} for name, rows in groups.items()},
            "shards": sorted(shards, key=lambda shard: (GROUPS.index(shard["group"]), shard["start"])),
            "compressed_bytes": sum(shard["bytes"] for shard in shards),
            "seconds": time.monotonic() - started, "new_shards": len(tasks),
        }
        _published(directory, manifest)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "complete.json", {"identity": identity, "sha256": file_hash(directory / "manifest.json")})
    return directory


class PlaneReader:
    def __init__(self, directory, games, groups=("fit", "e2")):
        self.directory = Path(directory)
        manifest = _verified_manifest(self.directory)
        self.identity = manifest["identity"]
        self.shards, frames = [], []
        offset = 0
        for name in groups:
            path = self.directory / f"{name}.parquet"
            if file_hash(path) != manifest["tables"][path.name]:
                raise ValueError(f"plane index checksum differs: {name}")
            frame = pd.read_parquet(path)[META]
            frames.append(frame)
            first = 0
            for shard in manifest["shards"]:
                if shard["group"] != name:
                    continue
                start, end = shard["start"], shard["end"]
                if first != start or not start < end <= len(frame):
                    raise ValueError("plane shard coverage differs from game index")
                lengths = frame.length.iloc[start:end].to_numpy()
                if lengths.sum() != shard["positions"]:
                    raise ValueError("plane shard position count differs")
                self.shards.append((offset + start, offset + end, shard, lengths))
                first = end
            if first != len(frame):
                raise ValueError("incomplete plane group")
            offset += len(frame)
        if not pd.concat(frames, ignore_index=True).equals(games[META].astype(np.int64).reset_index(drop=True)):
            raise ValueError("teacher game order, length, color or handicap differs from plane cache")

    def batches(self, first, last, size):
        """Yield complete games in teacher order, with bounded per-worker decompression."""
        if not self.shards or not 0 <= first < last <= self.shards[-1][1] or size < 1:
            raise ValueError("invalid plane batch range or size")
        rows, parts = [], []
        for start, end, shard, lengths in self.shards:
            if start >= last:
                break
            if end <= first:
                continue
            data = _decode(self.directory / shard["file"], lengths, shard["sha256"])
            for i in range(max(first, start), min(last, end)):
                a, b = data["offsets"][i-start:i-start+2]
                rows.append(i)
                parts.append(tuple(data[name][a:b] for name in ("bits", "globals", "target", "side")))
                if len(rows) == size:
                    yield rows, parts
                    rows, parts = [], []
        if rows:
            yield rows, parts
