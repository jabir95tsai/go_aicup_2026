"""Build game-disjoint splits and compact per-game effective-ply indices.

Only reads existing packed cache; never parses raw data or decompresses boards.
Split seed is fixed at 2026; training seed and model settings do not determine
the shared data identity. Repeated runs validate and reuse existing files.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import BoardCache, _close_mmap
from go_aicup.data.cv import make_strength_v1_split, strength_v1_overlap
from .config import DEFAULT_CONFIG, read_config, validate_config


def prepare(config):
    validate_config(config)
    directory = Path(config["paths"]["dataset_dir"])
    cache = BoardCache(Path(config["paths"]["cache_dir"]),
                       max_open_shards=config["loader"]["max_open_shards"])
    try:
        identity = {"format_version": 2, "split_seed": config["split_seed"],
                    "cache_build_id": cache.manifest["build_id"],
                    "cache_manifest_sha256": file_hash(cache.directory / "manifest.json"),
                    "split": config["split"], "positions": config["positions"],
                    "input_encoding": config["model"]["input_encoding"],
                    "code_sha256": {"prepare": file_hash(Path(__file__)),
                                    "cv": file_hash(Path(__file__).parents[3] / "data/cv.py")}}
        manifest_path = directory / "manifest.json"
        if directory.exists() and any(directory.iterdir()):
            if not manifest_path.exists():
                raise ValueError("incomplete/nonempty output; choose a new dataset directory")
            record = json.loads(manifest_path.read_text())
            if record["identity"] != identity:
                raise ValueError("preparation identity changed; choose a new dataset directory")
            expected_files = {"games.parquet"} | {
                f"{split}_{suffix}.npy" for split in ("train", "valid")
                for suffix in ("game", "offsets", "ply")}
            if (set(record["files"]) != expected_files
                    or record["build_id"] != object_hash(identity)):
                raise ValueError("invalid prepared manifest files or build identity")
            for name, digest in record["files"].items():
                if not (directory / name).is_file() or file_hash(directory / name) != digest:
                    raise ValueError(f"prepared file checksum mismatch: {name}")
            return manifest_path
        meta = pd.read_parquet(cache.directory / "games.parquet")
        table = make_strength_v1_split(meta, config["split_seed"], **config["split"])
        if not np.array_equal(table.game_index, np.arange(len(cache))):
            raise ValueError("metadata must have contiguous game indices")
        directory.mkdir(parents=True, exist_ok=True)
        # Bounds allocate at most one uint16 per actual move, not 18 planes.
        counts = np.zeros(len(cache), np.int64)
        split_reports = {}
        files = []
        for split in ("train", "valid"):
            games = table.loc[table.split == split, "game_index"].to_numpy(np.uint32)
            capacity = int(cache.lengths[games].sum())
            temporary = directory / f"{split}_ply.tmp.npy"
            plies_out = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint16, shape=(capacity,))
            offsets = np.zeros(len(games) + 1, np.int64)
            try:
                for row, game in enumerate(games):
                    if cache.lengths[game] > 65536:
                        raise ValueError(f"game {game}: ply exceeds uint16")
                    colors = cache.colors(int(game))
                    plies = np.flatnonzero((colors == cache.player_colors[game]) &
                                           (np.arange(len(colors)) >= config["positions"]["min_ply"]))
                    if len(plies) < config["positions"]["positions_per_rank"]:
                        raise ValueError(f"game_index={game}: only {len(plies)} effective positions; need seven")
                    counts[game] = len(plies)
                    offsets[row + 1] = offsets[row] + len(plies)
                    plies_out[offsets[row]:offsets[row + 1]] = plies
                plies_out.flush()
                # Publish an exact-length npy, copying in bounded chunks.
                target = directory / f"{split}_ply.npy"
                compact = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint16, shape=(int(offsets[-1]),))
                try:
                    for begin in range(0, len(compact), 1_000_000):
                        compact[begin:begin + 1_000_000] = plies_out[begin:min(begin + 1_000_000, len(compact))]
                    compact.flush()
                finally:
                    _close_mmap(compact)
            finally:
                _close_mmap(plies_out)
                temporary.unlink(missing_ok=True)
            np.save(directory / f"{split}_game.npy", games, allow_pickle=False)
            np.save(directory / f"{split}_offsets.npy", offsets, allow_pickle=False)
            files.extend(f"{split}_{suffix}.npy" for suffix in ("game", "ply", "offsets"))
            split_reports[split] = {"games": len(games), "positions": int(offsets[-1]),
                "per_rank_games": [int(np.sum(cache.rank_ids[games] == rank)) for rank in range(10)],
                "per_rank_positions": [int(counts[games][cache.rank_ids[games] == rank].sum()) for rank in range(10)]}
        table["effective_position_count"] = counts
        table.to_parquet(directory / "games.parquet", index=False)
        files.append("games.parquet")
        write_json(manifest_path, {"identity": identity, "build_id": object_hash(identity),
            "cache_dir": str(cache.directory), "splits": split_reports,
            "player_overlap": strength_v1_overlap(table),
            "protocol": "Game-disjoint; players may overlap. Centroids use training positions only.",
            "files": {name: file_hash(directory / name) for name in files}})
        return manifest_path
    finally:
        cache.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(prepare(read_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
