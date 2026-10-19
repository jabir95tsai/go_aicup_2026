"""Self-contained game splits, validation questions and player-ply indices.

Consumes board_cache's games.parquet and actual color memmaps. No CSV/SGF
parsing, board decompression, GPU allocation or training occurs here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import BoardCache, _close_mmap
from go_aicup.data.cv import (make_strength_v1_questions, make_strength_v1_split,
                              strength_v1_overlap)
from go_aicup.tasks.strength.shared.runtime import (check_index_hashes,
    file_hash, load_dataset_manifest, object_hash, write_json)


def write_sample_indices(cache: BoardCache, meta: pd.DataFrame, selected_games,
                         directory: Path, prefix: str) -> dict:
    selected = np.sort(np.asarray(selected_games, dtype=np.int64))
    if not len(selected) or len(np.unique(selected)) != len(selected):
        raise ValueError("game selection must be nonempty and unique")
    if len(meta) > np.iinfo(np.uint32).max or int(meta["length"].max()) > 65536:
        raise ValueError("game/ply indices exceed uint32/uint16 format")
    counts = meta["player_ply_count"].to_numpy(dtype=np.int64)
    if np.any(counts[selected] <= 0):
        raise ValueError("selected game has no labelled-player plies")
    size = int(counts[selected].sum())
    paths = [directory / f"{prefix}_game.npy", directory / f"{prefix}_ply.npy"]
    temporary = [path.with_name(path.name + ".tmp") for path in paths]
    arrays = [np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(size,))
              for path, dtype in zip(temporary, (np.uint32, np.uint16))]
    cursor = 0
    per_rank = np.zeros(10, dtype=np.int64)
    try:
        for number, game in enumerate(selected):
            plies = np.flatnonzero(cache.colors(int(game)) == cache.player_colors[game])
            if len(plies) != counts[game]:
                raise ValueError(f"player ply count mismatch for game_index={game}")
            end = cursor + len(plies)
            arrays[0][cursor:end] = game
            arrays[1][cursor:end] = plies
            per_rank[cache.rank_ids[game]] += len(plies)
            cursor = end
            if (number + 1) % 10_000 == 0:
                print(f"{prefix}: {number + 1:,}/{len(selected):,} games; {cursor:,} samples", flush=True)
        if cursor != size:
            raise ValueError("sample count does not match metadata")
        for array in arrays:
            array.flush()
    finally:
        for array in arrays:
            _close_mmap(array)
    for source, target in zip(temporary, paths):
        os.replace(source, target)
    return {"num_games": len(selected), "num_samples": size,
            "per_rank_samples": per_rank.tolist()}


def select_monitoring_games(table: pd.DataFrame, seed: int, games_per_rank: int) -> np.ndarray:
    """Sample whole validation games; independent of row order and game length."""
    if type(games_per_rank) is not int or games_per_rank < 1:
        raise ValueError("games_per_rank must be a positive integer")
    if not table["game_index"].is_unique:
        raise ValueError("duplicate game_index in monitoring source")
    rng = np.random.default_rng(seed)
    selected = []
    for rank_id in range(10):
        candidates = np.sort(table.loc[(table["split"] == "valid") &
            (table["rank_id"] == rank_id), "game_index"].to_numpy(dtype=np.int64))
        if len(candidates) < games_per_rank:
            raise ValueError(f"validation rank {rank_id} needs at least {games_per_rank} games")
        selected.append(rng.choice(candidates, games_per_rank, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.uint32)


def prepare(config: dict) -> Path:
    directory = Path(config["paths"]["dataset_dir"])
    cache = BoardCache(Path(config["paths"]["cache_dir"]),
                       max_open_shards=config["loader"]["max_open_shards"])
    try:
        identity = {"format_version": 2, "cache_build_id": cache.manifest["build_id"],
                    "split_config": config["split"], "questions_config": config["questions"],
                    "seed": config["seed"],
                    "generator_sha256": {name: file_hash(path) for name, path in {
                        "cv.py": Path(__file__).resolve().parents[3] / "data" / "cv.py",
                        "shared/prepare.py": Path(__file__),
                    }.items()}}
        if "monitoring" in config:
            identity["monitoring_config"] = config["monitoring"]
        build_path = directory / "build.json"
        if build_path.exists():
            if json.loads(build_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("preparation settings/code changed; choose a new dataset directory")
            if (directory / "manifest.json").exists():
                record = load_dataset_manifest(config)
                check_index_hashes(config, record)
                return directory / "manifest.json"
        else:
            for target in (directory,):
                if target.exists() and any(target.iterdir()):
                    raise ValueError(f"refusing to replace an unrelated nonempty directory: {target}")
            directory.mkdir(parents=True, exist_ok=True)
            write_json(build_path, identity)
        meta = pd.read_parquet(cache.directory / "games.parquet")
        if not np.array_equal(meta["game_index"], np.arange(len(meta))):
            raise ValueError("cache game_index must match metadata row order")
        table = make_strength_v1_split(meta, config["seed"], **config["split"])
        valid_questions, _ = make_strength_v1_questions(
            table, "valid", config["questions"]["validation"], config["seed"] + 2)
        tables = {"strength_v1_games.parquet": table,
                     "strength_v1_validation_questions.parquet": valid_questions}
        for name, frame in tables.items():
            temporary = directory / (name + ".tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, directory / name)
        selections = {
            "train": table.loc[table["split"] == "train", "game_index"].to_numpy(),
            "valid": table.loc[table["split"] == "valid", "game_index"].to_numpy(),
        }
        if "monitoring" in config:
            monitoring_games = select_monitoring_games(
                table, config["seed"], config["monitoring"]["games_per_rank"])
            selections["monitor"] = monitoring_games
            temporary = directory / "monitor_games.npy.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, monitoring_games, allow_pickle=False)
            os.replace(temporary, directory / "monitor_games.npy")
        sets = {name: write_sample_indices(cache, meta, games, directory, name)
                for name, games in selections.items()}
        files = {f"{prefix}_{kind}.npy": {
                    "sha256": file_hash(directory / f"{prefix}_{kind}.npy"),
                    "bytes": (directory / f"{prefix}_{kind}.npy").stat().st_size}
                 for prefix in sets for kind in ("game", "ply")}
        if "monitoring" in config:
            path = directory / "monitor_games.npy"
            files[path.name] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        table_files = {name: {"sha256": file_hash(directory / name),
                           "bytes": (directory / name).stat().st_size} for name in tables}
        record = dict(identity, complete=True, sets=sets, files=files, table_files=table_files,
                      player_overlap=strength_v1_overlap(table),
                      note="Game-disjoint split; players may overlap.")
        record["dataset_id"] = object_hash(record)
        write_json(directory / "manifest.json", record)
        print(json.dumps({"sets": sets, "player_overlap": record["player_overlap"]}, indent=2), flush=True)
        return directory / "manifest.json"
    finally:
        cache.close()


