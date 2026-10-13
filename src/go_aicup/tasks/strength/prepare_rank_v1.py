"""Explicit game splits, fixed questions and compact player-ply indices.

Consumes board_cache's games.parquet and actual color memmaps. No CSV/SGF
parsing, board decompression, GPU allocation or training occurs here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import BoardCache, _close_mmap
from go_aicup.data.cv import (make_strength_v1_questions, make_strength_v1_split,
                              strength_v1_overlap)
from go_aicup.tasks.strength.runtime import (DEFAULT_CONFIG, check_index_hashes,
    file_hash, load_dataset_manifest, object_hash, read_config, write_json)


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


def prepare(config: dict) -> Path:
    directory = Path(config["paths"]["dataset_dir"])
    cv_directory = Path(config["paths"]["cv_dir"])
    cache = BoardCache(Path(config["paths"]["cache_dir"]),
                       max_open_shards=config["loader"]["max_open_shards"])
    try:
        identity = {"format_version": 1, "cache_build_id": cache.manifest["build_id"],
                    "split_config": config["split"], "questions_config": config["questions"],
                    "seed": config["seed"], "cv_dir": str(cv_directory),
                    "generator_sha256": {name: file_hash(path) for name, path in {
                        "cv.py": Path(__file__).resolve().parents[2] / "data" / "cv.py",
                        "prepare_rank_v1.py": Path(__file__),
                    }.items()}}
        build_path = directory / "build.json"
        if build_path.exists():
            if json.loads(build_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("preparation settings/code changed; choose new dataset and CV directories")
            if (directory / "manifest.json").exists():
                record = load_dataset_manifest(config)
                check_index_hashes(config, record)
                return directory / "manifest.json"
        else:
            for target in (directory, cv_directory):
                if target.exists() and any(target.iterdir()):
                    raise ValueError(f"refusing to replace an unrelated nonempty directory: {target}")
            directory.mkdir(parents=True, exist_ok=True)
            cv_directory.mkdir(parents=True, exist_ok=True)
            write_json(build_path, identity)
        meta = pd.read_parquet(cache.directory / "games.parquet")
        if not np.array_equal(meta["game_index"], np.arange(len(meta))):
            raise ValueError("cache game_index must match metadata row order")
        table = make_strength_v1_split(meta, config["seed"], **config["split"])
        train_questions, train_eligibility = make_strength_v1_questions(
            table, "train", config["questions"]["train_calibration"], config["seed"] + 1)
        valid_questions, valid_eligibility = make_strength_v1_questions(
            table, "valid", config["questions"]["validation"], config["seed"] + 2)
        cv_tables = {"strength_v1_games.parquet": table,
                     "strength_v1_train_calibration_questions.parquet": train_questions,
                     "strength_v1_validation_questions.parquet": valid_questions,
                     "strength_v1_eligibility.parquet": pd.concat([train_eligibility, valid_eligibility], ignore_index=True)}
        for name, frame in cv_tables.items():
            temporary = cv_directory / (name + ".tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, cv_directory / name)
        selections = {
            "train": table.loc[table["split"] == "train", "game_index"].to_numpy(),
            "valid": table.loc[table["split"] == "valid", "game_index"].to_numpy(),
            "calibration": np.unique(np.concatenate(train_questions["games"].to_numpy())),
        }
        sets = {name: write_sample_indices(cache, meta, games, directory, name)
                for name, games in selections.items()}
        files = {f"{prefix}_{kind}.npy": {
                    "sha256": file_hash(directory / f"{prefix}_{kind}.npy"),
                    "bytes": (directory / f"{prefix}_{kind}.npy").stat().st_size}
                 for prefix in sets for kind in ("game", "ply")}
        cv_files = {name: {"sha256": file_hash(cv_directory / name),
                           "bytes": (cv_directory / name).stat().st_size} for name in cv_tables}
        record = dict(identity, complete=True, sets=sets, files=files, cv_files=cv_files,
                      player_overlap=strength_v1_overlap(table),
                      note="Game-disjoint split; players may overlap. Calibration uses training games only.")
        record["dataset_id"] = object_hash(record)
        write_json(directory / "manifest.json", record)
        print(json.dumps({"sets": sets, "player_overlap": record["player_overlap"]}, indent=2), flush=True)
        return directory / "manifest.json"
    finally:
        cache.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(f"Dataset manifest: {prepare(read_config(args.config))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
