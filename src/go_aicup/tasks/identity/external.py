"""External (OGS) identity-training players merged into the Yugen board cache (2026-11-18).

Input: the filtered OGS shards of scripts/identity/ogs_stream_filter.py (moves_XXXXX.npy, meta_XXXXX.npy,
owners.npy). Selection: ranked games; players are archive folder owners (humans; bots have no folder),
optionally also a human opponent, a start year and a minimum rank for both players; a (player, colour) appearance per side; players with at
least `min_games` such games; at most `max_players` of them (random, seed) and at most
`games_per_player` games each (random). A game shared by two selected players is stored once per player
(the cache keeps one judged colour per row). Each game is cut at its first pass (the board cache holds
points 0..360 only; the filter guarantees >= 60 plies before any pass).

Output: a BoardCache directory whose first 1,000,000 rows ARE the Yugen cache (its shard files symlinked,
same build id, same game_index), followed by the OGS rows; plus external_players.parquet
(game_index -> player key "ogs_<id>") and external.json. Everything that reads the default Yugen cache
is unchanged; only a config whose paths.cache_dir points at the merged directory sees the OGS rows.

    python -m go_aicup.tasks.identity.external build --filtered DIR --out DIR [--max-players N] ...
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import DEFAULT_OUTPUT, PLANE_BYTES, pack_planes

GAMES_PER_SHARD = 4096
PASS = 361


def load_filtered(directory: Path):
    """(meta structured array with a global "start" column, moves int16 memmaps per shard, owners)."""
    metas, moves = [], []
    for i in range(10_000):
        mp = directory / f"meta_{i:05d}.npy"
        if not mp.exists():
            break
        m = np.load(mp)
        metas.append(pd.DataFrame(m).assign(shard=i))
        moves.append(np.load(directory / f"moves_{i:05d}.npy", mmap_mode="r"))
    if not metas:
        raise ValueError(f"no filtered shards in {directory}")
    owners = np.load(directory / "owners.npy")
    return pd.concat(metas, ignore_index=True), moves, owners


def select(meta: pd.DataFrame, owners: np.ndarray, *, min_games: int, max_players: int | None,
           games_per_player: int, seed: int, min_year: int | None = None, min_rank: float | None = None,
           human_opponent: bool = False):
    """DataFrame of (player, row into meta, colour 0/1), sorted by player then game id.

    Only folder owners (humans) become players. human_opponent=True also requires the opponent to own a
    folder; with a partial download (owners = folders read so far) that is a lower bound, since an
    opponent's folder may simply not have arrived yet. Training uses only the player's own moves."""
    keep = (meta["ranked"] == 1) & (meta["black"] != meta["white"])
    if human_opponent:
        keep &= meta["black"].isin(owners) & meta["white"].isin(owners)
    if min_year is not None:
        keep &= meta["year"] >= min_year
    if min_rank is not None:
        keep &= (meta["br"] >= min_rank) & (meta["wr"] >= min_rank)
    rows = np.flatnonzero(keep.to_numpy())
    sides = pd.DataFrame({"player": np.concatenate([meta["black"].to_numpy()[rows], meta["white"].to_numpy()[rows]]),
                          "row": np.concatenate([rows, rows]),
                          "color": np.repeat(np.array([0, 1], np.int8), len(rows))})
    sides = sides[np.isin(sides["player"].to_numpy(), owners)]
    counts = sides.groupby("player").size()
    eligible = np.sort(counts.index[counts >= min_games].to_numpy())
    rng = np.random.default_rng([seed, 31])
    if max_players is not None and len(eligible) > max_players:
        eligible = np.sort(rng.choice(eligible, size=max_players, replace=False))
    sides = sides[sides["player"].isin(eligible)]
    picked = []
    for player, part in sides.groupby("player", sort=True):
        idx = part.index.to_numpy()
        if len(idx) > games_per_player:
            idx = rng.choice(idx, size=games_per_player, replace=False)
        picked.append(sides.loc[np.sort(idx)])
    out = pd.concat(picked, ignore_index=True)
    out["game"] = meta["game"].to_numpy()[out["row"].to_numpy()]
    return out.sort_values(["player", "game"]).reset_index(drop=True), {
        "games_kept": int(len(rows)), "eligible_players": int((counts >= min_games).sum()),
        "selected_players": int(len(eligible)), "selected_rows": int(len(out))}


def _game_moves(moves, shard: int, offset: int, length: int) -> np.ndarray:
    pts = np.asarray(moves[shard][offset:offset + length], np.int64)
    first_pass = np.flatnonzero(pts == PASS)
    return pts[:first_pass[0]] if len(first_pass) else pts


_STATE = {}


def _init(directory):
    directory, moves = Path(directory), []
    while (directory / f"moves_{len(moves):05d}.npy").exists():
        moves.append(np.load(directory / f"moves_{len(moves):05d}.npy", mmap_mode="r"))
    _STATE["moves"] = moves


def _write_shard(task):
    """Pack one shard of selected rows; returns its summary and per-row records."""
    from go_aicup.features.sgf_planes import moves_to_planes
    out, shard_id, src, colors, game_start, build_id = (task[k] for k in
                                                        ("out", "shard_id", "src", "colors", "game_start", "build_id"))
    games = [_game_moves(_STATE["moves"], int(a), int(b), int(c)) for a, b, c in src]
    lengths = np.array([len(g) for g in games], np.int64)
    total = int(lengths.sum())
    boards = np.empty((total, 2, PLANE_BYTES), np.uint8)
    all_moves = np.empty(total, np.int16)
    all_colors = np.empty(total, np.int8)
    records, offset = [], 0
    for i, (g, color) in enumerate(zip(games, colors)):
        c = (np.arange(len(g)) % 2).astype(np.int8)
        span = slice(offset, offset + len(g))
        boards[span] = pack_planes(moves_to_planes(g, c))
        all_moves[span], all_colors[span] = g, c
        records.append({"game_index": game_start + i, "shard_id": shard_id, "local_offset": offset,
                        "length": len(g), "player_color": int(color),
                        "player_ply_count": int(np.count_nonzero(c == color))})
        offset += len(g)
    prefix = Path(out) / f"shard_{shard_id:05d}"
    files = {}
    for name, array in ((f"{prefix.name}.npy", boards), (f"{prefix.name}_moves.npy", all_moves),
                        (f"{prefix.name}_colors.npy", all_colors)):
        np.save(Path(out) / name, array)
        files[name] = {"bytes": (Path(out) / name).stat().st_size, "sha256": file_hash(Path(out) / name),
                       "dtype": str(array.dtype), "shape": list(array.shape)}
    summary = {"format_version": 1, "build_id": build_id, "shard_id": shard_id, "rank": "external",
               "source_start": int(game_start), "game_start": int(game_start), "num_games": len(src),
               "total_ply": total, "player_ply_count": int(sum(r["player_ply_count"] for r in records)),
               "files": files, "seconds": 0.0}
    write_json(Path(out) / f"{prefix.name}.done.json", summary)
    return summary, records


def build(filtered: Path, out: Path, *, yugen: Path = DEFAULT_OUTPUT, workers: int = 12, **selection) -> dict:
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is nonempty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    meta, _, owners = load_filtered(filtered)
    chosen, info = select(meta, owners, **selection)
    base = json.loads((yugen / "manifest.json").read_text())
    base_games = pd.read_parquet(yugen / "games.parquet")
    n_base = len(base_games)
    # Yugen part: symlink every shard file and completion marker unchanged
    for summary in base["shards"]:
        prefix = f"shard_{summary['shard_id']:05d}"
        for name in list(summary["files"]) + [f"{prefix}.done.json"]:
            os.symlink((yugen / name).resolve(), out / name)
    first_id = max(s["shard_id"] for s in base["shards"]) + 1
    tasks = []
    for j, start in enumerate(range(0, len(chosen), GAMES_PER_SHARD)):
        part = chosen.iloc[start:start + GAMES_PER_SHARD]
        r = part["row"].to_numpy()
        src = np.stack([meta["shard"].to_numpy()[r], meta["offset"].to_numpy()[r], meta["length"].to_numpy()[r]], 1)
        tasks.append({"out": str(out), "shard_id": first_id + j, "src": src,
                      "colors": part["color"].to_numpy(), "game_start": n_base + start, "build_id": base["build_id"]})
    with Pool(workers, initializer=_init, initargs=(str(filtered),)) as pool:
        results = pool.map(_write_shard, tasks, chunksize=1)
    summaries = [s for s, _ in results]
    ext = pd.DataFrame([r for _, rs in results for r in rs])
    ext["rank_id"] = np.uint8(0)
    ext["rank"] = "OGS"
    ext["player_id"] = "ogs_" + chosen["player"].astype(str).to_numpy()
    ext["game_id"] = "ogs_" + chosen["game"].astype(str).to_numpy()
    ext["color"] = np.where(ext["player_color"] == 0, "B", "W")
    ext["source_row"] = np.int32(-1)
    for column in base_games.columns:                 # columns the external rows do not have
        if column not in ext:
            ext[column] = -1 if base_games[column].dtype.kind in "iu" else ""
    games = pd.concat([base_games, ext[base_games.columns]], ignore_index=True)
    for column, dtype in base_games.dtypes.items():
        games[column] = games[column].astype(dtype)
    assert np.array_equal(games["game_index"].to_numpy(), np.arange(len(games)))
    games.to_parquet(out / "games.parquet", index=False)
    pd.DataFrame({"game_index": ext["game_index"], "player": ext["player_id"]}).to_parquet(
        out / "external_players.parquet", index=False)
    manifest = {**base, "shards": base["shards"] + summaries, "num_games": len(games),
                "total_ply": int(base["total_ply"] + sum(s["total_ply"] for s in summaries)),
                "player_ply_count": int(base["player_ply_count"] + sum(s["player_ply_count"] for s in summaries)),
                "games": {"file": "games.parquet", "bytes": (out / "games.parquet").stat().st_size,
                          "sha256": file_hash(out / "games.parquet")},
                "external": {"source": str(filtered), "yugen": str(yugen), "selection": selection, **info}}
    write_json(out / "manifest.json", manifest)
    report = {**info, "selection": selection, "external_rows": int(len(ext)),
              "external_plies": int(ext["length"].sum()), "bytes": int(sum(
                  f["bytes"] for s in summaries for f in s["files"].values()))}
    write_json(out / "external.json", report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--filtered", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--yugen", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-games", type=int, default=20)
    p.add_argument("--max-players", type=int, default=None)
    p.add_argument("--games-per-player", type=int, default=40)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--min-rank", type=float, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--human-opponent", action="store_true", help="also require the opponent to own a folder")
    p.add_argument("--count-only", action="store_true", help="print the selection counts and stop")
    args = parser.parse_args(argv)
    selection = dict(min_games=args.min_games, max_players=args.max_players, games_per_player=args.games_per_player,
                     seed=args.seed, min_year=args.min_year, min_rank=args.min_rank,
                     human_opponent=args.human_opponent)
    if args.count_only:
        meta, _, owners = load_filtered(args.filtered)
        print(json.dumps(select(meta, owners, **selection)[1]))
        return 0
    print(json.dumps(build(args.filtered, args.out, yugen=args.yugen, workers=args.workers, **selection), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
