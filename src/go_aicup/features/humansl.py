"""KataGo HumanSL per-move log-likelihoods (RESULTS A-039).

For every game and humanSLProfile, one analysis-engine query covers all turns
with one visit and includePolicy; with the HumanSL net as the main model the
raw policy at turn t is p_profile(move | position, history), so the entry at
the played move is that profile's likelihood of the move. Pass-free games only
(the training set has none, A-002).

Output (one directory per shard): logp.npy float32 [total turns, profiles] in
game order, colors.npy int8 [total turns], offsets.npy int64 [games + 1],
done.npy bool [games, profiles], meta.json (profiles, settings, throughput).
Training data is strict (any engine error aborts); release files use
--allow-failures, which records rejected games as not done and continues.

    python -m go_aicup.features.humansl --games outputs/humansl_pilot/games.parquet \\
        --out outputs/humansl_pilot/shard_0 --shard 0 --num-shards 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.paths import PROCESSED, ROOT

CACHE = PROCESSED / "boards" / "strength_v1"
KATAGO = ROOT / "data" / "external" / "katago-v1.18.2" / "katago"
MODEL = ROOT / "data" / "external" / "humansl-v0" / "b18c384nbt-humanv0.bin.gz"
CONFIG = ROOT / "configs" / "strength" / "humansl_analysis.cfg"
PROFILES = ("rank_20k", "rank_15k", "rank_10k", "rank_7k", "rank_5k", "rank_3k", "rank_1k", "rank_1d",
            "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d", "rank_7d", "rank_9d")
COLUMNS = "ABCDEFGHJKLMNOPQRST"   # GTP skips I
RULES, KOMI = "japanese", 6.5     # Yugen games; komi is not in the SGF (A-001)


def gtp(move: int) -> str:
    row, col = divmod(int(move), 19)   # row 0 = top = GTP row 19
    return f"{COLUMNS[col]}{19 - row}"


def cache_games(indices, cache: Path = CACHE) -> list[tuple[np.ndarray, np.ndarray]]:
    """(moves, colors) for game_index values of a board cache (training or release)."""
    table = pd.read_parquet(cache / "games.parquet").set_index("game_index").loc[np.asarray(indices)]
    shards, out = {}, []
    for row in table.itertuples():
        if row.shard_id not in shards:
            prefix = cache / f"shard_{int(row.shard_id):05d}"
            shards[row.shard_id] = (np.load(f"{prefix}_moves.npy", mmap_mode="r"),
                                    np.load(f"{prefix}_colors.npy", mmap_mode="r"))
        moves, colors = shards[row.shard_id]
        span = slice(int(row.local_offset), int(row.local_offset + row.length))
        out.append((np.asarray(moves[span]), np.asarray(colors[span])))
    return out


def query(game: int, profile: int, moves, colors, name: str, komi: float = KOMI) -> bytes:
    return (json.dumps({
        "id": f"{game}:{profile}", "moves": [["BW"[c], gtp(m)] for m, c in zip(moves, colors)],
        "rules": RULES, "komi": komi, "boardXSize": 19, "boardYSize": 19,
        "analyzeTurns": list(range(len(moves))), "maxVisits": 1, "includePolicy": True,
        "overrideSettings": {"humanSLProfile": name, "ignorePreRootHistory": False}},
        separators=(",", ":")) + "\n").encode()


def run(games, profiles, out: Path, *, in_flight: int = 48, max_seconds: float | None = None,
        overrides: str | None = None, komi: float = KOMI, allow_failures: bool = False,
        katago: Path = KATAGO, model: Path = MODEL, config: Path = CONFIG) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "humansl_logs").mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(m) for m, _ in games], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    logp = np.full((offsets[-1], len(profiles)), np.nan, dtype=np.float32)
    remaining = np.repeat(lengths[:, None], len(profiles), axis=1)
    rejected = []
    slots, stop = threading.Semaphore(in_flight), threading.Event()
    with open(out / "katago_stderr.log", "wb") as stderr:
        command = [str(katago), "analysis", "-config", str(config), "-model", str(model)]
        if overrides:
            command += ["-override-config", overrides]
        engine = subprocess.Popen(command,
                                  cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                  bufsize=1 << 20)

        def feed():
            try:
                for g, (moves, colors) in enumerate(games):
                    for p, name in enumerate(profiles):
                        while not slots.acquire(timeout=1.0):
                            if stop.is_set():
                                return
                        if stop.is_set():
                            return
                        engine.stdin.write(query(g, p, moves, colors, name, komi))
                        engine.stdin.flush()
            finally:
                engine.stdin.close()

        writer = threading.Thread(target=feed, daemon=True)
        started = last = time.monotonic()
        evaluated = reported = 0
        writer.start()
        try:
            for line in engine.stdout:
                result = json.loads(line)
                if "error" in result:
                    if not allow_failures or "id" not in result:
                        raise RuntimeError(f"katago error: {result}")
                    rejected.append(result)   # whole query rejected; the game stays not done
                    slots.release()
                    continue
                if "warning" in result:
                    print("katago warning:", result, flush=True)
                    continue
                g, p = map(int, result["id"].split(":"))
                t = int(result["turnNumber"])
                probability = result["policy"][int(games[g][0][t])]
                if not probability > 0 and not allow_failures:
                    raise ValueError(f"game {g} turn {t}: played move has policy {probability}")
                logp[offsets[g] + t, p] = np.log(probability) if probability > 0 else np.nan
                evaluated += 1
                remaining[g, p] -= 1
                if remaining[g, p] == 0:
                    slots.release()
                now = time.monotonic()
                if now - last >= 30:
                    print(json.dumps({"positions": evaluated, "positions_per_second":
                                      round((evaluated - reported) / (now - last), 1),
                                      "games_done": int((remaining == 0).all(axis=1).sum())}), flush=True)
                    last, reported = now, evaluated
                if max_seconds is not None and now - started >= max_seconds:
                    stop.set()   # send no more; the engine drains sent queries, then exits on EOF
        finally:
            stop.set()
            if engine.poll() is None:
                engine.terminate()
            engine.wait(timeout=60)
            writer.join(timeout=10)
    seconds = time.monotonic() - started
    done = remaining == 0
    np.save(out / "logp.npy", logp)
    np.save(out / "offsets.npy", offsets)
    np.save(out / "colors.npy", np.concatenate([c for _, c in games]).astype(np.int8))
    np.save(out / "done.npy", done)
    meta = {"profiles": list(profiles), "rules": RULES, "komi": komi, "games": len(games),
            "games_complete": int(done.all(axis=1).sum()), "positions": evaluated, "seconds": round(seconds, 1),
            "positions_per_second": round(evaluated / seconds, 1), "model": model.name, "katago": str(katago)}
    meta["katago_returncode"] = engine.returncode
    meta["rejected_queries"] = len(rejected)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if rejected:
        (out / "rejected.json").write_text(json.dumps(rejected[:200], indent=1))
    if max_seconds is None and (engine.returncode != 0 or not (done.all() or allow_failures)):
        raise RuntimeError(f"katago exited with {engine.returncode}; {meta['games_complete']}/{len(games)} "
                           f"games complete (see {out / 'katago_stderr.log'})")
    return meta


FEATURE_SETS = ("mean", "phase", "vote", "full")
PHASES = (("open", 0, 60), ("mid", 60, 150), ("end", 150, None))   # ply ranges (A-046)


def side_features(values: np.ndarray, ply: np.ndarray, side: str, profiles, feature_set: str) -> dict:
    """Features of one player's moves; values [moves, profiles] log-likelihoods."""
    mean = values.mean(axis=0)
    out = {f"{side}_{p}": float(m) for p, m in zip(profiles, mean)}
    if feature_set in ("phase", "full"):
        for name, start, stop in PHASES:
            rows = (ply >= start) & (ply < stop if stop is not None else True)
            # A phase without moves (short games) takes the whole-game mean.
            phase = values[rows].mean(axis=0) if rows.any() else mean
            out.update({f"{side}_{name}_{p}": float(m) for p, m in zip(profiles, phase)})
    if feature_set in ("vote", "full"):
        z = values - values.max(axis=1, keepdims=True)
        vote = (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).mean(axis=0)
        out.update({f"{side}_vote_{p}": float(v) for p, v in zip(profiles, vote)})
    if feature_set == "full":
        out.update({f"{side}_std_{p}": float(v) for p, v in zip(profiles, values.std(axis=0))})
    return out


def shard_features(shard: Path, feature_set: str = "mean") -> pd.DataFrame:
    """Per complete game, features of the labelled player's (own_) and the
    opponent's (opp_) moves. "mean" (A-039..A-045): mean log-likelihood per
    profile; "phase" adds opening / middle / end means; "vote" adds the mean
    per-move softmax over profiles; "full" = phase + vote + per-profile std."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    table = pd.read_parquet(shard / "games.parquet")
    logp, offsets, done = (np.load(shard / f"{name}.npy") for name in ("logp", "offsets", "done"))
    colors = np.load(shard / "colors.npy") if (shard / "colors.npy").exists() else None
    profiles = json.loads((shard / "meta.json").read_text())["profiles"]
    rows = []
    for i, game in enumerate(table.itertuples()):
        values = logp[offsets[i]:offsets[i + 1]]
        if not done[i].all() or np.isnan(values).any():
            continue
        # Shards written before colors.npy existed are training games: black first, alternating (A-002).
        turn_colors = colors[offsets[i]:offsets[i + 1]] if colors is not None else np.arange(len(values)) % 2
        mine = turn_colors == game.player_color
        if not mine.any() or mine.all():
            continue
        ply = np.arange(len(values))
        rows.append({"game_index": game.game_index,
                     **side_features(values[mine], ply[mine], "own", profiles, feature_set),
                     **side_features(values[~mine], ply[~mine], "opp", profiles, feature_set)})
    return pd.DataFrame(rows)


def run_features(run: Path, feature_set: str = "mean") -> pd.DataFrame:
    """shard_features over every shard_* directory of a run."""
    parts = [shard_features(p, feature_set) for p in sorted(run.glob("shard_*")) if p.is_dir()]
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, required=True, help="parquet with game_index (and player_color)")
    parser.add_argument("--cache", type=Path, default=CACHE, help="board cache holding the games")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N games of this shard (throughput tests)")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--in-flight", type=int, default=48, help="queries (game x profile) outstanding")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--override-config", help="KataGo -override-config, e.g. numAnalysisThreads=512")
    parser.add_argument("--komi", type=float, default=KOMI)
    parser.add_argument("--allow-failures", action="store_true", help="release files: skip rejected games")
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.games)
    table = table.iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "games.parquet")
    meta = run(cache_games(table["game_index"], args.cache), args.profiles.split(","), args.out,
               in_flight=args.in_flight, max_seconds=args.max_seconds, overrides=args.override_config,
               komi=args.komi, allow_failures=args.allow_failures)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
