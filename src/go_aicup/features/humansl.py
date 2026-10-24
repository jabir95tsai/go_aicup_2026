"""KataGo HumanSL per-move log-likelihoods (RESULTS A-039).

For every game and humanSLProfile, one analysis-engine query covers all turns
with one visit and includePolicy; with the HumanSL net as the main model the
raw policy at turn t is p_profile(move | position, history), so the entry at
the played move is that profile's likelihood of the move. Pass-free games only
(the training set has none, A-002).

Output (one directory per shard): logp.npy float32 [total turns, profiles] in
game order, offsets.npy int64 [games + 1], done.npy bool [games, profiles],
meta.json (profiles, settings, throughput).

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


def cache_games(indices) -> list[tuple[np.ndarray, np.ndarray]]:
    """(moves, colors) for training-cache game_index values."""
    table = pd.read_parquet(CACHE / "games.parquet").set_index("game_index").loc[np.asarray(indices)]
    shards, out = {}, []
    for row in table.itertuples():
        if row.shard_id not in shards:
            prefix = CACHE / f"shard_{int(row.shard_id):05d}"
            shards[row.shard_id] = (np.load(f"{prefix}_moves.npy", mmap_mode="r"),
                                    np.load(f"{prefix}_colors.npy", mmap_mode="r"))
        moves, colors = shards[row.shard_id]
        span = slice(int(row.local_offset), int(row.local_offset + row.length))
        out.append((np.asarray(moves[span]), np.asarray(colors[span])))
    return out


def query(game: int, profile: int, moves, colors, name: str) -> bytes:
    return (json.dumps({
        "id": f"{game}:{profile}", "moves": [["BW"[c], gtp(m)] for m, c in zip(moves, colors)],
        "rules": RULES, "komi": KOMI, "boardXSize": 19, "boardYSize": 19,
        "analyzeTurns": list(range(len(moves))), "maxVisits": 1, "includePolicy": True,
        "overrideSettings": {"humanSLProfile": name, "ignorePreRootHistory": False}},
        separators=(",", ":")) + "\n").encode()


def run(games, profiles, out: Path, *, in_flight: int = 48, max_seconds: float | None = None,
        overrides: str | None = None, katago: Path = KATAGO, model: Path = MODEL, config: Path = CONFIG) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "humansl_logs").mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(m) for m, _ in games], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    logp = np.full((offsets[-1], len(profiles)), np.nan, dtype=np.float32)
    remaining = np.repeat(lengths[:, None], len(profiles), axis=1)
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
                        engine.stdin.write(query(g, p, moves, colors, name))
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
                    raise RuntimeError(f"katago error: {result}")
                if "warning" in result:
                    print("katago warning:", result, flush=True)
                    continue
                g, p = map(int, result["id"].split(":"))
                t = int(result["turnNumber"])
                probability = result["policy"][int(games[g][0][t])]
                if not probability > 0:
                    raise ValueError(f"game {g} turn {t}: played move has policy {probability}")
                logp[offsets[g] + t, p] = np.log(probability)
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
    np.save(out / "done.npy", done)
    meta = {"profiles": list(profiles), "rules": RULES, "komi": KOMI, "games": len(games),
            "games_complete": int(done.all(axis=1).sum()), "positions": evaluated, "seconds": round(seconds, 1),
            "positions_per_second": round(evaluated / seconds, 1), "model": model.name, "katago": str(katago)}
    meta["katago_returncode"] = engine.returncode
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if max_seconds is None and (engine.returncode != 0 or not done.all()):
        raise RuntimeError(f"katago exited with {engine.returncode}; {meta['games_complete']}/{len(games)} "
                           f"games complete (see {out / 'katago_stderr.log'})")
    return meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, required=True, help="parquet with a game_index column")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="first N games of this shard (throughput tests)")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--in-flight", type=int, default=48, help="queries (game x profile) outstanding")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--override-config", help="KataGo -override-config, e.g. numAnalysisThreads=512")
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.games)
    table = table.iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "games.parquet")
    meta = run(cache_games(table["game_index"]), args.profiles.split(","), args.out,
               in_flight=args.in_flight, max_seconds=args.max_seconds, overrides=args.override_config)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
