"""Strong-engine move-quality features with KataGo (RESULTS A-048).

One analysis-engine query per game covers every position 0..n (n moves, so the
position after the last move too), one visit, includePolicy, with the kata1
b18c384nbt network. Per position the engine reports Black's score lead and
winrate (reportAnalysisWinratesAs = BLACK) and the policy; the move from
position t is scored by the change to position t + 1 from the mover's side.

Output per shard: score.npy / winrate.npy float32 [positions], logp.npy and
top1.npy (played move's policy log-prob, whether it is the policy argmax;
NaN / 0 at each game's final position), offsets.npy int64 [games + 1] over
positions (n + 1 per game), colors.npy int8 [moves], done.npy bool [games],
meta.json.

    python -m go_aicup.features.katago_engine --games G.parquet --out OUT/shard_0
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

from go_aicup.features.humansl import CACHE, CONFIG, KATAGO, KOMI, RULES, cache_games, gtp
from go_aicup.paths import ROOT

MODEL = ROOT / "data" / "external" / "kata1-b18c384nbt-s9996604416" / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
LOSS_CLIP = 20.0                      # points; larger swings are capped (decided games, blunders)
LOSS_THRESHOLDS = (1.0, 3.0, 7.0)    # points


def query(game: int, moves, colors, komi: float) -> bytes:
    return (json.dumps({
        "id": str(game), "moves": [["BW"[c], gtp(m)] for m, c in zip(moves, colors)],
        "rules": RULES, "komi": komi, "boardXSize": 19, "boardYSize": 19,
        "analyzeTurns": list(range(len(moves) + 1)), "maxVisits": 1, "includePolicy": True},
        separators=(",", ":")) + "\n").encode()


def run(games, out: Path, *, in_flight: int = 48, komi: float = KOMI, allow_failures: bool = False,
        overrides: str | None = None, katago: Path = KATAGO, model: Path = MODEL, config: Path = CONFIG) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "rank" / "humansl" / "humansl_logs").mkdir(parents=True, exist_ok=True)
    positions = np.array([len(m) + 1 for m, _ in games], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(positions)]
    score, winrate, logp = (np.full(offsets[-1], np.nan, dtype=np.float32) for _ in range(3))
    top1 = np.zeros(offsets[-1], dtype=np.int8)
    remaining, rejected = positions.copy(), []
    slots = threading.Semaphore(in_flight)
    with open(out / "katago_stderr.log", "wb") as stderr:
        command = [str(katago), "analysis", "-config", str(config), "-model", str(model)]
        if overrides:
            command += ["-override-config", overrides]
        engine = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=stderr, bufsize=1 << 20)

        def feed():
            try:
                for g, (moves, colors) in enumerate(games):
                    slots.acquire()
                    engine.stdin.write(query(g, moves, colors, komi))
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
                    rejected.append(result)
                    slots.release()
                    continue
                if "warning" in result:
                    print("katago warning:", result, flush=True)
                    continue
                g, t = int(result["id"]), int(result["turnNumber"])
                row = offsets[g] + t
                score[row] = result["rootInfo"]["scoreLead"]
                winrate[row] = result["rootInfo"]["winrate"]
                moves = games[g][0]
                if t < len(moves):
                    policy = np.asarray(result["policy"], dtype=np.float64)
                    probability = policy[int(moves[t])]
                    if not probability > 0 and not allow_failures:
                        raise ValueError(f"game {g} turn {t}: played move has policy {probability}")
                    logp[row] = np.log(max(probability, 1e-6))
                    top1[row] = int(np.argmax(policy) == int(moves[t]))
                evaluated += 1
                remaining[g] -= 1
                if remaining[g] == 0:
                    slots.release()
                now = time.monotonic()
                if now - last >= 30:
                    print(json.dumps({"positions": evaluated, "positions_per_second":
                                      round((evaluated - reported) / (now - last), 1),
                                      "games_done": int((remaining == 0).sum())}), flush=True)
                    last, reported = now, evaluated
        finally:
            if engine.poll() is None:
                engine.terminate()
            engine.wait(timeout=60)
            writer.join(timeout=10)
    seconds = time.monotonic() - started
    done = remaining == 0
    for name, array in (("score", score), ("winrate", winrate), ("logp", logp), ("top1", top1),
                        ("offsets", offsets), ("done", done),
                        ("colors", np.concatenate([c for _, c in games]).astype(np.int8))):
        np.save(out / f"{name}.npy", array)
    meta = {"model": model.name, "rules": RULES, "komi": komi, "games": len(games),
            "games_complete": int(done.sum()), "positions": evaluated, "seconds": round(seconds, 1),
            "positions_per_second": round(evaluated / seconds, 1), "katago_returncode": engine.returncode,
            "rejected_queries": len(rejected)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if engine.returncode != 0 or not (done.all() or allow_failures):
        raise RuntimeError(f"katago exited with {engine.returncode}; {meta['games_complete']}/{len(games)} complete")
    return meta


def side_features(loss, wr_loss, logp, top1, side: str) -> dict:
    loss = np.clip(loss, 0.0, LOSS_CLIP)
    out = {f"{side}_eng_loss": float(loss.mean()), f"{side}_eng_wrloss": float(np.clip(wr_loss, 0.0, 1.0).mean()),
           f"{side}_eng_logp": float(logp.mean()), f"{side}_eng_top1": float(top1.mean())}
    out.update({f"{side}_eng_gt{int(t)}": float((loss > t).mean()) for t in LOSS_THRESHOLDS})
    return out


def shard_features(shard: Path) -> pd.DataFrame:
    """Per complete game: own_eng_* / opp_eng_* move-quality features (7 per side)."""
    table = pd.read_parquet(shard / "games.parquet")
    arrays = {name: np.load(shard / f"{name}.npy") for name in
              ("score", "winrate", "logp", "top1", "offsets", "done", "colors")}
    offsets, rows = arrays["offsets"], []
    move_offsets = offsets - np.arange(len(offsets))   # n moves per game of n + 1 positions
    for i, game in enumerate(table.itertuples()):
        if not arrays["done"][i]:
            continue
        span = slice(offsets[i], offsets[i + 1])
        score, winrate = arrays["score"][span].astype(np.float64), arrays["winrate"][span].astype(np.float64)
        colors = arrays["colors"][move_offsets[i]:move_offsets[i + 1]]
        sign = np.where(colors == 0, 1.0, -1.0)            # Black-perspective values -> mover's side
        loss = sign * (score[:-1] - score[1:])
        wr_loss = sign * (winrate[:-1] - winrate[1:])
        logp, top1 = arrays["logp"][span][:-1], arrays["top1"][span][:-1]
        mine = colors == game.player_color
        if not mine.any() or mine.all():
            continue
        rows.append({"game_index": game.game_index,
                     **side_features(loss[mine], wr_loss[mine], logp[mine], top1[mine], "own"),
                     **side_features(loss[~mine], wr_loss[~mine], logp[~mine], top1[~mine], "opp")})
    return pd.DataFrame(rows)


def run_features(run: Path) -> pd.DataFrame:
    return pd.concat([shard_features(p) for p in sorted(run.glob("shard_*")) if p.is_dir()], ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, required=True, help="parquet with game_index and player_color")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--in-flight", type=int, default=48, help="games outstanding")
    parser.add_argument("--komi", type=float, default=KOMI)
    parser.add_argument("--override-config")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.games).iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "games.parquet")
    meta = run(cache_games(table["game_index"], args.cache), args.out, in_flight=args.in_flight, komi=args.komi,
               allow_failures=args.allow_failures, overrides=args.override_config)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
