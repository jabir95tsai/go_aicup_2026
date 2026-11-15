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
from numba import njit

from go_aicup.features.humansl import CACHE, CONFIG, KATAGO, KOMI, RULES, cache_games, gtp
from go_aicup.paths import ROOT

MODEL = ROOT / "data" / "external" / "kata1-b18c384nbt-s9996604416" / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"
LOSS_CLIP = 20.0                      # points; larger swings are capped (decided games, blunders)
LOSS_THRESHOLDS = (1.0, 3.0, 7.0)    # points


DEAD = 0.5                            # |ownership| above this = settled
LIFE_COLUMNS = ("dead_b", "dead_w", "unsettled_b", "unsettled_w", "captured_b", "captured_w")
LIFE_JUMP, LIFE_CLIP = 5, 30         # stones


def query(game: int, moves, colors, komi: float, visits: int = 1, ownership: bool = False) -> bytes:
    request = {
        "id": str(game), "moves": [["BW"[c], gtp(m)] for m, c in zip(moves, colors)],
        "rules": RULES, "komi": komi, "boardXSize": 19, "boardYSize": 19,
        "analyzeTurns": list(range(len(moves) + 1)), "maxVisits": visits, "includePolicy": True}
    if ownership:
        request["includeOwnership"] = True
    return (json.dumps(request, separators=(",", ":")) + "\n").encode()


@njit(cache=True)
def replay_boards(moves, colors):
    """int8 [n + 1, 361]: the board before each move and after the last (+1 black, -1 white),
    captures removed. Row-major from the top-left, as moves and KataGo's ownership."""
    n = len(moves)
    out = np.zeros((n + 1, 361), dtype=np.int8)
    board = np.zeros(361, dtype=np.int8)
    stack = np.empty(361, dtype=np.int64)
    seen = np.zeros(361, dtype=np.int64)
    stamp = 0
    for t in range(n):
        move = int(moves[t])
        stone = 1 if colors[t] == 0 else -1
        board[move] = stone
        row, col = move // 19, move % 19
        for d in range(4):
            r = row + (-1 if d == 0 else 1 if d == 1 else 0)
            c = col + (-1 if d == 2 else 1 if d == 3 else 0)
            if r < 0 or r > 18 or c < 0 or c > 18 or board[r * 19 + c] != -stone:
                continue
            stamp += 1
            size, top, free = 0, 1, False
            stack[0] = r * 19 + c
            seen[r * 19 + c] = stamp
            while size < top:
                point = stack[size]
                size += 1
                pr, pc = point // 19, point % 19
                for e in range(4):
                    qr = pr + (-1 if e == 0 else 1 if e == 1 else 0)
                    qc = pc + (-1 if e == 2 else 1 if e == 3 else 0)
                    if qr < 0 or qr > 18 or qc < 0 or qc > 18:
                        continue
                    q = qr * 19 + qc
                    if board[q] == 0:
                        free = True
                    elif board[q] == -stone and seen[q] != stamp:
                        seen[q] = stamp
                        stack[top] = q
                        top += 1
            if not free:
                for i in range(top):
                    board[stack[i]] = 0
        out[t + 1] = board
    return out


def life_row(board: np.ndarray, placed_b: int, placed_w: int, owner: np.ndarray) -> tuple:
    """LIFE_COLUMNS of one position; owner > 0 = Black's (reportAnalysisWinratesAs = BLACK)."""
    black, white = board == 1, board == -1
    settled = np.abs(owner) > DEAD
    return (int((black & (owner < -DEAD)).sum()), int((white & (owner > DEAD)).sum()),
            int((black & ~settled).sum()), int((white & ~settled).sum()),
            placed_b - int(black.sum()), placed_w - int(white.sum()))


def run(games, out: Path, *, in_flight: int = 48, komi: float = KOMI, allow_failures: bool = False,
        overrides: str | None = None, katago: Path = KATAGO, model: Path = MODEL, config: Path = CONFIG,
        visits: int = 1, ownership: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "humansl_logs").mkdir(parents=True, exist_ok=True)
    positions = np.array([len(m) + 1 for m, _ in games], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(positions)]
    score, winrate, logp = (np.full(offsets[-1], np.nan, dtype=np.float32) for _ in range(3))
    top1 = np.zeros(offsets[-1], dtype=np.int8)
    life = np.zeros((offsets[-1], len(LIFE_COLUMNS)), dtype=np.int16) if ownership else None
    boards = {}
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
                    if ownership:
                        black = np.r_[0, np.cumsum(np.asarray(colors) == 0)]
                        boards[g] = (replay_boards(np.asarray(moves, dtype=np.int64),
                                                   np.asarray(colors, dtype=np.int64)), black)
                    engine.stdin.write(query(g, moves, colors, komi, visits, ownership))
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
                if ownership:
                    board, black = boards[g]
                    life[row] = life_row(board[t], int(black[t]), t - int(black[t]),
                                         np.asarray(result["ownership"], dtype=np.float32))
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
                    boards.pop(g, None)
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
    if ownership:
        np.save(out / "life.npy", life)
    meta = {"model": model.name, "ownership": ownership, "rules": RULES, "komi": komi, "visits": visits, "overrides": overrides, "games": len(games),
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


def life_features(d_own, d_opp, unsettled, side: str) -> dict:
    """Life-and-death events of one side's moves. d_own / d_opp: change over the move in that
    side's / the other side's lost stones (dead on the board + captured)."""
    return {f"{side}_life_die": float(np.clip(d_own, 0, LIFE_CLIP).mean()),
            f"{side}_life_die{LIFE_JUMP}": float((d_own >= LIFE_JUMP).mean()),
            f"{side}_life_letlive": float(np.clip(-d_opp, 0, LIFE_CLIP).mean()),
            f"{side}_life_letlive{LIFE_JUMP}": float((d_opp <= -LIFE_JUMP).mean()),
            f"{side}_life_unsettled": float(unsettled.mean())}


def shard_features(shard: Path) -> pd.DataFrame:
    """Per complete game: own_eng_* / opp_eng_* move-quality features (7 per side), plus
    own_life_* / opp_life_* (5 per side) when the shard was run with --ownership."""
    table = pd.read_parquet(shard / "games.parquet")
    arrays = {name: np.load(shard / f"{name}.npy") for name in
              ("score", "winrate", "logp", "top1", "offsets", "done", "colors")}
    life_all = np.load(shard / "life.npy").astype(np.float32) if (shard / "life.npy").exists() else None
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
        row = {"game_index": game.game_index,
               **side_features(loss[mine], wr_loss[mine], logp[mine], top1[mine], "own"),
               **side_features(loss[~mine], wr_loss[~mine], logp[~mine], top1[~mine], "opp")}
        if life_all is not None:
            life = life_all[span]
            lost = np.stack([life[:, 0] + life[:, 4], life[:, 1] + life[:, 5]], axis=1)   # black, white
            delta = lost[1:] - lost[:-1]
            mover = colors.astype(np.int64)
            steps = np.arange(len(mover))
            d_own, d_opp = delta[steps, mover], delta[steps, 1 - mover]
            unsettled = life[:-1, 2:4][steps, mover]
            row |= life_features(d_own[mine], d_opp[mine], unsettled[mine], "own")
            row |= life_features(d_own[~mine], d_opp[~mine], unsettled[~mine], "opp")
        rows.append(row)
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
    parser.add_argument("--visits", type=int, default=1, help="maxVisits per position (A-048 used 1)")
    parser.add_argument("--ownership", action="store_true", help="also write life.npy (LIFE_COLUMNS per position)")
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.games).iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "games.parquet")
    meta = run(cache_games(table["game_index"], args.cache), args.out, in_flight=args.in_flight, komi=args.komi,
               allow_failures=args.allow_failures, overrides=args.override_config, visits=args.visits,
               ownership=args.ownership)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
