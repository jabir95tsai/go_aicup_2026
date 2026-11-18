"""Stream-filter the OGS 2025 by-username SGF archive (A0 of the external-data plan, 2026-11-18).

Reads sgfs-by-username.tar.gz from stdin (e.g. tail -c +1 -F archive | python -m
scripts.identity.ogs_stream_filter OUT). Member names are
sgfs-by-username/<p>/<owner>/<game>-<black>-<white>.sgf; bots have no folder of their own (archive
README), so a player is human iff it owns a folder (checked after the pass with owners.npy).

Kept (each game once, by game id): 19x19, no setup stones, handicap <= 1, finished (RE not "?"/Void),
>= 60 moves, strictly alternating from black, no passes in the first 60 plies, ranked. Written in shards to OUT:
moves_XXXXX.npy (int16 row*19+col, pass 361, games back to back) and meta_XXXXX.npy (structured rows).
Counters and per-owner eligible-game counts go to OUT/stats.json at every shard and at the end; a
truncated stream still writes everything read so far ("truncated": 1).
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

NAME = re.compile(r"sgfs-by-username/[^/]+/(\d+)/(\d+)-(\d+)-(\d+)\.sgf$")
PROP = re.compile(r"(SZ|HA|RE|GC|DT|BR|WR)\[([^\]]*)\]")
MOVE = re.compile(r";([BW])\[([a-t]{2})?\]")
SPEED = {"live": 0, "blitz": 1, "correspondence": 2, "rapid": 3}
META = np.dtype([("game", "i8"), ("black", "i8"), ("white", "i8"), ("year", "i2"), ("speed", "i1"),
                 ("ranked", "i1"), ("ha", "i1"), ("br", "f4"), ("wr", "f4"), ("length", "i4"), ("offset", "i8")])
SHARD = 500_000
MAX_GAME = 80_000_000


def rank_value(s: str) -> float:
    """OGS-style: 30k -> 0, 1k -> 29, 1d -> 30, 9d -> 38; unknown -> nan."""
    m = re.match(r"(\d+)([kdp])", s or "")
    if not m:
        return float("nan")
    n, t = int(m.group(1)), m.group(2)
    return 30.0 - n if t == "k" else 29.0 + n if t == "d" else 38.0 + n


def parse(text: str):
    """-> (reason, None) for a rejected game, ("ok", (props, points)) otherwise."""
    props = dict(PROP.findall(text))
    if props.get("SZ") != "19":
        return "not19", None
    if "AB[" in text or "AW[" in text:
        return "setup", None
    if int(props.get("HA", "0") or 0) > 1:
        return "handicap", None
    if props.get("RE", "?") in ("?", "Void", ""):
        return "unfinished", None
    mv = MOVE.findall(text)
    if len(mv) < 60:
        return "short", None
    colors = np.fromiter((col == "W" for col, _ in mv), np.int8, len(mv))
    if not np.array_equal(colors, np.arange(len(mv)) % 2):
        return "not_alternating", None
    pts = np.array([(ord(p[1]) - 97) * 19 + (ord(p[0]) - 97) if p and p != "tt" else 361 for _, p in mv], np.int16)
    if (pts[:60] == 361).any():
        return "early_pass", None
    return "ok", (props, pts)


class Writer:
    def __init__(self, out: Path):
        self.out, self.shard, self.moves, self.meta, self.offset = out, 0, [], [], 0

    def add(self, row, pts):
        self.meta.append(row + (self.offset,))
        self.moves.append(pts)
        self.offset += len(pts)
        if len(self.meta) >= SHARD:
            self.flush()

    def flush(self):
        if self.meta:
            np.save(self.out / f"moves_{self.shard:05d}.npy", np.concatenate(self.moves))
            np.save(self.out / f"meta_{self.shard:05d}.npy", np.array(self.meta, dtype=META))
            self.shard += 1
            self.moves, self.meta, self.offset = [], [], 0


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    seen = np.zeros(MAX_GAME, bool)
    owners: set[int] = set()
    owner_eligible = Counter()          # owner -> eligible ranked games in own folder (opponent not yet checked)
    c = Counter()
    writer = Writer(out)
    started = time.time()

    def report(final: bool):
        hist = Counter(min(v, 200) for v in owner_eligible.values())
        stats = {"final": final, "seconds": round(time.time() - started), "counters": dict(c), "owners": len(owners),
                 "owners_ge20_ranked": sum(1 for v in owner_eligible.values() if v >= 20),
                 "owners_eligible_hist": {str(k): v for k, v in sorted(hist.items())}}
        (out / "stats.json").write_text(json.dumps(stats, indent=1))
        print(json.dumps({k: stats[k] for k in ("seconds", "owners", "owners_ge20_ranked")}
                         | {"members": c["members"], "kept": c["kept"]}), flush=True)

    tf = tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz")
    try:
        for member in tf:
            if not member.isfile():
                continue
            c["members"] += 1
            tf.members = []                     # a streaming TarFile still keeps every TarInfo otherwise
            if c["members"] % 2_000_000 == 0:
                report(False)
            m = NAME.search(member.name)
            if not m:
                c["bad_name"] += 1
                continue
            owner, game, black, white = map(int, m.groups())
            owners.add(owner)
            reason, got = parse(tf.extractfile(member).read().decode("utf-8", "replace"))
            c[reason] += 1
            if got is None:
                continue
            props, pts = got
            gc = props.get("GC", "").split(",")
            ranked = int("ranked" in gc)
            owner_eligible[owner] += ranked
            if not ranked:                      # counted above; only ranked games are stored (disk quota)
                c["unranked_not_stored"] += 1
                continue
            if game >= MAX_GAME or seen[game]:
                c["duplicate"] += 1
                continue
            seen[game] = True
            dt = props.get("DT", "")
            writer.add((game, black, white, int(dt[:4]) if dt[:4].isdigit() else -1,
                        next((v for k, v in SPEED.items() if k in gc), -1), ranked, int(props.get("HA", "0") or 0),
                        rank_value(props.get("BR", "")), rank_value(props.get("WR", "")), len(pts)), pts)
            c["kept"] += 1
    except (tarfile.ReadError, EOFError, OSError) as e:
        c["truncated"] = 1
        print(f"stream ended early: {e!r}", flush=True)
    writer.flush()
    np.save(out / "owners.npy", np.array(sorted(owners), np.int64))
    report(True)


if __name__ == "__main__":
    main()
