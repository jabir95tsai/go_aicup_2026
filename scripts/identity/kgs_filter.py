"""KGS high-dan archive (u-go.net) -> filtered shards in the ogs_stream_filter format (2026-11-23, user "可以 要拚";
external data approved by the user; declared in EXTERNAL_RESOURCES.md)

Input: the yearly / monthly KGS-*-19-*-.tar.gz files of https://u-go.net/gamerecords/ in RAW. Same game filter as
ogs_stream_filter.parse (19x19, no setup stones, handicap <= 1, finished, >= 60 moves, alternating from black, no pass
in the first 60 plies). KGS SGFs carry no ranked flag: every kept game is marked ranked. Players are keyed by name
(PB / PW) -> int id KGS_BASE + index; bot accounts (name pattern BOTS) are not owners, so with --human-opponent the
external build keeps only human-vs-human games. Game ids are KGS_GAME + running index (never collide with OGS ids).

    python -m scripts.identity.kgs_filter RAW OUT
"""
from __future__ import annotations

import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.identity.ogs_stream_filter import META, Writer, parse, rank_value

KGS_BASE = 10 ** 12
KGS_GAME = 10 ** 11
NAMES = re.compile(r"P([BW])\[([^\]]*)\]")
BOTS = re.compile(r"bot|gnu|zen|crazy|leela|pachi|aya|golois|kata|fuego|mogo|steenvreter|manyfaces|dolbaram|"
                  r"abacus|nice|orego|hira|gomorra|darkforest|alphago|elf|sai|minigo|lz", re.I)


def main() -> None:
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    writer = Writer(out)
    ids: dict[str, int] = {}
    c = Counter()
    gid = 0
    for path in sorted(raw.glob("KGS-*.tar.gz")):
        with tarfile.open(path, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".sgf"):
                    continue
                c["members"] += 1
                text = tf.extractfile(member).read().decode("utf-8", "replace")
                reason, got = parse(text)
                c[reason] += 1
                if got is None:
                    continue
                names = dict(NAMES.findall(text))
                b, w = names.get("B", ""), names.get("W", "")
                if not b or not w or b == w:
                    c["names"] += 1
                    continue
                props, pts = got
                key = [ids.setdefault(n, KGS_BASE + len(ids)) for n in (b, w)]
                dt = props.get("DT", "")
                writer.add((KGS_GAME + gid, key[0], key[1], int(dt[:4]) if dt[:4].isdigit() else -1, -1, 1,
                            int(props.get("HA", "0") or 0), rank_value(props.get("BR", "")),
                            rank_value(props.get("WR", "")), len(pts)), pts)
                gid += 1
                c["kept"] += 1
        print(path.name, dict(c), flush=True)
    writer.flush()
    humans = np.array(sorted(v for n, v in ids.items() if not BOTS.search(n)), np.int64)
    np.save(out / "owners.npy", humans)
    (out / "players.json").write_text(json.dumps({"players": len(ids), "humans": int(len(humans)),
                                                  "bots": sorted(n for n in ids if BOTS.search(n))[:200]}, indent=1))
    (out / "stats.json").write_text(json.dumps({"counters": dict(c), "shards": writer.shard}, indent=1))
    print(json.dumps({"counters": dict(c), "players": len(ids), "humans": int(len(humans))}), flush=True)


if __name__ == "__main__":
    main()
