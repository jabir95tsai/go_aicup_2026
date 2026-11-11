"""Audit (CLAUDE.md §4): do rank-test games appear in the training set?

Exact check on the move sequence (SGF properties such as HD ignored), plus a
long-prefix check (first 60 moves) to catch truncated/extended copies. Reports
per-question hit counts and, for hits, whether all hits in a question point to
one training player/rank.

    python -m scripts.strength.rank_test_duplicates
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MOVES = re.compile(r";([BW])\[([a-s]{0,2})\]")
PREFIX = 60
csv.field_size_limit(sys.maxsize)


def keys(sgf: str) -> tuple[str, str]:
    moves = "".join(c + m for c, m in MOVES.findall(sgf))
    prefix = "".join(c + m for c, m in MOVES.findall(sgf)[:PREFIX])
    return hashlib.blake2b(moves.encode(), digest_size=12).hexdigest(), hashlib.blake2b(prefix.encode(), digest_size=12).hexdigest()


def main() -> None:
    exact, pref = defaultdict(list), defaultdict(list)
    for path in sorted((ROOT / "data/raw/training").glob("train_*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                e, p = keys(row["sgf_content"])
                info = (row["rank"], row["player_id"], row["game_id"], row["color"])
                exact[e].append(info)
                pref[p].append(info)
    report = {}
    for part in ("public", "private"):
        rows = list(csv.DictReader((ROOT / f"data/raw/tests/rank_prediction_test_{part}.csv").open(newline="")))
        games = hit_exact = hit_prefix = 0
        questions = []
        for row in rows:
            n = int(row["num_games"])
            ex_hits, pr_hits = [], []
            for i in range(1, n + 1):
                e, p = keys(row[f"sgf_{i}"])
                games += 1
                if e in exact:
                    hit_exact += 1
                    ex_hits.append(exact[e])
                if p in pref:
                    hit_prefix += 1
                    pr_hits.append(pref[p])
            if ex_hits:
                ranks = sorted({info[0] for hits in ex_hits for info in hits})
                players = sorted({info[1] for hits in ex_hits for info in hits})
                questions.append(dict(question_id=row["question_id"], k=n, exact_games=len(ex_hits),
                                      prefix_games=len(pr_hits), ranks=ranks, players=players[:5]))
        report[part] = dict(games=games, exact_hit_games=hit_exact, prefix60_hit_games=hit_prefix,
                            questions_with_exact_hits=len(questions), examples=questions[:20])
        print(part, json.dumps({k: v for k, v in report[part].items() if k != "examples"}))
    # Null for the prefix check: how often do two different training games share 60-move prefixes?
    shared = sum(1 for v in pref.values() if len(v) > 1)
    report["train_prefix60_shared_keys"] = shared
    print("train keys sharing a 60-move prefix:", shared, "of", len(pref))
    out = ROOT / "outputs/rank/test_duplicates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
