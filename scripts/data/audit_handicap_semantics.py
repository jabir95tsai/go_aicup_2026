"""Check the official Q&A Q13 handicap semantics against the v1.1 training CSVs.

Two read-only tests (no model, no labels used as features):
1. Move labels reversed in HD>=2 games: if the first SGF "B" move were really
   black's (no stones on the board), it would often sit on a star point such as
   pd. If the stones are on the Q13 star points and the first move is white's,
   none of the early plies can land on those points.
2. The `color` column is the real colour: in HD>=2 games the stronger player
   gives the stones and plays white, so the share of color == B should fall
   for the top rank. A reversed column would show the opposite.

    python -m scripts.data.audit_handicap_semantics --out outputs/rank_handicap_audit_v11/semantics.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd

from go_aicup.paths import ROOT

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
# Q&A Q13 standard star points, SGF coordinates.
STONES = {2: "pd dp", 3: "pd dp dd", 4: "pd dp dd pp", 5: "pd dp dd pp jj",
          6: "pd dp dd pp dj pj", 7: "pd dp dd pp dj pj jj",
          8: "pd dp dd pp dj pj jd jp", 9: "pd dp dd pp dj pj jd jp jj"}
EARLY = 10
HD_RE = re.compile(r"HD\[(\d+)\]")
MOVE_RE = re.compile(r";([BW])\[([a-s]{0,2})\]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=ROOT / "data" / "raw" / "training")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    games = collections.Counter()
    early_on_stone = collections.Counter()
    first_moves = collections.defaultdict(collections.Counter)
    color_b = collections.defaultdict(lambda: [0, 0])  # (group, rank) -> [n_B, n]
    for rank in RANKS:
        frame = pd.read_csv(args.train_dir / f"train_{rank}.csv", usecols=["color", "sgf_content"])
        for color, sgf in zip(frame["color"], frame["sgf_content"]):
            hd = int(HD_RE.search(sgf).group(1))
            group = "HD0" if hd == 0 else "HD1" if hd == 1 else "HD2+"
            color_b[group, rank][0] += color == "B"
            color_b[group, rank][1] += 1
            moves = MOVE_RE.findall(sgf)
            games[hd] += 1
            first_moves[hd][f"{moves[0][0]}[{moves[0][1]}]"] += 1
            if hd >= 2 and any(m[1] in STONES[hd].split() for m in moves[:EARLY]):
                early_on_stone[hd] += 1

    report = {
        "games_by_hd": {str(h): games[h] for h in sorted(games)},
        f"hd2plus_games_with_a_move_on_q13_stone_points_in_first_{EARLY}_plies":
            {str(h): early_on_stone[h] for h in sorted(games) if h >= 2},
        "top_first_moves_by_hd": {str(h): first_moves[h].most_common(5) for h in sorted(games)},
        "color_B_share": {g: {r: round(color_b[g, r][0] / color_b[g, r][1], 4) for r in RANKS}
                          for g in ("HD0", "HD1", "HD2+")},
        "games_by_group_rank": {g: {r: color_b[g, r][1] for r in RANKS} for g in ("HD0", "HD1", "HD2+")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
