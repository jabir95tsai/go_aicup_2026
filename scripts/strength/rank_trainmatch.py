"""Do competition-2 test games appear elsewhere as the same game? (2026-11-21)

Day-0 only compared SGF strings verbatim. A test game can also be stored under the opponent's
perspective in the training set (same moves, the training row labels the opponent with a known rank),
D4-transformed, or appear in two test questions (two test players who played each other).
Key: the first P moves in all 8 D4 variants. Looked up against the 1M training games (raw.zip) and
against the other test games.

    python -m scripts.strength.rank_trainmatch
"""
import collections
import io
import json
import re
import zipfile

import pandas as pd

from go_aicup.paths import ROOT

P = 40
MV = re.compile(r';[BW]\[([a-s]{0,2})\]')
RANKS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
OUT = ROOT / 'outputs' / 'rank' / 'trainmatch'


def d4(seq, t):
    """D4 variant t (0-7) of a move list (same transform as scripts/identity/pid2_trainmatch.py)."""
    out = []
    for m in seq:
        if len(m) != 2:
            out.append(m)
            continue
        x, y = ord(m[0]) - 97, ord(m[1]) - 97
        for _ in range(t % 4):
            x, y = y, 18 - x
        if t >= 4:
            x = 18 - x
        out.append(chr(97 + x) + chr(97 + y))
    return tuple(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = {}
    with zipfile.ZipFile(ROOT / 'data' / 'raw' / 'raw.zip') as z:
        for r in RANKS:
            with z.open(f'raw/training/train_{r}.csv') as f:
                df = pd.read_csv(io.TextIOWrapper(f), usecols=['player_id', 'game_id', 'color', 'sgf_content'])
            for pid, gid, c, s in zip(df.player_id, df.game_id, df.color, df.sgf_content):
                m = MV.findall(s)
                if len(m) >= P:
                    train.setdefault(tuple(m[:P]), []).append((r, str(pid), str(gid), c))
            print(r, len(train), flush=True)
    rows, test_keys = [], collections.defaultdict(list)
    for part in ('public', 'private'):
        df = pd.read_csv(ROOT / 'data' / 'raw' / 'tests' / f'rank_prediction_test_{part}.csv')
        for _, row in df.iterrows():
            for i in range(1, 21):
                s = row.get(f'sgf_{i}')
                if not isinstance(s, str) or not s:
                    continue
                m = MV.findall(s)
                if len(m) < P:
                    continue
                key = tuple(m[:P])
                test_keys[key].append((row.question_id, i, row[f'color_{i}']))
                for t in range(8):
                    hit = train.get(d4(m[:P], t))
                    if hit:
                        for r, pid, gid, c in hit:
                            rows.append({'question_id': row.question_id, 'slot': i, 'test_color': row[f'color_{i}'],
                                         'd4': t, 'train_rank': r, 'train_player': pid, 'train_game': gid,
                                         'train_color': c, 'same_side': c == row[f'color_{i}']})
                        break
    hits = pd.DataFrame(rows)
    hits.to_csv(OUT / 'train_hits.csv', index=False)
    shared = [v for v in test_keys.values() if len({q for q, _, _ in v}) > 1]
    pd.DataFrame([{'games': v} for v in shared]).to_json(OUT / 'test_shared.json', orient='records')
    report = {'P': P, 'train_keys': len(train), 'test_games': int(sum(len(v) for v in test_keys.values())),
              'test_games_in_train': int(hits[['question_id', 'slot']].drop_duplicates().shape[0]) if len(hits) else 0,
              'questions_with_train_hit': int(hits['question_id'].nunique()) if len(hits) else 0,
              'hits_by_side': hits['same_side'].value_counts().to_dict() if len(hits) else {},
              'hits_by_d4': hits['d4'].value_counts().to_dict() if len(hits) else {},
              'test_games_shared_between_questions': len(shared)}
    (OUT / 'report.json').write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == '__main__':
    main()
