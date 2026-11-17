"""Do competition-1 games appear in the training set as D4-transformed or re-serialised copies?
(2026-11-17) Day-0 compared only verbatim SGF strings. Here: first P moves of every PID game,
all 8 D4 variants, looked up against the first P moves of all 1M training games (raw orientation).
Reports per file the share matched and, for the added set, how often a question's and a candidate's
matched training players coincide."""
import collections, json, re
import pandas as pd
from go_aicup.paths import ROOT

P = 40
MV = re.compile(r';[BW]\[([a-s]{0,2})\]')
T = ROOT / 'data/raw/training'
D = ROOT / 'data/raw/tests'
OUT = ROOT / 'outputs/pid/pid2_trainmatch'


def d4(seq, t):
    out = []
    for m in seq:
        if len(m) != 2:
            out.append(m); continue
        x, y = ord(m[0]) - 97, ord(m[1]) - 97
        for _ in range(t % 4):
            x, y = y, 18 - x
        if t >= 4:
            x = 18 - x
        out.append(chr(97 + x) + chr(97 + y))
    return tuple(out)


train = {}
dup = 0
for r in ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']:
    df = pd.read_csv(T / f'train_{r}.csv', usecols=['player_id', 'game_id', 'sgf_content'])
    for pid, gid, s in zip(df.player_id, df.game_id, df.sgf_content):
        m = MV.findall(s)
        if len(m) < P:
            continue
        k = tuple(m[:P])
        if k in train:
            dup += 1
        train.setdefault(k, []).append((r, pid, gid))
    print(r, len(train), flush=True)
report = {'P': P, 'train_keys': len(train), 'train_prefix_collisions': dup, 'files': {}}
links = collections.defaultdict(set)    # (file, owner) -> training players matched
for f, kind in [('candidates2_public', 'c'), ('candidates2_private', 'c'), ('test2_public', 't'),
                ('test2_private', 't'), ('candidates_public', 'c'), ('candidates_private', 'c'),
                ('test_public', 't'), ('test_private', 't')]:
    df = pd.read_csv(D / f'player_identification_{f}.csv')
    games = []
    if kind == 'c':
        games = list(zip(df.player_id, df.sgf_content))
    else:
        for _, row in df.iterrows():
            for i in range(1, 21):
                s = row.get(f'sgf_{i}')
                if isinstance(s, str) and s:
                    games.append((row.question_id, s))
    hit, by_t = 0, collections.Counter()
    for owner, s in games:
        m = MV.findall(s)
        if len(m) < P:
            continue
        for t in range(8):
            v = train.get(d4(m[:P], t))
            if v:
                hit += 1; by_t[t] += 1
                for r, pid, gid in v:
                    links[(f, owner)].add(pid)
                break
    report['files'][f] = {'games': len(games), 'matched': hit, 'by_d4': dict(by_t)}
    print(f, report['files'][f], flush=True)
for part in ('public', 'private'):
    cand = {o: p for (f, o), p in links.items() if f == f'candidates2_{part}'}
    ques = {o: p for (f, o), p in links.items() if f == f'test2_{part}'}
    linked = sum(1 for q, ps in ques.items() if any(ps & cp for cp in cand.values()))
    report[f'{part}_questions_linked_to_a_candidate'] = linked
(OUT / 'report.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != 'files'}))
