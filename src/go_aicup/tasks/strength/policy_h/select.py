"""Rebuild the C five-arm TRAIN/monitor/dev selection for the H policy cache.

Same rule, seed and cv.py helpers as scripts/strength/select_cached_policy.py; the
only difference is the handicap source (data/raw/raw.zip after the 11/18 move).
The C selection lived inside its policy ZIP, which is no longer on disk, so the
result must reproduce the hashes recorded by the C five-arm run 796606 byte for
byte; otherwise nothing is published.

    python -m go_aicup.tasks.strength.policy_h.select
"""
from pathlib import Path
import json
import shutil
import tempfile
import zipfile

import pandas as pd

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.cv import make_cached_policy_games, make_rank_questions
from go_aicup.paths import ROOT

BASE = ROOT / 'data/processed/strength_features/humansl'
OUT = BASE / 'policy_cnn_h/cached_AB_mondev_seed2027_h9'
RAW = ROOT / 'data/raw/raw.zip'
# Selection hashes recorded by the C five-arm run (outputs/.../five_arms/796606/identity.json).
C_HASHES = {
    'games.parquet': '42760b5816ddd5dd596268135871a143ef4daef665acea61c843a62cbbee495f',
    'train_questions.parquet': '85049419d9beba239f51a95e075a0a2fbcd83d17e1ffa8d1ccb7b55a656c0463',
    'monitor_questions.parquet': 'fe10441f254f55762bce710f2ce1821b41b52d964e9e351c4f1d23e76541b66b',
}
EXPECTED = {'train': (12451, 7549, 1683401), 'monitor': (2536, 1464, 324192), 'dev': (1419, 5781, 1286332)}


def main():
    seed = 2027
    if OUT.exists():
        raise FileExistsError(f'refusing to overwrite {OUT}')
    canonical = ROOT / 'data/processed/reproduce_paper/seed2026/games.parquet'
    meta = pd.read_parquet(canonical, columns=[
        'game_index', 'rank', 'game_id', 'rank_id', 'player_id', 'player_color', 'length'])
    with zipfile.ZipFile(RAW) as archive, archive.open('raw/training/handicap.csv') as stream:
        hd = pd.read_csv(stream, usecols=['rank', 'game_id', 'handicap'])
    meta = meta.merge(hd, on=['rank', 'game_id'], validate='one_to_one')
    c4x = BASE / 'c4x'
    paths = [c4x / f'{name}_hd{h}.parquet' for name in ('learn_A', 'learn_B', 'monitor') for h in (0, 1)]
    sources = [pd.read_parquet(p).assign(handicap=h) for p, h in zip(paths, [0, 1] * 3)]
    games = make_cached_policy_games(meta, pd.concat(sources[:4], ignore_index=True),
                                     pd.concat(sources[4:], ignore_index=True), seed=seed)
    games = games[['game_index', 'rank_id', 'player_id', 'player_color', 'handicap',
                   'length', 'role', 'color', 'reuse_pack']]
    question_sets, report = {}, {}
    for offset, (role, count) in enumerate([('train', 8000), ('monitor', 4000), ('dev', 8000)]):
        part = games[games.role.eq(role)].set_index('game_index', drop=False)
        questions = make_rank_questions(part, part.player_id.unique(), count, seed=seed + 10 + offset)
        question_sets[role] = questions
        report[role] = dict(players=int(part.player_id.nunique()), games=len(part),
                            positions=int(part.length.sum()), questions=len(questions),
                            reused_games=int(part.reuse_pack.sum()),
                            supplement_games=int((~part.reuse_pack).sum()),
                            supplement_positions=int(part.loc[~part.reuse_pack, 'length'].sum()))
        actual = tuple(report[role][k] for k in ('reused_games', 'supplement_games', 'supplement_positions'))
        if actual != EXPECTED[role]:
            raise ValueError(f'{role}: selection differs from plan: {actual} != {EXPECTED[role]}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=OUT.parent) as tmp:
        stage = Path(tmp) / OUT.name
        stage.mkdir()
        games.to_parquet(stage / 'games.parquet', index=False)
        games[games.reuse_pack].to_parquet(stage / 'reuse_games.parquet', index=False)
        games[~games.reuse_pack].to_parquet(stage / 'supplement_games.parquet', index=False)
        for role, questions in question_sets.items():
            questions.to_parquet(stage / f'{role}_questions.parquet', index=False)
        mismatch = {name: file_hash(stage / name) for name, digest in C_HASHES.items()
                    if file_hash(stage / name) != digest}
        if mismatch:
            raise ValueError(f'selection does not reproduce the C five-arm hashes: {mismatch}')
        identity = dict(seed=seed, reproduces='C five-arm selection (796606 selection_hashes)',
                        source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in
                                       [canonical, *paths, ROOT / 'src/go_aicup/data/cv.py', Path(__file__)]},
                        handicap_archive_member='data/raw/raw.zip:raw/training/handicap.csv',
                        handicap_table_hash=pd.util.hash_pandas_object(hd, index=False).sum().item(),
                        artifact_hashes={p.name: file_hash(p) for p in stage.glob('*.parquet')})
        write_json(stage / 'identity.json', identity)
        write_json(stage / 'report.json', report)
        write_json(stage / 'complete.json', {'status': 'complete', 'scope': 'selection only'})
        shutil.move(str(stage), OUT)
    print(json.dumps({'out': str(OUT), 'report': report}, indent=2))


if __name__ == '__main__':
    main()
