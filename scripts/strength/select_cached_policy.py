"""Select cache-first, player-disjoint policy questions; never encode or train."""
from pathlib import Path
import json
import zipfile

import pandas as pd

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.cv import make_cached_policy_games, make_rank_questions
from go_aicup.paths import ROOT


def main():
    seed = 2027
    base = ROOT / 'data/processed/strength_features/humansl'
    cache_root = base / 'c4x'
    out = base / 'policy_cnn/cached_AB_mondev_seed2027'
    if out.exists():
        raise FileExistsError(f'refusing to overwrite {out}')
    canonical = ROOT / 'data/processed/reproduce_paper/seed2026/games.parquet'
    meta = pd.read_parquet(canonical, columns=[
        'game_index', 'rank', 'game_id', 'rank_id', 'player_id', 'player_color', 'length'])
    with zipfile.ZipFile(ROOT / 'data/raw.zip') as archive:
        with archive.open('raw/training/handicap.csv') as stream:
            hd = pd.read_csv(stream, usecols=['rank', 'game_id', 'handicap'])
    meta = meta.merge(hd, on=['rank', 'game_id'], validate='one_to_one')
    paths = [cache_root / f'{name}_hd{h}.parquet'
             for name in ('learn_A', 'learn_B', 'monitor') for h in (0, 1)]
    sources = [pd.read_parquet(p).assign(handicap=h)
               for p, h in zip(paths, [0, 1] * 3)]
    cached = pd.concat(sources[:4], ignore_index=True)
    dev_cached = pd.concat(sources[4:], ignore_index=True)
    games = make_cached_policy_games(meta, cached, dev_cached, seed=seed)
    games = games[['game_index', 'rank_id', 'player_id', 'player_color', 'handicap',
                   'length', 'role', 'color', 'reuse_pack']]
    question_sets, report = {}, {}
    for offset, (role, count) in enumerate([('train', 8000), ('monitor', 4000), ('dev', 8000)]):
        part = games[games.role.eq(role)].set_index('game_index', drop=False)
        questions = make_rank_questions(part, part.player_id.unique(), count, seed=seed + 10 + offset)
        for q in questions.itertuples():
            chosen = part.loc[q.games]
            assert len(set(q.games)) == q.k
            assert chosen.player_id.nunique() == chosen.player_color.nunique() == chosen.rank_id.nunique() == 1
            assert str(chosen.player_id.iloc[0]) == str(q.player_id)
            assert chosen.rank_id.eq(q.rank_id).all()
        question_sets[role] = questions
        report[role] = dict(players=int(part.player_id.nunique()), games=len(part),
                            positions=int(part.length.sum()), questions=len(questions),
                            reused_games=int(part.reuse_pack.sum()),
                            supplement_games=int((~part.reuse_pack).sum()),
                            reused_positions=int(part.loc[part.reuse_pack, 'length'].sum()),
                            supplement_positions=int(part.loc[~part.reuse_pack, 'length'].sum()),
                            players_per_rank=part.groupby('rank_id').player_id.nunique().to_dict(),
                            questions_per_k=questions.groupby('k').size().to_dict())
    expected = {'train': (12451, 7549, 1683401), 'monitor': (2536, 1464, 324192),
                'dev': (1419, 5781, 1286332)}
    for role, values in expected.items():
        actual = tuple(report[role][k] for k in ('reused_games', 'supplement_games', 'supplement_positions'))
        if actual != values:
            raise ValueError(f'{role}: selection differs from plan: {actual} != {values}')
    out.mkdir(parents=True)
    games.to_parquet(out / 'games.parquet', index=False)
    games[games.reuse_pack].to_parquet(out / 'reuse_games.parquet', index=False)
    games[~games.reuse_pack].to_parquet(out / 'supplement_games.parquet', index=False)
    for role, questions in question_sets.items():
        questions.to_parquet(out / f'{role}_questions.parquet', index=False)
    identity = dict(seed=seed, train_per_rank=100, monitor_per_rank=20, games_per_player=20,
                    dev='36 players per rank from original C4x monitor', dev_per_rank=36,
                    colour='most cached games among colours with >=20 HD<=1 games; then most total; then black',
                    source_hashes={str(p.relative_to(ROOT)): file_hash(p) for p in
                                   [canonical, *paths, ROOT / 'src/go_aicup/data/cv.py', Path(__file__)]},
                    handicap_archive_member='data/raw.zip:raw/training/handicap.csv',
                    handicap_table_hash=pd.util.hash_pandas_object(hd, index=False).sum().item(),
                    artifact_hashes={p.name: file_hash(p) for p in out.glob('*.parquet')},
                    limitations=['new split requires retraining A0 and all compared arms',
                                 'C vectors may have seen selected players; overlap audited during extraction',
                                 'manifest membership does not verify packed file availability',
                                 'questions share games/players; bootstrap unit is player'])
    write_json(out / 'identity.json', identity)
    write_json(out / 'report.json', report)
    write_json(out / 'complete.json', {'status': 'complete', 'scope': 'selection only'})
    print(json.dumps({'out': str(out), 'report': report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
