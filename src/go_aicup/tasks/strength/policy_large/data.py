"""Read-only sources, sole-CV split and small reproducibility manifests."""
import zipfile
import json
import numpy as np
import torch
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash
from go_aicup.data.cv import make_large_policy_tables
from go_aicup.paths import ROOT
from go_aicup.features.humansl_torch import CKPT

CANONICAL = ROOT / 'data/processed/reproduce_paper/seed2026/games.parquet'
VECTORS = ROOT / 'outputs/rank/humansl/cvec_v2/learn/vectors.pt'


def prepare(config):
    m = pd.read_parquet(CANONICAL, columns=['game_index', 'rank', 'game_id', 'rank_id', 'player_id', 'player_color', 'length'])
    hd_path = ROOT / 'data/raw/training/handicap.csv'
    if hd_path.exists():
        hd = pd.read_csv(hd_path, usecols=['rank', 'game_id', 'handicap'])
    else:
        hd_path = ROOT / 'data/raw/raw.zip'
        with zipfile.ZipFile(hd_path) as z, z.open('raw/training/handicap.csv') as stream:
            hd = pd.read_csv(stream, usecols=['rank', 'game_id', 'handicap'])
    m = m.merge(hd, on=['rank', 'game_id'], validate='one_to_one')
    paths = [VECTORS.parent / f'learn_hd{h}.parquet' for h in (0, 1)]
    ids = pd.concat([pd.read_parquet(p) for p in paths]).game_index
    lookup = m.set_index('game_index')
    touched = set(lookup.loc[ids, 'player_id'].astype(str))
    keys = ['train_games_per_rank', 'monitor_players_per_rank', 'dev_players_per_rank',
            'games_per_holdout_player', 'train_questions', 'monitor_questions', 'dev_questions', 'split_seed']
    games, questions = make_large_policy_tables(m, touched, **{k: config[k] for k in keys})
    board_root = ROOT / 'data/processed/boards/strength_v1'
    board_path = board_root / 'games.parquet'
    board = pd.read_parquet(board_path).set_index('game_index').loc[games.game_index]
    for field in ('length', 'player_color'):
        np.testing.assert_array_equal(board[field].to_numpy(), games[field].to_numpy())
    manifest_path = board_root / 'manifest.json'
    if not json.loads(manifest_path.read_text())['complete']:
        raise ValueError('board cache is incomplete')
    board_files = []
    for shard, group in board.groupby('shard_id'):
        for name, dtype in [('moves', np.int16), ('colors', np.int8)]:
            path = board_root / f'shard_{int(shard):05d}_{name}.npy'
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            if value.ndim != 1 or value.dtype != dtype or (group.local_offset + group.length).max() > len(value):
                raise ValueError(f'invalid board cache source: {path}')
            board_files.append(path)
    vectors = torch.load(VECTORS, map_location='cpu', weights_only=False)['vectors']
    if tuple(vectors.shape) != (10, 2, 384) or not torch.isfinite(vectors).all():
        raise ValueError('C-v2 vectors have invalid shape or finite values')
    source_paths = [CANONICAL, hd_path, VECTORS, CKPT, *paths, manifest_path, board_path, *board_files]
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    report = {'sources': sources, 'c_touched_players': len(touched), 'roles': {}}
    for role, part in games.groupby('role'):
        report['roles'][role] = dict(games=len(part), players=int(part.player_id.nunique()),
            positions=int(part.length.sum()), players_by_rank=part.groupby('rank_id').player_id.nunique().to_dict(),
            questions=len(questions[role]), covered_games=len(set(i for row in questions[role].games for i in row)),
            c_touched_players=int(part.player_id.isin(touched).groupby(part.player_id).any().sum()))
    report['positions'] = int(games.length.sum())
    report['cache_bytes'] = {family: report['positions'] * (p*362*2 + p*4 + 3)
                             for family, p in [('H', 9), ('C', 10)]}
    report['tables_identity'] = object_hash({'games': pd.util.hash_pandas_object(games, index=False).astype(str).tolist(),
        'questions': {r: q.assign(games=q.games.map(lambda v: tuple(map(int, v)))).to_dict('records') for r, q in questions.items()}})
    return games, questions, report
