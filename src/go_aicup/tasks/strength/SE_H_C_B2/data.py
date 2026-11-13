"""Immutable, same-color fitting questions and game-held-out evaluation."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from .common import BUCKETS, atomic_table, source_identity, writer_lock


def check_games(games):
    if not games.game_index.is_unique or games[['game_index', 'player_id', 'player_color', 'rank_id']].isna().any().any():
        raise ValueError('duplicate/missing game metadata')
    if not games.player_color.isin([0, 1]).all() or not games.rank_id.isin(range(10)).all():
        raise ValueError('invalid color/rank')
    if games.groupby('player_id').rank_id.nunique().gt(1).any():
        raise ValueError('one player has multiple ranks')


def match_metadata(rows, canonical):
    reference = canonical.set_index('game_index').loc[rows.game_index]
    for column in ('rank_id', 'player_id', 'player_color', 'length', 'game_id'):
        if column in rows and not np.array_equal(rows[column].astype(str), reference[column].astype(str)):
            raise ValueError(f'game_index metadata mismatch: {column}')


def check_questions(questions, games):
    if not questions.question_id.is_unique or questions.empty:
        raise ValueError('empty or duplicate question IDs')
    lookup = games.set_index('game_index')
    for q in questions.itertuples():
        ids = np.asarray(q.games)
        if q.k not in BUCKETS or len(ids) != q.k or len(np.unique(ids)) != q.k:
            raise ValueError('invalid k or duplicate games within question')
        rows = lookup.loc[ids]
        color = {'B': 0, 'W': 1}.get(str(q.color), q.color)
        if (not rows.player_id.astype(str).eq(str(q.player_id)).all()
                or not rows.player_color.eq(int(color)).all() or not rows.rank_id.eq(q.rank_id).all()):
            raise ValueError('question must have one player, color and rank')


def sample_questions(games, count, seed, *, balanced=False, prefix='fit'):
    check_games(games)
    rng = np.random.default_rng(seed)
    rows = []
    for k in BUCKETS:
        eligible = games.groupby(['player_id', 'player_color'], sort=True).filter(lambda g: len(g) >= k)
        for rank in (range(10) if balanced else (None,)):
            pool = eligible if rank is None else eligible[eligible.rank_id.eq(rank)]
            players = [(p, [g for _, g in groups.groupby('player_color', sort=True)])
                       for p, groups in pool.groupby('player_id', sort=True)]
            if not players:
                raise ValueError(f'no eligible players for k={k}, rank={rank}')
            if balanced:
                capacity = sum(min(count, math.comb(len(g), k)) for _, colors in players for g in colors)
                if capacity < count:
                    raise ValueError(f'not enough unique questions for k={k}, rank={rank}')
            used = set()
            for _ in range(count):
                for attempt in range(100000):
                    player, colors = players[int(rng.integers(len(players)))]
                    group = colors[int(rng.integers(len(colors)))]
                    ids = np.sort(rng.choice(group.game_index.to_numpy(), k, replace=False))
                    signature = tuple(ids)
                    if not balanced or signature not in used:
                        break
                else:
                    raise ValueError('could not draw unique evaluation question')
                used.add(signature)
                rows.append(dict(question_id=f'{prefix}_{len(rows):07d}', player_id=str(player),
                                 rank_id=int(group.rank_id.iloc[0]), color=int(group.player_color.iloc[0]),
                                 k=k, games=ids))
    result = pd.DataFrame(rows)
    if any(set(result.loc[result.k.eq(k), 'rank_id']) != set(range(10)) for k in BUCKETS):
        raise ValueError('all ten fitting/evaluation classes are required per k')
    check_questions(result, games)
    return result


def make_pools(canonical, roles, c_games, monitor):
    check_games(canonical)
    match_metadata(roles, canonical)
    match_metadata(c_games, canonical)
    if not roles.game_index.is_unique:
        raise ValueError('duplicate role games')
    dev = roles[roles.role.isin(['S-fit', 'S-extra'])]
    select = roles[roles.role.eq('S-select')]
    full = roles[roles.role.isin(['S-fit', 'S-extra', 'S-select'])]
    unseen = roles[roles.role.eq('U')]
    if any(g.empty for g in (dev, select, full, unseen)):
        raise ValueError('missing data roles')
    if set(dev.player_id) & set(select.player_id) or set(full.player_id) & set(unseen.player_id):
        raise ValueError('B2 fitting/evaluation players overlap')
    if set(c_games.player_id) & (set(unseen.player_id) | set(select.player_id)):
        raise ValueError('C learning/monitor players overlap evaluation')
    pool = canonical[canonical.split.eq('valid') & canonical.player_id.isin(set(unseen.player_id))
                     & ~canonical.game_index.isin(monitor) & ~canonical.game_index.isin(c_games.game_index)].copy()
    if (set(pool.game_index) & set(canonical.loc[canonical.split.eq('train'), 'game_index'])
            or set(pool.game_index) & set(full.game_index) or set(pool.player_id) & set(full.player_id)):
        raise ValueError('newgames evaluation leakage')
    return dev, full, pool


def legacy_files(paths):
    files = []
    root = paths['human_root']
    for names, columns in [(('humansl_q1000', 'humansl_qsel', 'humansl_vs_rest'),
                            ('games.parquet', 'meta.json', 'logp.npy', 'offsets.npy', 'done.npy', 'colors.npy')),
                           (('humansl_rankvec_feat', 'humansl_rankvec_feat_vu'),
                            ('games.parquet', 'meta.json', 'feature.npy', 'done.npy'))]:
        for name in names:
            for shard in sorted((root / name).glob('shard_*')):
                if (shard / 'meta.json').exists():
                    files.extend(shard / col for col in columns if (shard / col).exists())
    files.extend(paths['a_run'] / name for name in
                 ('validation_unseen_category.npy', 'validation_seen_category.npy') if (paths['a_run'] / name).exists())
    directory = paths['se3_evaluation']
    if (directory / 'manifest.json').exists():
        files.append(directory / 'manifest.json')
        for shard in sorted(directory.glob('games_rank*.parquet')):
            files.extend([shard, shard.with_suffix('.json')])
    return files


def prepare(config):
    paths = {k: Path(v) for k, v in config['paths'].items()}
    output = paths['output']
    source = source_identity(config)
    inputs = [paths['dataset'] / 'games.parquet', paths['cache'] / 'games.parquet', paths['se3_monitor'],
              paths['human_root'] / 'humansl_rankvec/games.parquet']
    inputs += [paths['human_root'] / name / 'games.parquet'
               for name in ('humansl_q1000', 'humansl_qsel', 'humansl_vs_rest')]
    inputs += [paths['human_root'] / name / 'questions.parquet' for name in ('humansl_q1000', 'humansl_qsel')]
    identity = dict(source=source, inputs={str(p): file_hash(p) for p in [*inputs, *legacy_files(paths)]},
                    seed=config['seed'], synthetic=config['synthetic_per_k'], per_cell=config['evaluation_per_cell'])
    with writer_lock(output):
        manifest = output / 'manifest.json'
        if manifest.exists():
            load_prepared(config, identity)
            return json.loads((output / 'audit.json').read_text())
        canonical = pd.read_parquet(inputs[0])
        canonical['player_id'] = canonical.player_id.astype(str)
        match_metadata(canonical, pd.read_parquet(inputs[1]).reset_index())
        roles = pd.concat([pd.read_parquet(p) for p in inputs[4:7]], ignore_index=True)
        roles['player_id'] = roles.player_id.astype(str)
        c_games = pd.read_parquet(inputs[3]); c_games['player_id'] = c_games.player_id.astype(str)
        with np.load(paths['se3_monitor']) as data:
            monitor = data['games']
        dev, full, pool = make_pools(canonical, roles, c_games, monitor)
        qselect = pd.read_parquet(inputs[8]); qselect = qselect[qselect.role.eq('S-select')].copy()
        qu = pd.read_parquet(inputs[7]); qu = qu[qu.role.eq('U')].copy()
        check_questions(qselect, canonical); check_questions(qu, canonical)
        frames = {'games': canonical[canonical.game_index.isin(roles.game_index) | canonical.game_index.isin(pool.game_index)],
                  'roles': roles, 'newgames_pool': pool,
                  'questions/dev': sample_questions(dev, config['synthetic_per_k'], config['seed'], prefix='dev'),
                  'questions/final': sample_questions(full, config['synthetic_per_k'], config['seed'] + 1, prefix='final'),
                  'questions/select': qselect, 'questions/old_u': qu,
                  'questions/newgames': sample_questions(pool, config['evaluation_per_cell'], config['seed'],
                                                        balanced=True, prefix='newgames')}
        audit = {'pools': {n: dict(games=len(g), players=g.player_id.nunique())
                          for n, g in [('dev', dev), ('final', full), ('newgames', pool)]},
                 'newgames_positions': int(pool.effective_position_count.sum()),
                 'limitation': 'SE3 has seen these players; newgames excludes SE3 training/monitor games and C games.'}
        eval_ids = np.unique(np.concatenate(frames['questions/newgames'].games))
        audit['newgames_unique_selected'] = len(eval_ids)
        for name in ('dev', 'final', 'newgames'):
            audit[f'{name}_cells'] = {f'{k}/{r}': int(n) for (k, r), n in
                                     frames[f'questions/{name}'].groupby(['k', 'rank_id']).size().items()}
        files = {}
        for name, frame in frames.items():
            path = output / f'{name}.parquet'; atomic_table(path, frame)
            files[f'{name}.parquet'] = file_hash(path)
        write_json(output / 'audit.json', audit)
        files['audit.json'] = file_hash(output / 'audit.json')
        write_json(manifest, dict(identity=identity, files=files))
        return audit


def load_prepared(config, identity=None):
    root = Path(config['paths']['output'])
    record = json.loads((root / 'manifest.json').read_text())
    if identity is not None and identity != record['identity']:
        raise ValueError('prepared experiment identity changed')
    expected = (config['seed'], config['synthetic_per_k'], config['evaluation_per_cell'])
    actual = tuple(record['identity'][key] for key in ('seed', 'synthetic', 'per_cell'))
    if actual != expected:
        raise ValueError('prepared sampling config changed')
    if record['identity']['source'] != source_identity(config):
        raise ValueError('frozen sources or experiment code changed')
    for name, digest in {**record['identity']['inputs'], **{str(root / n): h for n, h in record['files'].items()}}.items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f'prepared source/artifact changed: {name}')
    return record


def questions(config, name):
    return pd.read_parquet(Path(config['paths']['output']) / f'questions/{name}.parquet')
