"""Immutable HD0/1 preparation with explicit store and legacy identity joins."""
from collections import OrderedDict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.cv import make_humansl_residual_roles
from go_aicup.data.store import GameStore
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table, writer_lock
from go_aicup.tasks.strength.SE_H_C_B2.data import sample_questions, check_games, match_metadata


class MoveStore:
    """Use GameStore or existing board-cache move shards, bounded to two shards."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.store = GameStore(directory=self.directory) if (self.directory / 'train_meta.parquet').exists() else None
        self.meta = self.store.meta.reset_index(drop=True) if self.store else pd.read_parquet(self.directory / 'games.parquet')
        self.shards = OrderedDict()
        self._verified = set()
        manifest = self.directory / 'manifest.json'
        self._hashes = {}
        if manifest.exists() and not self.store:
            record = json.loads(manifest.read_text())
            if not record.get('complete') or file_hash(self.directory / 'games.parquet') != record['games']['sha256']:
                raise ValueError('move store incomplete or canonical index checksum changed')
            self._hashes = {name: details['sha256'] for part in record['shards'] for name, details in part['files'].items()
                            if name.endswith(('_moves.npy', '_colors.npy'))}
    def game(self, index):
        if self.store:
            return self.store.game(index)
        row = self.meta.iloc[index]; shard = int(row.shard_id)
        if shard not in self.shards:
            if shard not in self._verified:
                for suffix in ('moves', 'colors'):
                    path = self.directory / f'shard_{shard:05d}_{suffix}.npy'
                    if path.name in self._hashes and file_hash(path) != self._hashes[path.name]:
                        raise ValueError(f'move shard checksum changed: {path.name}')
                self._verified.add(shard)
            self.shards[shard] = tuple(np.load(self.directory / f'shard_{shard:05d}_{suffix}.npy', mmap_mode='r')
                                       for suffix in ('moves', 'colors'))
        self.shards.move_to_end(shard)
        while len(self.shards) > 2:
            self.shards.popitem(last=False)
        start = int(row.local_offset); stop = start + int(row.length)
        return tuple(array[start:stop] for array in self.shards[shard])


def directory(config, phase='dev'):
    root = Path(config['data']['prepared_dir'])
    return root / 'release' if phase == 'release' else root


def games(config, phase='dev'):
    return pd.read_parquet(directory(config, phase) / 'games.parquet')


def questions(config, name, phase='dev'):
    aliases = {'dev': 'fit', 'M': 'monitor', 'u': 'U', 'release': 'release_fit'}
    return pd.read_parquet(directory(config, phase) / f'questions_{aliases.get(name, name)}.parquet')


def load_prepared(config, release=False):
    root = directory(config, 'release' if release else 'dev')
    record = json.loads((root / 'manifest.json').read_text())
    for name, digest in record['files'].items():
        if file_hash(root / name) != digest:
            raise ValueError(f'prepared artifact changed: {name}')
    for name, digest in record['identity']['sources'].items():
        if file_hash(Path(name)) != digest:
            raise ValueError(f'prepared source changed: {name}')
    expected = dict(split_seed=config['split_seed'], seed=config['seed'],
                    groups_per_update=config['training']['groups_per_update'],
                    fit_per_k=config.get('b2', {}).get('questions_per_k', 20000),
                    monitor_per_cell=config.get('monitoring', {}).get('questions_per_cell', config.get('monitor', {}).get('questions_per_cell', 125)))
    if record.get('dataset_id') != object_hash(record['identity']):
        raise ValueError('prepared dataset identity checksum changed')
    if any(record['identity'].get(key) != value for key, value in expected.items()):
        raise ValueError('prepared sampling config changed')
    return record


def eligibility(games):
    counts = games.groupby(['rank_id', 'player_id', 'player_color']).size()
    rows = []
    for rank in range(10):
        ranked = counts[counts.index.get_level_values('rank_id') == rank]
        for k in (5, 10, 15, 20):
            selected = ranked[ranked >= k]
            rows.append(dict(rank_id=rank, k=k, eligible_players=len(set(selected.index.get_level_values('player_id'))),
                             eligible_color_groups=len(selected)))
    return rows


def _align(meta, store):
    if meta.duplicated(['rank', 'game_id']).any() or store.meta.duplicated(['rank', 'game_id']).any():
        raise ValueError('duplicate rank/game_id')
    source = store.meta.reset_index(drop=True).reset_index(names='store_index')
    joined = meta.merge(source[['rank', 'game_id', 'store_index']], on=['rank', 'game_id'],
                        how='left', validate='one_to_one')
    if joined.store_index.isna().any():
        raise ValueError('canonical game missing from move store')
    source = source.set_index('store_index').loc[joined.store_index]
    for name in ('rank_id', 'player_id', 'color', 'length'):
        if name in source and name in joined and not np.array_equal(source[name].astype(str), joined[name].astype(str)):
            raise ValueError(f'move store identity mismatch: {name}')
    joined['store_index'] = joined.store_index.astype(np.int64)
    return joined


def prepare(config, *, release=False):
    cfg = config['data']; root = directory(config, 'release' if release else 'dev')
    if release:
        from .evaluate import load_lock
        load_lock(config)
    paths = [Path(cfg[k]) for k in ('metadata_path', 'handicap_path', 'folds_path')]
    legacy_paths = [Path(p) for p in cfg.get('legacy_split_paths', [cfg.get('legacy_split_path')])]
    paths.extend(legacy_paths)
    store_root = Path(cfg['store_dir'])
    store_meta = store_root / ('train_meta.parquet' if (store_root / 'train_meta.parquet').exists() else 'games.parquet')
    sources = {str(p): file_hash(p) for p in [*paths, store_meta]}
    if (store_root / 'manifest.json').exists():
        sources[str(store_root / 'manifest.json')] = file_hash(store_root / 'manifest.json')
    identity = dict(version=1, sources=sources, split_seed=config['split_seed'], seed=config['seed'],
                    release=release, min_ply=30, positions_per_side=15,
                    groups_per_update=config['training']['groups_per_update'],
                    fit_per_k=config.get('b2', {}).get('questions_per_k', 20000),
                    monitor_per_cell=config.get('monitoring', {}).get('questions_per_cell', config.get('monitor', {}).get('questions_per_cell', 125)))
    with writer_lock(root):
        if (root / 'manifest.json').exists():
            record = load_prepared(config, release)
            if record['identity'] != identity:
                raise ValueError('prepared identity changed; use a new prepared_dir')
            return record
        meta = pd.read_parquet(paths[0]); meta['player_id'] = meta.player_id.astype(str)
        if 'game_index' not in meta:
            meta = meta.reset_index()
        check_games(meta)
        store = MoveStore(store_root); meta = _align(meta, store)
        hd = pd.read_csv(paths[1], dtype={'game_id': str, 'rank': str})
        hd = hd.rename(columns={'handicap': 'hd', 'HD': 'hd'})
        if hd.duplicated(['rank', 'game_id']).any():
            raise ValueError('duplicate handicap rank/game_id')
        joined = meta.merge(hd[['rank', 'game_id', 'hd']], on=['rank', 'game_id'], how='left', validate='one_to_one')
        if joined.hd.isna().any() or not np.isfinite(joined.hd).all() or not joined.hd.ge(0).all() or not joined.hd.eq(joined.hd.astype(int)).all():
            raise ValueError('missing or invalid handicap')
        joined = joined[joined.hd.le(1)].copy()
        legacy = pd.concat([pd.read_parquet(p) for p in legacy_paths], ignore_index=True)
        legacy['player_id'] = legacy.player_id.astype(str)
        match_metadata(legacy, meta)
        if 'split' not in legacy and 'role' in legacy:
            legacy['split'] = legacy.role.map({'S-fit': 'valid_seen', 'S-extra': 'valid_seen',
                                               'S-select': 'valid_seen', 'U': 'valid_unseen'})
        if legacy['split'].isna().any():
            raise ValueError('unknown legacy role/split')
        if legacy.groupby('game_index')['split'].nunique().gt(1).any():
            raise ValueError('conflicting legacy game roles')
        legacy = legacy.drop_duplicates('game_index')
        if not legacy.game_index.is_unique or 'split' not in legacy:
            raise ValueError('legacy split requires unique game_index and split')
        joined = joined.merge(legacy[['game_index', 'split']].rename(columns={'split': 'legacy_pool'}),
                              on='game_index', how='left', validate='one_to_one')
        if joined.legacy_pool.isna().any():
            if 'legacy_split_paths' not in cfg:
                raise ValueError('legacy split missing games')
            joined['legacy_pool'] = joined.legacy_pool.fillna('train')
        folds = pd.read_parquet(paths[2]); folds['player_id'] = folds.player_id.astype(str)
        if set(folds.player_id) != set(meta.player_id):
            raise ValueError('fold source must cover the complete original canonical players before HD filtering')
        if 'rank_id' in folds:
            original_rank = meta.groupby('player_id').rank_id.first()
            if not np.array_equal(folds.rank_id.to_numpy(), original_rank.loc[folds.player_id].to_numpy()):
                raise ValueError('original fold player rank mismatch')
        roles = make_humansl_residual_roles(joined, folds, config['split_seed'], release=release)
        joined = joined.merge(roles[['player_id', 'role', 'oof_fold']], on='player_id', validate='many_to_one')
        own_counts, opp_counts = [], []
        for row in joined.itertuples():
            moves, colors = store.game(int(row.store_index))
            if (len(moves) != row.length or len(colors) != row.length or not np.isin(colors, [0, 1]).all()
                    or not np.array_equal(colors, np.arange(len(colors)) % 2)):
                # HD0/1 begins with black; validate turns, still sample by actual stored color.
                raise ValueError(f'invalid moves/colors: {row.rank}/{row.game_id}')
            own = int(np.sum(colors[30:] == row.player_color)); opp = len(colors[30:]) - own
            if min(own, opp) < 15:
                raise ValueError(f'fewer than fifteen positions per side: {row.rank}/{row.game_id} own={own} opp={opp}')
            own_counts.append(own); opp_counts.append(opp)
        joined['own_count'], joined['opp_count'] = own_counts, opp_counts
        training = joined[joined.role.eq('P' if release else 'D')]
        for fold in ('all', 1, 2, 3, 4):
            pool = training if fold == 'all' else training[training.oof_fold.ne(fold)]
            if any(int(pool.rank_id.eq(r).sum()) < config['training']['groups_per_update'] for r in range(10)):
                raise ValueError(f'fold {fold}: each rank needs groups_per_update games')
        fit = joined[joined.legacy_pool.isin(['valid_seen', 'valid_unseen'] if release else ['valid_seen'])
                     & joined.role.eq('P' if release else 'D')]
        pools = {'release_fit' if release else 'fit': fit}
        if not release:
            pools.update(monitor=joined[joined.role.eq('M')], U=joined[joined.role.eq('U')])
        frames = {'games': joined, 'players': roles}
        audit = {'pools': {}, 'question_frequency': {}}
        for name, pool in pools.items():
            cells = eligibility(pool)
            if any(c['eligible_players'] == 0 for c in cells):
                raise ValueError(f'{name}: a k/rank eligibility cell has no player')
            balanced = name in ('monitor', 'U')
            q = sample_questions(pool, identity['monitor_per_cell'] if balanced else identity['fit_per_k'],
                                 config['split_seed'], balanced=balanced, prefix=name)
            frames[f'questions_{name}'] = q
            audit['pools'][name] = dict(games=len(pool), players=pool.player_id.nunique(), eligibility=cells)
            frequencies = q.groupby(['k', 'player_id']).size().reset_index(name='questions')
            frames[f'question_frequency_{name}'] = frequencies
        files = {}
        for name, frame in frames.items():
            path = root / f'{name}.parquet'; atomic_table(path, frame); files[path.name] = file_hash(path)
        write_json(root / 'audit.json', audit); files['audit.json'] = file_hash(root / 'audit.json')
        record = dict(identity=identity, dataset_id=object_hash(identity), files=files)
        write_json(root / 'manifest.json', record)
        return record
