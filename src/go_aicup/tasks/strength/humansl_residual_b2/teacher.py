"""Resumable strong KataGo labels for actual training positions only."""
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features.katago_engine import MODEL
from go_aicup.tasks.strength.reproduce_paper.label_value import Engine, make_query, parse_root
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table, writer_lock
from .data import MoveStore, games, load_prepared


def teacher_identity(config, table):
    cfg = config['teacher']
    keys = ['game_index', 'rank', 'game_id', 'store_index', 'hd', 'length', 'player_color']
    return dict(version=1, games=object_hash(table.sort_values('game_index')[keys].to_dict('records')),
                engine_sha256=file_hash(Path(cfg['engine'])), model_sha256=file_hash(Path(cfg['model_path'])),
                config_sha256=file_hash(Path(cfg['analysis_config'])), visits=1, rules='japanese',
                komi={'0': 6.5, '1': .5}, min_ply=30, perspective='actual_side_to_move',
                ignorePreRootHistory=False, num_shards=cfg.get('num_shards', 8),
                source_files=load_prepared(config)['identity']['sources'],
                teacher_code_sha256=file_hash(Path(__file__)),
                dataset_code_sha256=file_hash(Path(__file__).with_name('dataset.py')))


def value_digest(values):
    return int.from_bytes(hashlib.blake2b(np.asarray(values, dtype=np.float32).tobytes(), digest_size=8).digest(), 'little')


def _shard(config, shard):
    return Path(config['data']['teacher_dir']) / f'shard_{shard:05d}'


class TeacherLabels:
    def __init__(self, config):
        self.config = config
        root = Path(config['data']['teacher_dir'])
        self.record = json.loads((root / 'identity.json').read_text())
        cfg = config['teacher']
        if self.record['num_shards'] != cfg.get('num_shards', 8) or cfg.get('visits', 1) != 1:
            raise ValueError('teacher sharding or visits changed')
        for key, path in [('engine_sha256', cfg['engine']), ('model_sha256', cfg['model_path']),
                          ('config_sha256', cfg['analysis_config'])]:
            if self.record[key] != file_hash(Path(path)):
                raise ValueError('teacher engine/model/settings identity changed')
        if (self.record['perspective'] != 'actual_side_to_move' or self.record['visits'] != 1
                or self.record.get('rules') != 'japanese' or self.record.get('komi') != {'0': 6.5, '1': .5}
                or self.record.get('min_ply') != 30 or self.record.get('ignorePreRootHistory') is not False):
            raise ValueError('unsupported teacher target identity')
        if self.record.get('source_files') != load_prepared(config)['identity']['sources']:
            raise ValueError('teacher prepared move sources changed')
        for field, path in [('teacher_code_sha256', Path(__file__)),
                            ('dataset_code_sha256', Path(__file__).with_name('dataset.py'))]:
            if self.record.get(field) != file_hash(path):
                raise ValueError('teacher encoder/labeler code identity changed')
        for path, digest in self.record.get('source_files', {}).items():
            if file_hash(Path(path)) != digest:
                raise ValueError('teacher move source identity changed')
        self.shards = {}
    def values(self, game, plies):
        shard = int(game.game_index) % self.record['num_shards']
        if shard not in self.shards:
            root = _shard(self.config, shard)
            marker = json.loads((root / 'manifest.json').read_text())
            if marker['teacher_id'] != object_hash(self.record) or marker['index_sha256'] != file_hash(root / 'index.parquet'):
                raise ValueError('teacher shard identity mismatch')
            index = pd.read_parquet(root / 'index.parquet').set_index('game_index')
            self.shards[shard] = (index, np.load(root / 'values.npy', mmap_mode='r'),
                                  np.load(root / 'done.npy', mmap_mode='r'), np.load(root / 'checksums.npy', mmap_mode='r'))
        index, values, done, checksums = self.shards[shard]
        row = index.loc[int(game.game_index)]
        if any(str(row[name]) != str(game[name]) for name in ('rank', 'game_id', 'hd', 'length', 'player_color')):
            raise ValueError('teacher game/HD/ply source identity mismatch')
        if not done[int(row.row)] or np.any(plies < 30) or np.any(plies >= int(row.length)):
            raise ValueError('missing training teacher targets')
        full = values[int(row.offset):int(row.offset) + int(row.length) - 30]
        if value_digest(full) != int(checksums[int(row.row)]):
            raise ValueError('teacher value checksum changed')
        selected = np.array(values[int(row.offset) + np.asarray(plies) - 30], dtype=np.float32)
        if not np.isfinite(selected).all() or np.any(np.abs(selected) > 1):
            raise ValueError('nonfinite or invalid teacher value')
        return selected


def label(config, *, phase='dev', shard=0, limit_games=None):
    started = time.monotonic()
    release = phase == 'release'
    if release:
        from .evaluate import load_lock
        load_lock(config)
    prepared = load_prepared(config, release=release)
    table = games(config, phase).sort_values('game_index').reset_index(drop=True)
    cfg = config['teacher']; num_shards = cfg.get('num_shards', 8)
    if type(num_shards) is not int or num_shards < 1 or cfg.get('visits', 1) != 1 or not 0 <= shard < num_shards:
        raise ValueError('invalid teacher shard')
    if file_hash(Path(cfg['model_path'])) != file_hash(MODEL):
        raise ValueError('teacher must be the registered strong KataGo network')
    # Role membership is deliberately absent: release can extend the same labels.
    identity_table = games(config, 'dev') if release else table
    identity = teacher_identity(config, identity_table)
    root = Path(config['data']['teacher_dir']); root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root):
        if (root / 'identity.json').exists():
            if json.loads((root / 'identity.json').read_text()) != identity:
                raise ValueError('teacher source/settings changed; use a new teacher_dir')
        else:
            write_json(root / 'identity.json', identity)
    source = table[table.game_index.mod(num_shards).eq(shard)].copy().reset_index(drop=True)
    source['row'] = np.arange(len(source)); source['offset'] = np.r_[0, np.cumsum(source.length.to_numpy()[:-1] - 30)]
    count = int(np.sum(source.length - 30)); out = _shard(config, shard)
    with writer_lock(out):
        marker = out / 'manifest.json'
        if marker.exists():
            record = json.loads(marker.read_text())
            if record['teacher_id'] != object_hash(identity) or record['positions'] != count:
                raise ValueError('teacher shard source changed')
            existing = pd.read_parquet(out / 'index.parquet')
            names = ['game_index', 'rank', 'game_id', 'hd', 'length', 'player_color', 'offset', 'row']
            if existing[names].to_dict('records') != source[names].to_dict('records'):
                raise ValueError('teacher position alignment changed')
            if file_hash(out / 'index.parquet') != record['index_sha256']:
                raise ValueError('teacher index checksum changed')
            values = np.load(out / 'values.npy', mmap_mode='r+')
            done = np.load(out / 'done.npy', mmap_mode='r+')
            checksums = np.load(out / 'checksums.npy', mmap_mode='r+')
        else:
            atomic_table(out / 'index.parquet', source.drop(columns=['role', 'oof_fold'], errors='ignore'))
            values = np.lib.format.open_memmap(out / 'values.npy', mode='w+', dtype=np.float32, shape=(count,))
            values[:] = np.nan; values.flush()
            done = np.lib.format.open_memmap(out / 'done.npy', mode='w+', dtype=np.bool_, shape=(len(source),))
            done[:] = False; done.flush()
            checksums = np.lib.format.open_memmap(out / 'checksums.npy', mode='w+', dtype=np.uint64, shape=(len(source),))
            checksums[:] = 0; checksums.flush()
            write_json(marker, dict(teacher_id=object_hash(identity), positions=count,
                                    index_sha256=file_hash(out / 'index.parquet')))
        training = source[source.role.eq('P' if release else 'D')]
        if limit_games is not None:
            if limit_games < 1:
                raise ValueError('limit_games must be positive')
            training = training.iloc[:limit_games]
        store = MoveStore(config['data']['store_dir'])
        setup_seconds = time.monotonic() - started
        new_games = new_positions = 0
        engine = None
        success = False
        try:
            engine = Engine(Path(cfg['engine']), Path(cfg['model_path']), Path(cfg['analysis_config']),
                            out / 'engine.log', cfg.get('timeout', 300))
            for game in training.itertuples():
                index = int(game.row)
                if done[index]:
                    existing = values[int(game.offset):int(game.offset) + int(game.length) - 30]
                    if not np.isfinite(existing).all() or np.any(np.abs(existing) > 1) or value_digest(existing) != int(checksums[index]):
                        raise ValueError('completed teacher row is corrupt')
                    continue
                moves, colors = store.game(int(game.store_index)); plies = np.arange(30, len(moves))
                query = make_query(f'{game.rank}/{game.game_id}', moves, colors, plies, 1, 'japanese', 6.5 if game.hd == 0 else .5)
                completed = set()
                for ply, response in engine.analyze(query):
                    _, value, _, _, _ = parse_root(response, int(colors[ply]))
                    values[int(game.offset) + ply - 30] = value; completed.add(ply)
                    new_positions += 1
                if completed != set(plies):
                    raise ValueError('teacher failed to return every required actual-side target')
                values.flush()
                checksums[index] = value_digest(values[int(game.offset):int(game.offset) + int(game.length) - 30]); checksums.flush()
                done[index] = True; done.flush()
                new_games += 1
            success = True
        finally:
            if engine is not None:
                engine.close()
            seconds = time.monotonic() - started
            metrics = dict(phase=phase, shard=shard, training_games=len(training), completed_games=int(done.sum()),
                           teacher_id=object_hash(identity), prepared_id=prepared['dataset_id'],
                           status='complete' if success else 'failed', seconds=seconds, setup_seconds=setup_seconds,
                           new_games=new_games, new_positions=new_positions,
                           positions_per_second=new_positions / seconds if seconds else 0.,
                           gpu_count=1, gpu_hours=seconds / 3600)
            write_json(out / 'metrics.json', metrics)
    return metrics


def main(argv=None):
    from .config import read_config, DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--phase', choices=['dev', 'release'], default='dev')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--limit-games', type=int)
    args = parser.parse_args(argv)
    print(json.dumps(label(read_config(args.config), phase=args.phase, shard=args.shard, limit_games=args.limit_games)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
