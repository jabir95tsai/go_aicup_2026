"""Resumable strong KataGo labels for actual training positions only."""
import argparse
import json
import os
import hashlib
import time
import signal
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features.katago_engine import MODEL
from go_aicup.tasks.strength.reproduce_paper.label_value import Engine, make_query, parse_root
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table
from .data import MoveStore, games, load_prepared


# This labeler preserves the original pre-action targets and whole-game commit format.
LEGACY_CODE = 'b0bb8f9457d3bb826fd23fa49343f663cd728cba6236b64a119d5b8594e322be'


@contextmanager
def teacher_lock(directory, *, wait=False):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.teacher.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        if (directory / '.writer.lock').exists():
            raise ValueError('legacy writer lock remains; confirm old job stopped before removing it')
        yield


@contextmanager
def stop_signals():
    stopped = [False]
    def stop(*_):
        stopped[0] = True
    previous = {s: signal.signal(s, stop) for s in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT)}
    try:
        yield stopped
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def compatible_identity(saved, current):
    # Only the explicitly supported original implementation may resume across this labeler upgrade.
    expected = dict(current)
    if saved.get('teacher_code_sha256') == LEGACY_CODE:
        expected['teacher_code_sha256'] = LEGACY_CODE
    return saved == expected


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
            if self.record.get(field) != file_hash(path) and not (field == 'teacher_code_sha256' and self.record.get(field) == LEGACY_CODE):
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


def _fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _durable_write(path, writer):
    """Plain write + fsync + atomic rename; shared memmap writes were silently lost on Nano4 WekaFS."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _read_from_disk(path):
    """Drop this node's clean cached pages first so verification re-reads what the filesystem holds."""
    with open(path, 'rb') as handle:
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        return np.load(handle, allow_pickle=False)


def _valid_rows(source, values, done, checksums):
    valid = np.zeros(len(source), dtype=bool)
    for game in source.itertuples():
        index = int(game.row)
        if not done[index]:
            continue
        segment = values[int(game.offset):int(game.offset) + int(game.length) - 30]
        valid[index] = (np.isfinite(segment).all() and not np.any(np.abs(segment) > 1)
                        and value_digest(segment) == int(checksums[index]))
    return valid


def _load_state(out, source, count):
    """Verified consolidated rows plus committed chunks; anything unverifiable is relabelled."""
    values = np.full(count, np.nan, dtype=np.float32)
    done = np.zeros(len(source), dtype=bool)
    checksums = np.zeros(len(source), dtype=np.uint64)
    discarded = 0
    if all((out / name).exists() for name in ('values.npy', 'done.npy', 'checksums.npy')):
        old_values, old_done, old_checksums = (_read_from_disk(out / name) for name in ('values.npy', 'done.npy', 'checksums.npy'))
        if old_values.shape != (count,) or old_done.shape != (len(source),) or old_checksums.shape != (len(source),):
            raise ValueError('teacher shard arrays do not match the index')
        valid = _valid_rows(source, old_values, old_done, old_checksums)
        discarded = int(old_done.sum() - valid.sum())
        for game in source[valid[source.row.to_numpy()]].itertuples():
            index, start = int(game.row), int(game.offset)
            stop = start + int(game.length) - 30
            values[start:stop] = old_values[start:stop]
            done[index] = True
            checksums[index] = old_checksums[index]
    lengths = source.length.to_numpy() - 30
    offsets = source.offset.to_numpy()
    for path in sorted((out / 'chunks').glob('chunk_*.npz')):
        with np.load(path, allow_pickle=False) as chunk:
            rows, digests, sizes, flat = chunk['rows'], chunk['checksums'], chunk['lengths'], chunk['values']
        if len(rows) != len(digests) or len(rows) != len(sizes) or int(sizes.sum()) != len(flat):
            raise ValueError(f'teacher chunk malformed: {path.name}')
        cursor = 0
        for row, digest, size in zip(rows.tolist(), digests.tolist(), sizes.tolist()):
            segment = flat[cursor:cursor + size]; cursor += size
            if (size != lengths[row] or not np.isfinite(segment).all() or np.any(np.abs(segment) > 1)
                    or value_digest(segment) != digest):
                raise ValueError(f'teacher chunk row invalid: {path.name}')
            values[offsets[row]:offsets[row] + size] = segment
            done[row] = True
            checksums[row] = digest
    return values, done, checksums, discarded


def _commit_chunk(out, sequence, pending):
    rows = np.asarray([row for row, _, _ in pending], dtype=np.int64)
    digests = np.asarray([digest for _, digest, _ in pending], dtype=np.uint64)
    sizes = np.asarray([len(segment) for _, _, segment in pending], dtype=np.int64)
    flat = np.concatenate([segment for _, _, segment in pending]).astype(np.float32)
    (out / 'chunks').mkdir(exist_ok=True)
    _durable_write(out / 'chunks' / f'chunk_{sequence:06d}.npz',
                   lambda handle: np.savez(handle, rows=rows, checksums=digests, lengths=sizes, values=flat))


def _publish_arrays(out, values, done, checksums):
    for name, array in (('values.npy', values), ('checksums.npy', checksums), ('done.npy', done)):
        _durable_write(out / name, lambda handle, array=array: np.save(handle, array))


def verify_shard(config, *, phase='dev', shard=0, limit_games=None):
    """Re-read published arrays from disk and check every requested training row."""
    table = games(config, phase).sort_values('game_index').reset_index(drop=True)
    num_shards = config['teacher'].get('num_shards', 8)
    source = table[table.game_index.mod(num_shards).eq(shard)].copy().reset_index(drop=True)
    source['row'] = np.arange(len(source)); source['offset'] = np.r_[0, np.cumsum(source.length.to_numpy()[:-1] - 30)]
    out = _shard(config, shard)
    record = json.loads((out / 'manifest.json').read_text())
    if file_hash(out / 'index.parquet') != record['index_sha256'] or record['positions'] != int(np.sum(source.length - 30)):
        raise ValueError('teacher index checksum changed')
    values, done, checksums = (_read_from_disk(out / name) for name in ('values.npy', 'done.npy', 'checksums.npy'))
    training = source[source.role.eq('P' if phase == 'release' else 'D')]
    if limit_games is not None:
        training = training.iloc[:limit_games]
    valid = _valid_rows(source, values, done, checksums)
    verified = int(valid[training.row.to_numpy()].sum())
    return dict(training_games=len(training), verified_games=verified, complete=verified == len(training))


def label(config, *, phase='dev', shard=0, limit_games=None):
    started = time.monotonic()
    release = phase == 'release'
    if release:
        from .evaluate import load_lock
        load_lock(config)
    prepared = load_prepared(config, release=release)
    table = games(config, phase).sort_values('game_index').reset_index(drop=True)
    cfg = config['teacher']; num_shards = cfg.get('num_shards', 8)
    chunk_games = cfg.get('chunk_games', 500)
    if type(num_shards) is not int or num_shards < 1 or cfg.get('visits', 1) != 1 or not 0 <= shard < num_shards:
        raise ValueError('invalid teacher shard')
    if type(chunk_games) is not int or chunk_games < 1:
        raise ValueError('invalid teacher chunk_games')
    if file_hash(Path(cfg['model_path'])) != file_hash(MODEL):
        raise ValueError('teacher must be the registered strong KataGo network')
    # Role membership is deliberately absent: release can extend the same labels.
    identity_table = games(config, 'dev') if release else table
    identity = teacher_identity(config, identity_table)
    root = Path(config['data']['teacher_dir']); root.mkdir(parents=True, exist_ok=True)
    with teacher_lock(root, wait=True):
        if (root / 'identity.json').exists():
            saved = json.loads((root / 'identity.json').read_text())
            if not compatible_identity(saved, identity):
                raise ValueError('teacher source/settings changed; use a new teacher_dir')
            identity = saved
        else:
            write_json(root / 'identity.json', identity)
    source = table[table.game_index.mod(num_shards).eq(shard)].copy().reset_index(drop=True)
    source['row'] = np.arange(len(source)); source['offset'] = np.r_[0, np.cumsum(source.length.to_numpy()[:-1] - 30)]
    count = int(np.sum(source.length - 30)); out = _shard(config, shard)
    with teacher_lock(out), stop_signals() as stopped:
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
        else:
            atomic_table(out / 'index.parquet', source.drop(columns=['role', 'oof_fold'], errors='ignore'))
            write_json(marker, dict(teacher_id=object_hash(identity), positions=count,
                                    index_sha256=file_hash(out / 'index.parquet')))
        values, done, checksums, discarded_games = _load_state(out, source, count)
        if discarded_games:
            # Publish the honest state now so external progress checks never count unverifiable rows.
            _publish_arrays(out, values, done, checksums)
        sequence = 1 + max([int(p.stem.split('_')[1]) for p in (out / 'chunks').glob('chunk_*.npz')], default=-1)
        training = source[source.role.eq('P' if release else 'D')]
        if limit_games is not None:
            if limit_games < 1:
                raise ValueError('limit_games must be positive')
            training = training.iloc[:limit_games]
        reuse = None
        if cfg.get('se3_reuse'):
            from .teacher_reuse import SE3Values
            reuse = SE3Values(cfg['se3_reuse'], identity)
            receipt = out / 'se3_import.json'
            if receipt.exists() and json.loads(receipt.read_text()) != reuse.receipt:
                raise ValueError('SE3 import source changed')
            write_json(receipt, reuse.receipt)
        store = MoveStore(config['data']['store_dir'])
        setup_seconds = time.monotonic() - started
        new_games = new_positions = imported_positions = 0
        engine = None
        success = False
        pending = []
        verification = None
        try:
            try:
                for game in training.itertuples():
                    if stopped[0]:
                        break
                    index = int(game.row)
                    if done[index]:
                        continue
                    moves, colors = store.game(int(game.store_index)); plies = np.arange(30, len(moves))
                    imported, targets = reuse.positions(game, colors) if reuse else (np.empty(0, np.int64), None)
                    segment = np.full(len(moves) - 30, np.nan, dtype=np.float32)
                    segment[imported - 30] = targets if targets is not None else []
                    imported_positions += len(imported)
                    missing = np.setdiff1d(plies, imported)
                    completed = set(imported)
                    if len(missing):
                        if engine is None:
                            engine = Engine(Path(cfg['engine']), Path(cfg['model_path']), Path(cfg['analysis_config']),
                                            out / 'engine.log', cfg.get('timeout', 300))
                        query = make_query(f'{game.rank}/{game.game_id}', moves, colors, missing, 1, 'japanese', 6.5 if game.hd == 0 else .5)
                        for ply, response in engine.analyze(query):
                            if ply not in missing or ply in completed:
                                raise ValueError('unexpected/duplicate teacher ply')
                            _, value, _, _, _ = parse_root(response, int(colors[ply]))
                            segment[ply - 30] = value; completed.add(ply)
                            new_positions += 1
                    if completed != set(plies) or not np.isfinite(segment).all() or np.any(np.abs(segment) > 1):
                        raise ValueError('teacher failed to return every required actual-side target')
                    digest = value_digest(segment)
                    values[int(game.offset):int(game.offset) + len(segment)] = segment
                    checksums[index] = digest; done[index] = True
                    pending.append((index, digest, segment))
                    new_games += 1
                    if len(pending) >= chunk_games:
                        _commit_chunk(out, sequence, pending); sequence += 1; pending = []
                success = not stopped[0]
            finally:
                if engine is not None:
                    engine.close()
                # Only whole, validated games are ever pending, so they are always safe to commit.
                if pending:
                    _commit_chunk(out, sequence, pending); sequence += 1; pending = []
                _publish_arrays(out, values, done, checksums)
                verification = verify_shard(config, phase=phase, shard=shard, limit_games=limit_games)
                if verification['verified_games'] != int(done[training.row.to_numpy()].sum()):
                    raise ValueError('published teacher arrays differ from committed labels on re-read')
                # Chunks are redundant only after the published arrays re-read correctly.
                for path in sorted((out / 'chunks').glob('chunk_*.npz')):
                    path.unlink()
        finally:
            seconds = time.monotonic() - started
            if success and not (verification or {}).get('complete'):
                status = 'incomplete'
            else:
                status = 'complete' if success else ('stopped' if stopped[0] else 'failed')
            metrics = dict(phase=phase, shard=shard, training_games=len(training),
                           completed_games=int(done[training.row.to_numpy()].sum()),
                           verified_games=(verification or {}).get('verified_games'),
                           discarded_games=discarded_games,
                           teacher_id=object_hash(identity), prepared_id=prepared['dataset_id'],
                           status=status, seconds=seconds, setup_seconds=setup_seconds,
                           new_games=new_games, new_positions=new_positions, imported_positions=imported_positions,
                           runtime_code_sha256=file_hash(Path(__file__)),
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
    parser.add_argument('--verify', action='store_true', help='only re-read published arrays and check completeness')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.verify:
        result = verify_shard(config, phase=args.phase, shard=args.shard, limit_games=args.limit_games)
        print(json.dumps(dict(result, shard=args.shard, phase=args.phase)))
        return 0 if result['complete'] else 1
    metrics = label(config, phase=args.phase, shard=args.shard, limit_games=args.limit_games)
    print(json.dumps(metrics))
    return 0 if metrics['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
