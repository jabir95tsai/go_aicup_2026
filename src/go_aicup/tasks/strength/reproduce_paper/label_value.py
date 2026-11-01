"""Label every effective pre-action position with a strong KataGo soft value.

Commands only run via main(). Each worker owns a separate shard directory.
Values are indexed exactly like PaperDataset, never by game_id alone.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import _close_mmap
from go_aicup.features.humansl import KATAGO, gtp
from go_aicup.features.katago_engine import MODEL
from go_aicup.paths import ROOT
from .config import DEFAULT_CONFIG, SPLIT_SEED, read_config
from .dataset import soft_value_target

ENGINE_CONFIG = ROOT / 'configs/strength/se_value_analysis.cfg'
FIELDS = {'p_black': np.float32, 'value': np.float32, 'score_black': np.float32,
          'score_stdev': np.float32, 'visits': np.int32, 'status': np.uint8}
# status: 0 pending, 1 success, 2 failed. Failed values remain NaN, never 0.5.


@contextmanager
def exclusive(directory):
    """Refuse simultaneous writers; a stale lock requires manual inspection."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f'{os.getpid()}\n'.encode())
        yield
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)


def source_data(directory, split):
    record = json.loads((directory / 'manifest.json').read_text())
    if record['identity'].get('split_seed') != SPLIT_SEED:
        raise ValueError('expected shared split_seed=2026')
    names = ['games.parquet'] + [f'{split}_{s}.npy' for s in ('game', 'offsets', 'ply')]
    for name in names:
        if file_hash(directory / name) != record['files'][name]:
            raise ValueError(f'source checksum mismatch: {name}')
    games, offsets, plies = [np.load(directory / name, mmap_mode='r') for name in names[1:]]
    if (len(offsets) != len(games) + 1 or offsets[0] != 0 or offsets[-1] != len(plies)
            or np.any(np.diff(offsets) < 7) or len(np.unique(games)) != len(games)):
        raise ValueError('invalid effective position index')
    table = pd.read_parquet(directory / 'games.parquet').set_index('game_index')
    if not table.index.is_unique or table.duplicated(['rank', 'game_id']).any():
        raise ValueError('duplicate game identity')
    if not (table.loc[games, 'split'] == split).all():
        raise ValueError('split/index mismatch')
    return record, table, games, offsets, plies


def make_query(identifier, moves, colors, plies, visits, rules, komi):
    """turnNumber=t is the position BEFORE move[t]; no synthetic moves."""
    moves, colors, plies = np.asarray(moves), np.asarray(colors), np.asarray(plies)
    if (len(moves) != len(colors) or not np.all((colors == 0) | (colors == 1))
            or not np.all((moves >= 0) & (moves <= 361))
            or not np.all((plies >= 0) & (plies < len(moves)))
            or len(np.unique(plies)) != len(plies)):
        raise ValueError('invalid moves/colors/analyzeTurns')
    # Only supply the prefix needed for the last requested position. No future
    # action is available to the teacher; all earlier history is retained.
    end = int(plies.max())
    return {'id': identifier,
            'moves': [['BW'[int(c)], 'pass' if m == 361 else gtp(m)]
                      for m, c in zip(moves[:end], colors[:end])],
            'analyzeTurns': [int(p) for p in plies], 'maxVisits': visits,
            'rules': rules, 'komi': komi, 'boardXSize': 19, 'boardYSize': 19,
            'overrideSettings': {'reportAnalysisWinratesAs': 'BLACK',
                                 'ignorePreRootHistory': False}}


def parse_root(result, color):
    root = result['rootInfo']
    probability, score = float(root['winrate']), float(root['scoreLead'])
    value = float(soft_value_target(probability, color))
    visits = root['visits']
    if type(visits) is not int or not 0 < visits <= np.iinfo(np.int32).max or not np.isfinite(score):
        raise ValueError('invalid teacher visits or score')
    stdev = float(root.get('scoreStdev', np.nan))
    if 'scoreStdev' in root and (not np.isfinite(stdev) or stdev < 0):
        raise ValueError('invalid scoreStdev')
    return probability, value, score, stdev, visits


class Engine:
    """One game in flight, bounded response wait and clean process teardown."""
    def __init__(self, binary, model, config, log, timeout):
        self.timeout = timeout
        self.log = log.open('ab')
        try:
            self.process = subprocess.Popen(
                [str(binary), 'analysis', '-config', str(config), '-model', str(model),
                 '-override-config', 'reportAnalysisWinratesAs=BLACK,ignorePreRootHistory=false'],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        except Exception:
            self.log.close()
            raise
        self.responses = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def analyze(self, query):
        self.process.stdin.write((json.dumps(query) + '\n').encode())
        self.process.stdin.flush()
        pending = set(query['analyzeTurns'])
        deadline = time.monotonic() + self.timeout
        while pending:
            try:
                line = self.responses.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty as error:
                raise TimeoutError('KataGo game query timed out') from error
            if line is None:
                raise RuntimeError('KataGo exited before finishing query')
            result = json.loads(line)
            if result.get('id') != query['id']:
                raise ValueError('unexpected KataGo response id')
            if 'error' in result:
                raise RuntimeError(str(result['error']))
            if 'warning' in result and 'rootInfo' not in result:
                continue
            if result.get('isDuringSearch', False):
                continue
            ply = result['turnNumber']
            if ply not in pending:
                raise ValueError('unexpected/duplicate KataGo turnNumber')
            pending.remove(ply)
            yield int(ply), result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=5)
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


def label(config, args):
    directory = Path(config['paths']['dataset_dir'])
    record, table, games, offsets, plies = source_data(directory, args.split)
    cache = Path(config['paths']['cache_dir'])
    if file_hash(cache / 'manifest.json') != record['identity']['cache_manifest_sha256']:
        raise ValueError('board cache source mismatch')
    if config['teacher'].get('target') != 'per_effective_pre_action_position':
        raise ValueError('teacher target must be per_effective_pre_action_position')
    visits = args.visits if args.visits is not None else config['teacher']['visits']
    if type(visits) is not int or visits < 1:
        raise ValueError('set positive --visits explicitly; production budget is not frozen')
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards or args.retries < 0:
        raise ValueError('invalid shard/retry settings')
    if args.timeout <= 0 or (args.limit_games is not None and args.limit_games < 1):
        raise ValueError('timeout and limit-games must be positive')
    if args.limit_games is not None and args.split != 'train':
        raise ValueError('pilot selection is training-only')
    # The registered strong network is deliberately required: arbitrary renamed
    # HumanSL weights cannot pass a filename-only check.
    if file_hash(args.model) != file_hash(MODEL):
        raise ValueError('teacher must be the registered kata1 strong network')
    rows = np.arange(len(games), dtype=np.int64)
    if args.limit_games is not None:
        rows = rows[:args.limit_games]  # fixed source order, independent of visits
    rows = rows[args.shard::args.num_shards]
    if not len(rows):
        raise ValueError('empty shard')
    local_offsets = np.r_[0, np.cumsum(offsets[rows + 1] - offsets[rows])]
    count = int(local_offsets[-1])
    identity = {'version': 1, 'target': 'per_effective_pre_action_position',
                'source_manifest_sha256': file_hash(directory / 'manifest.json'),
                'source_build_id': record['build_id'], 'split': args.split,
                'num_shards': args.num_shards, 'limit_games': args.limit_games,
                'engine_sha256': file_hash(args.katago), 'model_sha256': file_hash(args.model),
                'engine_config_sha256': file_hash(args.engine_config),
                'code_sha256': file_hash(Path(__file__)),
                'dataset_code_sha256': file_hash(Path(__file__).with_name('dataset.py')),
                'rules': config['teacher']['rules'], 'komi': config['teacher']['komi'],
                'visits': visits, 'winrate_perspective': 'BLACK',
                'value_perspective': 'designated_player', 'ignorePreRootHistory': False}
    out = args.out / args.split / f'shard_{args.shard:05d}'
    arrays, engine = {}, None
    def check_export_state():
        if (out.parent / '.writer.lock').exists():
            raise RuntimeError('export is active; do not start shard workers')
        if (out.parent / 'value.npy').exists():
            raise ValueError('labels already published; use a new output directory')
    check_export_state()
    with exclusive(out):
        check_export_state()
        manifest = out / 'manifest.json'
        expected = {'identity': identity, 'shard': args.shard, 'positions': count,
                    'rows_sha256': object_hash(rows.tolist())}
        if manifest.exists():
            if json.loads(manifest.read_text()) != expected:
                raise ValueError('teacher/source/sharding changed; choose a new output directory')
        else:
            if any(p.name != '.writer.lock' for p in out.iterdir()):
                raise ValueError('incomplete initialization; inspect directory before retrying')
            for name, dtype in FIELDS.items():
                array = np.lib.format.open_memmap(out / f'{name}.npy', mode='w+', dtype=dtype, shape=(count,))
                array[:] = 0 if name in ('status', 'visits') else np.nan
                array.flush()
                _close_mmap(array)
            write_json(manifest, expected)
        try:
            for name, dtype in FIELDS.items():
                arrays[name] = np.load(out / f'{name}.npy', mmap_mode='r+')
                if arrays[name].dtype != dtype or arrays[name].shape != (count,):
                    raise ValueError(f'invalid {name} array')
            if np.any(arrays['status'] > 2):
                raise ValueError('unknown label status')
            for start in range(0, count, 1_000_000):
                stop = min(count, start + 1_000_000)
                good = arrays['status'][start:stop] == 1
                p = arrays['p_black'][start:stop][good]
                v = arrays['value'][start:stop][good]
                if (not np.all(np.isfinite(p) & (p >= 0) & (p <= 1))
                        or not np.all(np.isfinite(v) & (np.abs(v) <= 1))
                        or not np.all(np.isfinite(arrays['score_black'][start:stop][good]))
                        or not np.all(arrays['visits'][start:stop][good] > 0)):
                    raise ValueError('corrupted successful teacher values')
            started = last_report = time.monotonic()
            produced = 0
            current_shard, move_array, color_array = None, None, None
            try:
                for local_game, row in enumerate(rows):
                    game = int(games[row])
                    meta = table.loc[game]
                    span = slice(int(local_offsets[local_game]), int(local_offsets[local_game + 1]))
                    effective = plies[offsets[row]:offsets[row + 1]]
                    missing = np.flatnonzero(arrays['status'][span] != 1)
                    if not len(missing):
                        continue
                    shard = int(meta.shard_id)
                    if shard != current_shard:
                        for old in (move_array, color_array):
                            if old is not None:
                                _close_mmap(old)
                        move_array = np.load(cache / f'shard_{shard:05d}_moves.npy', mmap_mode='r')
                        color_array = np.load(cache / f'shard_{shard:05d}_colors.npy', mmap_mode='r')
                        current_shard = shard
                    begin, end = int(meta.local_offset), int(meta.local_offset + meta.length)
                    moves, colors = move_array[begin:end], color_array[begin:end]
                    if (len(moves) != int(meta.length) or not np.all(colors[effective] == meta.player_color)
                            or np.any(effective < record['identity']['positions']['min_ply'])):
                        raise ValueError('effective position/color mismatch')
                    for attempt in range(args.retries + 1):
                        missing = np.flatnonzero(arrays['status'][span] != 1)
                        if not len(missing):
                            break
                        lookup = {int(effective[i]): span.start + int(i) for i in missing}
                        completed = []
                        try:
                            if engine is None:
                                engine = Engine(args.katago, args.model, args.engine_config,
                                                out / 'katago_stderr.log', args.timeout)
                            query = make_query(f'{game}:{attempt}', moves, colors, effective[missing],
                                               visits, identity['rules'], identity['komi'])
                            for ply, result in engine.analyze(query):
                                index = lookup[ply]
                                values = parse_root(result, int(colors[ply]))
                                for name, value in zip(('p_black', 'value', 'score_black', 'score_stdev', 'visits'), values):
                                    arrays[name][index] = value
                                completed.append(index)
                        except (OSError, RuntimeError, ValueError, KeyError, TypeError, TimeoutError) as error:
                            with (out / 'failures.jsonl').open('a') as log:
                                log.write(json.dumps({'game_index': game, 'rank': str(meta['rank']),
                                    'game_id': str(meta.game_id), 'attempt': attempt,
                                    'plies': list(lookup), 'error': str(error)}) + '\n')
                            if engine is not None:
                                engine.close()
                                engine = None
                        finally:
                            # Commit data before marking success. Interrupted writes
                            # are retried; existing successful positions are skipped.
                            for name in FIELDS:
                                if name != 'status':
                                    arrays[name].flush()
                            arrays['status'][list(lookup.values())] = 2
                            arrays['status'][completed] = 1
                            arrays['status'].flush()
                            produced += len(completed)
                    now = time.monotonic()
                    if now - last_report >= 30:
                        print(json.dumps({'games_scanned': local_game + 1, 'new_positions': produced,
                                          'positions_per_second': produced / (now - started)}), flush=True)
                        last_report = now
            finally:
                for old in (move_array, color_array):
                    if old is not None:
                        _close_mmap(old)
            success = int(np.count_nonzero(arrays['status'] == 1))
            write_json(out / 'summary.json', {'positions': count, 'success': success,
                'failed_or_pending': count - success, 'new_positions': produced,
                'seconds': time.monotonic() - started,
                'files': {f'{name}.npy': file_hash(out / f'{name}.npy') for name in FIELDS}})
            if success != count:
                raise RuntimeError(f'{count - success} positions missing; rerun same command to retry')
        finally:
            if engine is not None:
                engine.close()
            for array in arrays.values():
                _close_mmap(array)
    return out


def validate_values(path, count):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if values.dtype != np.float32 or values.shape != (count,):
            raise ValueError('invalid published value shape/dtype')
        for begin in range(0, count, 1_000_000):
            block = values[begin:begin + 1_000_000]
            if not np.all(np.isfinite(block) & (np.abs(block) <= 1)):
                raise ValueError('invalid published values')
    finally:
        _close_mmap(values)


def cleanup_export(target):
    """Remove only this command's artifacts after a verified publication.

    Keep the commit marker until last so interrupted cleanup can be resumed.
    Caller holds the split lock and every existing shard lock.
    """
    allowed = {f'{name}.npy' for name in FIELDS} | {
        'manifest.json', 'summary.json', 'failures.jsonl', 'katago_stderr.log',
        '.writer.lock', 'manifest.json.tmp', 'summary.json.tmp'}
    shards = list(target.glob('shard_*'))
    for shard in shards:
        if shard.is_symlink() or not shard.is_dir():
            raise ValueError(f'unexpected shard path: {shard}')
        if any(p.name not in allowed or not p.is_file() or p.is_symlink() for p in shard.iterdir()):
            raise ValueError(f'unexpected file in shard: {shard}')
    known = {'value.npy', 'labels_manifest.json', 'labels_manifest.json.tmp',
             'value.tmp.npy', '.writer.lock'} | {p.name for p in shards}
    if any(p.name not in known for p in target.iterdir()):
        raise ValueError('unexpected output files; refusing unrelated cleanup')
    for shard in shards:
        shutil.rmtree(shard)
    for name in ('value.tmp.npy', 'labels_manifest.json.tmp', 'labels_manifest.json'):
        (target / name).unlink(missing_ok=True)


def export_labels(config, args):
    """Publish a complete vector, then retain only value.npy in this split."""
    directory = Path(config['paths']['dataset_dir'])
    record, table, games, offsets, plies = source_data(directory, args.split)
    target = args.out / args.split
    with exclusive(target), ExitStack() as locks:
        destination = target / 'value.npy'
        marker = target / 'labels_manifest.json'
        for shard in sorted(target.glob('shard_*')):
            if not shard.is_dir() or shard.is_symlink():
                raise ValueError('unexpected shard path')
            locks.enter_context(exclusive(shard))
        if marker.exists():
            published = json.loads(marker.read_text())
            identity = published['identity']
            if not destination.exists() and (target / 'value.tmp.npy').exists():
                temporary = target / 'value.tmp.npy'
                if file_hash(temporary) != published['value_sha256']:
                    raise ValueError('uncommitted value checksum mismatch')
                validate_values(temporary, len(plies))
                os.replace(temporary, destination)
            if (identity['source_manifest_sha256'] != file_hash(directory / 'manifest.json')
                    or identity['split'] != args.split or not published['all_success']
                    or published['positions'] != len(plies)
                    or published['value_sha256'] != file_hash(destination)):
                raise ValueError('published labels conflict or checksum mismatch')
            validate_values(destination, len(plies))
            cleanup_export(target)
            return destination
        if destination.exists():
            # Finalized files intentionally carry no provenance sidecar.
            validate_values(destination, len(plies))
            if any(p.name not in ('value.npy', '.writer.lock') for p in target.iterdir()):
                raise ValueError('uncommitted publication; preserve shards for inspection')
            return destination
        paths = sorted(target.glob('shard_*/manifest.json'))
        if not paths:
            raise ValueError('no shards')
        manifests = [json.loads(p.read_text()) for p in paths]
        identity = manifests[0]['identity']
        if (identity['source_manifest_sha256'] != file_hash(directory / 'manifest.json')
                or identity['split'] != args.split or identity['limit_games'] is not None):
            raise ValueError('wrong source/split or pilot-only labels')
        if len(paths) != identity['num_shards'] or sorted(m['shard'] for m in manifests) != list(range(len(paths))):
            raise ValueError('missing/duplicate shards')
        temporary = target / 'value.tmp.npy'
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(plies),))
        try:
            for path, manifest in zip(paths, manifests):
                if manifest['identity'] != identity:
                    raise ValueError('cannot mix teacher identities')
                rows = np.arange(len(games))[manifest['shard']::identity['num_shards']]
                if manifest['rows_sha256'] != object_hash(rows.tolist()):
                    raise ValueError('shard row identity mismatch')
                summary = json.loads((path.parent / 'summary.json').read_text())
                for name, digest in summary['files'].items():
                    if file_hash(path.parent / name) != digest:
                        raise ValueError(f'shard checksum mismatch: {name}')
                values = np.load(path.parent / 'value.npy', mmap_mode='r')
                status = np.load(path.parent / 'status.npy', mmap_mode='r')
                try:
                    expected_count = int(np.sum(offsets[rows + 1] - offsets[rows]))
                    if (values.shape != (expected_count,) or status.shape != values.shape
                            or values.dtype != np.float32 or not np.all(status == 1)
                            or not np.all(np.isfinite(values) & (np.abs(values) <= 1))):
                        raise ValueError('incomplete/invalid labels; retry failed shards first')
                    cursor = 0
                    for row in rows:
                        size = int(offsets[row + 1] - offsets[row])
                        output[offsets[row]:offsets[row + 1]] = values[cursor:cursor + size]
                        cursor += size
                finally:
                    _close_mmap(values)
                    _close_mmap(status)
            output.flush()
        finally:
            _close_mmap(output)
        validate_values(temporary, len(plies))
        # Commit metadata first: a crash before the rename leaves shards intact.
        write_json(marker, {'identity': identity, 'positions': len(plies),
                            'value_sha256': file_hash(temporary), 'all_success': True})
        os.replace(temporary, destination)
        cleanup_export(target)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid'), required=True)
    parser.add_argument('--export', action='store_true', help='merge complete shards without starting KataGo')
    parser.add_argument('--visits', type=int)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit-games', type=int, help='fixed first N training games for a pilot')
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=600, help='seconds per game query')
    parser.add_argument('--katago', type=Path, default=KATAGO)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--engine-config', type=Path, default=ENGINE_CONFIG)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    print(export_labels(config, args) if args.export else label(config, args))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
