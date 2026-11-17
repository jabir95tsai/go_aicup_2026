"""Stream reused/replayed HumanSL inputs into atomic, resumable TRAIN/monitor policy shards."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import queue
import shutil
import tempfile
import time
import threading
import zipfile

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from .data import VECTORS

BASE = ROOT / 'data/processed/strength_features/humansl'
SELECTION = BASE / 'policy_cnn/cached_AB_mondev_seed2027'
C4X = BASE / 'c4x'
CACHE_ROOT = SELECTION / 'policy'
FIELDS = {'logp': np.float16, 'q': np.float32, 'target': np.int16, 'side': np.uint8,
          'offsets': np.int64, 'game_index': np.int64}


def zip_sample(archive, name, first=0, count=0):
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(stream)
        if fortran or dtype.hasobject:
            raise ValueError('unsupported packed array layout')
        width = int(np.prod(shape[1:])) * dtype.itemsize
        if first < 0 or first + count > shape[0]:
            raise ValueError('packed sample outside array')
        if count:
            stream.seek(first * width, 1)
            value = np.frombuffer(stream.read(count * width), dtype=dtype).reshape(count, *shape[1:])
        else:
            value = None
        return shape, dtype, value


def validate_inputs(rows, bits, glob, target, side, raw):
    n = int(rows.length.sum())
    if (bits.shape != (n, 993) or bits.dtype != np.uint8 or glob.shape != (n, 19)
            or glob.dtype != np.float32 or target.shape != (n,) or side.shape != (n,)
            or not np.isfinite(glob).all() or not np.isin(side, [0, 1]).all()
            or not np.issubdtype(target.dtype, np.integer) or not ((target >= 0) & (target <= 361)).all()):
        raise ValueError('invalid input planes or incomplete positions')
    offset = 0
    for row, (moves, colors) in zip(rows.itertuples(), raw):
        end = offset + row.length
        if (len(moves) != row.length or not np.array_equal(colors, np.arange(row.length) % 2)
                or not np.array_equal(target[offset:end], moves)
                or not np.array_equal(side[offset:end], colors != row.player_color)):
            raise ValueError(f'pack/replay move, color or order mismatch: {row.game_index}')
        offset = end


def prepare(config, out):
    from go_aicup.features.humansl_torch import CKPT, KATAGO_PY, _init_worker, encode_game
    from go_aicup.features.humansl import cache_games
    net_config = torch.load(CKPT, map_location='cpu', weights_only=False)['config']
    canonical = pd.read_parquet(ROOT / 'data/processed/reproduce_paper/seed2026/games.parquet')
    length_by_game = canonical.set_index('game_index').length
    parity_samples = []
    selected = json.loads((SELECTION / 'identity.json').read_text())
    if json.loads((SELECTION / 'complete.json').read_text())['status'] != 'complete':
        raise ValueError('selection incomplete')
    for name, digest in selected['artifact_hashes'].items():
        if file_hash(SELECTION / name) != digest:
            raise ValueError(f'selection checksum mismatch: {name}')
    for name, digest in selected['source_hashes'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f'selection source changed: {name}')
    games = pd.read_parquet(SELECTION / 'games.parquet')
    learn = games[games.role.isin(['train', 'monitor'])].reset_index(drop=True)
    if len(learn) != 24000 or learn.groupby('player_id').role.nunique().gt(1).any():
        raise ValueError('expected 20,000 TRAIN and 4,000 monitor games')
    units, coverage = [], set()
    for label in ('learn_A', 'learn_B'):
        archive_path = C4X / f'{label}.zip'
        expected = pd.concat([pd.read_parquet(C4X / f'{label}_hd{hd}.parquet')
                              for hd in (0, 1)]).set_index('game_index')
        seen = set()
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(n for n in archive.namelist() if n.endswith('/meta.json')):
                prefix = name.removesuffix('meta.json')
                hd = int(re.search(r'/pack_hd([01])_', prefix)[1])
                details = json.loads(archive.read(name))
                table = pd.read_parquet(io.BytesIO(archive.read(prefix + 'games.parquet')))
                if (not table.game_index.is_unique or set(table.game_index) & seen
                        or details['komi'] != (.5 if hd else 6.5)
                        or details['games'] != len(table) or 'plies' in table):
                    raise ValueError(f'incompatible pack manifest: {prefix}')
                reference = expected.loc[table.game_index]
                if any(not np.array_equal(table[c], reference[c]) for c in ('rank_id', 'player_color')):
                    raise ValueError(f'pack metadata mismatch: {prefix}')
                seen.update(table.game_index)
                n = int(length_by_game.loc[table.game_index].sum())
                if n != details['positions']:
                    raise ValueError('pack position count differs from canonical')
                schema = dict(bits=((n, 993), np.uint8), globals=((n, 19), np.float32),
                              target=((n,), np.int16), own=((n,), np.bool_), game=((n,), np.int32))
                for field, (shape, dtype) in schema.items():
                    actual_shape, actual_dtype, _ = zip_sample(archive, prefix + field + '.npy')
                    if actual_shape != shape or actual_dtype != dtype:
                        raise ValueError(f'incompatible pack array: {prefix}{field}')
                part = learn[learn.game_index.isin(table.game_index)]
                if not part.empty and (label, hd) not in parity_samples:
                    row = part.iloc[0]
                    slot = int(np.flatnonzero(table.game_index.to_numpy() == row.game_index)[0])
                    first = int(length_by_game.loc[table.game_index.iloc[:slot]].sum())
                    actual = {name: zip_sample(archive, prefix + name + '.npy', first, int(row.length))[2]
                              for name in ('bits', 'globals', 'target', 'own')}
                    bins = np.unpackbits(actual['bits'], axis=1)[:, :22 * 361].reshape(-1, 22, 19, 19)
                    raw = cache_games([row.game_index])
                    validate_inputs(part.iloc[:1], actual['bits'], actual['globals'], actual['target'],
                                    (~actual['own']).astype(np.uint8), raw)
                    _init_worker(net_config, .5 if hd else 6.5)
                    for got, wanted in zip((bins, actual['globals'], actual['target']), encode_game(raw[0])):
                        np.testing.assert_array_equal(got, wanted, err_msg='CPU ZIP/replay parity failed')
                    parity_samples.append((label, hd))
                if not part.empty:
                    if not part.reuse_pack.all() or not part.handicap.eq(hd).all():
                        raise ValueError('pack handicap or selection mismatch')
                    units.append(dict(archive=str(archive_path), prefix=prefix,
                                      games=part.game_index.tolist(), handicap=hd))
                    coverage.update(part.game_index)
        if seen != set(expected.index):
            raise ValueError(f'archive coverage differs from source table: {label}')
    if coverage != set(learn.loc[learn.reuse_pack, 'game_index']):
        raise ValueError('physical ZIP coverage differs from reuse manifest')
    for hd in (0, 1):
        ids = learn.loc[~learn.reuse_pack & learn.handicap.eq(hd), 'game_index'].tolist()
        for start in range(0, len(ids), 256):
            units.append(dict(archive=None, games=ids[start:start + 256], handicap=hd))
    source_paths = [Path(__file__), ROOT / 'scripts/strength/select_cached_policy.py',
                    ROOT / 'src/go_aicup/data/cv.py', ROOT / 'src/go_aicup/artifacts.py',
                    ROOT / 'src/go_aicup/features/humansl.py',
                    ROOT / 'src/go_aicup/features/humansl_torch.py',
                    ROOT / 'src/go_aicup/features/humansl_rankvec.py',
                    ROOT / 'src/go_aicup/tasks/strength/humansl_finetune.py',
                    Path(__file__).with_name('data.py'),
                    ROOT / 'src/go_aicup/tasks/strength/humansl_residual_b2/data.py',
                    *KATAGO_PY.rglob('*.py')]
    board = ROOT / 'data/processed/boards/strength_v1'
    board_meta = pd.read_parquet(board / 'games.parquet')
    if 'game_index' not in board_meta:
        board_meta = board_meta.reset_index()
    check = board_meta.set_index('game_index').loc[games.game_index]
    if any(not np.array_equal(games[c].astype(str), check[c].astype(str))
           for c in ('player_id', 'rank_id', 'player_color', 'length')):
        raise ValueError('canonical and move-store identities differ')
    shards = board_meta.loc[board_meta.game_index.isin(learn.game_index), 'shard_id'].unique()
    inputs = [CKPT, VECTORS, SELECTION / 'identity.json', SELECTION / 'games.parquet',
              ROOT / 'plan.md', board / 'manifest.json', board / 'games.parquet']
    inputs += [C4X / f'{label}.zip' for label in ('learn_A', 'learn_B')]
    inputs += [board / f'shard_{s:05d}_{field}.npy' for s in shards for field in ('moves', 'colors')]
    c_sources = sorted(VECTORS.parent.glob('learn_hd*.parquet'))
    if len(c_sources) != 2:
        raise ValueError('C-vector learning player coverage unavailable')
    c_ids = pd.concat([pd.read_parquet(p) for p in c_sources]).game_index
    seen = set(canonical.loc[canonical.game_index.isin(c_ids), 'player_id'])
    identity = dict(schema=2, config=config, sources={str(p.relative_to(ROOT)): file_hash(p)
                    for p in [*source_paths, *inputs, *c_sources]},
                    selection=selected, units=units, cpu_zip_replay_parity=[list(pair) for pair in parity_samples],
                    input_planes='RAM only; never persisted',
                    c_vector_learning_player_overlap={role: len(set(g.player_id) & seen)
                                                      for role, g in games.groupby('role')})
    identity_id = object_hash(identity)
    root = CACHE_ROOT
    root.mkdir(exist_ok=True)
    if (root / 'identity.json').exists():
        if json.loads((root / 'identity.json').read_text()) != dict(id=identity_id, **identity):
            raise ValueError('cache identity changed; use a new version directory')
    else:
        write_json(root / 'identity.json', dict(id=identity_id, **identity))
    out.mkdir(parents=True, exist_ok=True)
    for p in source_paths:
        target = out / 'source' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    shutil.copyfile(ROOT / 'plan.md', out / 'plan.md')
    write_json(out / 'config.json', config)
    write_json(out / 'identity.json', dict(id=identity_id, **identity))
    return learn, units, identity_id


def specifications(games, units, config):
    lookup = games.set_index('game_index', drop=False)
    for i, unit in enumerate(units):
        for j, start in enumerate(range(0, len(unit['games']), config['games_per_shard'])):
            rows = lookup.loc[unit['games'][start:start + config['games_per_shard']]].reset_index(drop=True)
            yield i, j, rows


def validate_shard(path, identity, rows):
    if not path.exists():
        return None
    marker = json.loads((path / 'complete.json').read_text())
    if (marker['identity'] != identity or marker['games'] != object_hash(rows.game_index.tolist())
            or marker['positions'] != int(rows.length.sum()) or set(marker['files']) != set(FIELDS)):
        raise ValueError(f'cache shard identity mismatch: {path}')
    n, g = int(rows.length.sum()), len(rows)
    shapes = dict(logp=(n, 10, 362), q=(n, 10), target=(n,), side=(n,),
                  offsets=(g + 1,), game_index=(g,))
    for name, dtype in FIELDS.items():
        filename = path / f'{name}.npy'
        array = np.load(filename, mmap_mode='r', allow_pickle=False)
        if (array.shape != shapes[name] or array.dtype != dtype
                or file_hash(filename) != marker['files'][name]):
            raise ValueError(f'cache shard corruption: {filename}')
    if (not np.array_equal(np.load(path / 'game_index.npy'), rows.game_index)
            or not np.array_equal(np.load(path / 'offsets.npy'), np.r_[0, np.cumsum(rows.length)])):
        raise ValueError('shard offsets/game order changed')
    return marker


def publish_shard(path, identity, rows, values, metrics):
    temporary = path.with_name(path.name + '.partial')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for name, dtype in FIELDS.items():
        np.save(temporary / f'{name}.npy', np.asarray(values[name], dtype=dtype), allow_pickle=False)
    marker = dict(identity=identity, games=object_hash(rows.game_index.tolist()),
                  positions=int(rows.length.sum()), metrics=metrics,
                  files={name: file_hash(temporary / f'{name}.npy') for name in FIELDS})
    write_json(temporary / 'complete.json', marker)
    temporary.rename(path)
    return marker


def encode_supplement(item):
    from go_aicup.features import humansl_torch as ht
    hd, raw = item
    ht._KOMI = .5 if hd else 6.5
    bins, glob, target = ht.encode_game(raw)
    if not np.isin(bins, [0, 1]).all():
        raise ValueError('non-binary replay planes')
    # Pack before IPC and unpack on GPU, reducing both IPC and H2D input bytes eightfold.
    return np.packbits(bins.reshape(len(bins), -1), axis=1), glob, target


def chunks(games, units, local_units, config, identity, net_config, ram, pool):
    from go_aicup.features.humansl_torch import _init_worker, encode_game
    from go_aicup.tasks.strength.humansl_residual_b2.data import MoveStore
    store = MoveStore(ROOT / 'data/processed/boards/strength_v1')
    length_by_game = store.meta.set_index('game_index').length
    lookup = games.set_index('game_index', drop=False)
    for i in local_units:
        unit = units[i]
        todo = []
        for j, start in enumerate(range(0, len(unit['games']), config['games_per_shard'])):
            rows = lookup.loc[unit['games'][start:start + config['games_per_shard']]].reset_index(drop=True)
            path = CACHE_ROOT / f'unit_{i:03d}_shard_{j:03d}'
            if validate_shard(path, identity, rows) is None:
                todo.append((path, rows))
        if not todo:
            continue
        if unit['archive']:
            with tempfile.TemporaryDirectory(dir=ram) as tmp:
                pack = Path(tmp)
                with zipfile.ZipFile(unit['archive']) as archive:
                    for name in (*['bits', 'globals', 'target', 'own', 'game'], 'games.parquet', 'meta.json'):
                        member = name if '.' in name else name + '.npy'
                        with archive.open(unit['prefix'] + member) as src, (pack / member).open('wb') as dst:
                            shutil.copyfileobj(src, dst)
                arrays = {name: np.load(pack / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                          for name in ('bits', 'globals', 'target', 'own', 'game')}
                table = pd.read_parquet(pack / 'games.parquet')
                details = json.loads((pack / 'meta.json').read_text())
                n = details['positions']
                lengths = length_by_game.loc[table.game_index].to_numpy()
                bounds = np.r_[0, np.cumsum(lengths)]
                if (n != int(bounds[-1]) or arrays['bits'].shape != (n, 993)
                        or arrays['globals'].shape != (n, 19)
                        or any(arrays[k].shape != (n,) for k in ('target', 'own', 'game'))
                        or not np.array_equal(arrays['game'], np.repeat(np.arange(len(table)), lengths))):
                    raise ValueError('pack position count, schema or game order mismatch')
                slots = pd.Series(np.arange(len(table)), index=table.game_index)
                for number, (path, rows) in enumerate(todo):
                    ids = np.concatenate([np.arange(bounds[s], bounds[s + 1]) for s in slots.loc[rows.game_index]])
                    bits = np.asarray(arrays['bits'][ids])
                    glob = np.asarray(arrays['globals'][ids])
                    target = np.asarray(arrays['target'][ids]).astype(np.int64)
                    side = (~arrays['own'][ids]).astype(np.uint8)
                    raw = [store.game(int(g)) for g in rows.game_index]
                    validate_inputs(rows, bits, glob, target, side, raw)
                    if number == 0:
                        _init_worker(net_config, .5 if unit['handicap'] else 6.5)
                        encoded = encode_game(raw[0])
                        count = int(rows.length.iloc[0])
                        bins = np.unpackbits(bits[:count], axis=1)[:, :22 * 361].reshape(-1, 22, 19, 19)
                        for actual, expected in zip((bins, glob[:count], target[:count]), encoded):
                            np.testing.assert_array_equal(actual, expected, err_msg='ZIP/replay parity failed')
                    yield path, rows, bits, glob, target, side
                del arrays
        else:
            todo = iter(todo)
            pending = deque()

            def submit_next():
                item = next(todo, None)
                if item is not None:
                    path, rows = item
                    raw = [store.game(int(g)) for g in rows.game_index]
                    futures = [pool.submit(encode_supplement, (unit['handicap'], item)) for item in raw]
                    pending.append((path, rows, raw, futures))

            submit_next()
            submit_next()
            while pending:
                path, rows, raw, futures = pending.popleft()
                parts = [future.result() for future in futures]
                # Keep the next two shards in the process pool, including while this shard is consumed.
                submit_next()
                bits, glob, target = [np.concatenate([p[j] for p in parts]) for j in range(3)]
                side = np.concatenate([(colors != color).astype(np.uint8)
                                       for (_, colors), color in zip(raw, rows.player_color)])
                validate_inputs(rows, bits, glob, target, side, raw)
                yield path, rows, bits, glob, target, side


@contextmanager
def input_pipeline(cached, replayed, replay_shards, stats):
    """Independent producers: four cached shards plus all missing replay shards fit in bounded RAM."""
    ready = queue.Queue(maxsize=replay_shards + 6)
    cached_slots = threading.Semaphore(4)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                ready.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def produce(kind, iterator):
        started, count = time.monotonic(), 0
        try:
            for item in iterator:
                if kind == 'cached':
                    while not stop.is_set() and not cached_slots.acquire(timeout=.1):
                        pass
                if stop.is_set() or not put((kind, item, None)):
                    break
                count += 1
        except BaseException as error:
            put((kind, None, error))
        else:
            put((kind, None, None))
        finally:
            iterator.close()
            stats[kind] = dict(shards=count, seconds=time.monotonic() - started)

    def consume():
        finished = 0
        while finished < 2:
            started = time.monotonic()
            kind, item, error = ready.get()
            stats['input_wait_seconds'] += time.monotonic() - started
            if error is not None:
                raise error
            if item is None:
                finished += 1
            else:
                if kind == 'cached':
                    cached_slots.release()
                yield item

    stats['input_wait_seconds'] = 0.
    with ThreadPoolExecutor(max_workers=2) as producers:
        producers.submit(produce, 'cached', cached)
        producers.submit(produce, 'replayed', replayed)
        try:
            yield consume()
        finally:
            stop.set()


def infer(forward, device, rows, bits, glob, target, side, size, copy_stream=None):
    from go_aicup.features.humansl_rankvec import unpack
    n = len(target)
    cuda = device.type == 'cuda'
    events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)) if cuda else None
    if cuda:
        events[0].record()

    def upload(array):
        value = torch.from_numpy(np.ascontiguousarray(array))
        return value.pin_memory().to(device, non_blocking=True) if cuda else value

    with torch.inference_mode():
        x, g = unpack(upload(bits)), upload(glob)
        targets, sides = upload(target).long(), upload(side).long()
        logp_out = torch.empty(n, 10, 362, dtype=torch.float16, device=device)
        q_out = torch.empty(n, 10, dtype=torch.float32, device=device)
        ranks = torch.arange(10, device=device).repeat(size)
        checks = []
        for start in range(0, n, size):
            count = min(size, n - start)
            take = (torch.arange(size, device=device) + start).clamp_max(n - 1)
            xb, gb, sb, tb = x[take], g[take], sides[take], targets[take]
            logits = forward(xb.repeat_interleave(10, 0), gb.repeat_interleave(10, 0),
                             ranks, sb.repeat_interleave(10, 0)).reshape(size, 10, 362)
            legal = (xb[:, 0] * (1 - xb[:, 1]) * (1 - xb[:, 2]) * (1 - xb[:, 6])).flatten(1).gt(0)
            legal = torch.cat((legal, torch.ones_like(legal[:, :1])), 1)
            lp = logits.float().masked_fill(~legal[:, None], -torch.inf).log_softmax(-1)
            q = lp.gather(2, tb[:, None, None].expand(-1, 10, 1)).squeeze(-1)
            cached = lp[:count].half()
            recovered = cached.float().gather(2, tb[:count, None, None].expand(-1, 10, 1)).squeeze(-1)
            checks.append(torch.stack(((cached.float().exp().sum(-1) - 1).abs().max(),
                          (recovered - q[:count]).abs().max(), legal.gather(1, tb[:, None]).all().float(),
                          (torch.isfinite(q).all() &
                           (torch.isfinite(cached) | ~legal[:count, None]).all()).float())))
            logp_out[start:start + count] = cached
            q_out[start:start + count] = q[:count]
        checks = torch.stack(checks)
        summary = torch.cat((checks[:, :2].amax(0), checks[:, 2:].amin(0)))
        values = dict(target=target, side=side, offsets=np.r_[0, np.cumsum(rows.length)],
                      game_index=rows.game_index.to_numpy())
        if cuda:
            if copy_stream is None:
                raise ValueError('CUDA inference requires the dedicated output copy stream')
            events[1].record()
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(events[1])
                for name, value in (('logp', logp_out), ('q', q_out), ('checks', summary)):
                    host = torch.empty_like(value, device='cpu', pin_memory=True)
                    host.copy_(value, non_blocking=True)
                    value.record_stream(copy_stream)
                    if name == 'checks':
                        summary_host = host.numpy()
                    else:
                        values[name] = host.numpy()
                events[2].record(copy_stream)
        else:
            values.update(logp=logp_out.numpy(), q=q_out.numpy())
            summary_host = summary.numpy()
    return values, summary_host, events


def validate_policy(values, checks):
    if checks[2] != 1:
        raise ValueError('played move excluded by C mask')
    if checks[3] != 1 or not np.isfinite(checks).all():
        raise ValueError('nonfinite legal policy or exact q')
    c_error = 0.
    target, side = values['target'], values['side']
    for a, b in zip(values['offsets'][:-1], values['offsets'][1:]):
        recovered = values['logp'][np.arange(a, b)[:, None], np.arange(10), target[a:b, None]].astype(np.float32)
        for s in (0, 1):
            keep = side[a:b] == s
            if not keep.any():
                raise ValueError('empty own/opponent game side')
            c_error = max(c_error, float(np.abs(values['q'][a:b][keep].mean(0) - recovered[keep].mean(0)).max()))
    if checks[0] > .01 or checks[1] > .05 or c_error > .002:
        raise ValueError('FP16 precision gate failed')
    return dict(max_mass_error=float(checks[0]), max_q_error=float(checks[1]), max_C_error=c_error)


def complete_shard(path, identity, rows, values, checks, events):
    started = time.monotonic()
    if events is not None:
        events[2].synchronize()
    metrics = validate_policy(values, checks)
    if events is not None:
        metrics['inference_gpu_ms'] = events[0].elapsed_time(events[1])
        metrics['output_copy_ms'] = events[1].elapsed_time(events[2])
    publish_shard(path, identity, rows, values, metrics)
    validate_shard(path, identity, rows)
    return dict(path=path.name, marker_sha256=file_hash(path / 'complete.json'),
                positions=len(values['target']), metrics=metrics,
                writer_seconds=time.monotonic() - started)


def worker(rank, games, units, assignments, config, identity, out, ram):
    from go_aicup.features.humansl_torch import _init_worker
    from go_aicup.tasks.strength.humansl_finetune import RankPolicy
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    model = RankPolicy(device, torch.load(VECTORS, map_location=device, weights_only=False)['vectors'])
    model.requires_grad_(False).eval()
    forward = torch.compile(model, dynamic=False) if config['compile_teacher'] else model
    copy_stream = torch.cuda.Stream(device=device)
    started, reports, stats = time.monotonic(), [], dict(writer_backpressure_seconds=0.)
    local = assignments[rank]
    cached = [i for i in local if units[i]['archive']]
    replayed = [i for i in local if not units[i]['archive']]
    replay_shards = sum((len(units[i]['games']) + config['games_per_shard'] - 1) // config['games_per_shard']
                        for i in replayed)
    with ProcessPoolExecutor(max_workers=config['encoder_workers'], mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(model.net.config,)) as pool:
        cached_inputs = chunks(games, units, cached, config, identity, model.net.config, ram, pool)
        replay_inputs = chunks(games, units, replayed, config, identity, model.net.config, ram, pool)
        with ThreadPoolExecutor(max_workers=1) as writer, input_pipeline(
                cached_inputs, replay_inputs, replay_shards, stats) as inputs:
            pending = deque()
            for path, rows, bits, glob, target, side in inputs:
                while pending and (pending[0].done() or len(pending) >= 2):
                    waited = time.monotonic()
                    report = pending.popleft().result()
                    stats['writer_backpressure_seconds'] += time.monotonic() - waited
                    reports.append(report)
                    print(json.dumps(dict(gpu=rank, shards=len(reports), positions=report['positions'],
                                          seconds=round(time.monotonic() - started, 1))), flush=True)
                values, checks, events = infer(forward, device, rows, bits, glob, target, side,
                                               config['inference_positions'], copy_stream)
                pending.append(writer.submit(complete_shard, path, identity, rows, values, checks, events))
            reports.extend(future.result() for future in pending)
    write_json(out / f'worker_{rank}.json', dict(identity=identity, shards=reports, pipeline=stats,
               positions=sum(r['positions'] for r in reports), seconds=time.monotonic() - started,
               gpu=torch.cuda.get_device_name(rank), peak_gpu_gib=torch.cuda.max_memory_allocated(rank) / 2**30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_cache.json')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if min(config[k] for k in ('gpus', 'encoder_workers', 'games_per_shard', 'inference_positions')) < 1:
        raise ValueError('invalid extraction budget')
    out = ROOT / config['out'] / os.environ.get('SLURM_JOB_ID', 'preflight')
    SELECTION.mkdir(parents=True, exist_ok=True)
    with (SELECTION / 'policy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        games, units, identity = prepare(config, out)
        root = CACHE_ROOT
        missing = 0
        for i, j, rows in specifications(games, units, config):
            if validate_shard(root / f'unit_{i:03d}_shard_{j:03d}', identity, rows) is None:
                missing += int(rows.length.sum()) * 7283 + 4096
        free = shutil.disk_usage(root).free
        if free < missing * 1.03 + 2 * 2**30:
            raise RuntimeError(f'insufficient disk: need {missing} output bytes plus reserve, free {free}')
        write_json(out / 'capacity.json', dict(positions=int(games.length.sum()), output_bytes=int(games.length.sum()) * 7283,
                   missing_bytes=missing, free_bytes=free, gpus=config['gpus'], input_plane_files=0))
        print(json.dumps(dict(stage='preflight_complete', identity=identity, positions=int(games.length.sum()),
                              output_gib=round(int(games.length.sum()) * 7283 / 2**30, 3))), flush=True)
        if args.validate_only:
            return
        for archive_path in sorted({unit['archive'] for unit in units if unit['archive']}):
            with zipfile.ZipFile(archive_path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError(f'ZIP CRC failure: {archive_path}:{bad}')
        if torch.cuda.device_count() != config['gpus']:
            raise RuntimeError('visible GPU count differs from configured extraction allocation')
        started = time.monotonic()
        assignments, loads = [[] for _ in range(config['gpus'])], [0] * config['gpus']
        lengths = games.set_index('game_index').length
        for i in sorted(range(len(units)), key=lambda i: int(lengths.loc[units[i]['games']].sum()), reverse=True):
            owner = int(np.argmin(loads))
            assignments[owner].append(i)
            loads[owner] += int(lengths.loc[units[i]['games']].sum())
        write_json(out / 'environment.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu_names=[torch.cuda.get_device_name(i) for i in range(config['gpus'])],
                   positions_by_gpu=loads, assignments=assignments))
        packed_input_bytes = int(games.length.sum()) * (993 + 19 * 4 + 8 + 1)
        available = int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        if available < packed_input_bytes * 2 + 32 * 2**30:
            raise RuntimeError('insufficient host RAM for packed input buffers plus extraction reserve')
        # Replayed inputs fit in bounded RAM; ZIP producers keep four additional shards per GPU.
        with tempfile.TemporaryDirectory(prefix='policy-extract-', dir='/dev/shm') as ram:
            os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(Path(ram) / 'inductor')
            os.environ['TRITON_CACHE_DIR'] = str(Path(ram) / 'triton')
            torch.multiprocessing.spawn(worker, args=(games, units, assignments, config, identity, out, ram),
                                        nprocs=config['gpus'], join=True)
        # Writers already re-read and verify every output while other shards are inferred.
        verified = {}
        for rank in range(config['gpus']):
            report = json.loads((out / f'worker_{rank}.json').read_text())
            if report['identity'] != identity:
                raise ValueError('worker verification identity mismatch')
            for shard in report['shards']:
                if shard['path'] in verified:
                    raise ValueError('duplicate worker shard receipt')
                verified[shard['path']] = shard['marker_sha256']
        markers = {}
        for i, j, rows in specifications(games, units, config):
            path = root / f'unit_{i:03d}_shard_{j:03d}'
            digest = file_hash(path / 'complete.json')
            if path.name in verified:
                if digest != verified[path.name]:
                    raise ValueError('verified marker changed after publication')
            else:
                # Resumed shards were not written in this run; verify their contents too.
                validate_shard(path, identity, rows)
            markers[path.name] = digest
        source = json.loads((root / 'identity.json').read_text())
        if any(file_hash(ROOT / name) != digest for name, digest in source['sources'].items()):
            raise ValueError('source changed during extraction; complete marker withheld')
        write_json(root / 'complete.json', dict(identity=identity, status='complete', shards=markers,
                   games=len(games), positions=int(games.length.sum()), roles=['train', 'monitor']))
        write_json(out / 'complete.json', dict(identity=identity, status='complete',
                   seconds=time.monotonic() - started, allocated_gpu_hours=config['gpus'] * (time.monotonic() - started) / 3600))


if __name__ == '__main__':
    main()
