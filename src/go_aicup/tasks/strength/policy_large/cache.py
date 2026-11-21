"""Whole-game VRAM shards, bounded board replay and one teacher pass per family."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import gc
import multiprocessing as mp
import threading
import time

import numpy as np
import pandas as pd
import torch
from go_aicup.artifacts import file_hash, write_json
from go_aicup.paths import ROOT
from .stopping import ABORT

_INIT_LOCK = threading.Lock()
_TABLE = _SHARDS = _NET_CONFIG = None
_LAST_HD = None
BOARD_BYTES = 91


def teacher_forward(fn, *args):
    # FX tracing patches process-wide state, so it must not overlap other teacher calls.
    with _INIT_LOCK:
        return fn(*args)


def split_games(lengths, parts):
    lengths = np.asarray(lengths, dtype=np.int64)
    if parts < 1 or len(lengths) < parts or (lengths <= 0).any():
        raise ValueError('invalid whole-game shard sizes')
    bounds = np.r_[0, np.cumsum(lengths)]
    cuts = [0]
    for i in range(1, parts):
        target = bounds[-1] * i / parts
        a = int(np.searchsorted(bounds, target))
        candidates = [min(len(lengths)-(parts-i), max(cuts[-1]+1, x)) for x in (a-1, a)]
        cuts.append(min(candidates, key=lambda x: abs(bounds[x]-target)))
    return list(zip(cuts, cuts[1:]+[len(lengths)]))


@dataclass
class Shard:
    first: int
    last: int
    logp: torch.Tensor
    q: torch.Tensor
    target: torch.Tensor
    side: torch.Tensor
    board: torch.Tensor | None = None
    reservation: torch.Tensor | None = None


def pack_mover_stones(inputs):
    """Binary planes 1 and 2, row-major and big-endian with zero padding."""
    stones = inputs[:, 1:3].reshape(len(inputs), 722).to(torch.uint8)
    stones = torch.nn.functional.pad(stones, (0, 6)).reshape(len(inputs), BOARD_BYTES, 8)
    shifts = torch.arange(7, -1, -1, device=inputs.device, dtype=torch.uint8)
    return (stones << shifts).sum(dim=-1, dtype=torch.uint8)


def profile_offsets(q, side):
    """Mean per-move logsumexp across profiles, separately for each game side."""
    normalizer = q.float().logsumexp(dim=-1)
    return torch.stack([normalizer[side.eq(player_side)].mean() for player_side in (0, 1)])


class PolicyCache:
    def __init__(self, games, shards, means, counts, offsets=None):
        self.games, self.shards = games, shards
        self.bounds = np.r_[0, np.cumsum(games.length.to_numpy(dtype=np.int64))]
        self.arrays = {'c': torch.as_tensor(means), 'counts': torch.as_tensor(counts)}
        self.offsets = offsets
        if offsets is not None and (offsets.shape != (len(games), 2) or not np.isfinite(offsets).all()):
            raise ValueError('invalid game-side profile offsets')
        previous = 0
        for shard in shards:
            if shard.first != previous or shard.last <= previous:
                raise ValueError('VRAM shard coverage is invalid')
            n = self.bounds[shard.last] - self.bounds[shard.first]
            p = means.shape[1] // 2
            if shard.logp.shape != (n,p,362) or shard.q.shape != (n,p) or shard.target.shape != (n,) or shard.side.shape != (n,):
                raise ValueError('VRAM shard shape mismatch')
            if shard.board is not None and (shard.board.shape != (n, BOARD_BYTES)
                                            or shard.board.dtype != torch.uint8):
                raise ValueError('invalid packed mover-stones cache')
            previous = shard.last
        if previous != len(games) or counts.shape != (len(games),2) or (np.asarray(counts) <= 0).any():
            raise ValueError('missing game or side in cache')

    def for_device(self, device):
        return CacheView(self, torch.device(device))


class CacheView:
    def __init__(self, cache, device):
        self.cache, self.device = cache, device
        self.games, self.bounds, self.arrays = cache.games, cache.bounds, cache.arrays
        self.peer_bytes = 0

    def counts(self, rows):
        return self.arrays['counts'][rows].flatten().to(self.device, dtype=torch.float32)

    def chunks(self, rows, chunk, *, board=False):
        rows = np.asarray(rows)
        if chunk < 1 or rows.ndim != 1 or (rows.size and rows.dtype.kind not in 'iu'):
            raise ValueError('invalid chunk or game indices')
        rows = rows.astype(np.int64)
        if (rows < 0).any() or (rows >= len(self.games)).any():
            raise IndexError('game index out of bounds')
        if board and any(shard.board is None for shard in self.cache.shards):
            raise ValueError('this policy cache has no mover-stone planes')
        indices, owners, used = np.empty(chunk,np.int64), np.empty(chunk,np.int64), 0

        def gather(n):
            ids, own = indices[:n], owners[:n]
            p = self.arrays['c'].shape[1] // 2
            lp = torch.empty(n,p,362,device=self.device,dtype=torch.float16)
            q = torch.empty(n,p,device=self.device,dtype=torch.float32)
            target = torch.empty(n,device=self.device,dtype=torch.long)
            slot = torch.empty(n,device=self.device,dtype=torch.long)
            packed = torch.empty(n,BOARD_BYTES,device=self.device,dtype=torch.uint8) if board else None
            for shard in self.cache.shards:
                begin, end = self.bounds[shard.first], self.bounds[shard.last]
                positions = np.flatnonzero((ids >= begin) & (ids < end))
                if not len(positions):
                    continue
                if shard.q.device != self.device and shard.q.is_cuda and self.device.type == 'cuda':
                    # Dtype conversions for target and side happen during the peer copy.
                    self.peer_bytes += len(positions) * (p * (362 * 2 + 4) + 2 * 8 + BOARD_BYTES * board)
                source = torch.as_tensor(ids[positions]-begin,device=shard.q.device)
                dest = torch.as_tensor(positions,device=self.device)
                lp[dest] = shard.logp[source].to(self.device)
                q[dest] = shard.q[source].to(self.device)
                target[dest] = shard.target[source].to(self.device,dtype=torch.long)
                slot[dest] = shard.side[source].to(self.device,dtype=torch.long) + torch.as_tensor(2*own[positions],device=self.device)
                if board:
                    packed[dest] = shard.board[source].to(self.device)
            return (lp,target,q,slot,packed) if board else (lp,target,q,slot)
        for ordinal,row in enumerate(rows):
            a,b = self.bounds[row:row+2]
            while a < b:
                if ABORT.is_set():
                    raise RuntimeError('another experiment worker failed')
                count = min(b-a,chunk-used)
                indices[used:used+count] = np.arange(a,a+count)
                owners[used:used+count] = ordinal
                used += count
                a += count
                if used == chunk:
                    yield gather(used)
                    used = 0
        if used:
            yield gather(used)


def init_encoder(net_config, board_root=None):
    global _TABLE,_SHARDS,_NET_CONFIG,_BOARD_ROOT,_LAST_HD
    torch.set_num_threads(1)
    _NET_CONFIG = net_config
    _BOARD_ROOT = Path(board_root) if board_root else ROOT/'data/processed/boards/strength_v1'
    _TABLE = pd.read_parquet(_BOARD_ROOT/'games.parquet',
                             columns=['game_index','length','shard_id','local_offset']).set_index('game_index')
    _SHARDS = {}
    _LAST_HD = None


def encode(item):
    global _LAST_HD
    from go_aicup.features.humansl_torch import _init_worker,encode_game
    index,hd,player_color,length = item
    row = _TABLE.loc[index]
    if int(row.length) != length:
        raise ValueError('canonical length differs from board cache')
    if hd != _LAST_HD:
        _init_worker(_NET_CONFIG,.5 if hd else 6.5)
        _LAST_HD = hd
    sid = int(row.shard_id)
    if sid not in _SHARDS:
        prefix = _BOARD_ROOT/f'shard_{sid:05d}'
        _SHARDS[sid] = (np.load(f'{prefix}_moves.npy',mmap_mode='r'),np.load(f'{prefix}_colors.npy',mmap_mode='r'))
    a,b = int(row.local_offset),int(row.local_offset+row.length)
    moves,colors = (v[a:b] for v in _SHARDS[sid])
    bins,glob,target = encode_game((moves,colors))
    bits = np.packbits(bins.reshape(len(bins),-1),axis=1)
    return bits,glob,target.astype(np.int16),(np.asarray(colors,dtype=np.uint8)^player_color)


def teacher_profiles(family, config):
    from go_aicup.tasks.strength.policy_h.extract import PROFILES

    return tuple(config.get('profiles_H', PROFILES)) if family == 'H' else tuple(range(10))


class TeacherWorker:
    """Keep one teacher module on its execution GPU across whole-game work ranges."""

    def __init__(self, family, config, device):
        from go_aicup.tasks.strength.policy_h.extract import ProfilePolicy
        from go_aicup.tasks.strength.humansl_finetune import RankPolicy

        self.family, self.device = family, torch.device(device)
        self.profiles = teacher_profiles(family, config)
        self.first_forward = True
        started = time.monotonic()
        with torch.cuda.device(self.device), torch.inference_mode(), _INIT_LOCK:
            if family == 'H':
                teacher = ProfilePolicy(self.device, self.profiles)
            else:
                arrays = torch.load(ROOT/'outputs/rank/humansl/cvec_v2/learn/vectors.pt',
                                    map_location=self.device, weights_only=False)
                teacher = RankPolicy(self.device, arrays['vectors'])
            self.teacher = teacher.eval().requires_grad_(False)
            self.compiled = torch.compile(teacher, dynamic=False) if config['compile_teacher'] else None
        self.setup_seconds = time.monotonic() - started


class TeacherStore:
    """Own resident result shards while execution devices fill disjoint game ranges."""

    def __init__(self, family, games, config, plane_batches, nominal_lengths=None, *,
                 resident_bytes=None, reset_peaks=True):
        self.family, self.games, self.plane_batches = family, games, plane_batches
        self.started = time.monotonic()
        self.devices = [torch.device('cuda', i) for i in config['family_devices'][family]]
        if len(self.devices) != 4:
            raise ValueError('each teacher family requires four VRAM shards')
        for device in self.devices:
            for other in self.devices:
                if device != other and not torch.cuda.can_device_access_peer(device, other):
                    raise ValueError('NVLink peer access is required for policy shards')
        self.profiles = teacher_profiles(family, config)
        p = len(self.profiles)
        size = config['inference_positions']
        self.size = size[family] if isinstance(size, dict) else size
        self.games_per_batch = config['games_per_batch']
        self.keep_stones = plane_batches is not None and family == 'H' and p == 15
        self.bytes_per_position = p * (362 * 2 + 4) + 3 + BOARD_BYTES * self.keep_stones
        lengths = games.length.to_numpy(dtype=np.int64)
        self.bounds = np.r_[0, np.cumsum(lengths)]
        self.ranges = split_games(lengths, len(self.devices))
        nominal = lengths if nominal_lengths is None else np.asarray(nominal_lengths, dtype=np.int64)
        full_bounds = np.r_[0, np.cumsum(nominal)]
        full_ranges = split_games(nominal, len(self.devices))
        self.nominal_card_bytes = tuple(
            int(full_bounds[last] - full_bounds[first]) * self.bytes_per_position for first, last in full_ranges
        )
        self.means = np.zeros((len(games), 2 * p), np.float32)
        self.counts = np.zeros((len(games), 2), np.int64)
        self.offsets = np.full((len(games), 2), np.nan, np.float32) if plane_batches is not None else None
        self.seen = np.zeros(len(games), np.uint8)
        self.claim_lock = threading.Lock()
        other_bytes = (0,) * len(self.devices) if resident_bytes is None else resident_bytes
        self.shards = []
        for device, (first, last), needed, existing in zip(
            self.devices, self.ranges, self.nominal_card_bytes, other_bytes, strict=True,
        ):
            with torch.cuda.device(device):
                if reset_peaks and plane_batches is not None:
                    torch.cuda.reset_peak_memory_stats(device)
                n = int(self.bounds[last] - self.bounds[first])
                padding = needed - n * self.bytes_per_position
                if padding < 0:
                    raise ValueError(f'{family}/{device}: sampled shard exceeds its nominal size')
                free, total = torch.cuda.mem_get_info(device)
                workspace = 16 * 2**30
                if needed + workspace > free or needed + existing + workspace > total * config['cache_fraction']:
                    raise ValueError(f'{family}/{device}: insufficient VRAM, cache={needed + existing} free={free} total={total}')
                board = torch.empty(n, BOARD_BYTES, device=device, dtype=torch.uint8) if self.keep_stones else None
                reservation = torch.zeros(padding, device=device, dtype=torch.uint8) if padding else None
                self.shards.append(Shard(first, last, torch.empty(n, p, 362, device=device, dtype=torch.float16),
                    torch.empty(n, p, device=device, dtype=torch.float32), torch.empty(n, device=device, dtype=torch.int16),
                    torch.empty(n, device=device, dtype=torch.uint8), board, reservation))
                if plane_batches is not None:
                    print({'stage':'teacher_cache','family':family,'device':str(device),'positions':n,
                           'nominal_positions':needed // self.bytes_per_position,'reserved_padding_bytes':padding,
                           'device_free_bytes':free,'device_total_bytes':total},flush=True)

    def process(self, worker, shard_index, first, last, stop, *, batches=None):
        """Compute a whole-game range and synchronize its resident writes before returning."""
        from go_aicup.features.humansl_rankvec import unpack

        shard = self.shards[shard_index]
        if (worker.family != self.family or worker.profiles != self.profiles
                or not shard.first <= first < last <= shard.last):
            raise ValueError('teacher work range or profile order differs from its resident shard')
        device = worker.device
        remote = device != shard.q.device
        if remote and not torch.cuda.can_device_access_peer(device, shard.q.device):
            raise ValueError('NVLink peer access is required for helper writes')
        with self.claim_lock:
            if self.seen[first:last].any():
                raise ValueError('teacher work range was already claimed')
            self.seen[first:last] = 2
        if batches is None:
            batches = self.plane_batches(first, last, self.games_per_batch)
        p, size = len(self.profiles), self.size
        maxima = np.zeros(3)
        cursor = first
        compiled_count = eager_count = 0
        input_seconds = inference_seconds = 0.
        first_forward_seconds = None
        with torch.cuda.device(device), torch.inference_mode():
            while not stop.is_set():
                input_started = time.monotonic()
                batch = next(batches, None)
                compute_started = time.monotonic()
                input_seconds += compute_started - input_started
                if batch is None:
                    break
                rows, parts = batch
                end = cursor + len(rows)
                if (not rows or end > last or list(rows) != list(range(cursor, end)) or len(parts) != len(rows)
                        or any(len(a[2]) != self.bounds[i+1] - self.bounds[i] for i, a in zip(rows, parts))):
                    raise ValueError('teacher input does not cover its whole-game range in order')
                bits, glob, target, side = [np.concatenate([a[j] for a in parts]) for j in range(4)]

                def upload(a):
                    return torch.from_numpy(np.ascontiguousarray(a)).pin_memory().to(device, non_blocking=True)

                x, g, t, s = unpack(upload(bits)), upload(glob), upload(target).long(), upload(side).long()
                n = len(target)
                logp = torch.empty(n, p, 362, device=device, dtype=torch.float16)
                q = torch.empty(n, p, device=device, dtype=torch.float32)
                error = torch.zeros(2, device=device)
                packed_stones = pack_mover_stones(x) if self.keep_stones else None
                profile_side = s
                if self.family == 'H':
                    colors = np.repeat(self.games.iloc[cursor:end].player_color.to_numpy(dtype=np.int64),
                                       np.diff(self.bounds[cursor:end+1]))
                    profile_side = s ^ torch.as_tensor(colors, device=device)
                for a in range(0, n, size):
                    b = min(a + size, n)
                    xb, gb = x[a:b], g[a:b]
                    conditions = torch.arange(p, device=device).repeat(b-a)
                    use_compiled = worker.compiled is not None and b-a == size
                    fn = worker.compiled if use_compiled else worker.teacher
                    call_started = time.monotonic()
                    logits = teacher_forward(fn, xb.repeat_interleave(p, 0), gb.repeat_interleave(p, 0), conditions,
                                             profile_side[a:b].repeat_interleave(p, 0)).reshape(b-a, p, 362)
                    if use_compiled:
                        if worker.first_forward:
                            torch.cuda.synchronize(device)
                            first_forward_seconds = time.monotonic() - call_started
                            worker.first_forward = False
                        compiled_count += 1
                    else:
                        eager_count += 1
                    legal = (xb[:,0] * (1-xb[:,1]) * (1-xb[:,2]) * (1-xb[:,6])).flatten(1).gt(0)
                    legal = torch.cat((legal, torch.ones_like(legal[:,:1])), 1)
                    if not legal.gather(1, t[a:b,None]).all():
                        raise ValueError('actual move excluded by legal mask')
                    lp = logits.float().masked_fill(~legal[:,None], -torch.inf).log_softmax(-1)
                    exact = lp.gather(2, t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    compressed = lp.half()
                    if not torch.isfinite(exact).all() or not (torch.isfinite(compressed) | ~legal[:,None]).all():
                        raise ValueError('nonfinite teacher policy')
                    recovered = compressed.float().gather(2, t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    error[0] = torch.maximum(error[0], (compressed.float().exp().sum(-1)-1).abs().max())
                    error[1] = torch.maximum(error[1], (recovered-exact).abs().max())
                    logp[a:b], q[a:b] = compressed, exact
                a = 0
                mean_error = 0.
                for i, part in zip(rows, parts):
                    b = a + len(part[2])
                    pair, recovered_pair, side_counts = [], [], []
                    rec = logp[a:b].float().gather(2, t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    for player_side in (0, 1):
                        mask = s[a:b].eq(player_side)
                        if not mask.any():
                            raise ValueError('empty own/opponent side')
                        pair.append(q[a:b][mask].mean(0))
                        recovered_pair.append(rec[mask].mean(0))
                        side_counts.append(int(mask.sum()))
                    average = torch.cat(pair)
                    if self.offsets is None:
                        self.means[i] = average.cpu().numpy()
                    else:
                        reduced = torch.cat((average, profile_offsets(q[a:b], s[a:b]))).cpu().numpy()
                        self.means[i], self.offsets[i] = reduced[:-2], reduced[-2:]
                    self.counts[i] = side_counts
                    mean_error = max(mean_error, float((average - torch.cat(recovered_pair)).abs().max()))
                    a = b
                errors = [*error.cpu().tolist(), mean_error]
                if errors[0] > .01 or errors[1] > .05 or errors[2] > .002:
                    raise ValueError(f'policy FP16 precision gate failed: {errors}')
                dst = slice(int(self.bounds[cursor]-self.bounds[shard.first]),
                            int(self.bounds[end]-self.bounds[shard.first]))
                shard.logp[dst].copy_(logp, non_blocking=True)
                shard.q[dst].copy_(q, non_blocking=True)
                shard.target[dst].copy_(t, non_blocking=True)
                shard.side[dst].copy_(s, non_blocking=True)
                if self.keep_stones:
                    shard.board[dst].copy_(packed_stones, non_blocking=True)
                maxima = np.maximum(maxima, errors)
                inference_seconds += time.monotonic() - compute_started
                cursor = end
                completed = cursor - first
                if completed == len(rows) or completed % 128 == 0:
                    print({'stage':'teacher','family':self.family,'device':str(device),
                           'owner':str(shard.q.device),'first':first,'games':completed,
                           'seconds':time.monotonic()-self.started},flush=True)
            if stop.is_set():
                raise RuntimeError('another teacher worker failed')
            if cursor != last:
                raise ValueError('teacher input ended before its work range')
            # The peer copy is queued on the source stream. The fit barrier publishes only synchronized ranges.
            torch.cuda.synchronize(device)
            counters = {'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)}
        self.seen[first:last] = 1
        positions = int(self.bounds[last] - self.bounds[first])
        setup_seconds, worker.setup_seconds = worker.setup_seconds, 0.
        return {
            'device':str(device),'owner':str(shard.q.device),'first':first,'last':last,
            'games':last-first,'errors':maxima.tolist(),'positions':positions,
            'peer_bytes':positions * self.bytes_per_position if remote else 0,
            'resident_padding_bytes':shard.reservation.numel() if shard.reservation is not None else 0,
            'nominal_cache_bytes':self.nominal_card_bytes[shard_index],
            'input_wait_seconds':input_seconds,'inference_seconds':inference_seconds,
            'setup_seconds':setup_seconds,'first_compiled_call_seconds':first_forward_seconds,
            'compiled_batches':compiled_count,'eager_batches':eager_count,**counters,
        }

    def finish(self):
        if not np.all(self.seen == 1) or self.counts.sum() != self.bounds[-1]:
            raise ValueError('teacher must visit every selected position exactly once')
        return PolicyCache(self.games, self.shards, self.means, self.counts, self.offsets)


def publish_teacher(out, family, policy_cache, report):
    if policy_cache.offsets is not None:
        path = Path(out)/f'offset_{family}.npy'
        temporary = path.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, policy_cache.offsets, allow_pickle=False)
        temporary.replace(path)
        report['profile_offsets'] = {'file':path.name,'sha256':file_hash(path),'sides':['own','opponent']}
    write_json(Path(out)/f'teacher_{family}.json', report)


def infer_family(family, games, config, out, cancel=None, plane_batches=None, nominal_lengths=None):
    from go_aicup.features.humansl_torch import CKPT

    started = time.monotonic()
    store = TeacherStore(family, games, config, plane_batches, nominal_lengths)
    stop = cancel if cancel is not None else threading.Event()
    workers = max(1, config['encoder_workers']//2)
    if plane_batches is None:
        with _INIT_LOCK:
            checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
            net_config = checkpoint['config']
            del checkpoint
        supply = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                     initializer=init_encoder, initargs=(net_config, config.get('board_root')))
    else:
        supply = nullcontext()

    def replay_batches(first, last, pool):
        indices, pending = iter(range(first, last)), deque()

        def submit():
            i = next(indices, None)
            if i is not None:
                row = games.iloc[i]
                pending.append((i, pool.submit(encode, (int(row.game_index), int(row.handicap),
                                                        int(row.player_color), int(row.length)))))

        for _ in range(max(2*store.games_per_batch, workers)):
            submit()
        while pending:
            rows, parts = [], []
            for _ in range(min(len(pending), store.games_per_batch)):
                i, future = pending.popleft()
                rows.append(i)
                parts.append(future.result())
                submit()
            yield rows, parts

    def work(index, pool):
        first, last = store.ranges[index]
        worker = TeacherWorker(family, config, store.devices[index])
        batches = replay_batches(first, last, pool) if plane_batches is None else None
        return store.process(worker, index, first, last, stop, batches=batches)

    with supply as pool, ThreadPoolExecutor(len(store.devices)) as threads:
        futures = [threads.submit(work, index, pool) for index in range(len(store.devices))]
        try:
            reports = [future.result() for future in as_completed(futures)]
        except BaseException:
            stop.set()
            raise
    policy_cache = store.finish()
    report = {'family':family,'games':len(games),'positions':int(store.bounds[-1]),'seconds':time.monotonic()-started,
        'teacher_passes_per_game':1,'workers':sorted(reports,key=lambda worker: worker['device']),
        'cache_bytes':int(store.bounds[-1])*store.bytes_per_position,'profiles':list(store.profiles),
        'microbatch_positions':store.size,'nominal_cache_bytes':sum(store.nominal_card_bytes),
        'max_nominal_card_cache_bytes':max(store.nominal_card_bytes),
        'input':'zstd_planes' if plane_batches is not None else 'board_replay'}
    publish_teacher(out, family, policy_cache, report)
    gc.collect()
    return policy_cache, report
