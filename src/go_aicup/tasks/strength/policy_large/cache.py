"""Whole-game VRAM shards, bounded board replay and one teacher pass per family."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import gc
import multiprocessing as mp
import threading
import time

import numpy as np
import pandas as pd
import torch
from go_aicup.artifacts import write_json
from go_aicup.paths import ROOT
from .stopping import ABORT

_INIT_LOCK = threading.Lock()
_TABLE = _SHARDS = _NET_CONFIG = None
_LAST_HD = None


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


class PolicyCache:
    def __init__(self, games, shards, means, counts):
        self.games, self.shards = games, shards
        self.bounds = np.r_[0, np.cumsum(games.length.to_numpy(dtype=np.int64))]
        self.arrays = {'c': torch.as_tensor(means), 'counts': torch.as_tensor(counts)}
        previous = 0
        for shard in shards:
            if shard.first != previous or shard.last <= previous:
                raise ValueError('VRAM shard coverage is invalid')
            n = self.bounds[shard.last] - self.bounds[shard.first]
            p = means.shape[1] // 2
            if shard.logp.shape != (n,p,362) or shard.q.shape != (n,p) or shard.target.shape != (n,) or shard.side.shape != (n,):
                raise ValueError('VRAM shard shape mismatch')
            previous = shard.last
        if previous != len(games) or counts.shape != (len(games),2) or (np.asarray(counts) <= 0).any():
            raise ValueError('missing game or side in cache')

    def for_device(self, device):
        return CacheView(self, torch.device(device))


class CacheView:
    def __init__(self, cache, device):
        self.cache, self.device = cache, device
        self.games, self.bounds, self.arrays = cache.games, cache.bounds, cache.arrays

    def counts(self, rows):
        return self.arrays['counts'][rows].flatten().to(self.device, dtype=torch.float32)

    def chunks(self, rows, chunk):
        rows = np.asarray(rows)
        if chunk < 1 or rows.ndim != 1 or (rows.size and rows.dtype.kind not in 'iu'):
            raise ValueError('invalid chunk or game indices')
        rows = rows.astype(np.int64)
        if (rows < 0).any() or (rows >= len(self.games)).any():
            raise IndexError('game index out of bounds')
        indices, owners, used = np.empty(chunk,np.int64), np.empty(chunk,np.int64), 0
        def gather(n):
            ids, own = indices[:n], owners[:n]
            p = self.arrays['c'].shape[1] // 2
            lp = torch.empty(n,p,362,device=self.device,dtype=torch.float16)
            q = torch.empty(n,p,device=self.device)
            target = torch.empty(n,device=self.device,dtype=torch.long)
            slot = torch.empty(n,device=self.device,dtype=torch.long)
            for shard in self.cache.shards:
                begin, end = self.bounds[shard.first], self.bounds[shard.last]
                positions = np.flatnonzero((ids >= begin) & (ids < end))
                if not len(positions):
                    continue
                source = torch.as_tensor(ids[positions]-begin,device=shard.q.device)
                dest = torch.as_tensor(positions,device=self.device)
                lp[dest] = shard.logp[source].to(self.device)
                q[dest] = shard.q[source].to(self.device)
                target[dest] = shard.target[source].to(self.device,dtype=torch.long)
                slot[dest] = shard.side[source].to(self.device,dtype=torch.long) + torch.as_tensor(2*own[positions],device=self.device)
            return lp,target,q,slot
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


def init_encoder(net_config):
    global _TABLE,_SHARDS,_NET_CONFIG
    torch.set_num_threads(1)
    _NET_CONFIG = net_config
    _TABLE = pd.read_parquet(ROOT/'data/processed/boards/strength_v1/games.parquet').set_index('game_index')
    _SHARDS = {}


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
        prefix = ROOT/f'data/processed/boards/strength_v1/shard_{sid:05d}'
        _SHARDS[sid] = (np.load(f'{prefix}_moves.npy',mmap_mode='r'),np.load(f'{prefix}_colors.npy',mmap_mode='r'))
    a,b = int(row.local_offset),int(row.local_offset+row.length)
    moves,colors = (v[a:b] for v in _SHARDS[sid])
    bins,glob,target = encode_game((moves,colors))
    bits = np.packbits(bins.reshape(len(bins),-1),axis=1)
    return bits,glob,target.astype(np.int16),(np.asarray(colors,dtype=np.uint8)^player_color)


def infer_family(family,games,config,out,cancel=None):
    from go_aicup.features.humansl_torch import CKPT
    from go_aicup.features.humansl_rankvec import unpack
    from go_aicup.tasks.strength.policy_h.extract import ProfilePolicy
    from go_aicup.tasks.strength.humansl_finetune import RankPolicy
    devices = [torch.device('cuda',i) for i in config['family_devices'][family]]
    if len(devices) != 4:
        raise ValueError('each teacher family requires four VRAM shards')
    for device in devices:
        for other in devices:
            if device != other and not torch.cuda.can_device_access_peer(device,other):
                raise ValueError('NVLink peer access is required for policy shards')
    p = 9 if family == 'H' else 10
    bounds = np.r_[0,np.cumsum(games.length.to_numpy(dtype=np.int64))]
    ranges = split_games(games.length,4)
    means,counts = np.zeros((len(games),2*p),np.float32),np.zeros((len(games),2),np.int64)
    seen = np.zeros(len(games),np.uint8)
    with _INIT_LOCK:
        checkpoint = torch.load(CKPT,map_location='cpu',weights_only=False)
        net_config = checkpoint['config']
        del checkpoint
    shards = []
    for device,(first,last) in zip(devices,ranges):
        with torch.cuda.device(device):
            n = int(bounds[last]-bounds[first])
            needed = n*(p*362*2+p*4+3)
            free,total = torch.cuda.mem_get_info(device)
            if needed+16*2**30 > min(free,total*config['cache_fraction']):
                raise ValueError(f'{family}/{device}: insufficient VRAM cache reserve')
            shards.append(Shard(first,last,torch.empty(n,p,362,device=device,dtype=torch.float16),
                torch.empty(n,p,device=device),torch.empty(n,device=device,dtype=torch.int16),
                torch.empty(n,device=device,dtype=torch.uint8)))
    started = time.monotonic()
    stop = cancel if cancel is not None else threading.Event()
    workers = max(1,config['encoder_workers']//2)
    context = mp.get_context('spawn')

    def worker(shard,pool):
        device = shard.q.device
        with torch.cuda.device(device),torch.inference_mode():
            with _INIT_LOCK:
                if family == 'H':
                    teacher = ProfilePolicy(device)
                else:
                    vectors = torch.load(ROOT/'outputs/rank/humansl/cvec_v2/learn/vectors.pt',map_location=device,weights_only=False)['vectors']
                    teacher = RankPolicy(device,vectors)
                teacher.eval().requires_grad_(False)
                forward = torch.compile(teacher,dynamic=False) if config['compile_teacher'] else teacher
            indices = iter(range(shard.first,shard.last))
            pending = deque()
            def submit():
                i = next(indices,None)
                if i is not None:
                    row = games.iloc[i]
                    pending.append((i,pool.submit(encode,(int(row.game_index),int(row.handicap),int(row.player_color),int(row.length)))))
            for _ in range(max(2*config['games_per_batch'],workers)):
                submit()
            maxima = np.zeros(3)
            completed = 0
            while pending:
                if stop.is_set():
                    raise RuntimeError('another teacher worker failed')
                rows,parts = [],[]
                for _ in range(min(len(pending),config['games_per_batch'])):
                    i,future = pending.popleft()
                    rows.append(i); parts.append(future.result()); submit()
                bits,glob,target,side = [np.concatenate([a[j] for a in parts]) for j in range(4)]
                def upload(a):
                    return torch.from_numpy(np.ascontiguousarray(a)).pin_memory().to(device,non_blocking=True)
                x,g,t,s = unpack(upload(bits)),upload(glob),upload(target).long(),upload(side).long()
                n = len(target)
                logp = torch.empty(n,p,362,device=device,dtype=torch.float16)
                q = torch.empty(n,p,device=device)
                error = torch.zeros(2,device=device)
                size = config['inference_positions']
                for a in range(0,n,size):
                    b = min(a+size,n)
                    xb,gb = x[a:b],g[a:b]
                    conditions = torch.arange(p,device=device).repeat(b-a)
                    indices_side = s[a:b]
                    if family == 'H':
                        colors = np.concatenate([np.full(len(part[2]),int(games.iloc[i].player_color),np.int64) for i,part in zip(rows,parts)])
                        indices_side = s[a:b] ^ torch.as_tensor(colors[a:b],device=device)
                    fn = forward if b-a == size else teacher
                    logits = teacher_forward(fn,xb.repeat_interleave(p,0),gb.repeat_interleave(p,0),conditions,
                                             indices_side.repeat_interleave(p,0)).reshape(b-a,p,362)
                    legal = (xb[:,0]*(1-xb[:,1])*(1-xb[:,2])*(1-xb[:,6])).flatten(1).gt(0)
                    legal = torch.cat((legal,torch.ones_like(legal[:,:1])),1)
                    if not legal.gather(1,t[a:b,None]).all():
                        raise ValueError('actual move excluded by legal mask')
                    lp = logits.float().masked_fill(~legal[:,None],-torch.inf).log_softmax(-1)
                    exact = lp.gather(2,t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    compressed = lp.half()
                    if not torch.isfinite(exact).all() or not (torch.isfinite(compressed)|~legal[:,None]).all():
                        raise ValueError('nonfinite teacher policy')
                    recovered = compressed.float().gather(2,t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    error[0] = torch.maximum(error[0],(compressed.float().exp().sum(-1)-1).abs().max())
                    error[1] = torch.maximum(error[1],(recovered-exact).abs().max())
                    logp[a:b],q[a:b] = compressed,exact
                a = 0
                mean_error = 0.
                for i,part in zip(rows,parts):
                    length = len(part[2]); b = a+length
                    pair,recovered_pair,side_counts = [],[],[]
                    rec = logp[a:b].float().gather(2,t[a:b,None,None].expand(-1,p,1)).squeeze(-1)
                    for player_side in (0,1):
                        mask = s[a:b].eq(player_side)
                        if not mask.any():
                            raise ValueError('empty own/opponent side')
                        pair.append(q[a:b][mask].mean(0)); recovered_pair.append(rec[mask].mean(0));side_counts.append(int(mask.sum()))
                    means[i] = torch.cat(pair).cpu().numpy()
                    counts[i] = side_counts
                    mean_error = max(mean_error,float((torch.cat(pair)-torch.cat(recovered_pair)).abs().max()))
                    dst = slice(int(bounds[i]-bounds[shard.first]),int(bounds[i+1]-bounds[shard.first]))
                    shard.logp[dst].copy_(logp[a:b]);shard.q[dst].copy_(q[a:b]);shard.target[dst].copy_(t[a:b]);shard.side[dst].copy_(s[a:b])
                    seen[i] += 1
                    a = b
                errors = [*error.cpu().tolist(),mean_error]
                if errors[0] > .01 or errors[1] > .05 or errors[2] > .002:
                    raise ValueError(f'policy FP16 precision gate failed: {errors}')
                maxima = np.maximum(maxima,errors)
                completed += len(rows)
                if completed % 128 == 0:
                    print({'stage':'teacher','family':family,'device':str(device),'games':completed,'seconds':time.monotonic()-started},flush=True)
            del teacher,forward
            torch.cuda.synchronize(device)
            return {'device':str(device),'games':completed,'errors':maxima.tolist()}
    with ProcessPoolExecutor(workers,mp_context=context,initializer=init_encoder,initargs=(net_config,)) as pool:
        with ThreadPoolExecutor(4) as threads:
            futures = [threads.submit(worker,shard,pool) for shard in shards]
            try:
                reports = [f.result() for f in as_completed(futures)]
            except BaseException:
                stop.set()
                raise
    if not np.all(seen == 1) or counts.sum() != bounds[-1]:
        raise ValueError('teacher must visit every selected position exactly once')
    cache = PolicyCache(games,shards,means,counts)
    report = {'family':family,'games':len(games),'positions':int(bounds[-1]),'seconds':time.monotonic()-started,
        'teacher_passes_per_game':1,'workers':reports,'cache_bytes':int(bounds[-1])*(p*362*2+p*4+3)}
    write_json(Path(out)/f'teacher_{family}.json',report)
    gc.collect()
    return cache,report
