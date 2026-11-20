"""TRAIN/monitor per-move HumanSL *profile* policy cache (H counterpart of the C cache).

Reuses policy_residual.persistent_cache unchanged (selection checks, ZIP/replay
input pipeline, atomic shards, resume, capacity and completion markers). Only the
teacher conditioning differs: instead of the 10 learned C v2 vectors e[rank, own/opp],
each position is evaluated under the official HumanSL metadata of 9 profiles
(rank_10k ... rank_4d, cpp makeBasicRankProfile: both players at that rank),
encoded for the colour to move, exactly as the engine H features.

The overrides are installed at import time so that the spawned GPU workers, which
re-import this module, see them as well. Shard schema: logp FP16 [n, 9, 362],
q FP32 [n, 9], target/side/offsets/game_index as the C cache. Same komi (HD0 6.5,
HD1 0.5) and legal mask (empty, not ko, plus pass) as the C cache.

Before extraction, a GPU parity check compares the vector-indexed teacher with an
unmodified HumanSL that encodes the metadata rows itself (one HD0 and one HD1 game).

    python -m go_aicup.tasks.strength.policy_h.extract [--validate-only]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.features.humansl import PROFILES as ALL_PROFILES
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_residual import persistent_cache as pc

PROFILES = ('rank_10k', 'rank_7k', 'rank_5k', 'rank_3k', 'rank_1k', 'rank_1d', 'rank_2d', 'rank_3d', 'rank_4d')
P = len(PROFILES)
SELECTION = pc.BASE / 'policy_cnn_h/cached_AB_mondev_seed2027_h9'
CONFIG = ROOT / 'configs/strength/policy_h_cache.json'
assert set(PROFILES) <= set(ALL_PROFILES)


def profile_rows(profiles=PROFILES):
    """Metadata rows [2 colours (black, white), P, 192] for the colour to move."""
    from go_aicup.features.humansl_torch import _katago, profile_metadata
    board, _, _, _ = _katago()
    return np.asarray([[profile_metadata(name).get_metadata_row(nextPlayer=pla, boardArea=361)
                        for name in profiles] for pla in (board.Board.BLACK, board.Board.WHITE)],
                      dtype=np.float32)


class ProfilePolicy(torch.nn.Module):
    """Frozen HumanSL whose metadata-encoder output is the encoded profile row [colour, profile]."""

    def __init__(self, device, profiles=PROFILES):
        super().__init__()
        from go_aicup.features.humansl_rankvec import load_frozen
        self.net, _ = load_frozen(device)   # instance encoder replaced by the identity
        self.register_buffer('vectors', self.encode(device, profiles))

    def encode(self, device, profiles=PROFILES):
        """Original metadata encoder (class forward; the instance one is the identity) on the profile rows."""
        encoder = self.net.metadata_encoder
        rows = torch.as_tensor(profile_rows(profiles), device=device)
        with torch.no_grad():
            return type(encoder).forward(encoder, rows.reshape(-1, rows.shape[-1]), None).reshape(2, len(profiles), -1).float()

    def forward(self, x, g, profile, colour):
        return self.net(x, g, input_meta=self.vectors[colour, profile])[0][0][:, 0, :].float()


def validate_shard(path, identity, rows):
    if not path.exists():
        return None
    marker = json.loads((path / 'complete.json').read_text())
    if (marker['identity'] != identity or marker['games'] != pc.object_hash(rows.game_index.tolist())
            or marker['positions'] != int(rows.length.sum()) or set(marker['files']) != set(pc.FIELDS)):
        raise ValueError(f'cache shard identity mismatch: {path}')
    n, g = int(rows.length.sum()), len(rows)
    shapes = dict(logp=(n, P, 362), q=(n, P), target=(n,), side=(n,), offsets=(g + 1,), game_index=(g,))
    for name, dtype in pc.FIELDS.items():
        filename = path / f'{name}.npy'
        array = np.load(filename, mmap_mode='r', allow_pickle=False)
        if array.shape != shapes[name] or array.dtype != dtype or file_hash(filename) != marker['files'][name]:
            raise ValueError(f'cache shard corruption: {filename}')
    if (not np.array_equal(np.load(path / 'game_index.npy'), rows.game_index)
            or not np.array_equal(np.load(path / 'offsets.npy'), np.r_[0, np.cumsum(rows.length)])):
        raise ValueError('shard offsets/game order changed')
    return marker


def mover_colour(rows, side):
    """0 = black to move, 1 = white: side (1 = opponent) XOR the labelled player's colour."""
    return np.asarray(side, dtype=np.int64) ^ np.repeat(rows.player_color.to_numpy().astype(np.int64),
                                                         rows.length.to_numpy())


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
        targets, colours = upload(target).long(), upload(mover_colour(rows, side))
        logp_out = torch.empty(n, P, 362, dtype=torch.float16, device=device)
        q_out = torch.empty(n, P, dtype=torch.float32, device=device)
        profiles = torch.arange(P, device=device).repeat(size)
        checks = []
        for start in range(0, n, size):
            count = min(size, n - start)
            take = (torch.arange(size, device=device) + start).clamp_max(n - 1)
            xb, gb, cb, tb = x[take], g[take], colours[take], targets[take]
            logits = forward(xb.repeat_interleave(P, 0), gb.repeat_interleave(P, 0),
                             profiles, cb.repeat_interleave(P, 0)).reshape(size, P, 362)
            legal = (xb[:, 0] * (1 - xb[:, 1]) * (1 - xb[:, 2]) * (1 - xb[:, 6])).flatten(1).gt(0)
            legal = torch.cat((legal, torch.ones_like(legal[:, :1])), 1)
            lp = logits.float().masked_fill(~legal[:, None], -torch.inf).log_softmax(-1)
            q = lp.gather(2, tb[:, None, None].expand(-1, P, 1)).squeeze(-1)
            cached = lp[:count].half()
            recovered = cached.float().gather(2, tb[:count, None, None].expand(-1, P, 1)).squeeze(-1)
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
        raise ValueError('played move excluded by the legal mask')
    if checks[3] != 1 or not np.isfinite(checks).all():
        raise ValueError('nonfinite legal policy or exact q')
    h_error = 0.
    target, side = values['target'], values['side']
    for a, b in zip(values['offsets'][:-1], values['offsets'][1:]):
        recovered = values['logp'][np.arange(a, b)[:, None], np.arange(P), target[a:b, None]].astype(np.float32)
        for s in (0, 1):
            keep = side[a:b] == s
            if not keep.any():
                raise ValueError('empty own/opponent game side')
            h_error = max(h_error, float(np.abs(values['q'][a:b][keep].mean(0) - recovered[keep].mean(0)).max()))
    if checks[0] > .01 or checks[1] > .05 or h_error > .002:
        raise ValueError('FP16 precision gate failed')
    return dict(max_mass_error=float(checks[0]), max_q_error=float(checks[1]), max_H_error=h_error)


def worker(rank, games, units, assignments, config, identity, out, ram):
    """persistent_cache.worker with the profile teacher; everything else is the shared code."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from collections import deque
    import multiprocessing as mp
    from go_aicup.features.humansl_torch import _init_worker
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    model = ProfilePolicy(device).requires_grad_(False).eval()
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
        cached_inputs = pc.chunks(games, units, cached, config, identity, model.net.config, ram, pool)
        replay_inputs = pc.chunks(games, units, replayed, config, identity, model.net.config, ram, pool)
        with ThreadPoolExecutor(max_workers=1) as writer, pc.input_pipeline(
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
                pending.append(writer.submit(pc.complete_shard, path, identity, rows, values, checks, events))
            reports.extend(future.result() for future in pending)
    write_json(out / f'worker_{rank}.json', dict(identity=identity, shards=reports, pipeline=stats,
               positions=sum(r['positions'] for r in reports), seconds=time.monotonic() - started,
               gpu=torch.cuda.get_device_name(rank), peak_gpu_gib=torch.cuda.max_memory_allocated(rank) / 2**30))


# Install the H teacher into the shared extraction module (also in spawned workers).
pc.SELECTION = SELECTION
pc.CACHE_ROOT = SELECTION / 'policy'
pc.validate_shard = validate_shard
pc.infer = infer
pc.validate_policy = validate_policy
pc.worker = worker


def parity(out):
    """Indexed profile vectors vs an unmodified HumanSL encoding the rows itself; one HD0 and one HD1 game.

    Run in FP32 (TF32 off): TF32 alone moves per-move log p by up to ~2.4e-3 (A-055), which would
    hide an indexing bug at this tolerance. Negative control: the neighbouring profile must differ clearly.
    (Colour cannot serve as a control: makeBasicRankProfile gives both players the same rank, so the
    black-to-move and white-to-move rows encode identically; the gap is recorded as colour_vector_max_abs.)
    Extraction itself keeps TF32 like the C cache (workers are separate processes)."""
    import pandas as pd
    from go_aicup.features import humansl_torch as ht
    from go_aicup.features.humansl import cache_games
    device = torch.device('cuda', 0)
    teacher = ProfilePolicy(device).eval()   # load_frozen turns TF32 on
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    teacher.vectors = teacher.encode(device)  # re-encode the profile rows in FP32
    from katago.train.load_model import load_model
    reference, _, _ = load_model(str(ht.CKPT), use_swa=False, device=device)
    reference.eval()
    rows_meta = torch.as_tensor(profile_rows(), device=device)
    games = pd.read_parquet(SELECTION / 'supplement_games.parquet')
    games = games[games.role.eq('train')]
    report = []
    for hd in (0, 1):
        row = games[games.handicap.eq(hd)].iloc[0]
        raw = cache_games([row.game_index])[0]
        ht._init_worker(teacher.net.config, .5 if hd else 6.5)
        bins, glob, target = ht.encode_game(raw)
        x = torch.as_tensor(bins, device=device).float()
        g = torch.as_tensor(glob, device=device)
        t = torch.as_tensor(target, device=device).long()
        colour = torch.as_tensor(np.array(raw[1]), device=device).long()
        legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).flatten(1).gt(0)
        legal = torch.cat((legal, torch.ones_like(legal[:, :1])), 1)
        worst, mean, swapped = 0., 0., 0.
        with torch.inference_mode():
            for p in range(P):
                theirs = reference(x, g, input_meta=rows_meta[colour, p])[0][0][:, 0, :].float()
                ours = teacher(x, g, torch.full_like(colour, p), colour)
                wrong = teacher(x, g, torch.full_like(colour, (p + 1) % P), colour)
                a, b, c = (v.masked_fill(~legal, -torch.inf).log_softmax(-1).gather(1, t[:, None])
                           for v in (ours, theirs, wrong))
                worst = max(worst, float((a - b).abs().max()))
                mean += float((a - b).abs().mean()) / P
                swapped = max(swapped, float((c - b).abs().max()))
        report.append(dict(handicap=hd, game_index=int(row.game_index), positions=len(target),
                           max_abs_q=worst, mean_abs_q=mean, neighbour_profile_max_abs_q=swapped))
    colour_gap = float((teacher.vectors[0] - teacher.vectors[1]).abs().max())
    write_json(out / 'parity.json', dict(profiles=PROFILES, precision='fp32 (tf32 off)', games=report,
                                         colour_vector_max_abs=colour_gap))
    if max(r['max_abs_q'] for r in report) > 1e-3 or min(r['neighbour_profile_max_abs_q'] for r in report) < 1e-2:
        raise ValueError(f'profile teacher parity failed: {report}')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    print(json.dumps(dict(stage='parity', games=report, colour_vector_max_abs=colour_gap)), flush=True)


def main():
    validate_only = '--validate-only' in sys.argv[1:]
    config = json.loads(CONFIG.read_text())
    if tuple(config['profiles']) != PROFILES:
        raise ValueError('config profiles differ from the extractor')
    out = ROOT / config['out'] / os.environ.get('SLURM_JOB_ID', 'preflight')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'h_module.json', {str(p.relative_to(ROOT)): file_hash(p)
                                       for p in sorted(Path(__file__).parent.glob('*.py'))})
    if not validate_only:
        parity(out)
    sys.argv = [sys.argv[0], '--config', str(CONFIG), *(['--validate-only'] if validate_only else [])]
    pc.main()


if __name__ == '__main__':
    main()
