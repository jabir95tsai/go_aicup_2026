"""One node: five GPUs share policy extraction in RAM, then train one arm per GPU."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from go_aicup.artifacts import write_json
from go_aicup.paths import ROOT
from .data import allocate, extract_worker, prepare, question_rows
from .model import ARMS
from .train import fit_head, train_worker


def finalize_cache(games, bounds, arrays):
    if not arrays['done'].all():
        raise ValueError('incomplete RAM policy cache')
    means, counts, max_error = [], [], 0.
    for i in range(len(games)):
        a, b = bounds[i:i + 2]
        q = arrays['q'][a:b].numpy()
        side = arrays['side'][a:b].numpy()
        target = arrays['target'][a:b].numpy().astype(np.int64)
        raw = arrays['logp'][a:b].numpy()
        recovered = raw[np.arange(b - a)[:, None], np.arange(10)[None], target[:, None]].astype(np.float32)
        cell, n = [], []
        for s in (0, 1):
            mask = side == s
            if not mask.any():
                raise ValueError('empty game side')
            cell.append(q[mask].mean(0))
            n.append(int(mask.sum()))
            max_error = max(max_error, float(np.abs(q[mask].mean(0) - recovered[mask].mean(0)).max()))
        means.append(np.concatenate(cell)), counts.append(n)
    if max_error > .002:
        raise ValueError(f'cached original C precision changed: {max_error}')
    arrays['c'] = torch.from_numpy(np.asarray(means, dtype=np.float32)).share_memory_()
    arrays['counts'] = torch.tensor(counts).share_memory_()
    return max_error


def summarize(config, out, identity):
    from scripts.strength.rank_hier_head import difference
    results, frames = {}, {}
    for arm in ARMS:
        results[arm] = json.loads((out / arm / 'result.json').read_text())
        marker = json.loads((out / arm / 'complete.json').read_text())
        if marker['identity'] != identity:
            raise ValueError('arm identity mismatch')
        frames[arm] = pd.read_parquet(out / arm / 'predictions.parquet')
    reference = frames['A0']
    for arm, frame in frames.items():
        if not frame[['question_id', 'player_id', 'rank_id', 'k']].equals(
                reference[['question_id', 'player_id', 'rank_id', 'k']]):
            raise ValueError('unpaired dev questions')
        np.testing.assert_allclose(frame.initial_score, reference.initial_score, atol=0, rtol=0)
    rng = np.random.default_rng(config['seed'] + 99)
    comparisons = [('S', 'A0'), ('F', 'S'), ('F', 'F-mask'), ('F', 'A0'),
                   ('F-mask', 'A0'), ('C-conv', 'A0'), ('C-conv', 'S'), ('C-conv', 'F')]
    diffs = {}
    for a, b in comparisons:
        diff = difference(frames[a].score.to_numpy(), frames[b].score.to_numpy(),
                          reference.k.to_numpy(), reference.player_id.to_numpy(), rng)
        diff['gate_pass'] = (diff['equal_k'] >= .005 and diff['prob_positive'] >= .90
                             and min(diff['by_k'].values()) >= -.005)
        diffs[f'{a}-{b}'] = diff
    return {'identity': identity, 'arms': results, 'differences': diffs,
            'F_primary_gate_pass': diffs['F-S']['gate_pass'] and diffs['F-F-mask']['gate_pass'],
            'limitations': ['one training seed; historically reused dev',
                            'selected active players with >=20 same-colour HD<=1 games',
                            'no H/HD flags; shared rank-balanced head differs from production B2',
                            'C vectors may have seen monitoring players; overlap disclosed in identity.json']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_residual.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['train_questions'] % 40 or config['monitor_questions'] % 40 or config['dev_questions'] % 40:
        raise ValueError('questions must balance all 40 rank/k cells')
    if (config['updates'] < 1 or config['position_chunk'] < 1 or config['inference_positions'] < 1
            or config['every'] < 1 or config['encoder_workers'] < 1):
        raise ValueError('invalid runtime budget')
    suffix = os.environ.get('SLURM_JOB_ID', 'local')
    out = args.out or ROOT / config['out'] / suffix
    if (out / 'complete.json').exists():
        print(json.dumps({'status': 'already_complete', 'out': str(out)}))
        return
    out.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only and not torch.cuda.device_count() >= len(ARMS):
        raise RuntimeError('five visible GPUs required: parallel extraction and one GPU per arm')
    torch.set_num_threads(4)
    started, free_before = time.monotonic(), shutil.disk_usage(out).free
    if free_before < 512 * 2**20:
        raise RuntimeError('insufficient space for small persistent artifacts')
    write_json(out / 'config.json', config)
    games, questions, identities = prepare(config, out)
    identity = identities['identity']
    if args.prepare_only:
        print(json.dumps(identities['roles']), flush=True)
        return
    arrays, bounds, ram_bytes = allocate(games)
    write_json(out / 'memory_plan.json', {'shared_policy_and_metadata_bytes': ram_bytes,
               'persistent_policy_bytes': 0, 'cache_backend': 'RAM-backed shared-memory tmpfs',
               'swap_policy': 'site tmpfs/OS memory policy; no application policy files',
               'gpu_count': len(ARMS), 'gpu_names': [torch.cuda.get_device_name(i) for i in range(len(ARMS))]})
    # Compilation cache also stays in RAM; OS removes it at normal context exit.
    with tempfile.TemporaryDirectory(prefix=f'policy-code-{suffix}-', dir='/dev/shm') as code_dir:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = code_dir
        os.environ['TRITON_CACHE_DIR'] = str(Path(code_dir) / 'triton')
        mp.spawn(extract_worker, args=(len(ARMS), games, bounds, arrays, config, out), nprocs=len(ARMS))
        error = finalize_cache(games, bounds, arrays)
        write_json(out / 'extraction_complete.json', {'identity': identity, 'positions': int(bounds[-1]),
                   'cached_C_mean_max_abs_error': error, 'seconds': time.monotonic() - started})
        train_rows = question_rows(games, questions['train'])
        initial = fit_head(arrays['c'].numpy(), questions['train'], train_rows)
        torch.save(initial, out / 'initial_head.pt')
        # Raw mean C is tiny; the full policy remains exclusively in shared RAM.
        np.savez(out / 'raw_c.npz', game_index=games.game_index.to_numpy(), c=arrays['c'].numpy())
        mp.spawn(train_worker, args=(games, bounds, arrays, questions, initial, config, identity, out),
                 nprocs=len(ARMS))
    result = summarize(config, out, identity)
    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    result['resources'] = {'wall_seconds': time.monotonic() - started, 'allocated_gpus': len(ARMS),
                          'estimated_allocated_gpu_hours': len(ARMS) * (time.monotonic() - started) / 3600,
                          'shared_cache_bytes': ram_bytes, 'persistent_bytes_before_final_report': used,
                          'filesystem_free_before': free_before, 'filesystem_free_after': shutil.disk_usage(out).free}
    write_json(out / 'result.json', result)
    write_json(out / 'complete.json', {'identity': identity, 'status': 'complete'})
    arrays.clear()
    gc.collect()
    print(json.dumps({'status': 'complete', 'out': str(out), 'persistent_mib': used / 2**20,
                      'differences': result['differences']}), flush=True)


if __name__ == '__main__':
    main()
