"""H five-arm dev evaluation; two GPU extraction workers, shared C reports and gates."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

from go_aicup.artifacts import file_hash
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_residual import dev_eval as shared
from . import five_arms as h
from .extract import P, PROFILES, ProfilePolicy, infer, profile_rows, validate_policy


def teacher_identity():
    from go_aicup.features.humansl_torch import CKPT
    recorded = json.loads(shared.policy_zip._Reader(h.ARCHIVE).read('policy/identity.json'))['sources']
    name = str(CKPT.relative_to(ROOT))
    if file_hash(CKPT) != recorded[name]:
        raise ValueError('H teacher differs from the TRAIN/monitor cache')
    if recorded.get(name) is None or json.loads(shared.policy_zip._Reader(h.ARCHIVE).read('policy/identity.json'))['config']['profiles'] != list(PROFILES):
        raise ValueError('H profile metadata differs from the TRAIN/monitor cache')
    from go_aicup.artifacts import object_hash
    return {name: recorded[name], 'profiles': list(PROFILES),
            'metadata_rows': object_hash(profile_rows().tolist())}


def check_policy(values, checks):
    metrics = validate_policy(values, checks)
    metrics['max_C_error'] = metrics.pop('max_H_error')
    return metrics


def extract_worker(rank, dev, config, directory):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bounds, arrays, stats = shared.extract(
        dev, config, torch.device('cuda', rank), conditions=P,
        model_factory=ProfilePolicy, infer_policy=infer, check_policy=check_policy)
    for name, value in arrays.items():
        np.save(Path(directory) / f'{rank}_{name}.npy', value, allow_pickle=False)
    stats['max_H_error'] = stats.pop('max_C_error')
    return bounds, stats


def extract(dev, config, device):
    if torch.cuda.device_count() != 2:
        raise ValueError('H dev evaluation requires exactly two visible GPUs')
    needed = int(dev.length.sum()) * (P * 362 * 2 + P * 4 + 3)
    if shutil.disk_usage('/dev/shm').free < needed * 1.1:
        raise ValueError('insufficient shared RAM for dev policy arrays')
    started = time.monotonic()
    parts = [dev.iloc[a:b].reset_index(drop=True)
             for a, b in ((0, len(dev) // 2), (len(dev) // 2, len(dev)))]
    with tempfile.TemporaryDirectory(prefix='policy_h_dev_', dir='/dev/shm') as directory:
        with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as pool:
            futures = [pool.submit(extract_worker, rank, part, config, directory)
                       for rank, part in enumerate(parts)]
            reports = [future.result() for future in futures]
        arrays = {name: np.concatenate([np.load(Path(directory) / f'{rank}_{name}.npy',
                                              mmap_mode='r', allow_pickle=False) for rank in range(2)])
                  for name in ('logp', 'q', 'target', 'side')}
    bounds = np.r_[0, np.cumsum(dev.length.to_numpy())].astype(np.int64)
    if any(len(value) != bounds[-1] for value in arrays.values()):
        raise ValueError('two-GPU extraction position count mismatch')
    for part, (part_bounds, _) in zip(parts, reports):
        np.testing.assert_array_equal(part_bounds, np.r_[0, np.cumsum(part.length)])
    stats = dict(seconds=time.monotonic() - started, positions=int(bounds[-1]), games=len(dev),
                 workers=[stats for _, stats in reports],
                 peak_gpu_gib=max(stats['peak_gpu_gib'] for _, stats in reports))
    return bounds, arrays, stats


def main():
    h.use_h_archive()
    original_selection, original_load = shared.policy_zip.selection, shared.policy_zip.load
    # Their default path arguments were bound to the C archive when imported.
    shared.policy_zip.selection = lambda name='games': original_selection(name, h.ARCHIVE)
    shared.policy_zip.load = lambda **kwargs: original_load(h.ARCHIVE, **kwargs)
    shared.build, shared.BranchHead = h.build, h.BranchHead
    shared.evaluate_role, shared.finalize_cache = h.evaluate_role, h.finalize_cache
    shared.teacher_identity, shared.extract = teacher_identity, extract
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', str(ROOT / 'configs/strength/policy_h_five_arms_dev.json')])
    shared.main()


if __name__ == '__main__':
    main()
