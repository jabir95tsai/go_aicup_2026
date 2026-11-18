"""Two independent GPU workers read the original policy ZIPs; no teacher inference."""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_h import five_arms as h
from go_aicup.tasks.strength.policy_residual import five_arms as c, policy_zip
from go_aicup.tasks.strength.policy_residual.data import question_rows

ARMS = ('A0', 'R-HC', 'Z-HC')
MODELS = ('CNN-R10', 'CNN-Z32')


def read_json(path):
    return json.loads(Path(path).read_text())


def archive_table(side, name):
    archive, prefix = ((h.ARCHIVE, h.NAME) if side == 'H' else
                       (policy_zip.ARCHIVE, policy_zip.NAME))
    with zipfile.ZipFile(archive) as z:
        return pd.read_parquet(io.BytesIO(z.read(f'{prefix}/{name}.parquet')))


def align(ids, values, wanted, width):
    ids, values = np.asarray(ids), np.asarray(values)
    if (ids.ndim != 1 or len(np.unique(ids)) != len(ids) or values.shape != (len(ids), width)
            or not np.isfinite(values).all() or set(ids) != set(wanted)):
        raise ValueError('feature IDs, dimensions, coverage or finite values invalid')
    return values[pd.Index(ids).get_indexer(wanted)]


def read_features(path, wanted, width):
    with np.load(path, allow_pickle=False) as data:
        return align(data['game_index'], data['c'], wanted, width)


def split_features(raw, residual, combined, profiles):
    n = len(raw)
    if (raw.shape != (n, 2 * profiles) or residual.shape != raw.shape
            or combined.shape != (n, 2 * (profiles + 32))):
        raise ValueError('unexpected frozen CNN feature dimensions')
    sides = combined.reshape(n, 2, profiles + 32)
    np.testing.assert_allclose(sides[:, :, :profiles].reshape(n, -1), raw, atol=2e-6, rtol=2e-6)
    return {'q': raw, 'r': residual, 'z': sides[:, :, profiles:].reshape(n, 64)}


def assemble(left, right):
    arrays = {'A0': np.column_stack((left['q'], right['q'])),
              'R-HC': np.column_stack((left['r'], left['q'], right['r'], right['q'])),
              'Z-HC': np.column_stack((left['z'], left['q'], right['z'], right['q']))}
    for arm, width in zip(ARMS, (38, 76, 166)):
        if arrays[arm].shape[1] != width or not np.isfinite(arrays[arm]).all():
            raise ValueError(f'invalid {arm} features')
    return arrays


def question_values(games, questions, values):
    return np.stack([values[rows].mean(0) for rows in question_rows(games, questions)]).astype(np.float32)


def preflight(config):
    games = archive_table('C', 'games')
    other = archive_table('H', 'games')
    pd.testing.assert_frame_equal(games, other)
    questions = {}
    for role in ('train', 'monitor', 'dev'):
        questions[role] = archive_table('C', f'{role}_questions')
        pd.testing.assert_frame_equal(questions[role], archive_table('H', f'{role}_questions'))
    c.validate_questions(games, questions)
    sources = {}
    for side, module in (('H', h), ('C', c)):
        run, dev = ROOT / config['runs'][side], ROOT / config['dev_runs'][side]
        run_record = read_json(run / 'identity.json')
        identity = run_record['identity']
        dev_id = read_json(dev / 'identity.json')
        marker = read_json(dev / 'complete.json')
        if (dev_id['run_identity'] != identity or dev_id['smoke_players']
                or marker['status'] != 'complete' or marker['identity'] != dev_id['identity']):
            raise ValueError(f'{side} dev does not belong to the completed full source run')
        for arm in MODELS:
            if read_json(run / arm / 'complete.json') != {'identity': identity, 'status': 'complete'}:
                raise ValueError(f'{side}/{arm} not complete')
            payload = torch.load(run / arm / 'model.pt', map_location='cpu', weights_only=False)
            if payload['identity'] != identity or payload['step'] != read_json(dev / arm / 'dev_result.json')['selected_step']:
                raise ValueError(f'{side}/{arm} checkpoint differs from dev source')
            for path in (run / arm / 'model.pt', dev / arm / 'dev_features.npz'):
                sources[str(path.relative_to(ROOT))] = file_hash(path)
        sources[str((dev / 'A0/dev_features.npz').relative_to(ROOT))] = file_hash(dev / 'A0/dev_features.npz')
        archive, prefix = ((h.ARCHIVE, h.NAME) if side == 'H' else (policy_zip.ARCHIVE, policy_zip.NAME))
        with zipfile.ZipFile(archive) as z:
            marker = json.loads(z.read(f'{prefix}/policy/complete.json'))
            expected = h.IDENTITY if side == 'H' else policy_zip.IDENTITY
            if (marker['identity'] != expected or marker['status'] != 'complete'
                    or run_record['cache_identity'] != expected):
                raise ValueError(f'{side} policy archive identity changed')
            for name in ('policy/complete.json', 'games.parquet', 'train_questions.parquet',
                         'monitor_questions.parquet', 'dev_questions.parquet'):
                sources[f'{side}:{name}'] = policy_zip._sha(z.read(f'{prefix}/{name}'))
    return games, questions, sources


@torch.no_grad()
def frozen_features(models, cache, chunk, profiles):
    """Share each gathered policy chunk across both models and retain only per-game sums."""
    rows = np.arange(len(cache.games))
    residual = torch.zeros(len(rows) * 2, profiles, device=cache.device)
    latent = torch.zeros(len(rows) * 2, 32, device=cache.device)
    for lp, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            r = models['CNN-R10'](lp, target, q)
            z = models['CNN-Z32'](lp, target, q)
        residual.index_add_(0, slot, (r - q).float())
        latent.index_add_(0, slot, z[:, profiles:].float())
    counts = cache.counts(rows)[:, None]
    raw = cache.arrays['c'].cpu().numpy()
    result = {'q': raw, 'r': raw + (residual / counts).reshape(-1, profiles * 2).cpu().numpy(),
              'z': (latent / counts).reshape(-1, 64).cpu().numpy()}
    if not all(np.isfinite(x).all() for x in result.values()):
        raise ValueError('nonfinite frozen features')
    return result


def extract_worker(rank, config, identity, out):
    side = ('H', 'C')[rank]
    directory = Path(out) / 'features' / side
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / 'complete.json'
    if complete.exists():
        marker = read_json(complete)
        if marker['identity'] != identity:
            raise ValueError('feature cache identity changed')
        if file_hash(directory / 'features.npz') != marker['sha256']:
            raise ValueError('feature cache checksum changed')
        return
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda', rank)
    module, profiles = (h, 9) if side == 'H' else (c, 10)
    # H's archive adapter mutates globals only inside this isolated spawn worker.
    if side == 'H':
        h.use_h_archive()
    started = time.monotonic()
    cache, error = module.load_cache(device, config['zip_threads'], on_device=True)
    load_seconds = time.monotonic() - started
    models = {}
    for arm in MODELS:
        payload = torch.load(ROOT / config['runs'][side] / arm / 'model.pt', map_location=device, weights_only=False)
        model = module.build(arm, config['seed']).to(device)
        model.load_state_dict(payload['model'])
        models[arm] = model.requires_grad_(False).eval()
    tick = time.monotonic()
    result = frozen_features(models, cache, config['position_chunk'], profiles)
    cnn_seconds = time.monotonic() - tick
    path = directory / 'features.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, game_index=cache.games.game_index.to_numpy(), **result)
    path.with_suffix('.tmp').replace(path)
    write_json(complete, {'identity': identity, 'status': 'complete', 'sha256': file_hash(path),
                         'seconds': time.monotonic() - started, 'cache_load_seconds': load_seconds,
                         'cnn_seconds': cnn_seconds, 'cached_q_error': error,
                         'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 2**30})
    print(json.dumps({'stage': 'features_complete', 'side': side, **read_json(complete)}), flush=True)


def load_side(out, side, wanted, identity):
    directory = Path(out) / 'features' / side
    marker = read_json(directory / 'complete.json')
    path = directory / 'features.npz'
    if marker['identity'] != identity or marker['status'] != 'complete' or marker['sha256'] != file_hash(path):
        raise ValueError('feature completion identity/checksum changed')
    profiles = 9 if side == 'H' else 10
    with np.load(path, allow_pickle=False) as data:
        return {name: align(data['game_index'], data[name], wanted, width)
                for name, width in (('q', profiles * 2), ('r', profiles * 2), ('z', 64))}


def dev_side(config, side, wanted):
    directory = ROOT / config['dev_runs'][side]
    profiles = 9 if side == 'H' else 10
    raw = read_features(directory / 'A0/dev_features.npz', wanted, 2 * profiles)
    residual = read_features(directory / 'CNN-R10/dev_features.npz', wanted, 2 * profiles)
    combined = read_features(directory / 'CNN-Z32/dev_features.npz', wanted, 2 * (profiles + 32))
    return split_features(raw, residual, combined, profiles)
