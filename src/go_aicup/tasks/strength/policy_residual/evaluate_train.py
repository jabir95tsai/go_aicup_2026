"""Evaluate frozen best/latest checkpoints on the original fixed training questions."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

from go_aicup.artifacts import file_hash, write_json
from go_aicup.paths import ROOT
from .data import Cache, VECTORS, allocate, extract_worker, question_rows
from .model import Head, Residual
from .run import finalize_cache
from .train import game_features, predict

ARMS = ('A0', 'S', 'F')


@torch.no_grad()
def evaluate_checkpoint(path, arm, cache, questions, rows, identity, chunk):
    saved = torch.load(path, map_location=cache.device, weights_only=False)
    if saved['identity'] != identity:
        raise ValueError('checkpoint identity mismatch')
    model, head = Residual(arm).to(cache.device), Head().to(cache.device)
    model.load_state_dict(saved['model'])
    head.load_state_dict(saved['head'])
    model.requires_grad_(False).eval()
    head.requires_grad_(False).eval()
    features = game_features(model, cache, np.arange(len(cache.games)), chunk)
    report, predictions, scores, probabilities = predict(head, features, questions, rows)
    penalty = float(head.linear.weight.square().sum() / (2 * len(questions)))
    report.update(step=saved['step'], head_l2=penalty, objective=report['ce'] + penalty)
    frame = questions[['question_id', 'player_id', 'rank_id', 'k']].copy()
    frame['pred_rank'], frame['score'] = predictions, scores
    for r in range(10):
        frame[f'p{r}'] = probabilities[:, r]
    return report, features.cpu().numpy(), frame


def evaluate_worker(rank, source, out, games, bounds, arrays, questions, config, identity):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    arm = ARMS[rank]
    cache = Cache(games, bounds, arrays, torch.device('cuda', rank))
    rows = question_rows(games, questions)
    history = json.loads((source / arm / 'learning_curve.json').read_text())
    reports = {}
    for label, filename in [('selected', 'model.pt'), ('last', 'last.pt')]:
        report, features, predictions = evaluate_checkpoint(
            source / arm / filename, arm, cache, questions, rows, identity, config['position_chunk'])
        report['monitor'] = next(x['monitor'] for x in history if x['step'] == report['step'])
        report['checkpoint_sha256'] = file_hash(source / arm / filename)
        reports[label] = report
        np.savez(out / f'{arm}_{label}_features.npz', game_index=games.game_index.to_numpy(), c=features)
        predictions.to_parquet(out / f'{arm}_{label}_predictions.parquet', index=False)
    write_json(out / f'{arm}.json', reports)
    print(json.dumps({'arm': arm, 'training_diagnostic': reports}), flush=True)


def validate_sources(source):
    from go_aicup.features.humansl_torch import CKPT
    marker = json.loads((source / 'complete.json').read_text())
    identity = json.loads((source / 'identity.json').read_text())
    if marker.get('status') != 'complete' or marker['identity'] != identity['identity']:
        raise ValueError('source experiment is incomplete')
    paths = {'games_sha256': source / 'games.parquet', 'vectors_sha256': VECTORS,
             'humansl_sha256': CKPT,
             'board_manifest_sha256': ROOT / 'data/processed/boards/strength_v1/manifest.json'}
    for name, path in paths.items():
        if file_hash(path) != identity[name]:
            raise ValueError(f'source changed: {path}')
    for name, expected in identity['sources'].items():
        if name in ('model.py', 'run.py', 'train.py', 'data.py'):
            path = Path(__file__).with_name(name)
        elif name == 'cv.py':
            path = ROOT / 'src/go_aicup/data/cv.py'
        else:
            path = ROOT / 'src/go_aicup' / name
        if file_hash(path) != expected:
            raise ValueError(f'source changed: {path}')
    return identity


def training_questions(games, questions):
    if games.game_index.duplicated().any() or questions.question_id.duplicated().any():
        raise ValueError('duplicate game/question identity')
    rows = question_rows(games, questions)
    for item, indices in zip(questions.itertuples(), rows):
        selected = games.iloc[indices]
        if (len(indices) != item.k or len(np.unique(indices)) != item.k
                or not selected.role.eq('train').all()
                or not selected.player_id.eq(item.player_id).all()
                or not selected.rank_id.eq(item.rank_id).all()
                or selected.player_color.nunique() != 1):
            raise ValueError('training question membership changed')
    if set(questions.k) != {5, 10, 15, 20} or set(questions.rank_id) != set(range(10)):
        raise ValueError('incomplete training rank/k cells')
    cells = questions.groupby(['k', 'rank_id']).size()
    if len(cells) != 40 or cells.nunique() != 1:
        raise ValueError('unbalanced training questions')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    identity = validate_sources(source)
    config = json.loads((source / 'config.json').read_text())
    games = pd.read_parquet(source / 'games.parquet')
    games = games[games.role.eq('train')].reset_index(drop=True)
    questions = pd.read_parquet(source / 'train_questions.parquet')
    training_questions(games, questions)
    raw = np.load(source / 'raw_c.npz')
    raw_rows = pd.Index(raw['game_index']).get_indexer(games.game_index)
    if (raw_rows < 0).any():
        raise ValueError('training games missing from original raw C')
    original_c = raw['c'][raw_rows].copy()
    workers = torch.cuda.device_count()
    if workers != 5:
        raise RuntimeError('diagnostic allocation requires five visible GPUs')
    out = source / 'train_diagnostic' / os.environ['SLURM_JOB_ID']
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(out / 'identity.json', {
        'source_identity': identity['identity'], 'source': str(source),
        'train_questions_sha256': file_hash(source / 'train_questions.parquet'),
        'evaluator_sha256': file_hash(Path(__file__)), 'training_updates': 0,
        'games': len(games), 'players': int(games.player_id.nunique()), 'questions': len(questions)})
    arrays, bounds, ram_bytes = allocate(games)
    with tempfile.TemporaryDirectory(prefix='policy-train-diagnostic-', dir='/dev/shm') as code_dir:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = code_dir
        os.environ['TRITON_CACHE_DIR'] = str(Path(code_dir) / 'triton')
        mp.spawn(extract_worker, args=(workers, games, bounds, arrays, config, out), nprocs=workers)
        cache_error = finalize_cache(games, bounds, arrays)
        baseline_error = float(np.abs(arrays['c'].numpy() - original_c).max())
        if not np.isfinite(baseline_error) or baseline_error > .002:
            raise ValueError(f're-extracted raw C changed: {baseline_error}')
        # Keep the original baseline; re-extraction supplies only frozen residual corrections.
        arrays['c'].copy_(torch.from_numpy(original_c))
        write_json(out / 'extraction_complete.json', {
            'positions': int(bounds[-1]), 'seconds': time.monotonic() - started,
            'cached_C_mean_max_abs_error': cache_error, 'original_C_mean_max_abs_error': baseline_error})
        mp.spawn(evaluate_worker, args=(source, out, games, bounds, arrays, questions,
                                       config, identity['identity']), nprocs=len(ARMS))
    result = {'arms': {arm: json.loads((out / f'{arm}.json').read_text()) for arm in ARMS},
              'source_identity': identity['identity'], 'training_updates': 0,
              'limitations': ['training metrics are in-sample',
                              'monitor metrics reused from the original fixed monitoring curves',
                              'all fixed training questions, including questions not sampled before checkpoint'],
              'resources': {'wall_seconds': time.monotonic() - started, 'allocated_gpus': workers,
                            'estimated_allocated_gpu_hours': workers * (time.monotonic() - started) / 3600,
                            'shared_cache_bytes': ram_bytes}}
    write_json(out / 'result.json', result)
    write_json(out / 'complete.json', {'status': 'complete', 'source_identity': identity['identity']})
    print(json.dumps({'status': 'complete', 'out': str(out), **result}), flush=True)


if __name__ == '__main__':
    main()
