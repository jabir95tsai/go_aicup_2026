"""Paired A0/S/F x two learning rates, with complete question epochs."""
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

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from .data import Cache, allocate, extract_worker, question_rows
from .evaluate_train import training_questions, validate_sources
from .model import Head, Residual
from .run import finalize_cache
from .train import atomic_save, backward_questions, game_features, predict, standardize_statistics, state

CASES = (('A0_lr1e-3', 'A0', .001), ('A0_lr3e-4', 'A0', .0003),
         ('S_lr1e-3', 'S', .001), ('S_lr3e-4', 'S', .0003),
         ('F_lr1e-3', 'F', .001), ('F_lr3e-4', 'F', .0003))


def epoch_batches(questions, seed, epoch, per_k):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    buckets = [rng.permutation(np.flatnonzero(questions.k.to_numpy() == k)) for k in K_BUCKETS]
    if per_k < 1 or len({len(b) for b in buckets}) != 1 or not len(buckets[0]) or len(buckets[0]) % per_k:
        raise ValueError('question buckets must have equal sizes divisible by per_k')
    return [np.concatenate([b[start:start + per_k] for b in buckets])
            for start in range(0, len(buckets[0]), per_k)]


def gradient_snapshot(model, head):
    groups = {'head': list(head.parameters())}
    groups.update({name: list(child.parameters()) for name, child in model.named_children()})
    snapshots, report = {}, {}
    for name, parameters in groups.items():
        before = torch.cat([p.detach().flatten() for p in parameters]).clone()
        grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten()
                          for p in parameters])
        bad = int((~torch.isfinite(grad)).sum())
        if bad or not torch.isfinite(before).all():
            raise ValueError(f'nonfinite parameters or gradients in {name}')
        snapshots[name] = (parameters, before)
        report[name] = {'gradient_norm': float(grad.norm()), 'parameter_norm': float(before.norm()),
                        'nonfinite_gradients': bad,
                        'missing_gradient_parameters': sum(p.grad is None for p in parameters)}
    return snapshots, report


def update_statistics(snapshots, report):
    for name, (parameters, before) in snapshots.items():
        after = torch.cat([p.detach().flatten() for p in parameters])
        if not torch.isfinite(after).all():
            raise ValueError(f'nonfinite optimizer update in {name}')
        norm = float((after - before).norm())
        report[name].update(update_norm=norm, update_ratio=norm / max(report[name]['parameter_norm'], 1e-12))
    return report


@torch.no_grad()
def evaluate_role(model, head, cache, questions, role, chunk, train_count, baseline=False):
    ids = np.flatnonzero(cache.games.role.eq(role))
    c = cache.arrays['c'][ids].to(cache.device) if baseline else game_features(model, cache, ids, chunk)
    if not torch.isfinite(c).all():
        raise ValueError('nonfinite game features')
    q = questions[role]
    rows = question_rows(cache.games.iloc[ids].reset_index(drop=True), q)
    report, pred, scores, probs = predict(head, c, q, rows)
    penalty = float(head.linear.weight.square().sum() / (2 * train_count))
    report.update(head_l2=penalty, objective=report['ce'] + penalty)
    frame = q[['question_id', 'player_id', 'rank_id', 'k']].copy()
    frame['pred_rank'], frame['score'] = pred, scores
    for rank in range(10):
        frame[f'p{rank}'] = probs[:, rank]
    return report, frame, c.cpu().numpy(), ids


def run_arm(case, cache, questions, initial_head, config, identity, out):
    name, arm, lr = case
    directory = out / name
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        if json.loads((directory / 'complete.json').read_text())['identity'] != identity:
            raise ValueError('completed arm identity changed')
        return
    torch.manual_seed(config['seed'])
    model, head = Residual(arm).to(cache.device), Head().to(cache.device)
    head.load_state_dict(initial_head)
    train_games = np.flatnonzero(cache.games.role.eq('train'))
    rows = question_rows(cache.games, questions['train'])
    chunk, count = config['position_chunk'], len(questions['train'])
    if arm == 'S':
        standardize_statistics(model, cache, train_games, chunk)
    lp, target, q, _ = next(cache.chunks(train_games[:1], chunk))
    torch.testing.assert_close(model(lp, target, q), q, atol=0, rtol=0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.}], lr=lr)
    history = [{'epoch': 0, 'step': 0, **{
        role: evaluate_role(model, head, cache, questions, role, chunk, count, baseline=True)[0]
        for role in ('train', 'monitor')}}]
    epoch, step, best = 0, 0, history[0]['monitor']['equal_k']
    seen = np.zeros(len(cache.games), dtype=bool)
    last_path = directory / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location=cache.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('resume identity changed')
        model.load_state_dict(saved['model'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, step, best, history, seen = (saved[k] for k in ('epoch', 'step', 'best', 'history', 'seen'))
    else:
        atomic_save(state(model, head, identity, 0), directory / 'initial.pt')
        atomic_save(state(model, head, identity, 0), directory / 'model.pt')
        write_json(directory / 'learning_curve.json', history)
    started = time.monotonic()
    for epoch in range(epoch + 1, config['epochs'] + 1):
        batches = epoch_batches(questions['train'], config['seed'], epoch, config['questions_per_k'])
        symmetry_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, 4]))
        visited, losses = np.zeros(count, dtype=np.int64), []
        model.train(), head.train()
        # A restarted partial epoch replaces only its own incomplete gradient log.
        with (directory / f'gradients_epoch_{epoch}.jsonl').open('w') as log:
            for chosen in batches:
                optimizer.zero_grad(set_to_none=True)
                targets = torch.as_tensor(questions['train'].iloc[chosen].rank_id.to_numpy(), device=cache.device)
                loss, unique = backward_questions(model, head, cache, [rows[i] for i in chosen], targets,
                                                  count, chunk, int(symmetry_rng.integers(8)))
                snapshots, gradients = gradient_snapshot(model, head)
                optimizer.step()
                gradients = update_statistics(snapshots, gradients)
                step += 1
                seen[unique] = True
                visited[chosen] += 1
                losses.append(loss)
                log.write(json.dumps({'epoch': epoch, 'step': step, 'batch_objective': loss,
                                      'groups': gradients}, allow_nan=False) + '\n')
                if step % 25 == 0:
                    log.flush()
                    print(json.dumps({'case': name, 'stage': 'train', 'epoch': epoch, 'step': step,
                                      'batch_objective_mean': float(np.mean(losses)),
                                      'seconds': time.monotonic() - started}), flush=True)
        if not np.all(visited == 1):
            raise ValueError('epoch did not visit every training question exactly once')
        model.eval(), head.eval()
        entry = {'epoch': epoch, 'step': step, 'questions_seen_this_epoch': int(visited.sum()),
                 'training_games_seen': int(seen.sum()),
                 'training_players_seen': int(cache.games.loc[seen].player_id.nunique()),
                 'batch_objective_mean': float(np.mean(losses)), 'seconds': time.monotonic() - started}
        for role in ('train', 'monitor'):
            entry[role] = evaluate_role(model, head, cache, questions, role, chunk, count)[0]
        if entry['monitor']['equal_k'] > best:
            best = entry['monitor']['equal_k']
            atomic_save(state(model, head, identity, step), directory / 'model.pt')
        history.append(entry)
        write_json(directory / 'learning_curve.json', history)
        saved = state(model, head, identity, step)
        saved.update(optimizer=optimizer.state_dict(), epoch=epoch, best=best, history=history, seen=seen)
        atomic_save(saved, last_path)
        print(json.dumps({'case': name, 'stage': 'epoch_complete', **entry}), flush=True)
    selected = torch.load(directory / 'model.pt', map_location=cache.device, weights_only=False)
    model.load_state_dict(selected['model'])
    head.load_state_dict(selected['head'])
    model.eval(), head.eval()
    reports = {}
    for role in ('train', 'monitor', 'dev'):
        report, frame, features, ids = evaluate_role(model, head, cache, questions, role, chunk, count)
        reports[role] = report
        frame.to_parquet(directory / f'{role}_predictions.parquet', index=False)
        np.savez(directory / f'{role}_features.npz', game_index=cache.games.iloc[ids].game_index.to_numpy(), c=features)
    write_json(directory / 'result.json', {'identity': identity, 'case': name, 'arm': arm, 'lr': lr,
               'completed_updates': step, 'selected_step': selected['step'], 'roles': reports,
               'seconds_this_session': time.monotonic() - started,
               'training_games_seen': int(seen.sum()),
               'training_players_seen': int(cache.games.loc[seen].player_id.nunique()),
               'parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
               'peak_gpu_gib': (torch.cuda.max_memory_allocated(cache.device) / 2**30
                                if cache.device.type == 'cuda' else None)})
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})


def train_worker(rank, games, bounds, arrays, questions, initial, config, identity, out):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    cache = Cache(games, bounds, arrays, torch.device('cuda', rank))
    cases = CASES[:2] if rank == 0 else (CASES[rank + 1],)
    for case in cases:
        run_arm(case, cache, questions, initial, config, identity, out)


def summarize(out, identity, seed):
    from scripts.strength.rank_hier_head import difference
    results, frames = {}, {}
    for name, _, _ in CASES:
        marker = json.loads((out / name / 'complete.json').read_text())
        if marker['identity'] != identity or marker['status'] != 'complete':
            raise ValueError('incomplete or mismatched arm')
        results[name] = json.loads((out / name / 'result.json').read_text())
        frames[name] = pd.read_parquet(out / name / 'dev_predictions.parquet')
    ref = frames[CASES[0][0]]
    for frame in frames.values():
        if not frame[['question_id', 'player_id', 'rank_id', 'k']].equals(
                ref[['question_id', 'player_id', 'rank_id', 'k']]):
            raise ValueError('unpaired dev questions')
    pairs = [(f'{a}_{lr}', f'{b}_{lr}') for lr in ('lr1e-3', 'lr3e-4')
             for a, b in (('S', 'A0'), ('F', 'A0'), ('F', 'S'))]
    pairs += [(f'{a}_lr3e-4', f'{a}_lr1e-3') for a in ('A0', 'S', 'F')]
    differences = {}
    for a, b in pairs:
        diff = difference(frames[a].score.to_numpy(), frames[b].score.to_numpy(),
                          ref.k.to_numpy(), ref.player_id.to_numpy(), np.random.default_rng(seed + 99))
        diff['gate_pass'] = diff['equal_k'] >= .005 and diff['prob_positive'] >= .90 and min(diff['by_k'].values()) >= -.005
        differences[f'{a}-{b}'] = diff
    return {'identity': identity, 'arms': results, 'differences': differences,
            'limitations': ['one seed; historically reused dev',
                            'C-vector learning overlaps 190/200 monitoring players',
                            'batch size and replacement scheme both differ from the old pilot',
                            'monitor only selects among initial and complete-epoch checkpoints; no dev tuning']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_residual_epoch.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['epochs'] != 4 or config['questions_per_k'] != 8:
        raise ValueError('approved budget is four epochs of 32 questions per update')
    source = ROOT / config['source']
    source_identity = validate_sources(source)
    games = pd.read_parquet(source / 'games.parquet')
    questions = {role: pd.read_parquet(source / f'{role}_questions.parquet') for role in ('train', 'monitor', 'dev')}
    if games.groupby('player_id').role.nunique().gt(1).any():
        raise ValueError('player overlap between roles')
    for role, q in questions.items():
        part = games[games.role.eq(role)].copy()
        # Reuse the existing membership validator for all three isolated roles.
        part['role'] = 'train'
        training_questions(part, q)
    steps = len(epoch_batches(questions['train'], config['seed'], 1, config['questions_per_k']))
    if steps != 250 or len(questions['train']) != 8000:
        raise ValueError('source must retain the fixed 8000 training questions')
    out = args.out or ROOT / config['out'] / os.environ.get('SLURM_JOB_ID', 'prepare')
    out.mkdir(parents=True, exist_ok=True)
    record = {'config': config, 'source_identity': source_identity['identity'],
              'question_hashes': {role: file_hash(source / f'{role}_questions.parquet') for role in questions},
              'initial_head_sha256': file_hash(source / 'initial_head.pt'),
              'sources': {'epoch.py': file_hash(Path(__file__))}, 'cases': CASES, 'steps_per_epoch': steps}
    identity = object_hash(record)
    if (out / 'identity.json').exists() and json.loads((out / 'identity.json').read_text())['identity'] != identity:
        raise ValueError('existing run identity changed')
    write_json(out / 'identity.json', dict(record, identity=identity))
    write_json(out / 'config.json', config)
    for name in ('games.parquet', 'train_questions.parquet', 'monitor_questions.parquet', 'dev_questions.parquet'):
        if not (out / name).exists():
            shutil.copyfile(source / name, out / name)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'out': str(out), 'steps_per_epoch': steps, 'cases': CASES}))
        return
    if (out / 'complete.json').exists():
        print(json.dumps({'status': 'already_complete', 'out': str(out)}))
        return
    if torch.cuda.device_count() != 5:
        raise RuntimeError('five visible GPUs required')
    if shutil.disk_usage(out).free < 512 * 2**20:
        raise RuntimeError('insufficient free disk space')
    torch.set_num_threads(4)
    started = time.monotonic()
    arrays, bounds, ram_bytes = allocate(games)
    write_json(out / 'memory_plan.json', {'shared_cache_bytes': ram_bytes, 'persistent_policy_bytes': 0})
    with tempfile.TemporaryDirectory(prefix='policy-epoch-', dir='/dev/shm') as code_dir:
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = code_dir
        os.environ['TRITON_CACHE_DIR'] = str(Path(code_dir) / 'triton')
        mp.spawn(extract_worker, args=(5, games, bounds, arrays, config, out), nprocs=5)
        error = finalize_cache(games, bounds, arrays)
        raw = np.load(source / 'raw_c.npz')
        if not np.array_equal(raw['game_index'], games.game_index.to_numpy()):
            raise ValueError('source raw C game order mismatch')
        drift = float(np.abs(arrays['c'].numpy() - raw['c']).max())
        if not np.isfinite(drift) or drift > .002:
            raise ValueError(f'original C changed: {drift}')
        arrays['c'].copy_(torch.from_numpy(raw['c']))
        write_json(out / 'extraction_complete.json', {'positions': int(bounds[-1]), 'seconds': time.monotonic() - started,
                   'cached_C_mean_max_abs_error': error, 'original_C_mean_max_abs_error': drift})
        np.savez(out / 'raw_c.npz', game_index=games.game_index.to_numpy(), c=arrays['c'].numpy())
        initial = torch.load(source / 'initial_head.pt', map_location='cpu', weights_only=False)
        mp.spawn(train_worker, args=(games, bounds, arrays, questions, initial, config, identity, out), nprocs=5)
    result = summarize(out, identity, config['seed'])
    result['resources'] = {'wall_seconds': time.monotonic() - started,
                           'estimated_allocated_gpu_hours': 5 * (time.monotonic() - started) / 3600,
                           'allocated_gpus': 5, 'shared_cache_bytes': ram_bytes,
                           'persistent_bytes_before_final_report': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    write_json(out / 'result.json', result)
    write_json(out / 'complete.json', {'identity': identity, 'status': 'complete'})
    arrays.clear()
    gc.collect()
    print(json.dumps({'status': 'complete', 'out': str(out), 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
