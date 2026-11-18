"""Single-GPU five-arm experiment from root plan.md: A0, R10, Z32, CNN-R10, CNN-Z32.

Reads the TRAIN/monitor per-move policy cache straight from its ZIP into RAM (policy_zip),
fits one shared TRAIN-only scaler/sklearn head, then trains the arms one after another on
the same GPU with identical question order and D4 schedule.  dev is not touched here.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from . import policy_zip
from .branches import BranchHead, Encoder as SetEncoder, save_predictions
from .data import Cache, question_rows
from .epoch import epoch_batches, gradient_snapshot, update_statistics
from .evaluate_train import training_questions
from .model import Head as OriginalHead, policy_image, probabilities, question_mean
from .run import finalize_cache
from .train import atomic_save, fit_head, predict, state

CASES = ('A0', 'R10', 'Z32', 'CNN-R10', 'CNN-Z32')
RESIDUAL = {'R10', 'CNN-R10'}


class CNNEncoder(nn.Module):
    """Two valid 5x5 convolutions over [P10, logP10, played-move marker x10] board images."""

    def __init__(self, arm):
        super().__init__()
        if arm not in ('CNN-R10', 'CNN-Z32'):
            raise ValueError(arm)
        self.arm = arm
        self.width = 10 if arm in RESIDUAL else 42
        self.conv = nn.Sequential(nn.Conv2d(30, 32, 5), nn.ReLU(), nn.Conv2d(32, 32, 5), nn.ReLU())
        self.hidden = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.out = nn.Linear(64, self.width if arm in RESIDUAL else self.width - 10)
        if arm in RESIDUAL:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, logp, target, q, symmetry=0):
        # Illegal points are P=0 and logP=0 (finite) after probabilities(); pass is not on the board,
        # so a played pass leaves the marker planes empty and is carried by is_pass.
        p, lp, legal = probabilities(logp)
        image = policy_image(p, lp, target, symmetry).contiguous(memory_format=torch.channels_last)
        c = self.conv(image).mean((-2, -1))
        x = torch.cat((c, q, q.exp(), lp[..., -1], target.eq(361).float()[:, None],
                       legal.sum(-1).float().log()[:, None]), 1)
        z = self.out(self.hidden(x)).float()
        return q + z if self.arm in RESIDUAL else torch.cat((q, z), 1)


def build(arm, seed):
    """Seed before construction: both set arms share the set body, both CNN arms the CNN/hidden body."""
    torch.manual_seed(seed)
    return CNNEncoder(arm) if arm.startswith('CNN') else SetEncoder(arm)


@torch.no_grad()
def game_features(model, cache, rows, chunk, symmetry=0):
    raw = cache.arrays['c'][rows].to(cache.device)
    if model.arm == 'A0':
        return raw
    residual = model.arm in RESIDUAL
    width = 10 if residual else model.width - 10
    total = torch.zeros(len(rows) * 2, width, device=cache.device)
    for lp, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            evidence = model(lp, target, q, symmetry)
        total.index_add_(0, slot, (evidence - q if residual else evidence[:, 10:]).float())
    added = total / cache.counts(rows)[:, None]
    if residual:
        return raw + added.reshape(-1, 20)
    return torch.cat((raw.reshape(-1, 10), added), 1).reshape(len(rows), -1)


def backward_questions(model, head, cache, rows, targets, train_count, chunk, symmetry=0):
    """Exact chain rule through side/game/question means with bounded activation memory."""
    unique, inverse = np.unique(np.concatenate(rows), return_inverse=True)
    ends = np.r_[0, np.cumsum([len(r) for r in rows])]
    local = [inverse[a:b] for a, b in zip(ends[:-1], ends[1:])]
    c = game_features(model, cache, unique, chunk, symmetry).detach()
    c.requires_grad_(model.arm != 'A0')
    loss = F.cross_entropy(head(question_mean(c, local)), targets)
    loss = loss + head.linear.weight.square().sum() / (2 * train_count)
    if not torch.isfinite(loss):
        raise ValueError('nonfinite question loss')
    loss.backward()
    if model.arm != 'A0':
        grad = c.grad.reshape(-1, model.width) / cache.counts(unique)[:, None]
        for lp, target, q, slot in cache.chunks(unique, chunk):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
                evidence = model(lp, target, q, symmetry)
            evidence.backward(grad[slot])
    return float(loss.detach()), unique


@torch.no_grad()
def branch_rms(model, head, c, raw, rows):
    x = question_mean(c, rows)
    raw_x = question_mean(raw, rows)
    q_only = torch.zeros_like(x)
    for side in (0, 1):
        q_only[:, side * model.width:side * model.width + 10] = raw_x[:, side * 10:side * 10 + 10]
    q_logits = head(q_only)
    added = head(x) - q_logits
    if not torch.isfinite(q_logits).all() or not torch.isfinite(added).all():
        raise ValueError('nonfinite branch logits')
    return {'q_logits_rms': float(q_logits.square().mean().sqrt()),
            'added_logits_rms': float(added.square().mean().sqrt())}


@torch.no_grad()
def evaluate_role(model, head, cache, questions, role, chunk, train_count):
    ids = np.flatnonzero(cache.games.role.eq(role))
    c = game_features(model, cache, ids, chunk)
    if not torch.isfinite(c).all():
        raise ValueError('nonfinite game features')
    q = questions[role]
    rows = question_rows(cache.games.iloc[ids].reset_index(drop=True), q)
    report, pred, scores, probs = predict(head, c, q, rows)
    penalty = float(head.linear.weight.square().sum() / (2 * train_count))
    report.update(head_l2=penalty, objective=report['ce'] + penalty)
    report.update(branch_rms(model, head, c, cache.arrays['c'][ids].to(cache.device), rows))
    frame = q[['question_id', 'player_id', 'rank_id', 'k']].copy()
    frame['pred_rank'], frame['score'] = pred, scores
    for rank in range(10):
        frame[f'p{rank}'] = probs[:, rank]
    return report, frame, c.cpu().numpy(), ids


def _evaluate_both(model, head, cache, questions, chunk, count, history, directory, step):
    entry = {}
    for role in ('train', 'monitor'):
        report, frame, _, _ = evaluate_role(model, head, cache, questions, role, chunk, count)
        if history:
            base = history[0][role]
            report['delta_from_initial'] = {key: report[key] - base[key]
                                            for key in ('ce', 'head_l2', 'objective', 'equal_k')}
            report['by_k_delta_from_initial'] = {k: report['by_k'][k] - base['by_k'][k] for k in report['by_k']}
        entry[role] = report
        save_predictions(directory, role, step, frame)
    return entry


def run_arm(arm, cache, questions, initial_head, config, identity, out):
    directory = out / arm
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        if json.loads((directory / 'complete.json').read_text())['identity'] != identity:
            raise ValueError('completed arm identity changed')
        return json.loads((directory / 'result.json').read_text())
    if cache.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(cache.device)
    model = build(arm, config['seed']).to(cache.device)
    head = BranchHead(model.width, initial_head).to(cache.device)
    rows = question_rows(cache.games, questions['train'])
    chunk, count = config['position_chunk'], len(questions['train'])
    train_games = np.flatnonzero(cache.games.role.eq('train'))
    lp, target, q, _ = next(cache.chunks(train_games[:1], chunk))
    if arm == 'A0' or arm in RESIDUAL:
        torch.testing.assert_close(model(lp, target, q).float(), q, atol=0, rtol=0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.}], lr=config['lr'])
    reference_head = OriginalHead().to(cache.device)
    reference_head.load_state_dict(initial_head)
    timing = {'train_seconds': 0., 'evaluation_seconds': 0., 'evaluations': 0}
    model.eval(), head.eval()
    tick = time.monotonic()
    history = [{'epoch': 0, 'step': 0, **_evaluate_both(model, head, cache, questions, chunk, count, [],
                                                         directory, 0)}]
    timing['evaluation_seconds'] += time.monotonic() - tick
    timing['evaluations'] += 1
    for role in ('train', 'monitor'):
        ids = np.flatnonzero(cache.games.role.eq(role))
        local = question_rows(cache.games.iloc[ids].reset_index(drop=True), questions[role])
        reference = predict(reference_head, cache.arrays['c'][ids].to(cache.device), questions[role], local)
        got = np.load(directory / f'{role}_step_0000.npz')['probabilities']
        np.testing.assert_allclose(got, reference[3], atol=2e-6, rtol=2e-5)
        if (abs(history[0][role]['ce'] - reference[0]['ce']) > 2e-6
                or history[0][role]['equal_k'] != reference[0]['equal_k']):
            raise ValueError('step-zero baseline mismatch')
        history[0][role]['delta_from_initial'] = {key: 0. for key in ('ce', 'head_l2', 'objective', 'equal_k')}
        history[0][role]['by_k_delta_from_initial'] = {str(k): 0. for k in K_BUCKETS}
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
        epoch, step, best, history, seen, timing = (saved[k] for k in
                                                     ('epoch', 'step', 'best', 'history', 'seen', 'timing'))
        torch.set_rng_state(saved['torch_rng'].cpu())
        if cache.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), cache.device)
        # A restarted partial epoch replays from the epoch boundary; restore that boundary's best.
        atomic_save(saved['best_state'], directory / 'model.pt')
    else:
        atomic_save(state(model, head, identity, 0), directory / 'initial.pt')
        atomic_save(state(model, head, identity, 0), directory / 'model.pt')
        write_json(directory / 'learning_curve.json', history)
    limit = config.get('max_updates')
    for epoch in range(epoch + 1, config['epochs'] + 1):
        batches = epoch_batches(questions['train'], config['seed'], epoch, config['questions_per_k'])
        symmetry_rng = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, 4]))
        visited, losses = np.zeros(count, dtype=np.int64), []
        model.train(), head.train()
        truncated = False
        with (directory / f'gradients_epoch_{epoch}.jsonl').open('w') as log:
            for chosen in batches:
                symmetry = int(symmetry_rng.integers(8))
                if limit is not None and step >= limit:
                    truncated = True
                    break
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                targets = torch.as_tensor(questions['train'].iloc[chosen].rank_id.to_numpy(), device=cache.device)
                loss, unique = backward_questions(model, head, cache, [rows[i] for i in chosen], targets,
                                                  count, chunk, symmetry)
                snapshots, gradients = gradient_snapshot(model, head)
                optimizer.step()
                gradients = update_statistics(snapshots, gradients)
                if cache.device.type == 'cuda':
                    torch.cuda.synchronize(cache.device)
                timing['train_seconds'] += time.monotonic() - tick
                step += 1
                seen[unique] = True
                visited[chosen] += 1
                losses.append(loss)
                log.write(json.dumps({'epoch': epoch, 'step': step, 'symmetry': symmetry,
                                      'batch_objective': loss, 'groups': gradients}, allow_nan=False) + '\n')
                if step % config['evaluate_every'] == 0:
                    model.eval(), head.eval()
                    tick = time.monotonic()
                    entry = {'epoch': epoch, 'step': step,
                             **_evaluate_both(model, head, cache, questions, chunk, count, history, directory, step)}
                    timing['evaluation_seconds'] += time.monotonic() - tick
                    timing['evaluations'] += 1
                    # Strict improvement only; ties keep the earlier step, and step 0 can win.
                    if entry['monitor']['equal_k'] > best:
                        best = entry['monitor']['equal_k']
                        atomic_save(state(model, head, identity, step), directory / 'model.pt')
                    entry['timing'] = dict(timing)
                    history.append(entry)
                    write_json(directory / 'learning_curve.json', history)
                    print(json.dumps({'case': arm, 'stage': 'evaluation', 'step': step,
                                      'monitor_equal_k': entry['monitor']['equal_k'],
                                      'monitor_by_k': entry['monitor']['by_k'],
                                      'train_equal_k': entry['train']['equal_k'], **timing}), flush=True)
                    model.train(), head.train()
                if step % 25 == 0:
                    log.flush()
                    print(json.dumps({'case': arm, 'stage': 'train', 'epoch': epoch, 'step': step,
                                      'batch_objective_mean': float(np.mean(losses)), **timing}), flush=True)
        if truncated:
            break
        if not np.all(visited == 1):
            raise ValueError('epoch did not visit every training question exactly once')
        history[-1].update(questions_seen_this_epoch=int(visited.sum()), training_games_seen=int(seen.sum()),
                           training_players_seen=int(cache.games.loc[seen].player_id.nunique()))
        write_json(directory / 'learning_curve.json', history)
        saved = state(model, head, identity, step)
        saved.update(optimizer=optimizer.state_dict(), epoch=epoch, best=best, history=history, seen=seen,
                     timing=timing, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(cache.device) if cache.device.type == 'cuda' else None,
                     best_state=torch.load(directory / 'model.pt', map_location=cache.device, weights_only=False))
        atomic_save(saved, last_path)
        print(json.dumps({'case': arm, 'stage': 'epoch_complete', 'epoch': epoch, 'step': step}), flush=True)
    selected = torch.load(directory / 'model.pt', map_location=cache.device, weights_only=False)
    model.load_state_dict(selected['model'])
    head.load_state_dict(selected['head'])
    model.eval(), head.eval()
    reports = {}
    for role in ('train', 'monitor'):
        report, frame, features, ids = evaluate_role(model, head, cache, questions, role, chunk, count)
        reports[role] = report
        frame.to_parquet(directory / f'{role}_predictions.parquet', index=False)
        np.savez(directory / f'{role}_features.npz', game_index=cache.games.iloc[ids].game_index.to_numpy(),
                 c=features)
    encoder = sum(p.numel() for p in model.parameters())
    head_n = sum(p.numel() for p in head.parameters())
    result = {'identity': identity, 'arm': arm, 'completed_updates': step, 'selected_step': selected['step'],
              'smoke': limit is not None, 'roles': reports, 'timing': timing,
              'seconds_per_update': timing['train_seconds'] / max(step, 1),
              'seconds_per_evaluation': timing['evaluation_seconds'] / max(timing['evaluations'], 1),
              'training_games_seen': int(seen.sum()),
              'training_players_seen': int(cache.games.loc[seen].player_id.nunique()),
              'encoder_parameters': encoder, 'head_parameters': head_n, 'parameters': encoder + head_n,
              'peak_gpu_gib': (torch.cuda.max_memory_allocated(cache.device) / 2**30
                               if cache.device.type == 'cuda' else None)}
    write_json(directory / 'result.json', result)
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})
    return result


def load_cache(device, threads, path=policy_zip.ARCHIVE, on_device=False):
    """on_device keeps every per-move array in GPU memory (about 39 GiB), so chunk gathers never cross PCIe."""
    loaded = policy_zip.load(path, roles=('train', 'monitor'), threads=threads)
    games = loaded.games
    bounds = np.r_[games.start.to_numpy(), int(games.start.iloc[-1] + games.length.iloc[-1])].astype(np.int64)
    if not np.array_equal(np.diff(bounds), games.length.to_numpy()):
        raise ValueError('cache games are not contiguous')
    arrays = {name: torch.from_numpy(value) for name, value in loaded.arrays.items()}
    arrays['done'] = torch.ones(len(games), dtype=torch.bool)
    error = finalize_cache(games, bounds, arrays)
    if on_device:
        del loaded
        for name in list(arrays):
            arrays[name] = arrays[name].to(device)
    return Cache(games.reset_index(drop=True), bounds, arrays, device), error


def selection_tables(path=policy_zip.ARCHIVE):
    questions = {role: policy_zip.selection(f'{role}_questions', path) for role in ('train', 'monitor')}
    hashes = {}
    reader = policy_zip._Reader(Path(path))
    for name in ('games.parquet', 'train_questions.parquet', 'monitor_questions.parquet', 'identity.json',
                 'policy/complete.json'):
        hashes[name] = policy_zip._sha(reader.read(name))
    return questions, hashes


def validate_questions(games, questions):
    if games.groupby('player_id').role.nunique().gt(1).any():
        raise ValueError('player overlap between roles')
    for role, q in questions.items():
        part = games[games.role.eq(role)].copy()
        part['role'] = 'train'
        training_questions(part, q)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_five_arms.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--arms', nargs='+', default=list(CASES))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(args.arms) - set(CASES):
        raise ValueError(f'unknown arms: {args.arms}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    questions, hashes = selection_tables()
    sources = {name: file_hash(Path(__file__).with_name(name)) for name in
               ('five_arms.py', 'policy_zip.py', 'branches.py', 'epoch.py', 'model.py', 'train.py', 'data.py',
                'run.py', 'evaluate_train.py')}
    record = {'config': config, 'cache_identity': policy_zip.IDENTITY, 'selection_hashes': hashes,
              'plan_sha256': file_hash(ROOT / 'plan.md'), 'sources': sources, 'cases': list(CASES)}
    # Code stays in src/; the run records hashes and the git commit instead of copying source files.
    import subprocess
    git = {'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                    text=True).stdout.strip(),
           'dirty_files': subprocess.run(['git', 'status', '--porcelain', '--', 'src', 'configs'], cwd=ROOT,
                                         capture_output=True, text=True).stdout.splitlines()}
    identity = object_hash(record)
    out = args.out or ROOT / config['out'] / os.environ.get('SLURM_JOB_ID', 'local')
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists() and json.loads((out / 'identity.json').read_text())['identity'] != identity:
        raise ValueError('existing run identity changed')
    write_json(out / 'identity.json', dict(record, identity=identity, git=git))
    write_json(out / 'config.json', config)
    started = time.monotonic()
    cache, c_error = load_cache(device, config['zip_threads'], on_device=config.get('cache_on_gpu', False))
    validate_questions(cache.games, questions)
    steps = len(epoch_batches(questions['train'], config['seed'], 1, config['questions_per_k']))
    load_seconds = time.monotonic() - started
    initial_path = out / 'initial_head.pt'
    if initial_path.exists():
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
    else:
        rows = question_rows(cache.games, questions['train'])
        initial = fit_head(cache.arrays['c'].cpu().numpy(), questions['train'], rows)
        atomic_save(initial, initial_path)
    import importlib.metadata
    write_json(out / 'environment.json', {
        'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'host': platform.node(),
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'packages': {n: importlib.metadata.version(n) for n in ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'disk_free_bytes': shutil.disk_usage(out).free,
        'cache_load_seconds': load_seconds, 'positions': int(cache.bounds[-1]), 'games': len(cache.games),
        'steps_per_epoch': steps, 'cached_C_mean_max_abs_error': c_error})
    print(json.dumps({'stage': 'cache_loaded', 'seconds': round(load_seconds, 1), 'positions': int(cache.bounds[-1]),
                      'steps_per_epoch': steps}), flush=True)
    results = {}
    for arm in args.arms:
        results[arm] = run_arm(arm, cache, questions, initial, config, identity, out)
    summary = {'identity': identity, 'wall_seconds': time.monotonic() - started,
               'arms': {arm: {k: r[k] for k in ('completed_updates', 'selected_step', 'seconds_per_update',
                                                 'seconds_per_evaluation', 'parameters', 'peak_gpu_gib')}
                        | {'monitor': r['roles']['monitor']['by_k'] | {'equal_k': r['roles']['monitor']['equal_k']}}
                        for arm, r in results.items()},
               'persistent_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'status': 'complete', **summary}), flush=True)


if __name__ == '__main__':
    main()
