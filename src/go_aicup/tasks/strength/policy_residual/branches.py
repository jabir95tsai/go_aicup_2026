"""Fixed-budget A0/R10/Z10/Z32 experiment from root plan.md."""
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
from .model import Head as OriginalHead, Residual as OriginalResidual, probabilities, candidate_inputs, question_mean
from torch import nn
from torch.nn import functional as F
from .epoch import epoch_batches, gradient_snapshot, update_statistics
from .run import finalize_cache
from .train import atomic_save, predict, state

CASES = tuple((a, a, .0003) for a in ('A0', 'R10', 'Z10', 'Z32'))


class Encoder(OriginalResidual):
    def __init__(self, arm):
        super().__init__('A0' if arm == 'A0' else 'F')
        self.arm = arm
        self.width = {'A0': 10, 'R10': 10, 'Z10': 20, 'Z32': 42}[arm]
        if arm.startswith('Z'):
            self.delta[-1] = nn.Linear(64, self.width - 10)

    def forward(self, logp, target, q, symmetry=0):
        if self.arm == 'A0':
            return q
        # Set attention is invariant to synchronous point/D4 permutations, including the mask.
        p, lp, legal = probabilities(logp)
        v = self.phi(candidate_inputs(p, lp, legal, target, q=q))
        chosen = v[torch.arange(len(v), device=v.device), target]
        weights = self.scorer(torch.cat((v, chosen[:, None].expand(-1, 362, -1),
                                        q[:, None].expand(-1, 362, -1)), -1)).squeeze(-1)
        weights = weights.masked_fill(~legal, -torch.inf).softmax(-1)
        summary = (weights[..., None] * v).sum(1)
        x = torch.cat((q, chosen, summary, legal.sum(-1).float().log()[:, None]), 1)
        z = self.delta(x).float()
        return q + z if self.arm == 'R10' else torch.cat((q, z), 1)


class BranchHead(nn.Module):
    def __init__(self, width, initial):
        super().__init__()
        self.width = width
        self.register_buffer('mean', torch.zeros(2 * width))
        self.register_buffer('scale', torch.ones(2 * width))
        self.linear = nn.Linear(2 * width, 10)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.copy_(initial['linear.bias'])
            for side in (0, 1):
                old, new = slice(side * 10, side * 10 + 10), slice(side * width, side * width + 10)
                self.mean[new] = initial['mean'][old]
                self.scale[new] = initial['scale'][old]
                self.linear.weight[:, new] = initial['linear.weight'][:, old]

    def forward(self, x):
        return self.linear((x - self.mean) / self.scale)


@torch.no_grad()
def game_features(model, cache, rows, chunk, symmetry=0):
    raw = cache.arrays['c'][rows].to(cache.device)
    if model.arm == 'A0':
        return raw
    width = 10 if model.arm == 'R10' else model.width - 10
    total = torch.zeros(len(rows) * 2, width, device=cache.device)
    for lp, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            evidence = model(lp, target, q, symmetry)
        added = evidence - q if model.arm == 'R10' else evidence[:, 10:]
        total.index_add_(0, slot, added.float())
    added = total / cache.counts(rows)[:, None]
    if model.arm == 'R10':
        return raw + added.reshape(-1, 20)
    return torch.cat((raw.reshape(-1, 10), added), 1).reshape(len(rows), -1)


def backward_questions(model, head, cache, rows, targets, train_count, chunk, symmetry=0):
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
    added_logits = head(x) - q_logits
    if not torch.isfinite(q_logits).all() or not torch.isfinite(added_logits).all():
        raise ValueError('nonfinite branch logits')
    return {'q_logits_rms': float(q_logits.square().mean().sqrt()),
            'added_logits_rms': float(added_logits.square().mean().sqrt())}


def save_predictions(directory, role, step, frame):
    path = directory / f'{role}_step_{step:04d}.npz'
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, probabilities=frame[[f'p{r}' for r in range(10)]].to_numpy(),
                            score=frame.score.to_numpy(), pred_rank=frame.pred_rank.to_numpy())
    path.with_suffix('.tmp').replace(path)


def paired_difference(a, b, k, player, rng):
    delta = a - b
    ids, inverse = np.unique(player, return_inverse=True)
    sums = np.zeros((len(ids), 4))
    counts = np.zeros_like(sums)
    for j, bucket in enumerate(K_BUCKETS):
        rows = k == bucket
        np.add.at(sums[:, j], inverse[rows], delta[rows])
        np.add.at(counts[:, j], inverse[rows], 1)
    draws = rng.integers(0, len(ids), (2000, len(ids)))
    denominator = counts[draws].sum(1)
    if (denominator == 0).any():
        raise ValueError('bootstrap draw has an empty k bucket')
    by_k = sums[draws].sum(1) / denominator
    values = by_k.mean(1)
    return {'bootstrap_95': np.quantile(values, [.025, .975]).tolist(),
            'by_k_bootstrap_95': {str(k): np.quantile(by_k[:, j], [.025, .975]).tolist()
                                  for j, k in enumerate(K_BUCKETS)},
            'bootstrap_draws': 2000, 'prob_positive': float((values > 0).mean())}


def gate(diff):
    return diff['equal_k'] >= .005 and diff['prob_positive'] >= .90 and min(diff['by_k'].values()) >= -.005


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


def run_arm(case, cache, questions, initial_head, config, identity, out):
    name, arm, lr = case
    directory = out / name
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        if json.loads((directory / 'complete.json').read_text())['identity'] != identity:
            raise ValueError('completed arm identity changed')
        return
    torch.manual_seed(config['seed'])
    model = Encoder(arm).to(cache.device)
    head = BranchHead(model.width, initial_head).to(cache.device)
    train_games = np.flatnonzero(cache.games.role.eq('train'))
    rows = question_rows(cache.games, questions['train'])
    chunk, count = config['position_chunk'], len(questions['train'])
    lp, target, q, _ = next(cache.chunks(train_games[:1], chunk))
    if arm in ('A0', 'R10'):
        torch.testing.assert_close(model(lp, target, q), q, atol=0, rtol=0)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.}], lr=lr)
    reference_head = OriginalHead().to(cache.device)
    reference_head.load_state_dict(initial_head)
    history = [{'epoch': 0, 'step': 0}]
    for role in ('train', 'monitor'):
        ids = np.flatnonzero(cache.games.role.eq(role))
        local_rows = question_rows(cache.games.iloc[ids].reset_index(drop=True), questions[role])
        reference = predict(reference_head, cache.arrays['c'][ids].to(cache.device), questions[role], local_rows)
        report, frame, _, _ = evaluate_role(model, head, cache, questions, role, chunk, count)
        np.testing.assert_allclose(frame[[f'p{r}' for r in range(10)]], reference[3], atol=2e-6, rtol=2e-5)
        if abs(report['ce'] - reference[0]['ce']) > 2e-6 or report['equal_k'] != reference[0]['equal_k']:
            raise ValueError('step-zero baseline mismatch')
        report['delta_from_initial'] = {key: 0. for key in ('ce', 'head_l2', 'objective', 'equal_k')}
        report['by_k_delta_from_initial'] = {str(k): 0. for k in K_BUCKETS}
        history[0][role] = report
        save_predictions(directory, role, 0, frame)
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
        torch.set_rng_state(saved['torch_rng'].cpu())
        if cache.device.type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), cache.device)
        atomic_save(saved['best_state'], directory / 'model.pt')
    elif (directory / 'initial.pt').exists():
        saved = torch.load(directory / 'initial.pt', map_location=cache.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('initial identity changed')
        model.load_state_dict(saved['model'])
        head.load_state_dict(saved['head'])
        atomic_save(saved, directory / 'model.pt')
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
                if step % config['evaluate_every'] == 0:
                    model.eval(), head.eval()
                    entry = {'epoch': epoch, 'step': step, 'seconds': time.monotonic() - started}
                    for role in ('train', 'monitor'):
                        report, frame, _, _ = evaluate_role(model, head, cache, questions, role, chunk, count)
                        report['delta_from_initial'] = {
                            key: report[key] - history[0][role][key]
                            for key in ('ce', 'head_l2', 'objective', 'equal_k')}
                        report['by_k_delta_from_initial'] = {
                            k: report['by_k'][k] - history[0][role]['by_k'][k] for k in report['by_k']}
                        entry[role] = report
                        save_predictions(directory, role, step, frame)
                    if entry['monitor']['equal_k'] > best:
                        best = entry['monitor']['equal_k']
                        atomic_save(state(model, head, identity, step), directory / 'model.pt')
                    history.append(entry)
                    write_json(directory / 'learning_curve.json', history)
                    print(json.dumps({'case': name, 'stage': 'evaluation', **entry}), flush=True)
                    model.train(), head.train()
                if step % 25 == 0:
                    log.flush()
                    print(json.dumps({'case': name, 'stage': 'train', 'epoch': epoch, 'step': step,
                                      'batch_objective_mean': float(np.mean(losses)),
                                      'seconds': time.monotonic() - started}), flush=True)
        if not np.all(visited == 1):
            raise ValueError('epoch did not visit every training question exactly once')
        history[-1].update(questions_seen_this_epoch=int(visited.sum()),
                           training_games_seen=int(seen.sum()),
                           training_players_seen=int(cache.games.loc[seen].player_id.nunique()))
        write_json(directory / 'learning_curve.json', history)
        saved = state(model, head, identity, step)
        saved.update(optimizer=optimizer.state_dict(), epoch=epoch, best=best, history=history, seen=seen,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(cache.device) if cache.device.type == 'cuda' else None,
                     best_state=torch.load(directory / 'model.pt', map_location=cache.device, weights_only=False))
        atomic_save(saved, last_path)
        print(json.dumps({'case': name, 'stage': 'epoch_complete', **history[-1]}), flush=True)
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
               'encoder_parameters': sum(p.numel() for p in model.parameters()),
               'head_parameters': sum(p.numel() for p in head.parameters()),
               'parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
               'peak_gpu_gib': (torch.cuda.max_memory_allocated(cache.device) / 2**30
                                if cache.device.type == 'cuda' else None)})
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})


def train_worker(rank, games, bounds, arrays, questions, initial, config, identity, out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(rank)
    cache = Cache(games, bounds, arrays, torch.device('cuda', rank))
    run_arm(CASES[rank], cache, questions, initial, config, identity, out)


def summarize(out, identity, seed):
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
    pairs = [('Z10', 'R10'), ('Z32', 'Z10'), ('Z32', 'A0'), ('Z32', 'R10'), ('R10', 'A0'), ('Z10', 'A0')]
    differences = {}
    for a, b in pairs:
        diff = paired_difference(frames[a].score.to_numpy(), frames[b].score.to_numpy(),
                          ref.k.to_numpy(), ref.player_id.to_numpy(), np.random.default_rng(seed + 99))
        delta = frames[a].score.to_numpy() - frames[b].score.to_numpy()
        diff['by_k'] = {str(k): float(delta[ref.k.to_numpy() == k].mean()) for k in K_BUCKETS}
        diff['equal_k'] = float(np.mean(list(diff['by_k'].values())))
        diff['gate_pass'] = gate(diff)
        differences[f'{a}-{b}'] = diff
    return {'identity': identity, 'arms': results, 'differences': differences,
            'Z32_candidate_gate': differences['Z32-A0']['gate_pass'] and differences['Z32-R10']['gate_pass'],
            'Z32_dimension_gate': differences['Z32-Z10']['gate_pass'],
            'limitations': ['one seed; historically reused dev',
                            'C-vector learning overlaps 190/200 monitoring players',
                            'architecture/initialization and parameter counts differ',
                            'monitor selection every 125 updates; dev never tunes training']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_branches.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config['epochs'] != 12 or config['questions_per_k'] != 8 or config['evaluate_every'] != 125:
        raise ValueError('approved budget is twelve epochs of 32 questions, evaluation every 125 updates')
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
              'raw_c_sha256': file_hash(source / 'raw_c.npz'),
              'plan_sha256': file_hash(ROOT / 'plan.md'),
              'requirements_sha256': file_hash(ROOT / 'requirements.txt'),
              'sources': {name: file_hash(Path(__file__).with_name(name)) for name in
                          ('branches.py', 'epoch.py', 'model.py', 'train.py', 'data.py', 'evaluate_train.py')}, 'cases': CASES, 'steps_per_epoch': steps}
    identity = object_hash(record)
    if (out / 'identity.json').exists() and json.loads((out / 'identity.json').read_text())['identity'] != identity:
        raise ValueError('existing run identity changed')
    write_json(out / 'identity.json', dict(record, identity=identity))
    write_json(out / 'config.json', config)
    for name in ('games.parquet', 'train_questions.parquet', 'monitor_questions.parquet', 'dev_questions.parquet',
                 'raw_c.npz', 'initial_head.pt'):
        if not (out / name).exists():
            (out / name).symlink_to(os.path.relpath((source / name).resolve(), out.resolve()))
    if not (out / 'requirements.txt').exists():
        shutil.copyfile(ROOT / 'requirements.txt', out / 'requirements.txt')
    if not (out / 'plan.md').exists():
        shutil.copyfile(ROOT / 'plan.md', out / 'plan.md')
    snapshot = out / 'source'
    snapshot.mkdir(exist_ok=True)
    for name in record['sources']:
        path = Path(__file__).with_name(name)
        if not (snapshot / name).exists():
            shutil.copyfile(path, snapshot / name)
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
    import platform
    import importlib.metadata
    import subprocess
    write_json(out / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
               'packages': {name: importlib.metadata.version(name) for name in
                            ('numpy', 'pandas', 'scikit-learn', 'pyarrow')},
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'devices': [torch.cuda.get_device_name(i) for i in range(5)],
               'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': platform.node(),
               'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
               'disk_free_bytes': shutil.disk_usage(out).free})
    arrays, bounds, ram_bytes = allocate(games)
    coverage = {}
    for role, qs in questions.items():
        part = games[games.role.eq(role)]
        coverage[role] = {'source_players': int(part.player_id.nunique()), 'source_games': len(part),
                         'positions': int(part.length.sum()), 'questions': len(qs),
                         'question_players': int(qs.player_id.nunique()),
                         'question_games': len(np.unique(np.concatenate(qs.games.to_list()))),
                         'by_k': {str(k): {'question_players': int(qs[qs.k.eq(k)].player_id.nunique()),
                                          'eligible_players': int(part.groupby('player_id').size().ge(k).sum()),
                                          'questions': int(qs.k.eq(k).sum())} for k in K_BUCKETS}}
    write_json(out / 'coverage.json', coverage)
    write_json(out / 'memory_plan.json', {'shared_cache_bytes': ram_bytes, 'persistent_policy_bytes': 0, 'persistent_budget_bytes': 200 * 2**20})
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
        initial = torch.load(source / 'initial_head.pt', map_location='cpu', weights_only=False)
        mp.spawn(train_worker, args=(games, bounds, arrays, questions, initial, config, identity, out), nprocs=4)
    result = summarize(out, identity, config['seed'])
    result['resources'] = {'wall_seconds': time.monotonic() - started,
                           'estimated_allocated_gpu_hours': 5 * (time.monotonic() - started) / 3600,
                           'allocated_gpus': 5, 'shared_cache_bytes': ram_bytes,
                           'persistent_budget_bytes': 200 * 2**20,
                           'slurm_stdout': f"outputs/slurm-rank-policy-branches-{os.environ.get('SLURM_JOB_ID')}.out",
                           'persistent_bytes_before_final_report': sum(p.stat().st_size for p in out.rglob('*') if p.is_file() and not p.is_symlink())}
    if result['resources']['persistent_bytes_before_final_report'] > 200 * 2**20:
        raise RuntimeError('persistent output exceeded the approved 200 MiB budget; artifacts retained')
    write_json(out / 'result.json', result)
    write_json(out / 'complete.json', {'identity': identity, 'status': 'complete'})
    arrays.clear()
    gc.collect()
    print(json.dumps({'status': 'complete', 'out': str(out), 'resources': result['resources']}), flush=True)


if __name__ == '__main__':
    main()
