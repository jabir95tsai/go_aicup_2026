"""Frozen H/C CNN fusion: three arms, four independently selected linear heads each."""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_residual.branches import gate, paired_difference
from go_aicup.tasks.strength.policy_residual.train import atomic_save
from .features import (ARMS, assemble, dev_side, extract_worker, load_side, preflight,
                       question_values, read_json)


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean', torch.zeros(width))
        self.register_buffer('scale', torch.ones(width))
        self.linear = nn.Linear(width, 10)

    def forward(self, x):
        return self.linear((x - self.mean) / self.scale)


def raw_columns(arm):
    if arm == 'A0':
        return np.arange(38)
    if arm == 'R-HC':
        return np.r_[18:36, 56:76]
    if arm == 'Z-HC':
        return np.r_[64:82, 146:166]
    raise ValueError(arm)


def initialize(x, raw, y, arm, device):
    """All three arms start at the same TRAIN-only H+C sklearn prediction."""
    if set(y.tolist()) != set(range(10)):
        raise ValueError('all ten classes required for each k')
    scaler = StandardScaler().fit(raw)
    fit = LogisticRegression(C=1., max_iter=5000).fit(scaler.transform(raw), y)
    full = StandardScaler().fit(x)
    head = Head(x.shape[1]).to(device)
    columns = raw_columns(arm)
    np.testing.assert_array_equal(x[:, columns], raw)
    with torch.no_grad():
        head.mean.copy_(torch.as_tensor(full.mean_, dtype=torch.float32, device=device))
        head.scale.copy_(torch.as_tensor(full.scale_, dtype=torch.float32, device=device))
        head.linear.weight.zero_()
        head.linear.weight[:, columns] = torch.as_tensor(fit.coef_, dtype=torch.float32, device=device)
        head.linear.bias.copy_(torch.as_tensor(fit.intercept_, dtype=torch.float32, device=device))
        logits = head(torch.as_tensor(x, device=device)).cpu().numpy()
    np.testing.assert_allclose(logits, fit.decision_function(scaler.transform(raw)), atol=2e-4, rtol=2e-4)
    return head


@torch.no_grad()
def evaluate(head, x, y, train_count):
    logits = head(x)
    probs = logits.softmax(1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise ValueError('nonfinite predictions')
    pred = score_rank(probs)
    delta = np.abs(pred - y)
    scores = np.where(delta == 0, 1., np.where(delta == 1, np.exp(-1.), 0.))
    ce = float(F.cross_entropy(logits, torch.as_tensor(y, device=x.device)))
    penalty = float(head.linear.weight.square().sum() / (2 * train_count))
    return {'score': float(scores.mean()), 'ce': ce, 'head_l2': penalty, 'objective': ce + penalty}, pred, scores, probs


def batches(count, batch, seed, epoch):
    if count % batch:
        raise ValueError('per-k question count must divide the fixed batch size')
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(count).reshape(-1, batch)


def fit_head(arm, k, inputs, questions, raw, config, identity, out, device):
    directory = Path(out) / arm / f'k{k}'
    directory.mkdir(parents=True, exist_ok=True)
    masks = {role: q.k.eq(k).to_numpy() for role, q in questions.items()}
    x = {role: torch.as_tensor(inputs[role][mask], device=device) for role, mask in masks.items()}
    y = {role: questions[role].loc[mask, 'rank_id'].to_numpy() for role, mask in masks.items()}
    count = len(y['train'])
    expected = config['epochs'] * count // config['questions_per_k']
    expected = min(expected, config.get('smoke_updates', expected))
    torch.manual_seed(config['seed'])
    head = initialize(inputs['train'][masks['train']], raw[masks['train']], y['train'], arm, device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=0.)
    epoch, step, best_score, history = 0, 0, -1., []
    best_state = None
    checkpoint = directory / 'last.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('B3 checkpoint identity changed')
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        epoch, step, best_score, best_state, history = (saved[name] for name in
                                                      ('epoch', 'step', 'best_score', 'best_state', 'history'))

    def monitor():
        nonlocal best_score, best_state
        reports = {role: evaluate(head, x[role], y[role], count)[0] for role in ('train', 'monitor')}
        history.append({'step': step, **reports})
        if reports['monitor']['score'] > best_score:
            best_score = reports['monitor']['score']
            best_state = {'head': copy.deepcopy(head.state_dict()), 'step': step}
        print(json.dumps({'stage': 'b3_monitor', 'arm': arm, 'k': k, **history[-1]}), flush=True)

    if not history:
        monitor()
    for epoch in range(epoch + 1, config['epochs'] + 1):
        if step >= expected:
            break
        order = batches(count, config['questions_per_k'], config['seed'], epoch)
        for ids in order:
            if step >= expected:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x['train'][ids]), torch.as_tensor(y['train'][ids], device=device))
            loss = loss + head.linear.weight.square().sum() / (2 * count)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite B3 objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError('missing/nonfinite B3 gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise ValueError('nonfinite B3 parameter')
            step += 1
            if step % config['evaluate_every'] == 0:
                monitor()
        # Replaying an interrupted epoch also restores its boundary's best and history.
        atomic_save({'identity': identity, 'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch, 'step': step, 'best_score': best_score,
                     'best_state': best_state, 'history': history}, checkpoint)
    if history[-1]['step'] != step:
        monitor()
    if step != expected:
        raise ValueError('incomplete fixed B3 update budget')
    head.load_state_dict(best_state['head'])
    atomic_save({'identity': identity, **best_state}, directory / 'model.pt')
    write_json(directory / 'learning_curve.json', history)
    result = {'selected_step': best_state['step'], 'completed_updates': step,
              'train_questions': count, 'parameters': sum(p.numel() for p in head.parameters()),
              'roles': {role: evaluate(head, x[role], y[role], count)[0] for role in ('train', 'monitor')}}
    write_json(directory / 'result.json', result)
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})
    return head, result


def predictions(heads, values, questions, count):
    frame = questions[['question_id', 'player_id', 'rank_id', 'k']].copy()
    for k in K_BUCKETS:
        mask = questions.k.eq(k).to_numpy()
        report, pred, scores, probs = evaluate(heads[k], values[mask], questions.loc[mask, 'rank_id'].to_numpy(), count[k])
        frame.loc[mask, 'pred_rank'] = pred
        frame.loc[mask, 'score'] = scores
        frame.loc[mask, [f'p{i}' for i in range(10)]] = probs
    frame['pred_rank'] = frame.pred_rank.astype(int)
    by_k = {str(k): float(frame.loc[frame.k.eq(k), 'score'].mean()) for k in K_BUCKETS}
    return frame, {'by_k': by_k, 'equal_k': float(np.mean(list(by_k.values())))}


def comparisons(frames, seed):
    ref = frames['A0']
    results = {}
    for a, b in (('R-HC', 'A0'), ('Z-HC', 'A0'), ('Z-HC', 'R-HC')):
        for frame in (frames[a], frames[b]):
            pd.testing.assert_frame_equal(frame[['question_id', 'player_id', 'rank_id', 'k']],
                                          ref[['question_id', 'player_id', 'rank_id', 'k']])
        delta = frames[a].score.to_numpy() - frames[b].score.to_numpy()
        diff = paired_difference(frames[a].score.to_numpy(), frames[b].score.to_numpy(),
                                 ref.k.to_numpy(), ref.player_id.to_numpy(), np.random.default_rng(seed + 99))
        diff['by_k'] = {str(k): float(delta[ref.k.eq(k)].mean()) for k in K_BUCKETS}
        diff['equal_k'] = float(np.mean(list(diff['by_k'].values())))
        diff['gate_pass'] = gate(diff)
        results[f'{a}-{b}'] = diff
    return results


def run(config, out, prepare_only=False):
    started = time.monotonic()
    games, questions, sources = preflight(config)
    code = [*Path(__file__).parent.glob('*.py')]
    code += [ROOT / f'src/go_aicup/tasks/strength/{p}' for p in
             ('policy_h/five_arms.py', 'policy_residual/five_arms.py', 'policy_residual/policy_zip.py',
              'policy_residual/data.py', 'policy_residual/branches.py', 'policy_residual/model.py',
              'policy_residual/train.py', 'policy_residual/run.py', 'policy_residual/evaluate_train.py',
              'policy_residual/epoch.py')]
    code += [ROOT / 'src/go_aicup/metrics.py', ROOT / 'src/go_aicup/artifacts.py']
    record = {'config': config, 'sources': sources,
              'code': {str(p.relative_to(ROOT)): file_hash(p) for p in code}}
    identity = object_hash(record)
    if prepare_only:
        print(json.dumps({'status': 'preflight_passed', 'identity': identity,
                          'roles': games.groupby('role').size().to_dict(), 'teacher_inference': False}), flush=True)
        return
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'identity.json').exists() and read_json(out / 'identity.json')['identity'] != identity:
        raise ValueError('existing run identity changed')
    write_json(out / 'identity.json', {'identity': identity, **record})
    write_json(out / 'config.json', config)
    if (out / 'complete.json').exists():
        if read_json(out / 'complete.json') != {'identity': identity, 'status': 'complete'}:
            raise ValueError('completion identity changed')
        return
    if torch.cuda.device_count() < 2:
        raise RuntimeError('this launcher requires two GPUs: H on GPU0, C on GPU1')
    tick = time.monotonic()
    torch.multiprocessing.spawn(extract_worker, args=(config, identity, out), nprocs=2, join=True)
    feature_wall_seconds = time.monotonic() - tick
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    write_json(out / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'gpus': [torch.cuda.get_device_name(i) for i in range(2)],
               'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    fit_games = games[games.role.isin(['train', 'monitor'])].reset_index(drop=True)
    features = assemble(load_side(out, 'H', fit_games.game_index, identity),
                        load_side(out, 'C', fit_games.game_index, identity))
    inputs = {arm: {role: question_values(fit_games, questions[role], features[arm])
                    for role in ('train', 'monitor')} for arm in ARMS}
    fit_q = {role: questions[role] for role in ('train', 'monitor')}
    trained, reports = {}, {}
    tick = time.monotonic()
    for arm in ARMS:
        trained[arm], reports[arm] = {}, {}
        for k in K_BUCKETS:
            trained[arm][k], reports[arm][str(k)] = fit_head(arm, k, inputs[arm], fit_q,
                         inputs['A0']['train'], config, identity, out, device)
    b3_seconds = time.monotonic() - tick
    # Smoke never reads dev feature values or reports a candidate gate.
    roles = ('train', 'monitor') if config.get('smoke_updates') else ('train', 'monitor', 'dev')
    if 'dev' in roles:
        dev_games = games[games.role.eq('dev')].reset_index(drop=True)
        dev = assemble(dev_side(config, 'H', dev_games.game_index), dev_side(config, 'C', dev_games.game_index))
    frames, scores = {}, {}
    counts = {k: int(questions['train'].k.eq(k).sum()) for k in K_BUCKETS}
    for arm in ARMS:
        scores[arm] = {}
        for role in roles:
            values = (question_values(dev_games, questions['dev'], dev[arm]) if role == 'dev' else inputs[arm][role])
            frame, scores[arm][role] = predictions(trained[arm], torch.as_tensor(values, device=device), questions[role], counts)
            frame.to_parquet(out / arm / f'{role}_predictions.parquet', index=False)
            if role == 'dev':
                frames[arm] = frame
    # Recheck small source artifacts so a concurrently replaced checkpoint/feature cannot be published.
    _, _, final_sources = preflight(config)
    if final_sources != sources:
        raise ValueError('upstream sources changed during this run')
    summary = {'identity': identity, 'smoke': bool(config.get('smoke_updates')), 'heads': reports, 'scores': scores,
               'differences': comparisons(frames, config['seed']) if frames else {},
               'feature_wall_seconds': feature_wall_seconds, 'b3_seconds': b3_seconds,
               'wall_seconds': time.monotonic() - started,
               'limitations': ['one seed; historically reused dev', 'upstream CNNs selected using the same monitor',
                               'split player-disjoint here does not certify upstream C-vector player-disjointness']}
    write_json(out / 'summary.json', summary)
    write_json(out / 'complete.json', {'identity': identity, 'status': 'complete'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_hc_b3.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    for name in ('epochs', 'questions_per_k', 'evaluate_every', 'position_chunk', 'zip_threads'):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if 'smoke_updates' in config and (type(config['smoke_updates']) is not int or config['smoke_updates'] <= 0):
        raise ValueError('smoke_updates must be a positive integer')
    if not np.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('lr must be finite and positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run(config, args.out or ROOT / config['out'] / os.environ.get('SLURM_JOB_ID', 'local'), args.prepare_only)


if __name__ == '__main__':
    main()
