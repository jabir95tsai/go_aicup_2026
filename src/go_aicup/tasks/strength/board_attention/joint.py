"""Four controlled question-level B2/attention GD arms; frozen C v2."""
from __future__ import annotations

import argparse
import copy
import json
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
from go_aicup.data.cv import make_rank_questions
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.board_attention import train as base
from go_aicup.tasks.strength.v2.aggregate import official_scores

ARMS = {'mean_gd': None, 'joint': 0., 'joint_kl001': .01, 'joint_kl01': .1}


def questions(table, split, count, seed):
    meta = table[table['set'].eq(split)].set_index('game_index')
    meta = meta.assign(color=meta.player_color.map({0: 'B', 1: 'W'}))
    q = make_rank_questions(meta, meta.player_id.unique(), count, seed)
    lookup = pd.Series(np.arange(len(table)), index=table.game_index)
    rows = [lookup.loc[list(g)].to_numpy() for g in q.games]
    for r, item in zip(rows, q.itertuples()):
        selected = table.iloc[r]
        if (len(set(r)) != item.k or not selected['set'].eq(split).all()
                or not selected.player_id.eq(item.player_id).all()
                or not selected.rank_id.eq(item.rank_id).all()
                or selected.player_color.nunique() != 1):
            raise ValueError('invalid same-player, same-colour question')
    return q, rows


class B2(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(4, 20))
        self.register_buffer('scale', torch.ones(4, 20))
        self.weight = nn.Parameter(torch.zeros(4, 10, 20))
        self.bias = nn.Parameter(torch.zeros(4, 10))

    def forward(self, x, bucket):
        x = (x - self.mean[bucket]) / self.scale[bucket]
        return torch.bmm(self.weight[bucket], x.unsqueeze(2)).squeeze(2) + self.bias[bucket]

    def penalty(self, per_k_count):
        # sklearn C=1: mean CE + ||W||^2/(2*N), averaged across four heads.
        return self.weight.square().sum() / (8 * per_k_count)


def fit_b2(c, q, rows):
    head = B2()
    x = np.stack([c[r].mean(0) for r in rows]).astype(np.float64)
    y = q.rank_id.to_numpy()
    for b, k in enumerate(K_BUCKETS):
        mask = q.k.to_numpy() == k
        scaler = StandardScaler().fit(x[mask])
        lr = LogisticRegression(C=1., max_iter=5000).fit(scaler.transform(x[mask]), y[mask])
        if not np.array_equal(lr.classes_, np.arange(10)):
            raise ValueError('B2 training must contain every rank')
        with torch.no_grad():
            head.mean[b].copy_(torch.from_numpy(scaler.mean_))
            head.scale[b].copy_(torch.from_numpy(scaler.scale_))
            head.weight[b].copy_(torch.from_numpy(lr.coef_))
            head.bias[b].copy_(torch.from_numpy(lr.intercept_))
        with torch.no_grad():
            got = head(torch.tensor(x[mask], dtype=torch.float32), torch.full((mask.sum(),), b)).numpy()
        np.testing.assert_allclose(got, lr.decision_function(scaler.transform(x[mask])), atol=2e-4, rtol=2e-4)
    return head


def weight_kl(weights, slot, groups):
    counts = torch.bincount(slot, minlength=groups).to(weights.dtype)
    if (counts == 0).any():
        raise ValueError('empty game/side')
    terms = weights * (weights.clamp_min(torch.finfo(weights.dtype).tiny).log() + counts[slot].log())
    return weights.new_zeros(groups).index_add_(0, slot, terms).reshape(-1, 2).mean(1)


def aggregate(values, game_rows):
    sizes = np.array([len(r) for r in game_rows])
    device = values.device
    owner = torch.as_tensor(np.repeat(np.arange(len(sizes)), sizes), device=device)
    flat = torch.as_tensor(np.concatenate(game_rows), device=device)
    out = values.new_zeros((len(sizes), *values.shape[1:])).index_add_(0, owner, values[flat])
    shape = (len(sizes),) + (1,) * (values.ndim - 1)
    return out / torch.as_tensor(sizes, device=device).reshape(shape)


def question_forward(attention, head, data, game_rows, bucket, targets, coefficient, per_k_count):
    unique, inverse = np.unique(np.concatenate(game_rows), return_inverse=True)
    edges = np.r_[0, np.cumsum([len(r) for r in game_rows])]
    local_rows = [inverse[a:b] for a, b in zip(edges[:-1], edges[1:])]
    batch = data.batch(unique)
    packed, lp, ply, slot, _ = batch
    if coefficient is None:
        scores = lp.new_zeros(len(lp))
    else:
        with torch.autocast(device_type=lp.device.type, dtype=torch.bfloat16, enabled=lp.is_cuda):
            scores = attention(packed, lp, ply)
    values, weights = base.pool(scores.float(), lp, slot, 2 * len(unique))
    x = aggregate(values.reshape(-1, 20), local_rows)
    kl = aggregate(weight_kl(weights, slot, 2 * len(unique)), local_rows).mean()
    logits = head(x, bucket)
    ce = F.cross_entropy(logits, targets)
    loss = ce + head.penalty(per_k_count) + (0. if coefficient is None else coefficient) * kl
    return loss, ce, kl


@torch.no_grad()
def game_features(attention, data, rows, uniform=False):
    attention.eval()
    features, effective, maximum, divergence = [], 0., 0., 0.
    for start in range(0, len(rows), 128):
        batch = data.batch(rows[start:start + 128])
        packed, lp, ply, slot, target = batch
        if uniform:
            scores = lp.new_zeros(len(lp))
        else:
            with torch.autocast(device_type=lp.device.type, dtype=torch.bfloat16, enabled=lp.is_cuda):
                scores = attention(packed, lp, ply)
        values, w = base.pool(scores.float(), lp, slot, 2 * len(target))
        groups = 2 * len(target)
        sq = w.new_zeros(groups).index_add_(0, slot, w.square())
        mx = w.new_zeros(groups).scatter_reduce_(0, slot, w, reduce='amax')
        effective += float((1 / sq).sum())
        maximum += float(mx.sum())
        divergence += float(weight_kl(w, slot, groups).sum())
        features.append(values.reshape(-1, 20).cpu().numpy())
    return np.concatenate(features), {'effective_moves': effective / (2 * len(rows)),
                                     'max_weight': maximum / (2 * len(rows)), 'kl': divergence / len(rows)}


@torch.no_grad()
def predict(head, c, q, rows):
    x = np.stack([c[r].mean(0) for r in rows])
    device = head.weight.device
    buckets = torch.tensor([K_BUCKETS.index(int(k)) for k in q.k], device=device)
    logits = head(torch.tensor(x, device=device), buckets)
    y = q.rank_id.to_numpy()
    probabilities = logits.softmax(1).cpu().numpy()
    pred = score_rank(probabilities)
    scores = official_scores(pred, y)
    by_k = {str(k): float(scores[q.k.to_numpy() == k].mean()) for k in K_BUCKETS}
    report = {'by_k': by_k, 'equal_k': float(np.mean(list(by_k.values()))),
              'ce': float(F.cross_entropy(logits, torch.tensor(y, device=device)))}
    return report, pred, scores, probabilities


def sample_questions(q, rng, per_cell):
    return np.concatenate([rng.choice(np.flatnonzero((q.k == k) & (q.rank_id == r)), per_cell, replace=False)
                           for k in K_BUCKETS for r in range(10)])


def snapshot(attention, head, identity, step):
    return {'attention': attention.state_dict(), 'b2': head.state_dict(), 'identity': identity, 'step': step}


def train_arm(data, config, arm, out, identity, initial, train_q, train_rows, hold_q, hold_rows, mean_c):
    coefficient = ARMS[arm]
    torch.manual_seed(config['seed'])
    attention = base.MoveWeights().to(data.device, memory_format=torch.channels_last)
    head = B2().to(data.device)
    attention.load_state_dict(initial['attention'])
    head.load_state_dict(initial['b2'])
    optimizer = torch.optim.AdamW([
        {'params': attention.parameters(), 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.}], lr=config['lr'])
    rng = np.random.default_rng(config['seed'])
    per_k_count = len(train_q) // 4
    step, bad = 0, 0
    baseline, *_ = predict(head, mean_c, hold_q, hold_rows)
    best = baseline['equal_k']
    history = [{'step': 0, 'hold': baseline}]
    if (out / 'last.pt').exists():
        saved = torch.load(out / 'last.pt', map_location=data.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('resume identity changed')
        attention.load_state_dict(saved['attention'])
        head.load_state_dict(saved['b2'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        step, bad, best, history = (saved[k] for k in ('step', 'bad', 'best', 'history'))
    else:
        base.atomic_save(snapshot(attention, head, identity, 0), out / 'best.pt')
    hold_games = np.flatnonzero(data.table['set'].eq('hold'))
    started = time.monotonic()
    recent = []
    while step < config['updates'] and not (step >= config['min_updates'] and bad >= config['patience']):
        attention.train()
        indices = sample_questions(train_q, rng, config['questions_per_cell'])
        q = train_q.iloc[indices]
        bucket = torch.tensor([K_BUCKETS.index(int(k)) for k in q.k], device=data.device)
        targets = torch.tensor(q.rank_id.to_numpy(), device=data.device)
        optimizer.zero_grad(set_to_none=True)
        loss, ce, kl = question_forward(attention, head, data, [train_rows[i] for i in indices],
                                         bucket, targets, coefficient, per_k_count)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite joint training loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise ValueError('nonfinite joint training gradient')
        optimizer.step()
        step += 1
        recent.append([float(loss.detach()), float(ce.detach()), float(kl.detach())])
        if step % 25 == 0:
            print(json.dumps({'arm': arm, 'step': step, 'loss_ce_kl': np.mean(recent, axis=0).tolist(),
                              'seconds': time.monotonic() - started}), flush=True)
        if step % config['every'] == 0 or step == config['updates']:
            c = mean_c.copy()
            c[hold_games], stats = game_features(attention, data, hold_games, uniform=coefficient is None)
            report, *_ = predict(head, c, hold_q, hold_rows)
            if report['equal_k'] > best:
                best, bad = report['equal_k'], 0
                base.atomic_save(snapshot(attention, head, identity, step), out / 'best.pt')
            else:
                bad = bad + 1 if step >= config['min_updates'] else 0
            history.append({'step': step, 'hold': report, 'weights': stats,
                            'train_loss_ce_kl': np.mean(recent, axis=0).tolist(),
                            'seconds': time.monotonic() - started})
            recent.clear()
            write_json(out / 'learning_curve.json', history)
            saved = snapshot(attention, head, identity, step)
            saved.update(optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                         bad=bad, best=best, history=history)
            base.atomic_save(saved, out / 'last.pt')
            print(json.dumps(history[-1]), flush=True)
    saved = torch.load(out / 'best.pt', map_location=data.device, weights_only=False)
    attention.load_state_dict(saved['attention'])
    head.load_state_dict(saved['b2'])
    base.atomic_save(saved, out / 'model.pt')
    c, stats = game_features(attention, data, np.arange(len(data.table)), uniform=coefficient is None)
    np.savez(out / 'features.npz', game_index=data.table.game_index.to_numpy(), c=c)
    hold_report, *_ = predict(head, c, hold_q, hold_rows)
    write_json(out / 'training_complete.json', {'identity': identity, 'updates': step, 'best_step': saved['step'],
               'hold': hold_report, 'weights_all': stats, 'seconds': time.monotonic() - started})
    return head, c


def run(config, arm):
    out = ROOT / config['out'] / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Existing loader validates teacher, board timing and player-disjoint split.
    data = base.load_data(out, torch.device('cuda'))
    manifest = json.loads((out / 'data.json').read_text())
    manifest.pop('seconds')
    sources = {name: file_hash(ROOT / name) for name in (
        'src/go_aicup/tasks/strength/board_attention/joint.py',
        'src/go_aicup/tasks/strength/board_attention/train.py', 'src/go_aicup/data/cv.py')}
    identity = object_hash({'config': config, 'arm': arm, 'data': manifest, 'sources': sources,
                            'table': data.table.to_json(orient='split')})
    if (out / 'complete.json').exists():
        if json.loads((out / 'complete.json').read_text())['identity'] != identity:
            raise ValueError('completed run identity changed')
        return
    write_json(out / 'run_identity.json', {'identity': identity, 'config': config, 'arm': arm,
                                          'sources': sources, 'torch': torch.__version__,
                                          'gpu': torch.cuda.get_device_name()})
    train_q, train_rows = questions(data.table, 'train', config['train_questions'], config['seed'])
    hold_q, hold_rows = questions(data.table, 'hold', config['hold_questions'], config['seed'] + 1)
    train_q.to_parquet(out / 'train_questions.parquet', index=False)
    hold_q.to_parquet(out / 'hold_questions.parquet', index=False)
    torch.manual_seed(config['seed'])
    mu, sd = base.normalisation(data)
    attention = base.MoveWeights(mu, sd).to(data.device, memory_format=torch.channels_last)
    mean_c, _ = game_features(attention, data, np.arange(len(data.table)), uniform=True)
    if (out / 'initial.pt').exists():
        initial = torch.load(out / 'initial.pt', map_location='cpu', weights_only=False)
        if initial['identity'] != identity:
            raise ValueError('initialisation identity changed')
        head = B2()
        head.load_state_dict(initial['b2'])
    else:
        head = fit_b2(mean_c, train_q, train_rows)
        initial = {'attention': copy.deepcopy(attention.cpu().state_dict()), 'b2': head.state_dict(),
                   'identity': identity, 'step': 0}
        base.atomic_save(initial, out / 'initial.pt')
    baseline_head = copy.deepcopy(head).to(data.device)
    del attention, head
    if (out / 'training_complete.json').exists():
        saved = torch.load(out / 'model.pt', map_location=data.device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('model identity changed')
        head = B2().to(data.device)
        head.load_state_dict(saved['b2'])
        c = np.load(out / 'features.npz')['c']
    else:
        head, c = train_arm(data, config, arm, out, identity, initial,
                            train_q, train_rows, hold_q, hold_rows, mean_c)
    # U labels are used only after this arm's checkpoint is frozen; never select on U.
    from scripts.strength.humansl_hd_filter import bootstrap
    result = {'identity': identity, 'arm': arm, 'hold': predict(head, c, hold_q, hold_rows)[0],
              'sklearn_hold': predict(baseline_head, mean_c, hold_q, hold_rows)[0], 'U': {},
              'limitation': 'HD0 C-only; fixed rank-balanced same-colour questions. One training seed. '
                            'U reused historically; draws share U players. No production promotion.'}
    for seed in (2027, 2028):
        q, rows = questions(data.table, 'U', config['u_questions'], seed)
        q.to_parquet(out / f'HD0_Q{seed}.parquet', index=False)
        report, pred, scores, proba = predict(head, c, q, rows)
        ref, ref_pred, ref_scores, _ = predict(baseline_head, mean_c, q, rows)
        result['U'][str(seed)] = {'joint': report, 'sklearn_mean': ref,
            'difference': {str(k): float((scores - ref_scores)[q.k.to_numpy() == k].mean()) for k in K_BUCKETS},
            'equal_k_difference': report['equal_k'] - ref['equal_k'],
            'player_bootstrap_95': bootstrap(ref_scores, scores, q)}
        predictions = q.drop(columns='games').assign(pred_rank=pred, score=scores,
                                                     sklearn_rank=ref_pred, sklearn_score=ref_scores)
        for r in range(10):
            predictions[f'p_{r}'] = proba[:, r]
        predictions.to_parquet(out / f'predictions_Q{seed}.parquet', index=False)
    write_json(out / 'result.json', result)
    write_json(out / 'complete.json', {'identity': identity, 'arm': arm, 'result': 'result.json'})
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/strength/board_attention_joint.json')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if (config['train_questions'] % 40 or config['hold_questions'] % 40 or config['u_questions'] % 40
            or min(config['questions_per_cell'], config['every'], config['patience'], config['lr']) <= 0
            or config['min_updates'] > config['updates']):
        raise ValueError('invalid joint experiment configuration')
    run(config, args.arm)


if __name__ == '__main__':
    main()
