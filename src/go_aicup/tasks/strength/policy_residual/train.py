from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from go_aicup.artifacts import write_json
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.tasks.strength.v2.aggregate import official_scores
from .data import Cache, question_rows
from .model import ARMS, Head, Residual, probabilities, question_mean, statistics


def atomic_save(value, path):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def fit_head(c, questions, rows):
    x = np.stack([c[r].mean(0) for r in rows]).astype(np.float64)
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=1., max_iter=5000).fit(scaler.transform(x), questions.rank_id)
    if not np.array_equal(lr.classes_, np.arange(10)):
        raise ValueError('head requires all ten ranks')
    head = Head()
    with torch.no_grad():
        head.mean.copy_(torch.from_numpy(scaler.mean_))
        head.scale.copy_(torch.from_numpy(scaler.scale_))
        head.linear.weight.copy_(torch.from_numpy(lr.coef_))
        head.linear.bias.copy_(torch.from_numpy(lr.intercept_))
        got = head(torch.tensor(x, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(got, lr.decision_function(scaler.transform(x)), atol=2e-4, rtol=2e-4)
    return head.state_dict()


@torch.no_grad()
def game_features(model, cache, rows, chunk, symmetry=0):
    if model.arm == 'A0':
        return cache.arrays['c'][rows].to(cache.device)
    counts = cache.counts(rows)
    total = torch.zeros(len(rows) * 2, 10, device=cache.device)
    for lp, target, q, slot in cache.chunks(rows, chunk):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
            evidence = model(lp, target, q, symmetry)
        total.index_add_(0, slot, evidence.float() - q)
    # Add corrections to one shared baseline, so all five step-0 logits are identical.
    return cache.arrays['c'][rows].to(cache.device) + (total / counts[:, None]).reshape(-1, 20)


def backward_questions(model, head, cache, rows, targets, train_count, chunk, symmetry=0):
    """Exact chain rule through side/game means, with bounded per-position activation memory."""
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
        grad = c.grad.reshape(-1, 10) / cache.counts(unique)[:, None]
        for lp, target, q, slot in cache.chunks(unique, chunk):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
                evidence = model(lp, target, q, symmetry)
            evidence.backward(grad[slot])
    return float(loss.detach()), unique


@torch.no_grad()
def predict(head, c, questions, rows):
    x = question_mean(c, rows)
    logits = head(x)
    y = questions.rank_id.to_numpy()
    probabilities_ = logits.softmax(1).cpu().numpy()
    pred = score_rank(probabilities_)
    scores = official_scores(pred, y)
    k = questions.k.to_numpy()
    by_k = {str(b): float(scores[k == b].mean()) for b in K_BUCKETS}
    report = {'by_k': by_k, 'equal_k': float(np.mean(list(by_k.values()))),
              'ce': float(F.cross_entropy(logits, torch.as_tensor(y, device=logits.device)))}
    return report, pred, scores, probabilities_


@torch.no_grad()
def standardize_statistics(model, cache, train_rows, chunk):
    total = torch.zeros(80, dtype=torch.float64, device=cache.device)
    square, n = total.clone(), 0
    for lp, target, q, slot in cache.chunks(train_rows, chunk):
        p, safe, legal = probabilities(lp)
        x = torch.cat((q, statistics(p, safe, legal, q)), 1).double()
        total += x.sum(0)
        square += x.square().sum(0)
        n += len(x)
    mean = total / n
    model.mean.copy_(mean.float())
    model.scale.copy_((square / n - mean.square()).clamp_min(0).sqrt().clamp_min(1e-5).float())


def sample_questions(q, rng, per_k):
    return np.concatenate([rng.choice(np.flatnonzero(q.k.to_numpy() == k), per_k, replace=False)
                           for k in K_BUCKETS])


def state(model, head, identity, step):
    return {'model': model.state_dict(), 'head': head.state_dict(), 'identity': identity, 'step': step}


def train_worker(rank, games, bounds, arrays, questions, initial_head, config, identity, out):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    arm = ARMS[rank]
    directory = out / arm
    directory.mkdir(exist_ok=True)
    if (directory / 'complete.json').exists():
        saved = json.loads((directory / 'complete.json').read_text())
        if saved['identity'] != identity:
            raise ValueError('completed arm identity changed')
        return
    torch.manual_seed(config['seed'])
    model, head = Residual(arm).to(device), Head().to(device)
    head.load_state_dict(initial_head)
    cache = Cache(games, bounds, arrays, device)
    train_games = np.flatnonzero(games.role.eq('train'))
    monitor_games = np.flatnonzero(games.role.eq('monitor'))
    q_train, q_mon = questions['train'], questions['monitor']
    train_rows = question_rows(games, q_train)
    mon_rows = question_rows(games.iloc[monitor_games].reset_index(drop=True), q_mon)
    chunk = config['position_chunk']
    if arm == 'S':
        standardize_statistics(model, cache, train_games, chunk)
    for lp, target, q, slot in cache.chunks(train_games[:1], chunk):
        torch.testing.assert_close(model(lp, target, q), q, atol=0, rtol=0)
    initial = game_features(model, cache, monitor_games, chunk)
    torch.testing.assert_close(initial, arrays['c'][monitor_games].to(device), atol=2e-5, rtol=2e-5)
    baseline, *_ = predict(head, initial, q_mon, mon_rows)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'weight_decay': config['weight_decay']},
        {'params': head.parameters(), 'weight_decay': 0.}], lr=config['lr'])
    rng, step, bad, best = np.random.default_rng(config['seed']), 0, 0, baseline['equal_k']
    history = [{'step': 0, 'monitor': baseline}]
    seen = np.zeros(len(games), dtype=bool)
    last_path = directory / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('resume identity changed')
        model.load_state_dict(saved['model'])
        head.load_state_dict(saved['head'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
        step, bad, best, history, seen = (saved[k] for k in ('step', 'bad', 'best', 'history', 'seen'))
    else:
        atomic_save(state(model, head, identity, 0), directory / 'initial.pt')
        atomic_save(state(model, head, identity, 0), directory / 'model.pt')
    started, recent, occurrences = time.monotonic(), [], 0
    while step < config['updates'] and not (step >= config['min_updates'] and bad >= config['patience']):
        model.train(), head.train()
        chosen = sample_questions(q_train, rng, config['questions_per_k'])
        symmetry = int(rng.integers(8))
        rows = [train_rows[i] for i in chosen]
        targets = torch.as_tensor(q_train.iloc[chosen].rank_id.to_numpy(), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, unique = backward_questions(model, head, cache, rows, targets, len(q_train), chunk, symmetry)
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise ValueError('nonfinite residual gradient')
        optimizer.step()
        seen[unique] = True
        occurrences += sum(len(r) for r in rows)
        step += 1
        recent.append(loss)
        if step % 25 == 0:
            print(json.dumps({'stage': 'train', 'arm': arm, 'step': step,
                              'loss': float(np.mean(recent)), 'seconds': time.monotonic() - started}), flush=True)
        if step % config['every'] == 0 or step == config['updates']:
            model.eval(), head.eval()
            c = game_features(model, cache, monitor_games, chunk)
            report, *_ = predict(head, c, q_mon, mon_rows)
            if report['equal_k'] > best:
                best, bad = report['equal_k'], 0
                atomic_save(state(model, head, identity, step), directory / 'model.pt')
            else:
                bad = bad + 1 if step >= config['min_updates'] else 0
            history.append({'step': step, 'monitor': report, 'train_loss': float(np.mean(recent)),
                            'seconds': time.monotonic() - started})
            recent.clear()
            write_json(directory / 'learning_curve.json', history)
            saved = state(model, head, identity, step)
            saved.update(optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                         bad=bad, best=best, history=history, seen=seen)
            atomic_save(saved, last_path)
            print(json.dumps({'arm': arm, **history[-1]}), flush=True)
    # No dev labels participate in gradients, early stopping, or checkpoint selection.
    best_state = torch.load(directory / 'model.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_state['model']), head.load_state_dict(best_state['head'])
    model.eval(), head.eval()
    test_games = np.flatnonzero(games.role.eq('dev'))
    q_test = questions['dev']
    test_rows = question_rows(games.iloc[test_games].reset_index(drop=True), q_test)
    c = game_features(model, cache, test_games, chunk)
    report, pred, scores, probs = predict(head, c, q_test, test_rows)
    original_head = Head().to(device)
    original_head.load_state_dict(initial_head)
    reference, ref_pred, ref_scores, ref_probs = predict(original_head, arrays['c'][test_games].to(device),
                                                       q_test, test_rows)
    frame = q_test[['question_id', 'player_id', 'rank_id', 'k']].copy()
    frame['pred_rank'], frame['score'] = pred, scores
    frame['initial_pred_rank'], frame['initial_score'] = ref_pred, ref_scores
    for r in range(10):
        frame[f'p{r}'] = probs[:, r]
        frame[f'initial_p{r}'] = ref_probs[:, r]
    frame.to_parquet(directory / 'predictions.parquet', index=False)
    np.savez(directory / 'features.npz', game_index=games.iloc[test_games].game_index.to_numpy(),
             c=c.cpu().numpy())
    result = {'arm': arm, 'identity': identity, 'dev': report, 'initial_sklearn_dev': reference,
              'selected_step': best_state['step'], 'completed_updates': step,
              'training_games_seen': int(seen.sum()), 'training_players_seen': int(games.loc[seen].player_id.nunique()),
              'game_occurrences_this_session': occurrences,
              'parameters': sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters()),
              'seconds': time.monotonic() - started, 'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
    write_json(directory / 'result.json', result)
    write_json(directory / 'complete.json', {'identity': identity, 'arm': arm})
