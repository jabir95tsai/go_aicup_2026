"""Same-name frozen H/C features and twenty independently monitored linear heads."""
from dataclasses import asdict
from pathlib import Path
import copy
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from go_aicup.artifacts import write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength.policy_hc_b3.train import Head, evaluate, batches
from go_aicup.tasks.strength.policy_residual.train import atomic_save
from .stopping import LossStopper, ABORT

ARMS = ('A0', 'R10', 'Z32', 'CNN-R10', 'CNN-Z32')
REGULARIZATION_COUNT = 2000


def assemble(arm, h, c, raw_h, raw_c):
    if arm == 'A0':
        return np.column_stack((raw_h, raw_c)).astype(np.float32, copy=False)
    if arm in ('R10', 'CNN-R10'):
        return np.column_stack((h, raw_h, c, raw_c)).astype(np.float32, copy=False)
    if arm in ('Z32', 'CNN-Z32'):
        zh = h.reshape(-1, 2, 41)[:, :, 9:].reshape(-1, 64)
        zc = c.reshape(-1, 2, 42)[:, :, 10:].reshape(-1, 64)
        np.testing.assert_allclose(h.reshape(-1, 2, 41)[:, :, :9].reshape(-1, 18), raw_h,
                                   atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(c.reshape(-1, 2, 42)[:, :, :10].reshape(-1, 20), raw_c,
                                   atol=2e-6, rtol=2e-6)
        return np.column_stack((zh, raw_h, zc, raw_c)).astype(np.float32, copy=False)
    raise ValueError(arm)


def raw_columns(arm):
    if arm == 'A0':
        return np.arange(38)
    if arm in ('R10', 'CNN-R10'):
        return np.r_[18:36, 56:76]
    if arm in ('Z32', 'CNN-Z32'):
        return np.r_[64:82, 146:166]
    raise ValueError(arm)


def question_values(games, questions, values):
    from go_aicup.tasks.strength.policy_residual.data import question_rows
    return np.stack([values[r].mean(0) for r in question_rows(games, questions)]).astype(np.float32)


def initial_state(arm, x, raw, y):
    if set(y.tolist()) != set(range(10)) or not np.isfinite(x).all():
        raise ValueError('invalid B3 TRAIN classes or features')
    scaler = StandardScaler().fit(raw)
    lr = LogisticRegression(C=REGULARIZATION_COUNT / len(y), max_iter=5000).fit(scaler.transform(raw), y)
    full = StandardScaler().fit(x)
    head = Head(x.shape[1])
    cols = raw_columns(arm)
    np.testing.assert_array_equal(x[:, cols], raw)
    with torch.no_grad():
        head.mean.copy_(torch.as_tensor(full.mean_, dtype=torch.float32))
        head.scale.copy_(torch.as_tensor(full.scale_, dtype=torch.float32))
        head.linear.weight.zero_()
        head.linear.weight[:, cols] = torch.as_tensor(lr.coef_, dtype=torch.float32)
        head.linear.bias.copy_(torch.as_tensor(lr.intercept_, dtype=torch.float32))
        np.testing.assert_allclose(head(torch.as_tensor(x)).numpy(), lr.decision_function(scaler.transform(raw)),
                                   atol=2e-4, rtol=2e-4)
    return head.state_dict()


def fit(arm, k, values, raw, questions, config, identity, out, device):
    directory = Path(out) / arm / f'k{k}'
    directory.mkdir(parents=True, exist_ok=True)
    masks = {role: q.k.eq(k).to_numpy() for role, q in questions.items() if role != 'dev'}
    x = {role: torch.as_tensor(values[role][m], device=device) for role, m in masks.items()}
    y = {role: questions[role].loc[m, 'rank_id'].to_numpy() for role, m in masks.items()}
    count = len(y['train'])
    batch = config['questions_per_k']
    if count % batch:
        raise ValueError('B3 batch must divide per-k TRAIN count')
    steps_per_epoch = count // batch
    head = Head(x['train'].shape[1]).to(device)
    head.load_state_dict(initial_state(arm, values['train'][masks['train']], raw['train'][masks['train']], y['train']))
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['lr'], weight_decay=0.)
    stopper = LossStopper(steps_per_epoch, config['patience'])
    best_score, selected, history = -1., None, []
    started, step, epoch, early = time.monotonic(), 0, 0, False

    def monitor():
        nonlocal best_score, selected
        reports = {role: evaluate(head, x[role], y[role], REGULARIZATION_COUNT)[0] for role in x}
        stop = stopper.observe(reports['monitor']['objective'], step)
        if reports['monitor']['score'] > best_score:
            best_score = reports['monitor']['score']
            selected = {'identity': identity, 'step': step, 'head': copy.deepcopy(head.state_dict())}
            atomic_save(selected, directory / 'model.pt')
        entry = {'step': step, **reports, 'stopping': asdict(stopper)}
        history.append(entry)
        write_json(directory / 'learning_curve.json', history)
        print({'stage': 'b3_monitor', 'arm': arm, 'k': k, **entry}, flush=True)
        return stop

    monitor()
    for epoch in range(1, config['epochs'] + 1):
        for ids in batches(count, batch, config['seed'], epoch):
            if ABORT.is_set():
                raise RuntimeError('another experiment worker failed')
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x['train'][ids]), torch.as_tensor(y['train'][ids], device=device))
            loss = loss + head.linear.weight.square().sum() / (2 * REGULARIZATION_COUNT)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite B3 training objective')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise ValueError('missing or nonfinite B3 gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise ValueError('nonfinite B3 parameters')
            step += 1
            if step % config['evaluate_every'] == 0 and monitor():
                early = True
                break
        atomic_save({'identity': identity, 'head': head.state_dict(), 'optimizer': optimizer.state_dict(),
                     'epoch': epoch, 'step': step, 'early_stopped': early, 'stopping': asdict(stopper),
                     'selected': selected, 'history': history}, directory / 'last.pt')
        if early:
            break
    if history[-1]['step'] != step:
        monitor()
    head.load_state_dict(selected['head'])
    result = {'identity': identity, 'selected_step': selected['step'], 'completed_updates': step,
              'epochs_entered': epoch, 'early_stopped': early, 'stopping': asdict(stopper),
              'train_questions': count, 'parameters': sum(p.numel() for p in head.parameters()),
              'seconds': time.monotonic() - started,
              'roles': {role: evaluate(head, x[role], y[role], REGULARIZATION_COUNT)[0] for role in x}}
    write_json(directory / 'result.json', result)
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})
    return head, result
