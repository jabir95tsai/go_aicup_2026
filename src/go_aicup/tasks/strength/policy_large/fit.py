"""Shared warm starts, exact chunked gradients and monitor-selected upstream models."""
from dataclasses import asdict
from pathlib import Path
import copy
import math
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from go_aicup.artifacts import write_json
from go_aicup.tasks.strength.policy_residual.data import question_rows
from go_aicup.tasks.strength.policy_residual.epoch import epoch_batches
from go_aicup.tasks.strength.policy_residual.train import atomic_save, predict
from .models import INITIALIZATION_LOCK, build, module_for
from .stopping import LossStopper, ABORT

HEAD_DENOMINATOR = 8000


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def raw_head(family):
    module = module_for(family)
    return module.Head() if family == 'H' else module.OriginalHead()


def fit_initial(family, raw, games, train_questions):
    raw = np.asarray(raw)
    rows = question_rows(games, train_questions)
    x = np.stack([raw[r].mean(0, dtype=np.float64) for r in rows])
    if not np.isfinite(x).all():
        raise ValueError('nonfinite warm-start TRAIN features')
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=HEAD_DENOMINATOR / len(train_questions), max_iter=5000).fit(
        scaler.transform(x), train_questions.rank_id)
    if not np.array_equal(lr.classes_, np.arange(10)) or np.any(lr.n_iter_ >= lr.max_iter):
        raise ValueError('invalid or unconverged warm-start LR')
    head = raw_head(family)
    with torch.no_grad():
        head.mean.copy_(torch.as_tensor(scaler.mean_))
        head.scale.copy_(torch.as_tensor(scaler.scale_))
        head.linear.weight.copy_(torch.as_tensor(lr.coef_))
        head.linear.bias.copy_(torch.as_tensor(lr.intercept_))
        np.testing.assert_allclose(head(torch.as_tensor(x, dtype=torch.float32)).numpy(),
                                   lr.decision_function(scaler.transform(x)), atol=2e-4, rtol=2e-4)
    return cpu_tree(head.state_dict())


def layout(games, q):
    rows = question_rows(games, q)
    unique = np.unique(np.concatenate(rows))
    return unique, [np.searchsorted(unique, r) for r in rows]


@torch.no_grad()
def evaluate(family, model, head, cache, questions, chunk):
    ids, rows = layout(cache.games, questions)
    c = module_for(family).game_features(model, cache, ids, chunk)
    report, pred, scores, probs = predict(head, c, questions, rows)
    l2 = float(head.linear.weight.square().sum()) / (2 * HEAD_DENOMINATOR)
    report.update(head_l2=l2, objective=report['ce'] + l2)
    if not all(math.isfinite(report[k]) for k in ('ce', 'equal_k', 'objective')):
        raise ValueError('nonfinite upstream monitor')
    return report, pred, scores, probs


def load_arm(family, arm, path, device):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['family'] != family or payload['arm'] != arm:
        raise ValueError('checkpoint family/arm changed')
    with INITIALIZATION_LOCK:
        model = build(family, arm, payload['seed'])
        head = module_for(family).BranchHead(model.width, payload['initial_head'])
    model.load_state_dict(payload['model'])
    head.load_state_dict(payload['head'])
    return model.to(device).eval(), head.to(device).eval()


def train_arm(family, arm, cache, questions, initial, config, identity, out):
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'last.pt').exists() or (directory / 'model.pt').exists():
        raise FileExistsError(f'refusing to restart without the volatile policy: {directory}')
    module, device = module_for(family), cache.device
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    with INITIALIZATION_LOCK:
        model = build(family, arm, config['seed']).to(device)
        head = module.BranchHead(model.width, initial).to(device)
        reference = raw_head(family).to(device)
        reference.load_state_dict(initial)
    chunk, per_k = config['position_chunk'], config['questions_per_k']
    train = questions['train'].reset_index(drop=True)
    train_rows = question_rows(cache.games, train)
    probe_n = max(1, config.get('train_probe_questions', 8000) // 40)
    probe = train.groupby(['k', 'rank_id'], sort=True).head(probe_n).reset_index(drop=True)
    steps_per_epoch = len(epoch_batches(train, config['seed'], 1, per_k))
    stopper = LossStopper(steps_per_epoch, config['patience'])
    optimizer = torch.optim.AdamW([
        {'params': list(model.parameters()), 'weight_decay': config['weight_decay']},
        {'params': list(head.parameters()), 'weight_decay': 0.}], lr=config['lr'])
    best_score, selected, history, step, early = -1., None, [], 0, False
    seen = np.zeros(len(cache.games), dtype=bool)
    started = time.monotonic()

    def payload():
        return {'family': family, 'arm': arm, 'seed': config['seed'], 'identity': identity, 'step': step,
                'model': cpu_tree(model.state_dict()), 'head': cpu_tree(head.state_dict()),
                'initial_head': initial}

    def monitor():
        nonlocal best_score, selected
        model.eval(), head.eval()
        reports = {role: evaluate(family, model, head, cache, q, chunk)[0]
                   for role, q in [('train_diagnostic', probe), ('monitor', questions['monitor'])]}
        if step == 0:
            ids, rows = layout(cache.games, questions['monitor'])
            raw = cache.arrays['c'][ids].to(device)
            with torch.no_grad():
                expected = predict(reference, raw, questions['monitor'], rows)[3]
                got = evaluate(family, model, head, cache, questions['monitor'], chunk)[3]
            np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-5)
        stop = stopper.observe(reports['monitor']['objective'], step)
        if reports['monitor']['equal_k'] > best_score:
            best_score = reports['monitor']['equal_k']
            selected = payload()
            selected['monitor'] = reports['monitor']
            atomic_save(selected, directory / 'model.pt')
        history.append({'step': step, **reports, 'stopping': asdict(stopper)})
        write_json(directory / 'learning_curve.json', history)
        print({'stage': 'monitor', 'family': family, 'arm': arm, **history[-1]}, flush=True)
        return stop

    monitor()
    for epoch in range(1, config['epochs'] + 1):
        rng = np.random.default_rng(np.random.SeedSequence([config['seed'], epoch, 4]))
        visited = np.zeros(len(train), dtype=np.int8)
        for batch in epoch_batches(train, config['seed'], epoch, per_k):
            if ABORT.is_set():
                raise RuntimeError('another experiment worker failed')
            model.train(), head.train()
            optimizer.zero_grad(set_to_none=True)
            y = torch.as_tensor(train.iloc[batch].rank_id.to_numpy(), device=device)
            loss, used = module.backward_questions(model, head, cache, [train_rows[i] for i in batch], y,
                                                   HEAD_DENOMINATOR, chunk, int(rng.integers(8)))
            if not math.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                    for p in list(model.parameters()) + list(head.parameters())):
                raise ValueError('missing/nonfinite upstream gradient or loss')
            optimizer.step()
            step += 1
            visited[batch] += 1
            seen[used] = True
            if step % config['evaluate_every'] == 0 and monitor():
                early = True
                break
        if not early and not np.all(visited == 1):
            raise ValueError('TRAIN epoch did not visit every question once')
        saved = payload()
        saved.update(optimizer=cpu_tree(optimizer.state_dict()), epoch=epoch, history=history,
                     stopping=asdict(stopper), selected=selected, early_stopped=early,
                     d4_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
        atomic_save(saved, directory / 'last.pt')
        if early:
            break
    if history[-1]['step'] != step:
        monitor()
    result = {'identity': identity, 'family': family, 'arm': arm, 'completed_updates': step,
              'selected_step': selected['step'], 'early_stopped': early, 'stopping': asdict(stopper),
              'steps_per_epoch': steps_per_epoch, 'seconds': time.monotonic() - started,
              'encoder_parameters': sum(p.numel() for p in model.parameters()),
              'head_parameters': sum(p.numel() for p in head.parameters()),
              'training_games_seen': int(seen.sum()), 'training_players_seen': int(cache.games.loc[seen].player_id.nunique()),
              'monitor': selected['monitor']}
    write_json(directory / 'result.json', result)
    write_json(directory / 'complete.json', {'identity': identity, 'status': 'complete'})
    return result
