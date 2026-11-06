"""CPU per-k AdamW training, exact mid-epoch resume, and fixed-epoch refit."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import score_rank, NEAR
from .common import A, S, H, C, BUCKETS, save_torch, writer_lock
from .data import load_prepared, questions
from .model import Head, probabilities


def score(pred, truth):
    error = np.abs(np.asarray(pred) - np.asarray(truth))
    return float(((error == 0).sum() + NEAR * (error == 1).sum()) / len(error))


def array_hash(values):
    import hashlib
    values = np.ascontiguousarray(values)
    return hashlib.sha256(str((values.dtype.str, values.shape)).encode() + values.tobytes()).hexdigest()


def train_head(x, y, valid_x, valid_y, directory, settings, seed, identity, *, fixed_epochs=None,
               stop_after_updates=None):
    """Inputs are unstandardized fitting questions only; validation never fits scaler."""
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int64)
    if (x.ndim != 2 or x.shape != (len(y), 60) or not len(y) or not np.isfinite(x).all()
            or set(y) != set(range(10))):
        raise ValueError('fitting requires finite 60-vectors and all ten classes')
    if fixed_epochs is None:
        valid_x, valid_y = np.asarray(valid_x, np.float32), np.asarray(valid_y, np.int64)
        if (valid_x.shape != (len(valid_y), 60) or not len(valid_y) or not np.isfinite(valid_x).all()
                or not np.isin(valid_y, range(10)).all()):
            raise ValueError('invalid validation questions')
    elif not 1 <= fixed_epochs <= settings['epochs']:
        raise ValueError('invalid final epoch budget')
    directory = Path(directory)
    signature = object_hash(dict(identity=identity, settings=settings, seed=seed, fixed_epochs=fixed_epochs,
                                 x=array_hash(x), y=array_hash(y),
                                 validation=None if fixed_epochs is not None else
                                 [array_hash(valid_x), array_hash(valid_y)]))
    steps = math.ceil(len(y) / settings['batch_size'])
    scaler = StandardScaler().fit(x)
    inputs = torch.from_numpy(scaler.transform(x).astype(np.float32))
    targets = torch.from_numpy(y)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Head()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'],
                                 betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings['epochs'] * steps, eta_min=settings['min_lr'])
    generator = torch.Generator().manual_seed(seed)
    state = dict(format='se_h_c_b2_head_v1', signature=signature, epoch=0, cursor=0, updates=0,
                 order=None, best_epoch=0, best_score=-1.0, patience=0, best_model=None,
                 history=[], loss_sum=0.0, samples=0, complete=False,
                 mean=scaler.mean_, scale=scaler.scale_)
    latest = directory / 'latest.pt'
    with writer_lock(directory):
        if latest.exists():
            state = torch.load(latest, weights_only=False, map_location='cpu')
            if state['signature'] != signature or state['format'] != 'se_h_c_b2_head_v1':
                raise ValueError('training checkpoint identity mismatch')
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler']); generator.set_state(state['rng'])
            if not np.array_equal(state['mean'], scaler.mean_) or not np.array_equal(state['scale'], scaler.scale_):
                raise ValueError('checkpoint scaler mismatch')

        def save():
            state.update(model=deepcopy(model.state_dict()), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), rng=generator.get_state())
            save_torch(latest, state)

        budget = fixed_epochs if fixed_epochs is not None else settings['epochs']
        while not state['complete'] and state['epoch'] < budget:
            if state['order'] is None:
                state['order'] = torch.randperm(len(y), generator=generator)
            model.train()
            while state['cursor'] < len(y):
                indices = state['order'][state['cursor']:state['cursor'] + settings['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('nonfinite/missing head gradients')
                optimizer.step(); scheduler.step()
                state['updates'] += 1; state['cursor'] += len(indices)
                state['loss_sum'] += float(loss.detach()) * len(indices); state['samples'] += len(indices)
                if stop_after_updates is not None and state['updates'] >= stop_after_updates:
                    save()
                    return None
            state['epoch'] += 1
            row = dict(epoch=state['epoch'], updates=state['updates'], lr=scheduler.get_last_lr()[0],
                       loss=state['loss_sum'] / state['samples'])
            if fixed_epochs is None:
                record = dict(model=model.state_dict(), mean=scaler.mean_, scale=scaler.scale_)
                p = probabilities({5: record}, valid_x, np.full(len(valid_y), 5))
                row['score'] = score(score_rank(p), valid_y)
                if row['score'] > state['best_score']:
                    state.update(best_score=row['score'], best_epoch=state['epoch'], patience=0,
                                 best_model=deepcopy(model.state_dict()))
                else:
                    state['patience'] += 1
            else:
                state.update(best_epoch=state['epoch'], best_model=deepcopy(model.state_dict()))
            state['history'].append(row)
            state.update(order=None, cursor=0, loss_sum=0.0, samples=0)
            state['complete'] = state['epoch'] >= budget or (fixed_epochs is None and state['patience'] >= settings['patience'])
            save()
            write_json(directory / 'history.json', state['history'])
        record = dict(format=state['format'], signature=signature, model=state['best_model'],
                      mean=state['mean'], scale=state['scale'], best_epoch=state['best_epoch'],
                      stopped_epoch=state['epoch'], updates=state['updates'], best_score=state['best_score'])
        save_torch(directory / 'model.pt', record)
        return record


def question_values(frame, q, columns):
    if not frame.index.is_unique or list(frame.columns) != columns or not np.isfinite(frame.to_numpy()).all():
        raise ValueError('invalid feature schema')
    values = []
    for row in q.itertuples():
        if len(row.games) != row.k or len(set(row.games)) != row.k:
            raise ValueError('incomplete/duplicate question games')
        values.append(frame.loc[list(row.games), columns].to_numpy(np.float64).mean(0))
    result = np.asarray(values, np.float32)
    if result.shape != (len(q), 60):
        raise ValueError('expected sixty features')
    return result


def train(config, stage, arm):
    from .features import assemble
    prepared = load_prepared(config)
    torch.set_num_threads(config['training']['threads'])
    root = Path(config['paths']['output'])
    columns = (S if arm == 'se3' else A) + H + C
    q = questions(config, stage)
    selection = questions(config, 'select') if stage == 'dev' else None
    needed = np.unique(np.concatenate(list(q.games) + ([] if selection is None else list(selection.games))))
    frame = assemble(config, arm, needed)
    x = question_values(frame, q, columns)
    vx = question_values(frame, selection, columns) if selection is not None else None
    dev_bundle = None
    if stage == 'final':
        dev_bundle = load_bundle(root / 'models/dev' / f'{arm}.pt', config, arm)
    identity = dict(prepared=object_hash(prepared), stage=stage, arm=arm, columns=columns,
                    questions=file_hash(root / f'questions/{stage}.parquet'),
                    selection=None if selection is None else file_hash(root / 'questions/select.parquet'),
                    development=None if dev_bundle is None else file_hash(root / 'models/dev' / f'{arm}.pt'))
    heads = {}
    for k in BUCKETS:
        mask = q.k.to_numpy() == k
        valid = selection.k.to_numpy() == k if selection is not None else None
        heads[k] = train_head(x[mask], q.rank_id.to_numpy()[mask], None if valid is None else vx[valid],
                              None if valid is None else selection.rank_id.to_numpy()[valid],
                              root / f'models/{stage}/{arm}/k{k:02d}', config['training'], config['seed'] + k,
                              identity, fixed_epochs=None if dev_bundle is None else dev_bundle['heads'][k]['best_epoch'])
    bundle = dict(format='se_h_c_b2_bundle_v1', arm=arm, stage=stage, columns=columns, heads=heads,
                  source=prepared['identity']['source'], settings=config['training'], seed=config['seed'], identity=identity)
    save_torch(root / f'models/{stage}/{arm}.pt', bundle)
    return {str(k): dict(best_epoch=h['best_epoch'], updates=h['updates']) for k, h in heads.items()}


def load_bundle(path, config=None, arm=None):
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    expected = (S if bundle['arm'] == 'se3' else A) + H + C
    if (bundle['format'] != 'se_h_c_b2_bundle_v1' or bundle['columns'] != expected
            or set(bundle['heads']) != set(BUCKETS) or (arm is not None and bundle['arm'] != arm)):
        raise ValueError('invalid B2 bundle/schema')
    if config is not None:
        prepared = load_prepared(config)
        if bundle['source'] != prepared['identity']['source'] or bundle['settings'] != config['training']:
            raise ValueError('bundle source/config mismatch')
    for head in bundle['heads'].values():
        if (np.shape(head['mean']) != (60,) or np.shape(head['scale']) != (60,)
                or not np.isfinite(head['mean']).all() or not np.isfinite(head['scale']).all()
                or (head['scale'] <= 0).any()):
            raise ValueError('invalid saved scaler')
        model = Head(); model.load_state_dict(head['model'])
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('invalid head weights')
    return bundle
