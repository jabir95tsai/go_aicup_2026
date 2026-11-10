"""Cheap go/no-go probe on cached frozen HumanSL readouts, run before any stage-one GPU training.

cache: one HumanSL forward per fixed position (15 own + 15 opponent per game, the same
       deterministic positions as the M monitor) for fit_all, fit and M games; stores the
       768-dim readout [mean_board(trunkfinal), trunkfinal[played point]] in float16.
run:   per head (linear, mlp) train four player-OOF folds on D positions, aggregate per game
       (prob and log-prob means per branch), M = average of the four fold models, then the same
       per-k 22-dim linear B2 as the main pipeline, fitted on both fit (the controlled H+C pool,
       paired comparison) and fit_all. The 72-dim stack arm only runs if b2.arms lists it.

The probe uses 30 positions per game while the full pipeline uses every ply>=30 position, so
its B2 scores are a conservative lower bound of what the same readout could reach.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.SE_H_C_B2.evaluate import report
from go_aicup.data.board_cache import RANKS
from .features import COLUMNS, LOGP_COLUMNS

P_COLUMNS = [f'p_{rank}' for rank in RANKS]

DEFAULTS = dict(workers=10, games_per_chunk=2000, games_per_batch=32, epochs=4, batch_positions=8192,
                lr=1e-3, weight_decay=1e-4, hidden=256, seed=2026)
POSITIONS = 30


def options(config):
    return {**DEFAULTS, **config.get('probe', {})}


def root(config):
    return Path(config['run_dir']) / 'probe'


def cache_rows(config):
    from .data import games, pool_game_ids
    ids = pool_game_ids(config, 'fit_all') | pool_game_ids(config, 'fit') | pool_game_ids(config, 'monitor')
    rows = games(config)
    return rows[rows.game_index.isin(ids)].sort_values('game_index').reset_index(drop=True)


def cache_identity(config, rows):
    from .data import load_extension, load_prepared
    return dict(format='humansl_residual_probe_cache_v1', prepared=load_prepared(config)['dataset_id'],
                extension=object_hash(load_extension(config)),
                human_sha256=file_hash(Path(config['model']['human_checkpoint'])),
                games=object_hash(rows.game_index.tolist()), positions='fixed 15 own + 15 opp (monitor rng, step 0)',
                readout='mean_board(trunkfinal) || trunkfinal[action]', dtype='float16')


def readout(features, actions):
    flat = features.flatten(2)
    played = flat.gather(2, actions[:, None, None].expand(-1, features.shape[1], 1)).squeeze(2)
    return torch.cat((flat.mean(2), played), dim=1)


class _Games(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, row):
        return self.dataset[(row, 0)]


def cache(config):
    """torchrun-sharded; chunks are published atomically and skipped when already complete."""
    from .dataset import SelectedDataset, collate_groups
    from .train import load_model
    opts = options(config)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if config['device'] == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    rows = cache_rows(config)
    identity = cache_identity(config, rows)
    directory = root(config) / 'cache'
    directory.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(directory / 'identity.json', identity)
    local = rows[rows.game_index.mod(world).eq(rank)].reset_index(drop=True)
    model = load_model(config, 1, device=str(device)).eval()
    started, positions = time.perf_counter(), 0
    size = opts['games_per_chunk']
    for start in range(0, len(local), size):
        target = directory / f'rank{rank:02d}_world{world:02d}_chunk{start // size:05d}.npz'
        chunk = local.iloc[start:start + size].reset_index(drop=True)
        marker = target.with_suffix('.json')
        expected = dict(identity=object_hash(identity), games=object_hash(chunk.game_index.tolist()))
        if target.exists() and marker.exists() and json.loads(marker.read_text()).get('identity') == expected \
                and json.loads(marker.read_text()).get('sha256') == file_hash(target):
            continue
        dataset = SelectedDataset(config, chunk, model.humansl.config, teacher_required=False, augment=False)
        loader = torch.utils.data.DataLoader(_Games(dataset), batch_size=opts['games_per_batch'],
                                             collate_fn=collate_groups, num_workers=opts['workers'],
                                             multiprocessing_context='spawn' if opts['workers'] else None)
        xs, gs, ss = [], [], []
        with torch.inference_mode():
            for batch in loader:
                spatial = batch['spatial'].to(device).float()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['precision'] == 'bf16'):
                    features = model._features(spatial, batch['global_inputs'].to(device).float(),
                                               batch['metadata'].to(device).float())
                values = readout(features.float(), batch['actions'].to(device).long())
                if not torch.isfinite(values).all():
                    raise ValueError('nonfinite HumanSL readout')
                xs.append(values.half().cpu().numpy()); gs.append(batch['game_index'].numpy()); ss.append(batch['sides'].numpy())
        dataset.close()
        x, g, s = np.concatenate(xs), np.concatenate(gs), np.concatenate(ss).astype(np.int8)
        if len(x) != POSITIONS * len(chunk) or set(g) != set(chunk.game_index):
            raise ValueError('probe cache chunk coverage mismatch')
        temporary = target.with_name(target.stem + '.tmp.npz')
        np.savez(temporary, x=x, game_index=g, side=s)
        temporary.replace(target)
        write_json(marker, dict(identity=expected, sha256=file_hash(target)))
        positions += len(x)
    elapsed = time.perf_counter() - started
    write_json(directory / f'metrics_rank{rank:02d}.json',
               dict(seconds=elapsed, new_positions=positions, positions_per_second=positions / elapsed if elapsed else None,
                    gpu_hours=elapsed / 3600 if device.type == 'cuda' else 0, games=len(local)))
    return dict(rank=rank, games=len(local), new_positions=positions)


def load_cache(config):
    directory = root(config) / 'cache'
    rows = cache_rows(config)
    identity = cache_identity(config, rows)
    if json.loads((directory / 'identity.json').read_text()) != identity:
        raise ValueError('probe cache identity changed; rebuild in a new run directory')
    parts = []
    for path in sorted(p for p in directory.glob('rank*_chunk*.npz') if '.tmp' not in p.name):
        marker = json.loads(path.with_suffix('.json').read_text())
        if marker['identity']['identity'] != object_hash(identity) or marker['sha256'] != file_hash(path):
            raise ValueError(f'probe cache chunk changed: {path.name}')
        with np.load(path) as data:
            parts.append({k: data[k] for k in ('x', 'game_index', 'side')})
    if not parts:
        raise ValueError('probe cache is empty; run the cache command first')
    result = {k: np.concatenate([p[k] for p in parts]) for k in ('x', 'game_index', 'side')}
    if set(result['game_index']) != set(rows.game_index) or len(result['x']) != POSITIONS * len(rows):
        raise ValueError('probe cache is incomplete')
    return rows, result


class Head(nn.Module):
    def __init__(self, kind, hidden):
        super().__init__()
        def make():
            return (nn.Linear(768, 10) if kind == 'linear'
                    else nn.Sequential(nn.Linear(768, hidden), nn.ReLU(), nn.Linear(hidden, 10)))
        self.heads = nn.ModuleList([make(), make()])
    def forward(self, x, side):
        return torch.where(side[:, None] == 0, self.heads[0](x), self.heads[1](x))


def train_head(kind, x, side, labels, opts, device):
    """Standardize on training positions only, then AdamW CE; returns a predict(x, side) closure."""
    generator = torch.Generator().manual_seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    total, square = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for start in range(0, len(x), 262144):  # chunked: a float64 copy of all positions would not fit
        part = x[start:start + 262144].astype(np.float64)
        total += part.sum(0); square += (part * part).sum(0)
    mean_np = total / len(x)
    std_np = np.sqrt(np.maximum(square / len(x) - mean_np ** 2, 0))
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device).clamp_min(1e-4)
    model = Head(kind, opts['hidden']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts['lr'], weight_decay=opts['weight_decay'])
    xt = torch.as_tensor(x); st = torch.as_tensor(side).long(); yt = torch.as_tensor(labels).long()
    steps = opts['epochs'] * ((len(x) + opts['batch_positions'] - 1) // opts['batch_positions'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .5 * (1 + np.cos(np.pi * s / steps)))
    for _ in range(opts['epochs']):
        order = torch.randperm(len(x), generator=generator)
        for start in range(0, len(x), opts['batch_positions']):
            index = order[start:start + opts['batch_positions']]
            batch = (xt[index].to(device).float() - mean) / std
            loss = nn.functional.cross_entropy(model(batch, st[index].to(device)), yt[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
    model.eval()
    def predict(xp, sp):
        out = []
        with torch.no_grad():
            for start in range(0, len(xp), 65536):
                batch = (torch.as_tensor(xp[start:start + 65536]).to(device).float() - mean) / std
                out.append(model(batch, torch.as_tensor(sp[start:start + 65536]).to(device).long()).log_softmax(-1).cpu().numpy())
        return np.concatenate(out)
    return predict


def game_features(game_index, side, logp):
    """Per game and branch: mean probability and mean log-probability (counts are 15/15)."""
    frame = pd.DataFrame(np.c_[np.exp(logp), logp], columns=[f'p{i}' for i in range(10)] + [f'l{i}' for i in range(10)])
    frame['game_index'] = game_index; frame['side'] = side
    means = frame.groupby(['game_index', 'side']).mean().unstack('side')
    result = pd.DataFrame(index=means.index)
    for s, name in ((0, 'own'), (1, 'opp')):
        for i in range(10):
            result[f'{name}_{i}'] = means[(f'p{i}', s)]
            result[f'{name}_lp_{i}'] = means[(f'l{i}', s)]
    result['own_count'] = 15; result['opp_count'] = 15
    return result.reset_index()[['game_index', 'own_count', 'opp_count', *COLUMNS, *LOGP_COLUMNS]]


def run(config):
    from .b2 import average_frames, fit_questions, probabilities
    from .data import questions
    from .evaluate import compare
    opts = options(config)
    device = torch.device('cuda') if torch.cuda.is_available() and config['device'] == 'cuda' else torch.device('cpu')
    started = time.perf_counter()
    rows, data = load_cache(config)
    info = rows.set_index('game_index')
    role = info.loc[data['game_index'], 'role'].to_numpy()
    fold_of = info.loc[data['game_index'], 'oof_fold'].to_numpy()
    labels = info.loc[data['game_index'], 'rank_id'].to_numpy()
    monitor = role == 'M'
    if (role[~monitor] != 'D').any():
        raise ValueError('probe cache may only contain D and M games')
    out = root(config)
    result = dict(options=opts, positions=int(len(data['x'])), games=int(len(rows)), arms={})
    q_monitor = questions(config, 'monitor')
    reference = None
    from .baseline import root as baseline_root
    reference_path = baseline_root(config) / 'evaluation' / 'monitor.parquet'
    if reference_path.exists():
        from .evaluate import read_predictions
        reference = read_predictions(reference_path)
    for kind in ('linear', 'mlp'):
        oof, mon = [], []
        for fold in (1, 2, 3, 4):
            train = (~monitor) & (fold_of != fold)
            held = (~monitor) & (fold_of == fold)
            predict = train_head(kind, data['x'][train], data['side'][train], labels[train], opts, device)
            oof.append(game_features(data['game_index'][held], data['side'][held],
                                     predict(data['x'][held], data['side'][held])))
            mon.append(game_features(data['game_index'][monitor], data['side'][monitor],
                                     predict(data['x'][monitor], data['side'][monitor])))
        oof_frame = pd.concat(oof, ignore_index=True)
        monitor_frame = average_frames(mon)
        oof_frame.to_parquet(out / f'{kind}_oof.parquet', index=False)
        monitor_frame.to_parquet(out / f'{kind}_monitor.parquet', index=False)
        mq = q_monitor
        for pool in ('fit_all', 'fit'):
            fq = questions(config, pool)
            for aggregation in ('prob', 'logp'):
                model = fit_questions(rows, fq, oof_frame[oof_frame.game_index.isin(set(np.concatenate(fq.games.to_numpy())))],
                                      aggregation)
                frame = monitor_frame[monitor_frame.game_index.isin(set(np.concatenate(mq.games.to_numpy())))]
                proba = probabilities(model, rows, mq, frame, aggregation)
                name = f'{kind}_new_{aggregation}_{pool}'
                entry = report(mq, proba)
                if reference is not None:
                    entry['gate_vs_controlled_hc'] = compare(mq, reference[P_COLUMNS].to_numpy(), proba)
                result['arms'][name] = entry
        if 'stack' in config.get('b2', {}).get('arms', ['new']):
            stack = _stack_arm(config, rows, oof_frame, monitor_frame, q_monitor, reference)
            result['arms'].update({f'{kind}_{k}': v for k, v in stack.items()})
    result['seconds'] = time.perf_counter() - started
    result['device'] = str(device)
    write_json(out / 'result.json', result)
    return {name: arm['equal_k'] for name, arm in result['arms'].items()}


def _stack_arm(config, rows, oof_frame, monitor_frame, q_monitor, reference):
    """H+C+probe stack on the controlled baseline fit pool, if those features already exist."""
    from . import baseline
    from .b2 import fit_questions, probabilities
    from .data import questions
    from .evaluate import compare
    pool = baseline.fit_pool(config)
    try:
        hc_fit = baseline.read_pool(config, pool)[2]
        hc_monitor = baseline.read_pool(config, 'monitor')[2]
    except (FileNotFoundError, ValueError) as error:
        return {'stack_skipped': dict(reason=str(error))}
    fq = questions(config, pool)
    results = {}
    for aggregation in ('prob', 'logp'):
        fit_frame = oof_frame[oof_frame.game_index.isin(hc_fit.game_index)]
        model = fit_questions(rows, fq, fit_frame, aggregation, hc_fit)
        frame = monitor_frame[monitor_frame.game_index.isin(hc_monitor.game_index)]
        proba = probabilities(model, rows, q_monitor, frame, aggregation, hc_monitor)
        entry = report(q_monitor, proba)
        if reference is not None:
            entry['gate_vs_controlled_hc'] = compare(q_monitor, reference[P_COLUMNS].to_numpy(), proba)
        results[f'stack_{aggregation}_{pool}'] = entry
    return results


def main():
    from .config import read_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--command', choices=('cache', 'run'), required=True)
    a = p.parse_args()
    config = read_config(a.config)
    print(json.dumps(cache(config) if a.command == 'cache' else run(config), indent=2))


if __name__ == '__main__':
    main()
