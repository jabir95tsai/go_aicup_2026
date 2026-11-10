"""Bounded, all-position probability and log-probability extraction; no persisted HumanSL activations."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.SE_H_C_B2.common import load_table, publish_table

BUCKETS = (5, 10, 15, 20)
COLUMNS = [f'{side}_{rank}' for side in ('own', 'opp') for rank in range(10)]
# Per-branch mean of log_softmax: log aggregation multiplies per-position evidence (A-025: +0.064 on U).
LOGP_COLUMNS = [f'{side}_lp_{rank}' for side in ('own', 'opp') for rank in range(10)]
FIT_POOLS = ('fit', 'fit_all')
INFERENCE_POOLS = ('monitor', 'U', 'mu_all')


def feature_columns(aggregation='prob'):
    if aggregation not in ('prob', 'logp'):
        raise ValueError('aggregation must be prob or logp')
    return COLUMNS if aggregation == 'prob' else LOGP_COLUMNS


def validate_features(frame):
    if not frame.game_index.is_unique or frame.empty:
        raise ValueError('empty or duplicate game features')
    x = frame[COLUMNS].to_numpy(np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('nonfinite or negative game probabilities')
    if not np.allclose(x.reshape(-1, 2, 10).sum(2), 1, atol=1e-5):
        raise ValueError('each branch must sum to one')
    if not frame[['own_count', 'opp_count']].gt(0).all().all():
        raise ValueError('both branches require effective positions')
    if set(LOGP_COLUMNS) & set(frame.columns):
        logp = frame[LOGP_COLUMNS].to_numpy(np.float64)
        if not np.isfinite(logp).all() or (logp > 1e-6).any():
            raise ValueError('mean log-probabilities must be finite and nonpositive')
    return frame


def mean_probabilities(logits, sides):
    """Softmax first, arithmetic position mean within each branch."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    sides = torch.as_tensor(sides, device=logits.device)
    if logits.ndim != 2 or logits.shape[1] != 10 or sides.shape != (len(logits),):
        raise ValueError('expected logits [N,10] and sides [N]')
    if not torch.isfinite(logits).all() or not ((sides == 0) | (sides == 1)).all():
        raise ValueError('invalid logits or side')
    p = logits.softmax(-1)
    if any(not (sides == s).any() for s in (0, 1)):
        raise ValueError('both own and opponent positions are required')
    return torch.cat([p[sides == s].mean(0) for s in (0, 1)]).cpu().numpy()


def mean_log_probabilities(logits, sides):
    """log_softmax first, arithmetic position mean within each branch."""
    logits = torch.as_tensor(logits, dtype=torch.float32)
    sides = torch.as_tensor(sides, device=logits.device)
    if logits.ndim != 2 or logits.shape[1] != 10 or sides.shape != (len(logits),):
        raise ValueError('expected logits [N,10] and sides [N]')
    if any(not (sides == s).any() for s in (0, 1)):
        raise ValueError('both own and opponent positions are required')
    logp = logits.log_softmax(-1)
    return torch.cat([logp[sides == s].mean(0) for s in (0, 1)]).cpu().numpy()


def validate_oof(rows, identity, *, phase=None, fold=None):
    """Reject in-sample features and stage-two ancestry from another fold."""
    required = {'phase', 'fold', 'stage', 'training_players'}
    if not required.issubset(identity):
        raise ValueError('checkpoint lacks OOF training identity')
    if phase is not None and identity['phase'] != phase:
        raise ValueError('checkpoint phase mismatch')
    if fold is not None and str(identity['fold']) != str(fold):
        raise ValueError('checkpoint fold mismatch')
    if set(rows.player_id.astype(str)) & set(map(str, identity['training_players'])):
        raise ValueError('OOF checkpoint has trained on an extraction player')
    if int(identity['stage']) == 2:
        parent = identity.get('stage1_identity')
        if parent is None or any(str(parent[k]) != str(identity[k]) for k in ('phase', 'fold')):
            raise ValueError('stage two requires same-fold stage-one ancestry')
        if set(map(str, parent['training_players'])) & set(rows.player_id.astype(str)):
            raise ValueError('stage-one ancestor has seen an OOF player')


def p2_values(games, frame, aggregation='prob'):
    validate_features(frame)
    columns = feature_columns(aggregation)
    if not set(columns) <= set(frame.columns):
        raise ValueError(f'features lack {aggregation} columns; re-extract with the v2 feature format')
    rows = games.set_index('game_index').loc[frame.game_index]
    if not rows.hd.isin([0, 1]).all() or not rows.player_color.isin([0, 1]).all():
        raise ValueError('P2 requires explicit HD0/1 and player color')
    hd1 = rows.hd.to_numpy() == 1
    color = rows.player_color.to_numpy()
    return np.column_stack([frame[columns].to_numpy(), hd1 & (color == 0), hd1 & (color == 1)])


def infer_games(model, dataset, rows, *, device='cpu', batch_size=300, fixed=False, precision='fp32'):
    """Single game/bounded batch on GPU; only twenty numbers per game survive."""
    model.eval()
    output = []
    with torch.inference_mode():
        for row in range(len(rows)):
            sums = np.zeros((2, 10), np.float64)
            log_sums = np.zeros((2, 10), np.float64)
            counts = np.zeros(2, np.int64)
            batches = [dataset[(row, 0)]] if fixed else dataset.iter_positions(row, batch_size=batch_size)
            for batch in batches:
                tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()
                           if k in ('spatial', 'global_inputs', 'metadata', 'actions', 'sides')}
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=precision == 'bf16'):
                    result = model(tensors['spatial'].float(), tensors['global_inputs'].float(),
                                   tensors['metadata'].float(), tensors['actions'].long(), tensors['sides'].long())
                logits = result['rank_logits'] if isinstance(result, dict) else result
                sides = tensors['sides'].cpu().numpy()
                logp = logits.float().log_softmax(-1).cpu().numpy()
                p = np.exp(logp)
                if not np.isfinite(logp).all():
                    raise ValueError('nonfinite inference probabilities')
                for side in (0, 1):
                    sums[side] += p[sides == side].sum(0, dtype=np.float64)
                    log_sums[side] += logp[sides == side].sum(0, dtype=np.float64)
                    counts[side] += np.count_nonzero(sides == side)
            if (counts == 0).any():
                raise ValueError(f'empty branch: game {rows.game_index.iloc[row]}')
            values = (sums / counts[:, None]).reshape(-1)
            log_values = (log_sums / counts[:, None]).reshape(-1)
            output.append(dict(game_index=int(rows.game_index.iloc[row]), own_count=int(counts[0]),
                               opp_count=int(counts[1]), **dict(zip(COLUMNS, values)),
                               **dict(zip(LOGP_COLUMNS, log_values))))
    return validate_features(pd.DataFrame(output))


def feature_path(config, phase, stage, candidate, pool, fold='all'):
    return Path(config['run_dir']) / 'features' / phase / f'stage{stage}' / candidate / pool / f'fold_{fold}.parquet'


def extraction_rows(config, pool, phase='dev', fold='all'):
    """Fit pools are OOF by fold; M/U/M+U pools may use any development fold model."""
    from .data import games, pool_game_ids
    if phase != 'dev':
        raise ValueError('only development-phase upstream models exist; release reuses them')
    rows = games(config)
    rows = rows[rows.game_index.isin(pool_game_ids(config, pool))].copy()
    if pool in FIT_POOLS:
        if str(fold) == 'all':
            raise ValueError('fitting features require a held-out fold, never the all-player model')
        rows = rows[rows.oof_fold.eq(int(fold))]
    elif pool not in INFERENCE_POOLS:
        raise ValueError(f'unknown extraction pool: {pool}')
    elif str(fold) not in ('all', '1', '2', '3', '4'):
        raise ValueError('inference pools use fold 1..4 (averaged later) or the optional all-D model')
    if rows.empty:
        raise ValueError('empty extraction pool')
    return rows.reset_index(drop=True)


def extract(config, checkpoint, *, phase='dev', pool='fit', candidate='best', fold='all', output=None):
    from .data import load_prepared
    from .dataset import SelectedDataset
    from .train import load_candidate
    started, cpu_started = time.perf_counter(), time.process_time()
    if pool in ('U', 'mu_all'):
        from .evaluate import load_lock
        load_lock(config)
    record = load_prepared(config)
    rows = extraction_rows(config, pool, phase, fold)
    device = config.get('device', 'cuda')
    if device == 'cuda':
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    model, payload = load_candidate(Path(checkpoint), config, device=device)
    upstream = payload['identity']
    if payload['dataset_id'] != record['dataset_id'] or payload['config_sha256'] != object_hash(config):
        raise ValueError('candidate was trained on a different dataset or configuration')
    if Path(checkpoint).name != f'{candidate}.pt':
        raise ValueError('checkpoint filename must match the selected candidate')
    validate_oof(rows, upstream, phase=phase, fold=fold)
    identity = dict(format='humansl_residual_features_v2', prepared=object_hash(record),
                    checkpoint=file_hash(Path(checkpoint)), upstream=upstream,
                    games=object_hash(rows.game_index.tolist()), pool=pool, candidate=candidate,
                    aggregation='softmax-mean+log_softmax-mean/own-opp/all-ply>=30', hd_komi='HD0:6.5/HD1:0.5')
    target = Path(output) if output else feature_path(config, phase, upstream['stage'], candidate, pool, fold)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    rows = rows[rows.game_index.mod(world).eq(rank)].reset_index(drop=True)
    target = target.with_name(f'{target.stem}.rank{rank}.parquet') if world > 1 else target
    def record_metrics(frame, reused):
        elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu_started
        positions = int(frame[['own_count', 'opp_count']].to_numpy().sum())
        allocation = os.environ.get('SLURM_GPUS_ON_NODE', str(world if device == 'cuda' else 0))
        allocated_gpus = int(allocation) if allocation.isdigit() else None
        write_json(target.with_suffix('.reuse.metrics.json' if reused else '.metrics.json'),
                   dict(reused=reused, elapsed_seconds=elapsed, cpu_seconds=cpu, games=len(frame),
                        positions=positions, computed_positions=0 if reused else positions,
                        positions_per_second=None if reused else positions / elapsed,
                        cpu_hours=cpu / 3600, allocated_gpus=allocated_gpus,
                        gpu_hours=None if allocated_gpus is None else elapsed * allocated_gpus / world / 3600,
                        cuda_compute_gpu_hours=elapsed / 3600 if device == 'cuda' else 0,
                        device=device, rank=rank, world_size=world,
                        scope='validation, model load, encoding, inference and atomic publication'))
    if rows.empty:
        frame = pd.DataFrame(columns=['game_index', 'own_count', 'opp_count', *COLUMNS, *LOGP_COLUMNS])
        record_metrics(frame, False)
        return frame
    if target.exists():
        frame = validate_features(load_table(target, identity))
        record_metrics(frame, True)
        return frame
    dataset = SelectedDataset(config, rows, model.humansl.config, teacher_required=False, augment=False)
    frame = infer_games(model, dataset, rows, device=device,
                        batch_size=config.get('evaluation', {}).get('batch_size', 300), precision=config.get('precision', 'fp32'))
    publish_table(target, frame, identity)
    record_metrics(frame, False)
    return frame


def read_features(path):
    path = Path(path)
    record = json.loads(path.with_suffix('.json').read_text())
    return validate_features(load_table(path, record['identity'])), record['identity']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--phase', choices=('dev',), default='dev')
    p.add_argument('--pool', choices=FIT_POOLS + INFERENCE_POOLS, required=True)
    p.add_argument('--candidate', choices=('best', 'final'), default='best')
    p.add_argument('--fold', default='all'); p.add_argument('--output')
    args = p.parse_args()
    from .config import read_config
    config = read_config(args.config)
    print(extract(config, args.checkpoint, phase=args.phase, pool=args.pool,
                  candidate=args.candidate, fold=args.fold, output=args.output).shape)


if __name__ == '__main__':
    main()
