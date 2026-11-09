"""Controlled H+C+P2+B2: whole-game likelihoods and player-OOF C vectors."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.features.humansl import PROFILES
from go_aicup.features.humansl_rankvec import game_logp, load_frozen, policy_logp
from go_aicup.features.humansl_torch import CKPT, _katago, profile_metadata
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.humansl import GameLR, RankModel, question_mean
from go_aicup.tasks.strength.reproduce_paper_3.train import atomic_payload
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table, load_table, publish_table, writer_lock
from go_aicup.tasks.strength.SE_H_C_B2.data import check_questions
from go_aicup.tasks.strength.SE_H_C_B2.evaluate import report
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from .b2 import load_model, save_model
from .data import games, load_prepared, questions
from .dataset import SelectedDataset
from .features import BUCKETS, extraction_rows

H_COLUMNS = [f'h_{side}_{profile}' for side in ('own', 'opp') for profile in PROFILES]
C_COLUMNS = [f'c_{side}_{rank}' for side in ('own', 'opp') for rank in range(10)]
COLUMNS = H_COLUMNS + C_COLUMNS


def root(config):
    return Path(config['run_dir']) / 'baseline' / 'dev'


def vector_path(config, fold):
    return root(config) / f'fold_{fold}' / 'vectors.pt'


def feature_path(config, pool, fold):
    return root(config) / 'features' / pool / f'fold_{fold}.parquet'


def training_rows(config, fold):
    if str(fold) not in ('all', '1', '2', '3', '4'):
        raise ValueError('C fold must be all or 1..4')
    rows = games(config)
    rows = rows[rows.role.eq('D')]
    if str(fold) != 'all':
        rows = rows[rows.oof_fold.ne(int(fold))]
    if rows.empty or set(rows.rank_id) != set(range(10)):
        raise ValueError('C training requires D players from all ten classes')
    return rows.reset_index(drop=True)


def vector_identity(config, rows, fold, prepared):
    return dict(format='controlled_hc_vectors_v1', phase='dev', fold=str(fold),
                prepared=object_hash(prepared), config=object_hash(config),
                human_checkpoint=file_hash(Path(config['model']['human_checkpoint'])),
                training_players=sorted(rows.player_id.astype(str).unique()),
                training_games=object_hash(rows.game_index.tolist()),
                input_recipe='whole-game/japanese/HD0:6.5/HD1:0.5/real-history',
                sampling='uniform-positions-without-replacement-per-update',
                code=file_hash(Path(__file__)))


def validate_ancestry(rows, identity, fold):
    if (identity.get('format') != 'controlled_hc_vectors_v1' or identity.get('phase') != 'dev'
            or str(identity.get('fold')) != str(fold)):
        raise ValueError('C vectors lack this controlled fold identity')
    if set(rows.player_id.astype(str)) & set(identity['training_players']):
        raise ValueError('controlled C extraction player was seen during vector training')


def load_runtime(config):
    # The existing loader owns the metadata-identity override used by C.
    if Path(config['model']['human_checkpoint']).resolve() != Path(CKPT).resolve():
        raise ValueError('C loader requires the declared ordinary HumanSL checkpoint')
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if config['device'] == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model, init = load_frozen(device)
    return model, init, device


def sample_positions(dataset, rows, count, rng, step):
    """Uniform full-game positions; encode only selected plies, with no input cache."""
    bounds = np.cumsum(rows.length.to_numpy(np.int64))
    if type(count) is not int or count < 1 or count > int(bounds[-1]):
        raise ValueError('C batch must be positive and fit the available full-game positions')
    selected = rng.choice(int(bounds[-1]), count, replace=False)
    row_ids = np.searchsorted(bounds, selected, side='right')
    starts = np.r_[0, bounds[:-1]]
    parts = [dataset[(int(row), step, selected[row_ids == row] - starts[row])]
             for row in np.unique(row_ids)]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def train_c(config, fold='all', *, stop_after=None):
    """Resume Adam/cosine C learning; M/U/R labels never enter this objective."""
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('C training uses one process per fold; extract supports sharded torchrun')
    started = time.perf_counter()
    options = config['baseline']
    budget, batch, lr = options['updates'], options['batch_positions'], options['lr']
    if (type(budget) is not int or budget < 1 or type(batch) is not int or batch < 1
            or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr <= 0):
        raise ValueError('baseline updates/batch_positions/lr require explicit positive budgets')
    end = budget if stop_after is None else stop_after
    if type(end) is not int or not 1 <= end <= budget:
        raise ValueError('stop_after must be within the C update budget')
    prepared = load_prepared(config)
    rows = training_rows(config, fold)
    identity = vector_identity(config, rows, fold, prepared)
    directory = vector_path(config, fold).parent
    with writer_lock(directory):
        model, init, device = load_runtime(config)
        dataset = SelectedDataset(config, rows, model.config, teacher_required=False, augment=False)
        vectors = torch.nn.Parameter(init.repeat(10, 2, 1))
        optimizer = torch.optim.Adam([vectors], lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .5 * (1 + math.cos(math.pi * s / budget)))
        rng = np.random.default_rng(config['split_seed'])
        latest, final = directory / 'latest.pt', vector_path(config, fold)
        step = 0
        source = final if final.exists() else latest
        try:
            if source.exists():
                payload = torch.load(resolve_checkpoint(source), map_location=device, weights_only=False)
                if payload['identity'] != identity:
                    raise ValueError('controlled C source changed; use a new run directory')
                with torch.no_grad():
                    vectors.copy_(payload['model']['vectors'])
                optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
                step = payload['state']['optimizer_steps']; rng.bit_generator.state = payload['state']['numpy_rng']
                if step == budget and final.exists() and end == budget:
                    return dict(fold=str(fold), updates=step, complete=True, path=str(final))
            if step > end:
                raise ValueError('controlled C checkpoint exceeds requested stop')
            start_step = step
            while step < end:
                sample = sample_positions(dataset, rows, batch, rng, step)
                x = torch.as_tensor(sample['spatial'], device=device).float()
                global_inputs = torch.as_tensor(sample['global_inputs'], device=device).float()
                target = torch.as_tensor(sample['actions'], device=device).long()
                labels = torch.as_tensor(sample['labels'], device=device).long()
                sides = torch.as_tensor(sample['sides'], device=device).long()
                nll = -policy_logp(model, x, global_inputs, vectors[labels, sides], target).mean()
                if not torch.isfinite(nll):
                    raise ValueError('nonfinite controlled C likelihood')
                optimizer.zero_grad(set_to_none=True); nll.backward()
                if vectors.grad is None or not torch.isfinite(vectors.grad).all():
                    raise FloatingPointError('nonfinite controlled C vector gradient')
                optimizer.step(); scheduler.step()
                step += 1
                if step % config['training']['checkpoint_every'] == 0 or step == end:
                    payload = dict(config=config, config_sha256=object_hash(config), dataset_id=prepared['dataset_id'],
                                   code_sha256=identity['code'], identity=identity,
                                   model={'vectors': vectors.detach().cpu()}, optimizer=optimizer.state_dict(),
                                   scheduler=scheduler.state_dict(), state=dict(optimizer_steps=step, numpy_rng=rng.bit_generator.state))
                    atomic_payload(latest, payload)
                    print(json.dumps(dict(fold=str(fold), step=step, nll=float(nll.detach()))), flush=True)
            if step == budget:
                atomic_payload(final, payload if 'payload' in locals() else torch.load(resolve_checkpoint(source), map_location='cpu', weights_only=False))
            elapsed = time.perf_counter() - started
            write_json(directory / 'metrics.json', dict(seconds=elapsed, updates=step - start_step,
                       positions=(step - start_step) * batch, positions_per_second=(step - start_step) * batch / elapsed,
                       gpu_hours=elapsed / 3600 if device.type == 'cuda' else 0))
            return dict(fold=str(fold), updates=step, complete=step == budget, path=str(final if step == budget else latest))
        finally:
            dataset.close()


def human_vectors(model, device):
    """Encode original fixed profile rows; bypass C's instance identity override."""
    board, _, _, _ = _katago()
    rows = np.asarray([[profile_metadata(profile).get_metadata_row(nextPlayer=pla, boardArea=361)
                        for profile in PROFILES] for pla in (board.Board.BLACK, board.Board.WHITE)])
    with torch.no_grad():
        values = type(model.metadata_encoder).forward(model.metadata_encoder,
                torch.as_tensor(rows.reshape(-1, rows.shape[-1]), device=device).float(), None)
    return values.reshape(2, len(PROFILES), 384)


def infer_likelihoods(model, vectors, profiles, dataset, rows, batch_size, device):
    output = []
    with torch.inference_mode():
        for row in range(len(rows)):
            h_sums, c_sums = np.zeros((2, len(PROFILES)), np.float64), np.zeros((2, 10), np.float64)
            counts = np.zeros(2, np.int64)
            for batch in dataset.iter_positions(row, batch_size=batch_size, min_ply=0):
                x = torch.as_tensor(batch['spatial'], device=device).float()
                g = torch.as_tensor(batch['global_inputs'], device=device).float()
                actions = torch.as_tensor(batch['actions'], device=device).long()
                sides = np.asarray(batch['sides'])
                colors = torch.as_tensor(sides ^ int(rows.player_color.iloc[row]), device=device).long()
                h = torch.stack([policy_logp(model, x, g, profiles[colors, p], actions)
                                 for p in range(len(PROFILES))], 1).cpu().numpy()
                c = game_logp(model, vectors, batch['spatial'], batch['global_inputs'], batch['actions'],
                              sides == 0, batch=batch_size * 10).cpu().numpy()
                if not np.isfinite(h).all() or not np.isfinite(c).all():
                    raise ValueError('nonfinite whole-game baseline likelihood')
                for side in (0, 1):
                    mask = sides == side
                    h_sums[side] += h[mask].sum(0, dtype=np.float64)
                    c_sums[side] += c[mask].sum(0, dtype=np.float64)
                    counts[side] += int(mask.sum())
            if (counts == 0).any():
                raise ValueError('baseline requires both branches for every full game')
            values = np.r_[(h_sums / counts[:, None]).ravel(), (c_sums / counts[:, None]).ravel()]
            output.append(dict(game_index=int(rows.game_index.iloc[row]), own_count=int(counts[0]),
                               opp_count=int(counts[1]), **dict(zip(COLUMNS, values))))
    return checked_features(pd.DataFrame(output))


def checked_features(frame):
    if frame.empty or not frame.game_index.is_unique or not np.isfinite(frame[COLUMNS].to_numpy()).all():
        raise ValueError('baseline features must be complete, unique finite likelihoods')
    if not frame[['own_count', 'opp_count']].gt(0).all().all():
        raise ValueError('baseline whole-game side is empty')
    return frame


def extract(config, pool, fold='all'):
    started = time.perf_counter()
    if pool == 'U':
        from .evaluate import load_lock
        load_lock(config)
    prepared = load_prepared(config)
    rows = extraction_rows(config, pool, 'dev', fold)
    path = vector_path(config, fold)
    payload = torch.load(resolve_checkpoint(path), map_location='cpu', weights_only=False)
    current = vector_identity(config, training_rows(config, fold), fold, prepared)
    if payload['identity'] != current or payload['state']['optimizer_steps'] != config['baseline']['updates']:
        raise ValueError('baseline vectors are incomplete or belong to different data/config')
    validate_ancestry(rows, current, fold)
    identity = dict(format='controlled_hc_features_v1', prepared=object_hash(prepared),
                    vectors=file_hash(path), upstream=current, pool=pool, whole_game=True,
                    games=object_hash(rows.game_index.tolist()), columns=COLUMNS)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    target = feature_path(config, pool, fold)
    rows = rows[rows.game_index.mod(world).eq(rank)].reset_index(drop=True)
    if world > 1:
        target = target.with_name(f'{target.stem}.rank{rank}.parquet')
    if target.exists():
        return checked_features(load_table(target, identity))
    model, _, device = load_runtime(config)
    vectors = payload['model']['vectors'].to(device)
    if vectors.shape != (10, 2, 384) or not torch.isfinite(vectors).all():
        raise ValueError('invalid controlled C vectors')
    dataset = SelectedDataset(config, rows, model.config, teacher_required=False, augment=False)
    try:
        frame = infer_likelihoods(model, vectors, human_vectors(model, device), dataset, rows,
                                 config['baseline']['batch_positions'], device)
        publish_table(target, frame, identity)
        elapsed = time.perf_counter() - started
        positions = int(frame[['own_count', 'opp_count']].to_numpy().sum())
        write_json(target.with_suffix('.metrics.json'), dict(seconds=elapsed, games=len(frame), positions=positions,
                   positions_per_second=positions / elapsed, gpu_hours=elapsed / 3600 if device.type == 'cuda' else 0))
        return frame
    finally:
        dataset.close()


def read_pool(config, pool):
    canonical = games(config)
    q = questions(config, 'fit' if pool == 'fit' else pool)
    expected = set(np.concatenate(q.games.to_numpy()).tolist())
    prepared_record = load_prepared(config)
    prepared = object_hash(prepared_record)
    frames, sources = [], []
    for fold in range(1, 5) if pool == 'fit' else ('all',):
        target = feature_path(config, pool, fold)
        paths = [target] if target.exists() else sorted(target.parent.glob(f'{target.stem}.rank*.parquet'))
        if not paths:
            raise ValueError(f'missing controlled baseline features: {target}')
        current = vector_identity(config, training_rows(config, fold), fold, prepared_record)
        for path in paths:
            marker = json.loads(path.with_suffix('.json').read_text())
            identity = marker['identity']; frame = checked_features(load_table(path, identity))
            rows = canonical.set_index('game_index').loc[frame.game_index].reset_index()
            if (identity.get('format') != 'controlled_hc_features_v1' or identity['prepared'] != prepared
                    or identity['pool'] != pool or identity.get('whole_game') is not True
                    or identity.get('columns') != COLUMNS or identity['upstream'] != current
                    or identity['vectors'] != file_hash(vector_path(config, fold))):
                raise ValueError('controlled baseline feature identity differs')
            validate_ancestry(rows, identity['upstream'], fold)
            if pool == 'fit' and not rows.oof_fold.eq(fold).all():
                raise ValueError('baseline OOF features assigned to wrong fold')
            frames.append(frame); sources.append(dict(path=str(path), sha256=file_hash(path), identity=identity))
    frame = checked_features(pd.concat(frames, ignore_index=True))
    if set(frame.game_index) != expected:
        raise ValueError('controlled baseline question features missing or unexpected')
    return canonical, q, frame, sources


def p2_values(canonical, frame):
    rows = canonical.set_index('game_index').loc[frame.game_index]
    if not rows.hd.isin([0, 1]).all() or not rows.player_color.isin([0, 1]).all():
        raise ValueError('controlled baseline supports explicit HD0/1 only')
    return np.c_[frame[COLUMNS].to_numpy(), rows.hd.eq(1) & rows.player_color.eq(0),
                 rows.hd.eq(1) & rows.player_color.eq(1)]


def fit(config):
    started = time.perf_counter()
    canonical, q, frame, sources = read_pool(config, 'fit')
    if not canonical.set_index('game_index').loc[frame.game_index, 'role'].eq('D').all():
        raise ValueError('controlled baseline B2 fits only D players')
    check_questions(q, canonical)
    x = question_mean(p2_values(canonical, frame), frame.game_index, q)
    identity = dict(format='controlled_hc_b2_v1', phase='dev', config=object_hash(config), sources=sources,
                    prepared=object_hash(load_prepared(config)), questions=object_hash(q.to_json(orient='records')),
                    training_players=sorted(q.player_id.astype(str).unique()), dimensions=52)
    target = root(config) / 'model.pkl'
    if target.exists():
        _, previous = load_model(target)
        if previous != identity:
            raise ValueError('controlled baseline B2 identity changed')
        return target
    model = RankModel('B2'); model.per_k = {}
    for k in BUCKETS:
        mask = q.k.eq(k).to_numpy(); labels = q.loc[mask, 'rank_id'].to_numpy()
        if set(labels) != set(range(10)):
            raise ValueError('controlled B2 requires all ten classes per k')
        fitted = GameLR(x[mask], labels)
        model.per_k[k] = fitted.scaler, fitted.model
    save_model(target, model, identity)
    elapsed = time.perf_counter() - started
    write_json(target.with_suffix('.metrics.json'), dict(seconds=elapsed, cpu_wall_hours=elapsed / 3600,
               questions=len(q), dimensions=52))
    return target


def evaluate(config, pool='monitor'):
    if pool not in ('monitor', 'U'):
        raise ValueError('baseline evaluation uses M or locked U only')
    target = root(config) / 'model.pkl'
    model, identity = load_model(target)
    if identity['config'] != object_hash(config) or identity['prepared'] != object_hash(load_prepared(config)):
        raise ValueError('controlled baseline B2 belongs to different data/config')
    if pool == 'U':
        from .evaluate import load_lock
        lock = load_lock(config)
        if not any(v['model'] == str(target) and v['sha256'] == file_hash(target) for v in lock['arms'].values()):
            raise ValueError('baseline U arm was not predeclared in lock')
    canonical, q, frame, sources = read_pool(config, pool)
    if set(q.player_id.astype(str)) & set(identity['training_players']):
        raise ValueError('controlled B2 fitting/evaluation players overlap')
    probability = model.probabilities(p2_values(canonical, frame), frame.game_index.to_numpy(), q)
    result = dict(report(q, probability), pool=pool, sources=sources, model=file_hash(target),
                  interpretation='development selection' if pool == 'monitor' else 'locked one-time U evaluation')
    output = q.drop(columns='games').assign(rank_prediction=score_rank(probability))
    for i, rank in enumerate(RANKS):
        output[f'p_{rank}'] = probability[:, i]
    path = root(config) / 'evaluation' / f'{pool}.parquet'
    if path.exists():
        if json.loads(path.with_suffix('.json').read_text()) != dict(report=result, sha256=file_hash(path)):
            raise ValueError('controlled baseline evaluation identity changed')
        return result
    atomic_table(path, output); write_json(path.with_suffix('.json'), dict(report=result, sha256=file_hash(path)))
    return result


def main():
    from .config import read_config
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--command', choices=('train-c', 'extract', 'fit', 'evaluate'), required=True)
    p.add_argument('--fold', default='all'); p.add_argument('--pool', choices=('fit', 'monitor', 'U'), default='monitor')
    p.add_argument('--stop-after', type=int)
    a = p.parse_args(); config = read_config(a.config)
    if a.command == 'train-c':
        result = train_c(config, a.fold, stop_after=a.stop_after)
    elif a.command == 'extract':
        result = dict(games=len(extract(config, a.pool, a.fold)))
    elif a.command == 'fit':
        result = dict(model=str(fit(config)))
    else:
        result = evaluate(config, a.pool)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
