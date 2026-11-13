"""Two-stage, complete-group training with consumer-owned sampler checkpoints."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import signal
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.features.humansl_torch import _katago
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.reproduce_paper.train import process_stats, synchronize, trim_training_log
from go_aicup.tasks.strength.reproduce_paper_3.train import atomic_payload
from go_aicup.tasks.strength.SE_H_C_B2.common import writer_lock
from go_aicup.tasks.strength.SE_H_C_B2.evaluate import report
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import (
    autocast_context, on_rank_zero, setup_distributed,
)
from go_aicup.tasks.strength.shared.training import (
    configure_torch, environment, log_event, restore_rng, rng_state, seed_process,
)
from .config import monitor_every, read_config, runtime_config


def run_directory(config, phase, fold, stage):
    return Path(config['run_dir']) / phase / f'fold_{fold}' / f'stage{stage}'


def code_identity():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / 'src/go_aicup' / name for name in (
        'data/cv.py', 'features/humansl_torch.py', 'tasks/strength/shared/runtime.py',
        'tasks/strength/shared/training.py', 'tasks/strength/shared/dataset.py',
        'tasks/strength/reproduce_paper/model.py', 'tasks/strength/reproduce_paper_3/train.py')]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in sorted(files)}


def load_model(config, stage, source=None, device='cpu'):
    from .model import ResidualClassifier
    _katago()
    from katago.train.load_model import load_model as load_human
    human, _, _ = load_human(config['model']['human_checkpoint'], use_swa=False, device=torch.device(device))
    model = ResidualClassifier(human, stage=stage, channels=config['model']['channels'], blocks=config['model']['blocks'])
    if source is not None:
        path = resolve_checkpoint(Path(source))
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['identity']['stage'] != 1:
            raise ValueError('stage two requires a first-stage candidate')
        model.load_stage1(payload['model'])
    return model.to(device)


def load_candidate(path, config, device='cpu'):
    path = resolve_checkpoint(Path(path))
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['config_sha256'] != object_hash(payload['config'])
            or payload['identity']['human_sha256'] != file_hash(Path(config['model']['human_checkpoint']))):
        raise ValueError('candidate config or HumanSL source changed')
    if payload['code_sha256'] != code_identity():
        raise ValueError('candidate code identity changed')
    model = load_model(config, payload['identity']['stage'], device=device)
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload


def validation_transition(state, score, step, patience=3):
    if not math.isfinite(score):
        raise ValueError('monitor score must be finite')
    updated = dict(state)
    improved = state['best_score'] is None or score > state['best_score']
    if improved:
        updated.update(best_score=float(score), best_step=int(step), non_improvements=0)
    else:
        updated['non_improvements'] += 1
    updated.update(pending_monitor=False, early_stopped=updated['non_improvements'] >= patience)
    return updated, improved


def learning_rate(options, completed):
    """Step schedule first, then optional cosine decay to min_lr_ratio at the budget end."""
    lr = options['lr']
    for boundary, value in options['lr_schedule']:
        if completed >= boundary:
            lr = value
    if options.get('lr_decay', 'none') == 'cosine':
        ratio = options.get('min_lr_ratio', 0.0)
        progress = min(max(completed / options['updates'], 0.0), 1.0)
        lr *= ratio + (1 - ratio) * .5 * (1 + math.cos(math.pi * progress))
    return lr


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def forward_batch(model, batch):
    return model(batch['spatial'].float(), batch['global_inputs'], batch['metadata'],
                 batch['actions'], batch['sides'])


@torch.no_grad()
def monitor(config, model, rank, world):
    from .data import games, questions
    from .dataset import SelectedDataset
    q = questions(config, 'monitor')
    ids = set(int(i) for row in q.games for i in row)
    rows = games(config)
    rows = rows[rows.game_index.isin(ids)].sort_values('game_index').reset_index(drop=True)
    dataset = SelectedDataset(config, rows, model.humansl.config, teacher_required=False, augment=False)
    model.eval()
    local = []
    for row in range(rank, len(rows), world):
        raw = dataset[(row, 0)]
        batch = to_device({k: torch.as_tensor(v) for k, v in raw.items()}, next(model.parameters()).device)
        with autocast_context(config, next(model.parameters()).device):
            result = forward_batch(model, batch)
        logp = result['rank_logits'].float().log_softmax(-1).cpu().numpy()
        probabilities = np.exp(logp)
        sides = np.asarray(raw['sides'])
        local.append((int(rows.iloc[row].game_index),
                      np.stack([probabilities[sides == side].mean(0) for side in (0, 1)]
                               + [logp[sides == side].mean(0) for side in (0, 1)])))
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, local)
        local = [entry for shard in gathered for entry in shard]
    lookup = dict(local)
    if set(lookup) != ids:
        raise ValueError('monitor game coverage mismatch')
    by_question = np.stack([np.mean([lookup[int(i)] for i in row], axis=0) for row in q.games])
    # Early stopping keeps the fixed equal-weight probability fusion (no fitted temperature).
    # The log-probability fusion is reported for diagnosis only: softmax(mean log p) at T=1.
    fused_logp = by_question[:, 2:].mean(1)
    fused_logp = np.exp(fused_logp - fused_logp.max(1, keepdims=True))
    fused_logp /= fused_logp.sum(1, keepdims=True)
    result = {}
    for name, values in [('own', by_question[:, 0]), ('opp', by_question[:, 1]),
                         ('fused', by_question[:, :2].mean(1)), ('fused_logp_t1', fused_logp)]:
        result[name] = report(q, values)
        result[name]['ce'] = float(-np.log(np.maximum(values[np.arange(len(q)), q.rank_id.to_numpy()], 1e-30)).mean())
        for k, bucket in result[name]['by_k'].items():
            mask = q.k.to_numpy() == int(k)
            bucket['ce'] = float(-np.log(np.maximum(values[mask, q.rank_id.to_numpy()[mask]], 1e-30)).mean())
    model.train()
    return result


def train(config, stage, phase='dev', fold='all', source=None, resume=True, stop_after=None):
    from .data import games, load_prepared
    from .dataset import SelectedDataset, collate_groups
    from .sampler import GroupBatchSampler
    from .loss import grouped_losses
    if stage == 2 and source is None:
        raise ValueError('stage two requires --source pointing to the same fold stage-one candidate')
    if stage == 1 and source is not None:
        raise ValueError('stage one starts from the fixed HumanSL model')
    if phase == 'dev' and stage == 2 and not config.get('smoke'):
        selection = json.loads((Path(config['run_dir']) / 'selection/stage1.json').read_text())
        expected = run_directory(config, phase, fold, 1) / f'{selection["selected"]}.pt'
        if Path(source).resolve() != expected.resolve():
            raise ValueError('stage two must use the stage-one complete-B2 selection for this fold')
    options = dict(config[f'stage{stage}'])
    if phase != 'dev':
        raise ValueError('release retraining was removed: release reuses the four development fold models')
    if options['updates'] is None or options['lr'] is None:
        raise ValueError('set explicit measured updates and learning rate before a long run')
    world_env = int(__import__('os').environ.get('WORLD_SIZE', '1'))
    runtime = runtime_config(config, world_env)
    configure_torch(runtime)
    device, rank, world = setup_distributed(runtime)
    seed_process(config['seed'] + rank)
    directory = run_directory(config, phase, fold, stage)
    lock = None
    previous_handlers = {}
    try:
        # The owner lock stays held for the entire run, including publication.
        def acquire():
            nonlocal lock
            lock = writer_lock(directory)
            lock.__enter__()
        on_rank_zero(acquire, rank)
        manifest = load_prepared(config)
        table = games(config)
        selected = table[table.role.eq('D')]
        if fold != 'all':
            selected = selected[~selected.oof_fold.eq(int(fold))]
        selected = selected.sort_values('game_index').reset_index(drop=True)
        if selected.empty:
            raise ValueError('empty training fold')
        identity = dict(phase=phase, fold=str(fold), stage=stage,
                        training_players=sorted(selected.player_id.astype(str).unique().tolist()),
                        human_sha256=file_hash(Path(config['model']['human_checkpoint'])),
                        stage1_source_sha256=file_hash(Path(source)) if source else None)
        if source:
            source_payload = torch.load(resolve_checkpoint(Path(source)), map_location='cpu', weights_only=False)
            old = source_payload['identity']
            if (old['phase'] != phase or str(old['fold']) != str(fold)
                    or old['training_players'] != identity['training_players']
                    or source_payload['dataset_id'] != manifest['dataset_id']
                    or source_payload['config_sha256'] != object_hash(config)
                    or source_payload['code_sha256'] != code_identity()
                    or old['human_sha256'] != identity['human_sha256']):
                raise ValueError('stage-one source has different fold, players, data or HumanSL identity')
            identity['stage1_identity'] = old
        code = code_identity()
        state = dict(optimizer_steps=0, best_score=None, best_step=None,
                     non_improvements=0, pending_monitor=False, early_stopped=False)
        model = load_model(config, stage, source, device)
        if config['training']['sync_batch_norm'] and world > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if config['training']['channels_last']:
            model.to(memory_format=torch.channels_last)
        parameters = [p for p in model.parameters() if p.requires_grad]
        common = dict(lr=options['lr'], weight_decay=config['training']['weight_decay'])
        optimizer = (torch.optim.AdamW(parameters, betas=tuple(config['training']['betas']), eps=config['training']['eps'], **common)
                     if config['training']['optimizer'] == 'adamw' else
                     torch.optim.SGD(parameters, momentum=0.9, **common))
        latest = directory / 'latest.pt'
        payload = None
        if latest.exists():
            if not resume:
                raise FileExistsError('run already has resumable state')
            payload = torch.load(resolve_checkpoint(latest), map_location='cpu', weights_only=False)
            if (payload['config_sha256'] != object_hash(config) or payload['identity'] != identity
                    or payload['dataset_id'] != manifest['dataset_id'] or payload['code_sha256'] != code
                    or payload['world_size'] != world):
                raise ValueError('resume config, fold, data, code or world size mismatch')
            model.load_state_dict(payload['model'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            state = payload['state']
            restore_rng(payload['rng'][rank], device, model)
        dataset = SelectedDataset(config, selected, model.humansl.config, teacher_required=True, augment=True)
        sampler = GroupBatchSampler(dataset, groups_per_update=config['training']['groups_per_update'],
                    updates=options['updates'], start_step=state['optimizer_steps'], rank=rank, world_size=world,
                    state=payload['sampler'] if payload else None,
                    microbatch_groups=config['training']['microbatch_groups'])
        loader_options = dict(batch_sampler=sampler, collate_fn=collate_groups,
                              num_workers=config['training']['workers'], pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(config['seed'] + rank))
        if config['training']['workers']:
            loader_options.update(prefetch_factor=1, multiprocessing_context='spawn', persistent_workers=True)
        loader = DataLoader(dataset, **loader_options)
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                         broadcast_buffers=False) if world > 1 else model
        stop = [False]
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            previous_handlers[signum] = signal.signal(signum, lambda *_: stop.__setitem__(0, True))
        stop_path = directory / '.stop-requested'
        on_rank_zero(lambda: stop_path.unlink(missing_ok=True), rank)
        on_rank_zero(lambda: write_json(directory / 'config.json', config), rank)
        if payload:
            on_rank_zero(lambda: trim_training_log(directory, state['optimizer_steps']), rank)
        on_rank_zero(lambda: write_json(directory / f'environment_{time.time_ns()}.json', environment(device, world)), rank)
        started = time.perf_counter()
        metrics = dict(data_seconds=0., compute_seconds=0., monitor_seconds=0., checkpoint_seconds=0.)
        def snapshot():
            local_rng = rng_state(device, model)
            rng = [None] * world
            if world > 1:
                dist.all_gather_object(rng, local_rng)
            else:
                rng[0] = local_rng
            return dict(config=config, config_sha256=object_hash(config), state=dict(state),
                        dataset_id=manifest['dataset_id'], code_sha256=code, identity=identity,
                        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng,
                        sampler=sampler.state_dict(state['optimizer_steps']), world_size=world)
        def save(name):
            tick = time.perf_counter()
            value = snapshot()
            on_rank_zero(lambda: atomic_payload(directory / name, value), rank)
            metrics['checkpoint_seconds'] += time.perf_counter() - tick
        def validate():
            nonlocal state
            tick = time.perf_counter()
            result = monitor(config, model, rank, world)
            updated, improved = validation_transition(state, result['fused']['equal_k'], state['optimizer_steps'])
            state = updated
            if improved:
                save('best.pt')
            on_rank_zero(lambda: write_json(directory / f'monitor_{state["optimizer_steps"]:08d}.json', result), rank)
            metrics['monitor_seconds'] += time.perf_counter() - tick
            save('latest.pt')
        if state['pending_monitor'] and phase == 'dev':
            validate()
        model.train()
        iterator = iter(loader)
        worker_pids = [worker.pid for worker in getattr(iterator, '_workers', [])]
        before_resources = process_stats(worker_pids)
        peak_rss = before_resources['rss_bytes']
        step_seconds = []
        replayed_plies = 0
        start_step = state['optimizer_steps']
        accumulation = config['training']['groups_per_update'] // world // config['training']['microbatch_groups']
        while state['optimizer_steps'] < options['updates'] and not state['early_stopped']:
            synchronize(device)
            step_start = time.perf_counter()
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(options, state['optimizer_steps'])
            optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros(4, device=device, dtype=torch.float64)
            for micro in range(accumulation):
                tick = time.perf_counter()
                batch = to_device(next(iterator), device)
                if config.get('smoke'):
                    # Each game is replayed through its largest selected pre-action ply.
                    replayed_plies += int(batch['ply'].reshape(-1, 30).max(1).values.sum().item()) + len(batch['ply']) // 30
                    synchronize(device)
                metrics['data_seconds'] += time.perf_counter() - tick
                tick = time.perf_counter()
                local_games = len(batch['actions']) // 30
                context = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
                with context:
                    with autocast_context(config, device):
                        output = forward_batch(wrapped, batch)
                    values = grouped_losses(output, batch['labels'], batch['values'], batch['actions'], stage=stage, games=local_games)
                    weight = local_games * world / (10 * config['training']['groups_per_update'])
                    (values['total'] * weight).backward()
                for index, name in enumerate(('total', 'rank', 'policy', 'value')):
                    if name in values:
                        losses[index] += values[name].detach().double() * weight
                metrics['compute_seconds'] += time.perf_counter() - tick
            invalid = (~torch.stack([torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]).all()).int()
            if world > 1:
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            if invalid.item():
                raise FloatingPointError('nonfinite gradient; latest complete update retained')
            optimizer.step()
            synchronize(device)
            step_seconds.append(time.perf_counter() - step_start)
            if config.get('smoke'):
                peak_rss = max(peak_rss, process_stats(worker_pids)['rss_bytes'])
            state['optimizer_steps'] += 1
            step = state['optimizer_steps']
            if world > 1:
                dist.all_reduce(losses)
                losses /= world
            on_rank_zero(lambda: log_event(directory, 0, dict(step=step, lr=optimizer.param_groups[0]['lr'],
                          **{name: float(losses[i]) for i, name in enumerate(('total', 'rank', 'policy', 'value'))})), rank)
            pending = phase == 'dev' and (step % monitor_every(config, f'stage{stage}') == 0 or step == options['updates'])
            state['pending_monitor'] = pending
            if pending or step % config['training']['checkpoint_every'] == 0:
                save('latest.pt')
            if pending:
                validate()
            requested = torch.tensor(int(stop[0] or stop_path.exists()
                                         or (stop_after is not None and step >= stop_after)), device=device)
            if world > 1:
                dist.all_reduce(requested, op=dist.ReduceOp.MAX)
            if requested.item():
                save('latest.pt')
                break
        complete = state['early_stopped'] or state['optimizer_steps'] == options['updates']
        save('latest.pt')
        if complete:
            save('final.pt')
        elapsed = time.perf_counter() - started
        updates = state['optimizer_steps'] - start_step
        resources = process_stats(worker_pids)
        measured = step_seconds[5:] if len(step_seconds) > 5 else step_seconds
        summary = dict(status='completed' if complete else 'paused', stage=stage, phase=phase, fold=fold,
                       state=state, seconds=elapsed, steps_per_second=updates / elapsed,
                       effective_positions_per_second=updates * config['training']['groups_per_update'] * 300 / elapsed,
                       gpu_hours=elapsed * world / 3600 if device.type == 'cuda' else 0,
                       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                       peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                       training_steps_per_second=len(measured) / sum(measured) if measured else 0,
                       replayed_plies=replayed_plies, replayed_plies_per_second=replayed_plies / elapsed,
                       peak_rss_bytes=peak_rss, resources=resources,
                       io_delta={k: resources[k] - before_resources[k] for k in ('read_bytes', 'write_bytes', 'rchar', 'wchar')},
                       disk_free_bytes=shutil.disk_usage(directory).free,
                       **metrics)
        per_rank = [None] * world
        resource_record = dict(rank=rank, seconds=elapsed, peak_rss_bytes=peak_rss,
                               gpu_allocated_bytes=summary['peak_gpu_allocated_bytes'],
                               gpu_reserved_bytes=summary['peak_gpu_reserved_bytes'],
                               io_delta=summary['io_delta'], **metrics)
        if world > 1:
            dist.all_gather_object(per_rank, resource_record)
        else:
            per_rank[0] = resource_record
        summary['per_rank'] = per_rank
        summary['peak_gpu_allocated_bytes'] = max(r['gpu_allocated_bytes'] for r in per_rank)
        summary['peak_gpu_reserved_bytes'] = max(r['gpu_reserved_bytes'] for r in per_rank)
        summary['gpu_hours'] = sum(r['seconds'] for r in per_rank) / 3600 if device.type == 'cuda' else 0
        on_rank_zero(lambda: write_json(directory / 'summary.json', summary), rank)
        return summary
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if lock is not None:
            lock.__exit__(None, None, None)
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--stage', type=int, choices=(1, 2), required=True)
    parser.add_argument('--phase', choices=('dev',), default='dev')
    parser.add_argument('--fold', choices=('all', '1', '2', '3', '4'), default='all')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--no-resume', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config) if args.config else read_config()
    print(json.dumps(train(config, args.stage, args.phase, args.fold, args.source, not args.no_resume)))


if __name__ == '__main__':
    main()
