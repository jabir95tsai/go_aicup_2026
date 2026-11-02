"""Eight-GPU training probe with complete groups and measured costs."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.shared.checkpoint import save_checkpoint, resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import (
    autocast_context, setup_distributed, on_rank_zero, barrier)
from go_aicup.tasks.strength.shared.training import (
    configure_torch, seed_process, environment, restore_rng, log_event)
from .config import read_config, validate_config
from .dataset import PaperDataset, GroupBatchSampler, collate_groups
from .losses import grouped_losses
from .model import StrengthEstimator

DEFAULT_PROBE_CONFIG = ROOT / 'configs/strength/se_smoke100.json'


def scheduler_for(optimizer, training):
    """Scheduler step counts completed optimizer updates, including on resume."""
    def multiplier(completed):
        if completed >= training['final_lr_after_updates']:
            return training['final_lr'] / training['lr']
        if completed >= training['lr_drop_after_updates']:
            return training['lr_after_drop'] / training['lr']
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def process_stats(worker_pids=()):
    """Linux /proc counters include live loader workers; I/O may hit page cache."""
    result = dict(rss_bytes=0, read_bytes=0, write_bytes=0, rchar=0, wchar=0)
    for pid in (os.getpid(), *worker_pids):
        try:
            status = Path(f'/proc/{pid}/status').read_text()
            result['rss_bytes'] += next(int(line.split()[1]) * 1024
                                       for line in status.splitlines() if line.startswith('VmRSS:'))
            for line in Path(f'/proc/{pid}/io').read_text().splitlines():
                key, value = line.split(':')
                if key in result:
                    result[key] += int(value)
        except (FileNotFoundError, ProcessLookupError):
            continue
    result['main_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return result


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(config, *, resume=None, stop_after=None):
    validate_config(config)
    options = config['training']
    groups = options['groups_per_update']
    limit = options['probe_updates'] if stop_after is None else stop_after
    if (type(limit) is not int or not 1 <= limit <= options['probe_updates'] <= options['updates']
            or options['probe_updates'] > 500):
        raise ValueError('probe must run 1..500 updates within the training budget')
    for key in ('microbatch_groups', 'checkpoint_every'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key} must be positive')
    if type(options['workers']) is not int or options['workers'] < 0:
        raise ValueError('workers must be a nonnegative integer')
    if config['precision'] not in ('fp32', 'bf16') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('probe supports fp32/bf16 and cpu/cuda')
    if not 0 <= options['warmup_steps'] < options['probe_updates']:
        raise ValueError('invalid warmup measurement window')
    if not (0 < options['lr_drop_after_updates'] < options['final_lr_after_updates'] < options['updates']
            and 0 < options['final_lr'] <= options['lr_after_drop'] <= options['lr']):
        raise ValueError('invalid learning-rate schedule')
    if config['loss_weights'] != dict(bt=1.0, policy=1.0, value=1.0):
        raise ValueError('probe requires the confirmed equal loss weights')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA probe requires exactly eight GPUs via torchrun')
    configure_torch(config)
    # Shared runtime's batch divisibility check uses sample counts; here the unit is a complete group.
    runtime = dict(config, training=dict(options, global_batch_size=groups),
                   evaluation=dict(global_batch_size=groups))
    device, rank, world = setup_distributed(runtime)
    directory = Path(config['paths']['output_dir'])
    dataset = None
    lock = directory / '.writer.lock'
    locked = False
    started = time.perf_counter()
    try:
        def initialize():
            directory.mkdir(parents=True, exist_ok=True)
            if resume is None and any(directory.iterdir()):
                raise ValueError('nonempty probe output; use --resume or a new output directory')
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(str(os.getpid()))
        on_rank_zero(initialize, rank)
        locked = True
        seed_process(config['seed'])
        dataset = PaperDataset(config['paths']['dataset_dir'], augment=True, seed=config['seed'],
                               cache_dir=config['paths']['cache_dir'],
                               label_dir=config['paths']['label_dir'],
                               max_open_shards=config['loader']['max_open_shards'])
        record = dataset.record
        if (record['identity']['split'] != config['split']
                or record['identity']['positions'] != config['positions']):
            raise ValueError('prepared data/config mismatch')
        dataset._ensure_open()
        identity = dict(prepared_manifest_sha256=file_hash(dataset.directory / 'manifest.json'),
                        games_sha256=file_hash(dataset.directory / 'games.parquet'),
                        value_sha256=file_hash(dataset.label_dir / 'value.npy'))
        manifest = dict(dataset_id=object_hash(identity), cache_build_id=record['identity']['cache_build_id'])
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('train.py', 'config.py', 'dataset.py', 'model.py', 'losses.py')}
        model = StrengthEstimator(**config['model']).to(device)
        if options['sync_batch_norm'] and world > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if options.get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.SGD(model.parameters(), lr=options['lr'],
                                    momentum=options['momentum'], weight_decay=options['weight_decay'])
        scheduler = scheduler_for(optimizer, options)
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        step = 0
        payload = None
        if resume is not None:
            payload = torch.load(resolve_checkpoint(Path(resume)), map_location=device, weights_only=False)
            if (payload['config_sha256'] != object_hash(config) or payload['dataset_id'] != manifest['dataset_id']
                    or payload['world_size'] != world or payload['code_sha256'] != code):
                raise ValueError('checkpoint config/data/world/code mismatch')
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            step = payload['state']['optimizer_steps']
            if not 0 <= step < limit:
                raise ValueError('checkpoint is already at or past the requested stop')
        wrapped = (DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None)
                   if world > 1 else model)
        if payload is not None:
            restore_rng(payload['rng_by_rank'][rank], device, model)
        sampler = GroupBatchSampler(dataset, groups_per_update=groups, updates=limit,
                                    start_step=step, rank=rank, world_size=world)
        loader_options = dict(num_workers=options['workers'], pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(config['seed']))
        if options['workers']:
            loader_options.update(multiprocessing_context='spawn', prefetch_factor=1)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_groups, **loader_options)
        iterator = iter(loader)
        # PyTorch exposes worker processes only on the iterator; used solely for Linux resource accounting.
        worker_pids = [worker.pid for worker in getattr(iterator, '_workers', [])]
        def metadata():
            write_json(directory / 'config.json', config)
            write_json(directory / 'data_identity.json', identity)
            write_json(directory / 'environment.json', environment(device, world))
        if payload is not None:
            def trim_uncommitted_log():
                path = directory / 'train.jsonl'
                if path.exists():
                    lines = [line for line in path.read_text().splitlines()
                             if json.loads(line)['step'] <= step]
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(''.join(line + '\n' for line in lines))
                    os.replace(temporary, path)
            on_rank_zero(trim_uncommitted_log, rank)
        on_rank_zero(metadata, rank)
        barrier()
        synchronize(device)
        setup_seconds = time.perf_counter() - started
        rows = []
        measured_wall_times = []
        local_groups = groups // world
        micro = min(options['microbatch_groups'], local_groups)
        params = list(model.parameters())
        for completed in range(step, limit):
            tick = time.perf_counter()
            before_io = process_stats(worker_pids)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            data_start = time.perf_counter()
            batch = next(iterator)
            data_seconds = time.perf_counter() - data_start
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros(4, device=device)
            score_means = torch.zeros(10, device=device)
            lr = optimizer.param_groups[0]['lr']
            for begin in range(0, local_groups, micro):
                end = min(begin + micro, local_groups)
                weight = (end - begin) / local_groups
                sync = wrapped.no_sync() if world > 1 and end < local_groups else nullcontext()
                with sync:
                    with autocast_context(config, device):
                        planes = batch['planes'][begin:end].reshape(-1, 18, 19, 19)
                        planes = planes.to(device, dtype=torch.float32, non_blocking=True,
                            memory_format=torch.channels_last if options.get('channels_last', False) else torch.contiguous_format)
                        output = wrapped(planes)
                        output = {name: value.reshape(end - begin, 10, 7, *value.shape[1:])
                                  for name, value in output.items()}
                        terms = grouped_losses(output,
                            batch['policy_target'][begin:end].to(device, non_blocking=True),
                            batch['value_target'][begin:end].to(device, dtype=torch.float32, non_blocking=True),
                            batch['value_weight'][begin:end].to(device, non_blocking=True))
                    (terms['total'] * weight).backward()
                losses += torch.stack([terms[key].detach() for key in ('total', 'bt', 'policy', 'value')]) * weight
                score_means += output['score'].detach().float().mean(dim=(0, 2)) * weight
            synchronize(device)
            forward_backward_seconds = time.perf_counter() - compute_start
            diagnostic_start = time.perf_counter()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
            previous = torch.cat([parameter.detach().reshape(-1) for parameter in params])
            param_norm = previous.norm()
            synchronize(device)
            diagnostics_before = time.perf_counter() - diagnostic_start
            update_start = time.perf_counter()
            optimizer.step()
            scheduler.step()
            synchronize(device)
            optimizer_seconds = time.perf_counter() - update_start
            diagnostic_start = time.perf_counter()
            current = torch.cat([parameter.detach().reshape(-1) for parameter in params])
            update_norm = (current - previous).norm()
            if not torch.isfinite(current).all() or not torch.isfinite(losses).all():
                raise FloatingPointError('nonfinite parameters or loss')
            if world > 1:
                dist.all_reduce(losses)
                dist.all_reduce(score_means)
                losses /= world
                score_means /= world
            values = losses.tolist()
            row = dict(step=completed + 1, lr=lr, next_lr=optimizer.param_groups[0]['lr'],
                       **dict(zip(('total', 'bt', 'policy', 'value'), values)),
                       grad_norm=float(grad_norm), parameter_norm=float(param_norm),
                       update_norm=float(update_norm), update_parameter_ratio=float(update_norm / param_norm.clamp_min(1e-30)),
                       score_mean_by_rank=score_means.tolist(), data_seconds=data_seconds,
                       forward_backward_seconds=forward_backward_seconds, optimizer_seconds=optimizer_seconds)
            synchronize(device)
            row['diagnostic_seconds'] = diagnostics_before + time.perf_counter() - diagnostic_start
            checkpoint_start = time.perf_counter()
            row['checkpoint_bytes'] = 0
            if (completed + 1) % options['checkpoint_every'] == 0 or completed + 1 == limit:
                path = save_checkpoint(directory, model, optimizer, scheduler, scaler, config, manifest,
                    dict(optimizer_steps=completed + 1, epoch=0, next_batch=completed + 1),
                    device, rank, world, code_sha256=code, tag='probe')
                row['checkpoint_bytes'] = path.stat().st_size
            row['checkpoint_seconds'] = time.perf_counter() - checkpoint_start
            after_io = process_stats(worker_pids)
            row['rss_bytes'] = after_io['rss_bytes']
            row['main_peak_rss_bytes'] = after_io['main_peak_rss_bytes']
            row['io_delta'] = {key: after_io[key] - before_io[key]
                               for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')}
            row['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            row['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
            row['step_seconds'] = time.perf_counter() - tick
            row['rank'] = rank
            rank_rows = [None] * world
            if world > 1:
                dist.all_gather_object(rank_rows, row)
            else:
                rank_rows[0] = row
            event = dict(step=completed + 1, ranks=rank_rows)
            log_event(directory, rank, event)
            rows.append(rank_rows)
            duration = torch.tensor(time.perf_counter() - tick, device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            if completed + 1 > options['warmup_steps']:
                measured_wall_times.append(float(duration))
        elapsed = time.perf_counter() - started
        measured = measured_wall_times
        checkpoint_seconds = sum(max(row['checkpoint_seconds'] for row in group) for group in rows)
        summary = dict(start_step=step, completed_updates=limit, updates_this_session=len(rows),
                       world_size=world, precision=config['precision'], sync_batch_norm=options['sync_batch_norm'],
                       channels_last=options.get('channels_last', False),
                       cudnn_benchmark=options['cudnn_benchmark'],
                       microbatch_groups=micro, global_groups=groups, positions_per_update=groups * 70,
                       setup_seconds=setup_seconds, wall_seconds=elapsed,
                       gpu_hours=elapsed * world / 3600 if device.type == 'cuda' else 0,
                       checkpoint_seconds=checkpoint_seconds, warmup_steps=options['warmup_steps'],
                       finite_checks_passed=True,
                       peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for group in rows for r in group),
                       peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for group in rows for r in group),
                       peak_node_rss_bytes=max(sum(r['rss_bytes'] for r in group) for group in rows),
                       bn_note='Training BN sees each microbatch; accumulation does not reproduce full-batch BN.',
                       measurement_note='End-to-end summary includes logging/collection; per-rank step_seconds excludes them. Synchronized diagnostics reduce throughput. RAM is sampled; shared pages may be counted twice.')
        if measured:
            seconds = float(np.mean(measured))
            summary.update(measured_steps=len(measured), mean_step_seconds=seconds,
                           p50_step_seconds=float(np.median(measured)), p95_step_seconds=float(np.percentile(measured, 95)),
                           steps_per_second=1 / seconds, positions_per_second=groups * 70 / seconds,
                           projected_training_hours=seconds * options['updates'] / 3600,
                           projected_training_gpu_hours=seconds * options['updates'] * world / 3600 if device.type == 'cuda' else 0)
        on_rank_zero(lambda: write_json(directory / f'probe_summary_{step:05d}_{limit:05d}.json', summary), rank)
        return summary
    finally:
        if dataset is not None:
            dataset.close()
        if locked and rank == 0:
            lock.unlink(missing_ok=True)
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_PROBE_CONFIG)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='Stop at this completed update for a resume check')
    args = parser.parse_args(argv)
    train(read_config(args.config), resume=args.resume, stop_after=args.stop_after)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
