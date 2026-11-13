"""Eight-GPU SE training with resumable updates; optional bounded performance probe."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import resource
import signal
import shutil
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
    autocast_context, setup_distributed, on_rank_zero, barrier, model_state_hash)
from go_aicup.tasks.strength.shared.training import (
    configure_torch, seed_process, environment, restore_rng, log_event)
from .config import DEFAULT_CONFIG, read_config, validate_config
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


def trim_training_log(directory, step):
    """Discard updates beyond the durable cursor, including a torn final line."""
    path = directory / 'train.jsonl'
    if not path.exists():
        return
    temporary = path.with_suffix('.tmp')
    with path.open() as source, temporary.open('w') as target:
        for line in source:
            if not line.endswith('\n'):
                break
            event = json.loads(line)
            if event['step'] <= step:
                target.write(line)
    os.replace(temporary, path)


def finalize_training(directory, model, config, manifest, code, step):
    """Publish and verify final weights before any resumable state is removed."""
    if step != config['training']['updates']:
        raise ValueError('cannot finalize incomplete training')
    payload = dict(artifact_format='strength_estimator_1',
                   model={name: value.detach().cpu() for name, value in model.state_dict().items()},
                   config=config, optimizer_steps=step, dataset_id=manifest['dataset_id'],
                   cache_build_id=manifest['cache_build_id'], code_sha256=code,
                   model_state_sha256=model_state_hash(model))
    path = directory / 'model.pt'
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location='cpu', weights_only=False)
    verifier = StrengthEstimator(**config['model'])
    verifier.load_state_dict(loaded['model'])
    if model_state_hash(verifier) != payload['model_state_sha256']:
        raise ValueError('final model reload verification failed')
    os.replace(temporary, path)
    write_json(path.with_suffix('.complete.json'), dict(status='completed', optimizer_steps=step,
                                                       file=path.name, sha256=file_hash(path)))


def archive_best_checkpoints(directory):
    """Commit the two best full checkpoints before deleting resumable storage."""
    archive = directory / 'best_checkpoints.json'
    if archive.exists():
        records = json.loads(archive.read_text())['checkpoints']
    else:
        best = directory / 'best.json'
        selected = json.loads(best.read_text())['checkpoints'] if best.exists() else []
        latest = directory / 'last.json'
        if latest.exists():
            selected.append(json.loads(latest.read_text()))
        selected = list({record['file']: record for record in selected}.values())
        if any(not np.isfinite(record['loss']) for record in selected):
            raise ValueError('invalid ranked checkpoint loss')
        selected = sorted(selected, key=lambda record: (record['loss'], -record['optimizer_steps'], record['file']))[:2]
        records = []
        for index, record in enumerate(selected[:2], start=1):
            if Path(record['file']).name != record['file']:
                raise ValueError('unsafe ranked checkpoint path')
            source = resolve_checkpoint(directory / 'checkpoints' / record['file'])
            if file_hash(source) != record['sha256']:
                raise ValueError('ranked checkpoint hash mismatch')
            target = directory / f'best_{index}.pt'
            temporary = target.with_suffix('.pt.tmp')
            shutil.copyfile(source, temporary)
            if file_hash(temporary) != record['sha256']:
                raise ValueError('archived checkpoint copy verification failed')
            os.replace(temporary, target)
            published = dict(record, file=target.name)
            write_json(target.with_suffix('.complete.json'), published)
            records.append(published)
        write_json(archive, {'checkpoints': records})
    if len(records) > 2:
        raise ValueError('invalid best checkpoint archive')
    for index, record in enumerate(records, start=1):
        if (record['file'] != f'best_{index}.pt' or not np.isfinite(record['loss'])
                or file_hash(resolve_checkpoint(directory / record['file'])) != record['sha256']):
            raise ValueError('archived checkpoint identity/hash mismatch')


def cleanup_training(directory, config):
    """Remove resumable state only after a complete final artifact is verified."""
    path = resolve_checkpoint(directory / 'model.pt')
    record = json.loads(path.with_suffix('.complete.json').read_text())
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (record.get('status') != 'completed'
            or payload.get('artifact_format') != 'strength_estimator_1'
            or payload['optimizer_steps'] != config['training']['updates']
            or object_hash(payload['config']) != object_hash(config)):
        raise ValueError('final model/config mismatch; refusing checkpoint cleanup')
    archive_best_checkpoints(directory)
    checkpoints = directory / 'checkpoints'
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    for name in ('last.json', 'last.json.tmp', 'best.json', 'best.json.tmp', '.stop-requested', 'train.tmp',
                 'model.pt.tmp', 'model.complete.json.tmp', 'best_1.pt.tmp', 'best_2.pt.tmp',
                 'best_1.complete.json.tmp', 'best_2.complete.json.tmp', 'best_checkpoints.json.tmp'):
        (directory / name).unlink(missing_ok=True)


def train(config, *, resume=None, stop_after=None, keep_distributed=False):
    validate_config(config)
    options = config['training']
    groups = options['groups_per_update']
    if type(groups) is not int or groups < 1 or type(options['updates']) is not int or options['updates'] < 1:
        raise ValueError('groups and updates must be positive integers')
    if options['optimizer'] != 'sgd':
        raise ValueError('SE training requires SGD')
    probe = 'probe_updates' in options
    budget = options['probe_updates'] if probe else options['updates']
    limit = budget if stop_after is None else stop_after
    if probe and (type(budget) is not int or not 1 <= budget <= 500):
        raise ValueError('probe must run 1..500 updates within the training budget')
    if type(limit) is not int or not 1 <= limit <= budget <= options['updates']:
        raise ValueError('stop must be a positive update within the training budget')
    for key in ('microbatch_groups', 'checkpoint_every'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'{key} must be positive')
    if type(options['workers']) is not int or options['workers'] < 0:
        raise ValueError('workers must be a nonnegative integer')
    if config['precision'] not in ('fp32', 'bf16') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('training supports fp32/bf16 and cpu/cuda')
    if probe and not 0 <= options['warmup_steps'] < budget:
        raise ValueError('invalid warmup measurement window')
    if not (0 < options['lr_drop_after_updates'] < options['final_lr_after_updates'] < options['updates']
            and 0 < options['final_lr'] <= options['lr_after_drop'] <= options['lr']):
        raise ValueError('invalid learning-rate schedule')
    if config['loss_weights'] != dict(bt=1.0, policy=1.0, value=1.0):
        raise ValueError('training requires the confirmed equal loss weights')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA training requires exactly eight GPUs via torchrun')
    configure_torch(config)
    # Shared runtime's batch divisibility check uses sample counts; here the unit is a complete group.
    runtime = dict(config, training=dict(options, global_batch_size=groups),
                   evaluation=dict(global_batch_size=groups))
    device, rank, world = setup_distributed(runtime)
    directory = Path(config['paths']['output_dir'])
    dataset = None
    lock = directory / '.writer.lock'
    stop_file = directory / '.stop-requested'
    locked = False
    started = time.perf_counter()
    handlers = {}
    stop_requested = False
    handoff = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    try:
        if not probe:
            for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
                handlers[signum] = signal.signal(signum, request_stop)
        def initialize():
            directory.mkdir(parents=True, exist_ok=True)
            completed_artifact = not probe and (directory / 'model.complete.json').is_file()
            if resume is None and any(directory.iterdir()) and not completed_artifact:
                raise ValueError('nonempty training output; use --resume or a new output directory')
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(str(os.getpid()))
            if not probe:
                stop_file.unlink(missing_ok=True)
        on_rank_zero(initialize, rank)
        locked = True
        if not probe:
            completed_artifact = [False]
            def finish_completed():
                if (directory / 'model.complete.json').exists():
                    cleanup_training(directory, config)
                    completed_artifact[0] = True
            on_rank_zero(finish_completed, rank)
            if world > 1:
                dist.broadcast_object_list(completed_artifact, src=0)
            if completed_artifact[0]:
                handoff = True
                return dict(start_step=options['updates'], completed_updates=options['updates'],
                            updates_this_session=0, status='completed', finite_checks_passed=True)
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
        if not probe:
            package = ROOT / 'src/go_aicup'
            for name in ('tasks/strength/shared/runtime.py', 'tasks/strength/shared/training.py',
                         'tasks/strength/shared/checkpoint.py', 'tasks/strength/shared/dataset.py',
                         'data/board_cache.py', 'artifacts.py'):
                code[name] = file_hash(package / name)
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
            checkpoint = resolve_checkpoint(Path(resume))
            if not probe and checkpoint.resolve() != resolve_checkpoint(directory / 'last.json').resolve():
                raise ValueError('resume must use this run latest checkpoint')
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if (payload['config_sha256'] != object_hash(config) or payload['dataset_id'] != manifest['dataset_id']
                    or payload['world_size'] != world or payload['code_sha256'] != code):
                raise ValueError('checkpoint config/data/world/code mismatch')
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            step = payload['state']['optimizer_steps']
            if not 0 <= step <= limit or (probe and step == limit):
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
        worker_pids = [worker.pid for worker in getattr(iterator, '_workers', [])] if probe else []
        session_id = time.time_ns()
        def metadata():
            write_json(directory / 'config.json', config)
            write_json(directory / 'data_identity.json', identity)
            current_environment = environment(device, world)
            if not (directory / 'environment.json').exists():
                write_json(directory / 'environment.json', current_environment)
            if not probe:
                write_json(directory / f'environment_{session_id}.json', current_environment)
        if payload is not None:
            on_rank_zero(lambda: trim_training_log(directory, step), rank)
        on_rank_zero(metadata, rank)
        barrier()
        synchronize(device)
        setup_seconds = time.perf_counter() - started
        rows = []
        measured_wall_times = []
        local_groups = groups // world
        micro = min(options['microbatch_groups'], local_groups)
        params = list(model.parameters())
        completed_updates = step
        loss_sum = 0.0
        loss_count = 0
        for completed in range(step, limit):
            tick = time.perf_counter()
            before_io = process_stats(worker_pids) if probe else None
            if probe and device.type == 'cuda':
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
                if probe:
                    score_means += output['score'].detach().float().mean(dim=(0, 2)) * weight
            if probe:
                synchronize(device)
            forward_backward_seconds = time.perf_counter() - compute_start
            if not probe:
                torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                finite = torch.stack([torch.isfinite(parameter).all() for parameter in params]).all()
                finite &= torch.isfinite(losses).all()
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    dist.all_reduce(losses)
                    losses /= world
                if not finite:
                    raise FloatingPointError('nonfinite parameters or loss')
                completed_updates = completed + 1
                values = losses.tolist()
                loss_sum += values[0]
                loss_count += 1
                event = dict(step=completed_updates, lr=lr, next_lr=optimizer.param_groups[0]['lr'],
                             **dict(zip(('total', 'bt', 'policy', 'value'), values)))
                on_rank_zero(lambda: log_event(directory, rank, event), rank)
                stopping = torch.tensor(int(stop_requested or (rank == 0 and stop_file.exists())), device=device)
                if world > 1:
                    dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
                if completed_updates % options['checkpoint_every'] == 0 or completed_updates == limit or stopping.item():
                    checkpoint_loss = loss_sum / loss_count
                    save_checkpoint(directory, model, optimizer, scheduler, scaler, config, manifest,
                        dict(optimizer_steps=completed_updates, epoch=0, next_batch=completed_updates,
                             checkpoint_loss=checkpoint_loss, checkpoint_loss_updates=loss_count),
                        device, rank, world, code_sha256=code, tag='training', loss=checkpoint_loss)
                    loss_sum, loss_count = 0.0, 0
                if stopping.item():
                    break
                continue
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
        if not probe:
            complete = completed_updates == options['updates']
            if complete:
                on_rank_zero(lambda: finalize_training(directory, model, config, manifest, code,
                                                        completed_updates), rank)
            summary = dict(start_step=step, completed_updates=completed_updates,
                           updates_this_session=completed_updates - step,
                           status='completed' if complete else 'stopped', finite_checks_passed=True)
            on_rank_zero(lambda: write_json(directory / f'training_session_{session_id}.json', summary), rank)
            on_rank_zero(lambda: write_json(directory / 'training_summary.json', summary), rank)
            if complete:
                on_rank_zero(lambda: cleanup_training(directory, config), rank)
                handoff = True
            return summary
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
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if dataset is not None:
            dataset.close()
        if locked and rank == 0:
            lock.unlink(missing_ok=True)
        if dist.is_initialized() and not (keep_distributed and handoff):
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int, help='Stop at this completed update for a resume check')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    postprocess = 'probe_updates' not in config['training'] and 'evaluation' in config
    summary = train(config, resume=args.resume, stop_after=args.stop_after, keep_distributed=postprocess)
    try:
        if postprocess and summary['status'] == 'completed':
            from .evaluate import evaluate
            evaluate(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
