"""SE3 training, recoverable validation transactions and best-score publication."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import resource
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
from go_aicup.tasks.strength.reproduce_paper.train import trim_training_log
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import setup_distributed, autocast_context, on_rank_zero, model_state_hash
from go_aicup.tasks.strength.shared.training import configure_torch, seed_process, environment, rng_state, restore_rng, log_event
from .config import DEFAULT_CONFIG, read_config, validate_config
from .dataset import GroupBatchSampler, collate_groups, validate_prepared
from .model import StrengthEstimator
from .losses import grouped_losses
from .evaluate import prepare_monitor, monitor, collective_result


def code_fingerprints():
    files = [Path(__file__).with_name(name) for name in
             ('config.py', 'model.py', 'losses.py', 'dataset.py', 'train.py', 'evaluate.py')]
    files += [ROOT / 'src/go_aicup' / name for name in
              ('artifacts.py', 'data/board_cache.py', 'data/cv.py',
               'tasks/strength/shared/runtime.py', 'tasks/strength/shared/training.py',
               'tasks/strength/shared/checkpoint.py', 'tasks/strength/shared/dataset.py',
               'tasks/strength/reproduce_paper/model.py', 'tasks/strength/reproduce_paper/dataset.py',
               'tasks/strength/reproduce_paper/prepare.py', 'tasks/strength/reproduce_paper/config.py',
               'tasks/strength/reproduce_paper/questions.py', 'tasks/strength/reproduce_paper/train.py')]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def runtime_config(config):
    return dict(config, training=dict(config['training'], global_batch_size=config['training']['groups_per_update']),
                evaluation=dict(global_batch_size=config['training']['groups_per_update']))


def scheduler_for(optimizer, training):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed:
        1. if completed < training['lr_drop_after_updates'] else training['lr_after_drop'] / training['lr'])


def validation_transition(state, result, step):
    """Pure, retryable transition. Ties replace best and reset patience."""
    if state['evaluated_step'] == step:
        return dict(state), False
    if not state['validation_pending'] or step != state['optimizer_steps'] or step < state['evaluated_step']:
        raise ValueError('invalid validation progress')
    score = result['equal_k']
    if not math.isfinite(score) or not 0 <= score <= 1 or set(result['buckets']) != {'5', '10', '15', '20'}:
        raise ValueError('invalid monitoring result')
    for row in result['buckets'].values():
        if (row['n'] < 1 or not math.isfinite(row['score']) or row['exact'] < 0 or row['adjacent'] < 0
                or row['exact'] + row['adjacent'] > row['n']
                or row['score'] != (row['exact'] + row['adjacent'] / math.e) / row['n']):
            raise ValueError('missing/nonfinite k bucket')
    if score != sum(result['buckets'][str(k)]['score'] for k in (5, 10, 15, 20)) / 4:
        raise ValueError('monitor Equal-k mismatch')
    updated = dict(state, evaluated_step=step, validation_pending=False)
    accepted = state['best_score'] is None or score >= state['best_score']
    if accepted:
        updated.update(best_score=score, best_step=step, lower_count=0)
    else:
        updated['lower_count'] += 1
    return updated, accepted


def atomic_payload(path, payload):
    """Reload before replacing any published reference; retain old sources on failure."""
    temporary = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location='cpu', weights_only=False)
    if (loaded['config_sha256'] != object_hash(loaded['config'])
            or loaded['config_sha256'] != payload['config_sha256']
            or loaded['state'] != payload['state'] or loaded['dataset_id'] != payload['dataset_id']
            or loaded['code_sha256'] != payload['code_sha256']
            or set(loaded['model']) != set(payload['model'])
            or any(not torch.equal(value.cpu(), loaded['model'][key]) for key, value in payload['model'].items())):
        raise ValueError('checkpoint reload verification failed')
    del loaded
    os.replace(temporary, path)
    write_json(path.with_suffix('.complete.json'), dict(file=path.name, sha256=file_hash(path),
               optimizer_steps=payload['state']['optimizer_steps']))
    resolve_checkpoint(path)
    return path


def load_artifact(path, config):
    payload = torch.load(resolve_checkpoint(path), map_location='cpu', weights_only=False)
    if payload['config_sha256'] != object_hash(config) or payload.get('artifact_format') != 'se3_classifier_1':
        raise ValueError('SE3 artifact config/format mismatch')
    if path.name == 'model.pt':
        state = payload['state']
        if (payload.get('stop_reason') not in ('patience', 'budget') or state['best_step'] is None
                or payload['best_step'] != state['best_step'] or payload['stopped_step'] < payload['best_step']
                or payload['stopped_step'] > config['training']['updates']):
            raise ValueError('invalid completed best model')
    return payload


def train(config, *, resume=None, stop_after=None, keep_distributed=False):
    validate_config(config)
    options = config['training']
    budget = options['updates']
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= budget):
        raise ValueError('stop_after must be within the update budget')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA training requires exactly eight GPUs')
    configure_torch(config)
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if config['device'] == 'cuda' else torch.device('cpu')
    else:
        device, rank, world = setup_distributed(runtime_config(config))
    directory = Path(config['paths']['output_dir'])
    lock = directory / '.writer.lock'
    stop_file = directory / '.stop-requested'
    locked, stopping = False, False
    dataset = validation = None
    handlers = {}
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    try:
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, request_stop)
        def initialize():
            directory.mkdir(parents=True, exist_ok=True)
            if resume is None and any(directory.iterdir()) and not (directory / 'model.complete.json').exists():
                raise ValueError('nonempty output; use --resume with this run last.json')
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, 'w') as handle:
                handle.write(str(os.getpid()))
            stop_file.unlink(missing_ok=True)
        on_rank_zero(initialize, rank)
        locked = True
        code = code_fingerprints()
        if (directory / 'model.complete.json').exists():
            payload = load_artifact(directory / 'model.pt', config)
            if payload['code_sha256'] != code:
                raise ValueError('completed model code mismatch')
            return dict(status='completed', completed_updates=payload['stopped_step'], updates_this_session=0,
                        best_step=payload['best_step'], stop_reason=payload['stop_reason'])
        identity = collective_result(lambda: validate_prepared(config))
        on_rank_zero(lambda: prepare_monitor(config), rank)
        monitor_record = json.loads((directory / 'monitor/manifest.json').read_text())
        identity['monitor_id'] = monitor_record['monitor_id']
        manifest = dict(dataset_id=object_hash(identity))
        dataset = PaperDataset(config['paths']['dataset_dir'], augment=True, seed=config['seed'],
            cache_dir=config['paths']['cache_dir'], label_dir=config['paths']['label_dir'],
            max_open_shards=config['loader']['max_open_shards'])
        validation = PaperDataset(config['paths']['dataset_dir'], split='valid', cache_dir=config['paths']['cache_dir'],
                                   max_open_shards=config['loader']['max_open_shards'])
        seed_process(config['seed'])
        model = StrengthEstimator(**config['model']).to(device)
        if options['sync_batch_norm'] and world > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if options['channels_last']:
            model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=tuple(options['betas']),
                                      eps=options['eps'], weight_decay=options['weight_decay'])
        scheduler = scheduler_for(optimizer, options)
        state = dict(optimizer_steps=0, evaluated_step=0, validation_pending=False,
                     best_score=None, best_step=None, lower_count=0)
        pending_rng = None
        if resume is not None:
            path = resolve_checkpoint(Path(resume))
            if path.resolve() != resolve_checkpoint(directory / 'last.json').resolve():
                raise ValueError('resume must use this run latest checkpoint')
            payload = load_artifact(path, config)
            if (payload['dataset_id'] != manifest['dataset_id'] or payload['world_size'] != world
                    or payload['code_sha256'] != code):
                raise ValueError('checkpoint data/world/code mismatch')
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            state = payload['state']
            pending_rng = payload['rng_by_rank'][rank]
            del payload
        start_step = state['optimizer_steps']
        if stop_after is not None and start_step > stop_after:
            raise ValueError('checkpoint past requested stop')
        sampler = GroupBatchSampler(dataset, groups_per_update=options['groups_per_update'], updates=budget,
                                    start_step=start_step, rank=rank, world_size=world)
        # Production uses one complete local batch: smaller CPU batches are engineering checks.
        local_groups = options['groups_per_update'] // world
        if config['device'] == 'cuda' and options['microbatch_groups'] != local_groups:
            raise ValueError('CUDA requires a complete local batch without gradient accumulation')
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
        if pending_rng is not None:
            restore_rng(pending_rng, device, model)
        def metadata():
            write_json(directory / 'config.json', config)
            write_json(directory / 'data_identity.json', identity)
            write_json(directory / f'environment_{time.time_ns()}.json', environment(device, world))
            if resume is not None:
                trim_training_log(directory, start_step)
        on_rank_zero(metadata, rank)
        def snapshot():
            local = rng_state(device, model)
            states = [local]
            if world > 1:
                states = [None] * world
                dist.all_gather_object(states, local)
            payload = dict(artifact_format='se3_classifier_1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), config=config, config_sha256=object_hash(config),
                dataset_id=manifest['dataset_id'], code_sha256=code, world_size=world,
                rng_by_rank=states, state=dict(state), monitor_id=identity['monitor_id'])
            return payload
        def save_latest():
            payload = snapshot()
            def publish():
                path = directory / 'checkpoints' / f'update_{state["optimizer_steps"]:08d}_{time.time_ns()}.pt'
                atomic_payload(path, payload)
                record = json.loads(path.with_suffix('.complete.json').read_text())
                write_json(directory / 'last.json', record)
                for old in path.parent.glob('update_*.pt'):
                    if old != path:
                        old.unlink()
                        old.with_suffix('.complete.json').unlink(missing_ok=True)
            on_rank_zero(publish, rank)
        def validate_pending():
            nonlocal state
            step = state['optimizer_steps']
            result_file = directory / 'validation' / f'step_{step:08d}.json'
            model_id = model_state_hash(model) if rank == 0 else None
            if world > 1:
                holder = [model_id]
                dist.broadcast_object_list(holder, src=0)
                model_id = holder[0]
            if result_file.exists():
                result = json.loads(result_file.read_text())
                if result['model_id'] != model_id or result['monitor_id'] != identity['monitor_id']:
                    raise ValueError('validation transaction identity mismatch')
            else:
                result = monitor(model, validation, config, device, rank, world)
                result.update(step=step, model_id=model_id, monitor_id=identity['monitor_id'])
            updated, accepted = validation_transition(state, result, step)
            # Derive the same counters from the pending checkpoint on every retry.
            if accepted:
                payload = snapshot()
                payload['state'] = dict(updated)
                def publish_best():
                    path = directory / 'best' / f'best_{step:08d}.pt'
                    atomic_payload(path, payload)
                    record = json.loads(path.with_suffix('.complete.json').read_text())
                    write_json(directory / 'best.json', dict(record, file=str(path.relative_to(directory))))
                    for old in path.parent.glob('best_*.pt'):
                        if old != path:
                            old.unlink()
                            old.with_suffix('.complete.json').unlink(missing_ok=True)
                on_rank_zero(publish_best, rank)
            on_rank_zero(lambda: write_json(result_file, result), rank)
            state = updated
            save_latest()
            def publish_log():
                rows = [json.loads(p.read_text()) for p in sorted((directory / 'validation').glob('step_*.json'))]
                path = directory / 'validation.jsonl'
                temporary = path.with_suffix('.tmp')
                temporary.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))
                os.replace(temporary, path)
            on_rank_zero(publish_log, rank)
        if state['validation_pending']:
            validate_pending()
        loader_options = dict(num_workers=options['workers'], pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(config['seed']))
        if options['workers']:
            loader_options.update(multiprocessing_context='spawn', prefetch_factor=1)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_groups, **loader_options)
        iterator = iter(loader)
        session_start = time.perf_counter()
        params = list(model.parameters())
        limit = stop_after or budget
        for current in range(start_step, limit):
            if state['lower_count'] >= config['monitoring']['patience']:
                break
            begin = time.perf_counter()
            batch = next(iterator)
            data_seconds = time.perf_counter() - begin
            optimizer.zero_grad(set_to_none=True)
            lr = optimizer.param_groups[0]['lr']
            before = [p.detach().clone() for p in params] if config['smoke'] else None
            planes = batch['planes'].reshape(-1, 18, 19, 19).to(device, dtype=torch.float32,
                memory_format=torch.channels_last if options['channels_last'] else torch.contiguous_format)
            with autocast_context(config, device):
                output = wrapped(planes)
            output = {name: value.reshape(local_groups, 10, 20, *value.shape[1:]) for name, value in output.items()}
            terms = grouped_losses(output, *(batch[name].to(device) for name in
                                   ('rank_id', 'policy_target', 'value_target', 'value_weight')))
            terms['total'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            finite = torch.stack([torch.isfinite(p).all() for p in params]).all()
            losses = torch.stack([terms[k].detach() for k in ('total', 'strength', 'policy', 'value')])
            finite &= torch.isfinite(losses).all()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                dist.all_reduce(losses)
                losses /= world
            if not finite:
                raise FloatingPointError('nonfinite parameters or losses')
            state['optimizer_steps'] = current + 1
            step = current + 1
            event = dict(step=step, lr=lr, next_lr=optimizer.param_groups[0]['lr'], grad_norm=float(grad_norm),
                         data_seconds=data_seconds, seconds=time.perf_counter() - begin,
                         **dict(zip(('total', 'strength', 'policy', 'value'), losses.tolist())))
            if before is not None:
                event['update_norm'] = float(torch.sqrt(sum((p.detach() - old).float().square().sum() for p, old in zip(params, before))))
                event['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
                event['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
                if event['grad_norm'] <= 0 or event['update_norm'] <= 0:
                    raise ValueError('smoke gradient/update must be nonzero')
            if before is not None:
                event['rss_peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                rank_events = [event.copy()]
                if world > 1:
                    rank_events = [None] * world
                    dist.all_gather_object(rank_events, event.copy())
                event['ranks'] = rank_events
            on_rank_zero(lambda: log_event(directory, rank, event), rank)
            stop = torch.tensor(int(stopping or (rank == 0 and stop_file.exists())), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            due = step % config['monitoring']['every'] == 0 or step == budget
            state['validation_pending'] = due
            if due or step % options['checkpoint_every'] == 0 or step == limit or stop.item():
                save_started = time.perf_counter()
                save_latest()
                on_rank_zero(lambda: write_json(directory / 'checkpoint_metrics.json',
                    dict(step=step, seconds=time.perf_counter() - save_started,
                         bytes=sum(p.stat().st_size for p in (directory / 'checkpoints').glob('*.pt')))), rank)
            if stop.item():
                stopping = True
                break
            if due:
                validate_pending()
        step = state['optimizer_steps']
        complete = not stopping and (state['lower_count'] >= config['monitoring']['patience'] or step == budget)
        reason = 'patience' if state['lower_count'] >= config['monitoring']['patience'] else 'budget'
        if complete:
            def finalize():
                record = json.loads((directory / 'best.json').read_text())
                path = directory / record['file']
                if file_hash(path) != record['sha256']:
                    raise ValueError('best checksum mismatch')
                payload = load_artifact(path, config)
                if (payload['state']['best_step'] != state['best_step'] or payload['dataset_id'] != manifest['dataset_id']
                        or payload['code_sha256'] != code):
                    raise ValueError('best state/data/code mismatch')
                payload = dict(payload, best_step=state['best_step'], stopped_step=step, stop_reason=reason,
                               monitor_score=state['best_score'])
                atomic_payload(directory / 'model.pt', payload)
                load_artifact(directory / 'model.pt', config)
            on_rank_zero(finalize, rank)
        summary = dict(status='completed' if complete else 'paused', completed_updates=step,
                       updates_this_session=step - start_step, best_step=state['best_step'],
                       monitor_score=state['best_score'], stop_reason=reason if complete else None,
                       seconds=time.perf_counter() - session_start)
        summary['gpu_hours'] = summary['seconds'] * world / 3600 if device.type == 'cuda' else 0
        on_rank_zero(lambda: write_json(directory / 'training_summary.json', summary), rank)
        return summary
    finally:
        for number, handler in handlers.items():
            signal.signal(number, handler)
        for data in (dataset, validation):
            if data is not None:
                data.close()
        if locked and rank == 0:
            lock.unlink(missing_ok=True)
        if dist.is_initialized() and not keep_distributed:
            dist.destroy_process_group()


def main(argv=None):
    from .evaluate import evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    try:
        summary = train(config, resume=args.resume, stop_after=args.stop_after, keep_distributed=True)
        # Smoke measures monitoring only; final full inference belongs to the production run.
        if summary['status'] == 'completed' and not config['smoke']:
            evaluate(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
