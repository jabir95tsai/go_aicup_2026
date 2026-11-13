"""Eight-GPU AdamW continuation, exhaustive game epochs, and two full evaluations."""
import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint, save_checkpoint
from go_aicup.tasks.strength.shared.runtime import autocast_context, setup_distributed, on_rank_zero, barrier
from go_aicup.tasks.strength.shared.training import (
    configure_torch, seed_process, environment, restore_rng, log_event)
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset, collate_groups
from go_aicup.tasks.strength.reproduce_paper.model import StrengthEstimator
from go_aicup.tasks.strength.reproduce_paper.losses import grouped_losses
from go_aicup.tasks.strength.reproduce_paper.train import (
    finalize_training, trim_training_log)
from .config import DEFAULT_CONFIG, read_config, validate_config
from .dataset import GroupBatchSampler

BASE = Path(__file__).parent.parent / 'reproduce_paper'


def code_fingerprints():
    code = {name: file_hash(BASE / name) for name in
            ('model.py', 'dataset.py', 'losses.py', 'train.py', 'config.py')}
    for name in ('tasks/strength/shared/runtime.py', 'tasks/strength/shared/training.py',
                 'tasks/strength/shared/checkpoint.py', 'tasks/strength/shared/dataset.py',
                 'data/board_cache.py', 'artifacts.py'):
        code[name] = file_hash(ROOT / 'src/go_aicup' / name)
    for name in ('train.py', 'config.py', 'dataset.py', 'evaluate.py'):
        code['reproduce_paper_2/' + name] = file_hash(Path(__file__).with_name(name))
    return code


def scheduler_for(optimizer, training):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda completed:
        1.0 if completed < training['lr_drop_after_updates'] else training['lr_after_drop'] / training['lr'])


def load_source(config, manifest, world):
    """Load only the pinned parent; its SGD state is deliberately discarded."""
    path = resolve_checkpoint(Path(config['paths']['source_checkpoint']))
    if file_hash(path) != config['source']['sha256']:
        raise ValueError('source checkpoint checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['state']['optimizer_steps'] != config['source']['optimizer_steps']
            or payload['dataset_id'] != manifest['dataset_id']
            or payload['cache_build_id'] != manifest['cache_build_id']
            or payload['world_size'] != world or payload['config']['training']['optimizer'] != 'sgd'
            or payload['config_sha256'] != object_hash(payload['config'])
            or any(payload['config'][key] != config[key] for key in
                   ('model', 'split_seed', 'seed', 'positions', 'split', 'teacher', 'loss_weights'))):
        raise ValueError('source checkpoint config/data/world/step mismatch')
    current = code_fingerprints()
    if any(current.get(name) != digest for name, digest in payload['code_sha256'].items()):
        raise ValueError('source checkpoint code mismatch')
    return payload


def milestone_config(config):
    result = copy.deepcopy(config)
    step = config['training']['lr_drop_after_updates']
    result['training']['updates'] = step
    result['paths']['output_dir'] = str(Path(config['paths']['output_dir']) / 'milestones' / f'update_{step:08d}')
    return result


def publish_milestone(config, model, manifest, code, identity):
    snapshot = milestone_config(config)
    directory = Path(snapshot['paths']['output_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    step = snapshot['training']['updates']
    if (directory / 'model.complete.json').exists():
        payload = torch.load(resolve_checkpoint(directory / 'model.pt'), map_location='cpu', weights_only=False)
        if object_hash(payload['config']) != object_hash(snapshot) or payload['optimizer_steps'] != step:
            raise ValueError('milestone configuration mismatch')
        return
    write_json(directory / 'data_identity.json', identity)
    write_json(directory / 'config.json', snapshot)
    trim_training_log(directory, 0)
    source_log = Path(config['paths']['output_dir']) / 'train.jsonl'
    if source_log.exists():
        (directory / 'train.jsonl').write_bytes(source_log.read_bytes())
    finalize_training(directory, model, snapshot, manifest, code, step)


def cleanup_training(directory, config):
    """Remove latest-only resume state after verifying the published final model."""
    path = resolve_checkpoint(directory / 'model.pt')
    record = json.loads(path.with_suffix('.complete.json').read_text())
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (record.get('status') != 'completed'
            or payload.get('artifact_format') != 'strength_estimator_1'
            or payload['optimizer_steps'] != config['training']['updates']
            or object_hash(payload['config']) != object_hash(config)):
        raise ValueError('final model/config mismatch; refusing checkpoint cleanup')
    checkpoints = directory / 'checkpoints'
    if checkpoints.exists():
        shutil.rmtree(checkpoints)
    for name in ('last.json', 'last.json.tmp', '.stop-requested', 'train.tmp',
                 'model.pt.tmp', 'model.complete.json.tmp'):
        (directory / name).unlink(missing_ok=True)


def train(config, *, resume=None, stop_after=None, keep_distributed=False):
    """Train to the next evaluation boundary; all cursors count AdamW updates."""
    validate_config(config)
    options = config['training']
    budget = options['updates']
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= budget):
        raise ValueError('stop_after must be within the new update budget')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA training requires exactly eight GPUs')
    configure_torch(config)
    runtime = dict(config, training=dict(options, global_batch_size=options['groups_per_update']),
                   evaluation=dict(global_batch_size=options['groups_per_update']))
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if config['device'] == 'cuda' else torch.device('cpu')
    else:
        device, rank, world = setup_distributed(runtime)
    directory = Path(config['paths']['output_dir'])
    lock = directory / '.writer.lock'
    stop_file = directory / '.stop-requested'
    locked = False
    dataset = None
    handlers = {}
    stopping = False
    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
    try:
        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            handlers[signum] = signal.signal(signum, request_stop)
        def initialize():
            directory.mkdir(parents=True, exist_ok=True)
            if resume is None and any(directory.iterdir()) and not (directory / 'model.complete.json').exists():
                raise ValueError('nonempty output; use --resume with this run latest checkpoint')
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, 'w') as handle:
                handle.write(str(os.getpid()))
            if not keep_distributed:
                stop_file.unlink(missing_ok=True)
        on_rank_zero(initialize, rank)
        locked = True
        complete = (directory / 'model.complete.json').is_file()
        if complete:
            on_rank_zero(lambda: cleanup_training(directory, config), rank)
            return dict(status='completed', completed_updates=budget,
                        cumulative_updates=config['source']['optimizer_steps'] + budget, updates_this_session=0)
        dataset = PaperDataset(config['paths']['dataset_dir'], augment=True, seed=config['seed'],
            cache_dir=config['paths']['cache_dir'], label_dir=config['paths']['label_dir'],
            max_open_shards=config['loader']['max_open_shards'])
        if (dataset.record['identity']['split'] != config['split']
                or dataset.record['identity']['positions'] != config['positions']):
            raise ValueError('prepared data/config mismatch')
        dataset._ensure_open()
        identity = dict(prepared_manifest_sha256=file_hash(dataset.directory / 'manifest.json'),
                        games_sha256=file_hash(dataset.directory / 'games.parquet'),
                        value_sha256=file_hash(dataset.label_dir / 'value.npy'))
        manifest = dict(dataset_id=object_hash(identity), cache_build_id=dataset.record['identity']['cache_build_id'])
        code = code_fingerprints()
        seed_process(config['seed'])
        model = StrengthEstimator(**config['model']).to(device)
        if options['sync_batch_norm'] and world > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if options.get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], betas=tuple(options['betas']),
                                      eps=options['eps'], weight_decay=options['weight_decay'])
        scheduler = scheduler_for(optimizer, options)
        scaler = torch.amp.GradScaler(device.type, enabled=False)
        step = 0
        if resume is None:
            payload = load_source(config, manifest, world)
            model.load_state_dict(payload['model'])
        else:
            checkpoint = resolve_checkpoint(Path(resume))
            if checkpoint.resolve() != resolve_checkpoint(directory / 'last.json').resolve():
                raise ValueError('resume must use this run latest checkpoint')
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            if (payload['config_sha256'] != object_hash(config) or payload['dataset_id'] != manifest['dataset_id']
                    or payload['world_size'] != world or payload['code_sha256'] != code):
                raise ValueError('checkpoint config/data/world/code mismatch')
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            scheduler.load_state_dict(payload['scheduler'])
            step = payload['state']['optimizer_steps']
        boundary = options['lr_drop_after_updates']
        limit = stop_after if stop_after is not None else boundary if step < boundary else budget
        if not 0 <= step <= limit:
            raise ValueError('checkpoint is past requested stop')
        if step < boundary < limit:
            limit = boundary
        snapshot_path = Path(milestone_config(config)['paths']['output_dir'])
        if step == boundary and not (snapshot_path / 'model.complete.json').exists():
            limit = boundary
        wrapped = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else model
        restore_rng(payload['rng_by_rank'][rank], device, model)
        del payload
        sampler = GroupBatchSampler(dataset, groups_per_update=options['groups_per_update'],
            updates=budget, start_step=step, rank=rank, world_size=world)
        def metadata():
            write_json(directory / 'config.json', config)
            write_json(directory / 'data_identity.json', identity)
            write_json(directory / 'source.json', dict(config['source'], file=config['paths']['source_checkpoint'],
                optimizer_reset=True, sampling_reset=True, steps_per_epoch=sampler.steps_per_epoch))
            write_json(directory / f'environment_{time.time_ns()}.json', environment(device, world))
            if resume is not None:
                trim_training_log(directory, step)
        on_rank_zero(metadata, rank)
        loader_options = dict(num_workers=options['workers'], pin_memory=device.type == 'cuda',
                              generator=torch.Generator().manual_seed(config['seed']))
        if options['workers']:
            loader_options.update(multiprocessing_context='spawn', prefetch_factor=1)
        # The sampler may prefetch beyond limit; only completed optimizer steps become durable.
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_groups, **loader_options)
        iterator = iter(loader)
        params = list(model.parameters())
        local_groups = options['groups_per_update'] // world
        micro = min(options['microbatch_groups'], local_groups)
        completed = step
        loss_sum, loss_count = 0.0, 0
        barrier()
        for current in range(step, limit):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            losses = torch.zeros(4, device=device)
            lr = optimizer.param_groups[0]['lr']
            for begin in range(0, local_groups, micro):
                end = min(begin + micro, local_groups)
                weight = (end - begin) / local_groups
                sync = wrapped.no_sync() if world > 1 and end < local_groups else nullcontext()
                with sync:
                    with autocast_context(config, device):
                        planes = batch['planes'][begin:end].reshape(-1, 18, 19, 19).to(device,
                            dtype=torch.float32, non_blocking=True,
                            memory_format=torch.channels_last if options.get('channels_last', False) else torch.contiguous_format)
                        output = wrapped(planes)
                        output = {name: value.reshape(end - begin, 10, 7, *value.shape[1:]) for name, value in output.items()}
                        terms = grouped_losses(output,
                            batch['policy_target'][begin:end].to(device, non_blocking=True),
                            batch['value_target'][begin:end].to(device, dtype=torch.float32, non_blocking=True),
                            batch['value_weight'][begin:end].to(device, non_blocking=True))
                    (terms['total'] * weight).backward()
                losses += torch.stack([terms[key].detach() for key in ('total', 'bt', 'policy', 'value')]) * weight
            torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            finite = torch.stack([torch.isfinite(p).all() for p in params]).all() & torch.isfinite(losses).all()
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                dist.all_reduce(losses)
                losses /= world
            if not finite:
                raise FloatingPointError('nonfinite parameters or loss')
            completed = current + 1
            values = losses.tolist()
            loss_sum += values[0]
            loss_count += 1
            event = dict(step=completed, cumulative_step=config['source']['optimizer_steps'] + completed,
                         lr=lr, next_lr=optimizer.param_groups[0]['lr'],
                         **dict(zip(('total', 'bt', 'policy', 'value'), values)))
            on_rank_zero(lambda: log_event(directory, rank, event), rank)
            stop = torch.tensor(int(stopping or (rank == 0 and stop_file.exists())), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if completed % options['checkpoint_every'] == 0 or completed == limit or stop.item():
                checkpoint_loss = loss_sum / loss_count
                epoch, within = divmod(completed, sampler.steps_per_epoch)
                save_checkpoint(directory, model, optimizer, scheduler, scaler, config, manifest,
                    dict(optimizer_steps=completed, cumulative_steps=event['cumulative_step'], epoch=epoch,
                         next_batch=within, checkpoint_loss=checkpoint_loss, checkpoint_loss_updates=loss_count),
                    device, rank, world, code_sha256=code, tag='adamw_training')
                loss_sum, loss_count = 0.0, 0
            if stop.item():
                stopping = True
                break
        if completed == boundary:
            on_rank_zero(lambda: publish_milestone(config, model, manifest, code, identity), rank)
        complete = completed == budget
        if complete:
            on_rank_zero(lambda: finalize_training(directory, model, config, manifest, code, completed), rank)
        summary = dict(start_step=step, completed_updates=completed, updates_this_session=completed - step,
            cumulative_updates=config['source']['optimizer_steps'] + completed,
            status='completed' if complete else 'stopped' if stopping or completed == stop_after else 'milestone',
            finite_checks_passed=True)
        on_rank_zero(lambda: write_json(directory / 'training_summary.json', summary), rank)
        if complete:
            on_rank_zero(lambda: cleanup_training(directory, config), rank)
        return summary
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        if dataset is not None:
            dataset.close()
        if locked and rank == 0:
            lock.unlink(missing_ok=True)
        if dist.is_initialized() and not keep_distributed:
            dist.destroy_process_group()


def main(argv=None):
    from .evaluate import evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--resume', type=Path, help='This experiment last.json; never the SGD parent')
    parser.add_argument('--stop-after', type=int, help='Stop at this many completed AdamW updates')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    directory = Path(config['paths']['output_dir'])
    snapshot = milestone_config(config)
    try:
        configure_torch(config)
        runtime = dict(config, training=dict(config['training'], global_batch_size=config['training']['groups_per_update']),
                       evaluation=dict(global_batch_size=config['training']['groups_per_update']))
        _, rank, _ = setup_distributed(runtime)
        on_rank_zero(lambda: (directory / '.stop-requested').unlink(missing_ok=True), rank)
        # Retry a failed midpoint evaluation before advancing beyond its durable checkpoint.
        if args.stop_after is None and (Path(snapshot['paths']['output_dir']) / 'model.complete.json').exists():
            evaluate(snapshot)
            if (directory / '.stop-requested').exists():
                return 0
        resume = args.resume
        while True:
            summary = train(config, resume=resume, stop_after=args.stop_after, keep_distributed=True)
            if summary['status'] == 'stopped':
                return 0
            if summary['status'] == 'completed':
                if args.stop_after is None:
                    evaluate(config)
                return 0
            evaluate(snapshot)
            if (directory / '.stop-requested').exists():
                return 0
            resume = directory / 'last.json'
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
