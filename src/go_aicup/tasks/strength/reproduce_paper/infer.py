"""Measure eight-GPU inference batch sizes on identical effective positions."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import (
    autocast_context, setup_distributed, on_rank_zero, barrier)
from go_aicup.tasks.strength.shared.training import configure_torch, environment
from .config import read_config, validate_config
from .dataset import PaperDataset
from .model import StrengthEstimator
from .train import synchronize, process_stats

DEFAULT_CONFIG = ROOT / 'configs/strength/se_infer_batch_benchmark.json'


@torch.inference_mode()
def infer_positions(model, dataset, config, device, *, batch_size, rank=0, world=1,
                    positions_per_rank=None, warmup_batches=0, channels_last=False, collect_resources=True):
    """Contiguous sample-index partitions, no padding, labels, D4 or BN updates.

    Return scalar scores in sample-index order; caller may save each rank's
    range as a shard. No collectives here, including empty partitions and OOM.
    """
    if (type(batch_size) is not int or batch_size < 1 or not 0 <= rank < world
            or type(warmup_batches) is not int or warmup_batches < 0
            or (positions_per_rank is not None and
                (type(positions_per_rank) is not int or positions_per_rank < 1))):
        raise ValueError('invalid inference batch, partition, warmup or position limit')
    if dataset.augment:
        raise ValueError('inference requires augment=False')
    start = len(dataset) * rank // world
    stop = len(dataset) * (rank + 1) // world
    if positions_per_rank is not None:
        stop = min(stop, start + positions_per_rank)
    setup_start = time.perf_counter()
    options = dict(num_workers=config['training']['workers'], pin_memory=device.type == 'cuda',
                   generator=torch.Generator().manual_seed(config['seed']))
    if options['num_workers']:
        options.update(multiprocessing_context='spawn', prefetch_factor=1)
    batches = [range(begin, min(begin + batch_size, stop)) for begin in range(start, stop, batch_size)]
    loader = DataLoader(dataset, batch_sampler=batches, **options)
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in getattr(iterator, '_workers', [])]
    was_training = model.training
    model.eval()
    scores = np.empty(stop - start, dtype=np.float32)
    before_io = process_stats(worker_pids) if collect_resources else dict(rss_bytes=0, read_bytes=0, write_bytes=0, rchar=0, wchar=0)
    peak_rss = before_io['rss_bytes']
    data_seconds = compute_seconds = 0.0
    steps = 0
    after_io = before_io
    if collect_resources and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        first_start = time.perf_counter()
        first = next(iterator, None)
        first_data_seconds = time.perf_counter() - first_start
        def forward(batch):
            planes = batch['planes'].to(device, dtype=torch.float32, non_blocking=True,
                memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            with autocast_context(config, device):
                result = model(planes)['score'].float()
            if not torch.isfinite(result).all():
                raise FloatingPointError('nonfinite inference scores')
            return result.cpu().numpy()
        if first is not None:
            for _ in range(warmup_batches):
                forward(first)
        synchronize(device)
        setup_seconds = time.perf_counter() - setup_start
        tick = time.perf_counter()
        batch = first
        del first
        cursor = 0
        while batch is not None:
            compute_start = time.perf_counter()
            values = forward(batch)
            compute_seconds += time.perf_counter() - compute_start
            count = len(values)
            expected = torch.arange(start + cursor, start + cursor + count)
            if not torch.equal(batch['sample_index'], expected):
                raise ValueError('inference sample order mismatch')
            scores[cursor:cursor + count] = values
            cursor += count
            steps += 1
            if collect_resources:
                after_io = process_stats(worker_pids)
                peak_rss = max(peak_rss, after_io['rss_bytes'])
            data_start = time.perf_counter()
            batch = next(iterator, None)
            data_seconds += time.perf_counter() - data_start
        synchronize(device)
        elapsed = time.perf_counter() - tick
        if cursor != stop - start:
            raise ValueError('incomplete inference coverage')
        report = dict(status='ok', rank=rank, sample_start=start, sample_stop=stop,
                      positions=len(scores), batches=steps, batch_size=batch_size,
                      setup_seconds=setup_seconds, first_data_seconds=first_data_seconds,
                      measured_seconds=elapsed, data_seconds=data_seconds, compute_seconds=compute_seconds,
                      rss_bytes=peak_rss, io_delta={key: after_io[key] - before_io[key]
                      for key in ('read_bytes', 'write_bytes', 'rchar', 'wchar')},
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if collect_resources and device.type == 'cuda' else 0,
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if collect_resources and device.type == 'cuda' else 0)
        return scores, report
    finally:
        # Release each candidate's loader before the next batch-size measurement.
        del iterator, loader
        model.train(was_training)


def benchmark(config):
    validate_config(config)
    options = config['inference']
    sizes = options['per_gpu_batch_sizes']
    if (not sizes or any(type(size) is not int or size < 1 for size in sizes)
            or sizes != sorted(set(sizes))):
        raise ValueError('per_gpu_batch_sizes must be positive, unique and increasing')
    if (type(options['positions_per_rank']) is not int or options['positions_per_rank'] < sizes[-1]
            or options['split'] not in ('train', 'valid')
            or type(options['warmup_batches']) is not int or options['warmup_batches'] < 1):
        raise ValueError('invalid inference window, split or warmup')
    if config['precision'] not in ('fp32', 'bf16') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('inference benchmark supports fp32/bf16 and cpu/cuda')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA inference benchmark requires exactly eight GPUs via torchrun')
    configure_torch(config)
    runtime = dict(config, training=dict(config['training'], global_batch_size=8),
                   evaluation=dict(global_batch_size=8))
    device, rank, world = setup_distributed(runtime)
    directory = Path(config['paths']['output_dir'])
    dataset = None
    locked = False
    lock = directory / '.writer.lock'
    started = time.perf_counter()
    try:
        def initialize():
            directory.mkdir(parents=True, exist_ok=True)
            if any(directory.iterdir()):
                raise ValueError('nonempty inference output; choose a new directory')
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(descriptor)
        on_rank_zero(initialize, rank)
        locked = True
        checkpoint_arg = Path(options['checkpoint'])
        if not checkpoint_arg.is_absolute():
            checkpoint_arg = ROOT / checkpoint_arg
        checkpoint = resolve_checkpoint(checkpoint_arg)
        checkpoint_sha256 = file_hash(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        dataset = PaperDataset(config['paths']['dataset_dir'], split=options['split'],
                               cache_dir=config['paths']['cache_dir'], augment=False,
                               max_open_shards=config['loader']['max_open_shards'])
        dataset._ensure_open()
        if (any(payload['code_sha256'].get(name) != file_hash(Path(__file__).with_name(name))
                for name in ('model.py', 'dataset.py'))
                or payload['config']['model'] != config['model']
                or dataset.record['identity']['split'] != config['split']
                or dataset.record['identity']['positions'] != config['positions']):
            raise ValueError('checkpoint/model/prepared configuration mismatch')
        identity_directory = (checkpoint.parent if (checkpoint.parent / 'data_identity.json').is_file()
                              else checkpoint.parent.parent)
        identity = json.loads((identity_directory / 'data_identity.json').read_text())
        if (object_hash(identity) != payload['dataset_id']
                or identity['prepared_manifest_sha256'] != file_hash(dataset.directory / 'manifest.json')
                or identity['games_sha256'] != file_hash(dataset.directory / 'games.parquet')):
            raise ValueError('checkpoint/prepared data identity mismatch')
        model = StrengthEstimator(**config['model']).to(device)
        model.load_state_dict(payload['model'])
        del payload
        if options['channels_last']:
            model = model.to(memory_format=torch.channels_last)
        def metadata():
            write_json(directory / 'config.json', config)
            write_json(directory / 'environment.json', environment(device, world))
        on_rank_zero(metadata, rank)
        code = {name: file_hash(Path(__file__).with_name(name)) for name in
                ('infer.py', 'model.py', 'dataset.py', 'config.py', 'train.py')}
        reference = None
        outcomes = []
        for size in sizes:
            barrier()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            try:
                scores, report = infer_positions(model, dataset, config, device, batch_size=size,
                    rank=rank, world=world, positions_per_rank=options['positions_per_rank'],
                    warmup_batches=options['warmup_batches'], channels_last=options['channels_last'])
                if reference is None:
                    reference = scores.copy()
                    np.save(directory / f'reference_rank_{rank:02d}.npy', reference, allow_pickle=False)
                    report.update(max_abs_error=0.0, relative_l2_error=0.0)
                else:
                    delta = scores.astype(np.float64) - reference
                    report.update(max_abs_error=float(np.max(np.abs(delta))) if len(delta) else 0.0,
                                  relative_l2_error=float(np.linalg.norm(delta) /
                                      max(float(np.linalg.norm(reference.astype(np.float64))), 1e-30)))
                del scores
            except torch.cuda.OutOfMemoryError:
                report = dict(status='oom', rank=rank, batch_size=size)
                torch.cuda.empty_cache()
            reports = [None] * world
            if world > 1:
                dist.all_gather_object(reports, report)
            else:
                reports[0] = report
            success = all(item['status'] == 'ok' for item in reports)
            outcome = dict(batch_size=size, status='ok' if success else 'oom', ranks=reports)
            if success:
                count = sum(item['positions'] for item in reports)
                elapsed = max(item['measured_seconds'] for item in reports)
                end_to_end = max(item['measured_seconds'] + item['setup_seconds'] for item in reports)
                outcome.update(positions=count, positions_per_second=count / elapsed,
                               setup_inclusive_positions_per_second=count / end_to_end,
                               max_abs_error=max(item['max_abs_error'] for item in reports),
                               peak_cuda_allocated_bytes=max(item['peak_cuda_allocated_bytes'] for item in reports),
                               peak_cuda_reserved_bytes=max(item['peak_cuda_reserved_bytes'] for item in reports))
            outcomes.append(outcome)
            summary = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha256,
                           split=options['split'], prepared_identity=identity, code_sha256=code,
                           world_size=world, precision=config['precision'], channels_last=options['channels_last'],
                           candidates=outcomes, wall_seconds=time.perf_counter() - started,
                           note='Identical sample ranges, eval mode, no augmentation. First batch/worker startup and repeated first-batch warmup are setup costs; both setup-inclusive and measured throughput are reported. Later candidates benefit from cache. Score differences are reported, not automatically accepted.')
            on_rank_zero(lambda: write_json(directory / 'benchmark.json', summary), rank)
            if not success:
                break
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
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    benchmark(read_config(args.config))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
