"""Reuse SE reports/inference while retaining the process group between stages."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.shared.runtime import setup_distributed, on_rank_zero, barrier
from go_aicup.tasks.strength.shared.training import configure_torch
from go_aicup.tasks.strength.reproduce_paper import evaluate as base


def evaluate(config):
    base.validate_config(config)
    batch_size = config['evaluation']['per_gpu_batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('evaluation per_gpu_batch_size must be positive')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA evaluation requires eight processes')
    configure_torch(config)
    runtime = dict(config, training=dict(config['training'], global_batch_size=8),
                   evaluation=dict(global_batch_size=8))
    owns_group = not dist.is_initialized()
    if owns_group:
        device, rank, world = setup_distributed(runtime)
    else:
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if config['device'] == 'cuda' else torch.device('cpu')
    directory = Path(config['paths']['output_dir']) / 'evaluation'
    lock = base.exclusive(directory)
    locked = False
    dataset = None
    try:
        on_rank_zero(lambda: lock.__enter__(), rank)
        locked = True
        on_rank_zero(lambda: base.prepare_questions(config), rank)
        identity, payload = base.source_identity(config, world)
        def initialize():
            path = directory / 'identity.json'
            if path.exists():
                if json.loads(path.read_text()) != identity:
                    raise ValueError('evaluation identity changed; preserve existing output')
            else:
                if any(path.name != '.writer.lock' for path in directory.iterdir()):
                    raise ValueError('incomplete evaluation output without identity')
                write_json(path, identity)
        on_rank_zero(initialize, rank)
        model = base.StrengthEstimator(**config['model']).to(device)
        model.load_state_dict(payload['model'])
        del payload
        if identity['channels_last']:
            model = model.to(memory_format=torch.channels_last)
        for split in ('train', 'valid'):
            dataset = base.PaperDataset(config['paths']['dataset_dir'], split=split,
                cache_dir=config['paths']['cache_dir'], augment=False,
                max_open_shards=config['loader']['max_open_shards'])
            dataset._ensure_open()
            path = directory / f'{split}_scores_rank{rank:02d}.npy'
            if path.with_suffix('.json').exists():
                base.load_shard(directory, split, rank, world, len(dataset), identity)
            else:
                scores, report = base.infer_positions(model, dataset, config, device, batch_size=batch_size,
                    rank=rank, world=world, channels_last=identity['channels_last'], collect_resources=False)
                base.save_array(path, scores)
                write_json(path.with_suffix('.json'), dict(identity=object_hash(identity),
                    start=report['sample_start'], stop=report['sample_stop'], sha256=file_hash(path)))
                del scores
            dataset.close()
            dataset = None
            barrier()
        return on_rank_zero(lambda: base.build_reports(config, identity), rank)
    finally:
        if dataset is not None:
            dataset.close()
        if locked:
            on_rank_zero(lambda: lock.__exit__(None, None, None), rank)
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()
