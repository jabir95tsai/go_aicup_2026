"""Global game permutations shared by all DDP ranks, reconstructed from step."""
import numpy as np

from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler as RandomGroupSampler


class GroupBatchSampler(RandomGroupSampler):
    """Visit every game once per epoch while retaining ten-rank/twenty-ply groups."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        counts = [len(rows) for rows in self.rows_by_rank]
        if len(set(counts)) != 1 or counts[0] % self.groups_per_update:
            raise ValueError('equal rank counts must be divisible by groups_per_update; no games may be dropped')
        games = dataset._indices[0]
        if len(np.unique(games)) != len(games):
            raise ValueError('training game indices must be unique')
        if np.any(np.diff(self.offsets) < 20):
            raise ValueError('every training game needs twenty distinct effective positions')
        self.steps_per_epoch = counts[0] // self.groups_per_update

    def __iter__(self):
        local_groups = self.groups_per_update // self.world_size
        begin = self.rank * local_groups
        previous_epoch = None
        for step in range(self.start_step, self.updates):
            epoch, within = divmod(step, self.steps_per_epoch)
            if epoch != previous_epoch:
                permutations = [np.random.default_rng(np.random.SeedSequence(
                    [self.seed, epoch, level, 1])).permutation(rows)
                    for level, rows in enumerate(self.rows_by_rank)]
                previous_epoch = epoch
            tasks = []
            for group in range(begin, begin + local_groups):
                rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, group, 2]))
                for rows in permutations:
                    row = rows[within * self.groups_per_update + group]
                    start, stop = self.offsets[row:row + 2]
                    indices = start + rng.choice(int(stop - start), size=20, replace=False)
                    tasks.extend((int(index), step) for index in indices)
            yield tasks


def collate_groups(samples):
    """Validate complete ranking groups and retain [groups,10,20,...] axes."""
    from torch.utils.data import default_collate
    if not samples or len(samples) % 200:
        raise ValueError('batch must contain complete ten-rank/twenty-position groups')
    if any('value_target' not in sample or 'value_weight' not in sample for sample in samples):
        raise ValueError('group batch requires teacher labels for every position')
    batch = default_collate(samples)
    batch = {name: tensor.reshape(-1, 10, 20, *tensor.shape[1:]) for name, tensor in batch.items()}
    import torch
    if not torch.all(batch['rank_id'] == torch.arange(10).reshape(1, 10, 1)):
        raise ValueError('group ranks must be ordered 0..9')
    for name in ('game_index', 'player_color'):
        if not torch.all(batch[name] == batch[name][..., :1]):
            raise ValueError('twenty positions must share a game and player color')
    if torch.any(batch['ply'].sort(dim=-1).values.diff(dim=-1) == 0):
        raise ValueError('twenty positions must be distinct')
    if (not torch.isfinite(batch['value_target']).all() or torch.any(batch['value_target'].abs() > 1)
            or not torch.all(batch['value_weight'] == 1)):
        raise ValueError('invalid teacher value or weight')
    return batch


def validate_prepared(config, *, teacher=True):
    """Validate and reuse existing SE indices without migrating their identity."""
    import copy
    import json
    from pathlib import Path
    from go_aicup.artifacts import file_hash
    from go_aicup.tasks.strength.reproduce_paper.prepare import prepare
    from go_aicup.tasks.strength.reproduce_paper.questions import prepare_questions
    from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
    directory = Path(config['paths']['dataset_dir'])
    if not (directory / 'manifest.json').is_file() or not (directory / 'questions_manifest.json').is_file():
        raise ValueError('existing SE prepared indices and fixed questions are required; no regeneration')
    legacy = copy.deepcopy(config)
    legacy['positions']['positions_per_rank'] = 7
    prepare(legacy)
    prepare_questions(legacy)
    record = json.loads((directory / 'manifest.json').read_text())
    for split in ('train', 'valid'):
        offsets = np.load(directory / f'{split}_offsets.npy', mmap_mode='r')
        games = np.load(directory / f'{split}_game.npy', mmap_mode='r')
        short = np.flatnonzero(np.diff(offsets) < 20)
        if len(short):
            raise ValueError(f'{split} game_index={int(games[short[0]])}: needs twenty effective positions')
    if teacher:
        data = PaperDataset(directory, cache_dir=config['paths']['cache_dir'], label_dir=config['paths']['label_dir'])
        try:
            data._ensure_open()
            for begin in range(0, len(data), 1_000_000):
                values = data._values[begin:begin + 1_000_000]
                if not np.isfinite(values).all() or np.any(np.abs(values) > 1):
                    raise ValueError('invalid teacher value')
        finally:
            data.close()
    identity = dict(prepared_manifest_sha256=file_hash(directory / 'manifest.json'),
                    games_sha256=file_hash(directory / 'games.parquet'),
                    questions_sha256=file_hash(directory / 'validation_questions.parquet'),
                    available_positions=record['identity']['positions'], sampled_positions_per_game=20)
    if teacher:
        identity['value_sha256'] = file_hash(Path(config['paths']['label_dir']) / 'value.npy')
    return identity
