"""Global game permutations shared by all DDP ranks, reconstructed from step."""
import numpy as np

from go_aicup.tasks.strength.reproduce_paper.dataset import GroupBatchSampler as RandomGroupSampler


class GroupBatchSampler(RandomGroupSampler):
    """Visit every game once per epoch while retaining ten-rank/seven-ply groups."""

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        counts = [len(rows) for rows in self.rows_by_rank]
        if len(set(counts)) != 1 or counts[0] % self.groups_per_update:
            raise ValueError('equal rank counts must be divisible by groups_per_update; no games may be dropped')
        games = dataset._indices[0]
        if len(np.unique(games)) != len(games):
            raise ValueError('training game indices must be unique')
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
                    indices = start + rng.choice(int(stop - start), size=7, replace=False)
                    tasks.extend((int(index), step) for index in indices)
            yield tasks
