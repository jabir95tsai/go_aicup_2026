"""Independent rank streams; durable state advances only on consumed updates."""
import copy
import numpy as np
from go_aicup.artifacts import object_hash


class GroupBatchSampler:
    def __init__(self, dataset, *, groups_per_update=128, microbatch_groups=1,
                 updates=100, start_step=0, rank=0, world_size=1, state=None):
        self.seed = dataset.seed
        self.rows = [np.flatnonzero(dataset.games.rank_id.to_numpy() == r).tolist() for r in range(10)]
        self.groups_per_update, self.microbatch_groups = groups_per_update, microbatch_groups
        self.updates, self.start_step, self.rank, self.world_size = updates, start_step, rank, world_size
        if (world_size < 1 or not 0 <= rank < world_size or groups_per_update < 1
                or groups_per_update % world_size or microbatch_groups < 1
                or (groups_per_update // world_size) % microbatch_groups
                or not 0 <= start_step <= updates):
            raise ValueError('invalid sampler batch/distribution/update configuration')
        if any(len(rows) < groups_per_update for rows in self.rows):
            raise ValueError('each rank needs at least global groups_per_update unique games')
        if not dataset.games.game_index.is_unique:
            raise ValueError('duplicate training games')
        self._permutations = {}
        self.identity = object_hash(dict(seed=self.seed, groups=groups_per_update,
                                        games=dataset.games.game_index.tolist(), ranks=dataset.games.rank_id.tolist()))
        self._durable = dict(identity=self.identity, completed_step=0,
                             ranks=[dict(round=0, cursor=0, pending=[]) for _ in range(10)])
        if state is not None:
            if state.get('identity') != self.identity or state.get('completed_step') != start_step:
                raise ValueError('sampler checkpoint identity or consumed step changed')
            self._durable = copy.deepcopy(state)
        else:
            self.state_dict(start_step)

    def _draw(self, state):
        result = []
        for level, stream in enumerate(state['ranks']):
            selected, seen, deferred = [], set(), []
            old = iter(stream['pending'])
            for row in old:
                if row not in seen:
                    selected.append(row); seen.add(row)
                else:
                    deferred.append(row)
                if len(selected) == self.groups_per_update:
                    deferred.extend(old)
                    break
            while len(selected) < self.groups_per_update:
                cached = self._permutations.get(level)
                if cached is None or cached[0] != stream['round']:
                    cached = (stream['round'], np.random.default_rng([self.seed, level, stream['round'], 57]).permutation(self.rows[level]))
                    self._permutations[level] = cached
                permutation = cached[1]
                while stream['cursor'] < len(permutation) and len(selected) < self.groups_per_update:
                    row = int(permutation[stream['cursor']]); stream['cursor'] += 1
                    if row in seen:
                        deferred.append(row)
                    else:
                        selected.append(row); seen.add(row)
                if stream['cursor'] == len(permutation):
                    stream['round'] += 1; stream['cursor'] = 0
            stream['pending'] = deferred
            result.append(selected)
        state['completed_step'] += 1
        return result

    def state_dict(self, completed_step=None):
        completed_step = self._durable['completed_step'] if completed_step is None else completed_step
        if completed_step < self._durable['completed_step'] or completed_step > self.updates:
            raise ValueError('sampler state must follow consumed optimizer updates')
        while self._durable['completed_step'] < completed_step:
            self._draw(self._durable)
        return copy.deepcopy(self._durable)

    def __len__(self):
        return (self.updates - self.start_step) * (self.groups_per_update // self.world_size // self.microbatch_groups)

    def __iter__(self):
        state = self.state_dict(self.start_step)
        local = self.groups_per_update // self.world_size
        for step in range(self.start_step, self.updates):
            selected = self._draw(state)
            begin = self.rank * local
            for group in range(begin, begin + local, self.microbatch_groups):
                yield [(selected[level][g], step)
                       for g in range(group, group + self.microbatch_groups) for level in range(10)]
