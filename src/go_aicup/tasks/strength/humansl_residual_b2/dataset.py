"""Selected-ply KataGo inputs: real history, bounded output, fixed rank_1d."""
import os
from pathlib import Path
import numpy as np
from go_aicup.features.humansl_torch import _katago, profile_metadata
from go_aicup.tasks.strength.shared.dataset import apply_d4, d4_id
from go_aicup.tasks.strength.reproduce_paper.dataset import transform_policy
from .data import MoveStore
from .sampler import GroupBatchSampler


def effective_plies(colors, player_color, minimum=30):
    colors = np.asarray(colors)
    if player_color not in (0, 1) or not np.isin(colors, [0, 1]).all():
        raise ValueError('invalid actual move or designated-player color')
    plies = np.arange(minimum, len(colors), dtype=np.int64)
    return plies[colors[plies] == player_color], plies[colors[plies] != player_color]


def encode_selected(moves, colors, plies, model_config, hd, *, transform=0):
    """Replay the prefix, calling KataGo feature encoding only at selected turns."""
    moves, colors, plies = np.asarray(moves), np.asarray(colors), np.asarray(plies, dtype=np.int64)
    if (hd not in (0, 1) or len(moves) != len(colors) or len(plies) == 0
            or np.any(plies < 0) or np.any(plies >= len(moves)) or len(np.unique(plies)) != len(plies)
            or np.any(moves[plies] < 0) or np.any(moves[plies] >= 361)):
        raise ValueError('unsupported HD, missing position, or pass gather')
    board, features, gamestate, _ = _katago()
    encoder = features.Features(model_config, 19)
    rules = dict(gamestate.GameState.RULES_JAPANESE, whiteKomi=6.5 if hd == 0 else .5)
    state = gamestate.GameState(19, rules)
    metadata = profile_metadata('rank_1d')
    wanted = {int(p): i for i, p in enumerate(plies)}
    outputs = [None] * len(plies)
    for ply in range(int(plies.max()) + 1):
        color = int(colors[ply])
        if color not in (0, 1):
            raise ValueError(f'invalid actual color at ply {ply}')
        pla = board.Board.BLACK if color == 0 else board.Board.WHITE
        if state.board.pla != pla:
            raise ValueError(f'move color disagrees with replay at ply {ply}')
        move = int(moves[ply])
        loc = board.Board.PASS_LOC if move == 361 else state.board.loc(move % 19, move // 19)
        if ply in wanted:
            spatial, global_inputs = state.get_input_features(encoder)
            action = int(encoder.loc_to_tensor_pos(loc, state.board))
            if not 0 <= action < 361:
                raise ValueError('classification gather requires a board action')
            x = spatial[0].astype(np.uint8)
            if transform:
                x = apply_d4(x, transform); action = transform_policy(action, transform)
            outputs[wanted[ply]] = dict(spatial=x, global_inputs=global_inputs[0].astype(np.float32),
                                       metadata=metadata.get_metadata_row(nextPlayer=pla, boardArea=361).astype(np.float32),
                                       actions=action)
        state.play(pla, loc)
    return {key: np.asarray([output[key] for output in outputs], dtype=np.int64 if key == 'actions' else None)
            for key in outputs[0]}


class SelectedDataset:
    def __init__(self, config, games, model_config, *, teacher_required=True, augment=True):
        self.config, self.games = config, games.reset_index(drop=True).copy()
        self.model_config, self.teacher_required, self.augment = model_config, teacher_required, augment
        self.seed, self.positions = config['seed'], config['training']['positions_per_side']
        self._store = self._teacher = None; self._pid = None
        if self.positions != 15 or not self.games.game_index.is_unique:
            raise ValueError('fifteen positions per side and unique games are required')
    def __len__(self):
        return len(self.games)
    def __getstate__(self):
        state = self.__dict__.copy(); state.update(_store=None, _teacher=None, _pid=None)
        return state
    def _open(self):
        if self._pid != os.getpid():
            self._store = MoveStore(self.config['data']['store_dir'])
            self._teacher = None; self._pid = os.getpid()
            if self.teacher_required:
                from .teacher import TeacherLabels
                self._teacher = TeacherLabels(self.config)
    def close(self):
        self._store = self._teacher = None; self._pid = None
    def _encode(self, row, plies, step=0):
        self._open(); game = self.games.iloc[row]
        moves, colors = self._store.game(int(game.store_index))
        plies = np.asarray(plies, np.int64)
        transform = d4_id(self.seed, step, int(game.game_index)) if self.augment else 0
        sample = encode_selected(moves, colors, plies, self.model_config, int(game.hd), transform=transform)
        sample.update(sides=(colors[plies] != game.player_color).astype(np.int64),
                      labels=np.full(len(plies), game.rank_id, np.int64),
                      game_index=np.full(len(plies), game.game_index, np.int64), ply=plies)
        if self.teacher_required:
            sample['values'] = self._teacher.values(game, plies)
        return sample
    def __getitem__(self, task):
        row, step = task[:2]; self._open(); game = self.games.iloc[row]
        _, colors = self._store.game(int(game.store_index))
        if len(task) == 3:
            plies = task[2]
        else:
            own, opp = effective_plies(colors, int(game.player_color))
            if min(len(own), len(opp)) < self.positions:
                raise ValueError('fewer than fifteen valid positions per side')
            rng = np.random.default_rng([self.seed, int(step), int(game.game_index), 71])
            plies = np.concatenate([rng.choice(side, self.positions, replace=False) for side in (own, opp)])
        return self._encode(row, plies, step)
    def __getitems__(self, tasks):
        return [self[task] for task in tasks]
    def iter_positions(self, row, batch_size=300, min_ply=30):
        if batch_size < 1 or min_ply < 0:
            raise ValueError('invalid inference batch or minimum ply')
        game = self.games.iloc[row]
        for start in range(min_ply, int(game.length), batch_size):
            yield self._encode(row, np.arange(start, min(start + batch_size, int(game.length))))
    def all_positions(self, row):
        return self._encode(row, np.arange(30, int(self.games.iloc[row].length)))


def collate_groups(samples):
    import torch
    return {key: torch.from_numpy(np.concatenate([sample[key] for sample in samples], axis=0))
            for key in samples[0]}
