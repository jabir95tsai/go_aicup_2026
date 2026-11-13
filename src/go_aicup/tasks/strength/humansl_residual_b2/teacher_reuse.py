"""Read verified SE3 targets by canonical game and pre-action ply."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from go_aicup.artifacts import file_hash, object_hash


class SE3Values:
    def __init__(self, options, identity):
        directory = Path(options['dataset_dir'])
        value_path = Path(options['value_path'])
        recovery = json.loads(Path(options['recovery_path']).read_text())
        training = json.loads(Path(options['training_identity_path']).read_text())
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (file_hash(manifest_path) != training['prepared_manifest_sha256']
                or file_hash(value_path) != training['value_sha256']
                or object_hash(manifest['identity']) != manifest['build_id']):
            raise ValueError('SE3 value/prepared identity mismatch')
        records = list(recovery['original_manifests'].values())
        if (len(records) != 8 or {r['shard'] for r in records} != set(range(8))
                or len({object_hash(r['identity']) for r in records}) != 1):
            raise ValueError('SE3 recovery must identify all eight matching shards')
        old = records[0]['identity']
        expected = dict(engine_sha256=identity['engine_sha256'], model_sha256=identity['model_sha256'],
                        engine_config_sha256=identity['config_sha256'], source_manifest_sha256=file_hash(manifest_path),
                        source_build_id=manifest['build_id'], split='train', num_shards=8, limit_games=None,
                        rules='japanese', komi=6.5, visits=1, winrate_perspective='BLACK',
                        value_perspective='designated_player', ignorePreRootHistory=False,
                        target='per_effective_pre_action_position')
        if any(old.get(k) != v for k, v in expected.items()):
            raise ValueError('SE3 teacher conditions differ')
        cache = Path(manifest['cache_dir'])
        if (file_hash(cache / 'manifest.json') != manifest['identity']['cache_manifest_sha256']
                or identity['source_files'].get(str((cache / 'games.parquet').resolve()))
                != json.loads((cache / 'manifest.json').read_text())['games']['sha256']):
            raise ValueError('SE3 canonical move source differs')
        for name in ('games.parquet', 'train_game.npy', 'train_offsets.npy', 'train_ply.npy'):
            if file_hash(directory / name) != manifest['files'][name]:
                raise ValueError(f'SE3 index checksum changed: {name}')
        self.games = pd.read_parquet(directory / 'games.parquet').set_index('game_index')
        ids = np.load(directory / 'train_game.npy', mmap_mode='r')
        self.rows = {int(g): i for i, g in enumerate(ids)}
        self.offsets = np.load(directory / 'train_offsets.npy', mmap_mode='r')
        self.plies = np.load(directory / 'train_ply.npy', mmap_mode='r')
        self.values = np.load(value_path, mmap_mode='r')
        if (not pd.Index(ids).is_unique or len(self.offsets) != len(ids) + 1
                or self.offsets[0] != 0 or np.any(np.diff(self.offsets) <= 0)
                or self.offsets[-1] != len(self.plies) or self.values.shape != self.plies.shape
                or self.values.dtype != np.float32):
            raise ValueError('SE3 value/index shape mismatch')
        if sum(r['positions'] for r in records) != len(self.values):
            raise ValueError('SE3 recovery position count differs')
        self.receipt = dict(importer_sha256=file_hash(Path(__file__)), value_sha256=training['value_sha256'], prepared_manifest_sha256=file_hash(manifest_path),
                            recovery_sha256=file_hash(Path(options['recovery_path'])),
                            training_identity_sha256=file_hash(Path(options['training_identity_path'])))

    def positions(self, game, colors):
        if int(game.hd) != 0 or int(game.game_index) not in self.rows:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        old = self.games.loc[int(game.game_index)]
        for key in ('rank', 'game_id', 'player_color', 'length'):
            if str(old[key]) != str(getattr(game, key)):
                raise ValueError(f'SE3 game alignment changed: {key}')
        row = self.rows[int(game.game_index)]
        start, stop = self.offsets[row:row + 2]
        plies = np.asarray(self.plies[start:stop], dtype=np.int64)
        expected = np.flatnonzero((np.arange(len(colors)) >= 30) & (colors == int(game.player_color)))
        values = np.asarray(self.values[start:stop])
        if (not np.array_equal(plies, expected) or not np.isfinite(values).all() or np.any(np.abs(values) > 1)):
            raise ValueError('SE3 ply/value alignment invalid')
        # Every imported position is the designated player's turn, hence its value perspective already matches.
        return plies, values
