"""Consolidate completed training H/C features, preferring the SE3 recomputation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from go_aicup.features import humansl
from go_aicup.tasks.strength.SE_H_C_B2.common import H, C
from go_aicup.tasks.strength.SE_H_C_B2.data import match_metadata

ROOT = Path('data/processed/strength_train_hc')
OUT = ROOT / 'standard'
CANONICAL = Path('data/processed/reproduce_paper/seed2026/games.parquet')
H_SOURCES = ('humansl_q1000', 'humansl_qsel', 'humansl_vs_rest', 'humansl_train20', 'humansl_allu')
C_SOURCES = ('humansl_rankvec_feat', 'humansl_rankvec_feat_vu', 'humansl_allu/rankvec')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(f'{OUT} already exists')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if sha256(CANONICAL) != manifest['canonical_games_sha256']:
        raise ValueError('canonical games checksum mismatch')
    metadata = pd.read_parquet(CANONICAL)
    for name, info in manifest['files'].items():
        p = ROOT / name
        if not p.is_file() or sha256(p) != info['sha256']:
            raise ValueError(f'source checksum mismatch: {p}')
    results = {}
    for kind, names, columns in (('h', H_SOURCES, H), ('c', C_SOURCES, C)):
        parts = []
        for name in names:
            for shard in sorted((ROOT / 'legacy' / name).glob('shard_*')):
                if not (shard / 'meta.json').exists():
                    continue
                info = json.loads((shard / 'meta.json').read_text())
                games = pd.read_parquet(shard / 'games.parquet')
                match_metadata(games, metadata)
                if kind == 'h':
                    if info['profiles'] != list(humansl.PROFILES) or info['rules'] != humansl.RULES or info['komi'] != humansl.KOMI:
                        raise ValueError(f'H recipe mismatch: {shard}')
                    frame = humansl.shard_features(shard)
                else:
                    if info['columns'] != [c.removeprefix('c_') for c in C]:
                        raise ValueError(f'C columns mismatch: {shard}')
                    values = np.load(shard / 'feature.npy', allow_pickle=False)
                    done = np.load(shard / 'done.npy', allow_pickle=False)
                    if values.shape != (len(games), 20) or done.shape != (len(games),) or done.dtype != np.bool_:
                        raise ValueError(f'invalid C shard: {shard}')
                    frame = pd.DataFrame(values[done], columns=C)
                    frame.insert(0, 'game_index', games.game_index.to_numpy()[done])
                if len(frame):
                    parts.append(frame[['game_index', *columns]])
        old = pd.concat(parts, ignore_index=True)
        if old.game_index.duplicated().any() or not np.isfinite(old[columns].to_numpy()).all():
            raise ValueError(f'invalid or duplicate legacy {kind} features')
        fresh = []
        for path in sorted((ROOT / 'se3_hc_b2' / kind).glob('*.parquet')):
            if not path.with_suffix('.json').is_file():
                raise ValueError(f'missing feature marker: {path}')
            frame = pd.read_parquet(path)
            fresh.append(frame[['game_index', *columns]])
        new = pd.concat(fresh, ignore_index=True)
        if new.game_index.duplicated().any() or not np.isfinite(new[columns].to_numpy()).all():
            raise ValueError(f'invalid or duplicate new {kind} features')
        overlap = old.game_index.isin(new.game_index)
        if int(overlap.sum()) != 148:
            raise ValueError(f'unexpected {kind} overlap: {overlap.sum()}')
        combined = pd.concat([old.loc[~overlap], new], ignore_index=True).set_index('game_index').sort_index()
        if len(combined) != manifest['coverage'][kind]['total'] or not combined.index.isin(metadata.game_index).all():
            raise ValueError(f'{kind} coverage mismatch')
        results[kind] = combined
        print(kind, 'games', len(combined), 'preferred newer', int(overlap.sum()), flush=True)
    index = results['h'].index.union(results['c'].index).sort_values()
    values = np.concatenate([results['h'].reindex(index).to_numpy(dtype=np.float64),
                             results['c'].reindex(index).to_numpy(dtype=np.float64)], axis=1)
    counts = {'h': int(np.isfinite(values[:, :30]).all(axis=1).sum()),
              'c': int(np.isfinite(values[:, 30:]).all(axis=1).sum()),
              'both': int(np.isfinite(values).all(axis=1).sum())}
    if any(counts[k] != manifest['coverage'][k]['total'] for k in counts):
        raise ValueError(f'coverage mismatch: {counts}')
    OUT.mkdir()
    np.save(OUT / 'game_index.npy', index.to_numpy(dtype=np.int64), allow_pickle=False)
    np.save(OUT / 'features.npy', values, allow_pickle=False)
    saved_index = np.load(OUT / 'game_index.npy', allow_pickle=False)
    saved_values = np.load(OUT / 'features.npy', allow_pickle=False)
    if not np.array_equal(saved_index, index.to_numpy()) or not np.array_equal(saved_values, values, equal_nan=True):
        raise ValueError('array round-trip failed')
    info = {'format': 'training_hc_v1', 'canonical_games_sha256': manifest['canonical_games_sha256'],
            'columns': H + C, 'shape': list(values.shape), 'dtype': str(values.dtype),
            'missing': 'NaN', 'coverage': counts, 'overlap_preference': 'se3_hc_b2',
            'overlap_games_per_component': 148,
            'sha256': {name: sha256(OUT / name) for name in ('game_index.npy', 'features.npy')}}
    (OUT / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
