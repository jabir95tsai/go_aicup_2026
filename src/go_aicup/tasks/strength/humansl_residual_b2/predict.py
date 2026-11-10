"""Four locked fold models (feature average) + release B2 inference and validated Public/Private CSV export."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table, publish_table
from go_aicup.tasks.strength.SE_H_C_B2.predict import release_questions, export
from .b2 import average_frames, load_model, model_path, probabilities
from .evaluate import load_lock
from .features import infer_games, read_features

FILENAME = 'humansl_residual_b2_predictions.parquet'


def predict(config, work, *, arm_name, batch_size=300):
    """Test inference with the four locked development fold models (feature average) + release B2."""
    from .dataset import SelectedDataset
    from .train import load_candidate
    from .data import load_prepared
    manifest = load_prepared(config)
    locked = load_lock(config)
    spec = locked['arms'].get(arm_name)
    if spec is None or 'stage' not in spec:
        raise ValueError('test inference requires a locked new/stack arm name')
    stage, candidate, arm, aggregation = spec['stage'], spec['candidate'], spec['arm'], spec['aggregation']
    work = Path(work)
    q = release_questions(work)
    if any((np.asarray(ids) < 0).any() for ids in q.games):
        raise ValueError('release cache is incomplete; every game must map to the board cache')
    rows = pd.read_parquet(work / 'cache/games.parquet')
    meta = pd.read_parquet(work / 'store_meta.parquet')
    rows = rows.copy()
    rows['store_index'] = np.arange(len(rows))
    mapping = meta.iloc[rows.store_row.to_numpy()]
    for key in ('hd', 'player_color'):
        if key in mapping:
            rows[key] = mapping[key].to_numpy()
    if 'player_color' not in rows:
        rows['player_color'] = mapping.color.map({'B': 0, 'W': 1}).to_numpy()
    if 'hd' not in rows:
        raise ValueError('release data must contain explicit HD metadata; prepare the store with HD0/1')
    if not rows.hd.isin([0, 1]).all():
        raise ValueError('only HD0/1 release games are supported')
    rows['rank_id'] = 0
    rows['player_id'] = 'release-unlabelled'
    runtime = json.loads(json.dumps(config))
    runtime['data']['store_dir'] = str(work / 'cache')
    device = config.get('device', 'cuda')
    frames, upstream_hashes = [], {}
    for fold, record in sorted(locked['fold_checkpoints'][str(stage)].items()):
        checkpoint = Path(record['path'])
        if file_hash(checkpoint) != record['sha256']:
            raise ValueError('locked fold checkpoint changed')
        feature_file = work / f'humansl_residual_b2_stage{stage}_{candidate}_fold{fold}_games.parquet'
        feature_identity = dict(checkpoint=record['sha256'], metadata=file_hash(work / 'store_meta.parquet'),
                                games=file_hash(work / 'cache/games.parquet'), stage=stage, fold=str(fold),
                                format='humansl_residual_features_v2', hd_komi='HD0:6.5/HD1:0.5')
        if feature_file.exists():
            frame, cached = read_features(feature_file)
            if cached != feature_identity:
                raise ValueError('release feature identity changed')
        else:
            model, payload = load_candidate(checkpoint, config, device=device)
            identity = payload['identity']
            if payload['dataset_id'] != manifest['dataset_id'] or payload['config_sha256'] != object_hash(config):
                raise ValueError('release model dataset/config identity mismatch')
            if identity['phase'] != 'dev' or str(identity['fold']) != str(fold) or int(identity['stage']) != stage:
                raise ValueError('test inference requires the locked development fold models')
            dataset = SelectedDataset(runtime, rows, model.humansl.config, teacher_required=False, augment=False)
            frame = infer_games(model, dataset, rows, device=device, batch_size=batch_size,
                                precision=config.get('precision', 'fp32'))
            publish_table(feature_file, frame, feature_identity)
        frames.append(frame); upstream_hashes[str(fold)] = record['sha256']
    frame = average_frames(frames)
    baseline = None
    if arm == 'stack':
        from .baseline import extract_rows
        baseline = extract_rows(config, rows, runtime, work / 'controlled_hc_games.parquet')
    bundle = model_path(config, stage, candidate, 'release', arm, aggregation, spec.get('fit_pool'))
    b2, identity = load_model(bundle)
    if (identity['phase'] != 'release' or identity['stage'] != stage or identity['config'] != object_hash(config)
            or identity['arm'] != arm or identity['aggregation'] != aggregation):
        raise ValueError('test inference requires the matching release B2 arm')
    proba = probabilities(b2, rows, q, frame, aggregation, baseline)
    pred = score_rank(proba)
    result = q.drop(columns='games').assign(rank_id=pred, rank=[RANKS[i] for i in pred])
    for i, rank in enumerate(RANKS):
        result[f'p_{rank}'] = proba[:, i]
    target = work / FILENAME
    atomic_table(target, result)
    write_json(target.with_suffix('.json'), dict(model=object_hash(dict(upstream=upstream_hashes, b2=file_hash(bundle))),
                                                 arm=arm_name, sha256=file_hash(target), questions=len(result)))
    return dict(questions=len(result), path=str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--work'); p.add_argument('--arm-name', help='arm key in selection_lock.json')
    p.add_argument('--prepare', action='store_true'); p.add_argument('--store-dir'); p.add_argument('--name')
    p.add_argument('--work-root'); p.add_argument('--sample'); p.add_argument('--output')
    a = p.parse_args()
    if a.prepare:
        from go_aicup.tasks.strength.v2.predict import prepare
        prepare(Path(a.store_dir), a.name, Path(a.work))
        result = dict(prepared=a.work)
    elif a.work_root:
        result = export(a.work_root, a.sample, a.output, prediction_name=FILENAME)
    else:
        from .config import read_config
        result = predict(read_config(a.config), a.work, arm_name=a.arm_name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
