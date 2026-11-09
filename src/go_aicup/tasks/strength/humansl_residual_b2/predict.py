"""Single full-P model inference and validated Public/Private CSV export."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.SE_H_C_B2.common import atomic_table, publish_table
from go_aicup.tasks.strength.SE_H_C_B2.predict import release_questions, export
from .b2 import load_model, model_path, probabilities
from .evaluate import load_lock
from .features import infer_games, read_features

FILENAME = 'humansl_residual_b2_predictions.parquet'


def predict(config, work, checkpoint, *, stage, candidate='final', batch_size=300):
    from .dataset import SelectedDataset
    from .train import load_candidate
    from .data import load_prepared
    manifest = load_prepared(config, release=True)
    load_lock(config)
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
    model, payload = load_candidate(Path(checkpoint), config, device=device)
    upstream = payload['identity']
    locked = load_lock(config)
    if payload['dataset_id'] != manifest['dataset_id'] or payload['config_sha256'] != object_hash(config):
        raise ValueError('release model dataset/config identity mismatch')
    if stage == 2:
        parent = Path(config['run_dir']) / 'release/fold_all/stage1/final.pt'
        parent_payload = torch.load(parent, map_location='cpu', weights_only=False)
        if (parent_payload['state']['optimizer_steps'] != locked['fixed_steps']['s1']
                or upstream['stage1_source_sha256'] != file_hash(parent)):
            raise ValueError('stage-two release ancestor differs from the locked s1 refit')
    if payload['state']['optimizer_steps'] != locked['fixed_steps'][f's{stage}']:
        raise ValueError('release model was not trained for the locked fixed update count')
    if upstream['phase'] != 'release' or str(upstream['fold']) != 'all' or int(upstream['stage']) != stage:
        raise ValueError('test inference requires the single full-P release model')
    bundle = model_path(config, stage, candidate, phase='release')
    b2, identity = load_model(bundle)
    if identity['phase'] != 'release' or identity['stage'] != stage or identity['config'] != object_hash(config):
        raise ValueError('test inference requires release OOF B2')
    feature_file = work / 'humansl_residual_b2_games.parquet'
    feature_identity = dict(checkpoint=file_hash(Path(checkpoint)), metadata=file_hash(work / 'store_meta.parquet'),
                            games=file_hash(work / 'cache/games.parquet'), stage=stage, hd_komi='HD0:6.5/HD1:0.5')
    if feature_file.exists():
        frame, cached = read_features(feature_file)
        if cached != feature_identity:
            raise ValueError('release feature identity changed')
    else:
        dataset = SelectedDataset(runtime, rows, model.humansl.config, teacher_required=False, augment=False)
        frame = infer_games(model, dataset, rows, device=device, batch_size=batch_size, precision=config.get('precision', 'fp32'))
        publish_table(feature_file, frame, feature_identity)
    proba = probabilities(b2, rows, q, frame)
    pred = score_rank(proba)
    result = q.drop(columns='games').assign(rank_id=pred, rank=[RANKS[i] for i in pred])
    for i, rank in enumerate(RANKS):
        result[f'p_{rank}'] = proba[:, i]
    target = work / FILENAME
    atomic_table(target, result)
    write_json(target.with_suffix('.json'), dict(model=object_hash(dict(upstream=file_hash(Path(checkpoint)),
        b2=file_hash(bundle))), source=feature_identity, sha256=file_hash(target), questions=len(result)))
    return dict(questions=len(result), path=str(target))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config'); p.add_argument('--work'); p.add_argument('--checkpoint')
    p.add_argument('--prepare', action='store_true'); p.add_argument('--store-dir'); p.add_argument('--name')
    p.add_argument('--stage', type=int, choices=(1, 2), default=2)
    p.add_argument('--candidate', choices=('best', 'final'), default='final')
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
        result = predict(read_config(a.config), a.work, a.checkpoint,
                         stage=a.stage, candidate=a.candidate)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
