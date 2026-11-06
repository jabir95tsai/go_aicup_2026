"""Official complete-question inference; sample-ordered merged Public/Private CSV."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.v2.predict import prepare as prepare_v2
from .common import S, H, C, BUCKETS, atomic_table
from .data import load_prepared
from .features import context, read_component
from .model import probabilities, validate_probabilities
from .train import load_bundle, question_values


def prepare_release(config, store, name, work):
    load_prepared(config)
    prepare_v2(Path(store), name, Path(work))
    return release_questions(work)


def release_questions(work):
    work = Path(work)
    meta = pd.read_parquet(work / 'store_meta.parquet')
    games = pd.read_parquet(work / 'cache/games.parquet')
    if not games.store_row.is_unique or not meta.index.is_unique:
        raise ValueError('duplicate release game mapping')
    cache_ids = pd.Series(games.game_index.to_numpy(), index=games.store_row)
    rows = []
    for question_id, group in meta.groupby('question_id', sort=False):
        k = int(group.num_games.iloc[0])
        if (k not in BUCKETS or not group.num_games.eq(k).all() or len(group) != k
                or group.color.nunique() != 1 or not group.color.isin(['B', 'W']).all()):
            raise ValueError('invalid declared question size or mixed color')
        ids = cache_ids.reindex(group.index).fillna(-1).to_numpy(np.int64)
        rows.append(dict(question_id=str(question_id), k=k, color=group.color.iloc[0], games=ids))
    q = pd.DataFrame(rows)
    if q.empty or not q.question_id.is_unique:
        raise ValueError('empty/duplicate release questions')
    return q


def aggregate_frames(bundle, q, frame, fallback=None):
    """A failed game causes whole-question fallback, never a shortened question."""
    if bundle['arm'] != 'se3' or bundle['stage'] != 'final' or list(frame.columns) != bundle['columns']:
        raise ValueError('release requires the final SE3 B2 and exact feature order')
    if not frame.index.is_unique:
        raise ValueError('duplicate release feature IDs')
    usable, reasons = [], []
    for row in q.itertuples():
        ids = np.asarray(row.games)
        if row.k not in BUCKETS or len(ids) != row.k:
            raise ValueError('invalid declared k')
        complete = len(set(ids)) == row.k and (ids >= 0).all() and np.isin(ids, frame.index).all()
        usable.append(complete)
        reasons.append('' if complete else 'missing_or_failed_game_features')
    usable = np.asarray(usable)
    proba = np.empty((len(q), 10), np.float32)
    if usable.any():
        values = question_values(frame, q[usable], bundle['columns'])
        proba[usable] = probabilities(bundle['heads'], values, q.loc[usable, 'k'].to_numpy())
    if (~usable).any():
        if fallback is None or not fallback.question_id.is_unique:
            raise ValueError('incomplete question requires an existing AH+C posterior')
        backup = fallback.copy(); backup['question_id'] = backup.question_id.astype(str)
        backup = backup.set_index('question_id').loc[q.loc[~usable, 'question_id']]
        if (backup['num_games'].to_numpy() != q.loc[~usable, 'k'].to_numpy()).any():
            raise ValueError('AH+C fallback k mismatch')
        validate_probabilities(backup[[f'p_{r}' for r in RANKS]].to_numpy())
        # A fallback must itself have complete AH+C features.
        if (backup.fallback_v2.any() or backup.fallback_other.any()
                or (backup.humansl_games != backup.num_games).any()):
            raise ValueError('fallback must contain complete AH+C posteriors')
        proba[~usable] = backup[[f'p_{r}' for r in RANKS]].to_numpy()
    validate_probabilities(proba)
    pred = score_rank(proba)
    output = q.drop(columns='games').assign(rank_id=pred, rank=[RANKS[i] for i in pred],
                                           fallback_ahc=~usable, fallback_reason=reasons)
    for i, rank in enumerate(RANKS):
        output[f'p_{rank}'] = proba[:, i]
    return output


def aggregate(config, work, fallback=None):
    root = Path(config['paths']['output']); work = Path(work)
    bundle = load_bundle(root / 'models/final/se3.pt', config, 'se3')
    q = release_questions(work)
    ctx = context(config, work)
    parts = [read_component(config, kind, release=work, ctx=ctx) for kind in ('se3', 'h', 'c')]
    frame = pd.concat(parts, axis=1, join='inner')
    backup = pd.read_parquet(fallback) if fallback is not None else None
    output = aggregate_frames(bundle, q, frame, backup)
    atomic_table(work / 'se3_hc_b2_predictions.parquet', output)
    write_json(work / 'se3_hc_b2_predictions.json', dict(model=file_hash(root / 'models/final/se3.pt'),
        source=ctx[3], input_metadata=file_hash(work / 'store_meta.parquet'),
        fallback=None if fallback is None else file_hash(fallback),
        fallback_questions=int(output.fallback_ahc.sum()), questions=len(output),
        sha256=file_hash(work / 'se3_hc_b2_predictions.parquet')))
    return dict(questions=len(output), fallback_ahc=int(output.fallback_ahc.sum()))


def export(work_root, sample, output):
    work_root, sample, output = Path(work_root), Path(sample), Path(output)
    template = pd.read_csv(sample, dtype=str)
    if list(template) != ['question_id', 'pred_rank'] or not template.question_id.is_unique:
        raise ValueError('official sample requires unique question_id,pred_rank')
    parts = []
    model_id = None
    for part in ('public', 'private'):
        path = work_root / f'rank_{part}/se3_hc_b2_predictions.parquet'
        frame = pd.read_parquet(path)
        marker = json.loads(path.with_suffix('.json').read_text())
        if not marker.get('model') or (model_id is not None and model_id != marker['model']):
            raise ValueError('Public/Private predictions use different models')
        model_id = marker['model']
        if marker['sha256'] != file_hash(path):
            raise ValueError('prediction checksum mismatch')
        validate_probabilities(frame[[f'p_{r}' for r in RANKS]].to_numpy())
        if not np.array_equal(score_rank(frame[[f'p_{r}' for r in RANKS]].to_numpy()), frame.rank_id):
            raise ValueError('prediction posterior/rank mismatch')
        if frame['rank'].tolist() != [RANKS[i] for i in frame.rank_id]:
            raise ValueError('rank label/index mismatch')
        parts.append(frame[['question_id', 'rank']].rename(columns={'rank': 'pred_rank'}))
    result = pd.concat(parts, ignore_index=True)
    if (not result.question_id.is_unique or set(result.question_id) != set(template.question_id)
            or not result.pred_rank.isin(RANKS).all()):
        raise ValueError('submission must cover Public and Private sample IDs exactly')
    result = result.set_index('question_id').loc[template.question_id].reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    result.to_csv(temporary, index=False, lineterminator='\r\n')
    pd.testing.assert_frame_equal(pd.read_csv(temporary, dtype=str), result)
    temporary.replace(output)
    return dict(output=str(output), questions=len(result), sha256=file_hash(output))
