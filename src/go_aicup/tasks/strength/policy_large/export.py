"""Infer frozen large-policy H+C heads on all official test games."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import zipfile

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.data.sgf import parse
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_hc_b3.train import Head
from . import b3, cache
from .models import module_for
from .run import frozen_features

ARMS = ('A0', 'R10', 'CNN-Z32')


def prepare(out):
    board = out/'boards'
    board.mkdir(parents=True, exist_ok=True)
    games, questions, moves, colors = [], [], [], []
    offset = 0
    with zipfile.ZipFile(ROOT/'data/raw/raw.zip') as archive:
        for part in ('public', 'private'):
            with archive.open(f'raw/tests/rank_prediction_test_{part}.csv') as stream:
                raw = pd.read_csv(stream, dtype=str, keep_default_na=False)
            if len(raw) != 400 or not raw.question_id.is_unique:
                raise ValueError('invalid official question IDs/count')
            for row in raw.to_dict('records'):
                k = int(row['num_games'])
                if k not in (5, 10, 15, 20):
                    raise ValueError('unsupported question size')
                ids, sides = [], []
                for slot in range(1, k+1):
                    game = parse(row[f'sgf_{slot}'])
                    hd = game.props.get('HD', [])
                    color = row[f'color_{slot}']
                    if not game.ok or hd not in (['0'], ['1']) or color not in ('B', 'W'):
                        raise ValueError('invalid SGF, HD or player color')
                    index = len(games)
                    ids.append(index); sides.append(color)
                    games.append(dict(game_index=index, shard_id=0, local_offset=offset,
                                      length=game.n_moves, player_color=int(color == 'W'), handicap=int(hd[0])))
                    moves.append(game.moves); colors.append(game.colors)
                    offset += game.n_moves
                if len(set(sides)) != 1:
                    raise ValueError('mixed-color question')
                questions.append(dict(question_id=row['question_id'], k=k, games=ids, part=part))
    games, questions = pd.DataFrame(games), pd.DataFrame(questions)
    if len(games) != 10000 or not questions.question_id.is_unique:
        raise ValueError('incomplete official test data')
    for part, group in questions.groupby('part'):
        if group.k.value_counts().to_dict() != {5:100, 10:100, 15:100, 20:100}:
            raise ValueError(f'{part}: invalid k coverage')
    games.to_parquet(board/'games.parquet', index=False)
    questions.to_parquet(out/'questions.parquet', index=False)
    np.save(board/'shard_00000_moves.npy', np.concatenate(moves))
    np.save(board/'shard_00000_colors.npy', np.concatenate(colors))
    return games, questions


def infer(family, games, config, run, out, identity):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    policy, report = cache.infer_family(family, games, config, out)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    compact = {'A0': policy.arrays['c'].numpy().copy()}
    device = torch.device('cuda', config['family_devices'][family][0])
    for arm in ARMS[1:]:
        compact[arm] = frozen_features(family, arm, policy, run, device, config['position_chunk'], identity)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    record = json.loads((run/'identity.json').read_text())
    identity = record['identity']
    if json.loads((run/'complete.json').read_text()) != {'identity':identity, 'status':'complete'}:
        raise ValueError('training run is incomplete')
    for relative, digest in record['code'].items():
        if file_hash(ROOT/relative) != digest and relative != 'src/go_aicup/tasks/strength/policy_large/cache.py':
            raise ValueError(f'training source changed: {relative}')
    for relative in record['data']['sources']:
        if relative.endswith(('vectors.pt', 'b18c384nbt-humanv0.ckpt')):
            if file_hash(ROOT/relative) != record['data']['sources'][relative]:
                raise ValueError('teacher source changed')
    games, questions = prepare(out)
    template = pd.read_csv(ROOT/'submissions/rank/751959-cv2-l10.csv', dtype=str)
    if not template.question_id.is_unique or set(template.question_id) != set(questions.question_id):
        raise ValueError('template and test IDs differ')
    if args.prepare_only:
        print({'games':len(games), 'questions':len(questions), 'positions':int(games.length.sum())})
        return
    if torch.cuda.device_count() != 8:
        raise ValueError('export requires eight GPUs')
    config = dict(record['config'], board_root=str(out/'boards'))
    with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as pool:
        futures = {f:pool.submit(infer, f, games, config, run, out, identity) for f in ('H', 'C')}
        features = {f:future.result() for f, future in futures.items()}
    feature_dir = ROOT/'data/processed/strength_features/humansl/policy_large/806639_test'
    feature_dir.mkdir(parents=True, exist_ok=True)
    if (feature_dir/'manifest.json').exists():
        raise FileExistsError(feature_dir/'manifest.json')
    games.to_parquet(feature_dir/'games.parquet', index=False)
    questions.to_parquet(feature_dir/'questions.parquet', index=False)
    files = {}
    for family, arms in features.items():
        for arm, values in arms.items():
            path = feature_dir/f'{family}_{arm}.npy'
            np.save(path, values, allow_pickle=False)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
            files[path.name] = dict(shape=list(values.shape), sha256=file_hash(path))
    reports = {}
    for arm in ARMS:
        values = b3.assemble(arm, features['H'][arm], features['C'][arm], features['H']['A0'], features['C']['A0'])
        x = b3.question_values(games, questions, values)
        path = feature_dir/f'B3_{arm}_questions.npy'
        np.save(path, x, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), x)
        files[path.name] = dict(shape=list(x.shape), sha256=file_hash(path))
        probs = np.full((len(questions), 10), np.nan, np.float32)
        checkpoints = {}
        for k in (5, 10, 15, 20):
            path = run/'B3'/arm/f'k{k}'/'model.pt'
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['identity'] != identity:
                raise ValueError('B3 checkpoint identity differs')
            head = Head(x.shape[1]).eval()
            head.load_state_dict(payload['head'])
            mask = questions.k.eq(k).to_numpy()
            with torch.inference_mode():
                probs[mask] = head(torch.from_numpy(x[mask])).softmax(-1).numpy()
            checkpoints[str(k)] = dict(sha256=file_hash(path), step=payload['step'])
        if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-6):
            raise ValueError('invalid submission probabilities')
        predictions = questions[['question_id', 'k', 'part']].copy()
        predictions['pred_rank'] = np.asarray(RANKS)[score_rank(probs)]
        for i in range(10):
            predictions[f'p{i}'] = probs[:, i]
        predictions.to_parquet(feature_dir/f'{arm}_predictions.parquet', index=False)
        submission = predictions.set_index('question_id').loc[template.question_id, ['pred_rank']].reset_index()
        destination = ROOT/f'submissions/rank/806639-hc-{arm.lower()}.csv'
        if destination.exists():
            raise FileExistsError(destination)
        temporary = destination.with_suffix('.csv.tmp')
        submission.to_csv(temporary, index=False, lineterminator='\r\n')
        pd.testing.assert_frame_equal(pd.read_csv(temporary, dtype=str), submission)
        temporary.replace(destination)
        reports[arm] = dict(path=str(destination.relative_to(ROOT)), sha256=file_hash(destination),
                            rows=len(submission), checkpoints=checkpoints,
                            changed_vs_template={p:int((submission.set_index('question_id').pred_rank.loc[q.question_id].to_numpy()
                                != template.set_index('question_id').pred_rank.loc[q.question_id].to_numpy()).sum())
                                for p,q in questions.groupby('part')})
    write_json(out/'submission_report.json', dict(identity=identity, training_run=str(run), submissions=reports))
    write_json(feature_dir/'manifest.json', dict(identity=identity, training_run=str(run), files=files,
        code_sha256=file_hash(Path(__file__)), cache_code_sha256=file_hash(Path(cache.__file__)),
        raw_zip_sha256=file_hash(ROOT/'data/raw/raw.zip'), submissions=reports,
        feature_order='H own/opponent then C own/opponent, using b3.assemble; question rows in questions.parquet',
        complete=True))
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
