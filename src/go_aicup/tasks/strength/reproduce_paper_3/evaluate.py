"""Fixed monitoring and full-validation classification; game-equal log probabilities."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import default_collate

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import autocast_context, on_rank_zero, setup_distributed
from go_aicup.tasks.strength.shared.training import rng_state, restore_rng, configure_torch
from .config import DEFAULT_CONFIG, read_config
from .model import StrengthEstimator

LOGP = [f'logp_{i}' for i in range(10)]
BUCKETS = (5, 10, 15, 20)


def atomic_table(path, frame):
    temporary = path.with_name(path.name + '.tmp')
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def check_questions(questions, table):
    metadata = table.set_index('game_index', verify_integrity=True)
    if questions.question_id.duplicated().any() or set(questions.k) != set(BUCKETS):
        raise ValueError('questions require unique IDs and all four k buckets')
    for q in questions.itertuples():
        if len(q.games) != q.k or len(set(q.games)) != q.k:
            raise ValueError('question games must be distinct and match k')
        try:
            rows = metadata.loc[q.games]
        except KeyError as exc:
            raise ValueError('question contains unknown game') from exc
        if any(not (rows[key] == value).all() for key, value in
               (('split', 'valid'), ('rank_id', q.rank_id), ('player_id', q.player_id), ('color', q.color))):
            raise ValueError('questions must use validation games of the same player/rank/color')


def prepare_monitor(config):
    """Publish once; verify source/selection on every reuse, never silently redraw."""
    root = Path(config['paths']['dataset_dir'])
    directory = Path(config['paths']['output_dir']) / 'monitor'
    identity = dict(questions_sha256=file_hash(root / 'validation_questions.parquet'),
                    prepared_sha256=file_hash(root / 'manifest.json'), monitoring=config['monitoring'],
                    positions_per_game=30, code_sha256=file_hash(Path(__file__)))
    manifest = directory / 'manifest.json'
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record['identity'] != identity:
            raise ValueError('monitor identity mismatch')
        for name, digest in record['files'].items():
            if Path(name).name != name or file_hash(directory / name) != digest:
                raise ValueError('monitor checksum mismatch')
        return record
    directory.mkdir(parents=True, exist_ok=True)
    questions = pd.read_parquet(root / 'validation_questions.parquet')
    check_questions(questions, pd.read_parquet(root / 'games.parquet'))
    cells = questions.groupby(['k', 'rank_id'])
    if len(cells) != 40:
        raise ValueError('all forty monitoring cells are required')
    count = min(config['monitoring']['questions'] // 40, int(cells.size().min()))
    if count < 1:
        raise ValueError('empty monitoring cell')
    selected = []
    for (k, level), rows in cells:
        rows = rows.sort_values('question_id')
        rng = np.random.default_rng(np.random.SeedSequence([config['monitoring']['seed'], k, level]))
        selected.append(rows.iloc[np.sort(rng.choice(len(rows), count, replace=False))])
    selected = pd.concat(selected, ignore_index=True)
    games = np.unique(np.concatenate(selected.games)).astype(np.int64)
    index = np.load(root / 'valid_game.npy', mmap_mode='r')
    offsets = np.load(root / 'valid_offsets.npy', mmap_mode='r')
    lookup = {int(game): row for row, game in enumerate(index)}
    samples = []
    for game in games:
        row = lookup[int(game)]
        start, stop = offsets[row:row + 2]
        if stop - start < 30:
            raise ValueError(f'game_index={game}: needs thirty positions')
        rng = np.random.default_rng(np.random.SeedSequence([config['monitoring']['seed'], int(game), 30]))
        samples.append(np.sort(start + rng.choice(int(stop - start), 30, replace=False)))
    atomic_table(directory / 'questions.parquet', selected)
    temporary = directory / 'positions.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez(handle, games=games, samples=np.asarray(samples, dtype=np.int64))
    os.replace(temporary, directory / 'positions.npz')
    record = dict(identity=identity, questions=len(selected), unique_games=len(games),
                  files={name: file_hash(directory / name) for name in ('questions.parquet', 'positions.npz')})
    record['monitor_id'] = object_hash(record)
    write_json(manifest, record)
    return record


def predict_questions(games, questions):
    lookup = games.set_index('game_index', verify_integrity=True)
    if games.position_count.le(0).any() or not np.isfinite(games[LOGP].to_numpy()).all():
        raise ValueError('invalid per-game log probabilities/counts')
    result = questions.drop(columns='games').copy()
    vectors = np.stack([lookup.loc[q.games, LOGP].to_numpy(np.float64).mean(axis=0) for q in questions.itertuples()])
    result[LOGP] = vectors
    result['predicted_rank_id'] = vectors.argmax(axis=1)
    return result


def official_scores(predictions):
    if set(predictions.k) != set(BUCKETS):
        raise ValueError('all k buckets are required')
    scores = {}
    for k in BUCKETS:
        rows = predictions.loc[predictions.k == k]
        difference = np.abs(rows.predicted_rank_id.to_numpy() - rows.rank_id.to_numpy())
        exact, adjacent = int(np.sum(difference == 0)), int(np.sum(difference == 1))
        scores[str(k)] = dict(n=len(rows), exact=exact, adjacent=adjacent,
                             score=(exact + adjacent / math.e) / len(rows))
    score = sum(row['score'] for row in scores.values()) / 4
    if not math.isfinite(score):
        raise ValueError('nonfinite validation score')
    return dict(equal_k=score, buckets=scores)


def collective_result(function):
    """Local inference errors must reach every rank before the next collective."""
    result, error = None, None
    try:
        result = function()
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    errors = [error]
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError('; '.join(e for e in errors if e))
    return result


@torch.inference_mode()
def infer_games(model, dataset, games, config, device, samples=None):
    """Accumulate FP32 log_softmax outputs in FP64, retaining only game means."""
    dataset._ensure_open()
    index, offsets, _ = dataset._indices
    lookup = {int(game): row for row, game in enumerate(index)}
    lengths, tasks = [], []
    for i, game in enumerate(games):
        row = lookup[int(game)]
        chosen = samples[i] if samples is not None else np.arange(offsets[row], offsets[row + 1])
        if len(chosen) == 0 or np.any(chosen < offsets[row]) or np.any(chosen >= offsets[row + 1]):
            raise ValueError('invalid inference positions')
        lengths.append(len(chosen))
        tasks.append(chosen)
    sums = np.zeros((len(games), 10), np.float64)
    if len(games):
        indices = np.concatenate(tasks)
        rows = np.repeat(np.arange(len(games)), lengths)
        size = config['evaluation']['per_gpu_batch_size']
        for begin in range(0, len(indices), size):
            batch = default_collate(dataset.__getitems__(indices[begin:begin + size]))
            planes = batch['planes'].to(device, dtype=torch.float32,
                memory_format=torch.channels_last if config['training']['channels_last'] else torch.contiguous_format)
            with autocast_context(config, device):
                logits = model(planes)['rank_logits']
            logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            if not np.isfinite(logp).all():
                raise ValueError('nonfinite rank log probabilities')
            np.add.at(sums, rows[begin:begin + size], logp)
        sums /= np.asarray(lengths)[:, None]
    frame = pd.DataFrame(sums, columns=LOGP)
    frame.insert(0, 'game_index', np.asarray(games, np.int64))
    frame['position_count'] = np.asarray(lengths, np.int64)
    return frame


def monitor(model, dataset, config, device, rank, world):
    run = Path(config['paths']['output_dir'])
    positions = np.load(run / 'monitor/positions.npz')
    games, samples = positions['games'], positions['samples']
    start, stop = len(games) * rank // world, len(games) * (rank + 1) // world
    saved_rng, training = rng_state(device, model), model.training
    begin = time.perf_counter()
    try:
        model.eval()
        if world > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, src=0)
        local = collective_result(lambda: infer_games(model, dataset, games[start:stop], config, device, samples[start:stop]))
        frames = [local]
        if world > 1:
            frames = [None] * world
            dist.all_gather_object(frames, local)
        result = official_scores(predict_questions(pd.concat(frames, ignore_index=True),
                                 pd.read_parquet(run / 'monitor/questions.parquet')))
        result['seconds'] = time.perf_counter() - begin
        return result
    finally:
        model.train(training)
        restore_rng(saved_rng, device, model)
        positions.close()


def build_reports(config, games):
    run = Path(config['paths']['output_dir'])
    directory = run / 'evaluation'
    table = pd.read_parquet(Path(config['paths']['dataset_dir']) / 'games.parquet')
    questions = pd.read_parquet(Path(config['paths']['dataset_dir']) / 'validation_questions.parquet')
    check_questions(questions, table)
    predicted = predict_questions(games, questions)
    report = official_scores(predicted)
    report['scope'] = 'full validation, all effective positions; fixed full questions; best monitoring checkpoint'
    diagnostics = games.merge(table, on='game_index', validate='one_to_one')
    diagnostics['predicted_rank_id'] = diagnostics[LOGP].to_numpy().argmax(axis=1)
    difference = np.abs(diagnostics.predicted_rank_id.to_numpy() - diagnostics.rank_id.to_numpy())
    report['single_game'] = dict(n=len(diagnostics), exact=int(np.sum(difference == 0)),
                                 adjacent=int(np.sum(difference == 1)),
                                 score=float(np.mean((difference == 0) + (difference == 1) / math.e)))
    atomic_table(directory / 'validation_games.parquet', diagnostics)
    atomic_table(directory / 'validation_predictions.parquet', predicted)
    pd.DataFrame(report['buckets']).T.to_csv(directory / 'official_scores.csv')
    details = []
    matrices = {}
    for name, rows in [('games', diagnostics), ('questions', predicted)]:
        true = rows.rank_id.to_numpy()
        guess = rows.predicted_rank_id.to_numpy()
        matrix = np.zeros((10, 10), np.int64)
        np.add.at(matrix, (true, guess), 1)
        matrices[name] = matrix
        for level in range(10):
            difference = np.abs(guess[true == level] - level)
            details.append(dict(scope=name, rank_id=level, n=len(difference),
                                score=float(np.mean((difference == 0) + (difference == 1) / math.e)) if len(difference) else None))
    for k in BUCKETS:
        rows = predicted.loc[predicted.k == k]
        matrix = np.zeros((10, 10), np.int64)
        np.add.at(matrix, (rows.rank_id, rows.predicted_rank_id), 1)
        matrices[f'k{k}'] = matrix
    pd.DataFrame(details).to_csv(directory / 'per_rank.csv', index=False)
    np.savez(directory / 'confusion_matrices.npz', **matrices)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = directory / 'plot'
    plot.mkdir(exist_ok=True)
    for name, matrix in matrices.items():
        figure, axis = plt.subplots()
        axis.imshow(matrix)
        axis.set(xlabel='Predicted rank', ylabel='True rank', title=name, xticks=range(10), yticks=range(10))
        figure.savefig(plot / f'confusion_{name}.png', dpi=140, bbox_inches='tight')
        plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(10, 7))
    log = run / 'train.jsonl'
    if log.exists() and log.stat().st_size:
        curve = pd.read_json(log, lines=True)
        for column in ('total', 'strength', 'policy', 'value'):
            axes[0].plot(curve.step, curve[column], label=column)
        axes[0].legend()
    validation = run / 'validation.jsonl'
    if validation.exists() and validation.stat().st_size:
        curve = pd.read_json(validation, lines=True)
        axes[1].plot(curve.step, curve.equal_k, marker='o', label='monitor Equal-k (30 positions/game)')
        axes[1].legend()
    axes[0].set(ylabel='Training loss')
    axes[1].set(xlabel='Completed updates', ylabel='Monitor score')
    figure.tight_layout()
    figure.savefig(plot / 'learning_curves.png', dpi=140)
    plt.close(figure)
    write_json(directory / 'result.json', report)
    return report


def evaluate(config, *, report_only=False):
    from .dataset import validate_prepared
    from .train import code_fingerprints, load_artifact, runtime_config
    if report_only:
        run = Path(config['paths']['output_dir'])
        directory = run / 'evaluation'
        identity = json.loads((directory / 'manifest.json').read_text())
        payload = load_artifact(run / 'model.pt', config)
        current_data = validate_prepared(config, teacher=False)
        recorded = json.loads((run / 'data_identity.json').read_text())
        if (identity['model_sha256'] != file_hash(resolve_checkpoint(run / 'model.pt'))
                or identity['data'] != current_data or identity['code_sha256'] != code_fingerprints()
                or any(recorded.get(k) != v for k, v in current_data.items())
                or payload['dataset_id'] != object_hash(recorded)):
            raise ValueError('report source identity mismatch')
        frames = []
        index = np.load(Path(config['paths']['dataset_dir']) / 'valid_game.npy')
        for r in range(identity['world_size']):
            path = directory / f'games_rank{r:02d}.parquet'
            marker = json.loads(path.with_suffix('.json').read_text())
            frame = pd.read_parquet(path)
            expected = index[len(index) * r // identity['world_size']:len(index) * (r + 1) // identity['world_size']]
            if (marker != dict(identity=object_hash(identity), sha256=file_hash(path))
                    or not np.array_equal(frame.game_index, expected)):
                raise ValueError('report shard checksum/data mismatch')
            frames.append(frame)
        return build_reports(config, pd.concat(frames, ignore_index=True))
    configure_torch(config)
    own_group = not dist.is_initialized()
    if own_group:
        device, rank, world = setup_distributed(runtime_config(config))
    else:
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if config['device'] == 'cuda' else torch.device('cpu')
    run = Path(config['paths']['output_dir'])
    directory = run / 'evaluation'
    dataset = None
    try:
        data = collective_result(lambda: validate_prepared(config, teacher=False))
        path = resolve_checkpoint(run / 'model.pt')
        payload = load_artifact(path, config)
        recorded = json.loads((run / 'data_identity.json').read_text())
        if any(recorded.get(k) != v for k, v in data.items()) or payload['dataset_id'] != object_hash(recorded):
            raise ValueError('evaluation data identity mismatch')
        if payload['code_sha256'] != code_fingerprints():
            raise ValueError('evaluation code identity mismatch')
        identity = dict(model_sha256=file_hash(path), data=data, world_size=world,
                        precision=config['precision'], batch_size=config['evaluation']['per_gpu_batch_size'],
                        code_sha256=code_fingerprints())
        def initialize():
            directory.mkdir(exist_ok=True)
            manifest = directory / 'manifest.json'
            if manifest.exists() and json.loads(manifest.read_text()) != identity:
                raise ValueError('evaluation shard identity mismatch')
            write_json(manifest, identity)
        on_rank_zero(initialize, rank)
        dataset = PaperDataset(config['paths']['dataset_dir'], split='valid', cache_dir=config['paths']['cache_dir'],
                               max_open_shards=config['loader']['max_open_shards'])
        dataset._ensure_open()
        games = dataset._indices[0]
        start, stop = len(games) * rank // world, len(games) * (rank + 1) // world
        shard = directory / f'games_rank{rank:02d}.parquet'
        def load_shard(path, expected):
            marker = json.loads(path.with_suffix('.json').read_text())
            frame = pd.read_parquet(path)
            if (marker != dict(identity=object_hash(identity), sha256=file_hash(path))
                    or not np.array_equal(frame.game_index, expected)
                    or frame.position_count.le(0).any() or not np.isfinite(frame[LOGP].to_numpy()).all()):
                raise ValueError('evaluation shard checksum/data mismatch')
            return frame
        def infer_or_load():
            if shard.with_suffix('.json').exists():
                return load_shard(shard, games[start:stop])
            model = StrengthEstimator(**config['model']).to(device)
            model.load_state_dict(payload['model'])
            if config['training']['channels_last']:
                model.to(memory_format=torch.channels_last)
            model.eval()
            frame = infer_games(model, dataset, games[start:stop], config, device)
            atomic_table(shard, frame)
            write_json(shard.with_suffix('.json'), dict(identity=object_hash(identity), sha256=file_hash(shard)))
            return load_shard(shard, games[start:stop])
        collective_result(infer_or_load)
        def report():
            frames = [load_shard(directory / f'games_rank{r:02d}.parquet',
                                games[len(games) * r // world:len(games) * (r + 1) // world]) for r in range(world)]
            return build_reports(config, pd.concat(frames, ignore_index=True))
        return on_rank_zero(report, rank)
    finally:
        if dataset is not None:
            dataset.close()
        if own_group and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    evaluate(read_config(args.config), report_only=args.report_only)


if __name__ == '__main__':
    main()
