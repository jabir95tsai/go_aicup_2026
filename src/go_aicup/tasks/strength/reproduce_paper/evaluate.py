"""Full-training centroids, fixed-validation scores and saved PNG plots."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, NEAR, rank_report
from go_aicup.tasks.strength.shared.checkpoint import resolve_checkpoint
from go_aicup.tasks.strength.shared.runtime import setup_distributed, on_rank_zero, barrier
from go_aicup.tasks.strength.shared.training import configure_torch
from .config import DEFAULT_CONFIG, read_config, validate_config
from .dataset import PaperDataset
from .infer import infer_positions
from .label_value import exclusive
from .model import StrengthEstimator
from .questions import prepare_questions, QUESTIONS_FILE


def source_identity(config, world):
    run = Path(config['paths']['output_dir'])
    model_path = resolve_checkpoint(run / 'model.pt')
    payload = torch.load(model_path, map_location='cpu', weights_only=False)
    if (payload.get('artifact_format') != 'strength_estimator_1'
            or payload['optimizer_steps'] != config['training']['updates']
            or any(payload['config'][key] != config[key] for key in ('model', 'split_seed', 'positions', 'split'))):
        raise ValueError('evaluation requires the completed SE model with matching configuration')
    for name in ('model.py', 'dataset.py'):
        if payload['code_sha256'][name] != file_hash(Path(__file__).with_name(name)):
            raise ValueError('model/dataset code mismatch')
    prepared = Path(config['paths']['dataset_dir'])
    data = json.loads((run / 'data_identity.json').read_text())
    if (object_hash(data) != payload['dataset_id']
            or data['prepared_manifest_sha256'] != file_hash(prepared / 'manifest.json')
            or data['games_sha256'] != file_hash(prepared / 'games.parquet')):
        raise ValueError('evaluation model/prepared data mismatch')
    record = json.loads((prepared / 'manifest.json').read_text())
    for name, digest in record['files'].items():
        if Path(name).name != name or file_hash(prepared / name) != digest:
            raise ValueError('prepared file checksum mismatch')
    prepare_questions(config)
    return dict(model_sha256=file_hash(model_path), dataset_id=payload['dataset_id'],
                questions_sha256=file_hash(prepared / QUESTIONS_FILE), world_size=world,
                batch_size=config['evaluation']['per_gpu_batch_size'], precision=config['precision'],
                channels_last=config['training'].get('channels_last', False),
                code_sha256={name: file_hash(Path(__file__).with_name(name)) for name in
                             ('evaluate.py', 'infer.py', 'model.py', 'dataset.py')}), payload


def save_array(path, values):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def load_shard(directory, split, rank, world, positions, identity):
    path = directory / f'{split}_scores_rank{rank:02d}.npy'
    record = json.loads(path.with_suffix('.json').read_text())
    start, stop = positions * rank // world, positions * (rank + 1) // world
    if (record['identity'] != object_hash(identity) or record['start'] != start or record['stop'] != stop
            or record['sha256'] != file_hash(path)):
        raise ValueError('score shard identity/checksum mismatch')
    scores = np.load(path, mmap_mode='r', allow_pickle=False)
    if scores.dtype != np.float32 or scores.shape != (stop - start,) or not np.isfinite(scores).all():
        raise ValueError('invalid score shard shape/dtype/values')
    return scores


def aggregate_positions(scores, games, offsets, table):
    """Position-weighted statistics; each effective position appears exactly once."""
    scores = np.asarray(scores)
    if (scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all()
            or len(offsets) != len(games) + 1 or offsets[0] != 0 or offsets[-1] != len(scores)
            or np.any(np.diff(offsets) < 1) or len(np.unique(games)) != len(games)):
        raise ValueError('invalid score/index coverage')
    metadata = table.set_index('game_index', verify_integrity=True).loc[games]
    ranks = metadata.rank_id.to_numpy(dtype=np.int64)
    if np.any((ranks < 0) | (ranks > 9)):
        raise ValueError('invalid rank IDs')
    counts = np.diff(offsets)
    sums = np.add.reduceat(scores, offsets[:-1], dtype=np.float64)
    per_game = pd.DataFrame(dict(game_index=games, rank_id=ranks, position_count=counts,
                                 score_sum=sums, score_mean=sums / counts))
    return per_game


def training_statistics(scores, games, offsets):
    ranks = games.rank_id.to_numpy()
    counts = np.bincount(ranks, weights=games.position_count, minlength=10).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError('training centroid requires all ten ranks')
    means = np.bincount(ranks, weights=games.score_sum, minlength=10) / counts
    low, high = float(scores.min()), float(scores.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    edges = np.linspace(low, high, 129)
    histograms = np.zeros((10, 128), dtype=np.int64)
    squares = np.zeros(10, dtype=np.float64)
    minimum = np.full(10, np.inf)
    maximum = np.full(10, -np.inf)
    # Bounded temporary rank/value arrays, even for the full 76M-position training set.
    for begin in range(0, len(scores), 1_000_000):
        end = min(begin + 1_000_000, len(scores))
        selected_ranks = ranks[np.searchsorted(offsets, np.arange(begin, end), side='right') - 1]
        values = scores[begin:end].astype(np.float64)
        squares += np.bincount(selected_ranks, weights=values * values, minlength=10)
        for level in np.unique(selected_ranks):
            selected = values[selected_ranks == level]
            histograms[level] += np.histogram(selected, bins=edges)[0]
            minimum[level] = min(minimum[level], float(selected.min()))
            maximum[level] = max(maximum[level], float(selected.max()))
    if not np.array_equal(histograms.sum(axis=1), counts):
        raise ValueError('training histogram coverage mismatch')
    statistics = pd.DataFrame(dict(rank_id=np.arange(10), rank=RANKS, positions=counts, mean=means,
        std=np.sqrt(np.maximum(0, squares / counts - means * means)), min=minimum, max=maximum))
    return statistics, edges, histograms


def nearest_centroid(scores, centroids):
    centroids = np.asarray(centroids, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if centroids.shape != (10,) or not np.isfinite(centroids).all() or not np.isfinite(scores).all():
        raise ValueError('invalid scalar scores or centroids')
    # argmin chooses the lower rank index on equal distances; centroids are never reordered.
    return np.abs(scores[:, None] - centroids).argmin(axis=1)


def question_predictions(games, questions, table, centroids):
    lookup = games.set_index('game_index', verify_integrity=True)
    metadata = table.set_index('game_index', verify_integrity=True)
    scores = []
    if set(questions.k) != set(K_BUCKETS) or questions.question_id.duplicated().any():
        raise ValueError('fixed questions must have unique IDs and all four k buckets')
    for question in questions.itertuples():
        selected = np.asarray(question.games, dtype=np.int64)
        if len(selected) != question.k or len(np.unique(selected)) != question.k:
            raise ValueError('question games must be distinct and match k')
        try:
            source = metadata.loc[selected]
            values = lookup.loc[selected]
        except KeyError as exc:
            raise ValueError('question contains a game outside validation') from exc
        if (not (source.split == 'valid').all() or not (source.rank_id == question.rank_id).all()
                or not (source.player_id == question.player_id).all() or not (source.color == question.color).all()):
            raise ValueError('question must contain validation games from the same player/rank/color')
        scores.append(float(values.score_sum.sum() / values.position_count.sum()))
    output = questions.drop(columns='games').copy()
    output['score_mean'] = scores
    output['predicted_rank_id'] = nearest_centroid(scores, centroids)
    return output


def confusion(true, predicted):
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (np.asarray(true, dtype=np.int64), np.asarray(predicted, dtype=np.int64)), 1)
    return matrix


def save_plots(directory, statistics, edges, histograms, matrices, curve, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = directory / 'plot'
    output.mkdir(exist_ok=True)
    def publish(figure, name):
        temporary = output / f'{name}.png.tmp'
        figure.savefig(temporary, format='png', dpi=160, bbox_inches='tight')
        plt.close(figure)
        os.replace(temporary, output / f'{name}.png')
    figure, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for level, axis in enumerate(axes.flat):
        density = histograms[level] / (histograms[level].sum() * np.diff(edges))
        axis.stairs(density, edges, fill=True, alpha=.65)
        axis.axvline(statistics['mean'].iloc[level], color='red', linestyle='--', label='Mean')
        axis.set_title(f'{RANKS[level]} (n={int(statistics.positions.iloc[level]):,})')
        axis.set_xlabel('Strength score')
        axis.set_ylabel('Density')
        axis.legend(fontsize=8)
    figure.suptitle('Training score distributions — all effective positions')
    figure.tight_layout()
    publish(figure, 'training_score_distributions')
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(RANKS, statistics['mean'], marker='o')
    axis.set(xlabel='Rank', ylabel='Mean strength score', title='Training centroids (position weighted)')
    publish(figure, 'rank_centroids')
    for name, matrix in matrices.items():
        figure, axis = plt.subplots(figsize=(7, 6))
        normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                               out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0)
        shown = axis.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for row, col in np.ndindex(matrix.shape):
            axis.text(col, row, str(matrix[row, col]), ha='center', va='center', fontsize=7,
                      color='white' if normalized[row, col] > .5 else 'black')
        axis.set(xticks=np.arange(10), xticklabels=RANKS, yticks=np.arange(10), yticklabels=RANKS,
                 xlabel='Predicted rank', ylabel='True rank', title=f'Validation {name} — counts / row-normalized color')
        figure.colorbar(shown, ax=axis, label='Fraction of true rank')
        publish(figure, f'confusion_{name}')
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, columns in zip(axes, (('total',), ('bt', 'policy', 'value'))):
        for column in columns:
            axis.plot(curve.step, curve[column], alpha=.2, linewidth=.6)
            axis.plot(curve.step, curve[column].rolling(100, min_periods=1).mean(), label=column)
        axis.set_ylabel('Training loss')
        axis.legend()
        axis.grid(alpha=.2)
    axes[0].set_title('Training learning curve — raw and trailing 100-update mean')
    axes[1].set_xlabel('Completed optimizer updates')
    publish(figure, 'learning_curve')
    figure, axis = plt.subplots(figsize=(7, 4))
    labels = [str(k) for k in K_BUCKETS]
    values = [scores[label]['score'] for label in labels]
    axis.bar(labels, values)
    axis.axhline(np.mean(values), color='red', linestyle='--', label='Equal-k mean')
    axis.set(xlabel='Games per question (k)', ylabel='Official score', ylim=(0, 1),
             title='Fixed validation questions — official scoring formula')
    for index, value in enumerate(values):
        axis.text(index, value + .015, f'{value:.4f}', ha='center')
    axis.legend()
    publish(figure, 'official_scores')


def build_reports(config, identity):
    run = Path(config['paths']['output_dir'])
    directory = run / 'evaluation'
    prepared = Path(config['paths']['dataset_dir'])
    table = pd.read_parquet(prepared / 'games.parquet')
    split_games = {}
    statistics = edges = histograms = None
    for split in ('train', 'valid'):
        indices = [np.load(prepared / f'{split}_{name}.npy', mmap_mode='r', allow_pickle=False)
                   for name in ('game', 'offsets')]
        positions = int(indices[1][-1])
        # About 293 MiB for the official training scores; no board tensors are gathered.
        scores = np.concatenate([load_shard(directory, split, rank, identity['world_size'], positions, identity)
                                 for rank in range(identity['world_size'])])
        games = aggregate_positions(scores, *indices, table)
        split_games[split] = games
        if split == 'train':
            statistics, edges, histograms = training_statistics(scores, games, indices[1])
        del scores
    centroids = statistics['mean'].to_numpy()
    statistics.to_csv(directory / 'rank_statistics.csv', index=False)
    save_array(directory / 'centroids.npy', centroids)
    temporary = directory / 'training_distributions.npz.tmp'
    with temporary.open('wb') as handle:
        np.savez(handle, edges=edges, counts=histograms)
    os.replace(temporary, directory / 'training_distributions.npz')
    for split, games in split_games.items():
        temporary = directory / f'{split}_games.parquet.tmp'
        games.to_parquet(temporary, index=False)
        os.replace(temporary, directory / f'{split}_games.parquet')
    questions = pd.read_parquet(prepared / QUESTIONS_FILE)
    predicted = question_predictions(split_games['valid'], questions, table, centroids)
    predicted.to_parquet(directory / 'validation_predictions.parquet', index=False)
    metrics = rank_report(predicted.predicted_rank_id, predicted.rank_id, predicted.k)[['n', 'score', 'acc', 'within1']]
    metrics.to_csv(directory / 'official_scores.csv')
    matrices = {'questions_all': confusion(predicted.rank_id, predicted.predicted_rank_id)}
    for bucket in K_BUCKETS:
        selected = predicted.loc[predicted.k == bucket]
        matrices[f'questions_k{bucket}'] = confusion(selected.rank_id, selected.predicted_rank_id)
    valid = split_games['valid']
    game_predicted = nearest_centroid(valid.score_mean, centroids)
    matrices['single_game'] = confusion(valid.rank_id, game_predicted)
    single_game = rank_report(game_predicted, valid.rank_id, np.ones(len(valid), dtype=int)).loc[
        'all', ['n', 'score', 'acc', 'within1']].to_dict()
    single_game['n'] = len(valid)
    by_rank = []
    for level, row in enumerate(matrices['questions_all']):
        count = int(row.sum())
        exact = int(row[level])
        near = int(row[max(0, level - 1):min(10, level + 2)].sum()) - exact
        by_rank.append(dict(rank=RANKS[level], n=count, score=(exact + near * NEAR) / count if count else None,
                            acc=exact / count if count else None, within1=(exact + near) / count if count else None))
    pd.DataFrame(by_rank).to_csv(directory / 'validation_by_rank.csv', index=False)
    for name, matrix in matrices.items():
        save_array(directory / f'confusion_{name}.npy', matrix)
    curve = pd.DataFrame([json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()])
    columns = ['step', 'total', 'bt', 'policy', 'value', 'lr']
    if (not set(columns).issubset(curve.columns) or not np.isfinite(curve[columns]).all().all()
            or curve.step.tolist() != list(range(1, config['training']['updates'] + 1))):
        raise ValueError('learning curve requires complete, finite, consecutive training records')
    curve[columns].to_csv(directory / 'learning_curve.csv', index=False)
    result = metrics.to_dict(orient='index')
    save_plots(directory, statistics, edges, histograms, matrices, curve, result)
    report = dict(status='completed', identity=identity, centroids=centroids.tolist(), by_k=result,
                  single_game=single_game, per_rank=by_rank,
                  equal_k=float(metrics.loc[[str(k) for k in K_BUCKETS], 'score'].mean()),
                  positions={split: int(frame.position_count.sum()) for split, frame in split_games.items()},
                  protocol='Game-disjoint; players may overlap. Training centroids and questions are position weighted; nearest-centroid ties choose the lower rank index. within1 includes exact predictions. Scores are local validation, not leaderboard scores.')
    write_json(directory / 'report.json', report)
    print(json.dumps(dict(equal_k=report['equal_k'], by_k=result), allow_nan=False), flush=True)
    return report


def evaluate(config):
    validate_config(config)
    batch_size = config['evaluation']['per_gpu_batch_size']
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('evaluation per_gpu_batch_size must be positive')
    if config['device'] == 'cuda' and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise ValueError('CUDA evaluation requires eight processes')
    configure_torch(config)
    runtime = dict(config, training=dict(config['training'], global_batch_size=8),
                   evaluation=dict(global_batch_size=8))
    if dist.is_initialized():
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device('cuda', int(os.environ['LOCAL_RANK'])) if config['device'] == 'cuda' else torch.device('cpu')
    else:
        device, rank, world = setup_distributed(runtime)
    directory = Path(config['paths']['output_dir']) / 'evaluation'
    lock = exclusive(directory)
    locked = False
    dataset = None
    try:
        on_rank_zero(lambda: lock.__enter__(), rank)
        locked = True
        on_rank_zero(lambda: prepare_questions(config), rank)
        identity, payload = source_identity(config, world)
        def initialize():
            path = directory / 'identity.json'
            if path.exists():
                if json.loads(path.read_text()) != identity:
                    raise ValueError('evaluation identity changed; preserve existing output')
            else:
                if any(path.name != '.writer.lock' for path in directory.iterdir()):
                    raise ValueError('incomplete evaluation output without identity')
                write_json(path, identity)
        on_rank_zero(initialize, rank)
        model = StrengthEstimator(**config['model']).to(device)
        model.load_state_dict(payload['model'])
        del payload
        if identity['channels_last']:
            model = model.to(memory_format=torch.channels_last)
        for split in ('train', 'valid'):
            dataset = PaperDataset(config['paths']['dataset_dir'], split=split,
                cache_dir=config['paths']['cache_dir'], augment=False,
                max_open_shards=config['loader']['max_open_shards'])
            dataset._ensure_open()
            path = directory / f'{split}_scores_rank{rank:02d}.npy'
            if path.with_suffix('.json').exists():
                load_shard(directory, split, rank, world, len(dataset), identity)
            else:
                scores, report = infer_positions(model, dataset, config, device, batch_size=batch_size,
                    rank=rank, world=world, channels_last=identity['channels_last'], collect_resources=False)
                save_array(path, scores)
                write_json(path.with_suffix('.json'), dict(identity=object_hash(identity),
                    start=report['sample_start'], stop=report['sample_stop'], sha256=file_hash(path)))
                del scores
            dataset.close()
            dataset = None
            barrier()
        result = on_rank_zero(lambda: build_reports(config, identity), rank)
        return result
    finally:
        if dataset is not None:
            dataset.close()
        if locked:
            on_rank_zero(lambda: lock.__exit__(None, None, None), rank)
        if dist.is_initialized():
            dist.destroy_process_group()


def evaluate_saved(config):
    directory = Path(config['paths']['output_dir']) / 'evaluation'
    with exclusive(directory):
        identity = json.loads((directory / 'identity.json').read_text())
        current, _ = source_identity(config, identity['world_size'])
        if current != identity:
            raise ValueError('saved evaluation source changed')
        return build_reports(config, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--report-only', action='store_true', help='Rebuild reports/plots on CPU from saved scores')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    (evaluate_saved if args.report_only else evaluate)(config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
