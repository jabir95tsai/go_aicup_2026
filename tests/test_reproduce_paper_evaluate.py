"""Synthetic-only checks for full-position statistics, decoding and final reports."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from go_aicup.data.board_cache import RANKS, convert_training
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.prepare import prepare
from go_aicup.tasks.strength.reproduce_paper.questions import prepare_questions
from go_aicup.tasks.strength.reproduce_paper.train import train
from go_aicup.tasks.strength.reproduce_paper.evaluate import (
    aggregate_positions, training_statistics, nearest_centroid, question_predictions,
    confusion, evaluate, evaluate_saved)


def test_all_position_centroids_and_histograms_are_position_weighted():
    table = pd.DataFrame(dict(game_index=np.arange(20), rank_id=np.repeat(np.arange(10), 2)))
    offsets = np.cumsum([0] + [1, 3] * 10)
    scores = np.concatenate([[level * 10, *([level * 10 + 4] * 3)] for level in range(10)]).astype(np.float32)
    games = aggregate_positions(scores, np.arange(20), offsets, table)
    statistics, edges, counts = training_statistics(scores, games, offsets)
    np.testing.assert_allclose(statistics['mean'], np.arange(10) * 10 + 3)
    np.testing.assert_allclose(statistics['std'], np.sqrt(3))
    np.testing.assert_array_equal(counts.sum(axis=1), [4] * 10)
    np.testing.assert_array_equal(statistics.positions, [4] * 10)
    assert len(edges) == 129 and games.iloc[1].position_count == 3
    assert games.iloc[1].score_sum == 12
    with pytest.raises(ValueError, match='coverage'):
        aggregate_positions(scores[:-1], np.arange(20), offsets, table)
    invalid = scores.copy()
    invalid[0] = np.nan
    with pytest.raises(ValueError, match='coverage'):
        aggregate_positions(invalid, np.arange(20), offsets, table)
    constant = np.ones_like(scores)
    stats, _, hist = training_statistics(constant, aggregate_positions(constant, np.arange(20), offsets, table), offsets)
    assert (stats['mean'] == 1).all() and hist.sum() == len(scores)


def test_questions_use_all_positions_and_centroid_ties_choose_lower_rank():
    table = pd.DataFrame(dict(game_index=np.arange(20), rank_id=0, player_id='p', color='B', split='valid'))
    games = pd.DataFrame(dict(game_index=np.arange(20), position_count=[1, 3] + [1] * 18,
                             score_sum=[0, 12] + [0] * 18))
    questions = pd.DataFrame([dict(question_id=i, player_id='p', color='B', rank_id=0,
        k=k, games=np.arange(k)) for i, k in enumerate((5, 10, 15, 20))])
    predicted = question_predictions(games, questions, table, np.arange(10))
    assert predicted.iloc[0].score_mean == pytest.approx(12 / 7)
    assert predicted.iloc[0].predicted_rank_id == 2
    np.testing.assert_array_equal(nearest_centroid([.5, 1.5], np.arange(10)), [0, 1])
    np.testing.assert_array_equal(nearest_centroid([9.5], np.arange(10)[::-1]), [0])
    matrix = confusion([0, 0, 1], [0, 1, 2])
    assert matrix[0, 0] == matrix[0, 1] == matrix[1, 2] == 1
    invalid = questions.copy(deep=True)
    invalid.at[0, 'games'] = np.zeros(5, dtype=int)
    with pytest.raises(ValueError, match='distinct'):
        question_predictions(games, invalid, table, np.arange(10))
    wrong = table.copy()
    wrong.loc[0, 'split'] = 'train'
    with pytest.raises(ValueError, match='validation'):
        question_predictions(games, questions, wrong, np.arange(10))
    with pytest.raises(ValueError, match='outside validation'):
        question_predictions(games.iloc[1:], questions, table, np.arange(10))


@pytest.fixture(scope='module')
def final_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('paper_evaluation')
    raw = root / 'raw'
    raw.mkdir()
    coords = [chr(97 + column) + chr(97 + row) for row in range(6) for column in range(10)]
    sgf = '(;SZ[19]' + ''.join(f';{"B" if ply % 2 == 0 else "W"}[{coord}]'
                               for ply, coord in enumerate(coords)) + ')'
    for rank in RANKS:
        with (raw / f'train_{rank}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['player_id', 'game_id', 'rank', 'color', 'sgf_content'])
            writer.writeheader()
            for game in range(24):
                writer.writerow(dict(player_id=f'{rank}_p', game_id=str(game), rank=rank,
                                     color='B', sgf_content=sgf))
    config = read_config()
    config.update(device='cpu', precision='fp32')
    config['paths'].update(cache_dir=str(root / 'cache'), dataset_dir=str(root / 'prepared'),
                           output_dir=str(root / 'run'), label_dir=str(root / 'labels'))
    config['split'] = dict(expected_games_per_rank=24, train_games_per_rank=4)
    config['questions']['validation'] = 40
    config['model'].update(channels=2, blocks=1)
    config['training'].update(groups_per_update=2, microbatch_groups=1, updates=4,
        lr_drop_after_updates=2, final_lr_after_updates=3, checkpoint_every=2,
        workers=0, deterministic=True, sync_batch_norm=False, channels_last=False)
    config['evaluation']['per_gpu_batch_size'] = 128
    convert_training(raw, Path(config['paths']['cache_dir']), workers=1, games_per_shard=100,
                     expected_games_per_rank=24)
    manifest = json.loads(prepare(config).read_text())
    prepare_questions(config)
    Path(config['paths']['label_dir']).mkdir()
    np.save(Path(config['paths']['label_dir']) / 'value.npy',
            np.zeros(manifest['splits']['train']['positions'], dtype=np.float32))
    train(config)
    return config


def test_full_evaluation_reuses_scores_and_rebuilds_plots(final_run, monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import evaluate as evaluator
    from go_aicup.tasks.strength.reproduce_paper import infer as inference
    def resource_monitor_called(*args, **kwargs):
        raise AssertionError('full evaluation must not collect resource usage')
    monkeypatch.setattr(inference, 'process_stats', resource_monitor_called)
    config = copy.deepcopy(final_run)
    result = evaluate(config)
    directory = Path(config['paths']['output_dir']) / 'evaluation'
    stats = pd.read_csv(directory / 'rank_statistics.csv')
    assert len(stats) == 10 and stats.positions.sum() == result['positions']['train']
    assert result['equal_k'] == pytest.approx(np.mean([result['by_k'][str(k)]['score'] for k in (5, 10, 15, 20)]))
    for k in (5, 10, 15, 20):
        row = result['by_k'][str(k)]
        assert row['score'] == pytest.approx(row['acc'] + (row['within1'] - row['acc']) / np.e)
        assert np.load(directory / f'confusion_questions_k{k}.npy').sum() == 10
    assert np.load(directory / 'confusion_questions_all.npy').sum() == 40
    assert np.load(directory / 'confusion_single_game.npy').sum() == 200
    assert len(list((directory / 'plot').glob('*.png'))) == 10
    assert len(pd.read_csv(directory / 'learning_curve.csv')) == 4
    assert not (directory / '.writer.lock').exists()
    score_files = list(directory.glob('*_scores_rank*.npy'))
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in score_files}
    def fail_inference(*args, **kwargs):
        raise AssertionError('finished shards must be reused')
    monkeypatch.setattr(evaluator, 'infer_positions', fail_inference)
    repeated = evaluate(config)
    assert repeated['centroids'] == result['centroids']
    rebuilt = evaluate_saved(config)
    assert rebuilt['by_k'] == result['by_k']
    for path, original in before.items():
        assert (path.stat().st_mtime_ns, path.read_bytes()) == original
    changed = copy.deepcopy(config)
    changed['evaluation']['per_gpu_batch_size'] += 1
    with pytest.raises(ValueError, match='identity changed'):
        evaluate(changed)
    assert not (directory / '.writer.lock').exists()
    corrupt = score_files[0]
    with corrupt.open('ab') as handle:
        handle.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        evaluate(config)
    assert not (directory / '.writer.lock').exists()


def test_train_main_runs_evaluation_only_after_full_completion(monkeypatch):
    from go_aicup.tasks.strength.reproduce_paper import train as trainer
    from go_aicup.tasks.strength.reproduce_paper import evaluate as evaluator
    config = read_config()
    calls = []
    monkeypatch.setattr(trainer, 'read_config', lambda path: config)
    monkeypatch.setattr(trainer, 'train', lambda *args, **kwargs: {'status': 'completed'})
    monkeypatch.setattr(evaluator, 'evaluate', lambda value: calls.append(value))
    assert trainer.main([]) == 0 and calls == [config]
    calls.clear()
    monkeypatch.setattr(trainer, 'train', lambda *args, **kwargs: {'status': 'stopped'})
    assert trainer.main([]) == 0 and not calls


def test_two_process_completed_entry_runs_evaluation(final_run, tmp_path):
    import os
    import subprocess
    import sys
    config = copy.deepcopy(final_run)
    run = tmp_path / 'two_rank_run'
    config['paths']['output_dir'] = str(run)
    path = tmp_path / 'two_rank_config.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '-m', 'go_aicup.tasks.strength.reproduce_paper.train', '--config', str(path)],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((run / 'evaluation' / 'report.json').read_text())
    assert report['identity']['world_size'] == 2
    assert len(list((run / 'evaluation').glob('*_scores_rank*.npy'))) == 4
    assert not (run / 'checkpoints').exists() and not (run / 'last.json').exists()
    assert not (run / 'evaluation' / '.writer.lock').exists()
