"""Fixed-question validation using synthetic metadata only; no GPU/raw data."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.tasks.strength.reproduce_paper.config import read_config
from go_aicup.tasks.strength.reproduce_paper.questions import prepare_questions


def prepared_metadata(directory):
    directory.mkdir()
    records = []
    for rank, name in enumerate(RANKS):
        for game in range(80):
            records.append(dict(game_index=rank * 80 + game, game_id=str(game),
                rank=name, rank_id=rank, player_id=f'p{rank}',
                color='B' if game % 2 == 0 else 'W',
                split='train' if game < 40 else 'valid'))
    table = pd.DataFrame(records)
    table.to_parquet(directory / 'games.parquet', index=False)
    write_json(directory / 'manifest.json', {
        'identity': {'split_seed': 2026},
        'files': {'games.parquet': file_hash(directory / 'games.parquet')},
    })
    config = read_config()
    config['paths']['dataset_dir'] = str(directory)
    config['questions']['validation'] = 80
    return config, table


def test_balanced_validation_questions_and_shared_seed(tmp_path):
    config, table = prepared_metadata(tmp_path / 'data')
    path = prepare_questions(config)
    record = json.loads(path.read_text())
    questions = pd.read_parquet(path.parent / 'validation_questions.parquet')
    assert len(questions) == 80
    assert (questions.groupby(['k', 'rank_id']).size() == 2).all()
    assert set(questions.k) == {5, 10, 15, 20}
    lookup = table.set_index('game_index')
    training = set(table.loc[table.split == 'train', 'game_index'])
    for q in questions.itertuples():
        assert len(q.games) == len(set(q.games)) == q.k
        assert not training.intersection(q.games)
        selected = lookup.loc[q.games]
        assert (selected.split == 'valid').all()
        assert (selected.player_id == q.player_id).all()
        assert (selected.color == q.color).all()
        assert (selected.rank_id == q.rank_id).all()
    eligibility = pd.read_parquet(path.parent / 'validation_eligibility.parquet')
    assert len(eligibility) == 40 and (eligibility.eligible_groups > 0).all()
    assert record['identity']['question_seed'] == 2026
    assert record['identity']['split_seed'] == 2026
    changed = copy.deepcopy(config)
    changed['seed'] = 2027
    before = file_hash(path.parent / 'validation_questions.parquet')
    assert prepare_questions(changed) == path
    assert file_hash(path.parent / 'validation_questions.parquet') == before
    # A separate build from identical metadata must choose identical questions.
    other, _ = prepared_metadata(tmp_path / 'other')
    other['seed'] = 1234
    other_path = prepare_questions(other)
    actual = pd.read_parquet(other_path.parent / 'validation_questions.parquet')
    pd.testing.assert_frame_equal(questions.drop(columns='games'), actual.drop(columns='games'))
    for expected, selected in zip(questions.games, actual.games):
        np.testing.assert_array_equal(expected, selected)


def test_changed_question_count_rejected(tmp_path):
    config, _ = prepared_metadata(tmp_path / 'data')
    prepare_questions(config)
    config['questions']['validation'] = 40
    with pytest.raises(ValueError, match='identity changed'):
        prepare_questions(config)


def test_corrupt_question_artifact_rejected(tmp_path):
    config, _ = prepared_metadata(tmp_path / 'data')
    path = prepare_questions(config)
    (path.parent / 'validation_questions.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare_questions(config)


def test_ineligible_validation_fails_before_publication(tmp_path):
    config, table = prepared_metadata(tmp_path / 'data')
    table.loc[(table.rank_id == 0) & (table.game_index >= 60), 'split'] = 'train'
    directory = tmp_path / 'data'
    table.to_parquet(directory / 'games.parquet', index=False)
    write_json(directory / 'manifest.json', {'identity': {'split_seed': 2026},
        'files': {'games.parquet': file_hash(directory / 'games.parquet')}})
    with pytest.raises(ValueError, match='no eligible players'):
        prepare_questions(config)
    assert not (directory / 'questions_manifest.json').exists()
    assert not (directory / 'validation_questions.parquet').exists()


def test_wrong_split_seed_and_source_corruption_rejected(tmp_path):
    config, _ = prepared_metadata(tmp_path / 'data')
    path = tmp_path / 'data' / 'manifest.json'
    record = json.loads(path.read_text())
    record['identity']['split_seed'] = 2027
    write_json(path, record)
    with pytest.raises(ValueError, match='seed=2026'):
        prepare_questions(config)
    record['identity']['split_seed'] = 2026
    write_json(path, record)
    (path.parent / 'games.parquet').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='games checksum mismatch'):
        prepare_questions(config)


@pytest.mark.parametrize('count', [0, 39, 41, True, 40.0])
def test_invalid_question_count_rejected(tmp_path, count):
    config, _ = prepared_metadata(tmp_path / 'data')
    config['questions']['validation'] = count
    with pytest.raises(ValueError, match='multiple of 40'):
        prepare_questions(config)
