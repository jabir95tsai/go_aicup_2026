"""Create fixed validation questions from the shared, prepared SE game split.

Run after prepare.py. This command reads metadata only, never raw SGF/boards.
All runs reuse these files regardless of model initialization/training seed.
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.cv import make_strength_v1_questions
from .config import DEFAULT_CONFIG, SPLIT_SEED, read_config, validate_config

QUESTION_SEED = 2026
QUESTIONS_FILE = 'validation_questions.parquet'
ELIGIBILITY_FILE = 'validation_eligibility.parquet'
MANIFEST_FILE = 'questions_manifest.json'


def prepare_questions(config):
    """Publish/reuse questions bound to one seed-2026 split and generator."""
    validate_config(config)
    directory = Path(config['paths']['dataset_dir'])
    source_path = directory / 'manifest.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source['identity'].get('split_seed') != SPLIT_SEED:
        raise ValueError('fixed SE questions require a split prepared with seed=2026')
    games_path = directory / 'games.parquet'
    games_hash = file_hash(games_path)
    if games_hash != source['files']['games.parquet']:
        raise ValueError('prepared games checksum mismatch')
    identity = {
        'format_version': 1, 'split_seed': SPLIT_SEED,
        'question_seed': QUESTION_SEED,
        'n_questions': config['questions']['validation'],
        'games_sha256': games_hash,
        'generator_sha256': {
            'questions.py': file_hash(Path(__file__)),
            'cv.py': file_hash(Path(__file__).parents[3] / 'data/cv.py'),
        },
    }
    manifest_path = directory / MANIFEST_FILE
    artifacts = [directory / name for name in (QUESTIONS_FILE, ELIGIBILITY_FILE)]
    if manifest_path.exists():
        record = json.loads(manifest_path.read_text(encoding='utf-8'))
        if record['identity'] != identity:
            raise ValueError('question identity changed; use a new versioned dataset directory')
        if set(record['files']) != {QUESTIONS_FILE, ELIGIBILITY_FILE}:
            raise ValueError('invalid question manifest files')
        for name, digest in record['files'].items():
            path = directory / name
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError(f'question checksum mismatch: {name}')
        return manifest_path
    if any(path.exists() or path.with_suffix(path.suffix + '.tmp').exists() for path in artifacts):
        raise ValueError('incomplete question output; inspect it before regenerating')
    table = pd.read_parquet(games_path).sort_values('game_index').reset_index(drop=True)
    # Sampling stays exclusively in data.cv, shared with other task adapters.
    questions, eligibility = make_strength_v1_questions(
        table, 'valid', n_questions=identity['n_questions'], seed=QUESTION_SEED)
    for path, frame in zip(artifacts, (questions, eligibility)):
        temporary = path.with_suffix(path.suffix + '.tmp')
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    write_json(manifest_path, {
        'identity': identity, 'build_id': object_hash(identity),
        'protocol': 'Same-player/color validation games; no repeated game within a question; game-disjoint from training.',
        'counts': {str(k): {str(rank): int(len(group)) for rank, group in bucket.groupby('rank_id')}
                   for k, bucket in questions.groupby('k')},
        'unique_games': len(set(int(game) for games in questions['games'] for game in games)),
        'unique_players': int(questions['player_id'].nunique()),
        'files': {path.name: file_hash(path) for path in artifacts},
    })
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(prepare_questions(read_config(args.config)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
