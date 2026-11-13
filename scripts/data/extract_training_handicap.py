"""Install official v1.1 CSVs and extract HD, verifying every row against v1.0.

Download the release archive, then run:
    python -m scripts.data.extract_training_handicap /path/to/aicup2026_go_training_v1.1.tar.xz
Existing training CSVs are required; reruns also accept already updated v1.1 files.
"""
import argparse
import csv
import json
import os
import re
import shutil
import tarfile
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from go_aicup.artifacts import file_hash
from go_aicup.paths import ROOT

SHA256 = 'e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e'
RELEASE = 'https://github.com/AILAB-NDHU/aicup2026-go-dataset/releases/tag/v1.1'
RANKS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
FIELDS = ['player_id', 'game_id', 'rank', 'color', 'sgf_content']
HD = re.compile(r'\(;SZ\[19\]HD\[([0-9]+)\](?=;)')


def without_hd(content):
    match = HD.match(content)
    return content if match is None else content.replace(f'HD[{match[1]}]', '', 1)


def validate_rank(updated, original, rank, writer):
    counts, by_color, seen = Counter(), {'B': Counter(), 'W': Counter()}, set()
    with updated.open(encoding='utf-8-sig', newline='') as new_file, original.open(
            encoding='utf-8-sig', newline='') as old_file:
        new_rows, old_rows = csv.DictReader(new_file), csv.DictReader(old_file)
        if new_rows.fieldnames != FIELDS or old_rows.fieldnames != FIELDS:
            raise ValueError(f'{rank}: unexpected CSV columns')
        for source_row, (new, old) in enumerate(zip_longest(new_rows, old_rows)):
            if new is None or old is None:
                raise ValueError(f'{rank}: row count changed')
            match = HD.match(new['sgf_content'])
            if match is None:
                raise ValueError(f'{rank} row {source_row}: missing/invalid root HD')
            if (new['rank'] != rank or new['color'] not in by_color
                    or new['game_id'] in seen):
                raise ValueError(f'{rank} row {source_row}: invalid identity/color')
            seen.add(new['game_id'])
            if ({**new, 'sgf_content': without_hd(new['sgf_content'])}
                    != {**old, 'sgf_content': without_hd(old['sgf_content'])}):
                raise ValueError(f'{rank} row {source_row}: changes beyond HD')
            handicap = int(match[1])
            counts[handicap] += 1
            by_color[new['color']][handicap] += 1
            writer.writerow([rank, new['game_id'], new['player_id'], new['color'],
                             source_row, handicap])
    return {'rows': sum(counts.values()), 'handicap_counts': dict(sorted(counts.items())),
            'by_color': {c: dict(sorted(v.items())) for c, v in by_color.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    if file_hash(args.archive) != SHA256:
        raise ValueError('archive SHA-256 does not match official v1.1')
    destination = ROOT / 'data/raw/training'
    expected = {f'training/train_{rank}.csv' for rank in RANKS}
    report = {'release': RELEASE, 'archive_sha256': SHA256,
              'all_rows_equal_after_removing_hd': True, 'training': {}, 'files': {}}
    # Stage on the same filesystem; validate all ten ranks before replacing any CSV.
    with tempfile.TemporaryDirectory(prefix='.v11-', dir=destination) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(args.archive, 'r|xz') as archive:
            for member in archive:
                if member.isdir() and member.name == 'training':
                    continue
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f'unexpected archive member: {member.name}')
                seen.add(member.name)
                with archive.extractfile(member) as source, (stage / Path(member.name).name).open('wb') as target:
                    shutil.copyfileobj(source, target)
        if seen != expected:
            raise ValueError('missing training CSVs in archive')
        with (stage / 'handicap.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['rank', 'game_id', 'player_id', 'color', 'source_row', 'handicap'])
            for rank in RANKS:
                name = f'train_{rank}.csv'
                result = validate_rank(stage / name, destination / name, rank, writer)
                if result['rows'] != 100_000:
                    raise ValueError(f'{rank}: expected 100,000 games')
                report['training'][rank] = result
                report['files'][name] = {'sha256': file_hash(stage / name),
                                         'previous_sha256': file_hash(destination / name)}
                print(rank, result, flush=True)
        report['files']['handicap.csv'] = {'sha256': file_hash(stage / 'handicap.csv')}
        (stage / 'handicap_manifest.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        for name in [*(f'train_{rank}.csv' for rank in RANKS), 'handicap.csv', 'handicap_manifest.json']:
            os.replace(stage / name, destination / name)
    print(f'Installed v1.1 and extracted 1,000,000 handicap rows: {destination}')


if __name__ == '__main__':
    main()
