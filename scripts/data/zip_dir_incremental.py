"""Move a directory into a ZIP one group of files at a time, under a tight disk quota.

Each group (one cache shard directory, then the remaining loose files) is appended,
the ZIP is closed and fsynced, every new member is read back and compared by SHA-256
with its source, and only then is the source group deleted.  Re-running resumes:
groups whose members are already in the ZIP are re-verified before deletion.

Usage: python -m scripts.data.zip_dir_incremental <dir> [--level 1]
Writes <dir>.zip (members prefixed by the directory name) and a journal under
data/processed/migrations/.
"""
import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

from go_aicup.paths import ROOT


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def groups(src):
    """Shard directories first (sorted), then every remaining file as one final group."""
    shard_dirs = sorted(p for p in src.rglob('*') if p.is_dir() and any(c.is_file() for c in p.iterdir())
                        and p != src and p.name.startswith('unit_'))
    for d in shard_dirs:
        yield d, sorted(p for p in d.iterdir() if p.is_file())
    rest = sorted(p for p in src.rglob('*') if p.is_file() and not any(d in p.parents for d in shard_dirs))
    if rest:
        yield src, rest


def check_marker(files):
    """Cross-check shard arrays against the hashes the extraction job recorded."""
    marker = next((f for f in files if f.name == 'complete.json'), None)
    if marker is None or not marker.parent.name.startswith('unit_'):
        return
    recorded = json.loads(marker.read_bytes())['files']
    for name, digest in recorded.items():
        if sha256((marker.parent / f'{name}.npy').read_bytes()) != digest:
            raise ValueError(f'source differs from complete.json before zipping: {marker.parent}/{name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()
    src = args.src.resolve()
    target = src.with_suffix('.zip')
    base = src.parent
    log_dir = ROOT / 'data/processed/migrations' / f'zip_{src.name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    journal = (log_dir / 'journal.jsonl').open('a')
    started = time.time()
    totals = dict(groups=0, files=0, bytes=0)
    for group, files in groups(src):
        arcs = {f: f.relative_to(base).as_posix() for f in files}
        if group != src:
            check_marker(files)
        digests = {f: sha256(f.read_bytes()) for f in files}
        present = set()
        if target.exists():
            with zipfile.ZipFile(target) as archive:
                present = set(archive.namelist())
        todo = [f for f in files if arcs[f] not in present]
        if todo:
            with zipfile.ZipFile(target, 'a', zipfile.ZIP_DEFLATED, compresslevel=args.level) as archive:
                for f in todo:
                    archive.write(f, arcs[f])
            with target.open('rb+') as handle:
                os.fsync(handle.fileno())
        with zipfile.ZipFile(target) as archive:
            names = archive.namelist()
            for f in files:
                if names.count(arcs[f]) != 1:
                    raise ValueError(f'member missing or duplicated: {arcs[f]}')
                if sha256(archive.read(arcs[f])) != digests[f]:
                    raise ValueError(f'read-back mismatch: {arcs[f]}')
        size = sum(f.stat().st_size for f in files)
        for f in files:
            f.unlink()
        journal.write(json.dumps(dict(group=group.relative_to(base).as_posix(), files=len(files), bytes=size,
                                      sha256={arcs[f]: digests[f] for f in files},
                                      zip_bytes=target.stat().st_size, t=round(time.time() - started, 1))) + '\n')
        journal.flush()
        os.fsync(journal.fileno())
        totals['groups'] += 1
        totals['files'] += len(files)
        totals['bytes'] += size
        if totals['groups'] % 25 == 0:
            print(json.dumps(dict(**totals, zip_bytes=target.stat().st_size,
                                  seconds=round(time.time() - started))), flush=True)
    for d in sorted((p for p in src.rglob('*') if p.is_dir()), key=lambda p: -len(p.parts)):
        d.rmdir()
    src.rmdir()
    print(json.dumps(dict(done=True, **totals, zip_bytes=target.stat().st_size,
                          seconds=round(time.time() - started))), flush=True)


if __name__ == '__main__':
    main()
