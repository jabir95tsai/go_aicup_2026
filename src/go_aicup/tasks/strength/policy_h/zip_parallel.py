"""Multi-process version of scripts/data/zip_dir_incremental.py for the H policy cache.

Same archive layout and safety rules (members prefixed by the directory name; shard
directories first, then the remaining loose files; every member read back and compared
by SHA-256 before its source is deleted; a journal under data/processed/migrations/;
re-running resumes), but the CPU work runs in parallel:

* worker processes read each file, hash it (SHA-256, CRC-32) and deflate it with the
  exact raw-deflate settings zipfile uses (compressobj(level, DEFLATED, -15));
* the main process appends the precompressed members in order (single writer);
* after each batch the ZIP is closed and fsynced, workers re-open it and verify their
  groups' members in parallel, and only then are the sources deleted.

Shard arrays are also cross-checked against the hashes in their complete.json before
being written, as the original tool does.

    python -m go_aicup.tasks.strength.policy_h.zip_parallel <dir> [--level 1] [--workers 11]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import time
import zipfile
import zlib
from pathlib import Path

from go_aicup.paths import ROOT


def groups(src):
    """Shard directories first (sorted), then every remaining file as one final group."""
    shard_dirs = sorted(p for p in src.rglob('*') if p.is_dir() and any(c.is_file() for c in p.iterdir())
                        and p != src and p.name.startswith('unit_'))
    out = [(d, sorted(p for p in d.iterdir() if p.is_file())) for d in shard_dirs]
    rest = sorted(p for p in src.rglob('*') if p.is_file() and not any(d in p.parents for d in shard_dirs))
    if rest:
        out.append((src, rest))
    return out


def compress(path, level, present):
    """(sha256, crc32, size, raw deflate blob or None when already in the ZIP) for one file."""
    data = Path(path).read_bytes()
    blob = None
    if not present:
        compressor = zlib.compressobj(level, zlib.DEFLATED, -15)
        blob = compressor.compress(data) + compressor.flush()
    return hashlib.sha256(data).hexdigest(), zlib.crc32(data), len(data), blob


def verify(target, members):
    """Read back members [(arcname, sha256)] from the closed ZIP; return the mismatching names."""
    bad = []
    with zipfile.ZipFile(target) as archive:
        for arc, digest in members:
            if hashlib.sha256(archive.read(arc)).hexdigest() != digest:
                bad.append(arc)
    return bad


class _Flushed:
    """Stand-in compressor: the data was deflated by a worker and written already."""

    @staticmethod
    def flush():
        return b''


def append(archive, path, arc, level, crc, size, blob):
    info = zipfile.ZipInfo.from_file(path, arc)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.compress_level = level
    with archive.open(info, 'w', force_zip64=size * 1.05 > zipfile.ZIP64_LIMIT) as handle:
        handle._compressor = _Flushed()
        handle._fileobj.write(blob)
        handle._file_size, handle._crc, handle._compress_size = size, crc, len(blob)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('--level', type=int, default=1)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--batch', type=int, default=32, help='groups per write/verify/delete cycle')
    args = parser.parse_args()
    src = args.src.resolve()
    target = src.with_suffix('.zip')
    base = src.parent
    log_dir = ROOT / 'data/processed/migrations' / f'zip_{src.name}'
    log_dir.mkdir(parents=True, exist_ok=True)
    journal = (log_dir / 'journal.jsonl').open('a')
    started = time.time()
    totals = dict(groups=0, files=0, bytes=0)
    work = groups(src)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for first in range(0, len(work), args.batch):
            batch = work[first:first + args.batch]
            present = set()
            if target.exists():
                with zipfile.ZipFile(target) as archive:
                    present = set(archive.namelist())
            jobs = []
            for group, files in batch:
                arcs = {f: f.relative_to(base).as_posix() for f in files}
                futures = {f: pool.submit(compress, str(f), args.level, arcs[f] in present) for f in files}
                jobs.append((group, files, arcs, futures))
            digests = {}
            with zipfile.ZipFile(target, 'a', zipfile.ZIP_DEFLATED, compresslevel=args.level) as archive:
                for group, files, arcs, futures in jobs:
                    results = {f: futures[f].result() for f in files}
                    digests.update({f: results[f][0] for f in files})
                    marker = next((f for f in files if f.name == 'complete.json'), None)
                    if group != src and marker is not None:
                        for name, digest in json.loads(marker.read_bytes())['files'].items():
                            if digests[marker.parent / f'{name}.npy'] != digest:
                                raise ValueError(f'source differs from complete.json before zipping: {group}/{name}')
                    for f in files:
                        sha, crc, size, blob = results[f]
                        if blob is not None:
                            append(archive, f, arcs[f], args.level, crc, size, blob)
            with target.open('rb+') as handle:
                os.fsync(handle.fileno())
            with zipfile.ZipFile(target) as archive:
                names = archive.namelist()
            counts = {}
            for name in names:
                counts[name] = counts.get(name, 0) + 1
            for group, files, arcs, _ in jobs:
                for f in files:
                    if counts.get(arcs[f]) != 1:
                        raise ValueError(f'member missing or duplicated: {arcs[f]}')
            checks = [pool.submit(verify, str(target), [(arcs[f], digests[f]) for f in files])
                      for _, files, arcs, _ in jobs]
            for check in checks:
                bad = check.result()
                if bad:
                    raise ValueError(f'read-back mismatch: {bad}')
            for group, files, arcs, _ in jobs:
                size = sum(f.stat().st_size for f in files)
                for f in files:
                    f.unlink()
                journal.write(json.dumps(dict(group=group.relative_to(base).as_posix(), files=len(files), bytes=size,
                                              sha256={arcs[f]: digests[f] for f in files},
                                              zip_bytes=target.stat().st_size,
                                              t=round(time.time() - started, 1))) + '\n')
                totals['groups'] += 1
                totals['files'] += len(files)
                totals['bytes'] += size
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps(dict(**totals, zip_bytes=target.stat().st_size,
                                  seconds=round(time.time() - started))), flush=True)
    for d in sorted((p for p in src.rglob('*') if p.is_dir()), key=lambda p: -len(p.parts)):
        d.rmdir()
    src.rmdir()
    print(json.dumps(dict(done=True, **totals, zip_bytes=target.stat().st_size,
                          seconds=round(time.time() - started))), flush=True)


if __name__ == '__main__':
    main()
