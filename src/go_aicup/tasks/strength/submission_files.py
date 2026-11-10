"""Rank submission files: short, distinct names and no duplicate content (2026-11-09).

The upload page truncates long file names ("ahc-p3-noA-t20-a...csv"), so files that share a
long prefix look identical, and a byte-identical copy written by a later job was uploaded by
mistake. New files are named ``<job>-<tag>.csv``:

- the job id comes first: it is unique per run and sorts chronologically;
- the tag names the arm in at most 8 characters (lower case, digits, hyphens);
- the whole name stays at 19 characters or fewer, so the page shows it in full.

RESULTS.md maps each file to its full lineage. If another file in the directory already has the
same content, nothing is written and that file is returned.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from go_aicup.paths import ROOT

RANK_DIR = ROOT / "submissions" / "rank"
MAX_TAG = 8
_TAG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def write_rank_csv(result: pd.DataFrame, tag: str, job: str, directory: Path = RANK_DIR) -> Path:
    """Write ``question_id,pred_rank`` rows as ``<job>-<tag>.csv`` unless the content already exists."""
    if len(tag) > MAX_TAG or not _TAG.fullmatch(tag):
        raise ValueError(f"tag must be <= {MAX_TAG} chars of [a-z0-9-]: {tag!r}")
    if list(result.columns) != ["question_id", "pred_rank"]:
        raise ValueError(f"unexpected columns {list(result.columns)}")
    content = result.to_csv(index=False, lineterminator="\r\n").encode()
    digest = hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    for existing in sorted(directory.glob("*.csv")):
        if existing.stat().st_size == len(content) and hashlib.sha256(existing.read_bytes()).hexdigest() == digest:
            print(f"{tag}: same content as {existing.name} (sha256 {digest[:8]}); not written", flush=True)
            return existing
    output = directory / f"{job}-{tag}.csv"
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(content)
    print(f"wrote {output} (sha256 {digest[:8]})", flush=True)
    return output
