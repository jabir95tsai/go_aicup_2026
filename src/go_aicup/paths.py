"""Repository paths, computed in one place.

Every module used to derive the repo root from its own `__file__` depth;
moving a file then silently redirected its reads and writes. All of them now
import `ROOT` from here instead.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
NOTEBOOKS = ROOT / "notebooks"
