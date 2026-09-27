"""Read access to the processed move stores written by `preprocess.py`.

`GameStore` memory-maps the flat move arrays, so opening the 1M-game training
store costs almost nothing and slices are zero-copy.

Colour normalisation (CLAUDE.md section 4): models should see moves as
"mine / theirs" relative to the player being judged, not black / white.
`perspective()` returns exactly that, using the store's `color` column (the
colour the labelled player held).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

MINE, THEIRS = 0, 1


class GameStore:
    def __init__(self, name: str = "train", directory: pathlib.Path = PROCESSED):
        self.name = name
        self.moves = np.load(directory / f"{name}_moves.npy", mmap_mode="r")
        self.colors = np.load(directory / f"{name}_colors.npy", mmap_mode="r")
        self.meta = pd.read_parquet(directory / f"{name}_meta.parquet")
        self._offset = self.meta["offset"].to_numpy(np.int64)
        self._length = self.meta["length"].to_numpy(np.int64)
        # colour held by the labelled player: 0 black, 1 white, -1 unknown
        if "color" in self.meta:
            self.player_color = (
                self.meta["color"].astype(str).map({"B": 0, "W": 1}).fillna(-1)
                .to_numpy(np.int8)
            )
        else:
            self.player_color = np.full(len(self.meta), -1, np.int8)

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def lengths(self) -> np.ndarray:
        return self._length

    def game(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        span = slice(self._offset[i], self._offset[i] + self._length[i])
        return self.moves[span], self.colors[span]

    def perspective(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(moves, side) with side MINE for the labelled player's moves."""
        moves, colors = self.game(i)
        side = (colors != self.player_color[i]).astype(np.int8)
        return moves, side

    def games(self, indices) -> list[tuple[np.ndarray, np.ndarray]]:
        return [self.game(int(i)) for i in indices]
