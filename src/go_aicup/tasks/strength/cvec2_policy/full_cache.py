"""H9 and C10 full-board policy shards in the original cvec2 game order."""
import json
import time

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import write_json
from go_aicup.features.humansl import PROFILES
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_large import cache
from go_aicup.tasks.strength.policy_large.run import parallel

from .planes import PlaneReader


def a_config():
    """A experiment settings; historical experiments use the original JSON."""
    config = json.loads((ROOT / "configs/strength/policy_large_200k.json").read_text())
    return config | {"profiles_H": list(PROFILES), "inference_positions": {"H": 128, "C": 256},
                     "cache_fraction": .85}


class JointCache:
    def __init__(self, h, c):
        self.games = h.games
        if not h.games.equals(c.games) or not np.array_equal(h.bounds, c.bounds):
            raise ValueError("H/C game rows or position offsets differ")
        h_cols = h.arrays["c"].shape[1]
        if h_cols not in (18, 30) or c.arrays["c"].shape != (len(self.games), 20):
            raise ValueError("expected H9 or H15 and C10 cache profiles")
        if not torch.equal(h.arrays["counts"], c.arrays["counts"]):
            raise ValueError("H/C per-side move counts differ")
        for hs, cs in zip(h.shards, c.shards, strict=True):
            if (hs.first, hs.last) != (cs.first, cs.last):
                raise ValueError("H/C shard game ranges differ")
            if not torch.equal(hs.target, cs.target.to(hs.target.device)):
                raise ValueError("H/C played moves differ")
            if not torch.equal(hs.side, cs.side.to(hs.side.device)):
                raise ValueError("H/C move sides differ")
            if h_cols == 30:
                if hs.board is None or cs.board is not None:
                    raise ValueError("expected mover-stone planes on H15 shards only")
                for start in range(0, len(hs.target), 65536):
                    end = start + 65536
                    left = torch.isfinite(hs.logp[start:end, 0])
                    right = torch.isfinite(cs.logp[start:end, 0]).to(left.device)
                    if not torch.equal(left, right):
                        raise ValueError("H/C legal move masks differ")
        self.arrays = {
            "counts": h.arrays["counts"],
            "c": torch.cat((h.arrays["c"], c.arrays["c"]), dim=1),
        }
        self.offsets = {"H": h.offsets, "C": c.offsets}
        if h_cols == 30 and any(values is None for values in self.offsets.values()):
            raise ValueError("H15 and C10 must provide both game-side profile offsets")
        for family, values in self.offsets.items():
            if values is not None:
                self.arrays[f"offset_{family}"] = torch.from_numpy(values)
        self.h, self.c = h, c

    def for_device(self, device):
        return JointView(self.h.for_device(device), self.c.for_device(device), self.arrays, self.games)


class JointView:
    def __init__(self, h, c, arrays, games):
        self.h, self.c, self.arrays = h, c, arrays
        self.device, self.games = h.device, games

    def counts(self, rows):
        return self.h.counts(rows)

    @property
    def peer_bytes(self):
        return self.h.peer_bytes + self.c.peer_bytes

    def chunks(self, rows, chunk, *, board=False):
        # Whole-game ranges and targets/sides were compared once at cache construction.
        for left, right in zip(self.h.chunks(rows, chunk, board=board), self.c.chunks(rows, chunk), strict=True):
            h, target, qh, slot, *stones = left
            c, _, qc, _ = right
            yield torch.cat((h, c), dim=1), target, torch.cat((qh, qc), dim=1), slot, *stones


def infer(games, out, planes_dir=None, config=None, nominal_lengths=None):
    """Stream each teacher into four GPU shards and join them as batches are read."""
    if torch.cuda.device_count() != 8:
        raise ValueError("cvec2 full-policy training requires eight GPUs")
    for device in range(8):
        for other in range(8):
            if device != other and not torch.cuda.can_device_access_peer(device, other):
                raise ValueError("cross-family GPU peer access is required for policy shards")
    if not games.game_index.is_unique or not games.player_color.isin([0, 1]).all():
        raise ValueError("invalid cvec2 game ordering or player colors")
    if not games.handicap.isin(range(10)).all():
        raise ValueError("invalid cvec2 handicap metadata")
    plane_batches = None
    if planes_dir is None:
        board = pd.read_parquet(ROOT / "data/processed/boards/strength_v1/games.parquet",
                                columns=["game_index", "length"]).set_index("game_index")
        length = board.loc[games.game_index, "length"].to_numpy(dtype=np.int64)
        if "length" in games and not np.array_equal(games.length, length):
            raise ValueError("cvec2 positions differ from the replay cache")
        games = games.assign(length=length)
        # Keep the original E1 replay for HD>=2; only HD1 receives komi 0.5.
        teacher_games = games.assign(handicap=games.handicap.eq(1).astype(np.int64))
    else:
        teacher_games = games
        plane_batches = PlaneReader(planes_dir, teacher_games).batches
    if config is None:
        config = a_config() if planes_dir is not None else json.loads(
            (ROOT / "configs/strength/policy_large_200k.json").read_text()
        )
    if planes_dir is not None and tuple(config["profiles_H"]) != PROFILES:
        raise ValueError("H15 teacher profiles must match the original A column order")
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    flags = matmul.allow_tf32, cudnn.allow_tf32
    try:
        matmul.allow_tf32 = cudnn.allow_tf32 = True
        result = parallel([
            (family, cache.infer_family, (family, teacher_games, config, out, cache.ABORT, plane_batches, nominal_lengths))
            for family in ("H", "C")
        ], 2)
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = flags
    alignment_started = time.monotonic()
    joint = JointCache(result["H"][0], result["C"][0])
    if planes_dir is not None:
        write_json(out / "teacher_alignment.json", {"games": len(games), "positions": int(games.length.sum()),
                   "seconds": time.monotonic() - alignment_started, "legal_masks_equal": True})
    joint.games = games
    return joint
