"""H9 and C10 full-board policy shards in the original cvec2 game order."""
import json

import numpy as np
import pandas as pd
import torch

from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_large import cache
from go_aicup.tasks.strength.policy_large.run import parallel


class JointCache:
    def __init__(self, h, c):
        self.games = h.games
        if not h.games.equals(c.games) or not np.array_equal(h.bounds, c.bounds):
            raise ValueError("H/C game rows or position offsets differ")
        if h.arrays["c"].shape != (len(self.games), 18) or c.arrays["c"].shape != (len(self.games), 20):
            raise ValueError("expected H9 and C10 cache profiles")
        if not torch.equal(h.arrays["counts"], c.arrays["counts"]):
            raise ValueError("H/C per-side move counts differ")
        for hs, cs in zip(h.shards, c.shards, strict=True):
            if (hs.first, hs.last) != (cs.first, cs.last):
                raise ValueError("H/C shard game ranges differ")
            if not torch.equal(hs.target, cs.target.to(hs.target.device)):
                raise ValueError("H/C played moves differ")
            if not torch.equal(hs.side, cs.side.to(hs.side.device)):
                raise ValueError("H/C move sides differ")
        self.arrays = {
            "counts": h.arrays["counts"],
            "c": torch.cat((h.arrays["c"], c.arrays["c"]), dim=1),
        }
        self.h, self.c = h, c

    def for_device(self, device):
        return JointView(self.h.for_device(device), self.c.for_device(device), self.arrays, self.games)


class JointView:
    def __init__(self, h, c, arrays, games):
        self.h, self.c, self.arrays = h, c, arrays
        self.device, self.games = h.device, games

    def counts(self, rows):
        return self.h.counts(rows)

    def chunks(self, rows, chunk):
        # Whole-game ranges and targets/sides were compared once at cache construction.
        for left, right in zip(self.h.chunks(rows, chunk), self.c.chunks(rows, chunk), strict=True):
            h, target, qh, slot = left
            c, _, qc, _ = right
            yield torch.cat((h, c), dim=1), target, torch.cat((qh, qc), dim=1), slot


def infer(games, out):
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
    board = pd.read_parquet(ROOT / "data/processed/boards/strength_v1/games.parquet",
                            columns=["game_index", "length"]).set_index("game_index")
    length = board.loc[games.game_index, "length"].to_numpy(dtype=np.int64)
    if "length" in games and not np.array_equal(games.length, length):
        raise ValueError("cvec2 positions differ from the replay cache")
    games = games.assign(length=length)
    # Keep the original E1 replay for HD>=2; only HD1 receives komi 0.5.
    teacher_games = games.assign(handicap=games.handicap.eq(1).astype(np.int64))
    config = json.loads((ROOT / "configs/strength/policy_large_200k.json").read_text())
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    flags = matmul.allow_tf32, cudnn.allow_tf32
    try:
        matmul.allow_tf32 = cudnn.allow_tf32 = True
        result = parallel([
            (family, cache.infer_family, (family, teacher_games, config, out, cache.ABORT))
            for family in ("H", "C")
        ], 2)
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = flags
    joint = JointCache(result["H"][0], result["C"][0])
    joint.games = games
    return joint
