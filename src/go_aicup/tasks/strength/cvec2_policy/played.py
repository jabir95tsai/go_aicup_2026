"""Reuse cvec2 played-move caches and fold missing H9 directly into normalization offsets."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import multiprocessing as mp
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from go_aicup.artifacts import write_json
from go_aicup.features import humansl
from go_aicup.paths import PROCESSED
from go_aicup.tasks.strength.policy_h.extract import PROFILES
from scripts.strength.rank_permove import OUT as C_CACHE
from scripts.strength.rank_permove import _side_mean

H_IDS = np.array([humansl.PROFILES.index(name) for name in PROFILES])
H_COLUMNS = np.r_[H_IDS, H_IDS + len(humansl.PROFILES)]
H_CACHE = (
    PROCESSED / "strength_features/humansl/p3_komi/h_train",
    PROCESSED / "strength_features/humansl/train20_hc/h_hd1",
)
H_MEAN_TOLERANCE = 0.015
C_MEAN_TOLERANCE = 0.005


def means_and_offsets(logp, slots, games):
    """Return raw and logsumexp means as [game, own/opp, channel] and [game, own/opp]."""
    if (logp.ndim != 2 or slots.shape != (len(logp),) or not np.isfinite(logp).all()
            or ((slots < 0) | (slots >= 2 * games)).any()):
        raise ValueError("invalid played-move likelihoods or side slots")
    normalizer = logsumexp(logp, axis=1, keepdims=True)
    joined = np.column_stack((logp, normalizer))
    means = _side_mean(joined, slots, None, 2 * games).reshape(games, 2, -1)
    return means[:, :, :-1].reshape(games, -1), means[:, :, -1]


def h_means(shard, table):
    """Read an original KataGo H shard, with games in its own saved order."""
    meta = json.loads((shard / "meta.json").read_text())
    profiles = meta["profiles"]
    ids = [profiles.index(name) for name in PROFILES]
    if meta["rules"] != humansl.RULES or meta["model"] != humansl.MODEL.name:
        raise ValueError(f"wrong H model or rules in {shard}")
    if not (meta["komi"] == np.where(table.handicap.eq(1), .5, 6.5)).all():
        raise ValueError(f"wrong H komi in {shard}")
    offsets, colors, done, logp = (np.load(shard / f"{name}.npy", allow_pickle=False)
                                  for name in ("offsets", "colors", "done", "logp"))
    if (offsets.shape != (len(table) + 1,) or offsets[0] != 0 or offsets[-1] != len(logp)
            or (np.diff(offsets) <= 0).any() or colors.shape != (len(logp),)
            or not np.isin(colors, [0, 1]).all()
            or logp.shape != (len(colors), len(profiles)) or done.shape != (len(table), len(profiles))
            or not done[:, ids].all()):
        raise ValueError(f"incomplete H played-move shard {shard}")
    game = np.repeat(np.arange(len(table)), np.diff(offsets))
    side = colors != table.player_color.to_numpy()[game]
    mean, norm = means_and_offsets(logp[:, ids], 2 * game + side, len(table))
    return mean, norm


def c_offsets(games, canon):
    target = pd.Index(games.game_index)
    offsets = np.full((len(games), 2), np.nan, dtype=np.float32)
    max_error = 0.0
    for name, hd in product(("valid", "t20"), (0, 1)):
        run = C_CACHE / f"c_{name}_hd{hd}"
        shards = sorted(run.glob("shard_*/meta.json"))
        if len(shards) != 8:
            raise ValueError(f"expected eight complete C shards in {run}")
        for marker in shards:
            shard = marker.parent
            table = pd.read_parquet(shard / "games.parquet")
            rows = target.get_indexer(table.game_index)
            done = np.load(shard / "done.npy", allow_pickle=False)
            if (not table.game_index.is_unique or (rows < 0).any()
                    or np.isfinite(offsets[rows]).any()
                    or not np.array_equal(games.iloc[rows].player_color, table.player_color)
                    or not (games.iloc[rows].handicap.eq(1) == bool(hd)).all()
                    or done.shape != (len(table),) or not done.all()):
                raise ValueError(f"C shard identity/coverage changed: {shard}")
            slot = np.load(shard / "move_slot.npy", allow_pickle=False)
            raw = np.load(shard / "move_logp.npy", allow_pickle=False).astype(np.float32)
            if raw.shape[1:] != (10,):
                raise ValueError(f"expected ten C likelihoods in {shard}")
            mean, norm = means_and_offsets(raw, slot, len(table))
            error = float(np.abs(mean - canon[rows]).max())
            if not np.isfinite(error) or error > C_MEAN_TOLERANCE:
                raise ValueError(f"C played-move means differ from cvec2 in {shard}: {error:.6g}")
            offsets[rows] = norm
            max_error = max(max_error, error)
    if not np.isfinite(offsets).all():
        raise ValueError("missing C played-move offsets")
    return offsets, max_error


def _h_worker(device, games, out):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for start in range(0, len(games), 2048):
        table = games.iloc[start:start + 2048]
        path = out / f"part_{start:06d}.npz"
        if not path.exists():
            # KataGo's existing extraction API writes raw arrays. Keep them on local scratch only.
            with TemporaryDirectory(prefix="cvec2-h9-") as scratch:
                scratch = Path(scratch)
                humansl.run(humansl.cache_games(table.game_index), PROFILES, scratch,
                            komi=6.5, overrides="numAnalysisThreads=64")
                mean, offsets = h_means(scratch, table)
                partial = path.with_suffix(".partial.npz")
                np.savez(partial, games=table.game_index, raw=mean, offsets=offsets)
                partial.replace(path)
        files.append(path)
    return files


def h_offsets(games, canon, out):
    target = pd.Index(games.game_index)
    offsets = np.full((len(games), 2), np.nan, dtype=np.float32)
    max_error = 0.0

    def ingest(ids, raw, norm, source):
        nonlocal max_error
        rows = target.get_indexer(ids)
        if ((rows < 0).any() or len(np.unique(rows)) != len(rows)
                or np.isfinite(offsets[rows]).any() or raw.shape != (len(rows), 18)
                or norm.shape != (len(rows), 2) or not np.isfinite(norm).all()):
            raise ValueError(f"invalid H normalization shard: {source}")
        error = float(np.abs(raw - canon[rows]).max())
        if not np.isfinite(error) or error > H_MEAN_TOLERANCE:
            raise ValueError(f"H played-move means differ from cvec2 in {source}: {error:.6g}")
        offsets[rows] = norm
        max_error = max(max_error, error)

    for source in H_CACHE:
        shards = sorted(source.glob("shard_*/meta.json"))
        if not shards:
            raise FileNotFoundError(f"missing original HD1 H played-move shards: {source}")
        for marker in shards:
            shard = marker.parent
            table = pd.read_parquet(shard / "games.parquet")
            rows = target.get_indexer(table.game_index)
            if ((rows < 0).any() or not np.array_equal(games.iloc[rows].player_color, table.player_color)
                    or not games.iloc[rows].handicap.eq(1).all()):
                raise ValueError(f"H shard identity changed: {shard}")
            raw, norm = h_means(shard, games.iloc[rows])
            ingest(table.game_index, raw, norm, shard)

    missing = games.loc[~np.isfinite(offsets).all(axis=1)]
    if missing.handicap.eq(1).any():
        raise ValueError("missing original HD1 H played-move coverage")
    reused = len(games) - len(missing)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(missing) and (any(not name for name in devices) or len(set(devices)) != len(devices)):
        raise ValueError("set CUDA_VISIBLE_DEVICES to distinct allocated GPUs for H backfill")
    if len(missing):
        with ProcessPoolExecutor(len(devices), mp_context=mp.get_context("spawn")) as pool:
            tasks = [pool.submit(_h_worker, device, missing.iloc[i::len(devices)],
                                 out / f"h_backfill_{i:02d}")
                     for i, device in enumerate(devices) if i < len(missing)]
            for task in tasks:
                for path in task.result():
                    with np.load(path, allow_pickle=False) as data:
                        ingest(data["games"], data["raw"], data["offsets"], path)
    if not np.isfinite(offsets).all():
        raise ValueError("missing H played-move offsets")
    return offsets, max_error, reused


def normalized_offsets(games, base, out):
    """Per game, own/opponent mean logsumexp for H9 and C10 in cvec2 order."""
    if not games.game_index.is_unique or base.shape != (len(games), 52):
        raise ValueError("invalid cvec2 games or feature matrix")
    c, c_error = c_offsets(games, base[:, 30:50])
    h, h_error, reused = h_offsets(games, base[:, H_COLUMNS], Path(out))
    write_json(Path(out) / "played_audit.json", {
        "games": len(games), "h_profiles": list(PROFILES),
        "c_reused_games": len(games), "h_reused_games": reused,
        "h_backfill_games": len(games) - reused,
        "h_raw_mean_max_error": h_error, "c_raw_mean_max_error": c_error,
    })
    return h, c
