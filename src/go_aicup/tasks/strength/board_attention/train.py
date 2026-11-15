"""Train board-conditioned move weights; frozen C v2, two direct rank CEs.

Run on a Slurm compute node. Existing post-move packed boards are shifted
within each game, then put in the mover's perspective; no SGF replay is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import BoardCache, DEFAULT_OUTPUT
from go_aicup.paths import ROOT

OUT = ROOT / "outputs/rank/humansl/board_attention"
PERMOVE = ROOT / "outputs/rank/humansl/permove"
VECTORS = ROOT / "outputs/rank/humansl/cvec_v2/learn/vectors.pt"


def before_move(packed, colors):
    """Shift post-move snapshots inside ONE game, then black/white -> mover/other."""
    if packed.shape != (len(colors), 2, 46) or not np.isin(colors, (0, 1)).all():
        raise ValueError("invalid packed game or colors")
    result = np.zeros_like(packed)
    result[1:] = packed[:-1]
    white = colors == 1
    result[white] = result[white, ::-1]
    return result


def unpack(packed):
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = ((packed[..., None] >> shifts) & 1).flatten(2)[..., :361]
    return bits.reshape(-1, 2, 19, 19).float()


def move_features(logp, ply):
    mean = logp.mean(1, keepdim=True)
    return torch.cat((logp - mean, logp.std(1, correction=0, keepdim=True), mean, ply[:, None] / 100), 1)


def pool(scores, logp, slot, groups):
    """Stable independent softmax over each (game, own/opponent) segment."""
    top = scores.new_full((groups,), -torch.inf).scatter_reduce_(0, slot, scores, reduce="amax")
    weight = (scores - top[slot]).exp()
    denom = scores.new_zeros(groups).index_add_(0, slot, weight)
    weight = weight / denom[slot]
    values = logp.new_zeros((groups, 10)).index_add_(0, slot, weight[:, None] * logp)
    return values.reshape(-1, 2, 10), weight


class MoveWeights(nn.Module):
    def __init__(self, mean=None, std=None):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(2, 8, 5), nn.ReLU(), nn.Conv2d(8, 8, 5), nn.ReLU(),
                                  nn.Conv2d(8, 8, 5), nn.ReLU(), nn.Flatten(), nn.Linear(392, 10))
        self.head = nn.Sequential(nn.Linear(23, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("mean", torch.zeros(13) if mean is None else mean)
        self.register_buffer("std", torch.ones(13) if std is None else std)

    def forward(self, packed, logp, ply):
        board = self.conv(unpack(packed).contiguous(memory_format=torch.channels_last))
        aux = (move_features(logp, ply) - self.mean) / self.std
        return self.head(torch.cat((board, aux), 1)).float().squeeze(1)


def paired_ce(values, targets):
    # Each side and each game has equal weight, independent of game length.
    return F.cross_entropy(values.flatten(0, 1), targets.repeat_interleave(2))


class Games:
    """Resident packed boards/log p, with CPU game boundaries and player metadata."""
    def __init__(self, table, packed, logp, side, ply, bounds):
        self.table, self.packed, self.logp = table, packed, logp
        self.side, self.ply, self.bounds = side, ply, bounds
        self.device = logp.device
        self.targets = torch.as_tensor(table.rank_id.to_numpy().astype(np.int64), device=self.device)

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        lengths = self.bounds[rows + 1] - self.bounds[rows]
        ids = np.concatenate([np.arange(self.bounds[r], self.bounds[r + 1]) for r in rows])
        ids = torch.as_tensor(ids, device=self.device)
        group = torch.as_tensor(np.repeat(np.arange(len(rows)), lengths), device=self.device)
        slot = 2 * group + self.side[ids].long()
        return self.packed[ids], self.logp[ids].float(), self.ply[ids].float(), slot, self.targets[rows]


def metadata():
    from scripts.strength import rank_cvec_v2 as cv2, rank_move_weights as old
    _, fit, valid, _ = cv2.data()
    table = pd.concat((fit.assign(pool=True), valid.assign(pool=False)), ignore_index=True)
    # Reuse the exact existing split, rather than generating another player partition.
    table = old.split_players(table[table.handicap == 0]).sort_values("game_index").reset_index(drop=True)
    if not table.game_index.is_unique or (table.groupby("player_id")["set"].nunique() != 1).any():
        raise ValueError("duplicate games or overlapping players")
    return table


def load_data(out, device):
    started = time.monotonic()
    table = metadata()
    cache = BoardCache(DEFAULT_OUTPUT, max_open_shards=16)
    gi = table.game_index.to_numpy()
    if not np.array_equal(cache.rank_ids[gi], table.rank_id) or not np.array_equal(cache.player_colors[gi], table.player_color):
        raise ValueError("board metadata differs from likelihood metadata")
    bounds = np.r_[0, np.cumsum(cache.lengths[gi], dtype=np.int64)]
    packed = np.empty((bounds[-1], 2, 46), np.uint8)
    logp = np.empty((bounds[-1], 10), np.float16)
    side = np.empty(bounds[-1], np.uint8)
    ply = np.empty(bounds[-1], np.int16)
    lookup = pd.Series(np.arange(len(table)), index=gi)
    loaded = np.zeros(len(table), bool)
    source = []
    for run in ("c_valid_hd0", "c_t20_hd0"):
        shards = sorted((PERMOVE / run).glob("shard_*"))
        if len(shards) != 8:
            raise ValueError(f"{run}: expected 8 likelihood shards")
        for shard in shards:
            info = json.loads((shard / "meta.json").read_text())
            if Path(info["vectors"]) != VECTORS.relative_to(ROOT) or info["finetuned"] is not None or info["komi"] != 6.5:
                raise ValueError(f"unexpected teacher: {shard}")
            games = pd.read_parquet(shard / "games.parquet")
            done = np.load(shard / "done.npy")
            slots = np.load(shard / "move_slot.npy", mmap_mode="r")
            plies = np.load(shard / "move_ply.npy", mmap_mode="r")
            values = np.load(shard / "move_logp.npy", mmap_mode="r")
            reference = np.load(shard / "feature.npy", mmap_mode="r")
            if not (len(slots) == len(plies) == len(values)) or np.any(np.diff(slots // 2) < 0):
                raise ValueError("likelihood positions are not in game order")
            counts = np.bincount(slots // 2, minlength=len(games))
            edges = np.r_[0, np.cumsum(counts)]
            worst = 0.0
            for local in np.flatnonzero(games.game_index.isin(gi).to_numpy()):
                game = int(games.game_index.iloc[local])
                row = int(lookup.loc[game])
                start, stop = bounds[row:row + 2]
                span = slice(edges[local], edges[local + 1])
                colors = cache.colors(game)
                sides = (colors != cache.player_colors[game]).astype(np.uint8)
                lp = np.asarray(values[span])
                if (loaded[row] or not done[local] or len(lp) != stop - start
                        or not np.array_equal(plies[span], np.arange(len(colors)))
                        or not np.array_equal(slots[span] % 2, sides) or not np.isfinite(lp).all()
                        or not (sides == 0).any() or not (sides == 1).any()):
                    raise ValueError(f"incomplete or misaligned likelihood: {game}")
                mean = np.concatenate([lp[sides == s].astype(np.float32).mean(0) for s in (0, 1)])
                worst = max(worst, float(np.abs(mean - reference[local]).max()))
                packed[start:stop] = before_move(cache.packed(game), colors)
                logp[start:stop], side[start:stop], ply[start:stop] = lp, sides, plies[span]
                loaded[row] = True
            if worst > 0.005:
                raise ValueError(f"mean no longer reproduces C v2: {worst}")
            source.append({"path": str(shard.relative_to(ROOT)), "meta_sha256": file_hash(shard / "meta.json"),
                           "mean_max_abs_error": worst})
            print(json.dumps({"loaded_games": int(loaded.sum()), "seconds": round(time.monotonic() - started, 1)}), flush=True)
    cache.close()
    if not loaded.all():
        raise ValueError(f"{int((~loaded).sum())} missing games")
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "games.parquet", index=False)
    manifest = {"splits": table.groupby("set").agg(games=("game_index", "size"), players=("player_id", "nunique")).to_dict("index"),
                "positions": int(bounds[-1]), "sources": source, "vectors_sha256": file_hash(VECTORS),
                "board_manifest_sha256": file_hash(DEFAULT_OUTPUT / "manifest.json"),
                "board_time": "before move; post-move cache shifted within game", "perspective": "mover, other",
                "seconds": time.monotonic() - started}
    write_json(out / "data.json", manifest)
    return Games(table, torch.as_tensor(packed, device=device), torch.as_tensor(logp, device=device),
                 torch.as_tensor(side, device=device), torch.as_tensor(ply, device=device), bounds)


def normalisation(data):
    total, square, count = torch.zeros(13, device=data.device, dtype=torch.float64), torch.zeros(13, device=data.device, dtype=torch.float64), 0
    # All and only training positions contribute to the fixed feature transform.
    for chunk in np.array_split(np.flatnonzero(data.table["set"].to_numpy() == "train"), 128):
        _, lp, ply, _, _ = data.batch(chunk)
        x = move_features(lp, ply).double()
        total += x.sum(0)
        square += x.square().sum(0)
        count += len(x)
    mean = total / count
    return mean.float(), ((square / count - mean.square()).clamp_min(1e-12).sqrt().clamp_min(1e-6)).float()


class PlayerSampler:
    def __init__(self, table):
        train = table[table["set"] == "train"]
        self.groups = [[g.index.to_numpy() for _, g in train[train.rank_id == r].groupby("player_id", sort=True)] for r in range(10)]

    def sample(self, n, rng):
        if any(len(groups) < n for groups in self.groups):
            raise ValueError("batch asks for more distinct players than available")
        return np.array([rng.choice(groups[p]) for groups in self.groups
                         for p in rng.choice(len(groups), n, replace=False)], dtype=np.int64)


def forward(model, batch):
    packed, lp, ply, slot, target = batch
    with torch.autocast(device_type=lp.device.type, dtype=torch.bfloat16, enabled=lp.is_cuda):
        scores = model(packed, lp, ply)
    values, weights = pool(scores.float(), lp, slot, 2 * len(target))
    return paired_ce(values, target), values, weights


@torch.no_grad()
def evaluate(model, data, rows, batch_games=128, collect=False):
    model.eval()
    sums = np.zeros(4)
    features = []
    for start in range(0, len(rows), batch_games):
        selected = rows[start:start + batch_games]
        batch = data.batch(selected)
        _, values, weights = forward(model, batch)
        target = batch[-1]
        ce = F.cross_entropy(values.flatten(0, 1), target.repeat_interleave(2), reduction="none").reshape(-1, 2)
        squared = weights.new_zeros(2 * len(target)).index_add_(0, batch[3], weights.square())
        maximum = weights.new_zeros(2 * len(target)).scatter_reduce_(0, batch[3], weights, reduce="amax")
        sums += np.array([*ce.sum(0).cpu().tolist(), float((1 / squared).sum()), float(maximum.sum())])
        if collect:
            features.append(values.float().cpu().numpy().reshape(-1, 20))
    result = {"ce_own": float(sums[0] / len(rows)), "ce_opp": float(sums[1] / len(rows)),
              "ce": float(sums[:2].sum() / (2 * len(rows))), "effective_moves": float(sums[2] / (2 * len(rows))),
              "max_weight": float(sums[3] / (2 * len(rows)))}
    return result, np.concatenate(features) if collect else None


def atomic_save(state, path):
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def benchmark(data, config, out):
    sampler = PlayerSampler(data.table)
    mu, sd = normalisation(data)
    results = []
    for n in (32, 64, 128):
        torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        model = MoveWeights(mu, sd).to(data.device, memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
        torch.cuda.reset_peak_memory_stats()
        elapsed, positions = [], 0
        try:
            for step in range(34):
                torch.cuda.synchronize()
                start = time.monotonic()
                batch = data.batch(sampler.sample(n, rng))
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = forward(model, batch)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= 2:
                    elapsed.append(time.monotonic() - start)
                    positions += len(batch[1])
            results.append({"per_rank": n, "positions_per_second": positions / sum(elapsed),
                            "step_seconds": float(np.mean(elapsed)), "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
        except torch.OutOfMemoryError:
            results.append({"per_rank": n, "oom": True})
        del model, optimizer
        torch.cuda.empty_cache()
        print(json.dumps(results[-1]), flush=True)
    good = [r for r in results if "oom" not in r]
    if not good:
        raise RuntimeError("all benchmark batches exhausted GPU memory")
    fastest = max(r["positions_per_second"] for r in good)
    # Prefer more optimizer updates per processed game if throughput is nearly equal.
    selected = min(r["per_rank"] for r in good if r["positions_per_second"] >= fastest * 0.95)
    hold = np.flatnonzero(data.table["set"].to_numpy() == "hold")
    model = MoveWeights(mu, sd).to(data.device, memory_format=torch.channels_last)
    torch.cuda.synchronize()
    start = time.monotonic()
    baseline, _ = evaluate(model, data, hold)
    torch.cuda.synchronize()
    monitor_seconds = time.monotonic() - start
    chosen = next(r for r in good if r["per_rank"] == selected)
    result = {"batches": results, "selected_per_rank": selected, "hold_seconds": monitor_seconds,
              "baseline": baseline, "estimated_train_monitor_hours":
              (chosen["step_seconds"] * config["updates"] + monitor_seconds * (config["updates"] // config["every"] + 1)) / 3600,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    write_json(out / "benchmark.json", result)
    print(json.dumps(result), flush=True)


def train(data, config, out):
    bench = json.loads((out / "benchmark.json").read_text())
    n = config["per_rank"]
    from go_aicup.artifacts import object_hash
    identity = json.loads((out / "data.json").read_text())
    identity.pop("seconds")
    data_id = object_hash(identity)
    rng = np.random.default_rng(config["seed"])
    torch.manual_seed(config["seed"])
    mu, sd = normalisation(data)
    model = MoveWeights(mu, sd).to(data.device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sampler = PlayerSampler(data.table)
    hold = np.flatnonzero(data.table["set"].to_numpy() == "hold")
    step, bad, best, best_step = 0, 0, bench["baseline"]["ce"], 0
    log = [{"step": 0, "hold": bench["baseline"]}]
    if (out / "last.pt").exists():
        saved = torch.load(out / "last.pt", map_location=data.device, weights_only=False)
        if saved["config"] != config or saved["per_rank"] != n or saved["data_id"] != data_id:
            raise ValueError("resume configuration changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["rng"]
        step, bad, best, best_step, log = (saved[k] for k in ("step", "bad", "best", "best_step", "log"))
    else:
        atomic_save({"model": model.state_dict(), "step": 0, "config": config}, out / "best.pt")
    start = time.monotonic()
    recent = []
    while step < config["updates"]:
        model.train()
        batch = data.batch(sampler.sample(n, rng))
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = forward(model, batch)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training CE")
        loss.backward()
        optimizer.step()
        recent.append(float(loss.detach()))
        step += 1
        if step % 25 == 0:
            print(json.dumps({"step": step, "train_ce": float(np.mean(recent)), "seconds": time.monotonic() - start}), flush=True)
        if step % config["every"] == 0 or step == config["updates"]:
            report, _ = evaluate(model, data, hold)
            if report["ce"] < best:
                best, best_step, bad = report["ce"], step, 0
                atomic_save({"model": model.state_dict(), "step": step, "config": config}, out / "best.pt")
            else:
                bad = bad + 1 if step >= config["min_updates"] else 0
            log.append({"step": step, "train_ce": float(np.mean(recent)), "hold": report,
                        "best_step": best_step, "seconds": time.monotonic() - start})
            recent.clear()
            write_json(out / "learning_curve.json", log)
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.bit_generator.state,
                         "step": step, "bad": bad, "best": best, "best_step": best_step, "log": log,
                         "config": config, "per_rank": n, "data_id": data_id}, out / "last.pt")
            print(json.dumps(log[-1]), flush=True)
            if step >= config["min_updates"] and bad >= config["patience"]:
                break
    saved = torch.load(out / "best.pt", map_location=data.device, weights_only=False)
    model.load_state_dict(saved["model"])
    atomic_save(saved, out / "model.pt")
    rows = np.arange(len(data.table))
    report, features = evaluate(model, data, rows, collect=True)
    np.savez(out / "features.npz", game_index=data.table.game_index.to_numpy(), c=features)
    write_json(out / "complete.json", {"updates": step, "best_step": best_step, "best_hold_ce": best,
                                      "seconds": time.monotonic() - start, "all_game_diagnostic": report})
    print(json.dumps({"complete": True, "updates": step, "best_step": best_step, "seconds": time.monotonic() - start}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("benchmark", "train"))
    p.add_argument("--config", type=Path, default=ROOT / "configs/strength/board_attention.json")
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    if config["updates"] < config["min_updates"] or min(config["every"], config["patience"], config["lr"]) <= 0:
        raise ValueError("invalid training budget")
    out = ROOT / config["out"]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    if args.command == "train" and (out / "complete.json").exists():
        print((out / "complete.json").read_text(), flush=True)
        return
    data = load_data(out, device)
    (benchmark if args.command == "benchmark" else train)(data, config, out)


if __name__ == "__main__":
    main()
