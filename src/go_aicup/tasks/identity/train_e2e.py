"""End-to-end fine-tuning of the competition-1 identity encoder with the B aggregation head (2026-11-17).

Two arms share every setting but one:
    joint    encoder + B head trained together: each sampled move goes through the encoder, then
             through B's per-move residual blocks (with its ply embedding); the game vector is the
             mean of the B outputs; GE2E loss with B's w, b.
    encoder  the encoder alone, game vector = plain mean of its move vectors (the original
             training), GE2E loss with the encoder's w, b. Control for "more training steps".

Both start from the 9,600-step encoder (paths.init_model, full state incl. projection) and the joint
arm from the seed-2026 B head (paths.init_head). Sampling = train_ge2e (64 players x 8 games x 16
moves, both colours, player D4, HD <= 1) minus the 5 % per-rank players held out from B training
(seqagg.holdout_players over the per-move cache order, paths.move_cache); those players carry the
monitor (pure-style official score, 3 fixed draws), which is logged every e2e.monitor_every steps
as a curve only: the final step is the model. New runs use pool 0 draw 100 with
bench_monitor=true; init_optimizer restores encoder moments, while freeze_bn fixes BN statistics. Step 0 must reproduce the frozen monitor scores
(e2e.expect_step0), i.e. the joint arm at LR 0 equals the frozen 9,600 encoder + B.

Single GPU. Writes into paths.output_dir: config.json, train.jsonl, last.pt (every
checkpoint_steps; a rerun resumes), and at the end model.pt ({"model": encoder state}, readable by
scripts/identity/pid2_seqagg load_encoder) plus head.pt for the joint arm (seqagg.load_head format).

    python -m go_aicup.tasks.identity.train_e2e --config configs/identity/pid_e2e_joint_smoke100.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from go_aicup.data.board_cache import BoardCache
from go_aicup.tasks.identity import data
from go_aicup.tasks.identity.ge2e import MIN_PLY, GE2EBatches, IdentityEncoder, ge2e_loss
from go_aicup.tasks.identity.seqagg import Head, holdout_players, load_head, monitor_score, monitor_set, move_vectors
from go_aicup.tasks.identity.train_ge2e import read_config, training_players
from go_aicup.tasks.strength.shared.runtime import autocast_context
from go_aicup.tasks.strength.shared.training import environment, scheduler_for, seed_process

ARMS = ("joint", "encoder")


def training_mode(encoder, freeze_bn=False):
    encoder.train()
    if freeze_bn:
        for module in encoder.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()


def restore_encoder_optimizer(optimizer, encoder, saved):
    """Transfer encoder moments and step only; keep the new LR and fresh head state."""
    current = encoder.state_dict()
    if list(current) != list(saved["model"]) or any(
        not torch.equal(v.cpu(), saved["model"][k].cpu()) for k, v in current.items()
    ):
        raise ValueError("optimizer source weights/buffers differ from init_model")
    groups = saved["optimizer"]["param_groups"]
    params = list(encoder.parameters())
    if len(groups) != 1 or len(groups[0]["params"]) != len(params):
        raise ValueError("optimizer source does not match encoder parameter order")
    for key in ("betas", "eps", "weight_decay"):
        if groups[0][key] != optimizer.param_groups[0][key]:
            raise ValueError(f"optimizer source differs in {key}")
    source = torch.optim.AdamW(params)
    source.load_state_dict(copy.deepcopy(saved["optimizer"]))
    for param in params:
        state = source.state.get(param)
        if not state or any(state[k].shape != param.shape for k in ("exp_avg", "exp_avg_sq")):
            raise ValueError("missing or incompatible encoder optimizer moments")
        if not torch.isfinite(state["exp_avg"]).all() or not torch.isfinite(state["exp_avg_sq"]).all():
            raise ValueError("non-finite encoder optimizer moments")
        if float(state["step"]) != saved["step"]:
            raise ValueError("encoder optimizer step differs from source checkpoint")
    optimizer.state.update(source.state)
    return int(saved["step"])


class PlyBatches(GE2EBatches):
    """GE2EBatches item plus the sampled plies (the B head's position input)."""

    def __getitem__(self, item: int):
        planes = super().__getitem__(item)
        _, plies = self.sample(self.start + item)
        return planes, torch.from_numpy(plies.astype(np.int64))


def joint_games(head: Head, moves: torch.Tensor, plies: torch.Tensor, shape) -> torch.Tensor:
    """moves [N*M*S, D] (player, game, move order), plies [N*M*S] -> game vectors [N, M, D]."""
    n, m, s = shape
    x = moves.float().reshape(n * m, s, -1)
    valid = torch.ones(n * m, s, dtype=torch.bool, device=moves.device)
    return head(x, plies.reshape(n * m, s), valid).reshape(n, m, -1)


def mean_games(moves: torch.Tensor, shape) -> torch.Tensor:
    n, m, s = shape
    return moves.float().reshape(n, m, s, -1).mean(dim=2)


def split_players(config: dict, player_games: list[np.ndarray]):
    """(training player games, monitor sets, monitor store) with the B holdout removed.
    The per-move cache lists the same players in the same order; checked game by game."""
    cache = Path(config["paths"]["move_cache"])
    games = np.load(cache / "games.npy")
    players = np.load(cache / "players.npy")
    rank_ids = np.load(cache / "rank_ids.npy")
    expected = np.concatenate([np.sort(g) for g in player_games])
    if len(rank_ids) != len(player_games) or not np.array_equal(games, expected):
        raise ValueError("training players differ from the per-move cache the B head was trained on")
    held = holdout_players(rank_ids)
    store = SimpleNamespace(games=games, extra={"players": players})
    sets, eligible = monitor_set(store, held)
    return [g for g, h in zip(player_games, held) if not h], sets, int(held.sum()), eligible, games


class Monitor:
    """Fresh per-move vectors of the monitor games from the current encoder, then the pure-style
    official score of the plain mean (A) and of a head."""

    def __init__(self, config, sets, cache_games, device):
        self.config, self.device = config, device
        rows = np.unique(np.concatenate([np.concatenate(g + q) for g, q, _, _ in sets]))
        remap = lambda r: np.searchsorted(rows, r)
        self.sets = [([remap(x) for x in g], [remap(x) for x in q], k, t) for g, q, k, t in sets]
        self.games = cache_games[rows]
        meta = data.load_meta()
        self.colors = meta["player_color"].to_numpy().astype(np.int8)[self.games]
        self.lengths = meta["length"].to_numpy()[self.games]
        self.cache = BoardCache(data.CACHE_DIR)

    def __call__(self, encoder, heads: dict) -> dict:
        min_ply = self.config["sampling"]["min_ply"]
        vec, off = move_vectors(encoder, self.config, self.cache, self.games, self.colors, self.lengths,
                                self.device, min_ply)
        training_mode(encoder, self.config["e2e"].get("freeze_bn", False))
        store = SimpleNamespace(vectors=vec.astype(np.float16), offsets=off, games=self.games,
                                first_ply=self.colors.astype(np.int64) + min_ply)
        from go_aicup.tasks.identity.seqagg import MoveStore
        store.batch = MoveStore.batch.__get__(store)
        out = {"A": monitor_score(None, store, self.sets, self.device)}
        for name, head in heads.items():
            out[name] = monitor_score(head, store, self.sets, self.device)
        return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    e2e = config["e2e"]
    if e2e["arm"] not in ARMS:
        raise ValueError(f"e2e.arm must be one of {ARMS}")
    if config["sampling"].get("min_ply", MIN_PLY) != 0:
        raise ValueError("ply positions assume min_ply 0")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", False))
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = bool(
        config["training"].get("allow_tf32", False))
    seed_process(config["seed"])
    out = Path(config["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text()) != config:
        raise ValueError("output directory belongs to a different config")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "environment.json").write_text(json.dumps(environment(device, 1), indent=2))
    log_file = (out / "train.jsonl").open("a")

    def log(record):
        record = {**record, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        print(json.dumps(record), flush=True)

    player_games, lengths, excluded, player_color, info = training_players(config)
    train_games, sets, held, eligible, cache_games = split_players(config, player_games)
    if e2e.get("bench_monitor", False):
        from go_aicup.tasks.identity.e2e_probe import BenchMonitor
        monitor = BenchMonitor(config, device)
        if np.isin(monitor.games, np.concatenate(player_games)).any():
            raise ValueError("bench monitor overlaps encoder training games")
    else:
        monitor = Monitor(config, sets, cache_games, device)

    encoder = IdentityEncoder(**config["model"])
    encoder.load_state_dict(torch.load(config["paths"]["init_model"], map_location="cpu", weights_only=False)["model"])
    encoder.to(device)
    head = frozen_b = None
    if e2e["arm"] == "joint":
        head, head_step = load_head(config["paths"]["init_head"], device)
        params = list(encoder.parameters()) + list(head.parameters())
    else:
        frozen_b, head_step = load_head(config["paths"]["init_head"], device)   # diagnostic only
        frozen_b.requires_grad_(False)
        params = list(encoder.parameters())
    training = config["training"]
    optimizer = torch.optim.AdamW(params, **config["optimizer"])
    optimizer_source_step = None
    if config["paths"].get("init_optimizer") and not (out / "last.pt").exists():
        saved = torch.load(config["paths"]["init_optimizer"], map_location="cpu", weights_only=False)
        optimizer_source_step = restore_encoder_optimizer(optimizer, encoder, saved)
        del saved
    scheduler = scheduler_for(optimizer, training["steps"], training["warmup_fraction"], training["minimum_lr_ratio"])
    start = 0
    last = out / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("resume requires the same config")
        encoder.load_state_dict(saved["encoder"])
        if head is not None:
            head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["step"]
    log({"event": "start", "arm": e2e["arm"], "training_players": len(train_games), "held_out": held,
         "monitor_players": eligible, "excluded_bench_players": excluded, "init_head_step": head_step,
         "start_step": start, "optimizer_source_step": optimizer_source_step,
         "freeze_bn": e2e.get("freeze_bn", False), "bench_monitor": e2e.get("bench_monitor", False),
         "per_gpu_boards": int(np.prod([config["batch"][k] for k in ("players", "games", "moves")]))})

    def run_monitor(step):
        t0 = time.time()
        heads = {"B": head} if head is not None else {"B_frozen": frozen_b}
        scores = monitor(encoder, heads)
        log({"event": "monitor", "step": step, **{f"monitor_{k}": v for k, v in scores.items()},
             "seconds": time.time() - t0})
        return scores

    if start == 0:
        scores = run_monitor(0)
        expect = e2e.get("expect_step0")      # {"A": mean, "B": B head (joint) or frozen B (encoder), "tol"}
        if expect:
            got_b = scores.get("B", scores.get("B_frozen"))
            if max(abs(scores["A"] - expect["A"]), abs(got_b - expect["B"])) > expect["tol"]:
                raise SystemExit(f"step-0 monitor does not reproduce the frozen models: {scores} vs {expect}")

    stop = [False]
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.__setitem__(0, True))

    def save(step):
        payload = {"encoder": encoder.state_dict(), "head": head.state_dict() if head is not None else None,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step,
                   "config": config}
        torch.save(payload, out / "last.pt.tmp")
        os.replace(out / "last.pt.tmp", last)

    batch = config["batch"]
    shape = (batch["players"], batch["games"], batch["moves"])
    dataset = PlyBatches(config["paths"]["cache_dir"], train_games, lengths, players=shape[0], games=shape[1],
                         moves=shape[2], steps=training["steps"], seed=config["seed"], rank=0, start_step=start,
                         max_open_shards=config["loader"]["max_open_shards"], player_color=player_color,
                         min_ply=0, player_d4=config["sampling"].get("player_d4", False))
    workers = config["loader"]["num_workers"]
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=True,
                                         prefetch_factor=config["loader"]["prefetch_factor"] if workers else None,
                                         persistent_workers=workers > 0,
                                         multiprocessing_context="spawn" if workers else None)
    training_mode(encoder, e2e.get("freeze_bn", False))
    if head is not None:
        head.train()
    step, clock, window = start, time.monotonic(), []
    for x, plies in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        with autocast_context(config, device):
            moves = encoder(x)
        if head is not None:
            games = joint_games(head, moves, plies.to(device, non_blocking=True), shape)
            loss, accuracy = ge2e_loss(games, head.w, head.b)
        else:
            loss, accuracy = ge2e_loss(mean_games(moves, shape), encoder.w, encoder.b)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, training["gradient_clip_norm"])
        optimizer.step()
        scheduler.step()
        step += 1
        now = time.monotonic()
        window.append(now - clock)
        clock = now
        if step % training["log_every"] == 0 or step == training["steps"]:
            w, b = (head.w, head.b) if head is not None else (encoder.w, encoder.b)
            log({"event": "train", "step": step, "loss": float(loss), "own_centroid_accuracy": float(accuracy),
                 "w": float(w), "b": float(b), "grad_norm": float(grad_norm), "lr": optimizer.param_groups[0]["lr"],
                 "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2 ** 30,
                 "mean_step_seconds": float(np.mean(window[-training["log_every"]:]))})
        if step % e2e["monitor_every"] == 0 or step == training["steps"]:
            run_monitor(step)
            clock = time.monotonic()
        if step % training["checkpoint_steps"] == 0 or stop[0]:
            save(step)
        if stop[0]:
            log({"event": "stopped", "step": step})
            return 1
    save(step)
    del loader
    torch.save({"model": encoder.state_dict(), "config": config, "steps": step}, out / "model.pt")
    if head is not None:
        torch.save({"model": head.state_dict(), "step": step, "arm": "B", "position": True,
                    "seed": config["seed"], "init_head_step": head_step}, out / "head.pt")
    log({"event": "done", "step": step,
         "train_seconds_mean_step": float(np.mean(window[min(20, len(window) - 1):]))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
