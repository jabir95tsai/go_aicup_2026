"""C4: HumanSL fine-tuned as a Yugen-rank-conditioned policy (RESULTS A-057).

Starts from HumanSL (plain weights) with the metadata encoder replaced by the
A-055 rank vectors (e[rank, own/opp]), trains the policy path (conv_spatial,
linear_global, blocks, norm_trunkfinal, policy_head) and the vectors on packed
positions (features/humansl_rankvec.py pack) of the F players. The network stays
in eval mode: its one batch norm (trunk end) keeps its running statistics.

Each process keeps its share of the pack_* shards (shard i on rank i % world)
in GPU memory and draws its own batches; gradients are averaged by DDP.
Monitor (Bm games, outputs/humansl_finetune/monitor): mean NLL of the played
move under the true rank on a fixed position subset, and the single-game official
score (sum of log p under each rank hypothesis -> softmax -> score_rank). Step 0
is C2 (A-055) exactly.

--mmi LAMBDA (C4b) adds a rank-discrimination term: every position is evaluated
under all 10 rank hypotheses and loss = NLL(true rank) + LAMBDA * CE(true rank |
softmax over ranks of log p(played move)). Positions per step are then --batch,
forward passes --batch x 10.

C4x (cross-fitted retry, RESULTS 2026-11-11) adds: --vectors (initial rank vectors),
--monitor (a directory of pack_* shards; default <data>/monitor/pack_0 as C4),
--rank-balance (each position weighted by 1 / its rank's share, so every rank
weighs the same) and --no-step-checkpoints (only last.pt). Defaults reproduce C4.
--mmi-adjacent: the rank CE runs over {r-1, r, r+1} of the true rank r only (3 forward passes per
position, 2 at D / 6D), so the gradient goes to separating neighbouring ranks, which is what
score_rank and the official metric reward. The monitor also reports the mid-range (1D/2D, 2D/3D,
3D/4D) single-game d' of the per-move log-likelihood ratio, own / opponent / all moves.

C4q (player-level loss, RESULTS 2026-11-12), --player-ce LAMBDA with --players FILE --steps N:
every process holds all shards; a step draws --player-batch players (rank uniform, then a player
of that rank uniform) and --player-positions random positions of each (all of the player's games,
both sides). Positions are evaluated under {r-1, r, r+1}; per player the mean log p of the own-side
and of the opponent-side positions are scaled by two learned factors and summed into one logit per
hypothesis; loss = NLL(true rank) + LAMBDA * CE(true rank | player logits). A player whose pooled
evidence points to the wrong rank has a large CE, so the gradient aims at player-level bias, not
at single positions. --block-dropout K: each step only K randomly chosen trunk blocks (same choice
on every process) receive their gradient; --freeze-nonblock keeps stem / trunk norm / policy head
fixed. The monitor adds player-level figures (per-move means averaged over a player's games).

    torchrun --standalone --nproc-per-node=8 -m go_aicup.tasks.strength.humansl_finetune \\
        --data outputs/humansl_finetune --out outputs/c4_s2 [--max-positions N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from go_aicup.features import humansl_rankvec as rv
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT

VECTORS = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec" / "vectors.pt"   # A-055 (moved 10/28)
TRAINABLE = ("conv_spatial.", "linear_global.", "blocks.", "norm_trunkfinal.", "policy_head.")
EVAL_POSITIONS = 409   # x RANKS evaluations per compiled monitor batch (as A-055 extract)


class RankPolicy(torch.nn.Module):
    """HumanSL whose metadata-encoder output is e[rank, side]; returns main-policy logits [B, 362]."""

    def __init__(self, device, vectors: torch.Tensor | None = None):
        super().__init__()
        self.net, init = rv.load_frozen(device)
        self.vectors = torch.nn.Parameter((vectors if vectors is not None else init.repeat(rv.RANKS, 2, 1)).to(device))
        for name, p in self.net.named_parameters():
            p.requires_grad_(name.startswith(TRAINABLE))

    def forward(self, x, g, rank, side):
        return self.net(x, g, input_meta=self.vectors[rank, side])[0][0][:, 0, :].float()


def legal_logp(logits, x, target):
    legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).reshape(len(x), rv.AREA) > 0
    legal = torch.cat([legal, torch.ones_like(legal[:, :1])], dim=1)
    return torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1).gather(1, target[:, None])[:, 0]


def load_shards(paths, device, players: pd.Series | None = None) -> dict:
    """Packed positions of the given pack_* directories, on device. players (game_index -> player_id)
    adds game_player (numpy, player code per game) and the positions grouped by player
    (player_order / player_start / player_count / player_rank) for C4q."""
    parts = {name: [] for name in ("bits", "globals", "target", "side", "rank", "game")}
    owners = []
    offset = 0
    for path in paths:
        games = pd.read_parquet(path / "games.parquet")
        game = np.load(path / "game.npy")
        parts["bits"].append(np.load(path / "bits.npy"))
        parts["globals"].append(np.load(path / "globals.npy"))
        parts["target"].append(np.load(path / "target.npy").astype(np.int64))
        parts["side"].append((~np.load(path / "own.npy")).astype(np.int64))
        parts["rank"].append(games["rank_id"].to_numpy().astype(np.int64)[game])
        parts["game"].append(game.astype(np.int64) + offset)
        offset += len(games)
        if players is not None:
            owners.append(players.loc[games["game_index"].to_numpy()].to_numpy())
    out = {name: torch.as_tensor(np.concatenate(value), device=device) for name, value in parts.items()}
    out["games"] = offset
    if players is not None:
        codes, _ = pd.factorize(np.concatenate(owners), sort=True)
        out["game_player"] = codes
        position_player = codes[np.concatenate(parts["game"])]
        order = np.argsort(position_player, kind="stable")
        count = np.bincount(position_player, minlength=codes.max() + 1)
        game_rank = np.zeros(offset, dtype=np.int64)
        game_rank[np.concatenate(parts["game"])] = np.concatenate(parts["rank"])
        player_rank = np.zeros(len(count), dtype=np.int64)
        player_rank[codes] = game_rank
        out["player_order"] = torch.as_tensor(order, device=device)
        out["player_start"] = torch.as_tensor(np.concatenate([[0], np.cumsum(count)[:-1]]), device=device)
        out["player_count"] = torch.as_tensor(count, device=device)
        out["player_rank"] = torch.as_tensor(player_rank, device=device)
    return out


def player_metrics(sides: np.ndarray, moves: np.ndarray, truth: np.ndarray, game_player: np.ndarray) -> dict:
    """Player-level monitor: per-move mean log p of each game (own / opp / all), averaged over the
    player's games. Mid-range d' as mid_dprime; official score of softmax(T * all-moves mean) at the
    best T of a fixed grid (the same optimistic rule at every step)."""
    views = {"own": sides[:, 0] / np.maximum(moves[:, :1], 1), "opp": sides[:, 1] / np.maximum(moves[:, 1:], 1),
             "all": sides.sum(1) / np.maximum(moves.sum(1, keepdims=True), 1)}
    n = np.bincount(game_player).astype(float)
    p_truth = np.zeros(len(n), dtype=np.int64)
    p_truth[game_player] = truth
    out, pooled = {}, {}
    for name, per_move in views.items():
        pooled[name] = np.stack([np.bincount(game_player, weights=per_move[:, r], minlength=len(n))
                                 for r in range(per_move.shape[1])], axis=1) / n[:, None]
        out[f"player_mid_d_{name}"] = round(float(np.nanmean([pair_dprime(pooled[name], p_truth, r)
                                                              for r in MID_PAIRS])), 4)
    best = 0.0
    for t in (50, 100, 200, 400, 800, 1600):
        z = t * (pooled["all"] - pooled["all"].max(1, keepdims=True))
        prob = np.exp(z)
        decoded = score_rank(prob / prob.sum(1, keepdims=True))
        best = max(best, float(((decoded == p_truth) + (np.abs(decoded - p_truth) == 1) / np.e).mean()))
    out["player_official"] = round(best, 4)
    out["players"] = int(len(n))
    return out


def setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    return rank, world, torch.device("cuda", local)


def all_sum(t: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(t)
    return t


MID_PAIRS = (4, 5, 6)   # 1D/2D, 2D/3D, 3D/4D


def pair_dprime(per_move: np.ndarray, truth: np.ndarray, r: int) -> float:
    """Signed single-game d' of the log-likelihood ratio log p(r+1) - log p(r) (per-move means)
    between games of rank r and r+1 (positive = the right direction)."""
    s = per_move[:, r + 1] - per_move[:, r]
    a, b = s[truth == r], s[truth == r + 1]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))


def mid_dprime(totals: np.ndarray, counts: np.ndarray, truth: np.ndarray) -> dict:
    """totals [games, 2 (own, opp), RANKS] sum of log p; counts [games, 2]. Mid-range (1D-4D) d'."""
    views = {"own": totals[:, 0] / np.maximum(counts[:, :1], 1), "opp": totals[:, 1] / np.maximum(counts[:, 1:], 1),
             "all": totals.sum(1) / np.maximum(counts.sum(1, keepdims=True), 1)}
    out = {}
    for name, per_move in views.items():
        pairs = [pair_dprime(per_move, truth, r) for r in MID_PAIRS]
        out[f"mid_d_{name}"] = round(float(np.nanmean(pairs)), 4)
        if name == "all":
            out["mid_d_pairs"] = [round(p, 4) for p in pairs]
    return out


@torch.no_grad()
def monitor(model: RankPolicy, forward, data: dict, probe: torch.Tensor, rank: int, world: int) -> dict:
    """NLL under the true rank on probe positions, single-game official score over all monitor games,
    and the mid-range (1D-4D) single-game d' of own / opponent / all moves (C4x)."""
    device = data["bits"].device
    nll = torch.zeros(2, device=device)
    for start in range(0, len(probe), 4096):
        index = probe[start:start + 4096]
        x = rv.unpack(data["bits"][index])
        logits = forward(x, data["globals"][index], data["rank"][index], data["side"][index])
        nll += torch.stack([-legal_logp(logits, x, data["target"][index]).sum(), torch.tensor(len(index), device=device)])
    totals = torch.zeros((data["games"] * 2, rv.RANKS), device=device)    # slot = game * 2 + side
    moves = torch.zeros(data["games"] * 2, device=device)
    mine = torch.nonzero(data["game"] % world == rank)[:, 0]
    ranks = torch.arange(rv.RANKS, device=device).repeat(EVAL_POSITIONS)
    for start in range(0, len(mine), EVAL_POSITIONS):
        index = mine[start:start + EVAL_POSITIONS]
        n = len(index)
        if n < EVAL_POSITIONS:   # fixed shape for the compiled forward
            index = torch.cat([index, index[-1:].repeat(EVAL_POSITIONS - n)])
        x = rv.unpack(data["bits"][index]).repeat_interleave(rv.RANKS, 0)
        logits = forward(x, data["globals"][index].repeat_interleave(rv.RANKS, 0), ranks,
                         data["side"][index].repeat_interleave(rv.RANKS, 0))
        logp = legal_logp(logits, x, data["target"][index].repeat_interleave(rv.RANKS, 0)).reshape(-1, rv.RANKS)[:n]
        slot = data["game"][index[:n]] * 2 + data["side"][index[:n]]
        totals.index_add_(0, slot, logp)
        moves.index_add_(0, slot, torch.ones(n, device=device))
    nll, totals, moves = all_sum(nll), all_sum(totals), all_sum(moves)
    truth = torch.zeros(data["games"], dtype=torch.long, device=device)
    truth.index_put_((data["game"],), data["rank"])
    sides = totals.reshape(-1, 2, rv.RANKS).cpu().numpy()
    moves, truth = moves.reshape(-1, 2).cpu().numpy(), truth.cpu().numpy()
    totals = sides.sum(1)
    prob = np.exp(totals - totals.max(1, keepdims=True))
    decoded = score_rank(prob / prob.sum(1, keepdims=True))
    official = (decoded == truth) + (np.abs(decoded - truth) == 1) / np.e
    extra = player_metrics(sides, moves, truth, data["game_player"]) if "game_player" in data else {}
    return {"monitor_nll": round(float(nll[0] / nll[1]), 5), "monitor_official": round(float(official.mean()), 4),
            "monitor_accuracy": round(float((totals.argmax(1) == truth).mean()), 4),
            **mid_dprime(sides, moves, truth), **extra}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dir with pack_* (F) and monitor/pack_0 (Bm)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-positions", type=int, help="stop after this many training positions (all ranks)")
    parser.add_argument("--batch", type=int, default=512, help="positions per process per step")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--vector-lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--mmi", type=float, default=0.0, help="weight of the rank-discrimination CE (C4b)")
    parser.add_argument("--monitor-every", type=int, default=2000)
    parser.add_argument("--monitor-positions", type=int, default=200_000)
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="steps between last.pt saves")
    parser.add_argument("--shards", type=int, help="use only the first N pack shards (pilot)")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--vectors", type=Path, default=VECTORS, help="initial rank vectors (vectors.pt)")
    parser.add_argument("--monitor", type=Path, help="dir of monitor pack_* shards (default <data>/monitor/pack_0)")
    parser.add_argument("--rank-balance", action="store_true", help="weight positions so every rank weighs the same")
    parser.add_argument("--no-step-checkpoints", action="store_true", help="keep only last.pt")
    parser.add_argument("--mmi-adjacent", action="store_true",
                        help="with --mmi: the rank CE is over the true rank and its neighbours only (3 hypotheses)")
    parser.add_argument("--player-ce", type=float, default=0.0, help="C4q: weight of the player-level rank CE")
    parser.add_argument("--players", type=Path, help="C4q: parquet game_index, player_id of learn and monitor games")
    parser.add_argument("--steps", type=int, help="C4q: optimiser steps")
    parser.add_argument("--player-batch", type=int, default=8, help="C4q: players per process per step")
    parser.add_argument("--player-positions", type=int, default=32, help="C4q: positions per player per step")
    parser.add_argument("--block-dropout", type=int, default=0,
                        help="each step only this many random trunk blocks are updated (0 = all)")
    parser.add_argument("--freeze-nonblock", action="store_true", help="train trunk blocks and vectors only")
    args = parser.parse_args(argv)
    if args.player_ce and (args.players is None or not args.steps or args.mmi):
        parser.error("--player-ce needs --players and --steps, and excludes --mmi")
    rank, world, device = setup()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    def log(entry: dict) -> None:
        if rank == 0:
            print(json.dumps(entry), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")

    shards = sorted(p for p in args.data.glob("pack_*") if (p / "meta.json").exists())[:args.shards]
    players = None
    if args.players is not None:
        players = pd.read_parquet(args.players).drop_duplicates("game_index").set_index("game_index")["player_id"]
    data = load_shards(shards if args.player_ce else shards[rank::world], device, players if args.player_ce else None)
    monitor_shards = [args.data / "monitor" / "pack_0"] if args.monitor is None else \
        sorted(p for p in args.monitor.glob("pack_*") if (p / "meta.json").exists())
    if not monitor_shards:
        raise FileNotFoundError(f"no monitor shards in {args.monitor}")
    mon = load_shards(monitor_shards, device, players)
    probe = torch.as_tensor(np.random.default_rng(args.seed).permutation(len(mon["target"]))[:args.monitor_positions],
                            device=device)
    # per-rank position weights: total / (RANKS * count_r) over all processes' data (mean weight 1)
    counts = all_sum(torch.bincount(data["rank"], minlength=rv.RANKS).double())
    rank_weight = (counts.sum() / (rv.RANKS * counts.clamp(min=1))).float() if args.rank_balance \
        else torch.ones(rv.RANKS, device=device)
    model = RankPolicy(device, torch.load(args.vectors, map_location=device)["vectors"])
    if args.freeze_nonblock:
        for name, p in model.net.named_parameters():
            p.requires_grad_(name.startswith("blocks."))
    trainable = [p for n, p in model.net.named_parameters() if p.requires_grad]
    block_params: dict[int, list] = {}
    for name, p in model.net.named_parameters():
        if p.requires_grad and name.startswith("blocks."):
            block_params.setdefault(int(name.split(".")[1]), []).append(p)
    if args.block_dropout and not 0 < args.block_dropout <= len(block_params):
        raise ValueError(f"--block-dropout must be within 1..{len(block_params)}")
    # C4q: log scale of the own-side / opponent-side mean log p in the player logit
    player_scale = torch.nn.Parameter(torch.full((2,), math.log(100.0), device=device))
    optimizer = torch.optim.AdamW([{"params": trainable, "lr": args.lr},
                                   {"params": [model.vectors], "lr": args.vector_lr}]
                                  + ([{"params": [player_scale], "lr": 1e-2}] if args.player_ce else []),
                                  weight_decay=0.0)
    local = torch.tensor(len(data["target"]) // args.batch, device=device)
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local)
    total = int(math.ceil(args.epochs * steps_per_epoch))
    if args.player_ce:
        steps_per_epoch = total = args.steps
        by_rank = [torch.nonzero((data["player_rank"] == r) & (data["player_count"] > 0))[:, 0] for r in range(rv.RANKS)]
    if args.max_positions:
        total = min(total, args.max_positions // (args.batch * world))
    base = [group["lr"] for group in optimizer.param_groups]
    step, started_epoch = 0, 0
    checkpoint = args.out / "last.pt"
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if "player_scale" in state:
            player_scale.data.copy_(state["player_scale"])
    trained = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    train_forward = trained if args.no_compile else torch.compile(trained)
    eval_forward = model if args.no_compile else torch.compile(model, dynamic=False)
    log({"event": "start", "world": world, "shards": len(shards), "positions_rank0": len(data["target"]),
         "steps_per_epoch": steps_per_epoch, "total_steps": total, "global_batch": args.batch * world,
         "trainable_parameters": sum(p.numel() for p in trainable), "resume_step": step,
         "trunk_blocks": len(block_params), "players": int((data["player_count"] > 0).sum()) if args.player_ce else None,
         "rank_positions": counts.long().tolist(), "rank_weight": [round(float(w), 4) for w in rank_weight],
         "monitor_shards": [str(p) for p in monitor_shards], "args": vars(args) | {
             "data": str(args.data), "out": str(args.out), "vectors": str(args.vectors),
             "monitor": str(args.monitor), "players": str(args.players)}})
    if step == 0:
        log({"step": 0, **monitor(model, eval_forward, mon, probe, rank, world)})
    started, window, window_disc, seen, monitor_seconds = time.monotonic(), [], [], 0, 0.0
    hypotheses = torch.arange(rv.RANKS, device=device).repeat(args.batch)
    ADJACENT = torch.tensor([-1, 0, 1], device=device)

    def save(name: str) -> None:
        if rank == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                        "player_scale": player_scale.detach()}, args.out / "tmp.pt")
            os.replace(args.out / "tmp.pt", args.out / name)

    while step < total:
        epoch = step // steps_per_epoch
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000 * epoch + rank)
        order = None if args.player_ce else torch.randperm(len(data["target"]), generator=generator, device=device)
        for local_step in range(step % steps_per_epoch, steps_per_epoch):
            if step >= total:
                break
            scale = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
            for group, lr in zip(optimizer.param_groups, base):
                group["lr"] = lr * scale
            if args.player_ce:
                draw = torch.Generator(device=device).manual_seed(args.seed + 7919 * (step * world + rank) + 1)
                p_rank = torch.randint(rv.RANKS, (args.player_batch,), generator=draw, device=device)
                u = torch.rand(args.player_batch, generator=draw, device=device)
                chosen = torch.stack([by_rank[r][int(u[i] * len(by_rank[r]))] for i, r in enumerate(p_rank.tolist())])
                u = torch.rand((args.player_batch, args.player_positions), generator=draw, device=device)
                offset = (u * data["player_count"][chosen][:, None]).long()
                index = data["player_order"][data["player_start"][chosen][:, None] + offset].reshape(-1)
            else:
                index = order[local_step * args.batch:(local_step + 1) * args.batch]
            x = rv.unpack(data["bits"][index])
            if args.player_ce:
                truth = data["rank"][index]
                hyp = (truth[:, None] + ADJACENT).clamp(0, rv.RANKS - 1)
                xr = x.repeat_interleave(len(ADJACENT), 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(xr, data["globals"][index].repeat_interleave(len(ADJACENT), 0), hyp.reshape(-1),
                                           data["side"][index].repeat_interleave(len(ADJACENT), 0))
                logp = legal_logp(logits, xr, data["target"][index].repeat_interleave(len(ADJACENT), 0)) \
                    .reshape(args.player_batch, args.player_positions, len(ADJACENT))
                nll = -logp[:, :, 1].mean()
                opp = data["side"][index].reshape(args.player_batch, args.player_positions, 1).float()
                own_mean = (logp * (1 - opp)).sum(1) / (1 - opp).sum(1).clamp(min=1)
                opp_mean = (logp * opp).sum(1) / opp.sum(1).clamp(min=1)
                z = player_scale[0].exp() * own_mean + player_scale[1].exp() * opp_mean          # [P, 3]
                valid = (p_rank[:, None] + ADJACENT >= 0) & (p_rank[:, None] + ADJACENT < rv.RANKS)
                discrimination = -torch.log_softmax(z.masked_fill(~valid, float("-inf")), dim=1)[:, 1].mean()
                loss = nll + args.player_ce * discrimination
                window_disc.append(discrimination.detach())
            elif args.mmi and args.mmi_adjacent:
                truth = data["rank"][index]
                hyp = (truth[:, None] + ADJACENT).clamp(0, rv.RANKS - 1)               # [B, 3]
                valid = (truth[:, None] + ADJACENT >= 0) & (truth[:, None] + ADJACENT < rv.RANKS)
                xr = x.repeat_interleave(len(ADJACENT), 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(xr, data["globals"][index].repeat_interleave(len(ADJACENT), 0), hyp.reshape(-1),
                                           data["side"][index].repeat_interleave(len(ADJACENT), 0))
                logp = legal_logp(logits, xr, data["target"][index].repeat_interleave(len(ADJACENT), 0)) \
                    .reshape(-1, len(ADJACENT))
                weight = rank_weight[truth]
                nll = -(weight * logp[:, 1]).mean()
                local = logp.masked_fill(~valid, float("-inf"))
                discrimination = (weight * -torch.log_softmax(local, dim=1)[:, 1]).mean()
                loss = nll + args.mmi * discrimination
                window_disc.append(discrimination.detach())
            elif args.mmi:
                xr = x.repeat_interleave(rv.RANKS, 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(xr, data["globals"][index].repeat_interleave(rv.RANKS, 0), hypotheses,
                                           data["side"][index].repeat_interleave(rv.RANKS, 0))
                logp = legal_logp(logits, xr, data["target"][index].repeat_interleave(rv.RANKS, 0)).reshape(-1, rv.RANKS)
                truth = data["rank"][index]
                weight = rank_weight[truth]
                nll = -(weight * logp.gather(1, truth[:, None])[:, 0]).mean()
                discrimination = (weight * torch.nn.functional.cross_entropy(logp, truth, reduction="none")).mean()
                loss = nll + args.mmi * discrimination
                window_disc.append(discrimination.detach())
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(x, data["globals"][index], data["rank"][index], data["side"][index])
                weight = rank_weight[data["rank"][index]]
                loss = nll = -(weight * legal_logp(logits, x, data["target"][index])).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.block_dropout:
                pick = torch.Generator().manual_seed(args.seed + 104729 * step)
                kept = set(torch.randperm(len(block_params), generator=pick)[:args.block_dropout].tolist())
                for i, block in enumerate(sorted(block_params)):
                    if i not in kept:
                        for p in block_params[block]:
                            p.grad = None
            optimizer.step()
            step += 1
            seen += len(index)
            window.append(nll.detach())
            if step % 100 == 0 or step == total:
                mean = all_sum(torch.stack(window).mean().reshape(1))[0] / world
                extra = {"train_rank_ce": round(float(all_sum(torch.stack(window_disc).mean().reshape(1))[0] / world), 5)} \
                    if window_disc else {}
                elapsed = time.monotonic() - started - monitor_seconds
                log({"step": step, "epoch": round(step / steps_per_epoch, 4), "train_nll": round(float(mean), 5), **extra,
                     "lr": optimizer.param_groups[0]["lr"], "positions_per_second": round(seen * world / elapsed, 1)})
                window, window_disc = [], []
            if step % args.monitor_every == 0 or step == total or step % steps_per_epoch == 0:
                tic = time.monotonic()
                log({"step": step, **monitor(model, eval_forward, mon, probe, rank, world),
                     "monitor_seconds": round(time.monotonic() - tic, 1)})
                monitor_seconds += time.monotonic() - tic
                if not args.no_step_checkpoints:
                    save(f"step{step:06d}.pt")
            if step % args.checkpoint_every == 0:
                save("last.pt")
    save("last.pt")
    log({"event": "done", "step": step, "seconds": round(time.monotonic() - started, 1)})
    if rank == 0:
        (args.out / "done.json").write_text(json.dumps({"step": step, "total_steps": total}))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
