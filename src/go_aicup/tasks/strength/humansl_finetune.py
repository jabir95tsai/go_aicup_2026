"""C4: HumanSL fine-tuned as a Yugen-rank-conditioned policy (RESULTS A-057).

Starts from HumanSL (plain weights) with the metadata encoder replaced by the
A-055 rank vectors (e[rank, own/opp]), trains the policy path (conv_spatial,
linear_global, blocks, norm_trunkfinal, policy_head) and the vectors on packed
positions (features/humansl_rankvec.py pack) of the F players. The network stays
in eval mode: its one batch norm (trunk end) keeps its running statistics.

Each process keeps its share of the pack_* shards (shard i on rank i % world)
in GPU memory and draws its own batches; gradients are averaged by DDP.
Monitor (Bm games, outputs/rank/humansl/humansl_finetune/monitor): mean NLL of the played
move under the true rank on a fixed position subset, and the single-game official
score (sum of log p under each rank hypothesis -> softmax -> score_rank). Step 0
is C2 (A-055) exactly.

--mmi LAMBDA (C4b) adds a rank-discrimination term: every position is evaluated
under all 10 rank hypotheses and loss = NLL(true rank) + LAMBDA * CE(true rank |
softmax over ranks of log p(played move)). Positions per step are then --batch,
forward passes --batch x 10.

    torchrun --standalone --nproc-per-node=8 -m go_aicup.tasks.strength.humansl_finetune \\
        --data outputs/rank/humansl/humansl_finetune --out outputs/rank/humansl/c4_s2 [--max-positions N] [--resume]
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

VECTORS = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec" / "vectors.pt"
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


def load_shards(paths, device) -> dict:
    """Packed positions of the given pack_* directories, on device."""
    parts = {name: [] for name in ("bits", "globals", "target", "side", "rank", "game")}
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
    out = {name: torch.as_tensor(np.concatenate(value), device=device) for name, value in parts.items()}
    out["games"] = offset
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


@torch.no_grad()
def monitor(model: RankPolicy, forward, data: dict, probe: torch.Tensor, rank: int, world: int) -> dict:
    """NLL under the true rank on probe positions, and single-game official score over all monitor games."""
    device = data["bits"].device
    nll = torch.zeros(2, device=device)
    for start in range(0, len(probe), 4096):
        index = probe[start:start + 4096]
        x = rv.unpack(data["bits"][index])
        logits = forward(x, data["globals"][index], data["rank"][index], data["side"][index])
        nll += torch.stack([-legal_logp(logits, x, data["target"][index]).sum(), torch.tensor(len(index), device=device)])
    totals = torch.zeros((data["games"], rv.RANKS), device=device)
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
        totals.index_add_(0, data["game"][index[:n]], logp)
    nll, totals = all_sum(nll), all_sum(totals)
    truth = torch.zeros(data["games"], dtype=torch.long, device=device)
    truth.index_put_((data["game"],), data["rank"])
    totals, truth = totals.cpu().numpy(), truth.cpu().numpy()
    prob = np.exp(totals - totals.max(1, keepdims=True))
    decoded = score_rank(prob / prob.sum(1, keepdims=True))
    official = (decoded == truth) + (np.abs(decoded - truth) == 1) / np.e
    return {"monitor_nll": round(float(nll[0] / nll[1]), 5), "monitor_official": round(float(official.mean()), 4),
            "monitor_accuracy": round(float((totals.argmax(1) == truth).mean()), 4)}


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
    args = parser.parse_args(argv)
    rank, world, device = setup()
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "log.jsonl"

    def log(entry: dict) -> None:
        if rank == 0:
            print(json.dumps(entry), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(entry) + "\n")

    shards = sorted(p for p in args.data.glob("pack_*") if (p / "meta.json").exists())[:args.shards]
    data = load_shards(shards[rank::world], device)
    mon = load_shards([args.data / "monitor" / "pack_0"], device)
    probe = torch.as_tensor(np.random.default_rng(args.seed).permutation(len(mon["target"]))[:args.monitor_positions],
                            device=device)
    model = RankPolicy(device, torch.load(VECTORS, map_location=device)["vectors"])
    trainable = [p for n, p in model.net.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable, "lr": args.lr},
                                   {"params": [model.vectors], "lr": args.vector_lr}], weight_decay=0.0)
    local = torch.tensor(len(data["target"]) // args.batch, device=device)
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.MIN)
    steps_per_epoch = int(local)
    total = int(math.ceil(args.epochs * steps_per_epoch))
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
    trained = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    train_forward = trained if args.no_compile else torch.compile(trained)
    eval_forward = model if args.no_compile else torch.compile(model, dynamic=False)
    log({"event": "start", "world": world, "shards": len(shards), "positions_rank0": len(data["target"]),
         "steps_per_epoch": steps_per_epoch, "total_steps": total, "global_batch": args.batch * world,
         "trainable_parameters": sum(p.numel() for p in trainable), "resume_step": step, "args": vars(args) | {
             "data": str(args.data), "out": str(args.out)}})
    if step == 0:
        log({"step": 0, **monitor(model, eval_forward, mon, probe, rank, world)})
    started, window, window_disc, seen, monitor_seconds = time.monotonic(), [], [], 0, 0.0
    hypotheses = torch.arange(rv.RANKS, device=device).repeat(args.batch)

    def save(name: str) -> None:
        if rank == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, args.out / "tmp.pt")
            os.replace(args.out / "tmp.pt", args.out / name)

    while step < total:
        epoch = step // steps_per_epoch
        generator = torch.Generator(device=device).manual_seed(args.seed + 1000 * epoch + rank)
        order = torch.randperm(len(data["target"]), generator=generator, device=device)
        for local_step in range(step % steps_per_epoch, steps_per_epoch):
            if step >= total:
                break
            scale = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
            for group, lr in zip(optimizer.param_groups, base):
                group["lr"] = lr * scale
            index = order[local_step * args.batch:(local_step + 1) * args.batch]
            x = rv.unpack(data["bits"][index])
            if args.mmi:
                xr = x.repeat_interleave(rv.RANKS, 0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(xr, data["globals"][index].repeat_interleave(rv.RANKS, 0), hypotheses,
                                           data["side"][index].repeat_interleave(rv.RANKS, 0))
                logp = legal_logp(logits, xr, data["target"][index].repeat_interleave(rv.RANKS, 0)).reshape(-1, rv.RANKS)
                truth = data["rank"][index]
                nll = -logp.gather(1, truth[:, None]).mean()
                discrimination = torch.nn.functional.cross_entropy(logp, truth)
                loss = nll + args.mmi * discrimination
                window_disc.append(discrimination.detach())
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = train_forward(x, data["globals"][index], data["rank"][index], data["side"][index])
                loss = nll = -legal_logp(logits, x, data["target"][index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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
                save(f"step{step:06d}.pt")
            if step % args.checkpoint_every == 0:
                save("last.pt")
    save("last.pt")
    log({"event": "done", "step": step, "seconds": round(time.monotonic() - started, 1)})
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
