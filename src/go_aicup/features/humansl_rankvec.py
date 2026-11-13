"""HumanSL conditioned on Yugen ranks (C2, RESULTS A-055).

The HumanSL network (b18c384nbt-humanv0.ckpt, plain weights: A-050) adds the
output of its metadata encoder (one 384-d vector) to the first trunk layer. Here
the whole network is frozen and that vector is replaced by learned ones: for
each Yugen rank r, e[r, 0] when the labelled player is to move and e[r, 1] when
the opponent is to move (10 x 2 x 384 parameters), initialised from the encoder
output of the rank_1d profile (humansl_torch.profile_metadata). They are fitted
by maximising log p(played move) under the game's true rank.

Legal moves: empty, not ko-banned points plus pass (input planes 0, 1, 2, 6);
suicide is not excluded (the A-050 check masked it with Board.would_be_legal,
which is slow in python).

pack    : input planes of each position, bit-packed, for learning
train   : fit the vectors on packed shards (role learn), monitor on role monitor
extract : per game, mean log p(played move) under each rank hypothesis for the
          labelled player's moves (own_r0..9) and the opponent's (opp_r0..9)

    python -m go_aicup.features.humansl_rankvec pack --games G.parquet --out RUN/pack_0 --shard 0 --num-shards 4
    python -m go_aicup.features.humansl_rankvec train --run RUN
    python -m go_aicup.features.humansl_rankvec extract --games G.parquet --vectors RUN/vectors.pt --out RUN2/shard_0
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.features.humansl import CACHE, cache_games
from go_aicup.features.humansl_torch import CHANNELS, CKPT, _init_worker, _katago, encode_game, profile_metadata

RANKS, PLANES, AREA = 10, 22, 361
SEED = 2027


def load_frozen(device, fp32: bool = False):
    """HumanSL with its metadata encoder replaced by the identity (input_meta is the trunk offset).

    fp32=True disables TF32 (the path check); TF32 moves per-move log p by up to
    2.4e-3 (p99 1.5e-3), below the engine's own noise (A-041 p99 5.9e-3), and
    is ~7x faster, so learning and extraction use it."""
    import torch
    _katago()
    from katago.train.load_model import load_model
    torch.backends.cuda.matmul.allow_tf32 = not fp32
    torch.backends.cudnn.allow_tf32 = not fp32
    model, _, _ = load_model(str(CKPT), use_swa=False, device=device)
    model.eval()
    model.requires_grad_(False)
    board, _, _, _ = _katago()
    row = profile_metadata("rank_1d").get_metadata_row(nextPlayer=board.Board.BLACK, boardArea=AREA)
    with torch.no_grad():
        init = model.metadata_encoder(torch.tensor(row[None], device=device), None)[0].clone()
    model.metadata_encoder.forward = lambda input_meta, extra_outputs=None: input_meta
    return model, init


def unpack(bits):
    """uint8 [B, ceil(22*361/8)] -> float [B, 22, 19, 19]."""
    import torch
    shifts = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    x = (bits[:, :, None] >> shifts) & 1
    return x.reshape(len(bits), -1)[:, :PLANES * AREA].reshape(-1, PLANES, 19, 19).float()


def policy_logp(model, x, g, meta, target):
    """log p(target) with occupied and ko-banned points masked."""
    import torch
    logits = model(x, g, input_meta=meta)[0][0][:, 0, :].float()
    legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).reshape(len(x), AREA) > 0
    legal = torch.cat([legal, torch.ones_like(legal[:, :1])], dim=1)
    logp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    return logp.gather(1, target[:, None])[:, 0]


def game_logp(model, vectors, bins, globals_, played, own, batch: int = 4096):
    """[moves, RANKS] log p(played move) when the labelled player is each rank (no grad)."""
    import torch
    device = vectors.device
    n = len(played)
    out = torch.empty((n, RANKS), device=device)
    step = max(1, batch // RANKS)
    with torch.no_grad():
        for start in range(0, n, step):
            stop = min(n, start + step)
            m = stop - start
            x = torch.as_tensor(bins[start:stop], device=device).float().repeat_interleave(RANKS, 0)
            g = torch.as_tensor(globals_[start:stop], device=device).repeat_interleave(RANKS, 0)
            side = torch.as_tensor(~own[start:stop], device=device).long()
            meta = vectors[:, side].permute(1, 0, 2).reshape(m * RANKS, CHANNELS)    # [m*R, 384], rank fastest
            target = torch.as_tensor(played[start:stop], device=device).repeat_interleave(RANKS, 0)
            out[start:stop] = policy_logp(model, x, g, meta, target).reshape(m, RANKS)
    return out


def _encode_or_none(item):
    """encode_game, or None for a game the replay rejects (it is then left out, not fatal)."""
    try:
        return encode_game(item)
    except Exception as error:   # noqa: BLE001 - release data may hold games the replay cannot follow
        print(json.dumps({"encode_failed": repr(error)[:200]}), flush=True)
        return None


def _games(path: Path, shard: int, num_shards: int, limit: int | None):
    table = pd.read_parquet(path).iloc[shard::num_shards]
    return table.iloc[:limit] if limit else table


def pack(table: pd.DataFrame, out: Path, cache: Path, workers: int, komi: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    table.reset_index(drop=True).to_parquet(out / "games.parquet")
    _katago()
    from katago.train.load_model import load_model
    import torch
    config = load_model(str(CKPT), use_swa=False, device=torch.device("cpu"))[0].config
    games = cache_games(table["game_index"], cache)
    parts = {"bits": [], "globals": [], "target": [], "own": [], "game": []}
    started = time.monotonic()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(config, komi)) as pool:
        for g, (bins, globals_, played) in enumerate(pool.imap(encode_game, games, chunksize=4)):
            if bins.shape[1] != PLANES:
                raise ValueError(f"expected {PLANES} input planes, got {bins.shape[1]}")
            parts["bits"].append(np.packbits(bins.reshape(len(bins), -1), axis=1))
            parts["globals"].append(globals_)
            parts["target"].append(played.astype(np.int16))
            parts["own"].append(games[g][1] == table["player_color"].iloc[g])
            parts["game"].append(np.full(len(played), g, dtype=np.int32))
    for name, value in parts.items():
        np.save(out / f"{name}.npy", np.concatenate(value))
    seconds = time.monotonic() - started
    positions = int(sum(len(t) for t in parts["target"]))
    meta = {"games": len(table), "positions": positions, "seconds": round(seconds, 1),
            "positions_per_second": round(positions / seconds, 1), "komi": 6.5 if komi is None else komi}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _load_packs(run: Path, device):
    import torch
    data = {name: [] for name in ("bits", "globals", "target", "own", "rank", "role")}
    for shard in sorted(p for p in run.glob("pack_*") if (p / "meta.json").exists()):
        games = pd.read_parquet(shard / "games.parquet")
        game = np.load(shard / "game.npy")
        for name in ("bits", "globals", "target", "own"):
            data[name].append(np.load(shard / f"{name}.npy"))
        data["rank"].append(games["rank_id"].to_numpy()[game])
        data["role"].append(games["role"].to_numpy()[game] == "learn")
    out = {name: np.concatenate(value) for name, value in data.items()}
    learn = np.flatnonzero(out.pop("role"))
    monitor = np.setdiff1d(np.arange(len(out["target"])), learn)
    tensors = {"bits": torch.as_tensor(out["bits"], device=device),
               "globals": torch.as_tensor(out["globals"], device=device),
               "target": torch.as_tensor(out["target"].astype(np.int64), device=device),
               "side": torch.as_tensor((~out["own"]).astype(np.int64), device=device),
               "rank": torch.as_tensor(out["rank"].astype(np.int64), device=device)}
    return tensors, learn, monitor


def train(run: Path, *, epochs: int, batch: int, lr: float, monitor_positions: int, every: int) -> dict:
    import torch
    device = torch.device("cuda")
    model, init = load_frozen(device)
    data, learn, monitor = _load_packs(run, device)
    rng = np.random.default_rng(SEED)
    probe = torch.as_tensor(rng.permutation(monitor)[:monitor_positions], device=device)
    vectors = torch.nn.Parameter(init.repeat(RANKS, 2, 1))
    optimizer = torch.optim.Adam([vectors], lr=lr)
    steps = epochs * (len(learn) // batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + np.cos(np.pi * s / steps)))

    def nll(index, vec):
        x = unpack(data["bits"][index])
        meta = vec[data["rank"][index], data["side"][index]]
        return -policy_logp(model, x, data["globals"][index], meta, data["target"][index])

    def monitor_loss():
        with torch.no_grad():
            return float(torch.cat([nll(probe[s:s + 2048], vectors) for s in range(0, len(probe), 2048)]).mean())

    log = [{"step": 0, "monitor_nll": monitor_loss()}]
    print(json.dumps(log[-1]), flush=True)
    started, step, recent = time.monotonic(), 0, []
    generator = torch.Generator(device=device).manual_seed(SEED)
    learn_t = torch.as_tensor(learn, device=device)
    for _ in range(epochs):
        order = learn_t[torch.randperm(len(learn_t), generator=generator, device=device)]
        for start in range(0, len(order) - batch + 1, batch):
            loss = nll(order[start:start + batch], vectors).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            recent.append(float(loss))
            if step % every == 0 or step == steps:
                log.append({"step": step, "train_nll": float(np.mean(recent)), "monitor_nll": monitor_loss(),
                            "seconds": round(time.monotonic() - started, 1)})
                recent = []
                print(json.dumps(log[-1]), flush=True)
    torch.save({"vectors": vectors.detach().cpu(), "init": init.cpu(), "steps": step, "lr": lr, "batch": batch,
                "epochs": epochs}, run / "vectors.pt")
    result = {"learn_positions": int(len(learn)), "monitor_positions": int(len(monitor)), "steps": step,
              "seconds": round(time.monotonic() - started, 1), "log": log,
              "vector_shift": [[round(float((vectors[r, s] - init).norm() / init.norm()), 4) for s in range(2)]
                               for r in range(RANKS)],
              "monitor_games": _monitor_games(run, model, vectors.detach(), data, monitor)}
    (run / "train.json").write_text(json.dumps(result, indent=1))
    return result


def _monitor_games(run: Path, model, vectors, data, monitor) -> dict:
    """Diagnostic: per monitor game, sum of log p over all moves under each rank -> argmax."""
    import torch
    from go_aicup.metrics import score_rank
    total, truth = [], []
    for shard in sorted(p for p in run.glob("pack_*") if (p / "meta.json").exists()):
        games = pd.read_parquet(shard / "games.parquet")
        game = np.load(shard / "game.npy")
        bits, glob = np.load(shard / "bits.npy", mmap_mode="r"), np.load(shard / "globals.npy", mmap_mode="r")
        target, own = np.load(shard / "target.npy"), np.load(shard / "own.npy")
        bounds = np.searchsorted(game, np.arange(len(games) + 1))
        for g in np.flatnonzero(games["role"].to_numpy() == "monitor"):
            span = slice(bounds[g], bounds[g + 1])
            x = unpack(torch.as_tensor(np.asarray(bits[span]), device=vectors.device)).to(torch.uint8)
            logp = game_logp(model, vectors, x, np.asarray(glob[span]), target[span].astype(np.int64), own[span])
            total.append(logp.sum(0).cpu().numpy())
            truth.append(int(games["rank_id"].iloc[g]))
    total, truth = np.array(total), np.array(truth)
    prob = np.exp(total - total.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    exact = total.argmax(1) == truth
    near = np.abs(total.argmax(1) - truth) == 1
    decoded = score_rank(prob)
    return {"games": len(truth), "argmax_accuracy": round(float(exact.mean()), 4),
            "argmax_official": round(float((exact + near / np.e).mean()), 4),
            "score_rank_official": round(float(((decoded == truth) + (np.abs(decoded - truth) == 1) / np.e).mean()), 4)}


def load_finetuned(model, path: Path, device):
    """Load a humansl_finetune checkpoint (RankPolicy state: net.* and vectors) into the frozen
    network; returns its rank vectors."""
    import torch
    state = torch.load(path, map_location=device)["model"]
    net = {k.removeprefix("net."): v for k, v in state.items() if k.startswith("net.")}
    model.load_state_dict(net, strict=True)
    return state["vectors"].detach()


def extract(table: pd.DataFrame, vectors_path: Path | None, out: Path, cache: Path, workers: int,
            batch_positions: int = 409, komi: float | None = None, finetuned: Path | None = None,
            per_move: bool = False) -> dict:
    """Positions of consecutive games are pooled into fixed batches (batch_positions x RANKS
    evaluations) for torch.compile: 10.5k vs 4.5k evaluations/s eager on an H200 (A-055).
    finetuned: a humansl_finetune checkpoint (network weights and vectors) instead of vectors_path.
    per_move: also save every position's log p (move_logp.npy float16 [N, RANKS], move_slot.npy
    = 2 * game row + (0 own / 1 opponent), move_ply.npy), in game then ply order."""
    import torch
    out.mkdir(parents=True, exist_ok=True)
    table.reset_index(drop=True).to_parquet(out / "games.parquet")
    device = torch.device("cuda")
    model, _ = load_frozen(device)
    if finetuned is not None:
        vectors = load_finetuned(model, finetuned, device)
    else:
        vectors = torch.load(vectors_path, map_location=device)["vectors"]
    forward = torch.compile(model, dynamic=False)
    games = cache_games(table["game_index"], cache)
    sums = torch.zeros((len(table) * 2, RANKS), device=device)
    counts = np.zeros(len(table) * 2, dtype=np.int64)
    pending: list[tuple] = []
    moves: list[tuple] = []

    def flush(final: bool) -> list[tuple]:
        bins, globals_, played, own, slot, ply = (np.concatenate(a) for a in zip(*pending))
        while len(played) >= batch_positions or (final and len(played)):
            n = min(batch_positions, len(played))
            chunk = [a[:n] for a in (bins, globals_, played, own)]
            if n < batch_positions:    # pad the last batch so the compiled shape never changes
                chunk = [np.concatenate([a, np.repeat(a[-1:], batch_positions - n, axis=0)]) for a in chunk]
            logp = game_logp(forward, vectors, *chunk, batch=batch_positions * RANKS)[:n]
            sums.index_add_(0, torch.as_tensor(slot[:n], device=device), logp)
            np.add.at(counts, slot[:n], 1)
            if per_move:
                moves.append((slot[:n].astype(np.int32), ply[:n], logp.cpu().numpy().astype(np.float16)))
            bins, globals_, played, own, slot, ply = (a[n:] for a in (bins, globals_, played, own, slot, ply))
        return [(bins, globals_, played, own, slot, ply)] if len(played) else []

    started = last = time.monotonic()
    positions = waiting = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model.config, komi)) as pool:
        for g, encoded in enumerate(pool.imap(_encode_or_none, games, chunksize=4)):
            own = games[g][1] == table["player_color"].iloc[g]
            if encoded is None or not own.any() or own.all():
                continue
            bins, globals_, played = encoded
            pending.append((bins, globals_, played, own, 2 * g + (~own).astype(np.int64),
                            np.arange(len(played), dtype=np.int16)))
            positions += len(played)
            waiting += len(played)
            if waiting >= batch_positions:
                pending = flush(final=False)
                waiting = sum(len(p[2]) for p in pending)
            now = time.monotonic()
            if now - last >= 30:
                print(json.dumps({"games_read": g + 1, "positions_per_second":
                                  round(positions / (now - started), 1)}), flush=True)
                last = now
        if pending:
            flush(final=True)
    seconds = time.monotonic() - started
    counts = counts.reshape(-1, 2)
    done = (counts > 0).all(axis=1)
    mean = (sums.reshape(-1, 2, RANKS).cpu().numpy() / np.maximum(counts, 1)[:, :, None]).reshape(len(table), -1)
    feature = np.where(done[:, None], mean, 0).astype(np.float32)
    np.save(out / "feature.npy", feature)
    np.save(out / "done.npy", done)
    if per_move:
        for name, part in zip(("move_slot", "move_ply", "move_logp"), zip(*moves)):
            np.save(out / f"{name}.npy", np.concatenate(part))
    meta = {"columns": [f"own_r{r}" for r in range(RANKS)] + [f"opp_r{r}" for r in range(RANKS)],
            "games": len(table), "games_complete": int(done.sum()), "positions": positions,
            "seconds": round(seconds, 1), "positions_per_second": round(positions / seconds, 1),
            "vectors": str(vectors_path), "finetuned": None if finetuned is None else str(finetuned),
            "precision": "tf32, torch.compile", "per_move": per_move,
            "komi": 6.5 if komi is None else komi}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_features(run: Path) -> pd.DataFrame:
    """game_index + own_r* / opp_r* over every completed shard_* of an extract run."""
    parts = []
    for shard in sorted(p for p in run.glob("shard_*") if (p / "meta.json").exists()):
        table = pd.read_parquet(shard / "games.parquet")
        feature, done = np.load(shard / "feature.npy"), np.load(shard / "done.npy")
        columns = json.loads((shard / "meta.json").read_text())["columns"]
        frame = pd.DataFrame(feature[done], columns=columns)
        frame.insert(0, "game_index", table["game_index"].to_numpy()[done])
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pack", "train", "extract"))
    parser.add_argument("--games", type=Path)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--monitor-positions", type=int, default=50_000)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--komi", type=float, help="pack / extract: white komi of the input features (default rules: 6.5)")
    parser.add_argument("--per-move", action="store_true", help="extract: also save per-position log p")
    parser.add_argument("--finetuned", type=Path, help="extract: humansl_finetune checkpoint (weights + vectors)")
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train(args.run, epochs=args.epochs, batch=args.batch, lr=args.lr,
                       monitor_positions=args.monitor_positions, every=args.every)
        print(json.dumps({k: v for k, v in result.items() if k != "log"}), flush=True)
        return 0
    table = _games(args.games, args.shard, args.num_shards, args.limit)
    if args.command == "pack":
        meta = pack(table, args.out, args.cache, args.workers, komi=args.komi)
    else:
        meta = extract(table, args.vectors, args.out, args.cache, args.workers, komi=args.komi,
                       finetuned=args.finetuned, per_move=args.per_move)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
