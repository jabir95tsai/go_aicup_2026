"""HumanSL trunk embeddings with PyTorch (RESULTS A-050, A-051).

KataGo's own python code (v1.18.2 source in data/external) loads
b18c384nbt-humanv0.ckpt (plain weights, not SWA: A-050) and computes the input
features while replaying each game in CPU worker processes; the GPU runs the
trunk. Per move, two 384-d vectors of "trunkfinal" (after the final norm and
activation) for the position before the move: the board mean, and the vector at
the played point. Per game they are averaged over the labelled player's moves
(own) and the opponent's moves (opp), for each metadata condition:
  rank_1d   cpp makeBasicRankProfile for 1d (both players)
  unknown   same profile with both ranks marked unknown
Unlike the analysis engine there is no NN cache, so transposed openings use
their own history (A-050).

Output per shard: embedding.npy float16 [games, conditions * 2 sides * 2 kinds * 384]
(order: condition, side own/opp, kind pool/played, channel), done.npy, meta.json.

    python -m go_aicup.features.humansl_torch --games G.parquet --out OUT/shard_0
"""
from __future__ import annotations

import argparse
import datetime
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.features.humansl import CACHE, cache_games
from go_aicup.paths import ROOT

KATAGO_PY = ROOT / "data" / "external" / "katago-src-v1.18.2" / "KataGo-1.18.2" / "python"
CKPT = ROOT / "data" / "external" / "humansl-v0" / "b18c384nbt-humanv0.ckpt"
CONDITIONS = ("rank_1d", "unknown")
CHANNELS = 384
INVERSE_RANK = {**{f"{d}d": 10 - d for d in range(1, 10)}, **{f"{k}k": 9 + k for k in range(1, 21)}}


def _katago():
    if str(KATAGO_PY) not in sys.path:
        sys.path.insert(0, str(KATAGO_PY))
    import katago.game.board as board
    import katago.game.features as features
    import katago.game.gamestate as gamestate
    import katago.game.sgfmetadata as sgfmetadata
    return board, features, gamestate, sgfmetadata


def profile_metadata(name: str):
    """cpp makeBasicRankProfile (sgfmetadata.cpp) for rank_<r>; "unknown" marks both ranks unknown."""
    _, _, _, sgfmetadata = _katago()
    rank = INVERSE_RANK["1d" if name == "unknown" else name.removeprefix("rank_")]
    meta = sgfmetadata.SGFMetadata(
        inverseBRank=rank, inverseWRank=rank, bIsHuman=True, wIsHuman=True, gameRatednessIsUnknown=True,
        tcIsUnknown=False, tcIsByoYomi=True, mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
        gameDate=datetime.date(2020, 3, 1), source=sgfmetadata.SGFMetadata.SOURCE_KGS)
    if name == "unknown":
        meta.bRankIsUnknown = meta.wRankIsUnknown = True
        meta.inverseBRank = meta.inverseWRank = 0
    return meta


_FEATURES = None
_KOMI = None   # None = the rules' default (Japanese, 6.5)


def _init_worker(config: dict, komi: float | None = None) -> None:
    global _FEATURES, _KOMI
    _, features, _, _ = _katago()
    _FEATURES = features.Features(config, 19)
    _KOMI = komi


def encode_game(item):
    """CPU worker: binary planes (uint8), global inputs, played tensor positions."""
    moves, colors = item
    board, _, gamestate, _ = _katago()
    rules = dict(gamestate.GameState.RULES_JAPANESE)
    if _KOMI is not None:
        rules["whiteKomi"] = float(_KOMI)
    state = gamestate.GameState(19, rules)
    bins, globals_, played = [], [], []
    for move, color in zip(moves, colors):
        pla = board.Board.BLACK if color == 0 else board.Board.WHITE
        if state.board.pla != pla:
            raise ValueError("move colour does not match the side to move")
        b, g = state.get_input_features(_FEATURES)
        bins.append(b[0].astype(np.uint8)), globals_.append(g[0])
        row, col = divmod(int(move), 19)
        loc = state.board.loc(col, row)
        played.append(_FEATURES.loc_to_tensor_pos(loc, state.board))
        state.play(pla, loc)
    return np.stack(bins), np.stack(globals_).astype(np.float32), np.asarray(played, dtype=np.int64)


def run(games, player_colors, out: Path, *, workers: int = 11, batch: int = 1024, device_name: str = "cuda") -> dict:
    import torch
    _katago()
    from katago.train.load_model import load_model
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, _, _ = load_model(str(CKPT), use_swa=False, device=device)
    model.eval()
    from katago.train.model_pytorch import ExtraOutputs
    metas = [profile_metadata(name) for name in CONDITIONS]
    board, _, _, _ = _katago()
    meta_rows = {pla: np.stack([m.get_metadata_row(nextPlayer=pla, boardArea=361) for m in metas])
                 for pla in (board.Board.BLACK, board.Board.WHITE)}
    width = len(CONDITIONS) * 2 * 2 * CHANNELS
    embedding = np.zeros((len(games), width), dtype=np.float16)
    done = np.zeros(len(games), dtype=bool)
    started = last = time.monotonic()
    positions = 0
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(model.config,)) as pool:
        for g, (bins, globals_, played) in enumerate(pool.imap(encode_game, games, chunksize=4)):
            colors = games[g][1]
            n, c = len(played), len(CONDITIONS)
            plas = np.where(colors == 0, board.Board.BLACK, board.Board.WHITE)
            meta = np.concatenate([meta_rows[p][None] for p in plas])            # [n, c, 192]
            pooled = torch.zeros((n, c, CHANNELS), device=device)
            at_move = torch.zeros((n, c, CHANNELS), device=device)
            with torch.no_grad():
                for start in range(0, n, max(1, batch // c)):
                    stop = min(n, start + max(1, batch // c))
                    m = stop - start
                    x = torch.tensor(np.repeat(bins[start:stop], c, axis=0), dtype=torch.float32, device=device)
                    y = torch.tensor(np.repeat(globals_[start:stop], c, axis=0), device=device)
                    z = torch.tensor(meta[start:stop].reshape(m * c, -1), device=device)
                    extra = ExtraOutputs(["trunkfinal"])
                    model(x, y, input_meta=z, extra_outputs=extra)
                    trunk = extra.returned["trunkfinal"].float().reshape(m, c, CHANNELS, -1)   # [m, c, 384, 361]
                    pooled[start:stop] = trunk.mean(dim=-1)
                    index = torch.tensor(played[start:stop], device=device)[:, None, None, None].expand(m, c, CHANNELS, 1)
                    at_move[start:stop] = trunk.gather(-1, index)[..., 0]
            mine = torch.tensor(colors == player_colors[g], device=device)
            if not mine.any() or mine.all():
                continue
            parts = []
            for k in range(c):
                for side in (mine, ~mine):
                    parts += [pooled[side, k].mean(dim=0), at_move[side, k].mean(dim=0)]
            embedding[g] = torch.cat(parts).cpu().numpy().astype(np.float16)
            done[g] = True
            positions += n
            now = time.monotonic()
            if now - last >= 30:
                print(json.dumps({"games_done": int(done.sum()), "positions_per_second":
                                  round(positions / (now - started), 1)}), flush=True)
                last = now
    seconds = time.monotonic() - started
    np.save(out / "embedding.npy", embedding)
    np.save(out / "done.npy", done)
    meta_out = {"conditions": list(CONDITIONS), "layout": "condition, side(own,opp), kind(pool,played), channel",
                "games": len(games), "games_complete": int(done.sum()), "positions": positions,
                "seconds": round(seconds, 1), "positions_per_second": round(positions / seconds, 1),
                "checkpoint": CKPT.name, "weights": "model (not swa)"}
    (out / "meta.json").write_text(json.dumps(meta_out, indent=1))
    return meta_out


def run_embeddings(run_dir: Path) -> pd.DataFrame:
    """game_index + emb_<i> columns over every shard_* of a run (complete games only)."""
    parts = []
    for shard in sorted(p for p in run_dir.glob("shard_*") if p.is_dir()):
        table = pd.read_parquet(shard / "games.parquet")
        embedding, done = np.load(shard / "embedding.npy"), np.load(shard / "done.npy")
        frame = pd.DataFrame(embedding[done].astype(np.float32), columns=[f"emb_{i}" for i in range(embedding.shape[1])])
        frame.insert(0, "game_index", table["game_index"].to_numpy()[done])
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args(argv)
    table = pd.read_parquet(args.games).iloc[args.shard::args.num_shards]
    if args.limit:
        table = table.iloc[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "games.parquet")
    meta = run(cache_games(table["game_index"], args.cache), table["player_color"].to_numpy(), args.out,
               workers=args.workers)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
