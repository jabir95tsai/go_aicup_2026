"""Stage 0 of the HumanSL PyTorch path (RESULTS A-050): reproduce the engine's policy.

Loads b18c384nbt-humanv0.ckpt with KataGo's own python code (v1.18.2 source in
data/external), replays games with its Board / Features, builds the humanSL
metadata row of each profile exactly as cpp/neuralnet/sgfmetadata.cpp
makeBasicRankProfile does, and compares log p(played move) with the analysis
engine values of A-039 (outputs/humansl_pilot/shard_0).

Pre-registered gate (A-050): over 100 games and 15 profiles, per-move |dlogp|
mean <= 3e-3 and p99 <= 2e-2, and the per-game feature (own / opp mean) max
|diff| p99 <= 1e-2. Engine-vs-engine noise (A-041): mean 1.1e-3, p99 5.9e-3.

    python scripts/humansl_torch_check.py --games 100 [--swa]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys

import numpy as np
import pandas as pd
import torch

from go_aicup.features.humansl import PROFILES, cache_games
from go_aicup.paths import ROOT

KATAGO_PY = ROOT / "data" / "external" / "katago-src-v1.18.2" / "KataGo-1.18.2" / "python"
CKPT = ROOT / "data" / "external" / "humansl-v0" / "b18c384nbt-humanv0.ckpt"
REFERENCE = ROOT / "outputs" / "humansl_pilot" / "shard_0"
sys.path.insert(0, str(KATAGO_PY))

from katago.game.board import Board  # noqa: E402
from katago.game.features import Features  # noqa: E402
from katago.game.gamestate import GameState  # noqa: E402
from katago.game.sgfmetadata import SGFMetadata  # noqa: E402
from katago.train.load_model import load_model  # noqa: E402

INVERSE_RANK = {**{f"{d}d": 10 - d for d in range(1, 10)}, **{f"{k}k": 9 + k for k in range(1, 21)}}


def profile_metadata(profile: str) -> SGFMetadata:
    """cpp makeBasicRankProfile for rank_<r> (modern openings, KGS source)."""
    rank = INVERSE_RANK[profile.removeprefix("rank_")]
    return SGFMetadata(inverseBRank=rank, inverseWRank=rank, bIsHuman=True, wIsHuman=True,
                       gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True,
                       mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
                       gameDate=datetime.date(2020, 3, 1), source=SGFMetadata.SOURCE_KGS)


def game_logp(model, features, moves, colors, metas, device) -> np.ndarray:
    """[moves, profiles] log p(played move) with illegal moves masked out."""
    state = GameState(19, GameState.RULES_JAPANESE)
    bins, globals_, meta_rows, legal, targets = [], [], [], [], []
    for move, color in zip(moves, colors):
        pla = Board.BLACK if color == 0 else Board.WHITE
        assert state.board.pla == pla, "move colour does not match the side to move"
        b, g = state.get_input_features(features)
        bins.append(b[0]), globals_.append(g[0])
        meta_rows.append([m.get_metadata_row(nextPlayer=pla, boardArea=361) for m in metas])
        board = state.board
        mask = np.zeros(features.pos_len * features.pos_len + 1, dtype=bool)
        for y in range(19):
            for x in range(19):
                loc = board.loc(x, y)
                mask[features.loc_to_tensor_pos(loc, board)] = board.would_be_legal(pla, loc)
        mask[-1] = True
        legal.append(mask)
        row, col = divmod(int(move), 19)
        loc = board.loc(col, row)
        targets.append(features.loc_to_tensor_pos(loc, board))
        state.play(pla, loc)
    n, p = len(moves), len(metas)
    bin_t = torch.tensor(np.repeat(np.stack(bins), p, axis=0), device=device)
    glob_t = torch.tensor(np.repeat(np.stack(globals_), p, axis=0), device=device)
    meta_t = torch.tensor(np.stack(meta_rows).reshape(n * p, -1), device=device)
    out = []
    with torch.no_grad():
        for start in range(0, n * p, 1024):
            stop = start + 1024
            outputs = model(bin_t[start:stop], glob_t[start:stop], input_meta=meta_t[start:stop])
            out.append(model.postprocess_output(outputs)[0][0][:, 0, :].float())
    logits = torch.cat(out).reshape(n, p, -1)
    logits = logits.masked_fill(~torch.tensor(np.stack(legal), device=device)[:, None, :], float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    index = torch.tensor(targets, device=device)[:, None, None].expand(n, p, 1)
    return logp.gather(-1, index)[..., 0].cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, swa, _ = load_model(str(CKPT), use_swa=args.swa, device=device)
    model = swa if args.swa else model
    model.eval()
    features = Features(model.config, model.pos_len)
    metas = [profile_metadata(p) for p in PROFILES]
    table = pd.read_parquet(REFERENCE / "games.parquet").iloc[:args.games]
    reference, offsets = np.load(REFERENCE / "logp.npy"), np.load(REFERENCE / "offsets.npy")
    diffs, feature_diffs = [], []
    for i, ((moves, colors), game) in enumerate(zip(cache_games(table["game_index"]), table.itertuples())):
        mine = game_logp(model, features, moves, colors, metas, device)
        engine = reference[offsets[i]:offsets[i + 1]]
        diffs.append(np.abs(mine - engine).ravel())
        own = colors == game.player_color
        feature_diffs.append(max(np.abs(mine[own].mean(0) - engine[own].mean(0)).max(),
                                 np.abs(mine[~own].mean(0) - engine[~own].mean(0)).max()))
    d, f = np.concatenate(diffs), np.array(feature_diffs)
    result = {"weights": "swa" if args.swa else "model", "games": len(f), "moves_x_profiles": int(len(d)),
              "per_move": {"mean": float(d.mean()), "p50": float(np.median(d)), "p99": float(np.quantile(d, 0.99)),
                           "max": float(d.max())},
              "per_game_feature": {"p99": float(np.quantile(f, 0.99)), "max": float(f.max())}}
    result["gate"] = bool(result["per_move"]["mean"] <= 3e-3 and result["per_move"]["p99"] <= 2e-2
                          and result["per_game_feature"]["p99"] <= 1e-2)
    print(json.dumps(result, indent=1))
    if args.out:
        (ROOT / args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
