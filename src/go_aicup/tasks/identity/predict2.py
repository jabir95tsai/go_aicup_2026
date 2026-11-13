"""Competition-1 added test set (pool2_* / pid2_*): two-colour opening + GE2E style + HD-and-colour term.

Every choice (opening parameters, raw or colour-swapped white moves, lambda per k, mu) is
read from the bench summary written by scripts/identity/pid2_bench.py; nothing is fitted on
the official files. No assignment: the organisers state that questions and candidates are
not one-to-one. One GPU for the vectors (cached in --out-dir).

    python -m go_aicup.tasks.identity.predict2 --store-dir S --pool pool2_public \\
        --questions pid2_public --out-dir outputs/pid/.../public2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, opening, twocolor
from go_aicup.tasks.identity.predict import PLIES, store_prefix, train_prefix

BENCH = ROOT / "outputs" / "pid" / "pid2_bench" / "pid_ge2e_v1_short" / "summary.json"


def store_vectors(store_dir: Path, name: str, work: Path, swap: bool, run_dir: Path) -> np.ndarray:
    """[games of the store, D] GE2E vectors of the player's own moves with the encoder of
    run_dir (first ply from its config's sampling), cached."""
    import torch
    from go_aicup.data.board_cache import BoardCache, write_store_cache
    from go_aicup.tasks.identity.ge2e import MIN_PLY, IdentityEncoder

    path = work / f"{name}_{run_dir.name}_{'swap' if swap else 'raw'}.npy"
    if path.exists():
        return np.load(path)
    moves, colors, meta = load_store(name, store_dir, mmap=False)
    cache_dir = work / name / "cache"
    if not (cache_dir / "games.parquet").exists():
        write_store_cache(moves, colors, meta, cache_dir)
    games = pd.read_parquet(cache_dir / "games.parquet")
    if len(games) != len(meta):
        raise ValueError(f"{name}: {len(meta) - len(games)} games were dropped by the board cache")
    rows = games["store_row"].to_numpy()
    player = (meta["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
    config = json.loads((run_dir / "config.json").read_text())
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config = {**config, "precision": "fp32"}
    model.to(device)
    vectors = np.empty((len(meta), config["model"]["embedding"]), np.float32)
    vectors[rows] = twocolor.embed_games(model, config, BoardCache(cache_dir), games["game_index"].to_numpy(),
                                         player, games["length"].to_numpy(), device, swap=swap,
                                         min_ply=config.get("sampling", {}).get("min_ply", MIN_PLY))
    np.save(path, vectors)
    return vectors


def predict(store_dir: Path, pool_name: str, question_name: str, work: Path, summary: dict):
    params = summary["opening"]
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    train_meta, train_pre = train_prefix("d4")
    pool_moves, pool_colors, pool_meta = load_store(pool_name, store_dir, mmap=False)
    q_moves, q_colors, q_meta = load_store(question_name, store_dir, mmap=False)
    pool_pre, pool_ok = store_prefix(pool_moves, pool_colors, pool_meta, "d4", min_plies)
    q_pre, q_ok = store_prefix(q_moves, q_colors, q_meta, "d4", min_plies)
    if not (pool_ok.all() and q_ok.all()):
        raise ValueError("a game is too short or does not alternate; handle it before predicting")
    train_color = train_meta["player_color"].to_numpy().astype(np.int64)
    populations = (np.flatnonzero(train_color == 0), np.flatnonzero(train_color == 1))
    n_train, n_pool = len(train_meta), len(pool_meta)
    prefix = np.concatenate([train_pre[:, :PLIES], pool_pre, q_pre])
    hashes = opening.prefix_hashes(prefix)
    is_white = lambda meta: (meta["color"].astype(str).to_numpy() == "W").astype(np.int64)
    color = np.concatenate([train_color, is_white(pool_meta), is_white(q_meta)])
    codes = np.concatenate([np.zeros(n_train, np.int64),
                            np.minimum(pool_meta["hd"].to_numpy(), 1) + 2 * is_white(pool_meta),
                            np.minimum(q_meta["hd"].to_numpy(), 1) + 2 * is_white(q_meta)])

    pool_players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(pool_players)))
    gallery = [n_train + np.flatnonzero(pool_players == c) for c in candidates]
    question_ids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    query = [n_train + n_pool + np.sort(rows_by_q[q]) for q in question_ids]
    ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[question_ids].to_numpy()

    swap = summary["style"] == "style-swap"
    run_dir = Path(summary["run_dir"])
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    vectors = np.concatenate([store_vectors(store_dir, pool_name, work, swap, run_dir),
                              store_vectors(store_dir, question_name, work, swap, run_dir)])
    shift = lambda games: [g - n_train for g in games]
    parts = {"opening": twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, params),
             "style": summary["w"] * twocolor.style_scores(vectors, shift(gallery), shift(query)),
             "category": twocolor.category_scores(codes, gallery, query, 4)}
    lam = np.array([summary["lambda_per_k"].get(str(k), summary["lambda"]) for k in ks], float)
    if not summary.get("use_per_k", False):
        lam[:] = summary["lambda"]
    scores = parts["opening"] + lam[:, None] * parts["style"] + summary["mu"] * parts["category"]
    ranked = assign.top5(scores)
    table = pd.DataFrame({"question_id": question_ids, "num_games": ks})
    for j in range(5):
        table[f"top{j + 1}"] = candidates[ranked[:, j]]
    info = {"questions": len(question_ids), "candidates": len(candidates), "summary": summary}
    return table, scores, parts, candidates, info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bench-summary", type=Path, default=BENCH)
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.bench_summary.read_text())
    table, scores, parts, candidates, info = predict(args.store_dir, args.pool, args.questions, args.out_dir, summary)
    np.save(args.out_dir / "scores.npy", scores)
    np.savez(args.out_dir / "parts.npz", **parts)
    np.save(args.out_dir / "candidates.npy", candidates)
    table.to_parquet(args.out_dir / "predictions.parquet", index=False)
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "summary"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
