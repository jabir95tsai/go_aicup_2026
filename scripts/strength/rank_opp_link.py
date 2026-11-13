"""Premise check: can a single game's moves be linked to a labelled training player? (RESULTS 2026-11-13)

Hypothesis behind the lever: the opponents in competition-2 test games are often among the 8,007
labelled training players; Yugen pairs players of similar strength, so a recognised opponent
gives the player's rank. The 11/11 check (rank_opponent_check) could not answer this because the
opening model recognises 7.6% of single games even when the player is in the gallery. Here the
single-game fingerprint is the two-colour GE2E style vector (competition 1, 11/12).

Gallery   every training player: normalised mean of the unit vectors of up to GALLERY of their
          HD<=1 games (own moves, both colours).
Queries   one game each, scored by cosine against the 8,007 centroids:
  seen       own moves of gallery players in games outside their gallery sample (one per player)
             -> top-1 recall; with the own column removed the same rows are the "unseen" null
  train_opp  opponent moves of training games (owner column masked: the owner is not the opponent)
  test_opp   opponent moves of every test game;  test_own  the test player's own moves
Statistics: top-1 cosine and top-1 minus top-2. In-gallery share of a set =
(share above the null q95 - 0.05) / (seen share above it - 0.05), reported for both statistics and
for seen players inside / outside the GE2E training set (the encoder is in-sample on the former).
Gate (pre-registered): seen top-1 recall >= 0.20 and estimated in-gallery share of test_opp >= 0.30.

    python -m scripts.strength.rank_opp_link plan
    python -m scripts.strength.rank_opp_link embed --part train --shard 0 --shards 6     # 1 GPU each
    python -m scripts.strength.rank_opp_link embed --part test
    python -m scripts.strength.rank_opp_link analyze
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data

RUN = ROOT / "outputs" / "pid" / "ge2e" / "pid_ge2e_v2_both_p0_1200"
STORE = ROOT / "outputs" / "rank_release_20261106" / "store"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
OUT = ROOT / "outputs" / "rank" / "opp_link"
PARTS = ("rank_public", "rank_private")
GALLERY, SEEN, TRAIN_OPP, SEED = 40, 4000, 6000, 2027
EXCLUDED_SPLITS = (0, 1, 2)          # players the GE2E run never trained on


def load_train() -> pd.DataFrame:
    meta = data.load_meta()
    meta = meta.merge(pd.read_csv(HANDICAP, usecols=["rank", "game_id", "handicap"]), on=["rank", "game_id"],
                      how="left", validate="one_to_one")
    if not np.array_equal(meta["game_index"].to_numpy(), np.arange(len(meta))):
        raise ValueError("meta order changed by the handicap join")
    return meta


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_train()
    rng = np.random.default_rng(SEED)
    ok = (meta["handicap"].to_numpy() <= 1)
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    groups = pd.Series(np.flatnonzero(ok)).groupby(player[ok]).apply(np.asarray)
    gallery, left = [], []
    for games in groups:
        games = rng.permutation(games)
        gallery.append(games[:GALLERY])
        if len(games) > GALLERY:
            left.append(games[GALLERY])                         # one held-out game per player
    gallery = np.concatenate(gallery)
    seen = rng.permutation(np.array(left))[:SEEN]
    free = ok.copy()
    free[gallery] = False
    free[seen] = False
    train_opp = rng.permutation(np.flatnonzero(free))[:TRAIN_OPP]
    rows = [pd.DataFrame({"set": name, "game_index": games, "side": side})
            for name, games, side in (("gallery", gallery, pcolor[gallery]), ("seen", seen, pcolor[seen]),
                                      ("train_opp", train_opp, 1 - pcolor[train_opp]))]
    work = pd.concat(rows, ignore_index=True).sort_values("game_index", kind="stable").reset_index(drop=True)
    work["side"] = work["side"].astype(np.int8)
    work.to_parquet(OUT / "plan.parquet", index=False)
    print(work["set"].value_counts().to_dict(), "players", len(groups), "seen players", len(left))


def load_model():
    import torch
    from go_aicup.tasks.identity.ge2e import MIN_PLY, IdentityEncoder

    config = json.loads((RUN / "config.json").read_text())
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)["model"])
    device = torch.device("cuda")
    return model.to(device), config, device, config.get("sampling", {}).get("min_ply", MIN_PLY)


def embed(args) -> None:
    from go_aicup.data.board_cache import BoardCache, write_store_cache
    from go_aicup.data.preprocess import load_store
    from go_aicup.tasks.identity import twocolor

    model, config, device, min_ply = load_model()
    if args.part == "train":
        work = pd.read_parquet(OUT / "plan.parquet")
        edges = np.linspace(0, len(work), args.shards + 1).astype(int)
        work = work.iloc[edges[args.shard]:edges[args.shard + 1]]
        lengths = data.load_meta()["length"].to_numpy()[work["game_index"].to_numpy()]
        vectors = twocolor.embed_games(model, config, BoardCache(data.CACHE_DIR), work["game_index"].to_numpy(),
                                       work["side"].to_numpy(), lengths, device, swap=True, min_ply=min_ply)
        np.save(OUT / f"train_{args.shard}of{args.shards}.npy", vectors)
        print(f"train shard {args.shard}/{args.shards}: {len(work)} games")
        return
    for name in PARTS:
        moves, colors, meta = load_store(name, STORE, mmap=False)
        cache_dir = OUT / "cache" / name
        if not (cache_dir / "games.parquet").exists():
            write_store_cache(moves, colors, meta, cache_dir)
        games = pd.read_parquet(cache_dir / "games.parquet")
        if len(games) != len(meta):
            raise ValueError(f"{name}: {len(meta) - len(games)} games were dropped by the board cache")
        rows = games["store_row"].to_numpy()
        own = (meta["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
        cache = BoardCache(cache_dir)
        for label, side in (("own", own), ("opp", (1 - own).astype(np.int8))):
            vectors = np.empty((len(meta), config["model"]["embedding"]), np.float32)
            vectors[rows] = twocolor.embed_games(model, config, cache, games["game_index"].to_numpy(), side,
                                                 games["length"].to_numpy(), device, swap=True, min_ply=min_ply)
            np.save(OUT / f"{name}_{label}.npy", vectors)
        print(f"{name}: {len(meta)} games, both sides")


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def top2(scores: np.ndarray):
    """(argmax, top-1 score, top-1 minus top-2) per row."""
    part = np.argpartition(scores, -2, axis=1)[:, -2:]
    pair = np.take_along_axis(scores, part, axis=1)
    first = pair.argmax(axis=1)
    rows = np.arange(len(scores))
    s1, s2 = pair[rows, first], pair[rows, 1 - first]
    return part[rows, first], s1, s1 - s2


def share(found: float, recall: float, alpha: float) -> float | None:
    return None if recall - alpha < 1e-9 else float((found - alpha) / (recall - alpha))


def analyze(args) -> None:
    meta = load_train()
    work = pd.read_parquet(OUT / "plan.parquet")
    edges = np.linspace(0, len(work), args.shards + 1).astype(int)
    vectors = np.concatenate([np.load(OUT / f"train_{i}of{args.shards}.npy") for i in range(args.shards)])
    if len(vectors) != len(work) or edges[-1] != len(work):
        raise ValueError("train shards do not match the plan")
    if not np.isfinite(vectors).all():
        raise ValueError("non-finite train vectors")
    vectors = unit(vectors.astype(np.float64))
    player = meta["player_id"].astype(str).to_numpy()
    rank_id = meta["rank_id"].to_numpy().astype(np.int64)
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)
    cand_rank = pd.Series(rank_id, index=player).groupby(level=0).first().loc[candidates].to_numpy()
    sets = {name: np.flatnonzero(work["set"].to_numpy() == name) for name in ("gallery", "seen", "train_opp")}
    game = work["game_index"].to_numpy()

    g = sets["gallery"]
    owner_g = column.loc[player[game[g]]].to_numpy()
    sums = np.zeros((len(candidates), vectors.shape[1]))
    np.add.at(sums, owner_g, vectors[g])
    if (np.bincount(owner_g, minlength=len(candidates)) == 0).any():
        print("players without an HD<=1 gallery game:", int((np.bincount(owner_g, minlength=len(candidates)) == 0).sum()))
    centroids = np.where(np.linalg.norm(sums, axis=1, keepdims=True) > 0, sums, 1e-9)
    centroids = unit(centroids)

    gallery_table, _ = data.pid_splits(data.load_meta())
    untrained = set(gallery_table.loc[gallery_table["split"].isin(EXCLUDED_SPLITS), "player_id"].astype(str))

    # seen queries and the unseen null (own column removed)
    s = sets["seen"]
    own = column.loc[player[game[s]]].to_numpy()
    scores = vectors[s] @ centroids.T
    arg, s1, gap = top2(scores)
    order = (scores > scores[np.arange(len(s)), own][:, None]).sum(axis=1)       # rank of the true player, 0 = top
    masked = scores.copy()
    masked[np.arange(len(s)), own] = -np.inf
    _, n1, ngap = top2(masked)
    in_train = ~np.isin(player[game[s]], list(untrained))
    stats = {"cos": (s1, n1), "gap": (gap, ngap)}
    result = {"gallery_players": int(len(candidates)), "gallery_games": int(len(g)),
              "seen": {"n": int(len(s)), "top1": float((order == 0).mean()), "top5": float((order < 5).mean()),
                       "top50": float((order < 50).mean()), "median_rank": float(np.median(order)),
                       "top1_ge2e_trained": float((order[in_train] == 0).mean()), "n_trained": int(in_train.sum()),
                       "top1_ge2e_untrained": float((order[~in_train] == 0).mean()),
                       "n_untrained": int((~in_train).sum())}}
    thresholds, recall = {}, {}
    for name, (seen_stat, null_stat) in stats.items():
        for label, q in (("q95", 0.95), ("q99", 0.99)):
            t = float(np.quantile(null_stat, q))
            thresholds[name, label] = t
            recall[name, label] = {"all": float((seen_stat > t).mean()),
                                   "trained": float((seen_stat[in_train] > t).mean()),
                                   "untrained": float((seen_stat[~in_train] > t).mean())}
            result["seen"][f"{name}_{label}"] = {"threshold": t, **recall[name, label]}

    def describe(q_vectors: np.ndarray, mask_column: np.ndarray | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
        sc = q_vectors @ centroids.T
        if mask_column is not None:
            sc[np.arange(len(sc)), mask_column] = -np.inf
        a, top, gp = top2(sc)
        out = {"n": int(len(sc)), "median_cos": float(np.median(top)), "median_gap": float(np.median(gp))}
        for name, stat in (("cos", top), ("gap", gp)):
            for label, alpha in (("q95", 0.05), ("q99", 0.01)):
                found = float((stat > thresholds[name, label]).mean())
                out[f"{name}_{label}"] = {"above": found,
                                          **{f"share_{k}": share(found, v, alpha) for k, v in recall[name, label].items()}}
        return out, a, gp

    result["null"] = {"median_cos": float(np.median(n1)), "median_gap": float(np.median(ngap))}
    result["seen"]["median_cos"], result["seen"]["median_gap"] = float(np.median(s1)), float(np.median(gap))

    t = sets["train_opp"]
    owner_t = column.loc[player[game[t]]].to_numpy()
    result["train_opp"], arg_t, gap_t = describe(vectors[t], owner_t)
    # lever: rank of the matched player against the owner's rank, by HD and by match strength
    owner_rank = rank_id[game[t]]
    hd = meta["handicap"].to_numpy()[game[t]]
    diff = cand_rank[arg_t] - owner_rank
    shuffled = cand_rank[np.random.default_rng(SEED).permutation(arg_t)] - owner_rank
    lever = {"shuffled": {"same": float((shuffled == 0).mean()), "within1": float((np.abs(shuffled) <= 1).mean())}}
    for label, keep in (("all", np.ones(len(t), bool)), ("gap>q95", gap_t > thresholds["gap", "q95"]),
                        ("gap>q99", gap_t > thresholds["gap", "q99"])):
        for h in (0, 1):
            m = keep & (hd == h)
            if m.any():
                lever[f"{label}_hd{h}"] = {"n": int(m.sum()), "same": float((diff[m] == 0).mean()),
                                           "within1": float((np.abs(diff[m]) <= 1).mean()),
                                           "mean_diff": float(diff[m].mean())}
    result["lever"] = lever

    for name in PARTS:
        for side in ("opp", "own"):
            v = np.load(OUT / f"{name}_{side}.npy")
            if not np.isfinite(v).all():
                raise ValueError(f"non-finite vectors in {name}_{side}")
            result[f"{name}_{side}"], _, _ = describe(unit(v.astype(np.float64)))

    test_share = [result[f"{name}_opp"]["gap_q95"]["share_all"] for name in PARTS]
    result["gate"] = {"seen_top1": result["seen"]["top1"], "test_opp_share_gap_q95": test_share,
                      "pass": bool(result["seen"]["top1"] >= 0.20 and all(x is not None and x >= 0.30 for x in test_share))}
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    p = sub.add_parser("embed")
    p.add_argument("--part", choices=("train", "test"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=6)
    p.set_defaults(fn=embed)
    p = sub.add_parser("analyze")
    p.add_argument("--shards", type=int, default=6)
    p.set_defaults(fn=analyze)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
