"""Are the added-set candidates (pool2_*, 10 games each) people who also appear, under other
ids and with other games, in the original candidate pools (pool_*, 100 black games) or in the
training set (8,007 players)?  Day-0 only showed that no game and no id is shared.

    python -m scripts.identity.pid2_overlap embed     # 1 GPU: vectors of the samples below
    python -m scripts.identity.pid2_overlap analyze   # CPU: nearest-neighbour tables

Method: GE2E vectors (white moves colour-swapped) of
  train      up to 20 HD<=1 games per training player, split 10 probe / 10 reference
  pool_*     30 of the 100 games of every original candidate
  pool2_*    every game of every added-set candidate
A player's centroid is the normalised mean of unit game vectors. Calibration on the training
set: a probe centroid (10 games) against every player's centroid gives the cosine to its own
reference (what a true match looks like) and the best cosine to anybody else (the null).
Each official set is then matched against each reference set; the share of players whose best
cosine exceeds the null's 99th percentile estimates the overlap. Encoder-trained and
encoder-unseen training players are calibrated separately (official players are unseen).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, twocolor

RUN = ROOT / "outputs" / "pid" / "ge2e" / "pid_ge2e_v4_d4_hd_9600"
STORE = ROOT / "outputs" / "pid" / "day0_1112" / "store"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
OUT = ROOT / "outputs" / "pid" / "pid2_overlap"
PARTS = {"pool_public": 30, "pool_private": 30, "pool2_public": None, "pool2_private": None}
TRAIN_GAMES, PROBE, SEED = 20, 10, 2026


def train_plan() -> pd.DataFrame:
    """game, player_id, color, length, probe (bool) and trained (player was an encoder class)."""
    meta = data.load_meta()
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy()
    games = np.flatnonzero(handicap <= 1)
    rng = np.random.default_rng(SEED)
    games = games[rng.permutation(len(games))]
    frame = pd.DataFrame({"game": games, "player_id": meta["player_id"].to_numpy()[games]})
    frame["order"] = frame.groupby("player_id").cumcount()
    frame = frame[frame["order"] < TRAIN_GAMES].copy()
    size = frame.groupby("player_id")["game"].transform("size")
    frame["probe"] = (size == TRAIN_GAMES) & (frame["order"] < PROBE)
    frame["color"] = meta["player_color"].to_numpy()[frame["game"]].astype(np.int8)
    frame["length"] = meta["length"].to_numpy()[frame["game"]]
    names = json.loads((RUN / "training_players.json").read_text()) if (RUN / "training_players.json").exists() else None
    if names is None:
        gallery, _ = data.pid_splits(meta)
        config = json.loads((RUN / "config.json").read_text())
        held = set(gallery.loc[gallery["split"].isin(config["exclude_splits"]), "player_id"])
        frame["trained"] = ~frame["player_id"].isin(held)
    else:
        frame["trained"] = frame["player_id"].isin(set(names["players"]))
    return frame.sort_values("game").reset_index(drop=True)


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

    OUT.mkdir(parents=True, exist_ok=True)
    model, config, device, min_ply = load_model()
    for name, per_player in PARTS.items():
        if (OUT / f"{name}.npz").exists():
            continue
        moves, colors, meta = load_store(name, STORE, mmap=False)
        cache_dir = OUT / "cache" / name
        if not (cache_dir / "games.parquet").exists():
            write_store_cache(moves, colors, meta, cache_dir)
        games = pd.read_parquet(cache_dir / "games.parquet")
        if len(games) != len(meta):
            raise ValueError(f"{name}: {len(meta) - len(games)} games were dropped by the board cache")
        games["player_id"] = meta["player_id"].astype(str).to_numpy()[games["store_row"]]
        games["color"] = (meta["color"].astype(str).to_numpy()[games["store_row"]] == "W").astype(np.int8)
        if per_player is not None:
            games = games.sample(frac=1.0, random_state=SEED)
            games = games[games.groupby("player_id").cumcount() < per_player].sort_values("game_index")
        vectors = twocolor.embed_games(model, config, BoardCache(cache_dir), games["game_index"].to_numpy(),
                                       games["color"].to_numpy(), games["length"].to_numpy(), device,
                                       swap=True, min_ply=min_ply)
        np.savez(OUT / f"{name}.npz", vector=vectors, player_id=games["player_id"].to_numpy().astype(str),
                 color=games["color"].to_numpy())
        print(f"{name}: {len(games)} games, {games['player_id'].nunique()} players", flush=True)
    if not (OUT / "train.npy").exists():
        plan = train_plan()
        plan.to_parquet(OUT / "train_plan.parquet", index=False)
        vectors = twocolor.embed_games(model, config, BoardCache(data.CACHE_DIR), plan["game"].to_numpy(),
                                       plan["color"].to_numpy(), plan["length"].to_numpy(), device,
                                       swap=True, min_ply=min_ply)
        np.save(OUT / "train.npy", vectors)
        print(f"train: {len(plan)} games, {plan['player_id'].nunique()} players", flush=True)


def centroids(vectors: np.ndarray, players: np.ndarray):
    unit = twocolor.normalise(vectors.astype(np.float64))
    names, inverse = np.unique(players, return_inverse=True)
    sums = np.zeros((len(names), unit.shape[1]))
    np.add.at(sums, inverse, unit)
    return names, twocolor.normalise(sums)


def best(query: np.ndarray, reference: np.ndarray, skip: np.ndarray | None = None):
    """top-1 cosine, its index and the top-2 cosine; skip[i] is a reference row excluded for query i."""
    sims = query @ reference.T
    if skip is not None:
        sims[np.arange(len(query)), skip] = -np.inf
    order = np.argsort(-sims, axis=1)[:, :2]
    rows = np.arange(len(query))
    return sims[rows, order[:, 0]], order[:, 0], sims[rows, order[:, 1]]


def quantiles(x: np.ndarray) -> dict:
    return {f"q{int(q * 100):02d}": round(float(np.quantile(x, q)), 4) for q in (0.01, 0.5, 0.9, 0.99)} | {
        "max": round(float(x.max()), 4)}


def analyze(args) -> None:
    plan = pd.read_parquet(OUT / "train_plan.parquet")
    train = np.load(OUT / "train.npy")
    probe = plan["probe"].to_numpy()
    players = plan["player_id"].to_numpy().astype(str)
    ref_names, ref = centroids(train[~probe], players[~probe])         # 10 games, or all for small players
    all_names, full = centroids(train, players)                        # up to 20 games: the reference for official sets
    probe_names, probe_c = centroids(train[probe], players[probe])
    own = np.searchsorted(ref_names, probe_names)
    assert np.array_equal(ref_names[own], probe_names)
    trained = plan.groupby("player_id")["trained"].first().loc[probe_names].to_numpy()

    result = {"train_players": len(all_names), "probe_players": len(probe_names), "calibration": {}, "match": {}}
    sims = probe_c @ ref.T
    own_cos = sims[np.arange(len(probe_names)), own]
    rank1 = sims.argmax(axis=1) == own
    other_cos, _, _ = best(probe_c, ref, own)
    for label, mask in (("encoder-trained", trained), ("encoder-unseen", ~trained)):
        threshold = float(np.quantile(other_cos[mask], 0.99))
        result["calibration"][label] = {
            "players": int(mask.sum()), "own_cosine": quantiles(own_cos[mask]),
            "best_other_cosine": quantiles(other_cos[mask]), "true_match_is_top1": round(float(rank1[mask].mean()), 4),
            "null_q99": round(threshold, 4), "true_match_above_null_q99": round(float((own_cos[mask] > threshold).mean()), 4)}
    threshold = result["calibration"]["encoder-unseen"]["null_q99"]

    official = {}
    for name in PARTS:
        saved = np.load(OUT / f"{name}.npz")
        official[name] = centroids(saved["vector"], saved["player_id"])
    references = {"train": (all_names, full), **official}
    rows = []
    for name, (names, cents) in official.items():
        for ref_name, (r_names, r_cents) in references.items():
            if ref_name == name:
                top, index, second = best(cents, r_cents, np.arange(len(names)))    # nearest other player of the same set
            else:
                top, index, second = best(cents, r_cents)
            entry = {"query": name, "reference": ref_name, "players": len(names), **quantiles(top),
                     "share_above_null_q99": round(float((top > threshold).mean()), 4),
                     "median_margin": round(float(np.median(top - second)), 4)}
            result["match"][f"{name}->{ref_name}"] = entry
            for q, t, i, s in zip(names, top, index, second):
                rows.append({"query_set": name, "query": q, "reference_set": ref_name, "match": r_names[i],
                             "cosine": t, "second": s})
    pd.DataFrame(rows).to_csv(OUT / "nearest.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["calibration"], indent=2))
    print(f"null threshold (encoder-unseen probes, q99 of best other cosine): {threshold:.4f}")
    print(pd.DataFrame(result["match"].values()).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("embed").set_defaults(fn=embed)
    sub.add_parser("analyze").set_defaults(fn=analyze)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
