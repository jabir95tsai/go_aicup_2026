"""Bench for the added competition-1 test set (600 candidates x 10 games of both colours,
questions of 1/2/3/5 games; cv.make_pid2_bank / cv.draw_pid2).

    python -m scripts.identity.pid2_bench embed [--run-dir RUN]   # 1 GPU: GE2E vectors of the bank games
    python -m scripts.identity.pid2_bench eval  [--run-dir RUN]   # CPU: variants, pool 0 selects, pool 1 reports

Outputs go to outputs/pid/pid2_bench/<run name>/.

Variants (plain top-5, no assignment; official score per k):
  floor        the 11/12 floor: black games only, opening + 8 x GE2E; a question without a
               black game scores nothing
  opening      two-colour opening (twocolor.opening_scores), parameters as the original set
  style-*      GE2E alone: black games only / white moves raw / white moves colour-swapped
  fused        opening + lambda * w * style (+ mu * HD-and-colour histogram); lambda, mu and the
               opening parameters are chosen on pool 0, everything is reported on pool 1
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.metrics import PID_WEIGHTS
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import assign, data, opening, twocolor
from go_aicup.tasks.identity.predict import load_params, train_prefix

OUT = ROOT / "outputs" / "pid" / "pid2_bench"
BANK = PROCESSED / "identity" / "pid2_bank.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
DRAWS = 5
LAMBDAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MUS = (0.0, 0.5, 1.0, 2.0)


def load_bank(meta: pd.DataFrame):
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    merged = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left", validate="1:1")
    handicap = merged["handicap"].to_numpy()
    assert not np.isnan(handicap).any()
    handicap = handicap.astype(np.int64)
    if BANK.exists():
        return pd.read_parquet(BANK), handicap
    gallery, _ = data.pid_splits(meta)
    bank = cv.make_pid2_bank(meta, handicap, gallery)
    bank.to_parquet(BANK, index=False)
    return bank, handicap


def embed(args) -> None:
    """Vectors of every bank game with the encoder of --run-dir (its config's sampling decides
    the first ply; white moves colour-swapped, plus the raw variant for the original encoder)."""
    import torch
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity.ge2e import MIN_PLY, IdentityEncoder

    meta = data.load_meta()
    bank, _ = load_bank(meta)
    games = np.sort(bank["game"].to_numpy())
    colors = meta["player_color"].to_numpy()[games].astype(np.int8)
    lengths = meta["length"].to_numpy()[games]
    config = json.loads((args.run_dir / "config.json").read_text())
    payload = torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=False)
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(payload["model"])
    sampling = config.get("sampling", {})
    min_ply = sampling.get("min_ply", MIN_PLY)
    device = torch.device("cuda")
    model.to(device)
    cache = BoardCache(data.CACHE_DIR)
    swap = twocolor.embed_games(model, config, cache, games, colors, lengths, device, swap=True, min_ply=min_ply)
    out = {"game_index": games, "swap": swap, "w": float(payload["model"]["w"])}
    if not sampling:
        white = colors == 1
        raw = swap.copy()
        raw[white] = twocolor.embed_games(model, config, cache, games[white], colors[white], lengths[white],
                                          device, swap=False)
        out["raw"] = raw
    target = OUT / args.run_dir.name
    target.mkdir(parents=True, exist_ok=True)
    np.savez(target / "embeddings.npz", **out)
    print(f"embedded {len(games)} games with {args.run_dir.name} (min_ply {min_ply})")


def score_table(scores, candidates, q) -> pd.Series:
    """Official score per k bucket of the added set (metrics.pid_report buckets are 5/10/15/20)."""
    ranked = candidates[assign.top5(scores)]
    hit = ranked == q["player_id"].to_numpy()[:, None]
    points = (hit * PID_WEIGHTS).sum(axis=1)
    ks = q["k"].to_numpy()
    return pd.Series({**{str(k): float(points[ks == k].mean()) for k in cv.PID2_K}, "all": float(points.mean())})


def evaluate(args) -> None:
    out_dir = OUT / args.run_dir.name
    meta, prefix = train_prefix("d4")
    bank, handicap = load_bank(meta)
    color = meta["player_color"].to_numpy().astype(np.int64)
    pid = meta["player_id"].astype(str).to_numpy()
    saved = np.load(out_dir / "embeddings.npz")
    position = np.full(len(meta), -1)
    position[saved["game_index"]] = np.arange(len(saved["game_index"]))
    w = float(saved["w"])
    base = load_params("d4")
    hashes = opening.prefix_hashes(prefix)
    codes = np.minimum(handicap, 1) + 2 * color
    bench_players = bank["player_id"].unique()
    outside = ~np.isin(pid, bench_players)
    populations = (np.flatnonzero(outside & (color == 0)), np.flatnonzero(outside & (color == 1)))

    grid = [dict(base, moves=m, beta=b, slot_moves=s)
            for m, b, s in itertools.product((4, 8, 12), (0.1, 0.3, 1.0), (0, 4))]
    parts = {}          # (pool, draw) -> dict of score matrices
    for pool, draw in itertools.product((0, 1), range(DRAWS)):
        candidates, gallery, q = cv.draw_pid2(bank, pool, draw)
        query = list(q["games"])
        black = lambda games: [g[color[g] == 0] for g in games]
        item = {"candidates": candidates, "q": q}
        item["opening"] = twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, base)
        if pool == 0:
            item["grid"] = [twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, p)
                            for p in grid]
        vec = {name: saved[name] if name in saved else saved["swap"] for name in ("swap", "raw")}
        gal_pos, qry_pos = [position[g] for g in gallery], [position[g] for g in query]
        item["style-swap"] = w * twocolor.style_scores(vec["swap"], gal_pos, qry_pos)
        item["style-raw"] = w * twocolor.style_scores(vec["raw"], gal_pos, qry_pos)
        item["category"] = twocolor.category_scores(codes, gallery, query, 4)
        # the floor: black games only on both sides
        gal_b, qry_b = black(gallery), black(query)
        has_q = np.array([len(g) > 0 for g in qry_b])
        has_c = np.array([len(g) > 0 for g in gal_b])
        floor = np.zeros((len(query), len(candidates)))
        style_black = np.zeros_like(floor)
        if has_q.any():
            sub_g = [g for g in gal_b if len(g)]
            sub_q = [g for g in qry_b if len(g)]
            only_b = (populations[0], populations[1])
            op = twocolor.opening_scores(prefix, hashes, sub_g, sub_q, color, only_b, base)
            st = w * twocolor.style_scores(vec["swap"], [position[g] for g in sub_g], [position[g] for g in sub_q])
            floor[np.ix_(has_q, has_c)] = op + 8.0 * st
            style_black[np.ix_(has_q, has_c)] = st
            floor[np.ix_(has_q, ~has_c)] = -1e9
            style_black[np.ix_(has_q, ~has_c)] = -1e9
        # a question without a black game: the floor file names five arbitrary candidates
        rng = np.random.default_rng(draw)
        floor[~has_q] = rng.random((int((~has_q).sum()), len(candidates)))
        style_black[~has_q] = floor[~has_q]
        item["floor"], item["style-black"] = floor, style_black
        parts[pool, draw] = item

    def mean_score(pool, fn) -> pd.Series:
        return pd.concat([score_table(fn(parts[pool, d]), parts[pool, d]["candidates"], parts[pool, d]["q"])
                          for d in range(DRAWS)], axis=1).mean(axis=1)

    # selection on pool 0
    grid_all = [float(mean_score(0, lambda it, i=i: it["grid"][i])["all"]) for i in range(len(grid))]
    best_grid = int(np.argmax(grid_all))
    style = max(("style-swap", "style-raw"), key=lambda n: float(mean_score(0, lambda it: it[n])["all"]))
    curve = {(lam, mu): float(mean_score(0, lambda it: it["grid"][best_grid] + lam * it[style]
                                         + mu * it["category"])["all"])
             for lam, mu in itertools.product(LAMBDAS, MUS)}
    lam, mu = max(curve, key=curve.get)
    per_k = {}
    for k in cv.PID2_K:
        ck = {l: float(mean_score(0, lambda it: it["grid"][best_grid] + l * it[style] + mu * it["category"])[str(k)])
              for l in LAMBDAS}
        per_k[k] = max(ck, key=ck.get)
    chosen = grid[best_grid]
    print("opening grid (pool 0, all):")
    for p, s in zip(grid, grid_all):
        print(f"  moves={p['moves']:2d} beta={p['beta']:.1f} slot={p['slot_moves']}: {s:.4f}")
    print(f"chosen opening {chosen}; style={style}; lambda={lam} mu={mu}; per-k lambda={per_k}")
    print("fusion curve (pool 0, all):", {f"{a}/{b}": round(v, 4) for (a, b), v in curve.items()})

    def chosen_opening(it):
        gallery_item = it.get("grid")
        return gallery_item[best_grid] if gallery_item is not None else it["opening_chosen"]

    # pool 1 needs the chosen opening parameters
    for d in range(DRAWS):
        candidates, gallery, q = cv.draw_pid2(bank, 1, d)
        parts[1, d]["opening_chosen"] = twocolor.opening_scores(prefix, hashes, gallery, list(q["games"]), color,
                                                                populations, chosen)

    def per_k_fused(it):
        scores = np.zeros_like(it[style])
        ks = it["q"]["k"].to_numpy()
        for k, l in per_k.items():
            rows = ks == k
            scores[rows] = (chosen_opening(it) + l * it[style] + mu * it["category"])[rows]
        return scores

    variants = {
        "floor (black only)": lambda it: it["floor"],
        "opening, original params": lambda it: it["opening"],
        "opening, chosen params": chosen_opening,
        "style black only": lambda it: it["style-black"],
        "style raw": lambda it: it["style-raw"],
        "style swap": lambda it: it["style-swap"],
        "category (HD x colour)": lambda it: it["category"],
        f"opening + {lam} x style": lambda it: chosen_opening(it) + lam * it[style],
        f"opening + {lam} x style + {mu} x category": lambda it: chosen_opening(it) + lam * it[style]
        + mu * it["category"],
        "fused, per-k lambda": per_k_fused,
    }
    rows = []
    for pool in (0, 1):
        for name, fn in variants.items():
            rows.append({"pool": pool, "variant": name, **mean_score(pool, fn).round(4).to_dict()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    # spread over draws for the two key rows on pool 1
    for name in ("floor (black only)", f"opening + {lam} x style + {mu} x category"):
        per_draw = [float(score_table(variants[name](parts[1, d]), parts[1, d]["candidates"], parts[1, d]["q"])["all"])
                    for d in range(DRAWS)]
        print(name, "pool 1 per draw:", np.round(per_draw, 4).tolist())
    table.to_csv(out_dir / "eval.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({
        "run_dir": str(args.run_dir), "opening": chosen, "style": style, "lambda": lam, "mu": mu, "w": w,
        "lambda_per_k": {str(k): v for k, v in per_k.items()},
        "curve_pool0": {f"{a}/{b}": v for (a, b), v in curve.items()}}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("embed", embed), ("eval", evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/pid/ge2e/pid_ge2e_v1_short")
        p.set_defaults(func=fn)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
