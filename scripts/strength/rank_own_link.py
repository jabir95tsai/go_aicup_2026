"""Are competition-2 test players among the labelled training players? k-game GE2E check (RESULTS 2026-11-13)

l10_check (11/11) scored each test question with the opening model against the 8,007 training
players: 7 questions beyond every unseen reference, and a mixture estimate of 12% / 20% training
players that the tool (29% recall) could not resolve. Here a question is the normalised mean of its
games' two-colour GE2E vectors, compared by cosine with the player centroids of rank_opp_link
(up to 40 HD<=1 games per player).

Reference: per colour and k in 5/10/15/20, REF questions of distinct training players built from
games outside their gallery sample ("seen"); the same rows with the own column removed are the
"unseen" null. Statistic: top-1 minus top-2 cosine.

    python -m scripts.strength.rank_own_link plan
    python -m scripts.strength.rank_own_link embed          # 1 GPU, minutes
    python -m scripts.strength.rank_own_link analyze [--base submissions/rank/751959-cvec2.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data

import scripts.strength.rank_opp_link as ol

OUT = ol.OUT / "own"
REF, SEED = 150, 2028
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    free = meta["handicap"].to_numpy() <= 1
    free[work["game_index"].to_numpy()] = False
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for color in (0, 1):
        idx = np.flatnonzero(free & (pcolor == color))
        groups = pd.Series(idx).groupby(player[idx]).apply(np.asarray)
        for k in K_BUCKETS:
            eligible = groups[groups.map(len) >= k]
            chosen = rng.choice(len(eligible), size=min(REF, len(eligible)), replace=False)
            for j in chosen:
                games = rng.choice(eligible.iloc[j], size=k, replace=False)
                rows.append(pd.DataFrame({"question": f"ref_{color}_{k}_{j}", "k": k, "color": color,
                                          "player_id": eligible.index[j], "game_index": games}))
    table = pd.concat(rows, ignore_index=True)
    table.to_parquet(OUT / "ref.parquet", index=False)
    print(table.groupby(["color", "k"])["question"].nunique().to_dict(), "games", len(table))


def embed(_args) -> None:
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity import twocolor

    model, config, device, min_ply = ol.load_model()
    table = pd.read_parquet(OUT / "ref.parquet")
    games = np.unique(table["game_index"].to_numpy())
    meta = data.load_meta()
    vectors = twocolor.embed_games(model, config, BoardCache(data.CACHE_DIR), games,
                                   meta["player_color"].to_numpy()[games].astype(np.int8),
                                   meta["length"].to_numpy()[games], device, swap=True, min_ply=min_ply)
    np.savez(OUT / "ref_vectors.npz", game_index=games, vectors=vectors)
    print(f"embedded {len(games)} reference games")


def centroids_and_ranks(meta: pd.DataFrame):
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    shards = sorted(ol.OUT.glob("train_*of*.npy"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    vectors = ol.unit(np.concatenate([np.load(p) for p in shards]).astype(np.float64))
    if len(vectors) != len(work):
        raise ValueError("train shards do not match the plan")
    player = meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)
    g = np.flatnonzero(work["set"].to_numpy() == "gallery")
    owner = column.loc[player[work["game_index"].to_numpy()[g]]].to_numpy()
    sums = np.zeros((len(candidates), vectors.shape[1]))
    np.add.at(sums, owner, vectors[g])
    empty = np.linalg.norm(sums, axis=1) == 0
    sums[empty] = 1e-9
    cand_rank = pd.Series(meta["rank_id"].to_numpy().astype(np.int64), index=player).groupby(level=0).first()
    return candidates, column, ol.unit(sums), cand_rank.loc[candidates].to_numpy(), np.bincount(owner, minlength=len(candidates))


def question_vectors(game_vectors: np.ndarray, labels: np.ndarray):
    """normalised mean of unit game vectors per label (first-appearance order)."""
    codes, names = pd.factorize(labels, sort=False)
    sums = np.zeros((len(names), game_vectors.shape[1]))
    np.add.at(sums, codes, ol.unit(game_vectors.astype(np.float64)))
    return np.asarray(names), ol.unit(sums)


def analyze(args) -> None:
    meta = ol.load_train()
    candidates, column, centroids, cand_rank, gallery_size = centroids_and_ranks(meta)
    ref = pd.read_parquet(OUT / "ref.parquet")
    packed = np.load(OUT / "ref_vectors.npz")
    position = pd.Series(np.arange(len(packed["game_index"])), index=packed["game_index"])
    names, vectors = question_vectors(packed["vectors"][position.loc[ref["game_index"]].to_numpy()],
                                      ref["question"].to_numpy())
    info = ref.groupby("question", sort=False)[["k", "color", "player_id"]].first().loc[names]
    own = column.loc[info["player_id"].astype(str)].to_numpy()
    scores = vectors @ centroids.T
    arg, _, gap = ol.top2(scores)
    hit = arg == own
    masked = scores.copy()
    masked[np.arange(len(own)), own] = -np.inf
    _, _, null_gap = ol.top2(masked)
    gallery_table, _ = data.pid_splits(data.load_meta())
    untrained = np.isin(info["player_id"].astype(str), gallery_table.loc[
        gallery_table["split"].isin(ol.EXCLUDED_SPLITS), "player_id"].astype(str))
    ks = info["k"].to_numpy()
    result = {"reference": {}, "test": {}}
    thresholds = {}
    for k in K_BUCKETS:
        m = ks == k
        t95, t99, tmax = (float(np.quantile(null_gap[m], 0.95)), float(np.quantile(null_gap[m], 0.99)),
                          float(null_gap[m].max()))
        thresholds[k] = (t95, t99, tmax)
        result["reference"][str(k)] = {
            "n": int(m.sum()), "top1": float(hit[m].mean()),
            "top1_ge2e_untrained": float(hit[m & untrained].mean()), "n_untrained": int((m & untrained).sum()),
            "top1_gallery<40": float(hit[m & (gallery_size[own] < 40)].mean()) if (m & (gallery_size[own] < 40)).any() else None,
            "median_gap_seen": float(np.median(gap[m])), "median_gap_null": float(np.median(null_gap[m])),
            "null_q95": t95, "null_q99": t99, "null_max": tmax,
            "recall_q95": float((gap[m] > t95).mean()), "recall_q99": float((gap[m] > t99).mean()),
            "recall_max": float((gap[m] > tmax).mean()),
            "precision_of_top1_above_max": float(hit[m & (gap > tmax)].mean()) if (m & (gap > tmax)).any() else None}

    base = pd.read_csv(ROOT / args.base).set_index("question_id")["pred_rank"] if args.base else None
    flagged = []
    for part in ol.PARTS:
        store_meta = pd.read_parquet(ol.STORE / f"{part}_meta.parquet")
        q_names, q_vectors = question_vectors(np.load(ol.OUT / f"{part}_own.npy"), store_meta["question_id"].to_numpy())
        q_k = store_meta.groupby("question_id", sort=False)["num_games"].first().loc[q_names].to_numpy()
        a, top, g = ol.top2(q_vectors @ centroids.T)
        out = {}
        for k in K_BUCKETS:
            m = q_k == k
            t95, t99, tmax = thresholds[k]
            r = result["reference"][str(k)]
            above = {"q95": float((g[m] > t95).mean()), "q99": float((g[m] > t99).mean()), "max": float((g[m] > tmax).mean())}
            out[str(k)] = {"n": int(m.sum()), "median_gap": float(np.median(g[m])), "above": above,
                           "share_q95": ol.share(above["q95"], r["recall_q95"], 0.05),
                           "share_q99": ol.share(above["q99"], r["recall_q99"], 0.01),
                           "share_max": ol.share(above["max"], r["recall_max"], 0.0)}
        limit = np.array([thresholds[k][2] for k in q_k])
        level = np.array([thresholds[k][1] for k in q_k])
        table = pd.DataFrame({"part": part, "question_id": q_names, "k": q_k, "gap": g, "top1_cos": top,
                              "player": candidates[a], "matched_rank": [RANKS[r] for r in cand_rank[a]],
                              "above_q99": g > level, "above_max": g > limit})
        if base is not None:
            table["base_rank"] = base.loc[q_names].to_numpy()
            diff = np.abs(table["matched_rank"].map(RANKS.index) - table["base_rank"].map(RANKS.index))
            table["rank_diff"] = diff
            for label, m in (("above_max", table["above_max"]), ("above_q99_only", table["above_q99"] & ~table["above_max"]),
                             ("below_q99", ~table["above_q99"])):
                if m.any():
                    out[f"agreement_{label}"] = {"n": int(m.sum()), "same": float((diff[m] == 0).mean()),
                                                 "within1": float((diff[m] <= 1).mean())}
        out["n_above_max"] = int(table["above_max"].sum())
        result["test"][part] = out
        flagged.append(table)
    pd.concat(flagged, ignore_index=True).to_csv(OUT / "test_questions.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    sub.add_parser("embed").set_defaults(fn=embed)
    p = sub.add_parser("analyze")
    p.add_argument("--base", type=Path, default=Path("submissions/rank/751959-cvec2.csv"))
    p.set_defaults(fn=analyze)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
