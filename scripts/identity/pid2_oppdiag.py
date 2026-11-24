"""Competition 1 added set: do a player's games share opponents more than random training games do? Diagnostic.
(2026-11-23, user "競賽一再拚一下"; RESULTS.md, registered before the run)

Valid-S (random 10 % of each player's games over 2021-2023): the f8 top-1 opponent of an owner's game recurs in
another of the owner's games for 3.2 % of games (bin >= q90), 2-3x the different-owner rate. If the organisers drew the
added-set games from a time window, repeats would be far more common and could verify "same player" between questions
(the question-merge bound is +0.099 on the bench). Here the opponent sides of every added-set candidate game
(1,200 x 10) and test game (pub2 / priv2) are linked to the 8,007 training players exactly as stage 0 (f8, same
galleries, same per-colour bin edges), then:
  (a) within-candidate repeat share (bin >= 1), against Valid-S 0.032 and the cross-candidate rate in the same pool
  (b) question -> its 824896 top-1 candidate: share of query games whose top-1 opponent is among that candidate's
      top-1 opponents, against random other candidates of the same pool
Gate for a merge / component plan: (a) >= 0.10 and (b) ratio >= 5. Otherwise closed.

    python -m scripts.identity.pid2_oppdiag embed --shard I --shards N
    python -m scripts.identity.pid2_oppdiag score --workers 40
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

import scripts.strength.rank_link2 as l2
import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_opp_link2 as o2
import scripts.strength.rank_opp_fuse as fu

OUT = ROOT / "outputs" / "pid" / "oppdiag"
VEC = OUT / "vectors"
PARTS = {"pool2_public": l2.POOL_CACHE["pool2_public"], "pool2_private": l2.POOL_CACHE["pool2_private"],
         "pid2_public": l2.ART / "public2_d4_9600" / "pid2_public" / "cache",
         "pid2_private": l2.ART / "private2_d4_9600" / "pid2_private" / "cache"}


def inputs(name: str):
    _, _, m = load_store(name, l2.PID_STORE, mmap=False)
    games = pd.read_parquet(PARTS[name] / "games.parquet")
    rows = games["store_row"].to_numpy()
    if len(games) != len(m) or not np.array_equal(games["length"].to_numpy(), m["length"].to_numpy()[rows]):
        raise ValueError(f"{name}: board cache does not match the store")
    own = (m["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
    return games["game_index"].to_numpy(), (1 - own).astype(np.int8), games["length"].to_numpy(), rows, m


def embed(args) -> None:
    import torch
    VEC.mkdir(parents=True, exist_ok=True)
    tag = f"{args.shard}of{args.shards}"
    encoders = l2.load_encoders()
    t0 = time.time()
    for name, cache_dir in PARTS.items():
        games, colour, length, _, _ = inputs(name)
        idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
        np.save(VEC / f"{name}_rows_{tag}.npy", idx)
        for enc, (model, config, head, device) in encoders.items():
            path = VEC / f"{name}_{enc}_{tag}.npy"
            if not path.exists():
                np.save(path, l2.head_game_vectors(model, config, head, device, cache_dir, games[idx], colour[idx],
                                                   length[idx]))
        print(f"shard {tag} {name}: {len(idx)} games, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


def score(args) -> None:
    from multiprocessing import get_context
    o2.VEC_SAVED = o2.VEC
    o2.VEC = VEC                                    # load_opp_vectors reads this module-level path
    meta = ol.load_train()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)
    train_games = np.load(l2.OUT / "train_games.npy")
    tv = l2.load_vectors("train", 4, len(train_games), None)
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    sums = l2.sums_by({n: v[np.searchsorted(train_games, gal)] for n, v in tv.items()},
                      column.loc[player[gal]].to_numpy(), len(candidates))
    cent = {n: l2.unit_rows(x) for n, x in sums.items()}
    del tv
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    blocks, size = [data.opening_prefix_d4(meta, PLIES)], len(meta)
    qs, qvec, pidx, metas = [], {e: [] for e in l2.ENCODERS}, [], {}
    for name in PARTS:
        moves, colors, m = load_store(name, l2.PID_STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, m, "d4", min_plies)
        blocks.append(pre)
        _, colour, _, rows, m = inputs(name)
        v = o2.load_opp_vectors(name, args.shards, len(rows), rows)
        opp = np.empty(len(m), np.int8)
        opp[rows] = colour
        keep = np.flatnonzero(ok)
        for e in l2.ENCODERS:
            qvec[e].append(v[e][keep])
        pidx.append(size + keep)
        qs.append(pd.DataFrame({"set": name, "part": name, "store_row": keep, "qcolor": opp[keep],
                                "owner": m["player_id" if name.startswith("pool") else "question_id"].astype(str)
                                .to_numpy()[keep]}))
        metas[name] = m
        size += len(m)
    q = pd.concat(qs, ignore_index=True)
    prefix = np.concatenate(blocks)
    held = pd.read_parquet(o2.OUT / "queries.parquet").query("set == 'ref'")["game_index"].to_numpy()
    galleries, population = {}, {}
    for c in (0, 1):
        own = (pcolor == c) & hd01
        keep = own.copy()
        keep[held] = False
        gl = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            gl[column[p]] = rows.to_numpy()
        galleries[c], population[c] = gl, np.flatnonzero(own)
    o2.G.update(q=q, prefix=prefix, hashes=opening.prefix_hashes(prefix), galleries=galleries, population=population,
                params=params, pidx=np.concatenate(pidx), qvec={e: np.concatenate(v) for e, v in qvec.items()},
                cent=cent, owner_col=np.full(len(q), -1, np.int64), w=float(np.load(l2.NEW_W)["w"]))
    tasks = []
    for (name, c), g in q.groupby(["set", "qcolor"], sort=True):
        idx = g.index.to_numpy()
        tasks += [((name, c, i), idx[i:i + o2.TASK]) for i in range(0, len(idx), o2.TASK)]
    t0 = time.time()
    res = []
    with get_context("fork").Pool(args.workers) as pool:
        for r in pool.imap_unordered(o2.run_task, tasks):
            res.append(r)
    idx = np.full((len(q), o2.TOP), -1, np.int32)
    val = np.full((len(q), o2.TOP), np.nan, np.float32)
    for r in res:
        idx[r["rows"]], val[r["rows"]] = r["f8"]
    edges = fu.link_table().attrs["edges"]
    gap = val[:, 0] - val[:, 1]
    b = np.zeros(len(q), np.int64)
    for c in (0, 1):
        m = q["qcolor"].to_numpy() == c
        b[m] = np.searchsorted(np.array(edges[c]), gap[m], side="right")
    q["j"], q["bin"], q["gap"] = idx[:, 0], b, gap
    q.to_parquet(OUT / "links.parquet", index=False)
    np.savez(OUT / "top10.npz", idx=idx, val=val)
    print(f"scored {len(q)} games, {time.time() - t0:.0f}s", flush=True)
    analyze(q)


def analyze(q: pd.DataFrame) -> dict:
    rng = np.random.default_rng(2042)
    result = {"games": q.groupby("set").size().to_dict(), "bin_share": {}}
    for name in PARTS:
        x = q[q["set"] == name]
        result["bin_share"][name] = {int(k): round(float(v), 4) for k, v in x["bin"].value_counts(normalize=True).sort_index().items()}
    # (a) within-candidate repeats
    a = {}
    for minbin in (0, 1, 2):
        for pool in ("pool2_public", "pool2_private"):
            x = q[(q["set"] == pool) & (q["bin"] >= minbin)]
            share, same_pairs, tot_pairs = [], 0, 0
            for _, g in x.groupby("owner"):
                js = g["j"].to_numpy()
                if len(js) < 2:
                    continue
                c = pd.Series(js).value_counts()
                share.append(((c[js].to_numpy() - 1) > 0).mean())
                same_pairs += int((c * (c - 1) // 2).sum())
                tot_pairs += len(js) * (len(js) - 1) // 2
            js, ow = x["j"].to_numpy(), x["owner"].to_numpy()
            i1, i2 = rng.integers(len(js), size=200000), rng.integers(len(js), size=200000)
            m = ow[i1] != ow[i2]
            cross = float((js[i1][m] == js[i2][m]).mean())
            a[f"{pool}_bin{minbin}"] = {"games": int(len(x)), "repeat_share": round(float(np.mean(share)), 4),
                                        "same_owner_pair_rate": round(same_pairs / max(tot_pairs, 1), 5),
                                        "cross_owner_pair_rate": round(cross, 5),
                                        "ratio": round(same_pairs / max(tot_pairs, 1) / max(cross, 1e-9), 1)}
    result["a_within_candidate"] = a
    # (b) questions vs their 824896 top-1 candidate
    sub = pd.read_csv(ROOT / "submissions" / "pid" / "824896-comb-pseudo.csv").set_index("question_id")["top_1"]
    b_res = {}
    for minbin in (0, 1):
        for tp, pp in (("pid2_public", "pool2_public"), ("pid2_private", "pool2_private")):
            cand = q[(q["set"] == pp) & (q["bin"] >= minbin)].groupby("owner")["j"].apply(set)
            names = np.array(cand.index)
            tq = q[(q["set"] == tp) & (q["bin"] >= minbin)]
            hit_true, hit_rand, n = 0, 0, 0
            for qid, g in tq.groupby("owner"):
                c = sub.get(qid)
                if c not in cand.index:
                    continue
                js = set(g["j"])
                hit_true += len(js & cand[c])
                other = names[rng.integers(len(names), size=20)]
                hit_rand += np.mean([len(js & cand[o]) for o in other if o != c])
                n += len(js)
            b_res[f"{tp}_bin{minbin}"] = {"query_links": n, "match_top1_candidate": round(hit_true / max(n, 1), 4),
                                         "match_random_candidate": round(hit_rand / max(n, 1), 5),
                                         "ratio": round(hit_true / max(hit_rand, 1e-9), 1)}
    result["b_question_to_candidate"] = b_res
    rep = [a[k]["repeat_share"] for k in a if k.endswith("bin1")]
    rat = [b_res[k]["ratio"] for k in b_res if k.endswith("bin0")]
    result["gate"] = {"repeat_share_bin1": rep, "question_ratio_bin0": rat,
                      "pass": bool(min(rep) >= 0.10 and min(rat) >= 5)}
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps(result, indent=1, default=int), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("embed")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.set_defaults(fn=embed)
    p = sub.add_parser("score")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=40)
    p.set_defaults(fn=score)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
