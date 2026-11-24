"""Competition 1 added set: shared-opponent term on top of the 824896 fusion (2026-11-23, user "再拚一下競賽一";
RESULTS.md, registered before the bench numbers are seen)

pid2_oppdiag: a question's game shares its f8 top-1 opponent with its 824896 top-1 candidate 7-10x more often than with
a random candidate, but only for 2-3 % of games. Term (per question q, candidate c, gallery games only):
  SOFT_tau  OPP[q, c] = sum_{g in q} sum_{h in gallery(c)} <p_g, p_h>,  p = softmax((f_i - f_1) / tau) over the f8 top-10
  HARD      OPP[q, c] = number of (g, h) pairs with the same f8 top-1 and both gaps in bin >= 1
S' = S + beta * OPP, S = the saved 824896-form scores of the qmerge2 bench draws (comb + pseudo, evaluate11).
Bench opponent links: the opponent side of every bank game (47,280; owner column masked), same f8 / galleries / bin
edges as rank_opp_link2 stage 0. Selection: draws pool 0 4600-4629 (arm x beta). Sealed: pool 1 4600-4629 and pool 0
4630-4659. Rule (11/20): pool 1 delta > 0 with z >= 2.5, every k >= -0.005, pool 0 delta > 0 -> export hook.

    python -m scripts.identity.pid2_oppcomp embed --shard I --shards N
    python -m scripts.identity.pid2_oppcomp score --workers 40
    python -m scripts.identity.pid2_oppcomp evaluate
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params

import scripts.strength.rank_link2 as l2
import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_opp_link2 as o2
import scripts.strength.rank_opp_fuse as fu

OUT = ROOT / "outputs" / "pid" / "oppcomp"
VEC = OUT / "vectors"
DRAWS = ROOT / "outputs" / "pid" / "likelihood" / "qmerge2" / "draws"
DEV = [(0, d) for d in range(4600, 4630)]
SEAL0 = [(0, d) for d in range(4630, 4660)]
SEAL1 = [(1, d) for d in range(4600, 4630)]
TAUS = (1.0, 3.5)
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
N_CAND = 8007


def bank():
    from scripts.identity.pid2_paired import BANK
    return pd.read_parquet(BANK)


def embed(args) -> None:
    import torch
    VEC.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    games = np.sort(bank()["game"].to_numpy())
    colour = (1 - meta["player_color"].to_numpy()[games]).astype(np.int8)
    length = meta["length"].to_numpy()[games]
    tag = f"{args.shard}of{args.shards}"
    idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
    np.save(VEC / f"bank_rows_{tag}.npy", idx)
    t0 = time.time()
    for enc, (model, config, head, device) in l2.load_encoders().items():
        path = VEC / f"bank_{enc}_{tag}.npy"
        if not path.exists():
            np.save(path, l2.head_game_vectors(model, config, head, device, data.CACHE_DIR, games[idx], colour[idx],
                                               length[idx]))
        print(f"shard {tag} {enc}: {len(idx)} games, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


def score(args) -> None:
    from multiprocessing import get_context
    o2.VEC = VEC
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
    games = np.sort(bank()["game"].to_numpy())
    v = o2.load_opp_vectors("bank", args.shards, len(games), None)
    q = pd.DataFrame({"set": "bank", "part": "train", "game_index": games,
                      "qcolor": (1 - pcolor[games]).astype(np.int8), "owner": player[games]})
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
    prefix = data.opening_prefix_d4(meta, PLIES)
    o2.G.update(q=q, prefix=prefix, hashes=opening.prefix_hashes(prefix), galleries=galleries, population=population,
                params=load_params("d4"), pidx=games, qvec=v, cent=cent,
                owner_col=column.loc[q["owner"]].to_numpy(), w=float(np.load(l2.NEW_W)["w"]))
    tasks = []
    for c, g in q.groupby("qcolor", sort=True):
        idx = g.index.to_numpy()
        tasks += [(("bank", c, i), idx[i:i + o2.TASK]) for i in range(0, len(idx), o2.TASK)]
    t0 = time.time()
    idx = np.full((len(q), o2.TOP), -1, np.int32)
    val = np.full((len(q), o2.TOP), np.nan, np.float32)
    with get_context("fork").Pool(args.workers) as pool:
        for r in pool.imap_unordered(o2.run_task, tasks):
            idx[r["rows"]], val[r["rows"]] = r["f8"]
    edges = fu.link_table().attrs["edges"]
    gap = val[:, 0] - val[:, 1]
    b = np.zeros(len(q), np.int64)
    for c in (0, 1):
        m = q["qcolor"].to_numpy() == c
        b[m] = np.searchsorted(np.array(edges[c]), gap[m], side="right")
    np.savez(OUT / "bank_links.npz", game=games, idx=idx, val=val, bin=b)
    print(f"scored {len(q)} bank games, {time.time() - t0:.0f}s", flush=True)


def link_matrix(idx: np.ndarray, val: np.ndarray, b: np.ndarray, arm: str) -> sp.csr_matrix:
    """rows = games, cols = 8,007 training players."""
    n = len(idx)
    if arm == "HARD":
        keep = b >= 1
        return sp.csr_matrix((keep.astype(np.float64), (np.arange(n), idx[:, 0])), shape=(n, N_CAND))
    tau = float(arm[4:])
    v = val.astype(np.float64)
    w = np.exp((v - v[:, :1]) / tau)
    w /= w.sum(1, keepdims=True)
    return sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), idx.shape[1]), idx.ravel())), shape=(n, N_CAND))


def group_sum(M: sp.csr_matrix, row_of: dict, groups: list) -> sp.csr_matrix:
    rows, cols = [], []
    for i, g in enumerate(groups):
        r = [row_of[x] for x in g]
        rows += [i] * len(r)
        cols += r
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(groups), M.shape[0]))
    return A @ M


def official_opp(prefix: str, qids: np.ndarray, cand: np.ndarray, cell: str) -> np.ndarray:
    """[len(qids), len(cand)] OPP of one added part (prefix pub2 / priv2) from the pid2_oppdiag official links."""
    arm, beta = cell.rsplit("_", 1)
    part = "public" if prefix == "pub2" else "private"
    diag = ROOT / "outputs" / "pid" / "oppdiag"
    links = pd.read_parquet(diag / "links.parquet")
    top = np.load(diag / "top10.npz")
    M = link_matrix(top["idx"], top["val"], links["bin"].to_numpy(), arm)
    out = []
    for name, keys in ((f"pid2_{part}", qids), (f"pool2_{part}", cand)):
        rows = np.flatnonzero((links["set"] == name).to_numpy())
        owners = links["owner"].to_numpy()[rows]
        pos = pd.Series(np.arange(len(keys)), index=keys)
        if not set(owners) <= set(keys):
            raise ValueError(f"{name}: link owners outside the {prefix} ids")
        A = sp.csr_matrix((np.ones(len(rows)), (pos.loc[owners].to_numpy(), rows)), shape=(len(keys), M.shape[0]))
        out.append(A @ M)
    return float(beta) * (out[0] @ out[1].T).toarray()


def official_points(S: np.ndarray, owner: np.ndarray) -> np.ndarray:
    from scripts.identity.pid2_cheap import official
    return official(S, owner)


def evaluate(_args) -> dict:
    from go_aicup.data import cv
    z = np.load(OUT / "bank_links.npz")
    row_of = {int(g): i for i, g in enumerate(z["game"])}
    bk = bank()
    arms = [f"SOFT{t:g}" for t in TAUS] + ["HARD"]
    mats = {a: link_matrix(z["idx"], z["val"], z["bin"], a) for a in arms}
    per = {}
    for pool, d in DEV + SEAL0 + SEAL1:
        f = DRAWS / f"pool{pool}_draw{d}.npz"
        saved = np.load(f)
        candidates, gallery, q = cv.draw_pid2(bk, pool, d)
        owner = np.searchsorted(candidates, q["player_id"].to_numpy())
        if not np.array_equal(owner, saved["owner"]) or saved["S"].shape != (len(q), len(candidates)):
            raise ValueError(f"draw {pool}/{d} does not match the saved scores")
        S, ks = saved["S"], saved["k"]
        base = official_points(S, owner)
        out = {"k": ks, "base": base}
        for a in arms:
            OQ = group_sum(mats[a], row_of, list(q["games"]))
            OC = group_sum(mats[a], row_of, gallery)
            OPP = (OQ @ OC.T).toarray()
            for beta in BETAS:
                out[f"{a}_{beta:g}"] = official_points(S + beta * OPP, owner)
            out[f"{a}_stats"] = (float((OPP[np.arange(len(q)), owner] > 0).mean()), float((OPP > 0).mean()))
        per[pool, d] = out
    cells = [f"{a}_{b:g}" for a in arms for b in BETAS]

    def delta(draws, cell):
        ds = np.array([per[x][cell].mean() - per[x]["base"].mean() for x in draws])
        byk = {int(k): float(np.mean([per[x][cell][per[x]["k"] == k].mean() - per[x]["base"][per[x]["k"] == k].mean()
                                       for x in draws])) for k in (1, 2, 3, 5)}
        return float(ds.mean()), float(ds.std(ddof=1) / np.sqrt(len(ds))), float((ds > 0).mean()), byk

    dev = {c: delta(DEV, c) for c in cells}
    chosen = max(cells, key=lambda c: dev[c][0])
    s1, s0 = delta(SEAL1, chosen), delta(SEAL0, chosen)
    z1 = s1[0] / s1[1] if s1[1] > 0 else 0.0
    passed = s1[0] > 0 and z1 >= 2.5 and min(s1[3].values()) >= -0.005 and s0[0] > 0
    stats = {a: [float(np.mean([per[x][f"{a}_stats"][0] for x in DEV])), float(np.mean([per[x][f"{a}_stats"][1] for x in DEV]))]
             for a in arms}
    result = {"dev": {c: {"delta": round(v[0], 5), "se": round(v[1], 5), "positive": v[2],
                          "by_k": {k: round(x, 4) for k, x in v[3].items()}} for c, v in dev.items()},
              "chosen": chosen, "nonzero_owner_vs_any": stats,
              "sealed_pool1": {"delta": s1[0], "se": s1[1], "z": z1, "positive": s1[2], "by_k": s1[3]},
              "sealed_pool0": {"delta": s0[0], "se": s0[1], "positive": s0[2], "by_k": s0[3]},
              "base_mean": {"dev": float(np.mean([per[x]["base"].mean() for x in DEV])),
                            "pool1": float(np.mean([per[x]["base"].mean() for x in SEAL1]))},
              "pass": bool(passed)}
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
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
    sub.add_parser("evaluate").set_defaults(fn=evaluate)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
