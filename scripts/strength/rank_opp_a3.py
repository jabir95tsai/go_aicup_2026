"""Competition 2: opponent identification with the overnight competition-1 encoder A3 added to the style term
(2026-11-24, user "用今晚訓練的 A3／A3L 新編碼器改善認人 繼續做"; RESULTS.md, registered before results)

f8' = opening + 8 w (0.25 cos_B + 0.75 mean(cos_A1, cos_A2, cos_A3)) against the same 8,007 training players (style
centroid = the rank_link2 gallery, A3 vectors added), same opening galleries and queries as rank_opp_link2 stage 0.
LLR rerank without new policy: the LLR of a new top-10 candidate is looked up from the f8 top-30 LLR already
computed (llr_g40.npz ranks 1-10 + llr_g40_expanded.npz ranks 11-30, 40-game theta); a candidate outside the old
top-30 gets the row minimum of the looked-up values (pre-registered, conservative). Then rank_opp_llr.fuse unchanged
(mu on reference half A, soft top-10 + DELTA, E1 selection, E2 for the chosen cell only), base cell 845986.

    python -m scripts.strength.rank_opp_a3 embed --shard I --shards N --encoder A3
    python -m scripts.strength.rank_opp_a3 score --workers 80 --encoder A3
    python -m scripts.strength.rank_opp_a3 llr --encoder A3
    python -m scripts.strength.rank_opp_a3 fuse --job ID --encoder A3
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT

import scripts.strength.rank_link2 as l2
import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_opp_link2 as o2

RUNS = {"A3": ("pid_e2e_ogsk_A3_YO7200/step7200", ROOT / "outputs/pid/ge2e/pid_e2e_ogsk_A3_YO7200/step7200/best.pt"),
        "A3L": ("pid_e2e_ogsk_A3L_B30_YO7200/step7200",
                ROOT / "outputs/pid/ge2e/pid_e2e_ogsk_A3L_B30_YO7200/step7200/best.pt")}
LLR_DIR = ROOT / "outputs" / "rank" / "opp_llr"


def out_dir(enc: str) -> Path:
    return ROOT / "outputs" / "rank" / f"opp_{enc.lower()}"


def embed(args) -> None:
    import torch
    from go_aicup.tasks.identity import data
    from go_aicup.tasks.identity.seqagg import load_head
    from scripts.identity.pid2_seqagg import load_encoder
    out = out_dir(args.encoder) / "vectors"
    out.mkdir(parents=True, exist_ok=True)
    run, head_file = RUNS[args.encoder]
    model, config, device = load_encoder(run)
    head, _ = load_head(head_file, device)
    meta = ol.load_train()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    length = meta["length"].to_numpy()
    q = pd.read_parquet(o2.OUT / "queries.parquet")
    tr = q[q["set"].isin(["u_opp", "s_opp"])]
    train_games = np.load(l2.OUT / "train_games.npy")
    jobs = [("own", data.CACHE_DIR, train_games, pcolor[train_games], length[train_games]),
            ("opp", data.CACHE_DIR, tr["game_index"].to_numpy(), tr["qcolor"].to_numpy().astype(np.int8),
             length[tr["game_index"].to_numpy()])]
    for name in ol.PARTS:
        cache_dir, games, own, glen, _ = l2.part_inputs(name, meta)
        jobs.append((name, cache_dir, games, (1 - own).astype(np.int8), glen))
    tag = f"{args.shard}of{args.shards}"
    t0 = time.time()
    for name, cache_dir, games, colour, glen in jobs:
        idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
        np.save(out / f"{name}_rows_{tag}.npy", idx)
        path = out / f"{name}_{tag}.npy"
        if not path.exists():
            vec = l2.head_game_vectors(model, config, head, device, cache_dir, games[idx], colour[idx], glen[idx])
            if not np.isfinite(vec).all():
                raise ValueError(f"{name}: non-finite vectors")
            np.save(path, vec)
        print(f"shard {tag} {name}: {len(idx)} games, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()


def gather(enc: str, name: str, shards: int, n: int, rows=None) -> np.ndarray:
    d = out_dir(enc) / "vectors"
    vec = np.empty((n, 256))
    seen = np.zeros(n, bool)
    for s in range(shards):
        idx = np.load(d / f"{name}_rows_{s}of{shards}.npy")
        vec[idx] = np.load(d / f"{name}_{s}of{shards}.npy")
        seen[idx] = True
    if not seen.all():
        raise ValueError(f"{enc} {name}: missing shards")
    vec = ol.unit(vec)
    if rows is not None:
        store = np.empty_like(vec)
        store[rows] = vec
        vec = store
    return vec


def score(args) -> None:
    """o2.score with a fourth encoder; writes scores.npz (f8 arrays), result.json and queries.parquet to out_dir."""
    from multiprocessing import get_context
    from go_aicup.data.preprocess import load_store
    from go_aicup.tasks.identity import data, opening
    from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix
    enc = args.encoder
    out = out_dir(enc)
    meta = ol.load_train()
    q = pd.read_parquet(o2.OUT / "queries.parquet")
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)
    train_games = np.load(l2.OUT / "train_games.npy")
    tv = l2.load_vectors("train", 4, len(train_games), None)
    tv[enc] = gather(enc, "own", args.shards, len(train_games))
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    names = list(l2.ENCODERS) + [enc]
    sums = l2.sums_by({n: v[np.searchsorted(train_games, gal)] for n, v in tv.items()},
                      column.loc[player[gal]].to_numpy(), len(candidates))
    cent = {n: l2.unit_rows(x) for n, x in sums.items()}
    qvec = {e: np.zeros((len(q), 256)) for e in names}
    ref_rows = np.flatnonzero(q["set"].to_numpy() == "ref")
    pos = np.searchsorted(train_games, q["game_index"].to_numpy()[ref_rows])
    for e in names:
        qvec[e][ref_rows] = tv[e][pos]
    del tv
    tr = np.flatnonzero(q["set"].isin(["u_opp", "s_opp"]).to_numpy())
    old = o2.load_opp_vectors("train", 4, len(tr), None)
    for e in l2.ENCODERS:
        qvec[e][tr] = old[e]
    qvec[enc][tr] = gather(enc, "opp", args.shards, len(tr))
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    blocks, offsets, size = [data.opening_prefix_d4(meta, PLIES)], {}, len(meta)
    for name in ol.PARTS:
        moves, colors, m = load_store(name, ol.STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, m, "d4", min_plies)
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
        rows = np.flatnonzero((q["part"] == name).to_numpy())
        _, _, _, _, store_rows = l2.part_inputs(name, meta)
        v = o2.load_opp_vectors(name, 4, len(store_rows), store_rows)
        for e in l2.ENCODERS:
            qvec[e][rows] = v[e]
        qvec[enc][rows] = gather(enc, name, args.shards, len(store_rows), store_rows)
    for e in names:
        if not np.isclose(np.linalg.norm(qvec[e], axis=1), 1.0).all():
            raise ValueError(f"{e}: query vector missing")
    prefix = np.concatenate(blocks)
    pidx = np.where(q["part"].to_numpy() == "train", q["game_index"].to_numpy(),
                    q["store_row"].to_numpy() + q["part"].map(offsets).fillna(0).to_numpy().astype(np.int64))
    held = q.loc[q["set"] == "ref", "game_index"].to_numpy()
    galleries, population = {}, {}
    for c in (0, 1):
        own = (pcolor == c) & hd01
        keep = own.copy()
        keep[held] = False
        gl = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            gl[column[p]] = rows.to_numpy()
        galleries[c], population[c] = gl, np.flatnonzero(own)
    owner_col = np.full(len(q), -1, np.int64)
    trainish = q["set"].isin(["ref", "u_opp", "s_opp"]).to_numpy()
    owner_col[trainish] = column.loc[q["owner"].to_numpy()[trainish]].to_numpy()
    # style = 0.25 B + 0.75 mean(A1, A2, enc): patch the module-level weights the worker reads (fork inherits)
    l2.ENCODERS = dict(l2.ENCODERS) | {enc: RUNS[enc]}
    l2.WEIGHTS = {"B": 0.25, "A1": 0.25, "A2": 0.25, enc: 0.25}
    o2.LAMBDAS = (8.0,)
    o2.G.update(q=q, prefix=prefix, hashes=opening.prefix_hashes(prefix), galleries=galleries, population=population,
                params=params, pidx=pidx, qvec=qvec, cent=cent, owner_col=owner_col, w=float(np.load(l2.NEW_W)["w"]),
                width=o2.TOP)
    tasks = []
    for (name, colour), g in q.groupby(["set", "qcolor"], sort=True):
        idx = g.index.to_numpy()
        tasks += [((name, colour, i), idx[i:i + o2.TASK]) for i in range(0, len(idx), o2.TASK)]
    t0 = time.time()
    n = len(q)
    save = {k: np.full((n, o2.TOP), -1, np.int32) for k in ("f8_idx", "seen_f8_idx")}
    save |= {k: np.full((n, o2.TOP), np.nan, np.float32) for k in ("f8_val", "seen_f8_val")}
    with get_context("fork").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(o2.run_task, tasks)):
            rows = r["rows"]
            save["f8_idx"][rows], save["f8_val"][rows] = r["f8"]
            if "seen_f8" in r:
                save["seen_f8_idx"][rows], save["seen_f8_val"][rows] = r["seen_f8"]
            if i % 10 == 0:
                print(f"task {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    old = np.load(o2.OUT / "scores.npz")
    save["owner_col"], save["cand_rank"] = owner_col, old["cand_rank"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "scores.npz", **save)
    (out / "result.json").write_text(json.dumps({"selected_arm": "f8", "style": l2.WEIGHTS}))
    shutil.copy(o2.OUT / "queries.parquet", out / "queries.parquet")
    report = {"seconds": round(time.time() - t0)}
    ref = q["set"].to_numpy() == "ref"
    colour = q["qcolor"].to_numpy()
    for label, z in (("f8", old), ("f8_new", save)):
        gap = z["f8_val"][:, 0] - z["f8_val"][:, 1]
        sgap = z["seen_f8_val"][:, 0] - z["seen_f8_val"][:, 1]
        hit = z["seen_f8_idx"][:, 0] == owner_col
        th = np.where(colour == 1, np.quantile(gap[ref & (colour == 1)], 0.99), np.quantile(gap[ref & (colour == 0)], 0.99))
        report[label] = {"seen_top1": float(hit[ref].mean()), "hit_q99": float(((sgap > th) & hit)[ref].mean()),
                         "seen_top10_recall": float((z["seen_f8_idx"] == owner_col[:, None]).any(1)[ref].mean())}
    (out / "score_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


def llr(args) -> None:
    out = out_dir(args.encoder)
    new = np.load(out / "scores.npz")
    top30 = np.load(o2.OUT / "scores_f8_top30.npz")
    old = np.load(o2.OUT / "scores.npz")
    if not np.array_equal(top30["f8_idx"][:, :10], old["f8_idx"]):
        raise ValueError("top-30 file does not extend the stage-0 top-10")
    L = np.load(LLR_DIR / "llr_g40.npz")
    X = np.load(LLR_DIR / "llr_g40_expanded.npz")
    if not np.array_equal(L["masked_rows"], X["qrows"]):
        raise ValueError("LLR row sets differ")
    res = {}
    for label, cand_new, cand_old, llr_old, rows in (
            ("masked", new["f8_idx"], np.hstack([top30["f8_idx"][:, :10], top30["f8_idx"][:, 10:30]]),
             np.hstack([L["masked_llr"], X["masked_extra_candidates"]]), L["masked_rows"]),
            ("seen", new["seen_f8_idx"], np.hstack([top30["seen_f8_idx"][:, :10], top30["seen_f8_idx"][:, 10:30]]),
             np.hstack([L["seen_llr"], X["seen_extra_candidates"]]), L["seen_rows"])):
        cn, co = cand_new[rows], cand_old[rows]
        match = cn[:, :, None] == co[:, None, :]                          # [r, 10 new, 30 old]
        found = match.any(2)
        val = np.where(found, (match * llr_old[:, None, :]).sum(2), np.nan)
        rowmin = np.nanmin(llr_old, axis=1)
        val = np.where(found, val, rowmin[:, None]).astype(np.float32)
        res[f"{label}_rows"], res[f"{label}_llr"] = rows, val
        res[f"{label}_found"] = found
        print(label, "new top-10 candidates with a looked-up LLR:", round(float(found.mean()), 4),
              "| top-1:", round(float(found[:, 0].mean()), 4), flush=True)
    np.savez(out / "llr_lookup.npz", **res)
    (out / "llr_report.json").write_text(json.dumps({k: float(res[k].mean()) for k in res if k.endswith("found")}))


def fuse(args) -> None:
    import scripts.strength.rank_opp_llr as rl
    out = out_dir(args.encoder)
    rl.o2.OUT = out                       # scores.npz / result.json / queries.parquet of f8'
    ns = argparse.Namespace(job=args.job, force=False, llr=str(out / "llr_lookup.npz"), alphas=args.alphas,
                            tag=args.tag, local_cells=None, experiment=None)
    rl.fuse(ns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("embed", embed), ("score", score), ("llr", llr), ("fuse", fuse)):
        p = sub.add_parser(name)
        p.add_argument("--encoder", default="A3", choices=list(RUNS))
        p.set_defaults(fn=fn)
        if name == "embed":
            p.add_argument("--shard", type=int, required=True)
            p.add_argument("--shards", type=int, required=True)
        if name == "score":
            p.add_argument("--shards", type=int, default=8)
            p.add_argument("--workers", type=int, default=80)
        if name == "fuse":
            p.add_argument("--job", default="local")
            p.add_argument("--alphas", default="0.35,0.45,0.55,0.65")
            p.add_argument("--tag", default="opp6-l10")
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
