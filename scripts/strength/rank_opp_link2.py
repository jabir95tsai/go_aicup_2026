"""Opponent identity link v2, stage 0: are competition-2 test opponents labelled training players?
(2026-11-23, user "A 和 B 都做，A 先"; registered in RESULTS.md before the run)

Premise: Yugen sets the handicap from the rank difference (HD0 = same rank, HD1 = one apart), so a recognised
opponent with a known label pins the test player's rank. The 11/13 check (rank_opp_link, old two-colour GE2E,
single-game top-1 0.08) could not tell whether opponents are in the pool. This reruns it with the detector of
rank_link2 (11/21): opening pp+slot log-likelihood o (D4) + lam * w * style, style = 0.25 cos_B + 0.75 mean(cos_A1,
cos_A2), against the same 8,007 training players (style centroid = up to 40 HD<=1 games, both colours; opening
gallery = every own-colour HD<=1 game except the reference games).

Single-game queries (all HD<=1):
  ref     own moves of the 1,182 encoder-unseen reference players (rank_link2 ref.parquet, 20 games each, outside
          every gallery): seen = own column kept, null = own column removed
  u_opp   opponent moves of every Valid-U game (strength folds seed 2027, fold 0); owner column removed
  s_opp   opponent moves of 20,000 Valid-S games (folds 1-4); owner column removed
  test    opponent moves of every competition-2 test game (Public + Private)
Saved per query and arm (opening only, style only, fused lam in LAMBDAS): top-10 candidates and scores.

    python -m scripts.strength.rank_opp_link2 plan
    python -m scripts.strength.rank_opp_link2 embed --shard I --shards N      # 1 GPU each
    python -m scripts.strength.rank_opp_link2 score --workers 8               # CPU
    python -m scripts.strength.rank_opp_link2 analyze                         # CPU, seconds
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
from scripts.strength.l10_check import score_questions

OUT = ROOT / "outputs" / "rank" / "opp_link2"
VEC = OUT / "vectors"
FOLDS = ROOT / "data" / "processed" / "cv" / "strength_folds_seed2027.parquet"
SEED = 2037
S_SAMPLE = 20000
TOP = 10
LAMBDAS = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
ARMS = ("o", "s", *[f"f{lam:g}" for lam in LAMBDAS])
TASK = 1200
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


# ---------------------------------------------------------------- plan
def valid_games(meta: pd.DataFrame) -> pd.DataFrame:
    import scripts.strength.rank_cvec_v2 as cv2
    ids = np.unique(np.concatenate([pd.read_parquet(cv2.OUT / f"ext_valid_hd{h}.parquet")["game_index"].to_numpy()
                                    for h in (0, 1)]))
    fold = pd.read_parquet(FOLDS).set_index("player_id")["fold"]
    ids = ids[meta["handicap"].to_numpy()[ids] <= 1]          # Valid-U keeps HD>=2 games; test games are HD<=1
    out = pd.DataFrame({"game_index": ids})
    out["player_id"] = meta["player_id"].astype(str).to_numpy()[ids]
    out["fold"] = fold.loc[out["player_id"]].to_numpy()
    return out


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    player = meta["player_id"].astype(str).to_numpy()
    rank_id = meta["rank_id"].to_numpy().astype(np.int64)
    hd = meta["handicap"].to_numpy()
    rng = np.random.default_rng(SEED)
    ref = pd.read_parquet(l2.OUT / "ref.parquet")
    valid = valid_games(meta)
    u = valid.loc[valid["fold"] == 0, "game_index"].to_numpy()
    s_all = valid.loc[valid["fold"] > 0, "game_index"].to_numpy()
    s = np.sort(rng.permutation(s_all)[:S_SAMPLE])
    gallery = pd.read_parquet(ol.OUT / "plan.parquet")
    gallery = set(gallery.loc[gallery["set"] == "gallery", "game_index"])
    if np.isin(ref["game_index"], list(gallery)).any():
        raise ValueError("reference game inside the style gallery")
    rows = [pd.DataFrame({"set": "ref", "part": "train", "game_index": ref["game_index"].to_numpy(), "store_row": -1,
                          "qcolor": ref["color"].to_numpy().astype(np.int8), "owner": ref["player_id"].astype(str),
                          "owner_rank": rank_id[ref["game_index"]], "hd": hd[ref["game_index"]]})]
    for name, games in (("u_opp", u), ("s_opp", s)):
        rows.append(pd.DataFrame({"set": name, "part": "train", "game_index": games, "store_row": -1,
                                  "qcolor": (1 - pcolor[games]).astype(np.int8), "owner": player[games],
                                  "owner_rank": rank_id[games], "hd": hd[games]}))
    for name in ol.PARTS:
        _, _, m = load_store(name, ol.STORE, mmap=False)
        own = (m["color"].astype(str).to_numpy() == "W").astype(np.int8)
        rows.append(pd.DataFrame({"set": "test", "part": name, "game_index": -1, "store_row": np.arange(len(m)),
                                  "qcolor": (1 - own).astype(np.int8), "owner": m["question_id"].astype(str),
                                  "owner_rank": -1, "hd": -1}))
    q = pd.concat(rows, ignore_index=True)
    q.to_parquet(OUT / "queries.parquet", index=False)
    print(json.dumps({"queries": q.groupby("set").size().to_dict(),
                      "colour": q.groupby(["set", "qcolor"]).size().unstack().to_dict(),
                      "u_players": int(valid.loc[valid["fold"] == 0, "player_id"].nunique()),
                      "s_pool_games": int(len(s_all))}, indent=1, default=int))


def plan_extra(_args) -> None:
    """stage-1 refinement (RESULTS 2026-11-23): every Valid-S HD<=1 game not already in s_opp."""
    meta = ol.load_train()
    q = pd.read_parquet(OUT / "queries.parquet")
    valid = valid_games(meta)
    s_all = valid.loc[valid["fold"] > 0, "game_index"].to_numpy()
    extra = np.setdiff1d(s_all, q.loc[q["set"] == "s_opp", "game_index"].to_numpy())
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    x = pd.DataFrame({"set": "s_opp", "part": "train", "game_index": extra, "store_row": -1,
                      "qcolor": (1 - pcolor[extra]).astype(np.int8),
                      "owner": meta["player_id"].astype(str).to_numpy()[extra],
                      "owner_rank": meta["rank_id"].to_numpy().astype(np.int64)[extra],
                      "hd": meta["handicap"].to_numpy()[extra]})
    x.to_parquet(OUT / "queries_extra.parquet", index=False)
    print(json.dumps({"extra": int(len(x)), "valid_s": int(len(s_all))}))


# ---------------------------------------------------------------- embed (opponent sides)
def embed(args) -> None:
    import torch
    VEC.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    q = pd.read_parquet(OUT / ("queries_extra.parquet" if args.extra else "queries.parquet"))
    tag = f"{args.shard}of{args.shards}"
    encoders = l2.load_encoders()
    t0 = time.time()
    train = q[q["set"].isin(["u_opp", "s_opp"])]
    jobs = [("train_extra" if args.extra else "train", data.CACHE_DIR, train["game_index"].to_numpy(),
             train["qcolor"].to_numpy(), meta["length"].to_numpy()[train["game_index"].to_numpy()], None)]
    for name in (() if args.extra else ol.PARTS):
        cache_dir, games, own, length, rows = l2.part_inputs(name, meta)
        jobs.append((name, cache_dir, games, (1 - own).astype(np.int8), length, rows))
    for name, cache_dir, games, colour, length, rows in jobs:
        idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
        np.save(VEC / f"{name}_rows_{tag}.npy", idx)
        for enc, (model, config, head, device) in encoders.items():
            path = VEC / f"{name}_{enc}_{tag}.npy"
            if path.exists():
                continue
            vec = l2.head_game_vectors(model, config, head, device, cache_dir, games[idx], colour[idx], length[idx])
            if not np.isfinite(vec).all():
                raise ValueError(f"{name} {enc}: non-finite vectors")
            np.save(path.with_suffix(".tmp.npy"), vec)
            path.with_suffix(".tmp.npy").replace(path)
            print(f"shard {tag} {name} {enc}: {len(idx)} games, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()
    print(f"shard {tag} done, {time.time() - t0:.0f}s", flush=True)


def load_opp_vectors(name: str, shards: int, n: int, rows) -> dict:
    out = {}
    for enc in l2.ENCODERS:
        vec = np.empty((n, 256), np.float64)
        seen = np.zeros(n, bool)
        for s in range(shards):
            idx = np.load(VEC / f"{name}_rows_{s}of{shards}.npy")
            vec[idx] = np.load(VEC / f"{name}_{enc}_{s}of{shards}.npy")
            seen[idx] = True
        if not seen.all():
            raise ValueError(f"{name} {enc}: missing shards")
        vec = ol.unit(vec)
        if rows is not None:
            store = np.empty_like(vec)
            store[rows] = vec
            vec = store
        out[enc] = vec
    return out


# ---------------------------------------------------------------- score
G: dict = {}


def topk(scores: np.ndarray, k: int = TOP):
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    val = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-val, axis=1)
    return np.take_along_axis(part, order, axis=1).astype(np.int32), np.take_along_axis(val, order, axis=1).astype(np.float32)


def run_task(task) -> dict:
    key, rows = task
    q = G["q"].iloc[rows]
    colour = int(q["qcolor"].iloc[0])
    o = score_questions(G["prefix"], G["hashes"], G["galleries"][colour], [np.array([p]) for p in G["pidx"][rows]],
                        G["population"][colour], colour, G["params"])
    vec = {enc: G["qvec"][enc][rows] for enc in l2.ENCODERS}
    s = l2.style(vec, G["cent"])
    owner = G["owner_col"][rows]
    has = owner >= 0
    out = {"rows": rows}
    if q["set"].iloc[0] == "ref":
        own_o, own_s = o[np.arange(len(rows)), owner], s[np.arange(len(rows)), owner]
        out["own_o"], out["own_s"] = own_o.astype(np.float32), own_s.astype(np.float32)
    arms = {"o": o, "s": s}
    for lam in LAMBDAS:
        arms[f"f{lam:g}"] = o + lam * G["w"] * s
    for arm, sc in arms.items():
        if q["set"].iloc[0] == "ref":
            out[f"seen_{arm}"] = topk(sc)
        sc = sc.copy()
        sc[np.flatnonzero(has), owner[has]] = -np.inf
        out[arm] = topk(sc)
    return out


def score(args) -> None:
    from multiprocessing import get_context
    meta = ol.load_train()
    main_q = pd.read_parquet(OUT / "queries.parquet")
    q = pd.read_parquet(OUT / "queries_extra.parquet") if args.extra else main_q
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)

    # style centroids exactly as rank_link2 analyze_training
    train_games = np.load(l2.OUT / "train_games.npy")
    train_vec = l2.load_vectors("train", 4, len(train_games), None)
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    sums = l2.sums_by({n: v[np.searchsorted(train_games, gal)] for n, v in train_vec.items()},
                      column.loc[player[gal]].to_numpy(), len(candidates))
    cent = {n: l2.unit_rows(x) for n, x in sums.items()}

    # query vectors in query order
    qvec = {enc: np.zeros((len(q), 256)) for enc in l2.ENCODERS}
    ref_rows = np.flatnonzero(q["set"].to_numpy() == "ref")
    pos = np.searchsorted(train_games, q["game_index"].to_numpy()[ref_rows])
    if not np.array_equal(train_games[pos], q["game_index"].to_numpy()[ref_rows]):
        raise ValueError("reference games missing from the rank_link2 vectors")
    for enc in l2.ENCODERS:
        qvec[enc][ref_rows] = train_vec[enc][pos]
    tr = np.flatnonzero(q["set"].isin(["u_opp", "s_opp"]).to_numpy())
    v = load_opp_vectors("train_extra" if args.extra else "train", args.shards, len(tr), None)
    for enc in l2.ENCODERS:
        qvec[enc][tr] = v[enc]
    del train_vec

    # opening prefixes: training games then each test store
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    blocks, offsets, size = [data.opening_prefix_d4(meta, PLIES)], {}, len(meta)
    for name in ol.PARTS:
        moves, colors, m = load_store(name, ol.STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, m, "d4", min_plies)
        if not ok.all():
            raise ValueError(f"{name}: games too short for the opening model")
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
        rows = np.flatnonzero((q["part"] == name).to_numpy())
        if args.extra:
            continue
        if not np.array_equal(q["store_row"].to_numpy()[rows], np.arange(len(m))):
            raise ValueError(f"{name}: query rows out of store order")
        _, _, _, _, store_rows = l2.part_inputs(name, meta)
        v = load_opp_vectors(name, args.shards, len(store_rows), store_rows)
        for enc in l2.ENCODERS:
            qvec[enc][rows] = v[enc]
    prefix = np.concatenate(blocks)
    pidx = np.where(q["part"].to_numpy() == "train", q["game_index"].to_numpy(),
                    q["store_row"].to_numpy() + q["part"].map(offsets).fillna(0).to_numpy().astype(np.int64))
    held = main_q.loc[main_q["set"] == "ref", "game_index"].to_numpy()   # same galleries as the main run
    galleries, population = {}, {}
    for colour in (0, 1):
        own = (pcolor == colour) & hd01
        keep = own.copy()
        keep[held] = False
        gl = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            gl[column[p]] = rows.to_numpy()
        galleries[colour], population[colour] = gl, np.flatnonzero(own)
    owner_col = np.full(len(q), -1, np.int64)
    trainish = q["set"].isin(["ref", "u_opp", "s_opp"]).to_numpy()
    owner_col[trainish] = column.loc[q["owner"].to_numpy()[trainish]].to_numpy()
    for enc in l2.ENCODERS:
        if not np.isclose(np.linalg.norm(qvec[enc], axis=1), 1.0).all():
            raise ValueError(f"{enc}: query vector missing")
    G.update(q=q, prefix=prefix, hashes=opening.prefix_hashes(prefix), galleries=galleries, population=population,
             params=params, pidx=pidx, qvec=qvec, cent=cent, owner_col=owner_col, w=float(np.load(l2.NEW_W)["w"]))
    tasks = []
    for (name, colour), g in q.groupby(["set", "qcolor"], sort=True):
        idx = g.index.to_numpy()
        tasks += [((name, colour, i), idx[i:i + TASK]) for i in range(0, len(idx), TASK)]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} tasks, {len(q)} queries, w={G['w']:.4f}", flush=True)
    t0 = time.time()
    results = []
    with get_context("fork").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
            results.append(r)
            if i % 10 == 0:
                print(f"task {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    n = len(q)
    save = {}
    for arm in ARMS:
        for pre in ("", "seen_"):
            save[f"{pre}{arm}_idx"] = np.full((n, TOP), -1, np.int32)
            save[f"{pre}{arm}_val"] = np.full((n, TOP), np.nan, np.float32)
    save["own_o"], save["own_s"] = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    for r in results:
        rows = r["rows"]
        for key, value in r.items():
            if key == "rows":
                continue
            if isinstance(value, tuple):
                save[f"{key}_idx"][rows], save[f"{key}_val"][rows] = value
            else:
                save[key][rows] = value
    save["owner_col"] = owner_col
    save["cand_rank"] = pd.Series(meta["rank_id"].to_numpy().astype(np.int64), index=player).groupby(level=0).first() \
        .loc[candidates].to_numpy()
    name = "scores_extra.npz" if args.extra else "scores.npz"
    np.savez(OUT / (name if not args.limit else "scores_limit.npz"), **save)
    print(f"saved {len(results)} tasks, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- analyze
def poisson_sf(x: int, mu: float) -> float:
    from scipy.stats import poisson
    return float(poisson.sf(x - 1, mu))


def analyze(args) -> dict:
    q = pd.read_parquet(OUT / "queries.parquet")
    z = np.load(OUT / args.scores)
    cand_rank, owner_col = z["cand_rank"], z["owner_col"]
    sets = q["set"].to_numpy()
    colour = q["qcolor"].to_numpy()
    ref = sets == "ref"
    players = np.array(sorted(q.loc[ref, "owner"].unique()))
    half_a = set(players[np.random.default_rng(SEED).permutation(len(players))[:len(players) // 2]])
    in_a = ref & q["owner"].isin(half_a).to_numpy()
    in_b = ref & ~in_a

    def stats(arm: str):
        val, idx = z[f"{arm}_val"], z[f"{arm}_idx"]
        sval, sidx = z[f"seen_{arm}_val"], z[f"seen_{arm}_idx"]
        gap = val[:, 0] - val[:, 1]
        sgap = sval[:, 0] - sval[:, 1]
        shit = sidx[:, 0] == owner_col
        return gap, idx[:, 0], sgap, shit

    def thresholds(gap, mask, qs):
        return {c: {qq: float(np.quantile(gap[mask & (colour == c)], qq)) for qq in qs} for c in (0, 1)}

    def above(gap, th, qq):
        t = np.where(colour == 1, th[1][qq], th[0][qq])
        return gap > t

    result = {"queries": q.groupby("set").size().to_dict(), "arms": {}}
    best = None
    for arm in ARMS:
        if not np.isfinite(z[f"{arm}_val"][:, 1]).all():
            continue
        gap, top1, sgap, shit = stats(arm)
        th_a = thresholds(gap, in_a, (0.99,))
        rec_a = float((above(sgap, th_a, 0.99) & shit)[in_a].mean())
        th_b = thresholds(gap, in_b, (0.99, 0.999))
        entry = {"seen_top1": float(shit[ref].mean()), "seen_top1_by_colour": {c: float(shit[ref & (colour == c)].mean())
                                                                            for c in (0, 1)},
                 "half_a_hit_at_q99": rec_a,
                 "half_b_hit_at_q99_own_threshold": float((above(sgap, th_b, 0.99) & shit)[in_b].mean())}
        result["arms"][arm] = entry
        if best is None or rec_a > best[1]:
            best = (arm, rec_a)
    arm = best[0]
    result["selected_arm"] = arm
    gap, top1, sgap, shit = stats(arm)
    th = thresholds(gap, ref, (0.95, 0.99, 0.999))
    result["thresholds"] = th
    mixture = {}
    for qq in (0.95, 0.99, 0.999):
        r_null = float(above(gap, th, qq)[ref].mean())
        r_seen = float(above(sgap, th, qq)[ref].mean())
        prec = float(shit[ref & above(sgap, th, qq)].mean())
        entry = {"null_rate": r_null, "seen_rate": r_seen, "seen_precision": prec}
        for name in ("u_opp", "s_opp", "test"):
            for sub, m in ((name, sets == name),
                           *(((f"{name}_{p}", (sets == name) & (q["part"] == p).to_numpy()) for p in ol.PARTS)
                             if name == "test" else ())):
                r = float(above(gap, th, qq)[m].mean())
                n = int(m.sum())
                entry[sub] = {"n": n, "rate": r, "count": int(above(gap, th, qq)[m].sum()),
                              "expected_if_none": round((1 - qq) * n, 1),
                              "poisson_p_if_none": poisson_sf(int(above(gap, th, qq)[m].sum()), (1 - qq) * n),
                              "pi_hat": (r - r_null) / (r_seen - r_null) if r_seen > r_null else None}
        mixture[str(qq)] = entry
    result["mixture"] = mixture

    # lever: matched player's rank against the owner's rank
    lever = {}
    opp = np.isin(sets, ["u_opp", "s_opp"])
    hd = q["hd"].to_numpy()
    owner_rank = q["owner_rank"].to_numpy()
    owner_black = colour == 1          # the queried side is white -> owner played black
    diff_opp = cand_rank[top1] - owner_rank
    diff_null = cand_rank[top1] - owner_rank          # ref rows: top1 of the masked scores = nearest other player
    for qq in (None, 0.99, 0.999):
        keep = np.ones(len(q), bool) if qq is None else above(gap, th, qq)
        label = "all" if qq is None else f"q{qq}"
        for h in (0, 1):
            for ob in (True, False):
                m = opp & keep & (hd == h) & (owner_black == ob)
                if m.sum() >= 5:
                    d = diff_opp[m]
                    lever[f"opp_{label}_hd{h}_{'ownerB' if ob else 'ownerW'}"] = {
                        "n": int(m.sum()), "same": float((d == 0).mean()), "within1": float((np.abs(d) <= 1).mean()),
                        "mean": float(d.mean()), "hist": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}}
        m = ref & keep
        if m.sum() >= 5:
            d = diff_null[m]
            lever[f"null_{label}"] = {"n": int(m.sum()), "same": float((d == 0).mean()),
                                      "within1": float((np.abs(d) <= 1).mean())}
    shuf = cand_rank[np.random.default_rng(SEED).permutation(top1[opp])] - owner_rank[opp]
    lever["shuffled"] = {"same": float((shuf == 0).mean()), "within1": float((np.abs(shuf) <= 1).mean())}
    result["lever"] = lever

    m999 = mixture["0.999"]
    m99 = mixture["0.99"]
    hd0 = opp & above(gap, th, 0.999) & (hd == 0)
    g2 = {"n": int(hd0.sum()), "within1": float((np.abs(diff_opp[hd0]) <= 1).mean()) if hd0.any() else None}
    pi_test, pi_u = m99["test"]["pi_hat"], m99["u_opp"]["pi_hat"]
    g1 = bool(pi_test is not None and pi_u is not None and pi_test >= 0.2 and pi_u >= 0.2
              and m999["test"]["poisson_p_if_none"] < 1e-4)
    result["gate"] = {"G1_in_pool": {"pi_test_q99": pi_test, "pi_u_q99": pi_u,
                                     "test_poisson_p_q999": m999["test"]["poisson_p_if_none"], "pass": g1},
                      "G2_lever_hd0_q999": {**g2, "pass": bool(g2["n"] >= 100 and (g2["within1"] or 0) >= 0.8)}}
    result["gate"]["pass"] = bool(g1 and result["gate"]["G2_lever_hd0_q999"]["pass"])
    (OUT / args.out).write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps(result, indent=1, default=int))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    sub.add_parser("plan-extra").set_defaults(fn=plan_extra)
    p = sub.add_parser("embed")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--extra", action="store_true", help="the remaining Valid-S games (queries_extra.parquet)")
    p.set_defaults(fn=embed)
    p = sub.add_parser("score")
    p.add_argument("--extra", action="store_true")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N tasks only (timing)")
    p.set_defaults(fn=score)
    p = sub.add_parser("analyze")
    p.add_argument("--scores", default="scores.npz")
    p.add_argument("--out", default="result.json")
    p.set_defaults(fn=analyze)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
