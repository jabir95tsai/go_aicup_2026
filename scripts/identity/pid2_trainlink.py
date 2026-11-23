"""Competition 1, added set: are the 1,200 added-set candidates labelled training players? Stage 0.
(2026-11-23, user "A 和 B 都做，A 先"; registered in RESULTS.md before the run)

If a candidate (10 games, both colours) is one of the 8,007 training players under another ID, linking adds about
125 reference games. The 11/15 check (pid2_overlap, v4 style only) left "10-20 % overlap, +-6 points, cannot
confirm or exclude". Here the detector of rank_link2 (11/21) is used on whole candidates:
  f = o + lam * w * s,  o = opening pp+slot log-likelihood summed over the candidate's black and white games
  (each colour against that colour's training galleries), s = style score of the candidate's 10-game vector.
Reference ("seen" / null): the 1,182 encoder-unseen training players; each gets b black + (10 - b) white HD<=1
games outside the rank_link2 style gallery, b drawn from the added-pool black-count distribution (seed 2038).
Seen keeps the own column; null removes it. Query games are removed from the opening galleries.

    python -m scripts.identity.pid2_trainlink plan
    python -m scripts.identity.pid2_trainlink embed            # 1 GPU, reference games
    python -m scripts.identity.pid2_trainlink score --workers 8
    python -m scripts.identity.pid2_trainlink analyze
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
from scripts.strength.l10_check import score_questions

OUT = ROOT / "outputs" / "pid" / "trainlink"
SEED = 2038
GAMES = 10
POOLS = ("pool2_public", "pool2_private")
ARMS = o2.ARMS
LAMBDAS = o2.LAMBDAS


def pool_meta(name: str) -> pd.DataFrame:
    _, _, m = load_store(name, l2.PID_STORE, mmap=False)
    return m


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    free = meta["handicap"].to_numpy() <= 1
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    free[work.loc[work["set"] == "gallery", "game_index"].to_numpy()] = False
    free &= np.isin(player, sorted(l2.unseen_players()))
    blacks = np.concatenate([(pool_meta(p)["color"].astype(str) == "B").groupby(pool_meta(p)["player_id"]).sum()
                             .to_numpy() for p in POOLS])
    rng = np.random.default_rng(SEED)
    rows = []
    by_player = pd.Series(np.flatnonzero(free)).groupby(player[free]).apply(np.asarray)
    for pid in sorted(by_player.index):
        games = by_player[pid]
        b = int(rng.choice(blacks))
        black = rng.permutation(games[pcolor[games] == 0])[:b]
        white = rng.permutation(games[pcolor[games] == 1])[:GAMES - b]
        if len(black) < b or len(white) < GAMES - b:
            continue
        pick = np.concatenate([black, white])
        rows.append(pd.DataFrame({"player_id": pid, "game_index": pick, "color": pcolor[pick]}))
    ref = pd.concat(rows, ignore_index=True)
    ref.to_parquet(OUT / "ref.parquet", index=False)
    print(json.dumps({"reference_players": int(ref["player_id"].nunique()), "games": int(len(ref)),
                      "black_share": float((ref["color"] == 0).mean()),
                      "pool_black_share": float(blacks.mean() / GAMES)}, indent=1))


def embed(_args) -> None:
    meta = ol.load_train()
    ref = pd.read_parquet(OUT / "ref.parquet")
    games = ref["game_index"].to_numpy()
    encoders = l2.load_encoders()
    t0 = time.time()
    for enc, (model, config, head, device) in encoders.items():
        path = OUT / f"ref_{enc}.npy"
        if path.exists():
            continue
        vec = l2.head_game_vectors(model, config, head, device, data.CACHE_DIR, games,
                                   ref["color"].to_numpy().astype(np.int8), meta["length"].to_numpy()[games])
        if not np.isfinite(vec).all():
            raise ValueError(f"{enc}: non-finite vectors")
        np.save(path, vec)
        print(f"{enc}: {len(games)} games, {time.time() - t0:.0f}s", flush=True)


G: dict = {}


def run_task(task) -> dict:
    rows = task
    groups = [G["groups"][i] for i in rows]
    o = np.zeros((len(rows), G["n_cand"]))
    for colour in (0, 1):
        sel = [(j, g[G["gcolour"][g] == colour]) for j, g in enumerate(groups)]
        sel = [(j, g) for j, g in sel if len(g)]
        if not sel:
            continue
        sc = score_questions(G["prefix"], G["hashes"], G["galleries"][colour], [G["pidx"][g] for _, g in sel],
                             G["population"][colour], colour, G["params"])
        o[[j for j, _ in sel]] += sc
    qv = l2.question_vectors(G["gvec"], groups)
    s = l2.style(qv, G["cent"])
    truth = G["truth"][rows]
    is_ref = truth >= 0
    out = {"rows": rows}
    arms = {"o": o, "s": s}
    for lam in LAMBDAS:
        arms[f"f{lam:g}"] = o + lam * G["w"] * s
    for arm, sc in arms.items():
        out[f"seen_{arm}"] = o2.topk(sc)
        sc = sc.copy()
        sc[np.flatnonzero(is_ref), truth[is_ref]] = -np.inf
        out[arm] = o2.topk(sc)
    return out


def score(args) -> None:
    from multiprocessing import get_context
    meta = ol.load_train()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    candidates = np.array(sorted(set(player)))
    column = pd.Series(np.arange(len(candidates)), index=candidates)
    train_games = np.load(l2.OUT / "train_games.npy")
    train_vec = l2.load_vectors("train", 4, len(train_games), None)
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    sums = l2.sums_by({n: v[np.searchsorted(train_games, gal)] for n, v in train_vec.items()},
                      column.loc[player[gal]].to_numpy(), len(candidates))
    cent = {n: l2.unit_rows(x) for n, x in sums.items()}
    del train_vec

    ref = pd.read_parquet(OUT / "ref.parquet")
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    blocks, size = [data.opening_prefix_d4(meta, PLIES)], len(meta)
    # flat game table: reference games then each pool's store rows
    gvec = {enc: [ol.unit(np.load(OUT / f"ref_{enc}.npy").astype(np.float64))] for enc in l2.ENCODERS}
    pidx, gcolour, usable = [ref["game_index"].to_numpy()], [ref["color"].to_numpy()], [np.ones(len(ref), bool)]
    qinfo = [pd.DataFrame({"set": "ref", "query": ref["player_id"].astype(str).unique()})]
    groups, start = [], 0
    for pid, g in ref.groupby("player_id", sort=False):
        groups.append(g.index.to_numpy())
    start = len(ref)
    for name in POOLS:
        moves, colors, m = load_store(name, l2.PID_STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, m, "d4", min_plies)
        blocks.append(pre)
        pidx.append(size + np.arange(len(m)))
        size += len(m)
        gcolour.append((m["color"].astype(str).to_numpy() == "W").astype(np.int64))
        usable.append(ok)
        _, _, _, _, rows = l2.part_inputs(name, meta)
        v = l2.load_vectors(name, 4, len(rows), rows)
        for enc in l2.ENCODERS:
            gvec[enc].append(v[enc])
        ids = m["player_id"].astype(str).to_numpy()
        names = list(dict.fromkeys(ids))
        idx = pd.Series(np.arange(len(m))).groupby(ids).apply(np.asarray)
        groups += [start + idx[p] for p in names]
        qinfo.append(pd.DataFrame({"set": name, "query": names}))
        start += len(m)
    prefix = np.concatenate(blocks)
    gvec = {enc: np.concatenate(v) for enc, v in gvec.items()}
    pidx, gcolour, usable = np.concatenate(pidx), np.concatenate(gcolour), np.concatenate(usable)
    qinfo = pd.concat(qinfo, ignore_index=True)
    truth = np.full(len(qinfo), -1, np.int64)
    r = (qinfo["set"] == "ref").to_numpy()
    truth[r] = column.loc[qinfo.loc[r, "query"]].to_numpy()
    held = ref["game_index"].to_numpy()
    galleries, population = {}, {}
    for colour in (0, 1):
        own = (pcolor == colour) & hd01
        keep = own.copy()
        keep[held] = False
        gl = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            gl[column[p]] = rows.to_numpy()
        galleries[colour], population[colour] = gl, np.flatnonzero(own)
    # unusable (too short) games drop out of the opening term only
    gcolour_open = np.where(usable, gcolour, -1)
    G.update(prefix=prefix, hashes=opening.prefix_hashes(prefix), galleries=galleries, population=population,
             params=params, pidx=pidx, gcolour=gcolour_open, gvec=gvec, cent=cent, groups=groups, truth=truth,
             n_cand=len(candidates), w=float(np.load(l2.NEW_W)["w"]))
    print(f"queries {qinfo['set'].value_counts().to_dict()}, unusable games {int((~usable).sum())}", flush=True)
    tasks = [np.arange(i, min(i + 150, len(qinfo))) for i in range(0, len(qinfo), 150)]
    t0 = time.time()
    results = []
    with get_context("fork").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_task, tasks)):
            results.append(res)
            print(f"task {i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    n = len(qinfo)
    save = {f"{p}{a}_{x}": (np.full((n, o2.TOP), -1, np.int32) if x == "idx" else np.full((n, o2.TOP), np.nan, np.float32))
            for p in ("", "seen_") for a in ARMS for x in ("idx", "val")}
    for res in results:
        for key, value in res.items():
            if key != "rows":
                save[f"{key}_idx"][res["rows"]], save[f"{key}_val"][res["rows"]] = value
    save["truth"] = truth
    qinfo.to_parquet(OUT / "queries.parquet", index=False)
    np.savez(OUT / "scores.npz", candidates=candidates, **save)
    print(f"saved, {time.time() - t0:.0f}s", flush=True)


def analyze(args) -> dict:
    q = pd.read_parquet(OUT / "queries.parquet")
    z = np.load(OUT / "scores.npz")
    truth = z["truth"]
    ref = (q["set"] == "ref").to_numpy()
    players = q.loc[ref, "query"].to_numpy()
    half_a = set(players[np.random.default_rng(SEED).permutation(len(players))[:len(players) // 2]])
    in_a = ref & q["query"].isin(half_a).to_numpy()
    in_b = ref & ~in_a

    def stats(arm):
        val, sval, sidx = z[f"{arm}_val"], z[f"seen_{arm}_val"], z[f"seen_{arm}_idx"]
        return val[:, 0] - val[:, 1], sval[:, 0] - sval[:, 1], sidx[:, 0] == truth

    result = {"queries": q["set"].value_counts().to_dict(), "arms": {}}
    best = None
    for arm in ARMS:
        gap, sgap, hit = stats(arm)
        t_a = float(np.quantile(gap[in_a], 0.99))
        rec = float(((sgap > t_a) & hit)[in_a].mean())
        t_b = float(np.quantile(gap[in_b], 0.99))
        result["arms"][arm] = {"seen_top1": float(hit[ref].mean()), "half_a_hit_at_q99": rec,
                               "half_b_hit_at_q99": float(((sgap > t_b) & hit)[in_b].mean())}
        if best is None or rec > best[1]:
            best = (arm, rec)
    arm = best[0]
    result["selected_arm"] = arm
    gap, sgap, hit = stats(arm)
    mixture = {}
    for qq in (0.95, 0.99, 0.999, 1.0):
        t = float(np.quantile(gap[ref], qq)) if qq < 1 else float(gap[ref].max())
        r_null, r_seen = float((gap[ref] > t).mean()), float((sgap[ref] > t).mean())
        entry = {"threshold": t, "null_rate": r_null, "seen_rate": r_seen,
                 "seen_precision": float(hit[ref & (sgap > t)].mean()) if (ref & (sgap > t)).any() else None}
        n_ref = int(ref.sum())
        for name in POOLS:
            m = (q["set"] == name).to_numpy()
            count = int((gap[m] > t).sum())
            mu = max((1 - qq), 1 / n_ref) * int(m.sum())
            entry[name] = {"n": int(m.sum()), "count": count, "rate": count / int(m.sum()),
                           "expected_if_none": round(mu, 2), "poisson_p_if_none": o2.poisson_sf(count, mu),
                           "pi_hat": (count / int(m.sum()) - r_null) / (r_seen - r_null) if r_seen > r_null else None}
        mixture["max" if qq == 1 else str(qq)] = entry
    result["mixture"] = mixture
    t999 = mixture["0.999"]["threshold"]
    links = []
    for name in POOLS:
        m = np.flatnonzero((q["set"] == name).to_numpy() & (gap > t999))
        for i in m:
            links.append({"pool": name, "candidate": q["query"].iloc[i], "player": str(z["candidates"][z[f"{arm}_idx"][i, 0]]),
                          "gap": float(gap[i])})
    pd.DataFrame(links).to_csv(OUT / "links_q999.csv", index=False)
    result["links_q999"] = len(links)
    m99 = mixture["0.99"]
    pis = [m99[p]["pi_hat"] for p in POOLS]
    pooled_count = sum(mixture["0.999"][p]["count"] for p in POOLS)
    pooled_mu = sum(mixture["0.999"][p]["expected_if_none"] for p in POOLS)
    result["gate"] = {"pi_q99": pis, "pooled_q999_count": pooled_count, "pooled_q999_expected": pooled_mu,
                      "pooled_q999_poisson_p": o2.poisson_sf(pooled_count, pooled_mu),
                      "pass": bool(all(p is not None and p >= 0.10 for p in pis)
                                   and o2.poisson_sf(pooled_count, pooled_mu) < 1e-4)}
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps(result, indent=1, default=int))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    sub.add_parser("embed").set_defaults(fn=embed)
    p = sub.add_parser("score")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=score)
    sub.add_parser("analyze").set_defaults(fn=analyze)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
