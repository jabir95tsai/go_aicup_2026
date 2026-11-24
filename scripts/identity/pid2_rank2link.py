"""Are competition-1 added-set candidates competition-2 test players? (2026-11-23, user "可以"; RESULTS.md, registered
before the run)

11/21 rank_link2 (b) used the added pools (10 games) as the gallery: simulated recall 0.003, no power. Here the other
direction: the 800 competition-2 test questions (5-20 same-colour games) are the gallery, the 1,200 added-set
candidates are the queries, one query per (candidate, colour) with the candidate's games of that colour. Score per
colour c against the colour-c questions only: f = o + lam * w * s (o = opening pp+slot log-likelihood, D4;
s = 0.25 cos_B + 0.75 mean(cos_A1, cos_A2) of the query vector against the question vector), statistic = top-1 minus
top-2. Reference: the 1,182 encoder-unseen players of rank_link2 (20 games, one colour): gallery = first g games
(g cycling 5 / 10 / 15), query = games 15-19; seen keeps the own entry, null removes it. Arm (o, s, lam in LAMS)
chosen on reference half A (seen hit above the half-A null q99). Rates above the null q99 / q99.9 / max, pi_hat,
Poisson excess. Gate: pooled excess above q99.9 with Poisson p < 1e-4 and pi_hat(q99) >= 0.05 -> write a link plan.

    python -m scripts.identity.pid2_rank2link
"""
from __future__ import annotations

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

OUT = ROOT / "outputs" / "pid" / "rank2link"
POOLS = ("pool2_public", "pool2_private")
LAMS = (2.0, 4.0, 8.0, 16.0)
SEED = 2041


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = ol.load_train()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    w = float(np.load(l2.NEW_W)["w"])

    # flat game table: prefix rows, own colour, unit vectors per encoder
    blocks, size = [data.opening_prefix_d4(meta, PLIES)], len(meta)
    vec = {e: [] for e in l2.ENCODERS}
    rows = {}                                   # name -> (prefix rows, colour, vector rows)
    nvec = 0
    train_games = np.load(l2.OUT / "train_games.npy")
    tv = l2.load_vectors("train", 4, len(train_games), None)
    for e in l2.ENCODERS:
        vec[e].append(tv[e])
    rows["train"] = (train_games, pcolor[train_games], np.arange(len(train_games)))
    nvec += len(train_games)
    for name, store in [(p, ol.STORE) for p in ol.PARTS] + [(p, l2.PID_STORE) for p in POOLS]:
        moves, colors, m = load_store(name, store, mmap=False)
        pre, ok = store_prefix(moves, colors, m, "d4", min_plies)
        blocks.append(pre)
        _, _, _, _, srows = l2.part_inputs(name, meta)
        v = l2.load_vectors(name, 4, len(srows), srows)
        for e in l2.ENCODERS:
            vec[e].append(v[e])
        col = (m["color"].astype(str).to_numpy() == "W").astype(np.int64)
        rows[name] = (size + np.arange(len(m)), col, nvec + np.arange(len(m)), m, ok)
        size += len(m)
        nvec += len(m)
    prefix = np.concatenate(blocks)
    hashes = opening.prefix_hashes(prefix)
    vec = {e: np.concatenate(v) for e, v in vec.items()}
    print(f"tables built {time.time() - t0:.0f}s", flush=True)

    def qvec(vrows):
        return {e: l2.unit_rows(vec[e][vrows].sum(0, keepdims=True))[0] for e in l2.ENCODERS}

    result = {"arms": {}}
    # reference
    ref = pd.read_parquet(l2.OUT / "ref.parquet").sort_values(["player_id", "order"])
    pos_train = pd.Series(np.arange(len(train_games)), index=train_games)
    ref_entries = []
    for i, (pid, g) in enumerate(ref.groupby("player_id", sort=True)):
        games = g["game_index"].to_numpy()
        gsize = (5, 10, 15)[i % 3]
        ref_entries.append({"player": pid, "colour": int(g["color"].iloc[0]),
                            "gal_prefix": games[:gsize], "gal_vec": pos_train.loc[games[:gsize]].to_numpy(),
                            "q_prefix": games[15:20], "q_vec": pos_train.loc[games[15:20]].to_numpy()})
    population = {c: np.flatnonzero((pcolor == c) & hd01) for c in (0, 1)}

    def scores(galleries, gal_vecs, queries, q_vecs, colour):
        o = score_questions(prefix, hashes, galleries, queries, population[colour], colour, params)
        gv = [qvec(v) for v in gal_vecs]
        qv = [qvec(v) for v in q_vecs]
        s = sum(l2.WEIGHTS[e] * (np.stack([x[e] for x in qv]) @ np.stack([x[e] for x in gv]).T) for e in l2.ENCODERS)
        return o, s

    ref_scores = {}
    for c in (0, 1):
        ent = [x for x in ref_entries if x["colour"] == c]
        o, s = scores([x["gal_prefix"] for x in ent], [x["gal_vec"] for x in ent],
                      [x["q_prefix"] for x in ent], [x["q_vec"] for x in ent], c)
        ref_scores[c] = (o, s, np.array([x["player"] for x in ent]))
        print(f"reference colour {c}: {len(ent)} players, {time.time() - t0:.0f}s", flush=True)

    # competition-2 questions as gallery, candidates' colour-c games as queries
    gal = {0: [], 1: []}
    for name in ol.PARTS:
        prow, col, vrow, m, ok = rows[name]
        for qid, g in m.groupby("question_id", sort=True):
            idx = g.index.to_numpy()
            c = int(col[idx[0]])
            gal[c].append({"qid": qid, "prefix": prow[idx], "vec": vrow[idx]})
    cand_scores = {}
    for c in (0, 1):
        queries = []
        for name in POOLS:
            prow, col, vrow, m, ok = rows[name]
            for pid, g in m.groupby("player_id", sort=True):
                idx = g.index.to_numpy()
                idx = idx[(col[idx] == c) & ok[idx]]
                if len(idx):
                    queries.append({"pool": name, "player": pid, "n": len(idx), "prefix": prow[idx], "vec": vrow[idx]})
        o, s = scores([x["prefix"] for x in gal[c]], [x["vec"] for x in gal[c]],
                      [x["prefix"] for x in queries], [x["vec"] for x in queries], c)
        cand_scores[c] = (o, s, queries)
        print(f"candidates colour {c}: {len(queries)} queries vs {len(gal[c])} questions, {time.time() - t0:.0f}s",
              flush=True)

    def arm_matrix(o, s, arm):
        return o if arm == "o" else (s if arm == "s" else o + float(arm[1:]) * w * s)

    def top2(x):
        part = np.sort(x, axis=1)[:, -2:]
        return x.argmax(1), part[:, 1] - part[:, 0]

    rng = np.random.default_rng(SEED)
    players = np.array(sorted({x["player"] for x in ref_entries}))
    half_a = set(players[rng.permutation(len(players))[:len(players) // 2]])
    arms = ["o", "s"] + [f"f{lam:g}" for lam in LAMS]

    def ref_stats(arm):
        out = {}
        for c in (0, 1):
            o, s, pl = ref_scores[c]
            x = arm_matrix(o, s, arm)
            arg, sgap = top2(x)
            hit = arg == np.arange(len(pl))
            xm = x.copy()
            xm[np.arange(len(pl)), np.arange(len(pl))] = -np.inf
            _, ngap = top2(xm)
            out[c] = (sgap, hit, ngap, np.isin(pl, list(half_a)))
        return out

    best = None
    for arm in arms:
        st = ref_stats(arm)
        th_a = {c: np.quantile(st[c][2][st[c][3]], 0.99) for c in (0, 1)}
        hit_a = np.mean(np.concatenate([((st[c][0] > th_a[c]) & st[c][1])[st[c][3]] for c in (0, 1)]))
        th_b = {c: np.quantile(st[c][2][~st[c][3]], 0.99) for c in (0, 1)}
        hit_b = np.mean(np.concatenate([((st[c][0] > th_b[c]) & st[c][1])[~st[c][3]] for c in (0, 1)]))
        top1 = np.mean(np.concatenate([st[c][1] for c in (0, 1)]))
        result["arms"][arm] = {"seen_top1": float(top1), "half_a_hit_q99": float(hit_a), "half_b_hit_q99": float(hit_b)}
        if best is None or hit_a > best[1]:
            best = (arm, hit_a)
    arm = best[0]
    result["selected_arm"] = arm
    st = ref_stats(arm)
    n_ref = sum(len(st[c][2]) for c in (0, 1))
    mixture = {}
    links = []
    for label, qq in (("q95", 0.95), ("q99", 0.99), ("q999", 0.999), ("max", 1.0)):
        th = {c: (np.quantile(st[c][2], qq) if qq < 1 else st[c][2].max()) for c in (0, 1)}
        r_null = float(np.mean(np.concatenate([st[c][2] > th[c] for c in (0, 1)])))
        r_seen = float(np.mean(np.concatenate([st[c][0] > th[c] for c in (0, 1)])))
        entry = {"thresholds": {str(c): float(th[c]) for c in (0, 1)}, "null_rate": r_null, "seen_rate": r_seen}
        for pool in POOLS:
            cnt, n = 0, 0
            for c in (0, 1):
                o, s, queries = cand_scores[c]
                arg, gap = top2(arm_matrix(o, s, arm))
                m = np.array([x["pool"] == pool for x in queries])
                above = (gap > th[c]) & m
                cnt += int(above.sum())
                n += int(m.sum())
                if label == "q999":
                    for i in np.flatnonzero(above):
                        links.append({"pool": pool, "candidate": queries[i]["player"], "colour": c,
                                      "n_games": queries[i]["n"], "question": gal[c][arg[i]]["qid"],
                                      "gap": float(gap[i])})
            mu = max(1 - qq, 1 / n_ref) * n
            entry[pool] = {"n": n, "count": cnt, "expected_if_none": round(mu, 2),
                           "poisson_p_if_none": o2.poisson_sf(cnt, mu),
                           "pi_hat": (cnt / n - r_null) / (r_seen - r_null) if r_seen > r_null else None}
        mixture[label] = entry
    result["mixture"] = mixture
    pd.DataFrame(links).to_csv(OUT / "links_q999.csv", index=False)
    pooled = sum(mixture["q999"][p]["count"] for p in POOLS)
    pooled_mu = sum(mixture["q999"][p]["expected_if_none"] for p in POOLS)
    pis = [mixture["q99"][p]["pi_hat"] for p in POOLS]
    result["gate"] = {"pooled_q999": pooled, "expected": pooled_mu, "poisson_p": o2.poisson_sf(pooled, pooled_mu),
                      "pi_q99": pis,
                      "pass": bool(o2.poisson_sf(pooled, pooled_mu) < 1e-4 and all(p is not None and p >= 0.05 for p in pis))}
    # robustness (added after the first result showed hub questions; RESULTS 2026-11-23)
    save = {"w": w}
    for c in (0, 1):
        o, s_, queries = cand_scores[c]
        save[f"cand_o{c}"], save[f"cand_s{c}"] = o, s_
        save[f"cand_pool{c}"] = np.array([x["pool"] for x in queries])
        save[f"cand_player{c}"] = np.array([x["player"] for x in queries])
        save[f"gal_qid{c}"] = np.array([x["qid"] for x in gal[c]])
        save[f"gal_k{c}"] = np.array([len(x["prefix"]) for x in gal[c]])
        ro, rs, rpl = ref_scores[c]
        save[f"ref_o{c}"], save[f"ref_s{c}"] = ro, rs
        save[f"ref_g{c}"] = np.array([len(x["gal_prefix"]) for x in ref_entries if x["colour"] == c])
    np.savez(OUT / "matrices.npz", **save)
    result["seconds"] = round(time.time() - t0, 1)
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps(result, indent=1, default=int))


if __name__ == "__main__":
    main()
