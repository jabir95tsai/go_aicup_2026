"""Opponent identity link v2: rerank the top-10 opponent candidates with the competition-1 player-conditional
move likelihood (2026-11-23, user "繼續訓練 我們還有時間"; RESULTS.md, registered before results)

The base (845986) links each opponent side to the 8,007 training players with f8 = opening + 8 w style; single-game
seen top-1 is only 0.33. Competition 1 gained about +0.03 from LLR: frozen HumanSL + C v2 policy at four rank
conditions, a 27-dim log-linear residual theta per player (MAP, prior N(theta_pop, 3^2 I)), score = sum over the
player's moves of log p_theta - log p_theta_pop. Here:
  gallery   every training player: GAL_GAMES own-side HD<=1 games from the rank_link2 style gallery
  queries   the stage-0 rows: ref (first REF_GAMES per reference player), u_opp, s_opp (20,000), test
  LLR[q, i] for the top-10 f8 candidates of each query (masked as in stage 0; ref also with the own column kept)
  f' = f8 + mu * LLR on those 10, mu chosen on reference half A (seen hit at the null q99 of f'), half B reported
Fusion as the base (soft top-10 on f', DELTA kernel on s_opp, bins from the ref null of f'); cells SOFT tau x alpha
and DELTA top-1 x alpha; selection on E1 against the base cell (845986 settings on f8); E2 opened for the chosen cell;
export `<job>-opp4-l10.csv` as an LB probe if both are > 0.

    python -m scripts.strength.rank_opp_llr plan
    python -m scripts.strength.rank_opp_llr policy --shard I --shards 8      # 1 GPU each
    python -m scripts.strength.rank_opp_llr llr                              # 1 GPU
    python -m scripts.strength.rank_opp_llr fuse --job ID                    # CPU
"""
from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT

import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_opp_link2 as o2

OUT = ROOT / "outputs" / "rank" / "opp_llr"
GAL_GAMES, REF_GAMES, POP_PLAYERS, POP_GAMES = 15, 10, 600, 10
SEED = 2040
QUERY_ID = 10_000_000
TEST_ID = 20_000_000
SIGMA = 3.0
MUS = (0.25, 0.5, 1.0, 2.0, 4.0)
HD_RE = re.compile(r"HD\[(\d+)\]")


# ---------------------------------------------------------------- plan
def test_hd(name: str) -> np.ndarray:
    _, _, m = o2.load_store(name, ol.STORE, mmap=False)
    raw = pd.read_csv(ROOT / "data" / "raw" / "tests" / f"rank_prediction_test_{name.split('_')[1]}.csv", dtype=str,
                      keep_default_na=False).set_index("question_id")
    out = []
    for qid, slot in zip(m["question_id"], m["slot"]):
        found = HD_RE.findall(raw.at[qid, f"sgf_{slot}"])
        out.append(int(found[0]) if len(found) == 1 else -1)
    return np.array(out)


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    hd = meta["handicap"].to_numpy()
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    rng = np.random.default_rng(SEED)
    picks = []
    for _, games in pd.Series(gal).groupby(player[gal]):
        picks.append(np.sort(rng.permutation(games.to_numpy())[:GAL_GAMES]))
    gal = np.concatenate(picks)
    rows = [pd.DataFrame({"id": gal, "source": "train", "game_index": gal, "store_row": -1,
                          "keep": pcolor[gal], "hd": hd[gal], "player": player[gal]})]
    q = pd.read_parquet(o2.OUT / "queries.parquet")
    q["qrow"] = np.arange(len(q))
    ref = q[q["set"] == "ref"]
    ref = ref[ref.groupby("owner").cumcount() < REF_GAMES]
    use = pd.concat([ref, q[q["set"].isin(["u_opp", "s_opp", "test"])]])
    tr = use[use["part"] == "train"]
    rows.append(pd.DataFrame({"id": QUERY_ID + tr["qrow"].to_numpy(), "source": "train",
                              "game_index": tr["game_index"].to_numpy(), "store_row": -1,
                              "keep": tr["qcolor"].to_numpy(), "hd": hd[tr["game_index"].to_numpy()], "player": ""}))
    for name in ol.PARTS:
        te = use[use["part"] == name]
        thd = test_hd(name)[te["store_row"].to_numpy()]
        if (thd < 0).any() or (thd > 1).any():
            raise ValueError(f"{name}: test game HD outside 0/1")
        rows.append(pd.DataFrame({"id": QUERY_ID + te["qrow"].to_numpy(), "source": name, "game_index": -1,
                                  "store_row": te["store_row"].to_numpy(), "keep": te["qcolor"].to_numpy(),
                                  "hd": thd, "player": ""}))
    jobs = pd.concat(rows, ignore_index=True)
    jobs["keep"] = jobs["keep"].astype(np.int8)
    jobs["hd"] = jobs["hd"].astype(np.int8)
    if (jobs["hd"] > 1).any():
        raise ValueError("HD>=2 job")
    jobs.to_parquet(OUT / "jobs.parquet", index=False)
    length = meta["length"].to_numpy()
    est = int(length[gal].sum() / 2 + length[tr["game_index"].to_numpy()].sum() / 2 + 115 * 10000)
    print(json.dumps({"gallery_games": int(len(gal)), "gallery_players": int(len(picks)),
                      "query_rows": int(len(use)), "by_set": use["set"].value_counts().to_dict(),
                      "positions_estimate": est}, indent=1))


def plan_g40(_args) -> None:
    """refinement (RESULTS 2026-11-23): every style-gallery game (<= 40 per player); jobs for the ones not in jobs.parquet."""
    meta = ol.load_train()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    hd = meta["handicap"].to_numpy()
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    rng = np.random.default_rng(SEED)
    picks, first = [], []
    for _, games in pd.Series(gal).groupby(player[gal]):
        perm = rng.permutation(games.to_numpy())
        picks.append(np.sort(perm[:40]))
        first.append(np.sort(perm[:GAL_GAMES]))
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    old = jobs.loc[jobs["id"] < QUERY_ID, "id"].to_numpy()
    if not np.array_equal(np.sort(np.concatenate(first)), np.sort(old)):
        raise ValueError("the first GAL_GAMES picks do not reproduce jobs.parquet")
    new = np.setdiff1d(np.concatenate(picks), old)
    out = pd.DataFrame({"id": new, "source": "train", "game_index": new, "store_row": -1, "keep": pcolor[new],
                        "hd": hd[new].astype(np.int8), "player": player[new]})
    out.to_parquet(OUT / "jobs_g40.parquet", index=False)
    print(json.dumps({"new_gallery_games": int(len(out)), "positions_estimate": int(meta["length"].to_numpy()[new].sum() / 2)}))


def plan_delta(_args) -> None:
    main = pd.read_parquet(o2.OUT / "queries.parquet")
    extra = pd.read_parquet(o2.OUT / "queries_extra.parquet")
    if (extra["game_index"].duplicated().any() or not extra["hd"].isin((0, 1)).all()
            or np.isin(extra["game_index"], main.loc[main["set"] == "s_opp", "game_index"]).any()):
        raise ValueError("invalid extra Valid-S opponent games")
    jobs = pd.DataFrame({"id": QUERY_ID + len(main) + np.arange(len(extra)), "source": "train",
                         "game_index": extra["game_index"], "store_row": -1,
                         "keep": extra["qcolor"].astype(np.int8), "hd": extra["hd"].astype(np.int8), "player": ""})
    jobs.to_parquet(OUT / "jobs_delta_extra.parquet", index=False)
    print(f"extra Valid-S opponent games: {len(jobs)}", flush=True)


# ---------------------------------------------------------------- moves
_STORES: dict = {}


def store_games(name: str, rows: np.ndarray):
    if name not in _STORES:
        moves, colors, m = o2.load_store(name, ol.STORE, mmap=False)
        _STORES[name] = (moves, colors, m["offset"].to_numpy(), m["length"].to_numpy())
    moves, colors, off, length = _STORES[name]
    return [(moves[off[r]:off[r] + length[r]].astype(np.int16), colors[off[r]:off[r] + length[r]].astype(np.int8))
            for r in rows]


def job_games(jobs: pd.DataFrame):
    from go_aicup.features.humansl import CACHE, cache_games
    out = [None] * len(jobs)
    tr = np.flatnonzero((jobs["source"] == "train").to_numpy())
    for i, g in zip(tr, cache_games(jobs["game_index"].to_numpy()[tr], CACHE)):
        out[i] = g
    for name in ol.PARTS:
        idx = np.flatnonzero((jobs["source"] == name).to_numpy())
        for i, g in zip(idx, store_games(name, jobs["store_row"].to_numpy()[idx])):
            out[i] = g
    return out


# ---------------------------------------------------------------- policy (GPU)
def policy(args) -> None:
    from scripts.identity.pid2_likelihood import extract_positions
    out = OUT / ("smoke" if args.limit else f"{args.prefix}{args.shard:02d}")
    if (out / "meta.json").exists() and not args.limit:
        print(f"{out} done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    jobs = pd.read_parquet(OUT / args.jobs).iloc[args.shard::args.shards]
    if args.limit:
        jobs = jobs.iloc[:args.limit]
    played = job_games(jobs)
    groups = []
    for h in (0, 1):
        idx = np.flatnonzero((jobs["hd"] == h).to_numpy())
        items = [(played[i][0], played[i][1], played[i][1] == jobs["keep"].iloc[i]) for i in idx]
        groups.append((h, items, jobs["id"].to_numpy()[idx]))
    result, meta = extract_positions(groups, args.workers)
    for name, value in result.items():
        np.save(out / f"{name}.npy", value)
    meta["games"] = int(len(jobs))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# ---------------------------------------------------------------- LLR (GPU)
def llr(args) -> None:
    import torch
    from scripts.identity.pid2_likelihood import DIM, fit, position_tensors
    if args.expanded and not args.g40:
        raise ValueError("--expanded requires --g40")
    expanded_path = OUT / "llr_g40_expanded.npz"
    if args.expanded and expanded_path.exists():
        raise FileExistsError(expanded_path)
    device = torch.device("cuda")
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    shards = sorted(p for p in OUT.glob("shard_*") if (p / "meta.json").exists())
    if len(shards) != args.shards:
        raise SystemExit(f"{len(shards)} of {args.shards} policy shards")
    if args.g40:
        extra = sorted(p for p in OUT.glob("g40_*") if (p / "meta.json").exists())
        if len(extra) != args.shards:
            raise SystemExit(f"{len(extra)} of {args.shards} g40 shards")
        shards += extra
        jobs = pd.concat([jobs, pd.read_parquet(OUT / "jobs_g40.parquet")], ignore_index=True)
    if args.expanded:
        shards += [OUT / "delta_policy" / f"shard_{i:02d}" for i in range(args.shards)]
        if not all((shard / "meta.json").is_file() for shard in shards):
            raise ValueError("incomplete extra Valid-S policy shards")
        extra_jobs = pd.read_parquet(OUT / "jobs_delta_extra.parquet")
        jobs = pd.concat([jobs, extra_jobs], ignore_index=True)
    arr = {k: np.concatenate([np.load(s / f"{k}.npy") for s in shards]) for k in ("game", "ply", "top", "lp", "slot")}
    if not np.array_equal(np.unique(arr["game"]), np.sort(jobs["id"].to_numpy())):
        raise ValueError("policy shards do not cover the jobs")
    games = dict(zip(jobs["id"].to_numpy(), job_games(jobs)))
    t0 = time.time()
    B, PHI, SLOT, span = position_tensors(arr, lambda g: games[int(g)][0], device)
    PHI = PHI[:, :, :DIM]
    del arr
    print(f"positions {len(SLOT)}, outside top-32 {float((SLOT == 32).float().mean()):.4f}, {time.time() - t0:.0f}s",
          flush=True)

    def positions_of(ids_per_owner):
        pos, own = [], []
        for i, ids in enumerate(ids_per_owner):
            for g in ids:
                s, e = span[int(g)]
                pos.append(np.arange(s, e))
                own.append(np.full(e - s, i))
        return (torch.from_numpy(np.concatenate(pos)).to(device), torch.from_numpy(np.concatenate(own)).to(device))

    meta = ol.load_train()
    candidates = np.array(sorted(set(meta["player_id"].astype(str))))
    gal = jobs[jobs["source"].eq("train") & (jobs["id"] < QUERY_ID)]
    by_player = gal.groupby("player")["id"].apply(np.asarray)
    rng = np.random.default_rng(SEED)
    pop_players = rng.choice(np.array(by_player.index), size=POP_PLAYERS, replace=False)
    pos_p, own_p = positions_of([np.concatenate([by_player[p][:POP_GAMES] for p in pop_players])])
    zero = torch.zeros(DIM, device=device, dtype=torch.float64)
    theta_pop = fit(PHI, B, SLOT, pos_p, own_p, 1, zero, 10.0)[0]
    print("theta_pop", np.round(theta_pop.cpu().numpy(), 3).tolist(), flush=True)
    present = [p for p in candidates if p in by_player.index]
    pos_g, own_g = positions_of([by_player[p] for p in present])
    theta_present = fit(PHI, B, SLOT, pos_g, own_g, len(present), theta_pop, SIGMA)
    theta = theta_pop[None].repeat(len(candidates), 1).clone()
    col = pd.Series(np.arange(len(candidates)), index=candidates)
    theta[torch.from_numpy(col.loc[present].to_numpy()).to(device)] = theta_present
    print(f"theta fitted for {len(present)} players, {time.time() - t0:.0f}s", flush=True)

    def score_candidates(rows, cand):
        res = np.zeros((len(rows), cand.shape[1]), np.float32)
        th = theta.float()
        tp = theta_pop.float()
        for s in range(0, len(rows), 2000):
            blk = rows[s:s + 2000]
            pos, own = positions_of([[QUERY_ID + r] for r in blk])
            cth = th[torch.from_numpy(cand[s:s + 2000].astype(np.int64)).to(device)]
            acc = torch.zeros(len(blk), cand.shape[1], device=device, dtype=torch.float64)
            for c0 in range(0, len(pos), 200_000):
                p, o = pos[c0:c0 + 200_000], own[c0:c0 + 200_000]
                f = PHI[p].float()
                zc = B[p][:, :, None] + torch.einsum("pjd,pcd->pjc", f, cth[o])
                z0 = B[p] + torch.einsum("pjd,d->pj", f, tp)
                ar = torch.arange(len(p), device=device)
                lc = zc[ar, SLOT[p]] - torch.logsumexp(zc, dim=1)
                l0 = z0[ar, SLOT[p]] - torch.logsumexp(z0, dim=1)
                acc.index_add_(0, o, (lc - l0[:, None]).double())
            res[s:s + 2000] = acc.cpu().numpy()
        return res

    z = np.load(o2.OUT / "scores.npz")
    arm = json.loads((o2.OUT / "result.json").read_text())["selected_arm"]
    q = pd.read_parquet(o2.OUT / "queries.parquet")
    qrows = np.sort(jobs.loc[jobs["id"].between(QUERY_ID, QUERY_ID + len(q) - 1), "id"].to_numpy() - QUERY_ID)
    out = {"qrows": qrows}
    if args.expanded:
        wide = np.load(o2.OUT / "scores_f8_top30.npz")
        original = np.load(OUT / "llr_g40.npz")
        if (arm != "f8" or not np.array_equal(qrows, original["qrows"])
                or not np.allclose(theta_pop.cpu().numpy(), original["theta_pop"], atol=1e-6)):
            raise ValueError("expanded LLR does not match the opp5 setup")
        for label, key in (("masked", arm), ("seen", f"seen_{arm}")):
            rows = original[f"{label}_rows"]
            cand = wide[f"{key}_idx"][rows]
            sample = np.random.default_rng(SEED).choice(len(rows), 128, replace=False)
            reproduced = score_candidates(rows[sample], cand[sample, :o2.TOP])
            diff = np.max(np.abs(reproduced - original[f"{label}_llr"][sample]))
            if not np.isfinite(diff) or diff > 0.002:
                raise ValueError(f"{label}: opp5 LLR sample max difference {diff}")
            print(f"{label}: baseline LLR sample max difference {diff:.4g}", flush=True)
            out[f"{label}_extra_candidates"] = score_candidates(rows, cand[:, o2.TOP:])
        extra_scores = np.load(o2.OUT / "scores_extra.npz")
        extra_query = pd.read_parquet(o2.OUT / "queries_extra.parquet")
        if (not np.array_equal(extra_jobs["game_index"], extra_query["game_index"])
                or not np.array_equal(extra_scores["cand_rank"], z["cand_rank"])):
            raise ValueError("extra Valid-S games or candidate ranks are out of order")
        out["extra_game_index"] = extra_jobs["game_index"].to_numpy()
        out["extra_games_llr"] = score_candidates(extra_jobs["id"].to_numpy() - QUERY_ID, extra_scores[f"{arm}_idx"])
        temporary = expanded_path.with_suffix(".partial.npz")
        np.savez_compressed(temporary, **out)
        temporary.replace(expanded_path)
        print(f"saved {expanded_path}, {time.time() - t0:.0f}s", flush=True)
        return

    for label, key in (("masked", arm), ("seen", f"seen_{arm}")):
        rows = qrows[q["set"].to_numpy()[qrows] == "ref"] if label == "seen" else qrows
        res = score_candidates(rows, z[f"{key}_idx"][rows])
        out[f"{label}_rows"], out[f"{label}_llr"] = rows, res
        print(f"{label}: {len(rows)} queries, {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / ("llr_g40.npz" if args.g40 else "llr.npz"), theta_pop=theta_pop.cpu().numpy(), **out)


# ---------------------------------------------------------------- fuse (CPU)
def reranked(mu: float, z, arm: str, L: dict, width: int = o2.TOP) -> dict:
    """Keep ten names after adding LLR within the first `width` f8 candidates."""
    out = {}
    for label, key in (("masked", arm), ("seen", f"seen_{arm}")):
        idx, val = z[f"{key}_idx"][:, :width].copy(), z[f"{key}_val"][:, :width].astype(np.float64)
        rows, llr_ = L[f"{label}_rows"], L[f"{label}_llr"][:, :width].astype(np.float64)
        v = val[rows] + mu * llr_
        order = np.argsort(-v, axis=1)
        idx[rows] = np.take_along_axis(idx[rows], order, axis=1)
        val[rows] = np.take_along_axis(v, order, axis=1)
        out[label] = (idx[:, :o2.TOP], val[:, :o2.TOP])
    return out


def link_rows(q, idx, val, cand_rank, edges):
    gap = val[:, 0] - val[:, 1]
    colour = q["qcolor"].to_numpy()
    bins = np.zeros(len(q), np.int64)
    for c in (0, 1):
        same = colour == c
        bins[same] = np.searchsorted(np.array(edges[c]), gap[same], side="right")
    return q.assign(bin=bins, rj=cand_rank[idx[:, 0]], gap=gap, oc=(1 - colour).astype(np.int64))


def fuse(args) -> None:
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_nonlin as nl
    import scripts.strength.rank_opp_fuse as fu
    import scripts.strength.rank_opp_refine as rf
    import scripts.strength.rank_prod_a0_fusion as prod
    from go_aicup.metrics import score_rank  # noqa: F401
    z = np.load(o2.OUT / "scores.npz")
    arm = json.loads((o2.OUT / "result.json").read_text())["selected_arm"]
    L = dict(np.load(OUT / ("llr_g40.npz" if args.experiment else args.llr)))
    q = pd.read_parquet(o2.OUT / "queries.parquet")
    sets, colour = q["set"].to_numpy(), q["qcolor"].to_numpy()
    cand_rank = z["cand_rank"]
    owner_col = z["owner_col"]
    ref_rows = L["seen_rows"]
    is_ref = np.zeros(len(q), bool)
    is_ref[ref_rows] = True
    players = np.array(sorted(q.loc[is_ref, "owner"].unique()))
    half_a = set(players[np.random.default_rng(SEED).permutation(len(players))[:len(players) // 2]])
    in_a = is_ref & q["owner"].isin(half_a).to_numpy()
    in_b = is_ref & ~in_a
    report = {"mu": {}}

    def gate_stats(rr, mask_th, mask_eval):
        idx, val = rr["masked"]
        sidx, sval = rr["seen"]
        gap, sgap = val[:, 0] - val[:, 1], sval[:, 0] - sval[:, 1]
        hit = sidx[:, 0] == owner_col
        th = {c: float(np.quantile(gap[mask_th & (colour == c)], 0.99)) for c in (0, 1)}
        t = np.where(colour == 1, th[1], th[0])
        return float(((sgap > t) & hit)[mask_eval].mean()), float(hit[mask_eval].mean())

    if args.experiment:
        mu = 1.0
    else:
        base_rr = reranked(0.0, z, arm, L)
        report["mu"]["0"] = dict(zip(("half_b_hit_q99", "half_b_top1"), gate_stats(base_rr, in_b, in_b)))
        best = None
        for mu in MUS:
            rr = reranked(mu, z, arm, L)
            hit_a, top_a = gate_stats(rr, in_a, in_a)
            hit_b, top_b = gate_stats(rr, in_b, in_b)
            report["mu"][str(mu)] = {"half_a_hit_q99": hit_a, "half_a_top1": top_a, "half_b_hit_q99": hit_b, "half_b_top1": top_b}
            if best is None or hit_a > best[1]:
                best = (mu, hit_a)
        mu = best[0]
    report["chosen_mu"] = mu

    def link_table(rr):
        idx, val = rr["masked"]
        gap = val[:, 0] - val[:, 1]
        edges = {c: [float(np.quantile(gap[is_ref & (colour == c)], x)) for x in fu.QS] +
                 [float(gap[is_ref & (colour == c)].max())] for c in (0, 1)}
        t = link_rows(q, idx, val, cand_rank, edges).loc[L["masked_rows"]]
        return t, {"idx": idx, "val": val, "cand_rank": cand_rank}, edges

    def soft_evidence(t, top, tau):
        u = t[t["set"] == "u_opp"]
        prob = rf._delta_prob(t[t["set"] == "s_opp"])
        return pd.DataFrame(rf.soft_ll(prob, u, top, tau), index=u["game_index"].to_numpy())

    rr = reranked(mu, z, arm, L)
    t_new, top_new, edges = link_table(rr)
    report["edges"] = edges
    reference = "BASE_0.35"
    if args.experiment:
        expanded = np.load(OUT / "llr_g40_expanded.npz")
        if not np.array_equal(expanded["qrows"], L["qrows"]):
            raise ValueError("expanded LLR queries do not match opp5")
        reference = "opp5"
        variants = {reference: (t_new, top_new)}
        if args.experiment == "wide":
            wide = np.load(o2.OUT / "scores_f8_top30.npz")
            combined = dict(L)
            for label in ("masked", "seen"):
                combined[f"{label}_llr"] = np.hstack([L[f"{label}_llr"], expanded[f"{label}_extra_candidates"]])
            report["retrieval"] = {}
            for width in (10, 20, 30):
                ranked = reranked(mu, wide, arm, combined, width)
                name = f"top{width}"
                hit, top1 = gate_stats(ranked, in_b, in_b)
                recall = (wide[f"seen_{arm}_idx"][in_b, :width] == owner_col[in_b, None]).any(1).mean()
                report["retrieval"][name] = {"f8_recall": float(recall), "top1": top1, "hit_q99": hit}
                if width > o2.TOP:
                    table, top, variant_edges = link_table(ranked)
                    variants[name] = (table, top)
                    report["retrieval"][name]["edges"] = variant_edges
        else:
            extra = pd.read_parquet(o2.OUT / "queries_extra.parquet")
            if not np.array_equal(extra["game_index"], expanded["extra_game_index"]):
                raise ValueError("extra Valid-S query order changed")
            extra.index = np.arange(len(q), len(q) + len(extra))
            with np.load(o2.OUT / "scores_extra.npz") as z_extra:
                val = z_extra[f"{arm}_val"].astype(np.float64) + mu * expanded["extra_games_llr"]
                order = np.argsort(-val, axis=1)[:, :2]
                add = link_rows(extra, np.take_along_axis(z_extra[f"{arm}_idx"], order, axis=1),
                                np.take_along_axis(val, order, axis=1), cand_rank, edges)
            variants["delta62k"] = (pd.concat([t_new, add]), top_new)
            groups = ["hd", "oc", "bin"]
            report["delta_counts"] = {name: {"games": int(len(s)), "bins": s.groupby(groups).size().reset_index(name="n").to_dict("records")}
                                      for name, (t, _) in variants.items() for s in [t[t["set"] == "s_opp"]]}
        ev = {name: soft_evidence(t, top, 3.5) for name, (t, top) in variants.items()}
        cells = {name: (name, 0.65) for name in variants}
    else:
        t_base, top_base = rf.full_table()
        u_n = t_new[t_new["set"] == "u_opp"]
        ev = {"BASE": soft_evidence(t_base, top_base, 3.5),
              "D1": pd.DataFrame(fu.game_ll(fu.fit_delta(t_new[t_new["set"] == "s_opp"]), u_n, False),
                                 index=u_n["game_index"].to_numpy())}
        taus = (2.0, 3.5, 7.0)
        ev.update({f"S{tau:g}": soft_evidence(t_new, top_new, tau) for tau in taus})
        alphas = tuple(float(a) for a in args.alphas.split(","))
        cells = {"BASE_0.35": ("BASE", 0.35)} | {f"D1_{a:g}": ("D1", a) for a in (0.25, 0.35)} | \
                {f"S{tau:g}_{a:g}": (f"S{tau:g}", a) for tau in taus for a in alphas}
    print(json.dumps(report, indent=1), flush=True)

    fit, eval_games, x_fit, x_eval = prod.train_data()
    o, (e,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.hstack([x_fit, o]), np.hstack([x_eval, e])
    pri = more.player_prior(fit[fit["role"] == "S"])
    qsets = prod.question_sets()
    if args.local_cells:
        selected = args.local_cells.split(",")
        if len(selected) < 2 or len(set(selected)) != len(selected) or not set(selected) <= cells.keys():
            raise ValueError("--local-cells requires a reference and at least one distinct, available candidate")
        reference = selected[0]
        cells = {cell: cells[cell] for cell in selected}
    ref_name, ref_alpha = cells[reference]
    qll = {name: {n: fu.question_ll(qq["games"], frame) for n, qq in qsets.items()} for name, frame in ev.items()}
    probs, e1 = {}, {}
    for seed in prod.SEEDS:
        model = prod._fit(fit, xf, pri, seed)
        for n, qq in qsets.items():
            probs[seed, n] = model.probabilities(xe, eval_games["game_index"].to_numpy(), qq)
        qq = qsets["E1"]
        k, truth = qq["k"].to_numpy(), qq["rank_id"].to_numpy()
        ref = rf.cell_scores(probs[seed, "E1"], qll[ref_name]["E1"], ref_alpha, truth)
        e1[seed] = {}
        for cell, (name, a) in cells.items():
            s = rf.cell_scores(probs[seed, "E1"], qll[name]["E1"], a, truth)
            e1[seed][cell] = {**{str(bb): float(s[k == bb].mean() - ref[k == bb].mean()) for bb in K_BUCKETS},
                              "equal_k": hf.equal_k(s, k) - hf.equal_k(ref, k)}
        print(seed, json.dumps({c: round(v["equal_k"], 4) for c, v in e1[seed].items()}), flush=True)
    mean_e1 = {c: {kk: round(float(np.mean([e1[s][c][kk] for s in prod.SEEDS])), 4)
                   for kk in [str(bb) for bb in K_BUCKETS] + ["equal_k"]} for c in cells}
    chosen = max([c for c in cells if c != reference], key=lambda c: mean_e1[c]["equal_k"])
    qq = qsets["E2"]
    k, truth = qq["k"].to_numpy(), qq["rank_id"].to_numpy()
    name, a = cells[chosen]
    e2 = {}
    for seed in prod.SEEDS:
        ref = rf.cell_scores(probs[seed, "E2"], qll[ref_name]["E2"], ref_alpha, truth)
        s = rf.cell_scores(probs[seed, "E2"], qll[name]["E2"], a, truth)
        e2[seed] = {**{str(bb): float(s[k == bb].mean() - ref[k == bb].mean()) for bb in K_BUCKETS},
                    "equal_k": hf.equal_k(s, k) - hf.equal_k(ref, k)}
    mean_e2 = {kk: round(float(np.mean([e2[s][kk] for s in prod.SEEDS])), 4)
               for kk in [str(bb) for bb in K_BUCKETS] + ["equal_k"]}
    report.update({"mean_E1_vs_base": mean_e1, "chosen": chosen, "chosen_E2_vs_base": mean_e2,
                   "E1_per_seed": {str(s): round(e1[s][chosen]["equal_k"], 4) for s in prod.SEEDS},
                   "E2_per_seed": {str(s): round(e2[s]["equal_k"], 4) for s in prod.SEEDS}})
    if args.experiment:
        report["reference"] = reference
        report["pass"] = (mean_e1[chosen]["equal_k"] >= 0 and mean_e2["equal_k"] >= 0.002
                          and all(mean_e2[str(k)] >= -0.005 for k in K_BUCKETS))
    elif args.local_cells:
        report["reference"] = reference
    else:
        report["pass"] = bool(mean_e1[chosen]["equal_k"] > 0 and mean_e2["equal_k"] > 0)
    print(json.dumps({k_: v for k_, v in report.items() if k_ != "mu"}, indent=1), flush=True)
    if not args.local_cells and (report["pass"] or args.force):
        t_exp, top_exp = variants[name] if args.experiment else (t_new, top_new)
        pool = t_exp[t_exp["set"].isin(["s_opp", "u_opp"])]
        test_index = t_exp.index[t_exp["set"] == "test"].to_numpy()
        test_rows = t_exp.loc[t_exp["set"] == "test"].set_index(["part", "store_row"])
        if name == "D1":
            per = None
        else:
            prob = rf._delta_prob(pool)
            tau = 3.5 if args.experiment else float(name[1:])

            def per(rows):
                return rf.soft_ll(prob, rows, top_exp, tau)

        def evidence(games, sub, table, key):
            pos = test_rows.index.get_indexer(pd.MultiIndex.from_arrays([sub["part"], sub["store_row"]]))
            rows = sub.assign(_row=test_index[pos]).set_index("_row")
            values = fu.game_ll(table, rows, False) if per is None else per(rows)
            return fu.question_ll(games, pd.DataFrame(values, index=sub[key].to_numpy()))

        tag = f"w{name[3:]}-l10" if args.experiment == "wide" else "bigd-l10" if args.experiment else args.tag
        report["export"] = fu.export(t_exp, f"DELTA-ALL_{a:g}", args.job, evidence=evidence, tag=tag)
    suffix = f"_{args.experiment}" if args.experiment else ""
    (OUT / f"fuse_{args.job}{suffix}.json").write_text(json.dumps(report, indent=1, default=int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    sub.add_parser("plan-g40").set_defaults(fn=plan_g40)
    sub.add_parser("plan-delta").set_defaults(fn=plan_delta)
    p = sub.add_parser("policy")
    p.add_argument("--jobs", default="jobs.parquet")
    p.add_argument("--prefix", default="shard_")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=policy)
    p = sub.add_parser("llr")
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--g40", action="store_true", help="theta on every style-gallery game (jobs_g40.parquet, g40_ shards)")
    p.add_argument("--expanded", action="store_true", help="score added top-30 candidates and extra Valid-S games")
    p.set_defaults(fn=llr)
    p = sub.add_parser("fuse")
    p.add_argument("--job", default="local")
    p.add_argument("--force", action="store_true")
    p.add_argument("--llr", default="llr.npz")
    p.add_argument("--alphas", default="0.25,0.35,0.45")
    p.add_argument("--tag", default="opp4-l10")
    p.add_argument("--local-cells", help="score only these comma-separated cells, using the first as the reference")
    p.add_argument("--experiment", choices=("wide", "delta"), help="compare one expanded opp5 experiment")
    p.set_defaults(fn=fuse)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
