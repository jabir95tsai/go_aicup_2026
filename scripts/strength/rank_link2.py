"""L10 v2: link competition-2 test questions to known players with the competition-1 added-set style models
(2026-11-21, registered in RESULTS.md before the run)

rank_link (11/13: opening model + old GE2E p0_1200) recognised 5 test questions as labelled training players
(L10, 11/11: 7). The competition-1 added-set style term has improved a lot since then; here it replaces the old
GE2E vector:  s = 0.25 cos_B + 0.75 * mean(cos_A1, cos_A2)
  B   pid_ge2e_v4_d4_hd_9600 encoder + seqB aggregation head
  A1  E2E encoder + head trained from scratch on Yugen + 40,000 OGS players
  A2  same with all 82,481 OGS players
Three questions, each with its own null:
 (a) training  is the test player one of the 8,007 labelled training players? fused f = o + lam * w * s
               (o = opening pp+slot log-likelihood, D4, as rank_link; lam chosen on reference half A);
               a detected question takes the matched player's rank (export).
 (b) pools     is the test player a competition-1 candidate (original pools 800 x 100 black games;
               added pools 1,200 x 10 games)? style only. Report only (more games would need H/C features).
 (c) pairs     are two test questions the same player? style only. Report only.
Reference ("seen" / null): every player the three encoders never trained on (pid splits 0-2, 1,182 with games),
one colour each (the one with more HD<=1 games outside the rank_opp_link gallery), 20 such games, nested
questions k = 5/10/15/20. Null = the same rows with the own column removed (a), against the pools (b), or pairs of
different players (c). The old fusion (o + lam * w_old * g_old) is rerun on the same reference for comparison.

    python -m scripts.strength.rank_link2 plan                       # CPU, minutes
    python -m scripts.strength.rank_link2 embed --shard I --shards N # 1 GPU each
    python -m scripts.strength.rank_link2 analyze                    # CPU, about an hour (opening scores)
    python -m scripts.strength.rank_link2 export --job ID [--base 751959-cvec2.csv]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_own_link as wl
from scripts.strength.l10_check import score_questions

OUT = ROOT / "outputs" / "rank" / "link2"
VEC = OUT / "vectors"
SEED = 2030
REF_GAMES, POOL_GAMES, POOL2_GAMES = 20, 20, 5
ENCODERS = {
    "B": ("pid_ge2e_v4_d4_hd_9600", ROOT / "outputs" / "pid" / "pid2_seqagg" / "head_B" / "best.pt"),
    "A1": ("pid_e2e_ogs_A1_YO9600/step9600", ROOT / "outputs" / "pid" / "ge2e" / "pid_e2e_ogs_A1_YO9600" / "step9600" / "best.pt"),
    "A2": ("pid_e2e_ogs_A2_YO9600/step9600", ROOT / "outputs" / "pid" / "ge2e" / "pid_e2e_ogs_A2_YO9600" / "step9600" / "best.pt")}
WEIGHTS = {"B": 0.25, "A1": 0.375, "A2": 0.375}
BENCH_CACHE = {"B": Path("/home/jason1122/pid2_seqagg_cache/bench"), "A1": Path("/work/jason1122/ogs_a1_cache/A1/bench"),
               "A2": Path("/home/jason1122/ogs_runs/a2_cache/A2/bench")}
PID_STORE = PROCESSED / "identity" / "artifacts" / "day0_1112" / "store"
ART = PROCESSED / "identity" / "artifacts" / "day0_1112"
POOL_CACHE = {"pool_public": ART / "insurance" / "pool_public" / "cache",
              "pool_private": ART / "insurance" / "pool_private" / "cache",
              "pool2_public": ART / "public2_d4_9600" / "pool2_public" / "cache",
              "pool2_private": ART / "private2_d4_9600" / "pool2_private" / "cache"}
POOLS = {"orig": ("pool_public", "pool_private"), "added": ("pool2_public", "pool2_private")}
PARTS = ("train", *ol.PARTS, *POOL_CACHE)
LAMBDAS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
AGREE_FP = 0.001
OLD_W = ROOT / "outputs" / "pid" / "pid2_bench" / ol.RUN.name / "summary.json"
NEW_W = ROOT / "outputs" / "pid" / "pid2_bench" / "pid_ge2e_v4_d4_hd_9600" / "embeddings.npz"
CHUNK = 12000
CHECK_GAMES = 300


# ---------------------------------------------------------------- plan
def unseen_players() -> set[str]:
    gallery_table, _ = data.pid_splits(data.load_meta())
    return set(gallery_table.loc[gallery_table["split"].isin(ol.EXCLUDED_SPLITS), "player_id"].astype(str))


def plan(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gallery = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy().astype(np.int8)
    free = meta["handicap"].to_numpy() <= 1
    free[gallery] = False
    free &= np.isin(player, sorted(unseen_players()))
    rng = np.random.default_rng(SEED)
    rows = []
    by_player = pd.Series(np.flatnonzero(free)).groupby(player[free]).apply(np.asarray)
    for pid in sorted(by_player.index):
        games = by_player[pid]
        counts = np.bincount(pcolor[games], minlength=2)
        color = int(rng.integers(2)) if counts[0] == counts[1] else int(counts.argmax())
        pick = rng.permutation(games[pcolor[games] == color])[:REF_GAMES]
        if len(pick) < REF_GAMES:
            continue
        rows.append(pd.DataFrame({"player_id": pid, "color": color, "order": np.arange(REF_GAMES), "game_index": pick}))
    ref = pd.concat(rows, ignore_index=True)
    ref.to_parquet(OUT / "ref.parquet", index=False)

    pool_rows = []
    for name in POOL_CACHE:
        _, _, m = load_store(name, PID_STORE, mmap=False)
        size = POOL_GAMES if name.startswith("pool_") else POOL2_GAMES
        indices = m.groupby("player_id").indices
        for pid in sorted(indices):
            idx = indices[pid]
            pick = rng.permutation(idx)[:size]
            pool_rows.append(pd.DataFrame({"part": name, "player_id": str(pid), "order": np.arange(len(pick)),
                                           "store_row": pick}))
    pool_ref = pd.concat(pool_rows, ignore_index=True)
    pool_ref.to_parquet(OUT / "pool_ref.parquet", index=False)

    train_games = np.unique(np.concatenate([gallery, ref["game_index"].to_numpy()]))
    np.save(OUT / "train_games.npy", train_games)
    print(json.dumps({"reference_players": int(ref["player_id"].nunique()),
                      "reference_colour": ref.groupby("color")["player_id"].nunique().to_dict(),
                      "pool_simulation": pool_ref.groupby("part")["player_id"].nunique().to_dict(),
                      "train_games": int(len(train_games)), "gallery_games": int(len(gallery))}, indent=1))


# ---------------------------------------------------------------- embed
def part_inputs(name: str, meta: pd.DataFrame):
    """(cache dir, cache game_index, own colour int8, length, store row or None) in cache order."""
    if name == "train":
        games = np.load(OUT / "train_games.npy")
        return (data.CACHE_DIR, games, meta["player_color"].to_numpy()[games].astype(np.int8),
                meta["length"].to_numpy()[games], None)
    if name in ol.PARTS:
        _, _, m = load_store(name, ol.STORE, mmap=False)
        cache_dir = ol.OUT / "cache" / name
    else:
        _, _, m = load_store(name, PID_STORE, mmap=False)
        cache_dir = POOL_CACHE[name]
    games = pd.read_parquet(cache_dir / "games.parquet")
    rows = games["store_row"].to_numpy()
    if len(games) != len(m) or not np.array_equal(games["length"].to_numpy(), m["length"].to_numpy()[rows]):
        raise ValueError(f"{name}: board cache does not match the store")
    own = (m["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
    return cache_dir, games["game_index"].to_numpy(), own, games["length"].to_numpy(), rows


def head_game_vectors(model, config, head, device, cache_dir, games, colour, length) -> np.ndarray:
    """game vectors exactly as pid2_style_export.head_vectors (per-move vectors stored as float16)."""
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, move_vectors

    out = np.empty((len(games), 256), np.float32)
    cache = BoardCache(cache_dir)
    for s in range(0, len(games), CHUNK):
        sl = slice(s, s + CHUNK)
        vec, off = move_vectors(model, config, cache, games[sl], colour[sl], length[sl], device, 0)
        store = MoveStore.__new__(MoveStore)
        store.vectors, store.offsets, store.games = vec.astype(np.float16), off, games[sl]
        store.first_ply = colour[sl].astype(np.int64)
        out[sl] = game_vectors(store=store, head=head, rows=np.arange(len(store.games)), device=device)
    return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (ol.unit(a.astype(np.float64)) * ol.unit(b.astype(np.float64))).sum(1)


def selfcheck(meta: pd.DataFrame, encoders: dict) -> dict:
    """Fresh path against the saved bench caches (new encoders) and the rank_opp_link shard (old GE2E)."""
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity import twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors

    pcol = meta["player_color"].to_numpy().astype(np.int8)
    report = {}
    for name, (model, config, head, device) in encoders.items():
        store = MoveStore(BENCH_CACHE[name], pcol)
        rows = np.arange(CHECK_GAMES)
        saved = game_vectors(head, store, rows, device)
        games = store.games[rows]
        fresh = head_game_vectors(model, config, head, device, data.CACHE_DIR, games, pcol[games],
                                  meta["length"].to_numpy()[games])
        cos = cosine_rows(saved, fresh)
        report[name] = {"min_cosine": float(cos.min()), "max_abs": float(np.abs(saved - fresh).max())}
        del store
    model, config, device, min_ply = ol.load_model()
    work = pd.read_parquet(ol.OUT / "plan.parquet").iloc[:CHECK_GAMES]
    games = work["game_index"].to_numpy()
    fresh = twocolor.embed_games(model, config, BoardCache(data.CACHE_DIR), games, work["side"].to_numpy(),
                                 meta["length"].to_numpy()[games], device, swap=True, min_ply=min_ply)
    saved = np.load(ol.OUT / "train_0of6.npy", mmap_mode="r")[:CHECK_GAMES]
    report["old"] = {"min_cosine": float(cosine_rows(saved, fresh).min())}
    bad = {k: v for k, v in report.items() if v["min_cosine"] < 0.999}
    print("selfcheck", json.dumps(report), flush=True)
    if bad:
        raise SystemExit(f"selfcheck failed: {bad}")
    return report


def load_encoders() -> dict:
    from go_aicup.tasks.identity.seqagg import load_head
    from scripts.identity.pid2_seqagg import load_encoder

    encoders = {}
    for name, (run, head_file) in ENCODERS.items():
        model, config, device = load_encoder(run)
        if config.get("sampling", {}).get("min_ply", 20) != 0:
            raise ValueError(f"{run}: ply positions assume min_ply 0")
        head, _ = load_head(head_file, device)
        encoders[name] = (model, config, head, device)
    return encoders


def run_selfcheck(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selfcheck.json").write_text(json.dumps(selfcheck(ol.load_train(), load_encoders()), indent=1))


def embed(args) -> None:
    import torch
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity import twocolor

    VEC.mkdir(parents=True, exist_ok=True)
    meta = ol.load_train()
    tag = f"{args.shard}of{args.shards}"
    if not (OUT / "selfcheck.json").exists():
        raise SystemExit("run `selfcheck` first")
    encoders = load_encoders()

    t0 = time.time()
    for part in PARTS:
        cache_dir, games, colour, length, rows = part_inputs(part, meta)
        idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
        np.save(VEC / f"{part}_rows_{tag}.npy", idx)
        for name, (model, config, head, device) in encoders.items():
            path = VEC / f"{part}_{name}_{tag}.npy"
            if path.exists():
                continue
            vec = head_game_vectors(model, config, head, device, cache_dir, games[idx], colour[idx], length[idx])
            if not np.isfinite(vec).all():
                raise ValueError(f"{part} {name}: non-finite vectors")
            np.save(path.with_suffix(".tmp.npy"), vec)
            path.with_suffix(".tmp.npy").replace(path)
            print(f"shard {tag} {part} {name}: {len(idx)} games, {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    path = VEC / f"ref_old_{tag}.npy"
    if not path.exists():
        model, config, device, min_ply = ol.load_model()
        ref = pd.read_parquet(OUT / "ref.parquet")
        games = np.unique(ref["game_index"].to_numpy())
        idx = np.array_split(np.arange(len(games)), args.shards)[args.shard]
        g = games[idx]
        vec = twocolor.embed_games(model, config, BoardCache(data.CACHE_DIR), g,
                                   meta["player_color"].to_numpy()[g].astype(np.int8), meta["length"].to_numpy()[g],
                                   device, swap=True, min_ply=min_ply)
        np.save(VEC / f"ref_old_rows_{tag}.npy", idx)
        np.save(path, vec)
    print(f"shard {tag} done, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- analyze helpers
def load_vectors(part: str, shards: int, n_cache: int, rows) -> dict[str, np.ndarray]:
    """unit game vectors per encoder, in cache order for train, in store order otherwise."""
    out = {}
    for name in ENCODERS:
        vec = np.empty((n_cache, 256), np.float64)
        seen = np.zeros(n_cache, bool)
        for s in range(shards):
            idx = np.load(VEC / f"{part}_rows_{s}of{shards}.npy")
            vec[idx] = np.load(VEC / f"{part}_{name}_{s}of{shards}.npy")
            seen[idx] = True
        if not seen.all():
            raise ValueError(f"{part} {name}: missing shards")
        vec = ol.unit(vec)
        if rows is not None:
            store = np.empty_like(vec)
            store[rows] = vec
            vec = store
        out[name] = vec
    return out


def sums_by(vectors: dict, owner: np.ndarray, n: int) -> dict:
    out = {}
    for name, v in vectors.items():
        s = np.zeros((n, v.shape[1]))
        np.add.at(s, owner, v)
        out[name] = s
    return out


def unit_rows(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(norm > 0, x / np.where(norm > 0, norm, 1), 0.0)


def question_vectors(vectors: dict, groups: list[np.ndarray]) -> dict:
    return {name: unit_rows(np.stack([v[g].sum(0) for g in groups])) for name, v in vectors.items()}


def style(qv: dict, cent: dict) -> np.ndarray:
    return sum(WEIGHTS[name] * (qv[name] @ cent[name].T) for name in ENCODERS)


def own_column(qv: dict, sums: dict, owner: np.ndarray, removed: list[np.ndarray] | None, vectors: dict | None):
    """style score of each question against its own candidate, with `removed` game rows taken out of the centroid."""
    total = 0.0
    for name in ENCODERS:
        s = sums[name][owner].copy()
        if removed is not None:
            s -= np.stack([vectors[name][r].sum(0) for r in removed])
        total = total + WEIGHTS[name] * (qv[name] * unit_rows(s)).sum(1)
    return total


def top2(scores: np.ndarray):
    return ol.top2(scores)


def k_thresholds(gap: np.ndarray, ks: np.ndarray) -> dict:
    return {int(k): float(gap[ks == k].max()) for k in K_BUCKETS}


def above(gap: np.ndarray, ks: np.ndarray, thresholds: dict) -> np.ndarray:
    return gap > np.array([thresholds[int(k)] for k in ks])


# ---------------------------------------------------------------- (a) training players
def reduce(o: np.ndarray, sims: dict, scales: dict) -> dict:
    out = {"arg_o": o.argmax(1)}
    for fam, sim in sims.items():
        arg, _, gap = top2(sim)
        out[f"arg_{fam}_style"], out[f"gap_{fam}_style"] = arg, gap
        for lam in LAMBDAS:
            arg, _, gap = top2(o + lam * scales[fam] * sim)
            out[f"arg_{fam}_{lam}"], out[f"gap_{fam}_{lam}"] = arg, gap
    return out


def arm_names(fam: str) -> list[str]:
    return [f"{fam}_{lam}" for lam in LAMBDAS] + [f"{fam}_style"]


def rule(table: pd.DataFrame, prefix: str, arm: str, fam: str, thresholds: dict, agree: bool):
    ks = table["k"].to_numpy()
    fused = above(table[f"{prefix}gap_{arm}"].to_numpy(), ks, thresholds)
    same = table[f"{prefix}arg_o"].to_numpy() == table[f"{prefix}arg_{fam}_style"].to_numpy()
    match = np.where(fused, table[f"{prefix}arg_{arm}"].to_numpy(), table[f"{prefix}arg_{fam}_style"].to_numpy())
    return (fused | same) if agree else fused, match, fused, same


def decide_family(ref: pd.DataFrame, test: pd.DataFrame, fam: str) -> tuple[dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    players = np.array(sorted(ref["player_id"].unique()))
    half_a = set(players[rng.permutation(len(players))[:len(players) // 2]])
    a = ref[ref["player_id"].isin(half_a)].reset_index(drop=True)
    b = ref[~ref["player_id"].isin(half_a)].reset_index(drop=True)
    th = lambda t, arm: k_thresholds(t[f"null_gap_{arm}"].to_numpy(), t["k"].to_numpy())
    result = {"seen_top1": {arm: {str(k): float((g[f"seen_arg_{arm}"] == g["truth"]).mean()) for k, g in ref.groupby("k")}
                            for arm in ("o", f"{fam}_style", *arm_names(fam)[:-1])}, "selection": {}}
    best = None
    for arm in arm_names(fam):
        flag, match, _, _ = rule(a, "seen_", arm, fam, th(a, arm), False)
        recall = float((flag & (match == a["truth"].to_numpy())).mean())
        result["selection"][arm] = recall
        if best is None or recall > best[1]:
            best = (arm, recall)
    arm = best[0]
    agree_a = float((a["null_arg_o"] == a[f"null_arg_{fam}_style"]).mean())
    use_agree = agree_a <= AGREE_FP
    result.update({"arm": arm, "agree_null_rate_a": agree_a, "use_agree": bool(use_agree)})
    t_a = th(a, arm)
    null_flag, _, null_fused, null_same = rule(b, "null_", arm, fam, t_a, use_agree)
    seen_flag, seen_match, _, _ = rule(b, "seen_", arm, fam, t_a, use_agree)
    right = seen_match == b["truth"].to_numpy()
    kb = b["k"].to_numpy()
    result["half_b"] = {
        "false_positive_rate": float(null_flag.mean()), "false_positives": int(null_flag.sum()),
        "false_positives_fused": int(null_fused.sum()), "false_positives_agree": int(null_same.sum()),
        "expected_false_positives_in_800": round(800 * float(null_flag.mean()), 2),
        "recall": float((seen_flag & right).mean()),
        "recall_by_k": {str(k): float((seen_flag & right)[kb == k].mean()) for k in K_BUCKETS},
        "precision_seen": float(right[seen_flag].mean()) if seen_flag.any() else None}
    t_all = th(ref, arm)
    flag, match, fused, same = rule(test, "", arm, fam, t_all, use_agree)
    result["test_thresholds"] = t_all
    result["test_flagged"] = int(flag.sum())
    result["implied_training_share"] = round(float(flag.mean()) / max(result["half_b"]["recall"], 1e-9), 4)
    return result, flag, match


def analyze_training(meta, train_vec, train_pos, ref, test_parts, old_ref) -> dict:
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()

    candidates, column, cent_old, cand_rank, _ = wl.centroids_and_ranks(meta)
    owner = column.loc[player[gal]].to_numpy()
    sums = sums_by({n: v[train_pos(gal)] for n, v in train_vec.items()}, owner, len(candidates))
    cent_new = {n: unit_rows(s) for n, s in sums.items()}
    scales = {"old": float(json.loads(OLD_W.read_text())["w"]), "new": float(np.load(NEW_W)["w"])}

    train_pre = data.opening_prefix_d4(meta, PLIES)
    blocks, offsets, size = [train_pre], {}, len(train_pre)
    for name, (q_meta, _, _) in test_parts.items():
        moves, colors, _ = load_store(name, ol.STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        if not ok.all():
            raise ValueError(f"{name}: {int((~ok).sum())} games too short for the opening model")
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
    prefix = np.concatenate(blocks)
    hashes = opening.prefix_hashes(prefix)
    held = ref["game_index"].to_numpy()

    ref_rows, test_rows = [], []
    for color in (0, 1):
        own = (pcolor == color) & hd01
        keep = own.copy()
        keep[held] = False
        galleries = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            galleries[column[p]] = rows.to_numpy()
        population = np.flatnonzero(own)

        part = ref[ref["color"] == color]
        qs = [(pid, k, g["game_index"].to_numpy()[np.argsort(g["order"].to_numpy())][:k])
              for pid, g in part.groupby("player_id", sort=True) for k in K_BUCKETS]
        queries = [q[2] for q in qs]
        truth = column.loc[[q[0] for q in qs]].to_numpy()
        o = score_questions(prefix, hashes, galleries, queries, population, color, params)
        qv = question_vectors({n: v for n, v in train_vec.items()}, [train_pos(q) for q in queries])
        s_new = style(qv, cent_new)
        qo = unit_rows(np.stack([old_ref(q).sum(0) for q in queries]))
        g_old = qo @ cent_old.T
        seen = reduce(o, {"old": g_old, "new": s_new}, scales)
        rows = np.arange(len(truth))
        for x in (o, s_new, g_old):
            x[rows, truth] = -np.inf
        null = reduce(o, {"old": g_old, "new": s_new}, scales)
        frame = pd.DataFrame({"player_id": [q[0] for q in qs], "k": [q[1] for q in qs], "color": color,
                              "truth": truth})
        for key, value in seen.items():
            frame[f"seen_{key}"] = value
        for key, value in null.items():
            frame[f"null_{key}"] = value
        ref_rows.append(frame)
        print(f"(a) colour {color}: {len(qs)} reference questions", flush=True)

        for name, (q_meta, vec, old_vec) in test_parts.items():
            index = np.flatnonzero(q_meta["color"].astype(str).to_numpy() == ("W" if color else "B"))
            if not len(index):
                continue
            groups = q_meta.iloc[index].groupby("question_id", sort=False)
            names = list(groups.groups)
            members = [index[groups.indices[q]] for q in names]
            o = score_questions(prefix, hashes, galleries, [offsets[name] + m for m in members], population, color,
                                params)
            s_new = style(question_vectors(vec, members), cent_new)
            g_old = unit_rows(np.stack([old_vec[m].sum(0) for m in members])) @ cent_old.T
            frame = pd.DataFrame({"part": name, "question_id": names, "color": color,
                                  "k": groups["num_games"].first().loc[names].to_numpy()})
            for key, value in reduce(o, {"old": g_old, "new": s_new}, scales).items():
                frame[key] = value
            test_rows.append(frame)
            print(f"(a) colour {color}: {name} {len(names)} questions", flush=True)

    ref_t, test_t = pd.concat(ref_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)
    ref_t.to_parquet(OUT / "a_ref_scores.parquet", index=False)
    test_t.to_parquet(OUT / "a_test_scores.parquet", index=False)
    return finish_training(ref_t, test_t, candidates, cand_rank, scales)


def finish_training(ref_t: pd.DataFrame, test_t: pd.DataFrame, candidates, cand_rank, scales) -> dict:
    result = {"scales": scales}
    for fam in ("old", "new"):
        res, flag, match = decide_family(ref_t, test_t, fam)
        result[fam] = res
        links = test_t.loc[flag, ["part", "question_id", "k", "color"]].copy()
        links["player"] = candidates[match[flag]]
        links["matched_rank"] = [wl.RANKS[r] for r in cand_rank[match[flag]]]
        links["gap"] = test_t.loc[flag, f"gap_{res['arm']}"].to_numpy()
        links.to_csv(OUT / f"a_links_{fam}.csv", index=False)
        result[fam]["links"] = links.drop(columns=["color"]).to_dict("records")
    return result


# ---------------------------------------------------------------- (b) competition-1 pools
def analyze_pools(ref_qv: dict, ref_k: np.ndarray, test_qv: dict, test_info: pd.DataFrame, pool_vec: dict,
                  pool_meta: dict) -> dict:
    pool_ref = pd.read_parquet(OUT / "pool_ref.parquet")
    result = {}
    for gname, parts in POOLS.items():
        vec = {n: np.concatenate([pool_vec[p][n] for p in parts]) for n in ENCODERS}
        start = np.cumsum([0] + [len(pool_meta[p]) for p in parts])
        ids = np.concatenate([pool_meta[p]["player_id"].astype(str).to_numpy() for p in parts])
        candidates = np.array(sorted(set(ids)))
        column = pd.Series(np.arange(len(candidates)), index=candidates)
        owner = column.loc[ids].to_numpy()
        n_games = np.bincount(owner, minlength=len(candidates))
        sums = sums_by(vec, owner, len(candidates))
        cent = {n: unit_rows(s) for n, s in sums.items()}

        _, _, null_gap = top2(style(ref_qv, cent))
        thresholds = k_thresholds(null_gap, ref_k)

        sim = pool_ref[pool_ref["part"].isin(parts)]
        qs = []
        for (p, pid), g in sim.groupby(["part", "player_id"], sort=True):
            rows = start[parts.index(p)] + g["store_row"].to_numpy()[np.argsort(g["order"].to_numpy())]
            for k in K_BUCKETS:
                if k <= len(rows) and k < n_games[column[pid]]:
                    qs.append((column[pid], k, rows[:k]))
        truth = np.array([q[0] for q in qs])
        sk = np.array([q[1] for q in qs])
        qv = question_vectors(vec, [q[2] for q in qs])
        scores = style(qv, cent)
        scores[np.arange(len(qs)), truth] = own_column(qv, sums, truth, [q[2] for q in qs], vec)
        arg, _, gap = top2(scores)
        hit = above(gap, sk, thresholds) & (arg == truth)
        recall = {str(k): (float(hit[sk == k].mean()) if (sk == k).any() else None) for k in K_BUCKETS}
        top1 = {str(k): (float((arg == truth)[sk == k].mean()) if (sk == k).any() else None) for k in K_BUCKETS}

        t_scores = style(test_qv, cent)
        t_arg, t_top, t_gap = top2(t_scores)
        tk = test_info["k"].to_numpy()
        flag = above(t_gap, tk, thresholds)
        rate = {str(k): float(flag[tk == k].mean()) for k in K_BUCKETS}
        flagged = test_info.loc[flag, ["question_id", "k", "color"]].assign(
            candidate=candidates[t_arg[flag]], gap=t_gap[flag], top1=t_top[flag])
        flagged.to_csv(OUT / f"b_links_{gname}.csv", index=False)
        result[gname] = {"candidates": int(len(candidates)), "null_thresholds": thresholds,
                         "null_median_gap": {str(k): float(np.median(null_gap[ref_k == k])) for k in K_BUCKETS},
                         "simulation_top1": top1, "simulation_recall_at_null_max": recall,
                         "test_median_gap": {str(k): float(np.median(t_gap[tk == k])) for k in K_BUCKETS},
                         "test_flagged": int(flag.sum()), "test_flag_rate_by_k": rate,
                         "test_flagged_by_colour": flagged.groupby("color").size().to_dict(),
                         "flagged": flagged.to_dict("records")}
        print(f"(b) {gname}: test flagged {int(flag.sum())}", flush=True)
    return result


# ---------------------------------------------------------------- (c) pairs among test questions
def pair_scores(qa: dict, qb: dict) -> np.ndarray:
    return sum(WEIGHTS[n] * (qa[n] @ qb[n].T) for n in ENCODERS)


def analyze_pairs(ref_qv, ref_k, ref_player, seen_pairs, test_qv, test_info, base) -> dict:
    null = pair_scores(ref_qv, ref_qv)
    same = ref_player[:, None] == ref_player[None, :]
    test = pair_scores(test_qv, test_qv)
    tk = test_info["k"].to_numpy()
    iu = np.triu_indices(len(tk), 1)
    result = {"cells": {}}
    listed = []
    for i, k1 in enumerate(K_BUCKETS):
        for k2 in K_BUCKETS[i:]:
            m = (~same) & (ref_k[:, None] == k1) & (ref_k[None, :] == k2)
            values = null[m]
            q9999, vmax = float(np.quantile(values, 0.9999)), float(values.max())
            cell = ((tk[iu[0]] == k1) & (tk[iu[1]] == k2)) | ((tk[iu[0]] == k2) & (tk[iu[1]] == k1))
            tv = test[iu][cell]
            entry = {"null_pairs": int(m.sum()), "null_q9999": q9999, "null_max": vmax, "test_pairs": int(cell.sum()),
                     "test_above_q9999": int((tv > q9999).sum()),
                     "expected_above_q9999": round(1e-4 * int(cell.sum()), 2), "test_above_max": int((tv > vmax).sum())}
            if (k1, k2) in seen_pairs:
                entry["seen_recall_at_max"] = float((seen_pairs[(k1, k2)] > vmax).mean())
                entry["seen_recall_at_q9999"] = float((seen_pairs[(k1, k2)] > q9999).mean())
            result["cells"][f"{k1}-{k2}"] = entry
            hits = np.flatnonzero(cell)[tv > vmax]
            for h in hits:
                a, b = iu[0][h], iu[1][h]
                qa, qb = test_info["question_id"].iloc[a], test_info["question_id"].iloc[b]
                listed.append({"q1": qa, "q2": qb, "k1": int(tk[a]), "k2": int(tk[b]), "score": float(test[a, b]),
                               "base1": base.get(qa), "base2": base.get(qb)})
    result["above_max"] = listed
    pd.DataFrame(listed).to_csv(OUT / "c_pairs.csv", index=False)
    print(f"(c) pairs above null max: {len(listed)}", flush=True)
    return result


# ---------------------------------------------------------------- analyze
def analyze(args) -> None:
    meta = ol.load_train()
    shards = args.shards
    train_games = np.load(OUT / "train_games.npy")
    train_vec = load_vectors("train", shards, len(train_games), None)
    train_pos = lambda games: np.searchsorted(train_games, games)
    ref = pd.read_parquet(OUT / "ref.parquet")
    if not np.isin(ref["game_index"], train_games).all():
        raise ValueError("reference games missing from the train vectors")

    ref_games = np.unique(ref["game_index"].to_numpy())
    old = np.empty((len(ref_games), 256))
    for s in range(shards):
        old[np.load(VEC / f"ref_old_rows_{s}of{shards}.npy")] = np.load(VEC / f"ref_old_{s}of{shards}.npy")
    old_by_game = pd.Series(np.arange(len(ref_games)), index=ref_games)
    old_unit = ol.unit(old)
    old_ref = lambda games: old_unit[old_by_game.loc[games].to_numpy()]
    test_parts = {}
    for name in ol.PARTS:
        _, _, q_meta = load_store(name, ol.STORE, mmap=False)
        cache_dir, games, _, _, rows = part_inputs(name, meta)
        vec = load_vectors(name, shards, len(games), rows)
        test_parts[name] = (q_meta, vec, ol.unit(np.load(ol.OUT / f"{name}_own.npy").astype(np.float64)))
    result = {"selfcheck": json.loads((OUT / "selfcheck.json").read_text())}

    # reference and test question vectors for (b) and (c)
    groups, ref_k, ref_player = [], [], []
    for pid, g in ref.groupby("player_id", sort=True):
        ordered = g["game_index"].to_numpy()[np.argsort(g["order"].to_numpy())]
        for k in K_BUCKETS:
            groups.append(train_pos(ordered[:k]))
            ref_k.append(k)
            ref_player.append(pid)
    ref_qv = question_vectors(train_vec, groups)
    ref_k, ref_player = np.array(ref_k), np.array(ref_player)
    seen_pairs = {}
    by_player = {pid: train_pos(g["game_index"].to_numpy()[np.argsort(g["order"].to_numpy())])
                 for pid, g in ref.groupby("player_id", sort=True)}
    for k, (a, b) in ((5, (slice(0, 5), slice(5, 10))), (10, (slice(0, 10), slice(10, 20)))):
        qa = question_vectors(train_vec, [r[a] for r in by_player.values()])
        qb = question_vectors(train_vec, [r[b] for r in by_player.values()])
        seen_pairs[(k, k)] = sum(WEIGHTS[n] * (qa[n] * qb[n]).sum(1) for n in ENCODERS)

    infos, members_all = [], {n: [] for n in ENCODERS}
    for name, (q_meta, vec, _) in test_parts.items():
        groups_q = q_meta.groupby("question_id", sort=False)
        names = list(groups_q.groups)
        qv = question_vectors(vec, [groups_q.indices[q] for q in names])
        for n in ENCODERS:
            members_all[n].append(qv[n])
        infos.append(pd.DataFrame({"part": name, "question_id": names,
                                   "k": groups_q["num_games"].first().loc[names].to_numpy(),
                                   "color": groups_q["color"].first().loc[names].astype(str).to_numpy()}))
    test_info = pd.concat(infos, ignore_index=True)
    test_qv = {n: np.concatenate(v) for n, v in members_all.items()}

    pool_vec, pool_meta = {}, {}
    for name in POOL_CACHE:
        _, _, m = load_store(name, PID_STORE, mmap=False)
        _, games, _, _, rows = part_inputs(name, meta)
        pool_vec[name] = load_vectors(name, shards, len(games), rows)
        pool_meta[name] = m
    result["b_pools"] = analyze_pools(ref_qv, ref_k, test_qv, test_info, pool_vec, pool_meta)

    base = pd.read_csv(ROOT / "submissions" / "rank" / args.base, dtype=str).set_index("question_id")["pred_rank"]
    result["c_pairs"] = analyze_pairs(ref_qv, ref_k, ref_player, seen_pairs, test_qv, test_info, base.to_dict())
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps({"b_pools": {g: {k: v for k, v in r.items() if k != "flagged"}
                                  for g, r in result["b_pools"].items()},
                      "c_pairs": result["c_pairs"]["cells"]}, indent=1, default=int), flush=True)
    result["a_training"] = analyze_training(meta, train_vec, train_pos, ref, test_parts, old_ref)
    write_final(result)


def decide_only(_args) -> None:
    candidates, _, _, cand_rank, _ = wl.centroids_and_ranks(ol.load_train())
    result = json.loads((OUT / "result.json").read_text())
    scales = {"old": float(json.loads(OLD_W.read_text())["w"]), "new": float(np.load(NEW_W)["w"])}
    result["a_training"] = finish_training(pd.read_parquet(OUT / "a_ref_scores.parquet"),
                                           pd.read_parquet(OUT / "a_test_scores.parquet"), candidates, cand_rank, scales)
    write_final(result)


def write_final(result: dict) -> None:
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps({"a_training": {f: {k: v for k, v in result["a_training"][f].items() if k != "links"}
                                     for f in ("old", "new")},
                      "a_links_new": result["a_training"]["new"]["links"],
                      "b_pools": {g: {k: v for k, v in r.items() if k != "flagged"} for g, r in result["b_pools"].items()},
                      "c_pairs": result["c_pairs"]["cells"]}, indent=1, default=int))


# ---------------------------------------------------------------- export
def export(args) -> None:
    from go_aicup.tasks.strength.submission_files import write_rank_csv

    base = pd.read_csv(ROOT / "submissions" / "rank" / args.base, dtype=str).set_index("question_id")["pred_rank"]
    links = pd.read_csv(OUT / "a_links_new.csv").set_index("question_id")
    l10 = pd.read_csv(ROOT / "outputs" / "rank" / "l10_real_test.questions.csv")
    old_l10 = set(l10.loc[l10["above_null_max"], "question_id"])
    result = base.copy()
    result.loc[links.index] = links["matched_rank"]
    changed = result != base
    detail = links.assign(before=base.loc[links.index], changed=changed.loc[links.index],
                          in_old_l10=links.index.isin(old_l10)).reset_index()
    detail["rank_diff"] = (detail["matched_rank"].map(wl.RANKS.index) - detail["before"].map(wl.RANKS.index)).abs()
    current = pd.read_csv(ROOT / "submissions" / "rank" / args.current, dtype=str).set_index("question_id")["pred_rank"]
    out = {"base": args.base, "links": int(len(links)), "old_l10_questions": sorted(old_l10),
           "old_l10_also_linked": int(detail["in_old_l10"].sum()),
           "same_as_base": float((~detail["changed"]).mean()) if len(detail) else None,
           "changed_public": int(changed[changed.index.str.startswith("pub")].sum()),
           "changed_private": int(changed[changed.index.str.startswith("priv")].sum()),
           "differs_from_current": {"file": args.current,
                                    "public": int((result != current)[result.index.str.startswith("pub")].sum()),
                                    "private": int((result != current)[result.index.str.startswith("priv")].sum())}}
    half_b = json.loads((OUT / "result.json").read_text())["a_training"]["new"]["half_b"]
    gate = (half_b["expected_false_positives_in_800"] <= 1.5 and half_b["precision_seen"] is not None
            and half_b["precision_seen"] >= 0.9)
    out["gate"] = {"expected_false_positives_in_800": half_b["expected_false_positives_in_800"],
                   "precision_seen": half_b["precision_seen"], "pass": bool(gate)}
    if changed.any() and gate:
        out["file"] = write_rank_csv(result.rename("pred_rank").reset_index(), args.tag, args.job).name
    detail.to_csv(OUT / f"export_{args.job}-{args.tag}.csv", index=False)
    (OUT / f"export_{args.job}-{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(detail.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=plan)
    sub.add_parser("selfcheck").set_defaults(fn=run_selfcheck)
    sub.add_parser("decide").set_defaults(fn=decide_only)
    p = sub.add_parser("embed")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.set_defaults(fn=embed)
    p = sub.add_parser("analyze")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--base", default="751959-cvec2.csv")
    p.set_defaults(fn=analyze)
    p = sub.add_parser("export")
    p.add_argument("--job", required=True)
    p.add_argument("--base", default="751959-cvec2.csv")
    p.add_argument("--current", default="751959-cv2-l10.csv")
    p.add_argument("--tag", default="l10v2")
    p.set_defaults(fn=export)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
