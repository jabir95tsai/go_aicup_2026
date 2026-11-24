"""Pair opponent sides across games using frozen style and opening caches."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from scripts.strength import rank_opp_next as n

MIN_PAIRS, PRECISION_A, PRECISION_B = 20, 0.90, 0.85
BOOST = n.TAU * np.log(4.0)
PAIR_COLS = ["left", "right", "similarity"]


def query_vectors(q):
    link, l2 = n.link, n.link.l2
    meta = link.ol.load_train()
    train_games = np.load(l2.OUT / "train_games.npy")
    ref = q.index[q["set"] == "ref"].to_numpy()
    gi = q.loc[ref, "game_index"].to_numpy()
    pos = np.searchsorted(train_games, gi)
    if not np.array_equal(train_games[pos], gi):
        raise ValueError("reference vectors do not align with the query games")
    result = np.empty((len(q), 256 * len(l2.ENCODERS)), np.float32)

    def copy(rows, vectors, positions=slice(None)):
        for i, name in enumerate(l2.ENCODERS):
            result[rows, 256*i:256*(i+1)] = np.sqrt(l2.WEIGHTS[name]) * vectors[name][positions]

    copy(ref, l2.load_vectors("train", 4, len(train_games), None), pos)
    train = q.index[q["set"].isin(["s_opp", "u_opp"])].to_numpy()
    copy(train, link.load_opp_vectors("train", 4, len(train), None))
    for part in link.ol.PARTS:
        rows = q.index[q["part"] == part].to_numpy()
        _, _, _, _, store_rows = l2.part_inputs(part, meta)
        if not np.array_equal(q.loc[rows, "store_row"], np.arange(len(store_rows))):
            raise ValueError(f"{part}: vector rows do not align with the test store")
        copy(rows, link.load_opp_vectors(part, 4, len(store_rows), store_rows))
    if not np.allclose(np.linalg.norm(result, axis=1), 1.0, atol=2e-5):
        raise ValueError("missing or invalid opponent vector")
    return result


def signatures(cache):
    jobs = pd.read_parquet(n.llr.OUT / "jobs.parquet")
    jobs = jobs[jobs["id"] >= n.llr.QUERY_ID].sort_values("id")
    query = jobs["id"].to_numpy() - n.llr.QUERY_ID
    if not np.array_equal(query, cache["masked_rows"]):
        raise ValueError("policy jobs and opponent LLR rows differ")
    games = n.llr.job_games(jobs)
    two = np.array([moves[colours == keep][:2] for (moves, colours), keep in zip(games, jobs["keep"])], dtype=np.int16)
    if two.shape != (len(jobs), 2):
        raise ValueError("game with fewer than two opponent moves")
    canon = n.link.data.canonical_d4(two).astype(np.int32)
    sig = canon[:, 0] * 362 + canon[:, 1]
    sig[(two >= 361).any(1)] = -1
    hashes = np.array([int.from_bytes(hashlib.blake2b(moves.tobytes(), digest_size=8).digest(), "little", signed=True)
                       for moves, _ in games], dtype=np.int64)
    return pd.DataFrame({"opening": sig, "game_hash": hashes}, index=query)


def mutual_pairs(vectors, opening, hashes, device):
    import torch

    if len(vectors) < 2:
        return pd.DataFrame(columns=PAIR_COLS)
    x = torch.as_tensor(vectors, device=device)
    key = torch.as_tensor(opening, device=device)
    game = torch.as_tensor(hashes, device=device)
    best, score = [], []
    for start in range(0, len(x), 512):
        sl = slice(start, start + 512)
        affinity = x[sl] @ x.T
        match = (key[sl, None] == key) & (key[sl, None] >= 0) & (game[sl, None] != game)
        value, index = affinity.masked_fill_(~match, -torch.inf).max(dim=1)
        best.append(index.cpu().numpy())
        score.append(value.cpu().numpy())
    best, score = np.concatenate(best), np.concatenate(score)
    all_rows = np.arange(len(best))
    use = np.isfinite(score) & (all_rows < best) & (best[best] == all_rows)
    return pd.DataFrame({"left": all_rows[use], "right": best[use], "similarity": score[use]})


def proposals(pairs, ranking, colour, edges):
    idx, score = ranking
    gap = score[:, 0] - score[:, 1]
    q99 = np.where(colour == 0, edges[0][1], edges[1][1])
    a, b = (pairs[key].to_numpy(dtype=int) for key in ("left", "right"))
    safer = gap[a] >= gap[b]
    donor, target = np.where(safer, a, b), np.where(safer, b, a)
    candidate = idx[donor, 0]
    matches = idx[target] == candidate[:, None]
    passed = (gap[donor] >= q99[donor]) & (gap[target] < q99[target]) & matches.any(axis=1)
    return pairs.loc[passed].assign(donor=donor[passed], target=target[passed], candidate=candidate[passed],
                                    slot=matches[passed].argmax(axis=1))


def select_cutoff(transfer):
    ranked = transfer.sort_values("similarity", ascending=False)
    similarity = ranked["similarity"].to_numpy()
    count = np.arange(1, len(ranked) + 1)
    precise = ranked["correct"].to_numpy().cumsum() / count
    boundary = np.r_[similarity[:-1] != similarity[1:], True] if len(similarity) else np.empty(0, dtype=bool)
    approved = boundary & (count >= MIN_PAIRS) & (precise >= PRECISION_A)
    return float(similarity[approved][-1]) if approved.any() else None


def boost(ranking, transfer):
    idx, score = (x.copy() for x in ranking)
    to = transfer["target"].to_numpy(dtype=int)
    slot = transfer["slot"].to_numpy(dtype=int)
    score[to, slot] += BOOST
    order = np.argsort(-score[to], axis=1)
    idx[to] = np.take_along_axis(idx[to], order, axis=1)
    score[to] = np.take_along_axis(score[to], order, axis=1)
    return idx, score


def run(q, scores, previous, rows, edges, cache):
    import torch

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("cross-game matching needs a GPU")
    sig = signatures(cache)
    vectors = query_vectors(q)
    work = q.loc[rows.index].join(sig, validate="one_to_one").copy()
    if work[["opening", "game_hash"]].isna().any().any():
        raise ValueError("query without opening or game hash")
    ref = work["set"] == "ref"
    players = np.array(sorted(work.loc[ref, "owner"].unique()))
    half_a = set(np.random.default_rng(n.llr.SEED).permutation(players)[:len(players)//2])
    work["pool"] = work["set"]
    work.loc[ref, "pool"] = np.where(work.loc[ref, "owner"].isin(half_a), "ref_a", "ref_b")
    work.loc[work["set"] == "test", "pool"] = work.loc[work["set"] == "test", "part"]
    matches = []
    for (pool, _), sub in work.groupby(["pool", "qcolor"], sort=True):
        r = sub.index.to_numpy()
        match = mutual_pairs(vectors[r], sub["opening"].to_numpy(), sub["game_hash"].to_numpy(), device)
        for key in ("left", "right"):
            match[key] = r[match[key].to_numpy(dtype=int)]
        matches.append(match.assign(pool=pool))
    pairs = pd.concat(matches, ignore_index=True)
    print(f"mutual nearest pairs: {len(pairs)}", flush=True)
    colour = q["qcolor"].to_numpy()
    seen = proposals(pairs[pairs["pool"].isin(["ref_a", "ref_b"])], previous["seen"], colour, edges)
    owner = scores["owner_col"]
    target, donor = (seen[col].to_numpy(dtype=int) for col in ("target", "donor"))
    seen["correct"] = (seen["candidate"].to_numpy() == owner[target]) & (owner[donor] == owner[target])
    cutoff = select_cutoff(seen[seen["pool"] == "ref_a"])
    report = {"cuda": torch.cuda.get_device_name(device), "boost": float(BOOST), "similarity_cutoff": cutoff,
              "matched_pairs": {k: int(v) for k, v in pairs["pool"].value_counts().items()}}
    passed = cutoff is not None
    for pool in ("ref_a", "ref_b"):
        full = seen[seen["pool"] == pool]
        chosen = full[full["similarity"] >= cutoff] if cutoff is not None else full.iloc[:0]
        after, _ = boost(previous["seen"], chosen)
        changed = chosen["target"].to_numpy(dtype=int)
        report[pool] = {"proposed": len(full), "selected": len(chosen),
                        "precision_all": float(full["correct"].mean()) if len(full) else None,
                        "precision_selected": float(chosen["correct"].mean()) if len(chosen) else None,
                        "top1_correct_delta": int((after[changed, 0] == owner[changed]).sum()
                                                  - (previous["seen"][0][changed, 0] == owner[changed]).sum())}
        if pool == "ref_b":
            passed = passed and len(chosen) >= MIN_PAIRS and chosen["correct"].mean() >= PRECISION_B
    report["pass"] = bool(passed)
    if not passed:
        return None, report
    masked = proposals(pairs[pairs["similarity"] >= cutoff], previous["masked"], colour, edges)
    report["masked_transfers"] = {k: int(v) for k, v in masked["pool"].value_counts().items()}
    return {"masked": boost(previous["masked"], masked),
            "seen": boost(previous["seen"], seen[seen["similarity"] >= cutoff])}, report
