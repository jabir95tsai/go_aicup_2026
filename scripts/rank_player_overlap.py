"""Competition-2 test: do players repeat across questions, or come from the training set?

Read-only check (RESULTS 2026-11-08), CPU only, no labels of the test set used.

Fingerprint per game: tokens from the D4-canonical first 60 plies (identity/data.py
canonical_d4): the player's own-move sequences of length 1-6, own move i (i = 1-12)
and the joint sequence of length 1-6. A question / profile is the sum of its games'
tokens, TF-IDF weighted (idf per colour over training games), cosine similarity.
Only HD<=1 training games are used (HD>=2 move labels are reversed, CLAUDE.md §1).

Check 1 (repeat players within a part): every same-colour question pair in the
Public and in the Private part (never mixed, Q&A Q10). Null = simulated parts of
400 questions drawn from distinct training players with the test's k / colour mix;
planted = the same with 100 players giving two disjoint questions. Threshold =
null pair-similarity quantile 1 - 1/pairs (about one false pair per part).

Check 2 (test players inside the training set): top-1 similarity of a question to
the 8,007 x colour training profiles. "in" = simulated questions whose player stays in
the gallery (minus the question's games); "out" = the player removed. Threshold =
99th percentile of "out"; the test share above it, against 1% (out) and the "in"
share, estimates how many test players are in training.

    python scripts/rank_player_overlap.py --out outputs/rank/player_overlap
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from go_aicup.data.preprocess import load_store
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import data as idata

PLIES = 60
BASE = 362  # moves are 0..360, PAD 361
SEQ, SLOT, JOINT = 6, 12, 6
K_BUCKETS = (5, 10, 15, 20)
STORE = ROOT / "outputs" / "rank_test_ahc_713770" / "store"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
SIMS = 5


def game_keys(prefix: np.ndarray, parity: np.ndarray) -> np.ndarray:
    """int64 [n, SEQ + SLOT + JOINT] token keys (value * 64 + code)."""
    prefix = prefix.astype(np.int64)
    own = np.where(parity[:, None] == 0, prefix[:, 0::2], prefix[:, 1::2])
    cols = []
    h = np.zeros(len(prefix), np.int64)
    for d in range(SEQ):
        h = h * BASE + own[:, d]
        cols.append(h * 64 + (1 + d))
    for i in range(SLOT):
        cols.append((i * BASE + own[:, i]) * 64 + (10 + i))
    h = np.zeros(len(prefix), np.int64)
    for t in range(JOINT):
        h = h * BASE + prefix[:, t]
        cols.append(h * 64 + (30 + t))
    return np.stack(cols, axis=1)


def test_prefix(part: str) -> tuple[pd.DataFrame, np.ndarray]:
    moves, colors, meta = load_store(f"rank_{part}", STORE, mmap=False)
    if (meta["length"] < PLIES).any() or not meta["alternating"].all() or (meta["first_color"] != 0).any():
        raise ValueError(f"{part}: short or non-alternating game")
    prefix = np.stack([moves[o:o + PLIES] for o in meta["offset"]])
    return meta.reset_index(drop=True), idata.canonical_d4(prefix)


def matrix(keys: np.ndarray, vocab: pd.Index) -> sp.csr_matrix:
    cols = vocab.get_indexer(keys.ravel())
    rows = np.repeat(np.arange(len(keys)), keys.shape[1])
    keep = cols >= 0
    return sp.csr_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(len(keys), len(vocab)))


def normalise(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    return sp.diags(1 / np.maximum(norm, 1e-12)) @ m


def summed(games: sp.csr_matrix, groups: list[np.ndarray]) -> sp.csr_matrix:
    rows = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    pick = sp.csr_matrix((np.ones(len(rows)), (rows, np.concatenate(groups))), shape=(len(groups), games.shape[0]))
    return pick @ games


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    return {q: round(float(np.quantile(x, float(q))), 4) for q in ("0.1", "0.5", "0.9", "0.99")} | {"n": int(len(x))}


def simulate_part(rng, eligible: dict, mix: pd.DataFrame, planted: int) -> list[dict]:
    """Questions {key, games} with the test's (k, colour) mix; `planted` keys give two disjoint questions."""
    slots = mix.sample(frac=1, random_state=int(rng.integers(1 << 31)))[["k", "color"]].to_numpy().tolist()
    out, used = [], set()
    pairs = slots[:2 * planted]
    for (k1, c1), (k2, c2) in zip(pairs[0::2], pairs[1::2]):
        if c1 != c2:
            k2, c2 = k1, c1
        pool = [key for key, g in eligible[c1].items() if len(g) >= k1 + k2 and key not in used]
        key = pool[rng.integers(len(pool))]
        used.add(key)
        g = rng.permutation(eligible[c1][key])
        out += [{"key": key, "color": c1, "k": k1, "games": g[:k1]}, {"key": key, "color": c1, "k": k2, "games": g[k1:k1 + k2]}]
    for k, c in slots[2 * planted:]:
        pool = [key for key, g in eligible[c].items() if len(g) >= k and key not in used]
        key = pool[rng.integers(len(pool))]
        used.add(key)
        out.append({"key": key, "color": c, "k": k, "games": rng.choice(eligible[c][key], k, replace=False)})
    return out


def pair_sims(qmat: sp.csr_matrix, colors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = (qmat @ qmat.T).toarray()
    i, j = np.triu_indices(len(colors), 1)
    same = colors[i] == colors[j]
    return sim[i[same], j[same]], i[same], j[same]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2027)

    meta = idata.load_meta()
    hd = pd.read_csv(HANDICAP, usecols=["rank", "game_id", "handicap"])
    meta = meta.merge(hd, on=["rank", "game_id"], how="left", validate="many_to_one")
    if meta["handicap"].isna().any():
        raise ValueError("handicap join incomplete")
    prefix = idata.opening_prefix_d4(meta, PLIES)
    keep = (meta["handicap"] <= 1).to_numpy()
    meta, prefix = meta[keep].reset_index(drop=True), prefix[keep]
    parity = meta["player_color"].to_numpy()
    train_keys = game_keys(prefix, parity)
    print(f"training games HD<=1: {len(meta):,}", flush=True)

    tests = {}
    for part in ("public", "private"):
        tmeta, tprefix = test_prefix(part)
        tparity = (tmeta["color"] == "W").to_numpy().astype(int)
        tests[part] = (tmeta, game_keys(tprefix, tparity))
    vocab = pd.Index(np.unique(train_keys))
    train = matrix(train_keys, vocab)
    color_name = np.where(parity == 0, "B", "W")
    idf = {}
    for c in ("B", "W"):
        df = np.asarray((train[color_name == c] > 0).sum(axis=0)).ravel()
        idf[c] = np.log((1 + (color_name == c).sum()) / (1 + df))

    # profiles: (player, colour) over HD<=1 games
    groups = meta.groupby(["player_id", color_name]).indices
    keys = list(groups)
    profile_raw = summed(train, [groups[k] for k in keys])
    profile_colors = np.array([k[1] for k in keys])
    profile = {c: normalise(profile_raw[profile_colors == c].multiply(idf[c])) for c in ("B", "W")}
    profile_keys = {c: [k for k in keys if k[1] == c] for c in ("B", "W")}
    eligible = {c: {k: groups[k] for k in profile_keys[c]} for c in ("B", "W")}
    report = {"training_games_hd01": int(len(meta)), "profiles": {c: len(profile_keys[c]) for c in ("B", "W")}}

    def question_matrix(games_mat, qs, colors):
        raw = summed(games_mat, [np.asarray(q) for q in qs])
        return sp.vstack([normalise(raw[i].multiply(idf[c])) for i, c in enumerate(colors)])

    # ---------------- test questions
    test_q = {}
    for part, (tmeta, tkeys) in tests.items():
        tm = matrix(tkeys, vocab)
        qs = tmeta.groupby("question_id", sort=True)
        rows = [np.asarray(idx) for _, idx in qs.indices.items()]
        info = qs.agg(color=("color", "first"), k=("num_games", "first"))
        info["k"] = info["k"].astype(int)
        test_q[part] = (info, question_matrix(tm, rows, info["color"].to_numpy()))
    mix = test_q["public"][0].reset_index(drop=True)

    # ---------------- check 1: repeat players within a part
    check1 = {}
    null_sims, planted_hits, planted_false, null_hits = [], [], [], []
    for s in range(SIMS):
        for planted in (0, 100):
            qs = simulate_part(rng, eligible, mix, planted)
            colors = np.array([q["color"] for q in qs])
            qm = question_matrix(train, [q["games"] for q in qs], colors)
            sims, i, j = pair_sims(qm, colors)
            same = np.array([qs[a]["key"] == qs[b]["key"] for a, b in zip(i, j)])
            if planted == 0:
                null_sims.append(sims)
            else:
                planted_hits.append((sims, same))
    pooled_null = np.concatenate(null_sims)
    n_pairs = int(np.mean([len(x) for x in null_sims]))
    tau = float(np.quantile(pooled_null, 1 - 1 / n_pairs))
    check1["threshold"] = round(tau, 4)
    check1["pairs_per_part"] = n_pairs
    check1["null_pairs_above"] = [int((x > tau).sum()) for x in null_sims]
    check1["planted_true_pairs_above_of_100"] = [int((s[m] > tau).sum()) for s, m in planted_hits]
    check1["planted_false_pairs_above"] = [int((s[~m] > tau).sum()) for s, m in planted_hits]
    check1["null_pair_sim"] = quantiles(pooled_null)
    check1["planted_same_player_sim"] = quantiles(np.concatenate([s[m] for s, m in planted_hits]))
    for part, (info, qm) in test_q.items():
        sims, i, j = pair_sims(qm, info["color"].to_numpy())
        top = np.argsort(-sims)[:20]
        check1[part] = {"pairs_above": int((sims > tau).sum()), "pair_sim": quantiles(sims),
                        "top_pairs": [[info.index[i[t]], info.index[j[t]], int(info["k"].iloc[i[t]]),
                                       int(info["k"].iloc[j[t]]), round(float(sims[t]), 4)] for t in top]}
    report["check1_repeat_within_part"] = check1
    print(json.dumps(check1), flush=True)

    # ---------------- check 2: test players in the training set
    def top1(qm, colors, own=None, own_minus=None):
        best = np.empty(qm.shape[0])
        correct = np.zeros(qm.shape[0], bool)
        for c in ("B", "W"):
            m = np.flatnonzero(colors == c)
            if not len(m):
                continue
            sim = (qm[m] @ profile[c].T).toarray()
            if own is not None:
                index = {k: n for n, k in enumerate(profile_keys[c])}
                for r, q in enumerate(m):
                    col = index[own[q]]
                    if own_minus is None:
                        sim[r, col] = -np.inf
                    else:
                        reduced = normalise(own_minus[q].multiply(idf[c]))
                        sim[r, col] = float((qm[q] @ reduced.T).toarray()[0, 0])
                        correct[q] = sim[r].argmax() == col
            best[m] = sim.max(axis=1)
        return best, correct

    qs = []
    for s in range(2):
        qs += simulate_part(rng, eligible, mix, 0)
    colors = np.array([q["color"] for q in qs])
    qm = question_matrix(train, [q["games"] for q in qs], colors)
    owners = [q["key"] for q in qs]
    row_of = {k: n for n, k in enumerate(keys)}
    minus = [profile_raw[row_of[q["key"]]] - summed(train, [np.asarray(q["games"])]) for q in qs]
    out_best, _ = top1(qm, colors, own=owners)
    in_best, in_correct = top1(qm, colors, own=owners, own_minus=minus)
    tau2 = float(np.quantile(out_best, 0.99))
    ks = np.array([q["k"] for q in qs])
    check2 = {"threshold_out_p99": round(tau2, 4),
              "out_top1": quantiles(out_best), "in_top1": quantiles(in_best),
              "in_top1_correct": round(float(in_correct.mean()), 4),
              "in_top1_correct_by_k": {str(k): round(float(in_correct[ks == k].mean()), 4) for k in K_BUCKETS},
              "in_share_above": round(float((in_best > tau2).mean()), 4)}
    for part, (info, tq) in test_q.items():
        best, _ = top1(tq, info["color"].to_numpy())
        share = float((best > tau2).mean())
        check2[part] = {"top1": quantiles(best), "share_above": round(share, 4),
                        "estimated_in_training": round((share - 0.01) / max(check2["in_share_above"] - 0.01, 1e-9), 3)}
    report["check2_players_in_training"] = check2
    print(json.dumps(check2), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
