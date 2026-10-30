"""L10: are the competition-2 test players in the training set? (Day-0 item 8)

Every training player is a candidate; a question is scored with the
competition-1 opening model (pp+slot, D4, A-029 hyper-parameters) on the
question colour's own plies (black 0,2,...; white 1,3,...). Statistic per
question: gap = best minus second-best candidate log-likelihood.

Reference questions (Valid-U of the seed-2027 V2 split, 100 per k) give two
null distributions: "seen" with the true player's other games in the gallery,
"unseen" with the true player's column removed. Their query games are removed
from every gallery first. The test questions are scored against the full
training galleries; the report is the share of test questions whose gap
exceeds the 95th percentile of the unseen null (~5% if the test players are
new, near the seen share if they are training players).

    python scripts/l10_check.py --store-dir data/processed --names rank_public rank_private
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

V2_DIR = PROCESSED / "strength_v2" / "seed2027"
CHUNK = 800  # query games per scoring call (memory ~ CHUNK x 8,007 x 8 bytes per array)


def own_plies(color: int, moves: int) -> np.ndarray:
    return np.arange(color, color + 2 * moves, 2)


def score_questions(prefix, hashes, galleries, query_games, population, color, params) -> np.ndarray:
    """[n_questions, n_candidates] pp+slot log-likelihood, chunked over questions."""
    plies = own_plies(color, params["moves"])
    slot_plies = own_plies(color, params["slot_moves"])
    pop = opening.population_prior(prefix, population, plies)
    slot_pop = opening.population_prior(prefix, population, slot_plies)
    slot_logp = opening.slot_log_prob(prefix, galleries, slot_plies, slot_pop, params["alpha"])
    out = np.zeros((len(query_games), len(galleries)))
    start = 0
    while start < len(query_games):
        stop, size = start, 0
        while stop < len(query_games) and (size == 0 or size + len(query_games[stop]) <= CHUNK):
            size += len(query_games[stop])
            stop += 1
        chunk = query_games[start:stop]
        flat = np.concatenate(chunk)
        base = opening.population_conditional(hashes, prefix, population, flat, plies, pop)
        scores = opening.prefix_backoff(prefix, galleries, chunk, plies, pop, beta=params["beta"],
                                        hashes=hashes, base=base)
        slot = opening._game_slot_scores(slot_logp, prefix[flat][:, slot_plies].astype(np.int64))
        out[start:stop] = scores + opening._sum_by_question(slot, chunk)
        start = stop
    return out


def gap(scores: np.ndarray) -> np.ndarray:
    top = np.sort(scores, axis=1)[:, -2:]
    return top[:, 1] - top[:, 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", default=["rank_public", "rank_private"])
    parser.add_argument("--per-k", type=int, default=100, help="reference questions per k")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "release" / "day0_audit" / "l10.json")
    args = parser.parse_args()
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    meta = data.load_meta()
    train_pre = data.opening_prefix_d4(meta, PLIES)
    player = meta["player_id"].astype(str).to_numpy()
    color_of = (meta["color"].astype(str).to_numpy() == "W").astype(int)
    candidates = np.array(sorted(set(player)))
    cand_index = {p: i for i, p in enumerate(candidates)}

    unseen = pd.read_parquet(V2_DIR / "strength_v2_valid_unseen_questions.parquet")
    rng = np.random.default_rng(2027)
    reference = pd.concat([g.iloc[rng.permutation(len(g))[:args.per_k]] for _, g in unseen.groupby("k")])
    held_out = np.concatenate(list(reference["games"]))

    tests = {}
    for name in args.names:
        moves, colors, q_meta = load_store(name, args.store_dir, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        tests[name] = (pre, ok, q_meta)

    result = {"params": params, "reference_questions": len(reference)}
    gaps = {"seen": [], "unseen": []}
    test_gaps = {name: [] for name in tests}
    for color in (0, 1):
        own = color_of == color
        keep = own.copy()
        keep[held_out] = False
        galleries = [np.array([], np.int64)] * len(candidates)
        frame = pd.DataFrame({"p": player[keep], "g": np.flatnonzero(keep)})
        for p, rows in frame.groupby("p")["g"]:
            galleries[cand_index[p]] = rows.to_numpy()
        population = np.flatnonzero(own)
        # one prefix array: training rows, then each test store's rows
        blocks, offsets = [train_pre], {}
        size = len(train_pre)
        for name, (pre, _, _) in tests.items():
            offsets[name] = size
            blocks.append(pre)
            size += len(pre)
        prefix = np.concatenate(blocks)
        hashes = opening.prefix_hashes(prefix)

        ref = reference[reference["color"] == ("W" if color else "B")]
        if len(ref):
            s = score_questions(prefix, hashes, galleries, list(ref["games"]), population, color, params)
            truth = np.array([cand_index[p] for p in ref["player_id"].astype(str)])
            gaps["seen"].append(gap(s))
            s[np.arange(len(s)), truth] = -np.inf
            gaps["unseen"].append(gap(s))
            print(f"color {color}: {len(ref)} reference questions scored", flush=True)
        for name, (pre, ok, q_meta) in tests.items():
            in_color = q_meta["color"].astype(str).to_numpy() == ("W" if color else "B")
            rows = q_meta[in_color & ok].groupby("question_id", sort=False).indices
            if not rows:
                continue
            index = np.flatnonzero(in_color & ok)
            queries = [offsets[name] + index[r] for r in rows.values()]
            test_gaps[name].append(gap(score_questions(prefix, hashes, galleries, queries, population,
                                                       color, params)))
            print(f"color {color}: {name} {len(queries)} questions scored", flush=True)

    seen, null = np.concatenate(gaps["seen"]), np.concatenate(gaps["unseen"])
    threshold = float(np.quantile(null, 0.95))
    result["threshold_gap_null95"] = threshold
    result["reference"] = {"seen_share_above": float((seen > threshold).mean()),
                           "unseen_share_above": float((null > threshold).mean()),
                           "seen_median_gap": float(np.median(seen)), "unseen_median_gap": float(np.median(null))}
    for name, parts in test_gaps.items():
        values = np.concatenate(parts) if parts else np.array([])
        result[name] = {"questions": len(values), "share_above": float((values > threshold).mean()) if len(values) else None,
                        "median_gap": float(np.median(values)) if len(values) else None}
    print(json.dumps(result, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
