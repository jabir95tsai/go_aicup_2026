"""Competition-1 test pipeline: one pool + its questions -> opening scores -> L6 -> top-5.

Public and private are run separately (different 400 players; CLAUDE.md
section 2). Model = A-029 default: pp+slot with D4 canonicalisation,
hyper-parameters read from the split-0 selection in
outputs/pid_opening_v1_d4/summary.json, then Sinkhorn + Hungarian (L6).

Population priors use training black games of players NOT in the pool (the
A-029 protocol; on the real pool this is presumably every training player,
see Day-0 audit item 3). A game is used for the opening only if it has at
least MIN_PLIES plies and strictly alternates B, W, ... over them; a question
with no usable game scores 0 for every candidate (count printed).

With --insurance-dir (GE2E vectors from identity/insurance.py, A-035 / A-053)
the scores become opening + lambda * w * cosine before L6; a question without
a usable opening game is then ranked by the insurance alone.

Outputs in --out-dir: scores.npy (questions x candidates, scores fed to L6:
opening log-likelihoods, fused when --insurance-dir is given), candidates.npy,
predictions.parquet and a provisional submission CSV (format fixed on 11/04).

    python -m go_aicup.tasks.identity.predict --store-dir data/processed --pool pool_public \\
        --questions pid_public --out-dir outputs/pid_submit/public
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, data, opening

PLIES = 60
SUMMARY = {"d4": ROOT / "outputs" / "pid" / "opening" / "pid_opening_v1_d4" / "summary.json",
           "raw": ROOT / "outputs" / "pid" / "opening" / "pid_opening_v1" / "summary.json"}
PAD = np.int16(361)  # never a board point; only sits beyond a short game's end


def handicap_ok(meta: pd.DataFrame) -> np.ndarray:
    """False for HD>=2 games: their B/W labels are reversed and the handicap stones are
    missing (Q&A Q13), so the "black" plies would be the opponent's. Stores without an
    hd column (written before 2026-11-08) keep every game."""
    if "hd" not in meta:
        return np.ones(len(meta), bool)
    return meta["hd"].to_numpy() <= 1


def black_plies(moves: int) -> np.ndarray:
    return np.arange(0, 2 * moves, 2)


def store_prefix(moves, colors, meta: pd.DataFrame, orient: str, min_plies: int):
    """(int16 [n, PLIES] prefix, bool [n] usable) for every game of a store.

    Short games are canonicalised over the plies they have; plies past the end
    are PAD and lie beyond every scored ply of a usable game.
    """
    prefix = np.full((len(meta), PLIES), PAD, np.int16)
    usable = np.zeros(len(meta), bool)
    lengths = np.minimum(meta["length"].to_numpy(), PLIES)
    offsets = meta["offset"].to_numpy()
    expected = np.arange(PLIES) % 2
    for i, (offset, length) in enumerate(zip(offsets, lengths)):
        prefix[i, :length] = moves[offset:offset + length]
        usable[i] = length >= min_plies and np.array_equal(
            colors[offset:offset + min_plies], expected[:min_plies])
    if orient == "d4":
        for length in np.unique(lengths):
            rows = np.flatnonzero(lengths == length)
            if length:
                prefix[rows, :length] = data.canonical_d4(prefix[rows, :length])
    return prefix, usable


def train_prefix(orient: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta = data.load_meta()
    prefix = data.opening_prefix_d4(meta, PLIES) if orient == "d4" else data.opening_prefix(meta, PLIES)
    return meta, prefix


def load_params(orient: str) -> dict:
    summary = json.loads(SUMMARY[orient].read_text())
    if summary["orient"] != orient:
        raise ValueError("selection summary orientation mismatch")
    best = summary["best"]["pp+slot"]
    return {"moves": int(best["moves"]), "beta": float(best["beta"]), "alpha": float(best["alpha"]),
            "slot_moves": int(best["slot_moves"]), "tau": float(summary["tau_hungarian"])}


def opening_scores(prefix, gallery_games, query_games, population, params) -> np.ndarray:
    """The eval script's pp+slot scorer (scripts/identity/identity_opening_eval.py)."""
    hashes = opening.prefix_hashes(prefix)
    plies = black_plies(params["moves"])
    pop = opening.population_prior(prefix, population, plies)
    base = opening.population_conditional(hashes, prefix, population, np.concatenate(query_games), plies, pop)
    scores = opening.prefix_backoff(prefix, gallery_games, query_games, plies, pop,
                                    beta=params["beta"], hashes=hashes, base=base)
    slot_plies = black_plies(params["slot_moves"])
    slot_pop = opening.population_prior(prefix, population, slot_plies)
    return scores + opening.slot_nb(prefix, gallery_games, query_games, slot_plies, slot_pop, params["alpha"])


def hd_scores(pool_hd: np.ndarray, pool_players: np.ndarray, candidates: np.ndarray, q_hd: np.ndarray,
              question_rows: list[np.ndarray], smoothing: float = 1.0) -> np.ndarray:
    """[questions, candidates] sum over a question's games of log P_c(HD of the game).

    P_c is the candidate's HD histogram over their pool games (HD 0..9, -1 = absent),
    smoothed toward the pooled histogram with `smoothing` pseudo-games. How often and
    how many stones a player takes is part of their identity and is independent of
    board orientation."""
    values = np.arange(-1, 10)
    index = {v: i for i, v in enumerate(values)}
    code = lambda hd: np.array([index[int(v)] for v in hd])
    pool_code, q_code = code(pool_hd), code(q_hd)
    prior = np.bincount(pool_code, minlength=len(values)) + 1.0
    prior /= prior.sum()
    logp = np.empty((len(candidates), len(values)))
    for j, c in enumerate(candidates):
        counts = np.bincount(pool_code[pool_players == c], minlength=len(values))
        logp[j] = np.log((counts + smoothing * prior) / (counts.sum() + smoothing))
    return np.stack([logp[:, q_code[rows]].sum(axis=1) for rows in question_rows])


def predict(store_dir: Path, pool_name: str, question_name: str, orient: str, method: str,
            params: dict | None = None, insurance_dir: Path | None = None,
            hd2plus: str = "keep", hd_weight: float = 0.0):
    """hd2plus: "keep" scores HD>=2 games like any other (their mislabelled plies are
    consistent between pool and questions, A-053 behaviour); "drop" leaves them out.
    hd_weight: weight of the HD-histogram term (hd_scores); 0 = off."""
    params = params or load_params(orient)
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    train_meta, train_pre = train_prefix(orient)
    pool_moves, pool_colors, pool_meta = load_store(pool_name, store_dir, mmap=False)
    q_moves, q_colors, q_meta = load_store(question_name, store_dir, mmap=False)
    pool_pre, pool_ok = store_prefix(pool_moves, pool_colors, pool_meta, orient, min_plies)
    q_pre, q_ok = store_prefix(q_moves, q_colors, q_meta, orient, min_plies)
    if hd2plus == "drop":
        pool_hd, q_hd = handicap_ok(pool_meta), handicap_ok(q_meta)
    else:
        pool_hd, q_hd = np.ones(len(pool_meta), bool), np.ones(len(q_meta), bool)
    pool_ok &= pool_hd
    q_ok &= q_hd

    pool_players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(pool_players)))
    train_pid = train_meta["player_id"].astype(str).to_numpy()
    population = np.flatnonzero((train_meta["color"].astype(str).to_numpy() == "B")
                                & ~np.isin(train_pid, candidates))
    overlap = int(np.isin(candidates, train_pid).sum())

    # One prefix array: population rows, then pool rows, then question rows.
    n_pop, n_pool = len(population), len(pool_meta)
    prefix = np.concatenate([train_pre[population], pool_pre, q_pre])
    gallery_games = [n_pop + np.flatnonzero((pool_players == c) & pool_ok) for c in candidates]
    question_ids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    query_games = [n_pop + n_pool + np.sort(rows_by_q[q][q_ok[rows_by_q[q]]]) for q in question_ids]
    scored = np.array([len(g) > 0 for g in query_games])
    scores = np.zeros((len(question_ids), len(candidates)))
    if scored.any():
        scores[scored] = opening_scores(prefix, gallery_games, [g for g, s in zip(query_games, scored) if s],
                                        np.arange(n_pop), params)
    insurance_info = {}
    if insurance_dir is not None:
        from go_aicup.tasks.identity.insurance import fused_scores
        scores, insurance_info = fused_scores(scores, insurance_dir, pool_name, question_name, candidates,
                                              pool_players, [rows_by_q[q][q_hd[rows_by_q[q]]] for q in question_ids],
                                              pool_keep=pool_hd)
    if hd_weight and "hd" in pool_meta and "hd" in q_meta:
        scores = scores + hd_weight * hd_scores(pool_meta["hd"].to_numpy(), pool_players, candidates,
                                                q_meta["hd"].to_numpy(), [rows_by_q[q] for q in question_ids])
    if method == "none":
        ranked = assign.top5(scores)
    else:
        log_q = assign.sinkhorn(assign.row_posterior(scores, params["tau"]))
        ranked = assign.hungarian_top5(log_q) if method == "hungarian" else assign.top5(log_q)
    info = {"questions": len(question_ids), "candidates": len(candidates),
            "unscored_questions": int((~scored).sum()), "unusable_pool_games": int((~pool_ok).sum()),
            "unusable_query_games": int((~q_ok).sum()),
            "hd2plus": hd2plus, "hd_weight": hd_weight,
            "hd2plus_pool_games": int((handicap_ok(pool_meta) == 0).sum()),
            "hd2plus_query_games": int((handicap_ok(q_meta) == 0).sum()),
            "empty_candidates": int(sum(len(g) == 0 for g in gallery_games)),
            "population_games": n_pop, "candidates_seen_in_train": overlap,
            "orient": orient, "assign": method, **params, **insurance_info}
    num_games = q_meta.groupby("question_id", sort=False)["num_games"].first()
    table = pd.DataFrame({"question_id": question_ids,
                          "num_games": num_games.loc[question_ids].to_numpy(),
                          "used_games": [len(g) for g in query_games]})
    for j in range(5):
        table[f"top{j + 1}"] = candidates[ranked[:, j]]
    return table, scores, candidates, info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--pool", required=True, help="e.g. pool_public")
    parser.add_argument("--questions", required=True, help="e.g. pid_public")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--orient", choices=("d4", "raw"), default="d4")
    parser.add_argument("--assign", choices=("hungarian", "sinkhorn", "none"), default="none")  # D-004
    parser.add_argument("--submission", type=Path, help="default <out-dir>/submission.csv")
    parser.add_argument("--insurance-dir", type=Path, help="identity/insurance.py work dir; fuse GE2E (A-053)")
    parser.add_argument("--hd2plus", choices=("keep", "drop"), default="keep",
                        help="HD>=2 games (reversed labels, no stones): keep as A-053 or leave out")
    parser.add_argument("--hd-weight", type=float, default=0.0, help="weight of the HD-histogram term; 0 = off")
    args = parser.parse_args(argv)
    table, scores, candidates, info = predict(args.store_dir, args.pool, args.questions, args.orient, args.assign,
                                              insurance_dir=args.insurance_dir, hd2plus=args.hd2plus,
                                              hd_weight=args.hd_weight)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "scores.npy", scores)
    np.save(args.out_dir / "candidates.npy", candidates)
    table.to_parquet(args.out_dir / "predictions.parquet", index=False)
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=2))
    # Official format (2026-11-11 sample_submission_player_v2.csv): question_id, top_1..top_5.
    submission = table[["question_id"]].assign(**{f"top_{j}": table[f"top{j}"] for j in range(1, 6)})
    path = args.submission or args.out_dir / "submission.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
