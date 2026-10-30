"""Competition-1 opening models on the synthetic pools (RESULTS A-027).

Pre-registered protocol: every hyper-parameter (black moves D, alpha, beta,
temperature) is chosen on split 0 by the official score over all questions;
splits 1 and 2 (player-disjoint from 0 and each other) are only reported.
Population priors use black games of players outside the evaluated pool.

`--orient d4` canonicalises every game's opening (gallery, queries and
population alike; RESULTS E0-001 default) and first checks that randomly
rotating the query games leaves their canonical form unchanged (A-029).

    PYTHONPATH=src python scripts/identity_opening_eval.py [--orient raw|d4]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, data, opening

OUT = {"raw": ROOT / "outputs" / "pid" / "opening" / "pid_opening_v1", "d4": ROOT / "outputs" / "pid" / "opening" / "pid_opening_v1_d4"}
TUNE_SPLIT, REPORT_SPLITS = 0, (1, 2)
PLIES = 60


def black_plies(moves: int) -> np.ndarray:
    return np.arange(0, 2 * moves, 2)


def evaluate(ranked_pos, candidates, q) -> pd.DataFrame:
    return pid_report(candidates[ranked_pos], q["player_id"].to_numpy(), q["k"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orient", choices=("raw", "d4"), default="raw")
    orient = parser.parse_args().orient
    out = OUT[orient]
    out.mkdir(parents=True, exist_ok=True)
    meta = data.load_meta()
    gallery, questions = data.pid_splits(meta)
    if orient == "raw":
        prefix = data.opening_prefix(meta, PLIES)
    else:
        prefix = data.opening_prefix_d4(meta, PLIES)
        rows = np.concatenate(list(questions["games"]))
        g = np.random.default_rng(2026).integers(0, 8, len(rows))
        rotated = data.d4_tables()[g[:, None], data.opening_prefix(meta, PLIES)[rows]]
        assert np.array_equal(data.canonical_d4(rotated), prefix[rows]), "D4 canonical form not invariant"
        print(f"D4 invariance check passed on {len(rows)} randomly rotated query games")
    hashes = opening.prefix_hashes(prefix)
    black = meta["color"].to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()

    pop_games = {}
    base_cache = {}

    def outside(split):
        if split not in pop_games:
            candidates, _, _ = data.split_view(gallery, questions, split)
            pop_games[split] = np.flatnonzero(black & ~np.isin(pid, candidates))
        return pop_games[split]

    def slot(split, moves, alpha):
        candidates, gal, q = data.split_view(gallery, questions, split)
        plies = black_plies(moves)
        pop = opening.population_prior(prefix, outside(split), plies)
        return opening.slot_nb(prefix, gal, list(q["games"]), plies, pop, alpha), candidates, q

    def pp(split, moves, beta):
        candidates, gal, q = data.split_view(gallery, questions, split)
        plies = black_plies(moves)
        pop = opening.population_prior(prefix, outside(split), plies)
        qry = list(q["games"])
        if (split, moves) not in base_cache:
            base_cache[split, moves] = opening.population_conditional(
                hashes, prefix, outside(split), np.concatenate(qry), plies, pop)
        s = opening.prefix_backoff(prefix, gal, qry, plies, pop, beta=beta, hashes=hashes,
                                   base=base_cache[split, moves])
        return s, candidates, q

    def scorer(split, model, moves, alpha=5.0, beta=1.0, slot_moves=4):
        if model == "slot":
            return slot(split, moves, alpha)
        s, c, q = pp(split, moves, beta)
        if model == "pp+slot":
            s = s + slot(split, slot_moves, alpha)[0]
        return s, c, q

    rows = []

    def record(name, split, ranked, candidates, q, **params):
        rep = evaluate(ranked, candidates, q)
        for k, r in rep.iterrows():
            rows.append({"variant": name, "split": split, "k": k, **r.to_dict(), **params})
        return float(rep.loc["all", "score"])

    # ---- stage 1: model / depth / smoothing, tuned on split 0 --------------
    grid = [("slot", m, a, 0.0, 0) for m in (4, 8) for a in (1.0, 5.0)]
    grid += [("pp", m, 5.0, b, 0) for m in (4, 8, 12) for b in (0.3, 1.0, 3.0)]
    grid += [("pp+slot", m, 5.0, b, sm) for m in (8, 12) for b in (0.3, 1.0, 3.0) for sm in (2, 4)]
    tune = []
    t0 = time.time()
    for model, moves, alpha, beta, slot_moves in grid:
        s, c, q = scorer(TUNE_SPLIT, model, moves, alpha, beta, slot_moves)
        score = record(f"{model}", TUNE_SPLIT, assign.top5(s), c, q,
                       moves=moves, alpha=alpha, beta=beta, slot_moves=slot_moves)
        tune.append((score, model, moves, alpha, beta, slot_moves))
        print(f"{model:7s} D={moves:2d} a={alpha:<3} b={beta:<3} sD={slot_moves} split0 all={score:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    best = {m: max(t for t in tune if t[1] == m) for m in ("slot", "pp", "pp+slot")}
    print("best on split 0:", best)

    # ---- stage 2: L6 temperature for the best combined model, on split 0 ---
    _, model, moves, alpha, beta, slot_moves = best["pp+slot"]
    cache = {s: scorer(s, model, moves, alpha, beta, slot_moves) for s in (TUNE_SPLIT, *REPORT_SPLITS)}
    temps = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    tau_scores = {}
    s, c, q = cache[TUNE_SPLIT]
    for tau in temps:
        log_q = assign.sinkhorn(assign.row_posterior(s, tau))
        tau_scores[tau] = (record("pp+slot+sinkhorn", TUNE_SPLIT, assign.top5(log_q), c, q, tau=tau),
                           record("pp+slot+hungarian", TUNE_SPLIT, assign.hungarian_top5(log_q), c, q, tau=tau))
        print(f"tau={tau:<5} sinkhorn={tau_scores[tau][0]:.4f} hungarian={tau_scores[tau][1]:.4f}")
    tau_sink = max(temps, key=lambda t: tau_scores[t][0])
    tau_hung = max(temps, key=lambda t: tau_scores[t][1])

    # ---- report splits ------------------------------------------------------
    for split in REPORT_SPLITS:
        for name in ("slot", "pp"):
            s, c, q = scorer(split, *best[name][1:])
            record(f"{name} (best)", split, assign.top5(s), c, q)
        s, c, q = scorer(split, "slot", 4, 5.0)
        record("slot D=4 a=5 (E5)", split, assign.top5(s), c, q)
        s, c, q = cache[split]
        record("pp+slot (best)", split, assign.top5(s), c, q)
        record("pp+slot+sinkhorn", split, assign.top5(assign.sinkhorn(assign.row_posterior(s, tau_sink))), c, q)
        record("pp+slot+hungarian", split,
               assign.hungarian_top5(assign.sinkhorn(assign.row_posterior(s, tau_hung))), c, q)

    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)
    report = res[res["split"].isin(REPORT_SPLITS) & res["variant"].isin(
        ["slot D=4 a=5 (E5)", "slot (best)", "pp (best)", "pp+slot (best)", "pp+slot+sinkhorn",
         "pp+slot+hungarian"])]
    table = report.pivot_table(index="variant", columns="k", values="score", aggfunc="mean")
    top1 = report.pivot_table(index="variant", columns="k", values="top1", aggfunc="mean")
    gap = report.pivot_table(index="variant", columns="split", values="score",
                             aggfunc=lambda v: v.iloc[-1])  # 'all' row is last per split
    summary = {"orient": orient, "best": {m: dict(zip(("score", "model", "moves", "alpha", "beta", "slot_moves"), b))
                        for m, b in best.items()},
               "tau_sinkhorn": tau_sink, "tau_hungarian": tau_hung,
               "tau_scores_split0": {str(k): v for k, v in tau_scores.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nofficial score, mean of splits 1-2\n", table[["5", "10", "15", "20", "all"]].round(4))
    print("\ntop-1, mean of splits 1-2\n", top1[["5", "10", "15", "20", "all"]].round(4))
    print("\nper split (all)\n", gap.round(4))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
