"""Competition-1 worst cases for the pool-level assignment (L6; RESULTS A-059), CPU only.

Question bank (synthetic splits 1-2, D4 opening pp+slot scores, A-029 parameters):
for every candidate and k in {5,10,15,20}, BANK questions from the candidate's
black games outside the gallery (consecutive blocks of one permutation; blocks
wrap around when a player has too few games, so a few questions share games).
Scenarios, 400 questions with 100 per k:
  one_to_one  every candidate once (the synthetic construction)
  replace     candidates drawn with replacement (about 63% distinct)
  p300        300 candidates; 100 of them asked twice
  p200        200 candidates, each asked twice
Accuracy: Gaussian noise added to the scores, sigma set so plain top-1 on the
one_to_one bank of split 1 is about each target (stylised degradation).
Methods: none (row top-5), sinkhorn, hungarian (tau from the opening summary).
Also the Day-0 item 7 statistics on the noisy scores: distinct row argmax and the
largest number of questions whose argmax is one candidate.

    python scripts/pid_worstcase.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, data
from go_aicup.tasks.identity.predict import load_params, opening_scores, train_prefix

OUT = ROOT / "outputs" / "pid" / "robustness" / "pid_worstcase"
SEED, BANK, REPS, SPLITS = 2027, 3, 30, (1, 2)
KS = (5, 10, 15, 20)
TARGETS = (None, 0.7, 0.5, 0.3)   # None: no noise
SCENARIOS = ("one_to_one", "replace", "p300", "p200")


def bank(meta, gal_by, candidates, rng):
    """Rows: (candidate position, k, b, games)."""
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    rows = []
    for c, player in enumerate(candidates):
        spare = np.setdiff1d(np.flatnonzero(black & (pid == player)), gal_by[c])
        for k in KS:
            order = rng.permutation(spare)
            for b in range(BANK):
                rows.append((c, k, b, np.take(order, np.arange(b * k, (b + 1) * k), mode="wrap")))
    return pd.DataFrame(rows, columns=["candidate", "k", "b", "games"])


def draw(scenario: str, n: int, rng) -> np.ndarray:
    """400 candidate positions (with repeats per scenario)."""
    if scenario == "one_to_one":
        return rng.permutation(n)
    if scenario == "replace":
        return rng.integers(0, n, 400)
    people = rng.permutation(n)[:300 if scenario == "p300" else 200]
    twice = people[:100] if scenario == "p300" else people
    return rng.permutation(np.concatenate([people, twice]))


def main() -> int:
    meta, prefix = train_prefix("d4")
    gallery, questions = data.pid_splits(meta)
    params = load_params("d4")
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    banks = {}
    for split in SPLITS:
        candidates, gal, _ = data.split_view(gallery, questions, split)
        b = bank(meta, gal, candidates, rng)
        population = np.flatnonzero(black & ~np.isin(pid, candidates))
        scores = opening_scores(prefix, gal, list(b["games"]), population, params)
        overlap = float(np.mean([len(np.unique(g)) < len(g) for g in b["games"]]))
        banks[split] = (candidates, b, scores, overlap)
        print(f"split {split}: bank {len(b)} questions, wrapped {overlap:.3f}", flush=True)

    candidates, b, scores, _ = banks[SPLITS[0]]
    base = b["b"].to_numpy() == 0
    truth = b["candidate"].to_numpy()[base]
    spread = scores[base].std()
    sigma_grid = spread * np.geomspace(0.01, 10, 60)
    accuracy = [float((np.argmax(scores[base] + s * rng.standard_normal(scores[base].shape), 1) == truth).mean())
                for s in sigma_grid]
    sigmas = {}
    for target in TARGETS:
        if target is None:
            sigmas["none"] = 0.0
        else:   # accuracy falls with sigma: first grid point at or below the target
            i = next(i for i, a in enumerate(accuracy) if a <= target)
            sigmas[f"{target}"] = float(sigma_grid[i])
    print("sigma by target top-1:", json.dumps(sigmas), flush=True)

    records = []
    for split, (candidates, b, scores, _) in banks.items():
        key = {(c, k, j): i for i, (c, k, j) in enumerate(zip(b["candidate"], b["k"], b["b"]))}
        for scenario in SCENARIOS:
            for level, sigma in sigmas.items():
                for rep in range(REPS):
                    who = draw(scenario, len(candidates), rng)
                    ks = rng.permutation(np.repeat(KS, 100))
                    seen: dict = {}
                    rows = []
                    for c, k in zip(who, ks):
                        j = seen.get((c, k), 0)
                        seen[(c, k)] = j + 1
                        rows.append(key[(c, k, j % BANK)])
                    s = scores[rows] + sigma * rng.standard_normal((len(rows), scores.shape[1]))
                    log_q = assign.sinkhorn(assign.row_posterior(s, params["tau"]))
                    ranked = {"none": assign.top5(s), "sinkhorn": assign.top5(log_q),
                              "hungarian": assign.hungarian_top5(log_q)}
                    best = s.argmax(axis=1)
                    diag = {"distinct_argmax": len(np.unique(best)), "max_load": int(np.bincount(best).max()),
                            "distinct_truth": len(np.unique(who))}
                    for method, r in ranked.items():
                        table = pid_report(candidates[r], candidates[who], ks)
                        records.append({"split": split, "scenario": scenario, "accuracy_target": level,
                                        "method": method, "rep": rep, **diag,
                                        **{f"score_{k}": float(table.loc[k, "score"]) for k in table.index},
                                        "top1_all": float(table.loc["all", "top1"])})
        print(f"split {split} done", flush=True)
    frame = pd.DataFrame(records)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "records.csv", index=False)
    cols = [c for c in frame.columns if c.startswith("score_")] + ["top1_all"]
    summary = frame.groupby(["accuracy_target", "scenario", "method"])[cols].mean().round(4)
    diag = frame[frame["method"] == "none"].groupby(["accuracy_target", "scenario"])[
        ["distinct_argmax", "max_load", "distinct_truth"]].agg(["mean", "min", "max"]).round(1)
    print(summary.to_string())
    print(diag.to_string())
    result = {"sigmas": sigmas, "wrapped_share": {s: v[3] for s, v in banks.items()},
              "summary": {f"{a}|{s}|{m}": row for (a, s, m), row in summary.to_dict("index").items()},
              "diagnostic": {f"{a}|{s}": {f"{c[0]}_{c[1]}": v for c, v in row.items()}
                             for (a, s), row in diag.to_dict("index").items()}}
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
