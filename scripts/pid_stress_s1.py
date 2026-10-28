"""Competition-1 S1 opening-drift stress test and gate G3 (RESULTS A-061, D-005), CPU only.

Rehearsal release (public = synthetic split 1, private = split 2; A-053 / A-056)
with its GE2E insurance vectors. A fraction p of the usable query games has the
60-ply prefix seen by the opening model swapped for another game's D4 prefix;
the insurance vectors are left alone ("the middle game is still the player").
  population  black training games of non-candidates, from HOLDOUT games held
              out of the population prior (disjoint from it; p = 0 uses the same
              reduced population so every p is comparable)
  adversarial query games of other questions (another candidate is the answer);
              diagnostic only
Methods, all without assignment (D-004): opening only, style only (insurance
alone) and fused (opening + lambda * w * cosine), with the A-029 parameters and
lambda as in the release. p in {30, 60, 100} is drawn DRAWS times.

Sanity: opening only at p = 100 (population) scores <= 0.02 (chance ~0.004).
G3: at p = 100 (population), fused >= style only, mean of the two splits.

    python scripts/pid_stress_s1.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign
from go_aicup.tasks.identity.insurance import fused_scores
from go_aicup.tasks.identity.predict import load_params, opening_scores, store_prefix, train_prefix

REHEARSAL = ROOT / "outputs" / "rehearsal"
OUT = ROOT / "outputs" / "pid_stress_s1"
SEED, HOLDOUT, DRAWS = 2027, 20_000, 5
PS = (30, 60, 100)
PARTS = {"public": 1, "private": 2}
SANITY_MAX = 0.02


def report(scores, candidates, truth, k) -> pd.DataFrame:
    return pid_report(candidates[assign.top5(scores)], truth, k)


def run_part(part: str, params: dict, train_meta, train_pre, rng) -> list[dict]:
    store = REHEARSAL / "store"
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    pool_moves, pool_colors, pool_meta = load_store(f"pool_{part}", store, mmap=False)
    q_moves, q_colors, q_meta = load_store(f"pid_{part}", store, mmap=False)
    pool_pre, pool_ok = store_prefix(pool_moves, pool_colors, pool_meta, "d4", min_plies)
    q_pre, q_ok = store_prefix(q_moves, q_colors, q_meta, "d4", min_plies)

    pool_players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(pool_players)))
    train_pid = train_meta["player_id"].astype(str).to_numpy()
    outside = np.flatnonzero((train_meta["color"].astype(str).to_numpy() == "B") & ~np.isin(train_pid, candidates))
    outside = rng.permutation(outside)
    population, source = np.sort(outside[HOLDOUT:]), outside[:HOLDOUT]

    # Rows: population | pool | questions | replacement sources.
    n_pop, n_pool, n_q = len(population), len(pool_meta), len(q_meta)
    prefix = np.concatenate([train_pre[population], pool_pre, q_pre, train_pre[source]])
    q_base, src_base = n_pop + n_pool, n_pop + n_pool + n_q
    gallery_games = [n_pop + np.flatnonzero((pool_players == c) & pool_ok) for c in candidates]
    question_ids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    used = [np.sort(rows_by_q[q][q_ok[rows_by_q[q]]]) for q in question_ids]   # question-store rows
    scored = np.array([len(u) > 0 for u in used])

    answers = pd.read_csv(REHEARSAL / "answers" / f"pid_{part}.csv", dtype={"question_id": str})
    answers = answers.set_index("question_id").loc[[str(q) for q in question_ids]]
    truth, k = answers["player_id"].astype(str).to_numpy(), answers["k"].to_numpy()
    owner = np.empty(n_q, object)          # true player of every question-store row
    for q, t in zip(question_ids, truth):
        owner[rows_by_q[q]] = t

    insurance, info = fused_scores(np.zeros((len(question_ids), len(candidates))), REHEARSAL / "identity_insurance",
                                   f"pool_{part}", f"pid_{part}", candidates, pool_players,
                                   [rows_by_q[q] for q in question_ids])
    print(part, "insurance", info, "unscored", int((~scored).sum()), flush=True)

    flat_used = np.concatenate(used)
    records = []

    def evaluate(query_rows, src, p, draw):
        opening = np.zeros_like(insurance)
        opening[scored] = opening_scores(prefix, gallery_games, [r for r, s in zip(query_rows, scored) if s],
                                         np.arange(n_pop), params)
        for method, scores in (("opening", opening), ("style", insurance), ("fused", opening + insurance)):
            rep = report(scores, candidates, truth, k)
            records.append({"part": part, "source": src, "p": p, "draw": draw, "method": method,
                            **{f"k{b}": float(rep.loc[str(b), "score"]) for b in (5, 10, 15, 20)},
                            "all": float(rep.loc["all", "score"])})
        last = records[-3:]
        print(part, src, p, draw, {r["method"]: round(r["all"], 4) for r in last}, flush=True)

    evaluate([q_base + u for u in used], "none", 0, 0)
    for src in ("population", "adversarial"):
        for p in PS:
            for draw in range(DRAWS):
                swap = rng.permutation(flat_used)[:round(len(flat_used) * p / 100)]
                target = {}
                if src == "population":
                    target = dict(zip(swap, src_base + rng.choice(HOLDOUT, len(swap), replace=False)))
                else:
                    for row in swap:     # a query game of another question's (another candidate's) player
                        other = flat_used[owner[flat_used] != owner[row]]
                        target[row] = q_base + rng.choice(other)
                query_rows = [np.array([target.get(r, q_base + r) for r in u]) for u in used]
                evaluate(query_rows, src, p, draw)
    return records


def main() -> int:
    rng = np.random.default_rng(SEED)
    params = load_params("d4")
    train_meta, train_pre = train_prefix("d4")
    records = []
    for part in PARTS:
        records += run_part(part, params, train_meta, train_pre, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records)
    table.to_csv(OUT / "records.csv", index=False)

    cols = ["k5", "k10", "k15", "k20", "all"]
    mean = table.groupby(["source", "p", "method", "part"])[cols].mean()          # over draws
    by_split = mean.groupby(level=["source", "p", "method"]).mean()                 # over the two splits
    pivot = by_split["all"].unstack("method")
    sanity = float(by_split.loc[("population", 100, "opening"), "all"])
    g3 = by_split.loc[("population", 100)]
    result = {
        "params": params, "holdout": HOLDOUT, "draws": DRAWS, "seed": SEED,
        "sanity_opening_p100": sanity, "sanity_pass": sanity <= SANITY_MAX,
        "g3_fused": float(g3.loc["fused", "all"]), "g3_style": float(g3.loc["style", "all"]),
        "g3_pass": bool(g3.loc["fused", "all"] >= g3.loc["style", "all"]),
        "by_split_mean": {f"{s}|{p}|{m}": by_split.loc[(s, p, m)].round(4).to_dict() for s, p, m in by_split.index},
        "per_split": {f"{s}|{p}|{m}|{part}": mean.loc[(s, p, m, part)].round(4).to_dict()
                      for s, p, m, part in mean.index},
        "draw_sd_all": {"|".join(map(str, i)): round(float(v), 4) for i, v in
                        table.groupby(["source", "p", "method", "part"])["all"].std().dropna().items()},
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(pivot.round(4).to_string())
    print(json.dumps({k: result[k] for k in ("sanity_opening_p100", "sanity_pass", "g3_fused", "g3_style", "g3_pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
