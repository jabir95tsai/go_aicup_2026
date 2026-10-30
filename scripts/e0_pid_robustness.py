"""E0 (competition 1): how much of the synthetic 0.929 rests on fixed orientation and habit-heavy players.

Baseline = the A-005/A-006 probe: per-slot naive Bayes on black's first 4 moves,
player-disjoint synthetic pools splits 0-2 (400 candidates x 100 games each).
  (a) strata: score by the true player's first-move concentration in the gallery
  (b) stress: query games get a random D4 transform (gallery untouched),
      either one transform per question or one per game; plus the defence
      (D4-canonicalise gallery and queries) under the same stress.
This is a stress test, not a claim that the real test set is rotated.

  python scripts/e0_pid_robustness.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eda_report import N_POINTS, canonical_d4, d4_tables, nb_opening_scores
from go_aicup.data import cv
from go_aicup.data.store import GameStore
from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT

PLIES = np.arange(0, 8, 2)  # black's first 4 moves
OUT = ROOT / "outputs" / "pid" / "robustness" / "pid_e0_robustness"
SEED = 2026


def main() -> int:
    store = GameStore("train")
    meta = store.meta
    offsets = meta["offset"].to_numpy(np.int64)
    prefix = np.asarray(store.moves)[offsets[:, None] + np.arange(8)].astype(np.int64)
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    tables = d4_tables().astype(np.int64)
    gallery, questions = cv.load("pid_gallery"), cv.load("pid_questions")
    rng = np.random.default_rng(SEED)

    rows, strata = [], []
    for split in (0, 1, 2):  # the three mutually disjoint pools
        g = gallery[gallery["split"] == split]
        q = questions[questions["split"] == split].sort_values("question_id")
        cand = np.array(sorted(g["player_id"].astype(str).unique()))
        gal_games = [g.loc[g["player_id"].astype(str) == p, "game"].to_numpy() for p in cand]
        qry_games = list(q["games"])
        truth, k = q["player_id"].astype(str).to_numpy(), q["k"].to_numpy()
        outside = black & ~np.isin(pid, cand)

        # rotated copy of the prefix table: only query rows change
        flat = np.concatenate(qry_games)
        per_q = np.repeat(rng.integers(0, 8, len(qry_games)), [len(x) for x in qry_games])
        per_g = rng.integers(0, 8, len(flat))
        variants = {}
        for name, t in [("none", None), ("rot_per_question", per_q), ("rot_per_game", per_g)]:
            p = prefix.copy()
            if t is not None:
                p[flat] = tables[t[:, None], prefix[flat]]
            variants[f"raw / {name}"] = p
            if name != "rot_per_question":  # canonical form is invariant to both stresses
                variants[f"d4canon / {name}"] = canonical_d4(p)

        for name, p in variants.items():
            pop = np.stack([np.bincount(p[outside, j], minlength=N_POINTS) for j in PLIES])
            pop = (pop + 1) / (pop + 1).sum(axis=1, keepdims=True)
            scores = nb_opening_scores(p, gal_games, qry_games, PLIES, pop)
            ranked = cand[np.argsort(-scores, axis=1)[:, :5]]
            rep = pid_report(ranked, truth, k)
            rep["variant"], rep["split"] = name, split
            rows.append(rep.reset_index())
            if name == "raw / none":
                # habit strength = share of the player's most common first move in the gallery
                conc = {c: np.bincount(prefix[gg, 0], minlength=N_POINTS).max() / len(gg)
                        for c, gg in zip(cand, gal_games)}
                hit = ranked[:, 0] == truth
                pos = np.array([list(r).index(t) if t in r else -1 for r, t in zip(ranked, truth)])
                strata.append(pd.DataFrame({"split": split, "k": k, "conc": [conc[t] for t in truth],
                                            "score": np.where(pos >= 0, np.exp(-np.maximum(pos, 0)), 0.0),
                                            "top1": hit}))

    res = pd.concat(rows)
    cols = ["5", "10", "15", "20", "all"]
    score = res.pivot_table(index="variant", columns="k", values="score", aggfunc="mean")[cols]
    sd = res.pivot_table(index="variant", columns="k", values="score", aggfunc="std")[cols]
    st = pd.concat(strata)
    st["band"] = pd.cut(st["conc"], [0, 0.6, 0.8, 0.95, 1.0001],
                        labels=["<0.60", "0.60-0.80", "0.80-0.95", ">=0.95"])
    band = st.pivot_table(index="band", columns="k", values="score", aggfunc="mean", observed=True)
    band["all"] = st.groupby("band", observed=True)["score"].mean()
    band["n_questions"] = st.groupby("band", observed=True).size()

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "stress.csv", index=False)
    st.to_csv(OUT / "strata.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(
        {"score": score.round(4).to_dict(orient="index"),
         "score_sd_over_splits": sd.round(4).to_dict(orient="index")}, indent=2))
    pd.set_option("display.width", 160)
    print("official score, mean over splits 0-2\n", score.round(3))
    print("\nsd over splits\n", sd.round(3))
    print("\nby first-move concentration (raw / none)\n", band.round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
