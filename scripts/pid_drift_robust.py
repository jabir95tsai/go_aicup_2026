"""Competition-1 opening drift: Day-0 detector calibration and the robust opening aggregate (RESULTS A-062).

Synthetic splits 0-2 (training indices), A-029 opening parameters, A-035 GE2E
vectors (`embeddings_split{s}.npz`). Drift as in A-061 (population version): a
fraction p of the query games takes the opening of a game held out of the
population prior. Per-game opening scores are computed once per split (query
games + HOLDOUT replacement games); a question's score is the sum over its
games, so every drift draw is an index substitution.

Robust fusion: drift.robust_opening(eps) + lambda * w * cos, (eps, lambda)
chosen on split 0 only by the mean over p in {0, 30, 60, 100}; splits 1-2 report.
Gate: (a) p = 0 fused >= current (eps 0, lambda 8) - 0.005; (b) p = 60 fused >=
style only. Detector: p_hat (prefix-match depth) and opening/style top-1
agreement per p, as the Day-0 reference table.

    python scripts/pid_drift_robust.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, data, drift, opening
from go_aicup.tasks.identity.predict import load_params, opening_scores, train_prefix
from identity_ge2e_eval import insurance_scores

RUN = ROOT / "outputs" / "pid" / "ge2e" / "pid_ge2e_v1_short"
OUT = ROOT / "outputs" / "pid" / "robustness" / "pid_drift_robust"
SEED, HOLDOUT, FOREIGN, DRAWS = 2027, 20_000, 5_000, 5
SPLITS, PS = (0, 1, 2), (0, 30, 60, 100)
EPS = (0.0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3)
LAMBDAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CURRENT = (0.0, 8.0)
COLS = ["k5", "k10", "k15", "k20", "all"]


def scores_of(scores, candidates, q) -> dict:
    rep = pid_report(candidates[assign.top5(scores)], q["player_id"].to_numpy(), q["k"].to_numpy())
    return {f"k{b}": float(rep.loc[str(b), "score"]) for b in (5, 10, 15, 20)} | {"all": float(rep.loc["all", "score"])}


def run_split(split, meta, prefix, hashes, gallery, questions, params, w, rng):
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    candidates, gal, q = data.split_view(gallery, questions, split)
    outside = rng.permutation(np.flatnonzero(black & ~np.isin(pid, candidates)))
    population, source = np.sort(outside[HOLDOUT:]), outside[:HOLDOUT]
    foreign = outside[HOLDOUT:HOLDOUT + FOREIGN]

    q_games = [np.asarray(g) for g in q["games"]]
    flat = np.concatenate(q_games)
    train_rows = np.concatenate([flat, source])                    # per-game row -> training index
    per_game = opening_scores(prefix, gal, [np.array([g]) for g in train_rows], population, params)
    starts = np.cumsum([0] + [len(g) for g in q_games])
    base_rows = [np.arange(starts[i], starts[i + 1]) for i in range(len(q_games))]
    check = float(np.abs(np.stack([per_game[r].sum(axis=0) for r in base_rows])
                         - opening_scores(prefix, gal, q_games, population, params)).max())
    ins = insurance_scores(RUN, split, gal, q, w)

    pool_rows = np.concatenate(gal)
    pool_depth = drift.match_depth(prefix, pool_rows, pool_rows, in_reference=True, hashes=hashes)
    foreign_depth = drift.match_depth(prefix, pool_rows, foreign, hashes=hashes)
    print(f"split {split}: per-game sum check {check:.2e}; depth pool {pool_depth.mean():.2f} "
          f"foreign {foreign_depth.mean():.2f}", flush=True)

    records, detector = [], []
    style = scores_of(ins, candidates, q)
    for p in PS:
        for draw in range(DRAWS if p else 1):
            swap = rng.permutation(len(flat))[:round(len(flat) * p / 100)]
            mapping = np.arange(len(flat))
            mapping[swap] = len(flat) + rng.choice(HOLDOUT, len(swap), replace=False)
            rows = [mapping[r] for r in base_rows]
            base = {"split": split, "p": p, "draw": draw}
            records.append(base | {"method": "style", "eps": np.nan, "lam": np.nan} | style)
            plain = drift.robust_opening(per_game, rows, 0.0)
            records.append(base | {"method": "opening", "eps": 0.0, "lam": np.nan} | scores_of(plain, candidates, q))
            for eps in EPS:
                o = plain if eps == 0 else drift.robust_opening(per_game, rows, eps)
                for lam in LAMBDAS:
                    records.append(base | {"method": "fused", "eps": eps, "lam": lam}
                                   | scores_of(o + lam * ins, candidates, q))
                    if (eps, lam) == CURRENT:
                        current = records[-1]["all"]
            query_depth = drift.match_depth(prefix, pool_rows, train_rows[np.concatenate(rows)], hashes=hashes)
            detector.append(base | drift.drift_estimate(pool_depth, query_depth, foreign_depth)
                            | {"agreement": drift.agreement(plain, ins)})
        print(f"split {split} p={p} (last draw): style {style['all']:.4f}, current {current:.4f}, "
              f"p_hat {detector[-1]['p_hat']:.3f}, agreement {detector[-1]['agreement']:.3f}", flush=True)
    return records, detector, check


def main() -> int:
    rng = np.random.default_rng(SEED)
    w = float(torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)["model"]["w"])
    meta, prefix = train_prefix("d4")
    hashes = opening.prefix_hashes(prefix)
    gallery, questions = data.pid_splits(meta)
    params = load_params("d4")
    records, detector, checks = [], [], {}
    for split in SPLITS:
        r, d, checks[split] = run_split(split, meta, prefix, hashes, gallery, questions, params, w, rng)
        records += r
        detector += d
    OUT.mkdir(parents=True, exist_ok=True)
    table, det = pd.DataFrame(records), pd.DataFrame(detector)
    table.to_csv(OUT / "records.csv", index=False)
    det.to_csv(OUT / "detector.csv", index=False)

    fused = table[table["method"] == "fused"].groupby(["split", "eps", "lam", "p"])[COLS].mean()   # over draws
    curve = fused.xs(0, level="split")["all"].groupby(level=["eps", "lam"]).mean()                 # mean over p
    eps, lam = max(curve.index, key=lambda i: (curve[i], -i[0], -i[1]))
    report = fused.loc[[1, 2]].groupby(level=["eps", "lam", "p"]).mean()                           # splits 1-2
    other = table[table["method"] != "fused"].groupby(["method", "split", "p"])[COLS].mean()
    other12 = other.loc[(slice(None), [1, 2]), :].groupby(level=["method", "p"]).mean()
    style = float(other12.loc[("style", 0), "all"])
    chosen, current = report.loc[(eps, lam)], report.loc[CURRENT]
    gate_a = float(chosen.loc[0, "all"]) >= float(current.loc[0, "all"]) - 0.005
    gate_b = float(chosen.loc[60, "all"]) >= style
    result = {
        "seed": SEED, "holdout": HOLDOUT, "draws": DRAWS, "w": w, "per_game_sum_check": checks,
        "split0_curve_mean_over_p": {f"{e}|{l}": round(float(v), 4) for (e, l), v in curve.items()},
        "chosen": {"eps": eps, "lam": lam, "on_grid_edge": eps in (EPS[0], EPS[-1]) or lam in (LAMBDAS[0], LAMBDAS[-1])},
        "splits12": {"chosen": chosen.round(4).to_dict("index"), "current": current.round(4).to_dict("index"),
                     "style": other12.loc["style"].round(4).to_dict("index"),
                     "opening": other12.loc["opening"].round(4).to_dict("index")},
        "gate": {"a_p0": [float(chosen.loc[0, "all"]), float(current.loc[0, "all"])], "a_pass": gate_a,
                 "b_p60": [float(chosen.loc[60, "all"]), style], "b_pass": gate_b, "pass": gate_a and gate_b},
        "detector": det.groupby(["split", "p"])[["p_hat", "agreement"]].agg(["mean", "min", "max"]).round(4)
                       .pipe(lambda d: {f"{s}|{p}": {f"{a}_{b}": v for (a, b), v in row.items()}
                                        for (s, p), row in d.iterrows()}),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"chosen eps={eps} lam={lam} (split-0 mean over p {curve[(eps, lam)]:.4f}; current {curve[CURRENT]:.4f})")
    for name, part in (("chosen", chosen), ("current", current)):
        print(name, "\n", part.round(4).to_string())
    print("style / opening (splits 1-2)\n", other12.round(4).to_string())
    print(json.dumps(result["gate"]))
    print(det.groupby(["split", "p"])[["p_hat", "agreement"]].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
