"""Scoring variants of the style term on the added-set bench, fused with the opening and
HD-and-colour terms: cosine vs PLDA (twocolor.plda_scores), one encoder vs the mean of several.

The weight of each variant's style term is chosen on pool 0; pool 1 reports the paired
difference to the first variant (cosine of the first run, the current submission's form).

    python -m scripts.identity.pid2_scoring --draws 20 --runs pid_ge2e_v2_both_p0_1200 pid_ge2e_v2_both_p0_600
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import opening, twocolor
from go_aicup.tasks.identity.predict import train_prefix
from scripts.identity.pid2_paired import BANK, BENCH, HANDICAP, points

OUT = ROOT / "outputs" / "pid" / "pid2_scoring"
COS_LAMBDAS = (8.0, 16.0, 32.0)
PLDA_LAMBDAS = (1.0, 2.0, 4.0, 8.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--first", type=int, default=100)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--tag", default="scoring")
    args = parser.parse_args()

    meta, prefix = train_prefix("d4")
    bank = pd.read_parquet(BANK)
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy().astype(np.int64)
    color = meta["player_color"].to_numpy().astype(np.int64)
    pid = meta["player_id"].astype(str).to_numpy()
    hashes = opening.prefix_hashes(prefix)
    codes = np.minimum(handicap, 1) + 2 * color
    outside = ~np.isin(pid, bank["player_id"].unique())
    populations = (np.flatnonzero(outside & (color == 0)), np.flatnonzero(outside & (color == 1)))
    summary = json.loads((BENCH / args.runs[0] / "summary.json").read_text())
    runs = []
    for name in args.runs:
        saved = np.load(BENCH / name / "embeddings.npz")
        position = np.full(len(meta), -1)
        position[saved["game_index"]] = np.arange(len(saved["game_index"]))
        runs.append((saved["swap"], position, float(saved["w"])))

    rows = []
    for pool in (0, 1):
        for draw in range(args.first, args.first + args.draws):
            candidates, gallery, q = cv.draw_pid2(bank, pool, draw)
            _, other, _ = cv.draw_pid2(bank, 1 - pool, draw)      # stands for the other part's gallery
            query = list(q["games"])
            base = (twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, summary["opening"])
                    + summary["mu"] * twocolor.category_scores(codes, gallery, query, 4))
            cos, plda = [], []
            for vectors, position, w in runs:
                gal, qry = [position[g] for g in gallery], [position[g] for g in query]
                cos.append(w * twocolor.style_scores(vectors, gal, qry))
                plda.append(twocolor.plda_scores(vectors, gal, qry, [position[g] for g in other]))
            styles = {"cos first": (cos[0], COS_LAMBDAS), "plda first": (plda[0], PLDA_LAMBDAS)}
            if len(runs) > 1:
                styles["cos mean"] = (np.mean(cos, axis=0), COS_LAMBDAS)
                styles["plda mean"] = (np.mean(plda, axis=0), PLDA_LAMBDAS)
            ks = q["k"].to_numpy()
            for name, (style, lambdas) in styles.items():
                for lam in lambdas:
                    pts = points(base + lam * style, candidates, q)
                    rows.append({"pool": pool, "draw": draw, "variant": name, "lambda": lam,
                                 **{str(k): pts[ks == k].mean() for k in cv.PID2_K}, "all": pts.mean()})
            print(f"pool {pool} draw {draw}", flush=True)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{args.tag}.csv", index=False)
    cols = [str(k) for k in cv.PID2_K] + ["all"]
    select = frame[frame["pool"] == 0].groupby(["variant", "lambda"], sort=False)["all"].mean()
    print("pool 0 mean by variant and lambda:\n" + select.round(4).to_string())
    chosen = {v: select[v].idxmax() for v in select.index.get_level_values(0).unique()}
    report = frame[frame["pool"] == 1]
    picked = {v: report[(report["variant"] == v) & (report["lambda"] == lam)].set_index("draw")[cols]
              for v, lam in chosen.items()}
    reference = picked["cos first"]
    for name, table in picked.items():
        diff = table - reference
        print(f"\npool 1, {name}, lambda {chosen[name]}:")
        print(pd.DataFrame({"mean": table.mean(), "diff": diff.mean(),
                            "se": diff.std(ddof=1) / np.sqrt(len(diff)), "pos": (diff > 0).mean()}).round(4).to_string())


if __name__ == "__main__":
    main()
