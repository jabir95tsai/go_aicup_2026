"""Paired comparison of style encoders on the added-set bench at fixed fusion (2026-11-16, CPU).

Every run is scored on the same draws with the same opening term, lambda = 16, mu = 1 and
its own learned w: fused = opening + 16 * w * cosine + category, and the cosine alone.
The first run is the base of the gate; --also names a second reference (the current
9,600-step model) that is reported, not gated.

Besides the official score, each run gets its top-1 error rates over ALL questions split by
the rank gap between the wrong candidate and the true player (same rank / one rank / two or
more), so a fall of the same-rank share cannot hide a rise of cross-rank errors.

Gate per new run against the base: pool 1 delta >= +0.010, delta / se >= 2.5, every k bucket
>= -0.005, and pool 0 delta > 0. Pool 0 draws 100-119, pool 1 draws 100-129.

    python -m scripts.identity.pid2_encoder_compare --base pid_ge2e_v4_d4_hd_4800 \\
        --also pid_ge2e_v4_d4_hd_9600 --runs pid_ge2e_v7_band_4800 pid_ge2e_v7_pick_4800
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, twocolor
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_paired import BENCH, points
from scripts.identity.pid2_transductive import load

OUT = ROOT / "outputs" / "pid" / "pid2_encoder_compare"
PARAMS = "pid_ge2e_v4_d4_hd_9600"          # opening parameters and bench environment
FIRST, DRAWS = 100, {0: 20, 1: 30}
LAMBDA, MU = 16.0, 1.0
GATE = {"pool1": 0.010, "z": 2.5, "bucket": -0.005}
ERRORS = ("wrong_same", "wrong_adjacent", "wrong_far")


def error_rates(scores, candidates, q, rank_of) -> dict:
    """Share of all questions whose top-1 is wrong, by rank gap to the true player; per k and all."""
    ranks = rank_of[candidates].to_numpy()
    truth = np.searchsorted(candidates, q["player_id"].to_numpy())
    top = scores.argmax(axis=1)
    gap = np.abs(ranks[top] - ranks[truth])
    wrong = top != truth
    parts = {"wrong_same": wrong & (gap == 0), "wrong_adjacent": wrong & (gap == 1), "wrong_far": wrong & (gap >= 2)}
    ks = q["k"].to_numpy()
    return {name: bucket_means(mask.astype(float), ks) for name, mask in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--also", default=None)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    names = [args.base] + ([args.also] if args.also else []) + args.runs
    out_dir = OUT / "__".join(args.runs)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = load(PARAMS)
    meta = data.load_meta()
    rank_of = pd.Series(meta["rank_id"].to_numpy().astype(np.int64), index=meta["player_id"].astype(str).to_numpy())
    rank_of = rank_of[~rank_of.index.duplicated()]
    vectors = {}
    for name in names:
        saved = np.load(BENCH / name / "embeddings.npz")
        position = np.full(len(meta), -1)
        position[saved["game_index"]] = np.arange(len(saved["game_index"]))
        assert np.array_equal(position, env["position"]), f"{name}: vectors cover different games"
        assert np.isfinite(saved["swap"]).all()
        vectors[name] = (saved["swap"], float(saved["w"]))
    print({name: round(w, 4) for name, (_, w) in vectors.items()}, flush=True)

    rows, errors = [], []
    for pool in (0, 1):
        for draw in range(FIRST, FIRST + (args.limit or DRAWS[pool])):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
            query = list(q["games"])
            ks = q["k"].to_numpy()
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], env["summary"]["opening"])
                    + MU * twocolor.category_scores(env["codes"], gallery, query, 4))
            pos = env["position"]
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            for name, (vec, w) in vectors.items():
                cosine = twocolor.style_scores(vec, g, qq)
                for what, scores in (("fused", rest + LAMBDA * w * cosine), ("style only", cosine)):
                    rows.append({"pool": pool, "draw": draw, "arm": name, "what": what,
                                 **bucket_means(points(scores, candidates, q), ks)})
                    for kind, values in error_rates(scores, candidates, q, rank_of).items():
                        errors.append({"pool": pool, "draw": draw, "arm": name, "what": what, "error": kind, **values})
            print(f"pool {pool} draw {draw} done", flush=True)
    frame, err = pd.DataFrame(rows), pd.DataFrame(errors)
    frame.to_csv(out_dir / "per_draw.csv", index=False)
    err.to_csv(out_dir / "errors_per_draw.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    report = {"base": args.base, "also": args.also, "runs": args.runs, "gate": GATE, "tables": {}, "verdict": {}}
    for reference in [args.base] + ([args.also] if args.also else []):
        for pool in (0, 1):
            for what in ("fused", "style only"):
                part = frame[(frame["pool"] == pool) & (frame["what"] == what)]
                table = paired(part, reference)
                diff = part.pivot(index="draw", columns="arm", values=COLS)
                for c in COLS[:-1]:
                    table[f"se_{c}"] = [float((diff[c][a] - diff[c][reference]).std(ddof=1) / np.sqrt(len(diff)))
                                        for a in table["arm"]]
                report["tables"][f"vs {reference} | pool {pool} | {what}"] = table.to_dict(orient="records")
                print(f"\n== {what}, pool {pool}, difference to {reference} ==")
                print(table.round(4).to_string(index=False))
    for pool in (0, 1):
        part = err[(err["pool"] == pool) & (err["what"] == "fused")]
        table = part.groupby(["arm", "error"], sort=False)[COLS].mean().reset_index()
        report["tables"][f"errors | pool {pool} | fused"] = table.to_dict(orient="records")
        print(f"\n== top-1 error rate over all questions by rank gap, fused, pool {pool} ==")
        print(table.round(4).to_string(index=False))
        style = err[(err["pool"] == pool) & (err["what"] == "style only")]
        report["tables"][f"errors | pool {pool} | style only"] = (
            style.groupby(["arm", "error"], sort=False)[COLS].mean().reset_index().to_dict(orient="records"))

    def row(reference, pool, arm):
        return next(r for r in report["tables"][f"vs {reference} | pool {pool} | fused"] if r["arm"] == arm)

    for run in args.runs:
        p0, p1 = row(args.base, 0, run), row(args.base, 1, run)
        z = p1["d_all"] / p1["se_all"] if p1["se_all"] > 0 else float("nan")
        buckets = {str(k): p1[f"d_{k}"] for k in cv.PID2_K}
        report["verdict"][run] = {
            "pool0_delta": p0["d_all"], "pool1_delta": p1["d_all"], "pool1_se": p1["se_all"], "pool1_z": z,
            "pool1_buckets": buckets,
            "pass": bool(p0["d_all"] > 0 and p1["d_all"] >= GATE["pool1"] and z >= GATE["z"]
                         and min(buckets.values()) >= GATE["bucket"])}
    print(json.dumps(report["verdict"], indent=1))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
