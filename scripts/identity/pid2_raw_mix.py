"""Added set: equal-weight mean of the raw and the D4-normalised opening terms (2026-11-16, CPU).

fused = (opening_raw + opening_d4) / 2 + 16 * w * cosine + category; the weight is not
searched and nothing else changes (see pid2_raw_opening.py). The gate is against the raw
arm; the difference to the D4 base in use is reported next to it.

    pool 0 draws 100-119: mix - raw >= +0.003, otherwise pool 1 is not scored
    pool 1 draws 100-129: mix - raw >= +0.003, delta / se >= 2.5, every k bucket >= -0.002

    python -m scripts.identity.pid2_raw_mix
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.tasks.identity import data, opening, twocolor
from go_aicup.tasks.identity.predict import PLIES
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_paired import BENCH, points
from scripts.identity.pid2_raw_opening import DRAWS, FIRST, OUT, WEIGHTS
from scripts.identity.pid2_transductive import load

GATE = {"pool0": 0.003, "pool1": 0.003, "z": 2.5, "bucket": -0.002}


def run_pool(env, reps, pool: int, n: int) -> pd.DataFrame:
    summary, pos = env["summary"], env["position"]
    rows = []
    for draw in range(FIRST, FIRST + n):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
        query = list(q["games"])
        ks = q["k"].to_numpy()
        style = summary["w"] * twocolor.style_scores(env["vectors"], [pos[g] for g in gallery], [pos[g] for g in query])
        rest = WEIGHTS[0] * style + WEIGHTS[1] * twocolor.category_scores(env["codes"], gallery, query, 4)
        op = {arm: twocolor.opening_scores(prefix, hashes, gallery, query, env["color"], env["populations"],
                                           summary["opening"]) for arm, (prefix, hashes) in reps.items()}
        op["mix"] = 0.5 * (op["d4"] + op["raw"])
        for arm, term in op.items():
            rows.append({"pool": pool, "draw": draw, "arm": arm, **bucket_means(points(term + rest, candidates, q), ks)})
        print(f"pool {pool} draw {draw}: " + "  ".join(f"{r['arm']} {r['all']:.4f}" for r in rows[-3:]), flush=True)
    return pd.DataFrame(rows)


def stats(frame: pd.DataFrame, pool: int, base: str) -> dict:
    part = frame[frame["pool"] == pool]
    row = paired(part, base).set_index("arm").loc["mix"]
    wide = part.pivot(index="draw", columns="arm", values=COLS)
    se_k = {c: float((wide[c]["mix"] - wide[c][base]).std(ddof=1) / np.sqrt(len(wide))) for c in COLS}
    return {"base": base, "draws": int(row["draws"]), "mix": {c: float(row[c]) for c in COLS},
            "delta": {c: float(row[f"d_{c}"]) for c in COLS}, "se": se_k,
            "z": float(row["d_all"] / se_k["all"]), "pos": float(row["pos"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    args = parser.parse_args()
    out_dir = OUT / f"{args.run}_mix"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load(args.run)
    env["vectors"] = np.load(BENCH / args.run / "embeddings.npz")[env["summary"]["style"].split("-")[1]]
    raw = data.opening_prefix(data.load_meta(), PLIES)
    reps = {"d4": (env["prefix"], env["hashes"]), "raw": (raw, opening.prefix_hashes(raw))}

    frame = run_pool(env, reps, 0, DRAWS[0])
    report = {"run": args.run, "gate": GATE, "pool0": {b: stats(frame, 0, b) for b in ("raw", "d4")}}
    report["pool0_pass"] = bool(report["pool0"]["raw"]["delta"]["all"] >= GATE["pool0"])
    if report["pool0_pass"]:
        frame = pd.concat([frame, run_pool(env, reps, 1, DRAWS[1])], ignore_index=True)
        report["pool1"] = {b: stats(frame, 1, b) for b in ("raw", "d4")}
        s = report["pool1"]["raw"]
        report["pool1_pass"] = bool(s["delta"]["all"] >= GATE["pool1"] and s["z"] >= GATE["z"]
                                    and min(s["delta"][str(k)] for k in cv.PID2_K) >= GATE["bucket"])
    report["verdict"] = ("candidate (pipeline unchanged)" if report.get("pool1_pass")
                         else "mix not adopted" if report["pool0_pass"] else "stopped at pool 0; pool 1 not scored")
    frame.to_csv(out_dir / "per_draw.csv", index=False)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
