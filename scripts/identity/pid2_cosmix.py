"""Cosine mixing of two style models on the added-set bench (2026-11-17).

Arms are (per-move bench cache, B head) pairs as in pid2_seqagg compare. The base arm is the current
base (9,600-step encoder + B s2026). For the --mix arm the fused score uses
    opening + 16 * w * ((1 - a) * cos_base + a * cos_mix) + category,
a chosen on development draws pool 0 100-119 from --weights (a = 1 is the mix arm alone). Sealed draws
(pool 0 first..first+19, pool 1 first..first+29) are scored once for the base, every arm alone and the
chosen mix. Gates (written in RESULTS.md before the run): --gate name:min_gain applies the confirm rule
(pool 1 delta >= min_gain, z >= 2.5, every k >= -0.005, pool 0 delta > 0) to that arm ("mix" = the
chosen mix).

    python -m scripts.identity.pid2_cosmix --arms base=CACHE:HEAD e=CACHE:HEAD --base base --mix e \\
        --weights 0.25 0.5 0.75 --sealed-first 900 --gate mix:0.010 --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.data import cv
from go_aicup.tasks.identity import data, twocolor
from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_seqagg import BENCH
from scripts.identity.pid2_transductive import load

RUN = "pid_ge2e_v4_d4_hd_9600"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--mix", required=True)
    parser.add_argument("--weights", type=float, nargs="+", required=True)
    parser.add_argument("--sealed-first", type=int, required=True)
    parser.add_argument("--gate", nargs="+", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    env = load(RUN)
    saved = np.load(BENCH / RUN / "embeddings.npz")
    colors = data.load_meta()["player_color"].to_numpy().astype(np.int8)
    vectors, info = {}, {}
    for spec in args.arms:
        name, rest = spec.split("=", 1)
        cache_dir, _, head_file = rest.partition(":")
        store = MoveStore(Path(cache_dir) / "bench", colors)
        if not (np.array_equal(store.games, saved["game_index"]) and store.finite):
            raise ValueError(f"{name}: bench cache does not match the bench games or is not finite")
        head, step = load_head(Path(head_file), device)
        info[name] = {"cache": cache_dir, "head": head_file, "step": step}
        vectors[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
        print(f"{name}: vectors ready (head step {step})", flush=True)
    scale = 16.0 * float(saved["w"])
    pos = env["position"]

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
        g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
        cos = {name: twocolor.style_scores(v, g, qq) for name, v in vectors.items()}
        return candidates, q, rest, cos

    # development choice of the mixing weight
    dev = []
    for d in range(100, 120):
        candidates, q, rest, cos = draw(0, d)
        row = {"draw": d, "base": points(rest + scale * cos[args.base], candidates, q).mean()}
        for a in args.weights:
            mixed = (1 - a) * cos[args.base] + a * cos[args.mix]
            row[f"a{a:g}"] = points(rest + scale * mixed, candidates, q).mean()
        dev.append(row)
    dev = pd.DataFrame(dev).set_index("draw")
    gains = dev.drop(columns="base").sub(dev["base"], axis=0).mean()
    best = max(gains.index, key=lambda c: (round(gains[c], 12), -float(c[1:])))   # ties -> smaller a
    a = float(best[1:])
    choice = {"a": a, "dev_gain": float(gains[best]), "dev_gains": gains.to_dict()}
    print("dev:", json.dumps(choice), flush=True)

    rows = []
    for pool, n in ((0, 20), (1, 30)):
        for d in range(args.sealed_first, args.sealed_first + n):
            candidates, q, rest, cos = draw(pool, d)
            ks = q["k"].to_numpy()
            arms = {name: c for name, c in cos.items()}
            arms["mix"] = (1 - a) * cos[args.base] + a * cos[args.mix]
            for name, c in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name,
                             **bucket_means(points(rest + scale * c, candidates, q), ks)})
        print(f"sealed pool {pool} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "sealed_per_draw.csv", index=False)
    report = {"arms": info, "base": args.base, "mix": args.mix, "choice": choice,
              "sealed_first": args.sealed_first, "tables": {}, "gate": {}}
    for pool in (0, 1):
        part = frame[frame["pool"] == pool]
        t = paired(part, args.base)
        wide = part.pivot(index="draw", columns="arm", values=COLS)
        for c in COLS[:-1]:
            t[f"se_{c}"] = [float((wide[c][x] - wide[c][args.base]).std(ddof=1) / np.sqrt(len(wide)))
                            for x in t["arm"]]
        report["tables"][f"pool{pool}"] = t.to_dict(orient="records")
        print(f"pool {pool}, difference to {args.base}:\n{t.round(4).to_string(index=False)}", flush=True)
    t0 = pd.DataFrame(report["tables"]["pool0"]).set_index("arm")
    t1 = pd.DataFrame(report["tables"]["pool1"]).set_index("arm")
    for spec in args.gate:
        name, gain = spec.split(":")
        z = float(t1.loc[name, "d_all"] / t1.loc[name, "se_all"])
        buckets = {str(k): float(t1.loc[name, f"d_{k}"]) for k in cv.PID2_K}
        passed = (t1.loc[name, "d_all"] >= float(gain) and z >= 2.5 and min(buckets.values()) >= -0.005
                  and t0.loc[name, "d_all"] > 0)
        report["gate"][name] = {"min_gain": float(gain), "pool1_delta": float(t1.loc[name, "d_all"]),
                                "pool1_se": float(t1.loc[name, "se_all"]), "z": z, "buckets": buckets,
                                "pool0_delta": float(t0.loc[name, "d_all"]), "pass": bool(passed)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("GATE:", json.dumps(report["gate"], indent=1), flush=True)


if __name__ == "__main__":
    main()
