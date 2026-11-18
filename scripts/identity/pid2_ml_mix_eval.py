"""Sealed check of the two passed changes together (2026-11-17, registered in RESULTS.md before the run).

base    opening + 16 w cos_B + category (hung-seqB)
ml      base + multi-level terms (stage-1 choice: N 15, beta 0.1, lambda1 = lambda2 = 1)
mix     opening + 16 w (0.5 cos_B + 0.5 cos_E2E9600) + category
mix_ml  mix + the same multi-level terms
No weight is re-tuned. Fresh sealed draws 1100- (pool 0 1100-1119, pool 1 1100-1129). Gate for mix_ml
against mix: pool 1 delta >= +0.005, z >= 2.5, every k >= -0.005, pool 0 delta > 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.data import cv
from go_aicup.tasks.identity import data, twocolor
from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
from scripts.identity.pid2_diag_d123a import bucket_means, paired
from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_seqagg import BENCH
from scripts.identity.pid2_transductive import load

RUN = "pid_ge2e_v4_d4_hd_9600"
ARMS = {"B": ("/home/jason1122/pid2_seqagg_cache", "outputs/pid/pid2_seqagg/head_B/best.pt"),
        "E": ("/home/jason1122/e2e_scratch32_cache/step9600",
              "outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt")}
FIRST = 1100
OUT = STAGE0 / "ml_mix"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e-cache", default=ARMS["E"][0])
    parser.add_argument("--e-head", default=ARMS["E"][1])
    parser.add_argument("--a-from", default="outputs/pid/e2e_scratch/mix9600/report.json")
    parser.add_argument("--first", type=int, default=FIRST)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    ARMS["E"] = (args.e_cache, args.e_head)
    out_dir, first = args.out, args.first
    out_dir.mkdir(parents=True, exist_ok=True)
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    n, beta, l1, l2 = choice["n"], choice["beta"], choice["lambda1"], choice["lambda2"]
    a = json.loads(Path(args.a_from).read_text())["choice"]["a"]
    device = torch.device("cuda")
    env = load(RUN)
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    colors = meta["player_color"].to_numpy().astype(np.int8)
    vec = {}
    for name, (cache, head_file) in ARMS.items():
        store = MoveStore(Path(cache) / "bench", colors)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite
        head, _ = load_head(Path(head_file), device)
        vec[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
    scale = 16.0 * float(saved["w"])
    bench = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[bench["game_index"]] = np.arange(len(bench["game_index"]))
    local = bench["keys"]
    popfile = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popfile["keys"], env["color"][popfile["game_index"]])
    pos = env["position"]
    rows = []
    for pool, count in ((0, 20), (1, 30)):
        for d in range(first, first + count):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
            query = list(q["games"])
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], env["summary"]["opening"])
                    + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            cb, ce = (twocolor.style_scores(vec[k], g, qq) for k in ("B", "E"))
            keys_of = lambda games: local[row[games]]
            ml = (l1 * level_scores(keys_of, env["color"], pop, gallery, query, n, "L1")[beta]
                  + l2 * level_scores(keys_of, env["color"], pop, gallery, query, n, "L2")[beta])
            base = rest + scale * cb
            mix = rest + scale * ((1 - a) * cb + a * ce)
            ks = q["k"].to_numpy()
            for name, sc in (("base", base), ("ml", base + ml), ("mix", mix), ("mix_ml", mix + ml)):
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"pool {pool} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "sealed_per_draw.csv", index=False)
    report = {"choice": choice, "a": a, "first": first, "e_arm": ARMS["E"], "tables": {}}
    for pool in (0, 1):
        for ref in ("base", "mix"):
            t = paired(frame[frame["pool"] == pool], ref)
            report["tables"][f"pool{pool} vs {ref}"] = t.to_dict(orient="records")
            print(f"pool {pool}, difference to {ref}:\n{t.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"]["pool1 vs mix"]).set_index("arm").loc["mix_ml"]
    t0 = pd.DataFrame(report["tables"]["pool0 vs mix"]).set_index("arm").loc["mix_ml"]
    z = float(t1["d_all"] / t1["se_all"])
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    report["gate"] = {"pool1_delta": float(t1["d_all"]), "z": z, "buckets": buckets,
                      "pool0_delta": float(t0["d_all"]),
                      "pass": bool(t1["d_all"] >= 0.005 and z >= 2.5 and min(buckets.values()) >= -0.005
                                   and t0["d_all"] > 0)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("GATE mix_ml vs mix:", json.dumps(report["gate"]), flush=True)


if __name__ == "__main__":
    main()
