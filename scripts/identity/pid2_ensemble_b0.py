"""B0: do already-trained encoders add to the current added-set base? (2026-11-18, registered in RESULTS.md
before the run.)

Base = hung-seqB-e2emix-ml: opening + 16 w (cos_B + cos_E9600) / 2 + category + stage-1 L1/L2.
Candidate members: the bench game embeddings ("swap" vectors, mean over moves) of earlier encoders,
the mean-aggregated vectors of the base encoder itself (A9600), and E2E 14,400 steps + its head. Excluded:
v6_gal_p1_4800 (trained on pool-1 gallery games). The style term becomes
    (cos_B + cos_E9600 + sum_m beta_m cos_m) / (2 + sum_m beta_m)
Greedy forward selection on development draws pool 0 100-119: each round tries every remaining member
with beta in {0.25, 0.5, 1}; the best is added if it raises the development mean by >= 0.002; at most 3
members (blend limit 5). Sealed draws 1500- (pool 0 1500-1519, pool 1 1500-1529) scored once.
Gate against the base: pool 1 delta >= +0.010, z >= 2.5, every k >= -0.005, pool 0 delta > 0.
"""
from __future__ import annotations

import json
import time
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
OUT = Path("outputs/pid/pid2_ensemble_b0")
HEADED = {"B": ("/home/jason1122/pid2_seqagg_cache", "outputs/pid/pid2_seqagg/head_B/best.pt"),
          "E9600": ("/home/jason1122/e2e_scratch32_cache/step9600",
                    "outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt"),
          "E14400": ("/home/jason1122/e2e_scratch32_cache/step14400",
                     "outputs/pid/ge2e/pid_e2e_scratch32_14400/step14400/best.pt")}
MEAN_RUNS = ["pid_ge2e_v1_short", "pid_ge2e_v2_both", "pid_ge2e_v2_both_p0", "pid_ge2e_v2_both_p0_600",
             "pid_ge2e_v2_both_p0_1200", "pid_ge2e_v2_both_p0_2400", "pid_ge2e_v3_d4_1200", "pid_ge2e_v3_d4_2400",
             "pid_ge2e_v4_d4_hd_2400", "pid_ge2e_v4_d4_hd_4800", "pid_ge2e_v4_d4_hd_9600", "pid_ge2e_v4_d4_hd_x8_2400",
             "pid_ge2e_v5_sub82_4800", "pid_ge2e_v7_band_4800", "pid_ge2e_v7_pick_4800"]
BETAS = (0.25, 0.5, 1.0)
MAX_ADD = 3
MIN_STEP = 0.002
DEV = (0, range(100, 120))
SEALED = ((0, range(1500, 1520)), (1, range(1500, 1530)))
GATE = {"pool1": 0.010, "z": 2.5, "bucket": -0.005, "pool0": 0.0}


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    vec = {}
    for name, (cache, head_file) in HEADED.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite
        head, _ = load_head(Path(head_file), device)
        vec[name] = twocolor.normalise(game_vectors(head, store, np.arange(len(store.games)), device).astype(np.float64))
        del store
    for run in MEAN_RUNS:
        e = np.load(BENCH / run / "embeddings.npz")
        assert np.array_equal(e["game_index"], saved["game_index"]), run
        name = "A9600" if run == RUN else run.replace("pid_ge2e_", "")
        vec[name] = twocolor.normalise(e["swap"].astype(np.float64))
    members = [m for m in vec if m not in ("B", "E9600")]
    print(f"{len(members)} candidate members, vectors in {time.time() - started:.0f}s", flush=True)
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda games: b1["keys"][row[games]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popf["keys"], color[popf["game_index"]])

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
        rest = rest + (s1["lambda1"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]]
                       + s1["lambda2"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]])
        g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
        cos = {k: twocolor.style_scores(v, g, qq) for k, v in vec.items()}
        return candidates, q, rest, cos

    def fused(rest, cos, chosen):
        num = cos["B"] + cos["E9600"] + sum(b * cos[m] for m, b in chosen)
        return rest + scale * num / (2.0 + sum(b for _, b in chosen))

    dev = [draw(DEV[0], d) for d in DEV[1]]
    print(f"development draws ready ({time.time() - started:.0f}s)", flush=True)
    score = lambda chosen: float(np.mean([points(fused(r, c, chosen), cand, q).mean() for cand, q, r, c in dev]))
    chosen, current, path = [], score([]), []
    base_dev = current
    for step in range(MAX_ADD):
        trials = {(m, b): score(chosen + [(m, b)]) for m in members if m not in {x for x, _ in chosen} for b in BETAS}
        (m, b), val = max(trials.items(), key=lambda kv: (round(kv[1], 12), -kv[0][1]))
        top = sorted(((v - current, k) for k, v in trials.items()), reverse=True)[:8]
        path.append({"step": step, "best": [m, b], "gain": val - current,
                     "top": [[f"{k[0]}:{k[1]}", round(g, 5)] for g, k in top]})
        print(f"step {step}: best {m} beta {b} gain {val - current:+.4f} ({time.time() - started:.0f}s)", flush=True)
        if val - current < MIN_STEP:
            break
        chosen.append((m, b))
        current = val
    choice = {"chosen": chosen, "dev_base": base_dev, "dev_gain": current - base_dev, "path": path}
    (OUT / "dev_choice.json").write_text(json.dumps(choice, indent=1))
    print("dev choice:", json.dumps({k: choice[k] for k in ("chosen", "dev_gain")}), flush=True)
    if not chosen:
        print("no member added on development draws; sealed draws not opened", flush=True)
        return

    rows = []
    for pool, draws in SEALED:
        for d in draws:
            candidates, q, rest, cos = draw(pool, d)
            ks = q["k"].to_numpy()
            arms = {"base": fused(rest, cos, []), "b0": fused(rest, cos, chosen)}
            for i in range(1, len(chosen)):
                arms[f"first{i}"] = fused(rest, cos, chosen[:i])
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"sealed pool {pool} done ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"choice": choice, "tables": {}}
    for pool, _ in SEALED:
        t = paired(frame[frame["pool"] == pool], "base")
        report["tables"][f"pool{pool}"] = t.to_dict(orient="records")
        print(f"pool {pool}:\n{t.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"]["pool1"]).set_index("arm").loc["b0"]
    t0 = pd.DataFrame(report["tables"]["pool0"]).set_index("arm").loc["b0"]
    z = float(t1["d_all"] / t1["se_all"])
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    passed = (t1["d_all"] >= GATE["pool1"] and z >= GATE["z"] and min(buckets.values()) >= GATE["bucket"]
              and t0["d_all"] > GATE["pool0"])
    report["gate"] = {"pool1_delta": float(t1["d_all"]), "pool1_se": float(t1["se_all"]), "z": z, "buckets": buckets,
                      "pool0_delta": float(t0["d_all"]), "pass": bool(passed), "rule": GATE}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("GATE:", "PASS" if passed else "FAIL", json.dumps(report["gate"]), flush=True)


if __name__ == "__main__":
    main()
