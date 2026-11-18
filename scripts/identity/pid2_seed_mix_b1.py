"""B1: does a second, independently seeded E2E-scratch 9,600-step encoder (seed 2027, Yugen only) add to the
current fusion? (2026-11-19, registered in RESULTS.md before the run; fallback if the OGS short run fails.)

mix_ml9600   opening + 16 w (0.5 cos_B + 0.5 cos_E9600) + category + ML      (current base)
mix_mlS      opening + 16 w (0.5 cos_B + 0.5 cos_S2027) + category + ML      (seed replicate, reference)
mix3s_ml     opening + 16 w (0.5 cos_B + 0.25 cos_E9600 + 0.25 cos_S2027) + category + ML
No weight is tuned. Fresh sealed draws 1700- (pool 0 1700-1719, pool 1 1700-1729). Switch rule for mix3s_ml
against mix_ml9600: pool 1 delta >= +0.005, z >= 2.5, every k >= -0.005, pool 0 delta > 0.
"""
from __future__ import annotations

import json
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
        "E9600": ("/home/jason1122/e2e_scratch32_cache/step9600",
                  "outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt"),
        "S2027": ("/home/jason1122/ogs_short_cache/S2027",
                  "outputs/pid/ge2e/pid_e2e_scratch32_9600_s2027/step9600/best.pt")}
FIRST = 1700
OUT = STAGE0 / "seed_mix_b1"
REF = "mix_ml9600"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    n, beta, l1, l2 = choice["n"], choice["beta"], choice["lambda1"], choice["lambda2"]
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
        for d in range(FIRST, FIRST + count):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
            query = list(q["games"])
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], env["summary"]["opening"])
                    + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            c = {k: twocolor.style_scores(vec[k], g, qq) for k in ARMS}
            keys_of = lambda games: local[row[games]]
            ml = (l1 * level_scores(keys_of, env["color"], pop, gallery, query, n, "L1")[beta]
                  + l2 * level_scores(keys_of, env["color"], pop, gallery, query, n, "L2")[beta])
            arms = {"base": rest + scale * c["B"],
                    "mix_ml9600": rest + scale * (0.5 * c["B"] + 0.5 * c["E9600"]) + ml,
                    "mix_mlS": rest + scale * (0.5 * c["B"] + 0.5 * c["S2027"]) + ml,
                    "mix3s_ml": rest + scale * (0.5 * c["B"] + 0.25 * c["E9600"] + 0.25 * c["S2027"]) + ml}
            ks = q["k"].to_numpy()
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"pool {pool} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"choice": choice, "first": FIRST, "tables": {}, "switch": {}}
    for pool in (0, 1):
        for ref in ("base", REF):
            t = paired(frame[frame["pool"] == pool], ref)
            report["tables"][f"pool{pool} vs {ref}"] = t.to_dict(orient="records")
            print(f"pool {pool}, difference to {ref}:\n{t.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"][f"pool1 vs {REF}"]).set_index("arm")
    t0 = pd.DataFrame(report["tables"][f"pool0 vs {REF}"]).set_index("arm")
    for arm in ("mix_mlS", "mix3s_ml"):
        z = float(t1.loc[arm, "d_all"] / t1.loc[arm, "se_all"])
        buckets = {str(k): float(t1.loc[arm, f"d_{k}"]) for k in cv.PID2_K}
        report["switch"][arm] = {"pool1_delta": float(t1.loc[arm, "d_all"]), "z": z, "buckets": buckets,
                                 "pool0_delta": float(t0.loc[arm, "d_all"]),
                                 "pass": bool(t1.loc[arm, "d_all"] >= 0.005 and z >= 2.5
                                              and min(buckets.values()) >= -0.005 and t0.loc[arm, "d_all"] > 0)}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("SWITCH vs mix_ml9600:", json.dumps(report["switch"], indent=1), flush=True)


if __name__ == "__main__":
    main()
