"""A2 (E2E-scratch 9,600 steps on Yugen + all 82,481 OGS players x 10 games) against A1 (40,000 x 20)
inside the current added-set fusion (2026-11-19, registered in RESULTS.md before the run).

rest = opening + category + stage-1 L1/L2; ogsw form: score = rest + 16 w 1.5 (0.25 cos_B + 0.75 cos_X).
  A1w   X = A1 (= ogsw.csv, reference)        A2w   X = A2
  A12w  X = mean(cos_A1, cos_A2)              alone_A1 / alone_A2 = rest + 16 w cos_X (observation)
No weight is tuned. Sealed draws 2000- (pool 0 2000-2019, pool 1 2000-2029). Switch rule (same-path) against
A1w for A2w and A12w: pool 1 delta >= +0.005, z >= 2.5, every k >= -0.005, pool 0 delta > 0. If both pass,
the larger pool 1 delta wins; if neither passes, keep ogsw.
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
ENC = {"B": ("/home/jason1122/pid2_seqagg_cache", "outputs/pid/pid2_seqagg/head_B/best.pt"),
       "A1": ("/work/jason1122/ogs_a1_cache/A1", "outputs/pid/ge2e/pid_e2e_ogs_A1_YO9600/step9600/best.pt"),
       "A2": ("/home/jason1122/ogs_runs/a2_cache/A2", "outputs/pid/ge2e/pid_e2e_ogs_A2_YO9600/step9600/best.pt")}
SEALED = ((0, range(2000, 2020)), (1, range(2000, 2030)))
REF = "A1w"
OUT = Path("outputs/pid/ogs/a2_eval")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    vec = {}
    for name, (cache, head_file) in ENC.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, _ = load_head(Path(head_file), device)
        vec[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda games: b1["keys"][row[games]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popf["keys"], color[popf["game_index"]])
    rows = []
    for pool, draws in SEALED:
        for d in draws:
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
            query = list(q["games"])
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                            env["populations"], env["summary"]["opening"])
                    + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4)
                    + s1["lambda1"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]]
                    + s1["lambda2"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]])
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            c = {k: twocolor.style_scores(v, g, qq) for k, v in vec.items()}
            c["A12"] = 0.5 * (c["A1"] + c["A2"])
            arms = {f"{x}w": rest + scale * 1.5 * (0.25 * c["B"] + 0.75 * c[x]) for x in ("A1", "A2", "A12")}
            arms.update({f"alone_{x}": rest + scale * c[x] for x in ("A1", "A2")})
            ks = q["k"].to_numpy()
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"pool {pool} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"tables": {}, "switch": {}}
    for pool in (0, 1):
        t = paired(frame[frame["pool"] == pool], REF)
        report["tables"][f"pool{pool} vs {REF}"] = t.to_dict(orient="records")
        print(f"pool {pool}, difference to {REF}:\n{t.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"][f"pool1 vs {REF}"]).set_index("arm")
    t0 = pd.DataFrame(report["tables"][f"pool0 vs {REF}"]).set_index("arm")
    for arm in ("A2w", "A12w"):
        z = float(t1.loc[arm, "d_all"] / t1.loc[arm, "se_all"])
        buckets = {str(k): float(t1.loc[arm, f"d_{k}"]) for k in cv.PID2_K}
        report["switch"][arm] = {"pool1_delta": float(t1.loc[arm, "d_all"]), "z": z, "buckets": buckets,
                                 "pool0_delta": float(t0.loc[arm, "d_all"]),
                                 "pass": bool(t1.loc[arm, "d_all"] >= 0.005 and z >= 2.5
                                              and min(buckets.values()) >= -0.005 and t0.loc[arm, "d_all"] > 0)}
    passed = [a for a in report["switch"] if report["switch"][a]["pass"]]
    report["chosen"] = max(passed, key=lambda a: report["switch"][a]["pool1_delta"]) if passed else REF
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("SWITCH:", json.dumps(report["switch"]), "CHOSEN:", report["chosen"], flush=True)


if __name__ == "__main__":
    main()
