"""Cross-site short-run comparison: Yugen-only (Y) vs Yugen+OGS (YO) encoders, 960 steps each
(2026-11-18, registered in RESULTS.md before the run).

Both arms are scored inside the current fusion on the added-set bench (unseen Yugen players):
    rest = opening + category + stage-1 L1/L2 (as hung-seqB-e2emix-ml)
    alone_X = rest + 16 w cos_X              mix_X = rest + 16 w (0.5 cos_B + 0.5 cos_X)
No weight is tuned. Sealed draws 1600- (pool 0 1600-1619, pool 1 1600-1629). Decision (YO against Y, mix):
pool 1 delta >= +0.010 overall, pool 0 delta > 0, and the k=1 gain not cancelled by k=2/3/5 losses
(every k >= -0.005). If neither arm learned to identify (alone_Y and alone_YO both below the opening-only
fusion), the short run is recorded as undecidable.
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
from scripts.identity.pid2_diag_d123a import bucket_means, paired
from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_seqagg import BENCH
from scripts.identity.pid2_transductive import load

RUN = "pid_ge2e_v4_d4_hd_9600"
B = ("/home/jason1122/pid2_seqagg_cache", "outputs/pid/pid2_seqagg/head_B/best.pt")
SEALED = ((0, range(1600, 1620)), (1, range(1600, 1630)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs=2, required=True, help="Y=CACHE:HEAD YO=CACHE:HEAD")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1600, help="first sealed draw (A1 uses 1700)")
    args = parser.parse_args()
    sealed = tuple((pool, range(args.first, args.first + len(r))) for pool, r in SEALED)
    args.out.mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    specs = {"B": B}
    for spec in args.arms:
        name, rest = spec.split("=", 1)
        cache, _, head = rest.partition(":")
        specs[name] = (cache, head)
    assert set(specs) == {"B", "Y", "YO"}
    vec, steps = {}, {}
    for name, (cache, head_file) in specs.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, steps[name] = load_head(Path(head_file), device)
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
    for pool, draws in sealed:
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
            arms = {"no_style": rest, "B": rest + scale * c["B"]}
            for x in ("Y", "YO"):
                arms[f"alone_{x}"] = rest + scale * c[x]
                arms[f"mix_{x}"] = rest + scale * (0.5 * c["B"] + 0.5 * c[x])
            ks = q["k"].to_numpy()
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"pool {pool} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "sealed_per_draw.csv", index=False)
    report = {"steps": steps, "tables": {}}
    for pool, _ in sealed:
        for ref in ("B", "mix_Y", "alone_Y"):
            t = paired(frame[frame["pool"] == pool], ref)
            report["tables"][f"pool{pool} vs {ref}"] = t.to_dict(orient="records")
            print(f"pool {pool}, difference to {ref}:\n{t.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"]["pool1 vs mix_Y"]).set_index("arm").loc["mix_YO"]
    t0 = pd.DataFrame(report["tables"]["pool0 vs mix_Y"]).set_index("arm").loc["mix_YO"]
    a1 = pd.DataFrame(report["tables"]["pool1 vs B"]).set_index("arm")
    learned = {x: float(a1.loc[f"alone_{x}", "all"] - a1.loc["no_style", "all"]) for x in ("Y", "YO")}
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    report["decision"] = {"pool1_delta": float(t1["d_all"]), "pool1_se": float(t1["se_all"]),
                          "pool0_delta": float(t0["d_all"]), "buckets": buckets,
                          "alone_minus_no_style": learned,
                          "undecidable": bool(max(learned.values()) <= 0),
                          "continue": bool(t1["d_all"] >= 0.010 and t0["d_all"] > 0 and min(buckets.values()) >= -0.005)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("DECISION:", json.dumps(report["decision"]), flush=True)


if __name__ == "__main__":
    main()
