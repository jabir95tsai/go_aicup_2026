"""Style-term weight sweep for the added set after A1 (2026-11-19, registered in RESULTS.md before the run).

Fusion as ogsmix.csv: rest = opening + category + stage-1 L1/L2, score = rest + 16 w m (a cos_B + (1 - a) cos_A1).
ogsmix itself is a = 0.5, m = 1. Grid a in {0, 0.25, 0.5, 0.75, 1}, m in {0.75, 1, 1.25, 1.5, 2, 2.5}.
Selection on fresh development draws (pool 0 1800-1819, pool 1 1800-1829, mean over all 50 draws).
If the chosen setting is ogsmix the sealed draws are not opened. Otherwise the chosen setting is scored once
on sealed draws 1900- (pool 0 1900-1919, pool 1 1900-1929) against ogsmix. Switch rule (same-path change):
pool 1 delta >= +0.005, z >= 2.5, every k >= -0.005, pool 0 delta > 0; otherwise keep ogsmix.
"""
from __future__ import annotations

import itertools
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
       "A1": ("/work/jason1122/ogs_a1_cache/A1", "outputs/pid/ge2e/pid_e2e_ogs_A1_YO9600/step9600/best.pt")}
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
M_GRID = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
DEV = ((0, range(1800, 1820)), (1, range(1800, 1830)))
SEALED = ((0, range(1900, 1920)), (1, range(1900, 1930)))
REF = "a0.5_m1"
OUT = Path("outputs/pid/ogs/weight_sweep")


def tag(a: float, m: float) -> str:
    return f"a{a:g}_m{m:g}"


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

    def score(draws, settings):
        rows = []
        for pool, rng in draws:
            for d in rng:
                candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
                query = list(q["games"])
                rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                                env["populations"], env["summary"]["opening"])
                        + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4)
                        + s1["lambda1"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]]
                        + s1["lambda2"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]])
                g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
                c = {k: twocolor.style_scores(v, g, qq) for k, v in vec.items()}
                ks = q["k"].to_numpy()
                for a, m in settings:
                    sc = rest + scale * m * (a * c["B"] + (1 - a) * c["A1"])
                    rows.append({"pool": pool, "draw": d, "arm": tag(a, m),
                                 **bucket_means(points(sc, candidates, q), ks)})
            print(f"pool {pool} draws {rng.start}- done", flush=True)
        return pd.DataFrame(rows)

    dev = score(DEV, list(itertools.product(A_GRID, M_GRID)))
    dev.to_csv(OUT / "dev_per_draw.csv", index=False)
    table = dev.groupby("arm")[["all", "1", "2", "3", "5"]].mean().sort_values("all", ascending=False)
    print(table.round(4).to_string(), flush=True)
    chosen = table.index[0]
    a, m = (float(x[1:]) for x in chosen.split("_"))
    report = {"dev_table": table.reset_index().to_dict(orient="records"), "chosen": chosen,
              "edge": {"a": a in (A_GRID[0], A_GRID[-1]), "m": m in (M_GRID[0], M_GRID[-1])}}
    if chosen == REF:
        report["switch"] = {"pass": False, "note": "development chose ogsmix; sealed draws not opened"}
    else:
        sealed = score(SEALED, [(0.5, 1.0), (a, m)])
        sealed.to_csv(OUT / "sealed_per_draw.csv", index=False)
        t = {pool: paired(sealed[sealed["pool"] == pool], REF).set_index("arm").loc[chosen] for pool in (0, 1)}
        for pool in (0, 1):
            print(f"pool {pool}, {chosen} minus {REF}:\n{t[pool].round(4).to_string()}", flush=True)
        buckets = {str(k): float(t[1][f"d_{k}"]) for k in cv.PID2_K}
        z = float(t[1]["d_all"] / t[1]["se_all"])
        report["switch"] = {"pool1_delta": float(t[1]["d_all"]), "pool1_se": float(t[1]["se_all"]), "z": z,
                            "buckets": buckets, "pool0_delta": float(t[0]["d_all"]),
                            "pass": bool(t[1]["d_all"] >= 0.005 and z >= 2.5 and min(buckets.values()) >= -0.005
                                         and t[0]["d_all"] > 0)}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("RESULT:", json.dumps({k: report[k] for k in ("chosen", "edge", "switch")}), flush=True)


if __name__ == "__main__":
    main()
