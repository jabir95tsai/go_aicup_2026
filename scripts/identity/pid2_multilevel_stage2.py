"""Multi-level opening back-off on the added competition-1 set, stage 2 (2026-11-18).

Three coarser levels (multilevel.game_keys2) on top of the current base candidate
hung-seqB-e2emix-ml (opening + 16 w (0.5 cos_B + 0.5 cos_E2E9600) + category + stage-1 L1/L2,
N 15, beta 0.1, lambda1 = lambda2 = 1):

    L3 reply5    5x5 window around the opponent's previous move (coarser L2)
    L4 reply-off offset from the opponent's previous move, no board content
    L5 follow    offset from the player's own previous move, no board content

Same count likelihood as stage 1 (pid2_multilevel_fuse.level_scores), same population sample.

    python -m scripts.identity.pid2_multilevel_stage2 keys   # CPU: stage-2 keys, bench + population
    python -m scripts.identity.pid2_multilevel_stage2 run    # 1 GPU (style vectors) + CPU

Development pool 0 draws 100-119 choose N in {15, 30}, beta, lambda3/4/5 in {0, .25, .5, 1}
(highest mean; ties -> smaller lambda sum). Sealed draws 1400- (pool 0 1400-1419, pool 1
1400-1429) scored once. Gate against the base (registered in RESULTS.md before the run): pool 1
delta >= +0.005, z >= 2.5, every k >= -0.005, pool 0 delta > 0.
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.tasks.identity import multilevel, twocolor
from scripts.identity.pid2_diag_d123a import bucket_means, paired
from scripts.identity.pid2_multilevel_fuse import BETAS, OUT as STAGE1, STAGE0, PopTable, level_scores
from scripts.identity.pid2_paired import points

RUN = "pid_ge2e_v4_d4_hd_9600"
OUT = STAGE0 / "stage2"
N_MAX = 30
NS = (15, 30)
LAMS = (0.0, 0.25, 0.5, 1.0)
LEVELS = {"L3": (0, 1), "L4": (2, 3), "L5": (4, 5)}
DEV = (0, range(100, 120))
SEALED = ((0, range(1400, 1420)), (1, range(1400, 1430)))
GATE = {"pool1": 0.005, "z": 2.5, "bucket": -0.005, "pool0": 0.0}
STYLE = {"B": ("/home/jason1122/pid2_seqagg_cache", "outputs/pid/pid2_seqagg/head_B/best.pt"),
         "E": ("/home/jason1122/e2e_scratch32_cache/step9600",
               "outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt")}

_PREFIX = None
_COLOR = None


def _init(prefix, color):
    global _PREFIX, _COLOR
    _PREFIX, _COLOR = prefix, color


def _keys2(game: int) -> np.ndarray:
    return multilevel.game_keys2(_PREFIX[game].astype(np.int64), int(_COLOR[game]), N_MAX)


def keys(args) -> None:
    from scripts.identity.pid2_transductive import load
    env = load(RUN)
    prefix, color = env["prefix"], env["color"]
    assert prefix.shape[1] >= 2 * N_MAX and (prefix[:, :2 * N_MAX] < 361).all()
    bench = np.load(STAGE0 / "keys.npz")
    pop = np.load(STAGE1 / "pop_keys.npz")
    # the window parameter refactor must leave the stage-1 keys unchanged
    for g, k in zip(bench["game_index"][::997], bench["keys"][::997]):
        assert np.array_equal(multilevel.game_keys(prefix[g].astype(np.int64), int(color[g]), 15), k)
    started = time.time()
    out = {}
    with Pool(args.workers, initializer=_init, initargs=(prefix, color)) as p:
        for name, games in (("bench", bench["game_index"]), ("pop", pop["game_index"])):
            out[name] = np.stack(p.map(_keys2, games.tolist(), chunksize=512))
            print(f"{name}: {out[name].shape} in {time.time() - started:.0f}s; none share "
                  f"{[round(float((out[name][:, :, j] == 0).mean()), 3) for j in (1, 3, 5)]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "keys2.npz", bench_index=bench["game_index"], bench_keys=out["bench"],
             pop_index=pop["game_index"], pop_keys=out["pop"])


def run(args) -> None:
    import torch
    from go_aicup.tasks.identity import data
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    started = time.time()
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    vec = {}
    for name, (cache, head_file) in STYLE.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite
        head, _ = load_head(Path(head_file), device)
        vec[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
    scale = 16.0 * float(saved["w"])
    print(f"style vectors in {time.time() - started:.0f}s", flush=True)

    b1 = np.load(STAGE0 / "keys.npz")
    k2 = np.load(OUT / "keys2.npz")
    assert np.array_equal(b1["game_index"], k2["bench_index"])
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))

    def rows_of(games):
        r = row[games]
        assert (r >= 0).all(), "game without local keys"
        return r

    keys1 = lambda games: b1["keys"][rows_of(games)]
    keys2 = lambda games: k2["bench_keys"][rows_of(games)]
    pop1f = np.load(STAGE1 / "pop_keys.npz")
    assert np.array_equal(pop1f["game_index"], k2["pop_index"])
    assert not np.isin(k2["pop_index"], b1["game_index"]).any(), "population overlaps the bench"
    pop1 = PopTable(pop1f["keys"], color[pop1f["game_index"]])
    pop2 = PopTable(k2["pop_keys"], color[k2["pop_index"]], ns=NS, levels=LEVELS)
    print(f"population tables in {time.time() - started:.0f}s", flush=True)

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4))
        g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
        cb, ce = (twocolor.style_scores(vec[k], g, qq) for k in ("B", "E"))
        ml1 = (s1["lambda1"] * level_scores(keys1, color, pop1, gallery, query, s1["n"], "L1")[s1["beta"]]
               + s1["lambda2"] * level_scores(keys1, color, pop1, gallery, query, s1["n"], "L2")[s1["beta"]])
        base = rest + scale * (0.5 * cb + 0.5 * ce) + ml1
        ml2 = {(n, lv): level_scores(keys2, color, pop2, gallery, query, n, lv, LEVELS)
               for n in NS for lv in LEVELS}
        return candidates, q, base, ml2

    OUT.mkdir(parents=True, exist_ok=True)
    combos = [(n, b, l3, l4, l5) for n in NS for b in BETAS for l3 in LAMS for l4 in LAMS for l5 in LAMS]
    name_of = lambda c: f"n{c[0]}_b{c[1]:g}_l{c[2]:g}_{c[3]:g}_{c[4]:g}"
    dev_rows = []
    pool, draws = DEV
    for d in draws:
        candidates, q, base, ml = draw(pool, d)
        r = {"draw": d}
        for c in combos:
            n, b, l3, l4, l5 = c
            sc = base + l3 * ml[n, "L3"][b] + l4 * ml[n, "L4"][b] + l5 * ml[n, "L5"][b]
            r[name_of(c)] = points(sc, candidates, q).mean()
        dev_rows.append(r)
        print(f"dev draw {d} ({time.time() - started:.0f}s)", flush=True)
    dev = pd.DataFrame(dev_rows).set_index("draw")
    dev.to_csv(OUT / "dev_grid.csv")
    means = dev.mean()
    key_of = {name_of(c): c for c in combos}
    base_name = name_of((NS[0], BETAS[0], 0.0, 0.0, 0.0))
    best = max(means.index, key=lambda a: (round(means[a], 12), -sum(key_of[a][2:])))
    n, beta, l3, l4, l5 = key_of[best]
    choice = {"chosen": {"n": n, "beta": beta, "lambda3": l3, "lambda4": l4, "lambda5": l5},
              "dev_gain": float(means[best] - means[base_name]),
              "dev_top": (means - means[base_name]).sort_values(ascending=False).head(10).to_dict(),
              "on_boundary": bool(max(l3, l4, l5) == LAMS[-1])}
    (OUT / "dev_choice.json").write_text(json.dumps(choice, indent=1))
    print("dev choice:", json.dumps(choice, indent=1), flush=True)
    if l3 == l4 == l5 == 0:
        print("development chose no stage-2 term; sealed draws not opened", flush=True)
        return

    rows = []
    for pool, draws in SEALED:
        for d in draws:
            candidates, q, base, ml = draw(pool, d)
            ks = q["k"].to_numpy()
            t = {lv: lam * ml[n, lv][beta] for lv, lam in (("L3", l3), ("L4", l4), ("L5", l5))}
            arms = {"base": base, "s2": base + t["L3"] + t["L4"] + t["L5"]}
            for lv, lam in (("L3", l3), ("L4", l4), ("L5", l5)):
                if lam:
                    arms[f"{lv}_only"] = base + t[lv]
            for name, sc in arms.items():
                rows.append({"pool": pool, "draw": d, "arm": name, **bucket_means(points(sc, candidates, q), ks)})
        print(f"sealed pool {pool} done ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "sealed_per_draw.csv", index=False)
    report = {"choice": choice, "stage1": s1, "tables": {}}
    for pool, _ in SEALED:
        table = paired(frame[frame["pool"] == pool], "base")
        report["tables"][f"pool{pool}"] = table.to_dict(orient="records")
        print(f"pool {pool}:\n{table.round(4).to_string(index=False)}", flush=True)
    t1 = pd.DataFrame(report["tables"]["pool1"]).set_index("arm").loc["s2"]
    t0 = pd.DataFrame(report["tables"]["pool0"]).set_index("arm").loc["s2"]
    z = float(t1["d_all"] / t1["se_all"])
    buckets = {str(k): float(t1[f"d_{k}"]) for k in cv.PID2_K}
    passed = (t1["d_all"] >= GATE["pool1"] and z >= GATE["z"]
              and min(buckets.values()) >= GATE["bucket"] and t0["d_all"] > GATE["pool0"])
    report["gate"] = {"pool1_delta": float(t1["d_all"]), "pool1_se": float(t1["se_all"]), "z": z,
                      "buckets": buckets, "pool0_delta": float(t0["d_all"]), "pass": bool(passed), "rule": GATE}
    report["seconds"] = time.time() - started
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("GATE:", "PASS" if passed else "FAIL", json.dumps(report["gate"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("keys")
    p.add_argument("--workers", type=int, default=12)
    p.set_defaults(func=keys)
    sub.add_parser("run").set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
