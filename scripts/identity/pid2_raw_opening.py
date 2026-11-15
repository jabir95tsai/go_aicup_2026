"""Added set: does the opening model identify better on raw board orientation than on the
D4-normalised one (2026-11-15, CPU)?

Only the coordinates the opening model reads change. Arm A (base) builds the population
prior, the galleries and the questions on the D4-normalised prefixes, arm B on the raw
ones; style vectors, fusion weights (opening + 16 * w * cosine + 1 * category), opening
parameters, bank, draws and the category term are shared. No assignment, nothing is tuned.

    smoke   pool 0, draws 100-101: engineering checks and timing, no pool 1 output
    full    pool 0 draws 100-119; pool 1 draws 100-129 only if pool 0 gives delta >= +0.003

    python -m scripts.identity.pid2_raw_opening --stage smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening, twocolor
from go_aicup.tasks.identity.predict import PLIES
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_paired import BANK, BENCH, points
from scripts.identity.pid2_transductive import load

OUT = ROOT / "outputs" / "pid" / "pid2_raw_opening"
PHASE1 = ROOT / "outputs" / "pid" / "pid2_diag"
FIRST = 100
DRAWS = {0: 20, 1: 30}
WEIGHTS = (16.0, 1.0)
OPENING = {"moves": 8, "beta": 0.1, "alpha": 5.0, "slot_moves": 4}
GATE = {"pool0": 0.003, "pool1": 0.005, "z": 2.5, "bucket": -0.002}
ARMS = ("d4", "raw")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def engineering(env, run: str, raw: np.ndarray) -> dict:
    """Both representations hold the same games in the same order: the cached D4 prefix is
    the canonical image of the raw one row by row, every entry is a board point."""
    d4 = env["prefix"]
    summary = env["summary"]
    saved = np.load(BENCH / run / "embeddings.npz")
    meta = data.load_meta()
    report = {
        "games": int(len(meta)),
        "shape_d4": list(d4.shape), "shape_raw": list(raw.shape),
        "d4_is_canonical_image_of_raw": bool(np.array_equal(data.canonical_d4(raw), d4)),
        "all_points_valid": bool(((raw >= 0) & (raw < 361) & (d4 >= 0) & (d4 < 361)).all()),
        "rows_changed_by_d4": float((raw != d4).any(axis=1).mean()),
        "min_game_length": int(meta["length"].min()),
        "prefix_covers_opening": bool(meta["length"].min() >= PLIES and PLIES >= 2 * OPENING["moves"]),
        "opening_matches_summary": bool(all(summary["opening"][k] == v for k, v in OPENING.items())),
        "weights_match_summary": bool((summary["lambda"], summary["mu"]) == WEIGHTS),
        "w": summary["w"], "w_matches_vectors": bool(float(saved["w"]) == summary["w"]),
        "style": summary["style"],
        "vectors_cover_bank": bool(np.array_equal(saved["game_index"], np.sort(env["bank"]["game"].to_numpy()))),
        "sha256": {
            "embeddings.npz": sha256(BENCH / run / "embeddings.npz"),
            "summary.json": sha256(BENCH / run / "summary.json"),
            "pid2_bank.parquet": sha256(BANK),
            f"train_prefix_{PLIES}.npy": sha256(data.OUT_DIR / f"train_prefix_{PLIES}.npy"),
            f"train_prefix_{PLIES}_d4.npy": sha256(data.OUT_DIR / f"train_prefix_{PLIES}_d4.npy"),
        },
    }
    report["failed"] = [k for k, v in report.items() if v is False]
    if report["shape_d4"] != report["shape_raw"]:
        report["failed"].append("shape")
    return report


def run_pool(env, reps: dict, pool: int, n: int) -> pd.DataFrame:
    summary = env["summary"]
    lam, mu = WEIGHTS
    pos = env["position"]
    rows = []
    for draw in range(FIRST, FIRST + n):
        t0 = time.time()
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
        query = list(q["games"])
        ks = q["k"].to_numpy()
        style = summary["w"] * twocolor.style_scores(env["vectors"], [pos[g] for g in gallery],
                                                     [pos[g] for g in query])
        rest = lam * style + mu * twocolor.category_scores(env["codes"], gallery, query, 4)
        top = {}
        for arm in ARMS:
            prefix, hashes = reps[arm]
            op = twocolor.opening_scores(prefix, hashes, gallery, query, env["color"], env["populations"],
                                         summary["opening"])
            top[arm] = (op + rest).argmax(axis=1)
            rows.append({"pool": pool, "draw": draw, "arm": arm, "what": "fused",
                         **bucket_means(points(op + rest, candidates, q), ks)})
            rows.append({"pool": pool, "draw": draw, "arm": arm, "what": "opening only",
                         **bucket_means(points(op, candidates, q), ks)})
        changed = (top["d4"] != top["raw"]).astype(float)
        rows.append({"pool": pool, "draw": draw, "arm": "raw", "what": "top-1 changed", **bucket_means(changed, ks)})
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        print(f"pool {pool} draw {draw}: d4 {rows[-5]['all']:.4f} raw {rows[-3]['all']:.4f}  "
              f"{time.time() - t0:.1f}s  peak RSS {rss:.1f} GiB", flush=True)
    return pd.DataFrame(rows)


def check_base(frame: pd.DataFrame, run: str) -> dict:
    """The D4 arm against the phase-1 diagnostic's full score, draw by draw."""
    old = pd.read_csv(PHASE1 / run / "d1_per_draw.csv")
    old = old[old["arm"] == "full"].set_index(["pool", "draw"])[COLS]
    new = frame[(frame["arm"] == "d4") & (frame["what"] == "fused")].set_index(["pool", "draw"])[COLS]
    diff = float(np.abs(old.loc[new.index].to_numpy() - new.to_numpy()).max())
    return {"draws": len(new), "max_abs_diff": diff, "reproduced": bool(diff < 1e-12)}


def compare(frame: pd.DataFrame, pool: int) -> dict:
    part = frame[frame["pool"] == pool]
    out = {}
    for what in ("fused", "opening only"):
        table = paired(part[part["what"] == what], "d4")
        row = table[table["arm"] == "raw"].iloc[0]
        se = float(row["se_all"])
        out[what] = {"draws": int(row["draws"]),
                     "d4": {c: float(table[table["arm"] == "d4"].iloc[0][c]) for c in COLS},
                     "raw": {c: float(row[c]) for c in COLS},
                     "delta": {c: float(row[f"d_{c}"]) for c in COLS},
                     "se": se, "z": float(row["d_all"] / se) if se > 0 else float("nan"),
                     "pos": float(row["pos"])}
        frame_d = part[part["what"] == what].pivot(index="draw", columns="arm", values=COLS)
        out[what]["se_k"] = {c: float((frame_d[c]["raw"] - frame_d[c]["d4"]).std(ddof=1) / np.sqrt(len(frame_d)))
                             for c in COLS}
    changed = part[part["what"] == "top-1 changed"]
    out["top1_changed"] = {c: float(changed[c].mean()) for c in COLS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    out_dir = OUT / (args.run if args.stage == "full" else f"{args.run}_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    env = load(args.run)                                     # D4 prefix, hashes, populations, bank
    env["vectors"] = np.load(BENCH / args.run / "embeddings.npz")[env["summary"]["style"].split("-")[1]]
    raw = data.opening_prefix(data.load_meta(), PLIES)
    reps = {"d4": (env["prefix"], env["hashes"]), "raw": (raw, opening.prefix_hashes(raw))}
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    report = {"run": args.run, "stage": args.stage, "weights": list(WEIGHTS), "opening": OPENING, "gate": GATE,
              "engineering": engineering(env, args.run, raw)}
    print(json.dumps(report["engineering"], indent=1), flush=True)

    def save() -> None:
        report["seconds"] = time.time() - started
        report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
        (out_dir / "report.json").write_text(json.dumps(report, indent=1))

    save()
    if report["engineering"]["failed"]:
        raise SystemExit(f"engineering check failed: {report['engineering']['failed']}")

    frame = run_pool(env, reps, 0, 2 if args.stage == "smoke" else DRAWS[0])
    frame.to_csv(out_dir / "per_draw.csv", index=False)
    report["base_check_pool0"] = check_base(frame, args.run)
    report["pool0"] = compare(frame, 0)
    print(json.dumps({k: report[k] for k in ("base_check_pool0", "pool0")}, indent=1), flush=True)
    save()
    if not report["base_check_pool0"]["reproduced"]:
        raise SystemExit("the D4 base is not reproduced")
    if args.stage == "smoke":
        print(f"smoke done in {report['seconds']:.0f}s, peak RSS {report['peak_rss_gib']:.1f} GiB")
        return

    report["pool0_pass"] = bool(report["pool0"]["fused"]["delta"]["all"] >= GATE["pool0"])
    if not report["pool0_pass"]:
        report["verdict"] = "stopped at pool 0; raw opening not adopted, pool 1 not scored"
        save()
        print(report["verdict"])
        return

    frame = pd.concat([frame, run_pool(env, reps, 1, DRAWS[1])], ignore_index=True)
    frame.to_csv(out_dir / "per_draw.csv", index=False)
    report["base_check_pool1"] = check_base(frame[frame["pool"] == 1], args.run)
    report["pool1"] = s = compare(frame, 1)
    fused = s["fused"]
    report["pool1_pass"] = bool(report["base_check_pool1"]["reproduced"] and fused["delta"]["all"] >= GATE["pool1"]
                                and fused["z"] >= GATE["z"]
                                and min(fused["delta"][str(k)] for k in cv.PID2_K) >= GATE["bucket"])
    report["verdict"] = "candidate (pipeline unchanged)" if report["pool1_pass"] else "raw opening not adopted"
    save()
    print(json.dumps({k: report[k] for k in ("base_check_pool1", "pool1", "pool1_pass", "verdict")}, indent=1))
    print(f"done in {report['seconds']:.0f}s, peak RSS {report['peak_rss_gib']:.1f} GiB -> {out_dir}")


if __name__ == "__main__":
    main()
