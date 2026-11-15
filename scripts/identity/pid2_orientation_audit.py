"""Orientation audit of the official added set (2026-11-15, CPU): do the candidate galleries
and the questions sit in the same board orientation? Needed before a raw-orientation
opening model (scripts/identity/pid2_raw_opening.py) is applied to the official files.

No predicted identity is used. Per part (public / private) and player colour, the gallery
games are compared with the question games through total-variation distances of

    d4        which of the 8 board symmetries maps the first 16 plies to their canonical image
    corner    quadrant of the game's first move (4 quadrants + centre lines)
    own_j     the player's own j-th move as an absolute point, j = 0..3

each weighted per game and per unit (candidate / question), for all games and by HD 0 / 1.

Reference   the same statistics on the bench (cv.draw_pid2, pools 0 and 1, draws 100-129):
            training games in their recorded orientation, the same gallery / question sizes.
Control     the official question games with a random board symmetry applied to a share of
            them (5%..100%), 20 seeds: what the statistics look like under rotation.

Rule, fixed before the data were read (primary: d4, corner, own_0, per game, all HD):
    anomalous      a primary cell lies above every reference draw and at least 20% of the
                   way from the reference mean to the fully rotated control
    not flagged    no such cell, and the fully rotated control lies above every reference
                   draw in every primary cell
    undetermined   otherwise
"Not flagged" means no gross mismatch was seen; it cannot rule out every per-game rotation.

    python -m scripts.identity.pid2_orientation_audit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data
from go_aicup.tasks.identity.predict import PLIES, store_prefix
from scripts.identity.pid2_paired import BANK, HANDICAP

OUT = ROOT / "outputs" / "pid" / "pid2_orientation"
STORE = ROOT / "outputs" / "pid" / "day0_1112" / "store"
DEPTH = 16
OWN = 4
SHARES = (0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS = 20
PRIMARY = ("d4", "corner", "own_0")
TABLES = data.d4_tables()


def d4_index(prefix: np.ndarray) -> np.ndarray:
    """Index of the symmetry giving the lexicographically smallest image (first on ties)."""
    images = TABLES[:, prefix]
    alive = np.ones(images.shape[:2], bool)
    for t in range(images.shape[2]):
        vals = np.where(alive, images[:, :, t], np.int16(32767))
        alive &= vals == vals.min(axis=0)
    return alive.argmax(axis=0)


def corner(points: np.ndarray) -> np.ndarray:
    r, c = np.divmod(points.astype(np.int64), 19)
    return np.where((r == 9) | (c == 9), 4, 2 * (r > 9) + (c > 9))


def features(prefix: np.ndarray, color: np.ndarray) -> dict:
    """Per game: integer codes of each statistic and the number of bins."""
    pre = prefix[:, :DEPTH]
    out = {"d4": (d4_index(pre), 8), "corner": (corner(pre[:, 0]), 5)}
    rows = np.arange(len(pre))
    for j in range(OWN):
        out[f"own_{j}"] = (pre[rows, color + 2 * j].astype(np.int64), 361)
    return out


def unit_weights(unit: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(unit, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def tv(a, wa, b, wb, bins) -> float:
    pa = np.bincount(a, weights=wa, minlength=bins)
    pb = np.bincount(b, weights=wb, minlength=bins)
    return float(0.5 * np.abs(pa / pa.sum() - pb / pb.sum()).sum())


def compare(gal: dict, qry: dict) -> dict:
    """gal / qry = {"prefix", "color", "hd", "unit"}. -> {(color, hd, weighting, stat): tv};
    a cell with an empty side is left out."""
    fg, fq = features(gal["prefix"], gal["color"]), features(qry["prefix"], qry["color"])
    out = {}
    for color, hd in ((c, h) for c in (0, 1) for h in ("all", 0, 1)):
        mg = (gal["color"] == color) & ((gal["hd"] == hd) if hd != "all" else True)
        mq = (qry["color"] == color) & ((qry["hd"] == hd) if hd != "all" else True)
        if not mg.any() or not mq.any():
            continue
        weights = {"game": (np.ones(mg.sum()), np.ones(mq.sum())),
                   "unit": (unit_weights(gal["unit"][mg]), unit_weights(qry["unit"][mq]))}
        for name, (wg, wq) in weights.items():
            for stat, (codes, bins) in fg.items():
                out["BW"[color], str(hd), name, stat] = tv(codes[mg], wg, fq[stat][0][mq], wq, bins)
        out["BW"[color], str(hd), "game", "n_gallery"] = int(mg.sum())
        out["BW"[color], str(hd), "game", "n_query"] = int(mq.sum())
    return out


def rotated(qry: dict, share: float, rng) -> dict:
    pick = rng.random(len(qry["prefix"])) < share
    g = np.where(pick, rng.integers(0, 8, len(pick)), 0)
    return {**qry, "prefix": TABLES[g[:, None], qry["prefix"][:, :DEPTH]]}


def official(part: str) -> tuple[dict, dict, dict]:
    sides, info = [], {}
    for name, unit_col in ((f"pool2_{part}", "player_id"), (f"pid2_{part}", "question_id")):
        moves, colors, meta = load_store(name, STORE, mmap=False)
        prefix, usable = store_prefix(moves, colors, meta, "raw", DEPTH)
        if not usable.all():
            raise ValueError(f"{name}: {int((~usable).sum())} games shorter than {DEPTH} plies or not alternating")
        color = (meta["color"].astype(str).to_numpy() == "W").astype(np.int64)
        sides.append({"prefix": prefix[:, :DEPTH], "color": color, "hd": np.minimum(meta["hd"].to_numpy(), 1),
                      "unit": meta[unit_col].astype(str).to_numpy()})
        info[name] = {"games": len(meta), "units": int(meta[unit_col].nunique()),
                      "white_share": float(color.mean()), "hd_counts": meta["hd"].value_counts().to_dict()}
    return sides[0], sides[1], info


def bench(prefix, color, handicap, bank, pool: int, draw: int) -> tuple[dict, dict]:
    candidates, gallery, q = cv.draw_pid2(bank, pool, draw)

    def side(groups) -> dict:
        games = np.concatenate(groups)
        unit = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        return {"prefix": prefix[games, :DEPTH], "color": color[games], "hd": np.minimum(handicap[games], 1),
                "unit": unit}

    return side(gallery), side(list(q["games"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--first", type=int, default=100)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    meta = data.load_meta()
    prefix = data.opening_prefix(meta, PLIES)
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy().astype(np.int64)
    color = meta["player_color"].to_numpy().astype(np.int64)
    bank = pd.read_parquet(BANK)
    rows = []
    for pool in (0, 1):
        for draw in range(args.first, args.first + args.draws):
            gal, qry = bench(prefix, color, handicap, bank, pool, draw)
            for key, value in compare(gal, qry).items():
                rows.append(("reference", f"pool{pool}", draw, 0.0, *key, value))
            if draw < args.first + 5:                      # the audit must also see rotation on the bench
                rng = np.random.default_rng([draw, pool, 7])
                for key, value in compare(gal, rotated(qry, 1.0, rng)).items():
                    rows.append(("reference rotated", f"pool{pool}", draw, 1.0, *key, value))
    print(f"reference done in {time.time() - started:.0f}s", flush=True)

    info = {}
    for part in ("public", "private"):
        gal, qry, info[part] = official(part)
        for key, value in compare(gal, qry).items():
            rows.append(("official", part, 0, 0.0, *key, value))
        for share in SHARES:
            for seed in range(SEEDS):
                rng = np.random.default_rng([seed, int(share * 100), part == "private"])
                for key, value in compare(gal, rotated(qry, share, rng)).items():
                    rows.append(("control", part, seed, share, *key, value))
    frame = pd.DataFrame(rows, columns=["kind", "part", "draw", "share", "color", "hd", "weight", "stat", "value"])
    frame.to_csv(OUT / "all_values.csv", index=False)

    keys = ["color", "hd", "weight", "stat"]
    ref = frame[frame["kind"] == "reference"].groupby(keys)["value"].agg(
        ref_mean="mean", ref_p95=lambda v: v.quantile(0.95), ref_max="max").reset_index()
    ref_rot = frame[frame["kind"] == "reference rotated"].groupby(keys)["value"].mean().rename("ref_rotated")
    off = frame[frame["kind"] == "official"][["part", *keys, "value"]].rename(columns={"value": "official"})
    ctrl = frame[frame["kind"] == "control"].groupby(["part", *keys, "share"])["value"].mean().unstack("share")
    ctrl.columns = [f"rot_{int(s * 100)}" for s in ctrl.columns]
    summary = off.merge(ref, on=keys).merge(ref_rot.reset_index(), on=keys).merge(ctrl.reset_index(), on=["part", *keys])
    counts = summary[summary["stat"].isin(["n_gallery", "n_query"])]
    summary = summary[~summary["stat"].isin(["n_gallery", "n_query"])].copy()
    summary["above_ref_max"] = summary["official"] > summary["ref_max"]
    summary["position"] = (summary["official"] - summary["ref_mean"]) / (summary["rot_100"] - summary["ref_mean"])
    shares = [f"rot_{int(s * 100)}" for s in SHARES]
    seen = summary[shares].to_numpy() > summary[["ref_max"]].to_numpy()
    summary["detect_from"] = [SHARES[int(np.argmax(r))] if r.any() else np.nan for r in seen]
    summary.to_csv(OUT / "summary.csv", index=False)

    primary = summary[(summary["hd"] == "all") & (summary["weight"] == "game") & summary["stat"].isin(PRIMARY)]
    anomalous = primary[primary["above_ref_max"] & (primary["position"] >= 0.2)]
    detects = bool((primary["rot_100"] > primary["ref_max"]).all())
    verdict = "anomalous" if len(anomalous) else ("not flagged" if detects else "undetermined")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    show = ["part", "color", "hd", "weight", "stat", "official", "ref_mean", "ref_p95", "ref_max", "ref_rotated",
            "rot_5", "rot_10", "rot_25", "rot_50", "rot_100", "position", "detect_from", "above_ref_max"]
    print("\n== primary cells (per game, all HD) ==")
    print(primary[show].round(4).to_string(index=False))
    print("\n== per unit, all HD ==")
    print(summary[(summary["hd"] == "all") & (summary["weight"] == "unit")][show].round(4).to_string(index=False))
    print("\n== by HD (per game) ==")
    print(summary[(summary["hd"] != "all") & (summary["weight"] == "game")][show].round(4).to_string(index=False))
    print("\n== official game counts ==")
    print(counts[["part", "color", "hd", "stat", "official"]].to_string(index=False))
    print(json.dumps(info, indent=1, default=int))
    report = {"verdict": verdict, "control_detected_in_every_primary_cell": detects,
              "anomalous_cells": anomalous[["part", "color", "stat", "official", "ref_max", "position"]]
              .to_dict(orient="records"),
              "cells_above_ref_max": int(summary["above_ref_max"].sum()), "cells": int(len(summary)),
              "primary_above_ref_max": primary[primary["above_ref_max"]][["part", "color", "stat", "official",
                                                                         "ref_max", "position"]]
              .to_dict(orient="records"),
              "reference_draws": 2 * args.draws, "control_seeds": SEEDS, "depth": DEPTH, "official": info,
              "seconds": time.time() - started}
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "official"}, indent=1, default=float))


if __name__ == "__main__":
    main()
