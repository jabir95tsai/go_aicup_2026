"""Multi-level opening back-off on the added competition-1 test set, stage 0 (2026-11-17).

Feasibility diagnostic only: no parameter is fitted and no score is computed. For every
own move among a question game's first N_OWN moves, on the development draws
(pool 0, draws 100-119):

    L0 hit   the true candidate has a gallery game of the same colour with the same
             exact (D4-canonical) prefix through this move
    L1 / L2  the true candidate has a gallery game of the same colour with the same
             local outcome key (multilevel.corner_keys / reply_keys)

and the same for the other candidates (mean over the 590 others, from the number of
candidates holding the key). Gate 0 (written before the run): on moves the true
candidate misses at L0, L1 or L2 true-candidate hit rate minus other-candidate hit rate
>= 0.05.

    python -m scripts.identity.pid2_multilevel keys    # CPU: local keys of the bank games
    python -m scripts.identity.pid2_multilevel diag    # CPU: hit rates per k / move range
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import multilevel, opening
from go_aicup.tasks.identity.predict import train_prefix
from scripts.identity.pid2_bench import load_bank

OUT = ROOT / "outputs" / "pid" / "pid2_multilevel"
N_OWN = 15
DEV_DRAWS = range(100, 120)
GATE = 0.05
_C_CAND = np.int64(0x2545F4914F6CDD1D)
_C_COLOR = np.int64(0x5851F42D4C957F2D)

_PREFIX = None
_COLOR = None


def _init(prefix, color):
    global _PREFIX, _COLOR
    _PREFIX, _COLOR = prefix, color


def _keys_of(game: int) -> np.ndarray:
    return multilevel.game_keys(_PREFIX[game].astype(np.int64), int(_COLOR[game]), N_OWN)


def build_keys(args) -> None:
    meta, prefix = train_prefix("d4")
    bank, _ = load_bank(meta)
    color = meta["player_color"].to_numpy().astype(np.int64)
    games = np.sort(bank["game"].unique())
    with Pool(args.workers, initializer=_init, initargs=(prefix, color)) as pool:
        keys = np.stack(pool.map(_keys_of, games.tolist(), chunksize=256))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "keys.npz", game_index=games, keys=keys)
    print(f"keys of {len(games)} games: {keys.shape}; L1 none {np.mean(keys[:, :, 1] == 0):.3f}, "
          f"L2 none {np.mean(keys[:, :, 3] == 0):.3f}")


def _combine(key, cand, col):
    with np.errstate(over="ignore"):
        return key ^ ((cand + 1) * _C_CAND) ^ ((col + 1) * _C_COLOR)


def _with_color(key, col):
    with np.errstate(over="ignore"):
        return key ^ ((col + 1) * _C_COLOR)


def level_rates(gal_key, gal_cand, gal_col, q_key, q_cand, q_col, n_cand):
    """(true hit [n], other-candidate hit rate [n], has state [n]) for query move keys."""
    ok = gal_key != 0
    triples = np.unique(_combine(gal_key[ok], gal_cand[ok], gal_col[ok]))
    # number of distinct candidates holding each (colour, key)
    pairs = np.unique(np.stack([_with_color(gal_key[ok], gal_col[ok]), gal_cand[ok]], axis=1), axis=0)
    ck, counts = np.unique(pairs[:, 0], return_counts=True)

    def member(sorted_arr, x):
        pos = np.minimum(np.searchsorted(sorted_arr, x), len(sorted_arr) - 1)
        return sorted_arr[pos] == x, pos

    has = q_key != 0
    true_hit, _ = member(triples, _combine(q_key, q_cand, q_col))
    true_hit &= has
    found, pos = member(ck, _with_color(q_key, q_col))
    n_with = np.where(found & has, counts[pos], 0)
    other = (n_with - true_hit) / (n_cand - 1)
    return true_hit, other, has


def diag(args) -> None:
    meta, prefix = train_prefix("d4")
    bank, _ = load_bank(meta)
    color = meta["player_color"].to_numpy().astype(np.int64)
    saved = np.load(OUT / "keys.npz")
    row = np.full(len(meta), -1)
    row[saved["game_index"]] = np.arange(len(saved["game_index"]))
    local = saved["keys"]                                        # [games, N_OWN, 4]
    hashes = opening.prefix_hashes(prefix[saved["game_index"], :2 * N_OWN]).astype(np.int64)
    own_idx = np.arange(N_OWN)

    def move_keys(games):
        """flat per-move arrays for a list of games: (L0, L1, L2 outcome keys, colour, own index)."""
        r = row[games]
        col = color[games]
        plies = col[:, None] + 2 * own_idx[None, :]              # [g, N_OWN]
        l0 = hashes[r[:, None], plies + 1]
        return (l0.ravel(), local[r][:, :, 1].ravel(), local[r][:, :, 3].ravel(),
                np.repeat(col, N_OWN), np.tile(own_idx, len(games)))

    records = []
    for d in DEV_DRAWS:
        candidates, gallery, q = cv.draw_pid2(bank, 0, d)
        n_cand = len(candidates)
        index = {p: i for i, p in enumerate(candidates)}
        g_games = np.concatenate(gallery)
        g_cand = np.repeat(np.arange(n_cand), [len(g) for g in gallery])
        gk = move_keys(g_games)
        g_cand_m = np.repeat(g_cand, N_OWN)
        q_games = np.concatenate(q["games"].to_list())
        q_cand = np.repeat([index[p] for p in q["player_id"]], [len(g) for g in q["games"]])
        q_k = np.repeat(q["k"].to_numpy(), [len(g) for g in q["games"]])
        qk = move_keys(q_games)
        q_cand_m, q_k_m = np.repeat(q_cand, N_OWN), np.repeat(q_k, N_OWN)
        frame = {"draw": d, "k": q_k_m, "own": qk[4]}
        for name, j in (("L0", 0), ("L1", 1), ("L2", 2)):
            hit, other, has = level_rates(gk[j], g_cand_m, gk[3], qk[j], q_cand_m, qk[3], n_cand)
            frame[f"{name}_true"], frame[f"{name}_other"], frame[f"{name}_has"] = hit, other, has
        records.append(pd.DataFrame(frame))
    moves = pd.concat(records, ignore_index=True)

    def summarise(sub: pd.DataFrame) -> dict:
        out = {"moves": len(sub)}
        for name in ("L1", "L2"):
            has = sub[f"{name}_has"]
            out[f"{name}_state"] = float(has.mean())
            s = sub[has]
            out[f"{name}_true"] = float(s[f"{name}_true"].mean())
            out[f"{name}_other"] = float(s[f"{name}_other"].mean())
            out[f"{name}_diff"] = out[f"{name}_true"] - out[f"{name}_other"]
        return out

    rows = []
    l0_rate = {}
    for rng_name, lo, hi in (("own 0-7", 0, 8), ("own 8-14", 8, N_OWN), ("own 0-14", 0, N_OWN)):
        span = moves[(moves["own"] >= lo) & (moves["own"] < hi)]
        l0_rate[rng_name] = {"L0_true": float(span["L0_true"].mean()), "L0_other": float(span["L0_other"].mean())}
        for subset, sel in (("all moves", np.ones(len(span), bool)), ("L0 miss", ~span["L0_true"].to_numpy())):
            part = span[sel]
            for k in (*cv.PID2_K, "all"):
                s = part if k == "all" else part[part["k"] == k]
                rows.append({"range": rng_name, "subset": subset, "k": k, **summarise(s)})
    table = pd.DataFrame(rows)
    # draw-to-draw spread of the gate statistic
    gate_rows = moves[(moves["own"] < N_OWN) & ~moves["L0_true"]]
    per_draw = {name: [float(g.loc[g[f"{name}_has"], f"{name}_true"].mean()
                             - g.loc[g[f"{name}_has"], f"{name}_other"].mean())
                       for _, g in gate_rows.groupby("draw")] for name in ("L1", "L2")}
    gate = {name: {"diff": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))}
            for name, v in per_draw.items()}
    passed = any(g["diff"] >= GATE for g in gate.values())
    pd.set_option("display.width", 200)
    print("L0 (exact prefix) hit rates:", json.dumps(l0_rate, indent=1))
    print(table.round(4).to_string(index=False))
    print("gate statistic (own 0-14, L0 miss, all k; mean over 20 draws):", gate)
    print("GATE 0:", "PASS" if passed else "FAIL", f"(threshold {GATE})")
    table.to_csv(OUT / "diag.csv", index=False)
    (OUT / "diag.json").write_text(json.dumps({"l0": l0_rate, "gate": gate, "threshold": GATE,
                                               "passed": passed, "draws": list(DEV_DRAWS),
                                               "n_own": N_OWN}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("keys")
    p.add_argument("--workers", type=int, default=12)
    p.set_defaults(func=build_keys)
    sub.add_parser("diag").set_defaults(func=diag)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
