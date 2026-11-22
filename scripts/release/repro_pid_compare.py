"""Competition 1 reproducibility audit, step 3 (2026-11-22): compare the clean-tree rebuild with the
main repo. Submission files byte for byte; intermediate arrays by max abs difference (first place
a difference appears); original test-set predictions against the final file's pub_ / priv_ rows.

    python -m scripts.release.repro_pid_compare --main /work/jason1122/go_aicup_2026 \
        --new /work/jason1122/repro_pid_1122 --job <repro job id> --out <report.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["top_1", "top_2", "top_3", "top_4", "top_5"]
FILES = {"lik": "817714-lik.csv", "likeb": "820880-likeb.csv", "comb-probe": "821010-comb-probe.csv",
         "comb-pseudo": "824896-comb-pseudo.csv"}
DAY0_DIRS = ["2_d4_9600", "2_d4_9600_seqB", "2_d4_9600_e2eogs9600", "2_d4_9600_e2eogsA2", "2_d4_9600_seqB_ml"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows_diff(a: Path, b: Path) -> dict:
    x = pd.read_csv(a, dtype=str).set_index("question_id")
    y = pd.read_csv(b, dtype=str).set_index("question_id").loc[x.index]
    out = {}
    for pre in ("pub_", "priv_", "pub2_", "priv2_"):
        m = x.index.str.startswith(pre)
        out[pre] = {"top1": int((x.loc[m, "top_1"] != y.loc[m, "top_1"]).sum()),
                    "any": int((x.loc[m, COLS] != y.loc[m, COLS]).any(axis=1).sum())}
    return out


def npz_diff(a: Path, b: Path) -> dict:
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    out = {}
    for k in sorted(set(x.files) | set(y.files)):
        if k not in x.files or k not in y.files:
            out[k] = "missing"
            continue
        u, v = x[k], y[k]
        if u.shape != v.shape:
            out[k] = f"shape {u.shape} vs {v.shape}"
        elif u.dtype.kind in "fc":
            out[k] = float(np.nanmax(np.abs(u.astype(np.float64) - v.astype(np.float64)))) if u.size else 0.0
        else:
            out[k] = "equal" if np.array_equal(u, v) else f"{int((u != v).sum())} differ"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    M, N = a.main, a.new
    rep: dict = {"files": {}, "day0": {}, "likelihood": {}, "original_set": {}}
    for tag, ref in FILES.items():
        new = N / "submissions" / "pid" / f"{a.job}-{tag}.csv"
        old = M / "submissions" / "pid" / ref
        rep["files"][tag] = {"ref": ref, "exists": new.exists()}
        if new.exists():
            rep["files"][tag].update(identical=sha(new) == sha(old), sha_new=sha(new)[:8], sha_ref=sha(old)[:8],
                                     rows=rows_diff(old, new))
    ml_new, ml_old = N / "submissions/pid/hung-seqB-ml.csv", M / "submissions/pid/hung-seqB-ml.csv"
    if ml_new.exists():
        rep["files"]["hung-seqB-ml"] = {"identical": sha(ml_new) == sha(ml_old), "rows": rows_diff(ml_old, ml_new)}
    for part in ("public", "private"):
        for suf in DAY0_DIRS:
            d = f"{part}{suf}"
            for f in ("parts.npz",):
                pn, po = N / "outputs/pid/day0_1112" / d / f, M / "outputs/pid/day0_1112" / d / f
                if pn.exists() and po.exists():
                    rep["day0"][f"{d}/{f}"] = npz_diff(po, pn)
        h_new = N / "outputs/pid/day0_1112" / f"{part}_hungarian" / "submission.csv"
        h_old = M / "outputs/pid/day0_1112" / f"{part}_hungarian" / "submission.csv"
        if h_new.exists():
            rep["original_set"][part] = {"identical_to_old_run": sha(h_new) == sha(h_old)}
            final = pd.read_csv(M / "submissions/pid/824896-comb-pseudo.csv", dtype=str).set_index("question_id")
            x = pd.read_csv(h_new, dtype=str).set_index("question_id")
            rep["original_set"][part]["rows_vs_824896"] = int((final.loc[x.index, COLS] != x[COLS]).any(axis=1).sum())
    for f in ("official/positions.npz", "official/pub2.npz", "official/priv2.npz"):
        pn, po = N / "outputs/pid/likelihood" / f, M / "outputs/pid/likelihood" / f
        if pn.exists() and po.exists():
            rep["likelihood"][f] = npz_diff(po, pn)
    for f in ("rich/official.npz", "rich/bench.npz"):
        pn, po = N / "data/processed/identity/likelihood" / f, M / "data/processed/identity/likelihood" / f
        if pn.exists() and po.exists():
            rep["likelihood"][f] = npz_diff(po, pn)
    for n in ("B", "A1", "A2"):
        for part in ("public", "private"):
            pn = N / f"outputs/pid/likelihood/official/vectors_{n}_{part}.npy"
            po = M / f"outputs/pid/likelihood/official/vectors_{n}_{part}.npy"
            if pn.exists() and po.exists():
                rep["likelihood"][f"vectors_{n}_{part}"] = float(np.abs(np.load(pn) - np.load(po)).max())
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
