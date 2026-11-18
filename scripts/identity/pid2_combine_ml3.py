"""Write the two E2E14400 probe files mix3ml.csv and ml14400.csv (2026-11-19).

Same fusion as hung-seqB-e2emix-ml.csv (opening + category + 16 * style + ML, lambda1/lambda2 from stage 1)
with only the added-set style term changed, matching the sealed arms of pid2_ml_mix_compare.py (job 795581):
  mix3ml   style = 0.5 seqB + 0.25 E2E9600 + 0.25 E2E14400
  ml14400  style = 0.5 seqB + 0.5 E2E14400
Original test-set rows copied from 772748-hung.csv. Checks that the current base file is reproduced first.
Refuses to overwrite.

    python -m scripts.identity.pid2_combine_ml3
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT
from go_aicup.tasks.identity.assign import top5

DAY0 = ROOT / "outputs" / "pid" / "day0_1112"
SUB = ROOT / "submissions" / "pid"
OUT = ROOT / "outputs" / "pid" / "pid2_multilevel" / "ml_mix_compare" / "combine_report.json"
COLS = [f"top_{i}" for i in range(1, 6)]
STYLE = {"mix3ml": {"_seqB": 0.5, "_e2e9600": 0.25, "_e2e14400": 0.25},
         "ml14400": {"_seqB": 0.5, "_e2e14400": 0.5}}


def main() -> None:
    base = pd.read_csv(SUB / "772748-hung.csv", dtype=str).set_index("question_id")
    cur = pd.read_csv(SUB / "hung-seqB-e2emix-ml.csv", dtype=str).set_index("question_id")
    choice = json.loads((ROOT / "outputs/pid/pid2_multilevel/stage1/dev_choice.json").read_text())["chosen"]
    l1, l2 = choice["lambda1"], choice["lambda2"]
    outs = {name: base.copy() for name in STYLE}
    for name in outs:
        assert not (SUB / f"{name}.csv").exists(), name
    report = {name: {"style": STYLE[name], "parts": {}} for name in outs}
    for part, prefix in (("public", "pub2"), ("private", "priv2")):
        src = {s: DAY0 / f"{part}2_d4_9600{s}" for s in ("_seqB", "_e2e9600", "_e2e14400", "_seqB_ml")}
        pred = pd.read_parquet(src["_seqB"] / "predictions.parquet")
        ids, ks = pred.question_id.astype(str).tolist(), pred.num_games.to_numpy()
        cand = np.load(src["_seqB"] / "candidates.npy", allow_pickle=True).astype(str)
        p = {}
        for s, d in src.items():
            assert pd.read_parquet(d / "predictions.parquet").question_id.astype(str).tolist() == ids
            assert np.array_equal(np.load(d / "candidates.npy", allow_pickle=True).astype(str), cand)
            p[s] = dict(np.load(d / "parts.npz"))
        for s in ("_e2e9600", "_e2e14400", "_seqB_ml"):
            np.testing.assert_array_equal(p[s]["opening"], p["_seqB"]["opening"])
            np.testing.assert_array_equal(p[s]["category"], p["_seqB"]["category"])
        rest = p["_seqB"]["opening"] + p["_seqB"]["category"]
        ml = l1 * p["_seqB_ml"]["ml_L1"] + l2 * p["_seqB_ml"]["ml_L2"]
        cur_style = 0.5 * (p["_seqB"]["style"] + p["_e2e9600"]["style"])
        cur_top = cur.loc[ids, COLS].to_numpy()
        np.testing.assert_array_equal(cand[top5(rest + 16 * cur_style + ml)], cur_top)
        for name, weights in STYLE.items():
            style = sum(w * p[s]["style"] for s, w in weights.items())
            sc = rest + 16 * style + ml
            assert np.isfinite(sc).all() and sc.shape == (400, 600)
            ranked = cand[top5(sc)]
            outs[name].loc[ids, COLS] = ranked
            ch = ranked[:, 0] != cur_top[:, 0]
            report[name]["parts"][prefix] = {"top1_changed_vs_e2emix_ml": int(ch.sum()),
                                             "by_k": {int(k): int(ch[ks == k].sum()) for k in (1, 2, 3, 5)}}
    orig = base.index[base.index.str.startswith(("pub_", "priv_"))]
    for name, out in outs.items():
        pd.testing.assert_frame_equal(out.loc[orig], base.loc[orig])
        for q, row in out.iterrows():
            assert len(set(row)) == 5 and all(v.split("_")[0] == q.split("_")[0] for v in row)
        target = SUB / f"{name}.csv"
        out.to_csv(target)
        pd.testing.assert_frame_equal(pd.read_csv(target, dtype=str).set_index("question_id"), out)
        report[name]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
