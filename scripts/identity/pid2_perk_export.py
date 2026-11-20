"""Per-k style multiplier on the official added set (A-a of pid2_cheap; passed the 11/20 switch rule
on sealed S1, 2026-11-20).

ogsa12w added-set score per part: S = opening + category + L1 + L2 + 16 * sum_n w_n style_n with
(w_B, w_A1, w_A2) = 1.5 * (0.25, 0.375, 0.375), i.e. style multiplier m = 1.5. Here the style term of
a k-game question is scaled by m_k / 1.5 with m_k read from outputs/pid/cheap/report.json; nothing is
refitted. The rebuilt base must reproduce the ogsa12w top-5 exactly. Original test-set rows are
copied from ogsa12w. Writes submissions/pid/<tag>-perk.csv; refuses to overwrite.

    python -m scripts.identity.pid2_perk_export --tag <job or date tag>
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity.assign import top5
from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1
from scripts.identity.pid2_qmerge_export import BASE, COLS, DAY0, ENC, SUB
from scripts.identity.pid2_style_export import STORE

CHEAP = ROOT / "outputs" / "pid" / "cheap" / "report.json"
BASE_M = 1.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    target = SUB / f"{args.tag}-perk.csv"
    assert not target.exists(), target
    rep = json.loads(CHEAP.read_text())
    assert rep["s1"]["verdict"]["perk"]["pass"], "per-k did not pass the switch rule"
    mk = {int(k): float(v) for k, v in rep["perk"]["chosen"].items()}
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")
    out = base.copy()
    report = {"m_k": mk, "base": str(BASE.relative_to(ROOT)), "parts": {}}
    for part, prefix in (("public", "pub2"), ("private", "priv2")):
        parts = {}
        for name, (suffix, _) in ENC.items():
            d = DAY0 / f"{part}2_d4_9600{suffix}"
            parts[name] = dict(np.load(d / "parts.npz"))
            if name == "B":
                cand = np.load(d / "candidates.npy", allow_pickle=True).astype(str)
                qids = pd.read_parquet(d / "predictions.parquet").question_id.astype(str).tolist()
        ml = dict(np.load(DAY0 / f"{part}2_d4_9600_seqB_ml" / "parts.npz"))
        rest = (parts["B"]["opening"] + parts["B"]["category"]
                + choice["lambda1"] * ml["ml_L1"] + choice["lambda2"] * ml["ml_L2"])
        style = 16 * sum(w * parts[n]["style"] for n, (_, w) in ENC.items())
        np.testing.assert_array_equal(cand[top5(rest + style)], base.loc[qids, COLS].to_numpy())
        _, _, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
        k_of = q_meta.groupby("question_id", sort=False)["num_games"].first()
        ks = k_of.loc[qids].to_numpy().astype(int)
        assert set(ks) == {1, 2, 3, 5} and all((ks == k).sum() == 100 for k in (1, 2, 3, 5)), np.unique(ks, return_counts=True)
        factor = np.array([mk[k] / BASE_M for k in ks])[:, None]
        ranked = cand[top5(rest + factor * style)]
        out.loc[qids, COLS] = ranked
        ch = ranked[:, 0] != base.loc[qids, "top_1"].to_numpy()
        report["parts"][prefix] = {"top1_changed": int(ch.sum()),
                                   "by_k": {k: int(ch[ks == k].sum()) for k in (1, 2, 3, 5)}}
        print(prefix, json.dumps(report["parts"][prefix]), flush=True)
    orig = base.index[base.index.str.startswith(("pub_", "priv_"))]
    pd.testing.assert_frame_equal(out.loc[orig], base.loc[orig])
    assert len(out) == 1600
    for q, row in out.iterrows():
        assert len(set(row)) == 5 and all(v.split("_")[0] == q.split("_")[0] for v in row)
    out.to_csv(target)
    pd.testing.assert_frame_equal(pd.read_csv(target, dtype=str).set_index("question_id"), out)
    report["path"] = str(target.relative_to(ROOT))
    report["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (CHEAP.parent / "perk_export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
