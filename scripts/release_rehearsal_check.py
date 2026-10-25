"""Gate of the one-command rehearsal (RESULTS A-054), CPU only.

Compares scripts/run_release.sh output on the fake release with the rehearsal
outputs that already passed their gates (same raw files):
  - four submission files, 400 rows each;
  - competition 1: top-5 identical to identity/predict.py with the A-053
    insurance vectors (outputs/rehearsal/identity_insurance);
  - competition 2: HumanSL classes identical to the A-042 extraction path under
    the current model on >= 398/400 questions per file; V2 fallback files
    identical to the A-030 rehearsal submissions.

    python scripts/release_rehearsal_check.py --out outputs/rehearsal_t3_1025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from go_aicup.paths import ROOT
from go_aicup.tasks.identity.predict import predict
from go_aicup.tasks.strength.humansl_predict import MODEL, aggregate

REHEARSAL = ROOT / "outputs" / "rehearsal"
SAME_MIN = 398


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    result, ok = {}, True
    files = ["pid_public", "pid_private", "rank_public", "rank_private"]
    rows = {name: len(pd.read_csv(out / "submissions" / f"{name}.csv")) if (out / "submissions" / f"{name}.csv").exists()
            else 0 for name in files}
    result["submission_rows"] = rows
    ok &= all(n == 400 for n in rows.values())
    for part in ("public", "private"):
        new = pd.read_parquet(out / "pid" / part / "predictions.parquet")
        ref, _, _, _ = predict(REHEARSAL / "store", f"pool_{part}", f"pid_{part}", "d4", "hungarian",
                               insurance_dir=REHEARSAL / "identity_insurance")
        cols = [f"top{j}" for j in range(1, 6)]
        merged = new.merge(ref, on="question_id", suffixes=("", "_ref"))
        same = int((merged[cols].to_numpy() == merged[[c + "_ref" for c in cols]].to_numpy()).all(axis=1).sum())
        result[f"pid_{part}_top5_identical"] = same
        ok &= same == len(merged) == 400
    for name in ("rank_public", "rank_private"):
        old_work = REHEARSAL / "strength" / name
        ref = aggregate(old_work, MODEL, None)[["question_id", "rank_id"]]
        new = pd.read_parquet(out / "rank" / name / "humansl_predictions.parquet")[["question_id", "rank_id", "fallback_v2"]]
        merged = new.merge(ref, on="question_id", suffixes=("", "_ref"))
        same = int((merged["rank_id"] == merged["rank_id_ref"]).sum())
        v2_new = pd.read_csv(out / "submissions" / f"{name}_v2_fallback.csv")
        v2_old = pd.read_csv(old_work / "submission.csv")
        v2_same = bool(v2_new.equals(v2_old))
        result[name] = {"humansl_classes_identical": same, "questions": len(merged),
                        "fallback_v2": int(merged["fallback_v2"].sum()), "v2_fallback_identical": v2_same}
        ok &= same >= SAME_MIN and v2_same
    result["pass"] = bool(ok)
    print(json.dumps(result, indent=1))
    (out / "rehearsal_check.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
