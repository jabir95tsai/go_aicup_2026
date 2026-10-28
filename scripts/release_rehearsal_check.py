"""Gate of the one-command rehearsal (RESULTS A-054, A-056), CPU only.

Compares scripts/run_release.sh output on the fake release with the rehearsal
outputs that already passed their gates (same raw files):
  - eight submission files (pid x2, rank AH+C x2, AH x2, V2 x2), 400 rows each;
  - competition 1: top-5 identical to identity/predict.py (no L6 assignment, the
    A-059 default) with the A-053 insurance vectors (outputs/rehearsal/identity_insurance);
  - competition 2: HumanSL classes identical to the A-042 extraction path under
    the current model on >= 398/400 questions per file; V2 fallback files
    identical to the A-030 rehearsal submissions;
  - AH+C (A-056): no question falls back or misses a game, and the release-path C
    features match the offline extraction of the same games (matched by move
    sequence) with per-game max |diff| p99 <= 2e-3. Diagnostic only: official
    scores of AH+C and AH against the rehearsal answers.
A-063 adds: twenty files (competition-1 LB-probe variants, all-player AH+C); each
competition-1 variant's top-5 matches a recomputation with the same flags on
>= 398/400 questions; all-player AH+C complete and matching its parquet; the Day-0
audit reports the opening/style agreement for both parts. Diagnostic only: every
competition-1 file's official score against the rehearsal answers.

    python scripts/release_rehearsal_check.py --out outputs/rehearsal_t3_1025
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import numpy as np

from go_aicup.features.humansl import cache_games
from go_aicup.metrics import K_BUCKETS, pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity.predict import predict
from go_aicup.tasks.strength.humansl_predict import MODEL, RANKVEC_RUNS, aggregate, rankvec_features
from go_aicup.tasks.strength.v2.aggregate import official_scores

REHEARSAL = ROOT / "outputs" / "rehearsal"
SAME_MIN, C_DIFF_P99 = 398, 2e-3
# Competition-1 files: suffix -> predict() flags as run_release.sh passes them (A-059, A-060, A-062).
PID_VARIANTS = {"": {"method": "none"},
                "_hungarian": {"method": "hungarian"},
                "_sinkhorn": {"method": "sinkhorn"},
                "_opening_only": {"method": "none", "insurance": False},
                "_style_only": {"method": "none", "use_opening": False},
                "_robust": {"method": "none", "robust_eps": 0.1, "insurance_lambda": 32}}
RANK_SUFFIXES = ("", "_ah", "_allplayers", "_v2_fallback")


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Row-for-row equality as text (CSV reads question_id as int, parquet keeps str)."""
    return a.astype(str).reset_index(drop=True).equals(b.astype(str).reset_index(drop=True))


def _keys(games) -> list[bytes]:
    return [moves.tobytes() + colors.tobytes() for moves, colors in games]


def c_feature_match(work: Path, offline: pd.DataFrame, offline_keys: dict) -> dict:
    """Release-path C features vs the offline extraction of the same games."""
    release = rankvec_features([work / "rankvec"]).set_index("game_index")
    keys = _keys(cache_games(release.index.to_numpy(), work / "cache"))
    pairs = [(g, offline_keys[k]) for g, k in zip(release.index, keys) if k in offline_keys]
    if not pairs:
        return {"matched_games": 0, "pass": False}
    new, old = zip(*pairs)
    diff = np.abs(release.loc[list(new)].to_numpy() - offline.loc[list(old)].to_numpy()).max(axis=1)
    return {"release_games": len(release), "matched_games": len(pairs),
            "max_diff_p99": float(np.quantile(diff, 0.99)), "max_diff_max": float(diff.max()),
            "pass": bool(np.quantile(diff, 0.99) <= C_DIFF_P99)}


def answer_scores(predictions: pd.DataFrame, answers: pd.DataFrame) -> dict:
    left = predictions[["question_id", "rank_id"]].astype({"question_id": str})
    merged = left.merge(answers[["question_id", "rank_id", "k"]].astype({"question_id": str}), on="question_id",
                        suffixes=("", "_true"), validate="one_to_one")
    scores, k = official_scores(merged["rank_id"].to_numpy(), merged["rank_id_true"].to_numpy()), merged["k"].to_numpy()
    by = {str(b): round(float(scores[k == b].mean()), 4) for b in K_BUCKETS}
    return by | {"equal_k": round(float(np.mean(list(by.values()))), 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    result, ok = {}, True
    files = ([f"pid_{p}{s}" for p in ("public", "private") for s in PID_VARIANTS]
             + [f"rank_{p}{s}" for p in ("public", "private") for s in RANK_SUFFIXES])
    rows = {name: len(pd.read_csv(out / "submissions" / f"{name}.csv")) if (out / "submissions" / f"{name}.csv").exists()
            else 0 for name in files}
    result["submission_rows"] = rows
    ok &= all(n == 400 for n in rows.values())
    cols = [f"top{j}" for j in range(1, 6)]
    for part in ("public", "private"):
        answers = pd.read_csv(REHEARSAL / "answers" / f"pid_{part}.csv").astype({"question_id": str})
        for suffix, flags in PID_VARIANTS.items():
            flags = dict(flags)
            insurance = REHEARSAL / "identity_insurance" if flags.pop("insurance", True) else None
            new = pd.read_parquet(out / "pid" / f"{part}{suffix}" / "predictions.parquet")
            ref, _, _, _ = predict(REHEARSAL / "store", f"pool_{part}", f"pid_{part}", "d4", flags.pop("method"),
                                   insurance_dir=insurance, **flags)
            merged = new.merge(ref, on="question_id", suffixes=("", "_ref"))
            same = int((merged[cols].to_numpy() == merged[[c + "_ref" for c in cols]].to_numpy()).all(axis=1).sum())
            scored = new.astype({"question_id": str}).merge(answers, on="question_id", validate="one_to_one")
            report = pid_report(scored[cols].to_numpy(), scored["player_id"].to_numpy(), scored["k"].to_numpy())
            result[f"pid_{part}{suffix}"] = {"top5_identical": same, "questions": len(merged),
                                             "diagnostic_answers": report["score"].round(4).to_dict()}
            # The primary keeps A-054's exact gate; probe files re-embed, so allow SAME_MIN (A-063).
            ok &= len(merged) == 400 and same >= (400 if suffix == "" else SAME_MIN)
    drift = json.loads((out / "day0_audit.json").read_text()).get("opening_drift", {})
    summary = (out / "SUMMARY.md").read_text()
    result["opening_drift"] = drift
    agreement_ok = all("agreement_opening_style" in drift.get(p, {}) for p in ("public", "private"))
    result["agreement_reported"] = agreement_ok and "agreement" in summary
    ok &= result["agreement_reported"]
    offline = rankvec_features(RANKVEC_RUNS).set_index("game_index")
    offline_keys = dict(zip(_keys(cache_games(offline.index.to_numpy())), offline.index))
    for name in ("rank_public", "rank_private"):
        old_work = REHEARSAL / "strength" / name
        ref = aggregate(old_work, MODEL, None)[["question_id", "rank_id"]]
        new = pd.read_parquet(out / "rank" / name / "humansl_predictions.parquet")[["question_id", "rank_id", "fallback_v2"]]
        merged = new.merge(ref, on="question_id", suffixes=("", "_ref"))
        same = int((merged["rank_id"] == merged["rank_id_ref"]).sum())
        ah_file = pd.read_csv(out / "submissions" / f"{name}_ah.csv")
        ah_file_matches = same_rows(ah_file, pd.read_parquet(out / "rank" / name / "humansl_predictions.parquet")
                                    [["question_id", "rank"]])
        v2_new = pd.read_csv(out / "submissions" / f"{name}_v2_fallback.csv")
        v2_old = pd.read_csv(old_work / "submission.csv")
        v2_same = bool(v2_new.equals(v2_old))
        primary = pd.read_parquet(out / "rank" / name / "rankvec_predictions.parquet")
        primary_file = pd.read_csv(out / "submissions" / f"{name}.csv")
        complete = bool(not (primary["fallback_v2"] | primary["fallback_other"]).any()
                        and (primary["humansl_games"] == primary["num_games"]).all())
        allplayers = pd.read_parquet(out / "rank" / name / "allplayers_predictions.parquet")
        allplayers_ok = bool(not (allplayers["fallback_v2"] | allplayers["fallback_other"]).any()
                             and (allplayers["humansl_games"] == allplayers["num_games"]).all()
                             and same_rows(pd.read_csv(out / "submissions" / f"{name}_allplayers.csv"),
                                           allplayers[["question_id", "rank"]]))
        c_match = c_feature_match(out / "rank" / name, offline, offline_keys)
        answers = pd.read_csv(REHEARSAL / "answers" / f"{name}.csv")
        result[name] = {"ah_classes_identical_to_a042_path": same, "questions": len(merged),
                        "ah_fallback_v2": int(merged["fallback_v2"].sum()), "ah_file_matches": ah_file_matches,
                        "v2_fallback_identical": v2_same,
                        "ahc_complete": complete, "ahc_file_matches": same_rows(primary_file, primary[["question_id", "rank"]]),
                        "ahc_vs_ah_same_class": int((primary["rank_id"].to_numpy()
                                                     == new.set_index("question_id").loc[primary["question_id"], "rank_id"].to_numpy()).sum()),
                        "allplayers_complete_and_matches": allplayers_ok,
                        "c_features": c_match,
                        "diagnostic_answers": {"AH_C": answer_scores(primary, answers),
                                               "AH": answer_scores(pd.read_parquet(out / "rank" / name / "humansl_predictions.parquet"), answers),
                                               "AH_C_allplayers": answer_scores(allplayers, answers)}}
        ok &= (same >= SAME_MIN and v2_same and ah_file_matches and complete and result[name]["ahc_file_matches"]
               and c_match["pass"] and allplayers_ok)
    result["pass"] = bool(ok)
    print(json.dumps(result, indent=1))
    (out / "rehearsal_check.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
