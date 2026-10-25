"""Competition-1 insurance rehearsal gate (RESULTS A-053), CPU only.

Fake release (scripts/make_fake_release.py): pid_public / pool_public = synthetic
split 1, pid_private / pool_private = split 2. Pre-registered:
  (a) release-path game vectors (identity/insurance.py) vs the A-035 training-path
      vectors (embeddings_split{1,2}.npz): cosine >= 0.999 for every game;
  (b) fused + Hungarian top-5 from identity/predict.py --insurance-dir equals the
      validated path (same opening scores + training-path vectors) on >= 398/400
      questions per file, any difference being a near tie;
  (c) the score is within 0.005 of A-035's fused + Hungarian (1.0 on both splits).

    python scripts/identity_insurance_rehearsal.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from identity_ge2e_eval import insurance_scores
from rehearsal_check import question_sources
from go_aicup.data.preprocess import load_store
from go_aicup.metrics import pid_report
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import assign, data
from go_aicup.tasks.identity.insurance import RUN, load_run
from go_aicup.tasks.identity.predict import load_params, opening_scores, predict, train_prefix

REHEARSAL = ROOT / "outputs" / "rehearsal"
WORK = REHEARSAL / "identity_insurance"
SPLITS = {"public": 1, "private": 2}
COS_MIN, SAME_MIN, SCORE_TOL, TIE = 0.999, 398, 0.005, 0.01


def cosines(name: str, sources: np.ndarray, split: int) -> np.ndarray:
    release = np.load(WORK / name / "ge2e.npz")
    train = np.load(RUN / f"embeddings_split{split}.npz")
    position = {int(g): i for i, g in enumerate(train["game_index"])}
    a = release["vector"].astype(np.float64)
    b = train["vector"][[position[int(sources[r])] for r in release["store_row"]]].astype(np.float64)
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def main() -> int:
    _, _, w, lam = load_run()
    meta, prefix = train_prefix("d4")
    gallery, questions = data.pid_splits(meta)
    params = load_params("d4")
    black = meta["color"].astype(str).to_numpy() == "B"
    pid = meta["player_id"].astype(str).to_numpy()
    ok, result = True, {}
    for part, split in SPLITS.items():
        answers = pd.read_csv(REHEARSAL / "answers" / f"pid_{part}.csv")
        q_split = questions[questions["split"] == split]
        source_games = dict(zip(q_split["question_id"], q_split["games"]))
        q_sources = question_sources(REHEARSAL / "store", f"pid_{part}", answers, source_games)
        _, _, pool_meta = load_store(f"pool_{part}", REHEARSAL / "store")
        pool_sources = pool_meta["game_id"].str[1:].astype(int).to_numpy()
        cos = np.concatenate([cosines(f"pid_{part}", q_sources, split), cosines(f"pool_{part}", pool_sources, split)])
        ok_a = bool(cos.min() >= COS_MIN)

        candidates, gal, q = data.split_view(gallery, questions, split)
        population = np.flatnonzero(black & ~np.isin(pid, candidates))
        fused_ref = opening_scores(prefix, gal, list(q["games"]), population, params) + \
            lam * insurance_scores(RUN, split, gal, q, w)
        log_q = assign.sinkhorn(assign.row_posterior(fused_ref, params["tau"]))
        reference = pd.DataFrame(candidates[assign.hungarian_top5(log_q)], columns=[f"top{j}" for j in range(1, 6)])
        reference["source_question_id"] = q["question_id"].to_numpy()
        top1_margin = np.sort(log_q, axis=1)[:, -1] - np.sort(log_q, axis=1)[:, -2]
        margins = dict(zip(q["question_id"], top1_margin))

        table, _, _, info = predict(REHEARSAL / "store", f"pool_{part}", f"pid_{part}", "d4", "hungarian",
                                    insurance_dir=WORK)
        table["question_id"] = table["question_id"].astype(int)
        merged = answers.merge(table, on="question_id").merge(reference, on="source_question_id", suffixes=("", "_ref"))
        cols = [f"top{j}" for j in range(1, 6)]
        same = (merged[cols].to_numpy() == merged[[c + "_ref" for c in cols]].to_numpy()).all(axis=1)
        gaps = [float(margins[s]) for s in merged.loc[~same, "source_question_id"]]
        ok_b = bool(same.sum() >= SAME_MIN and all(g < TIE for g in gaps))
        report = pid_report(merged[cols].to_numpy(), merged["player_id"].astype(str).to_numpy(), merged["k"])
        score = float(report.loc["all", "score"])
        ok_c = abs(score - 1.0) <= SCORE_TOL
        ok &= ok_a and ok_b and ok_c
        result[part] = {"cos_min": round(float(cos.min()), 6), "cos_p01": round(float(np.quantile(cos, 0.01)), 6),
                        "games": int(len(cos)), "top5_identical": int(same.sum()), "questions": int(len(merged)),
                        "differing_top1_margins": [round(g, 4) for g in gaps], "score_by_k": report["score"].round(4).to_dict(),
                        "a": ok_a, "b": ok_b, "c": ok_c, "insurance_info": {k: info[k] for k in info if "insurance" in k}}
        print(part, json.dumps(result[part]), flush=True)
    result["pass"] = bool(ok)
    print(json.dumps({"identity_insurance_rehearsal": "PASS" if ok else "FAIL"}))
    (WORK / "rehearsal.json").write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
