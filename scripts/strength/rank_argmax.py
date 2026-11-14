"""Compare cvec2 argmax with score_rank; export from the saved test probabilities.

python -m scripts.strength.rank_argmax export --job 1113
python -m scripts.strength.rank_argmax evaluate
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.submission_files import write_rank_csv
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs/rank/humansl/argmax"


def export(job):
    from scripts.strength.rank_identity_merge import L10, overlay

    saved = pd.read_parquet(ROOT / "outputs/rank/humansl/question_probs/cvec2.parquet")
    base = pd.read_csv(ROOT / "submissions/rank/751959-cvec2.csv").set_index("question_id")
    assert len(saved) == len(base) == 800 and saved.question_id.is_unique and base.index.is_unique
    assert set(saved.question_id) == set(base.index)
    saved = saved.set_index("question_id").loc[base.index]
    p = saved[[f"p_{r}" for r in range(10)]].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(axis=1), 1)
    assert np.array_equal(np.asarray(RANKS)[score_rank(p)], base.pred_rank)
    result = pd.Series(np.asarray(RANKS)[p.argmax(axis=1)], index=base.index, name="pred_rank")
    path = write_rank_csv(result.reset_index(), "argmax", job)
    overlay(path.name, job, "arg-l10")
    l10 = pd.read_csv(L10)
    overrides = l10[l10.above_null_max].set_index("question_id").top1_rank
    result.loc[overrides.index] = overrides
    ref = pd.read_csv(ROOT / "submissions/rank/751959-cv2-l10.csv").set_index("question_id").pred_rank
    changed = result != ref.loc[result.index]
    report = {part: int(changed[changed.index.str.startswith(prefix)].sum())
              for part, prefix in (("public", "pub_"), ("private", "priv_"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def evaluate():
    import go_aicup.tasks.strength.humansl as hs
    from scripts.strength import humansl_hd_filter as hf, rank_cvec_v2 as cv2, rank_more_players as more

    games, fit, ev, xs = cv2.data()
    sets = {n: q for n, q in hf.question_sets(games, rebuild=False).items() if n in ("E1", "E2")}
    keep = (fit.handicap <= 1).to_numpy()
    prior = more.player_prior(fit[keep & (fit.role == "S").to_numpy()])
    xf, xe = xs["v2"]
    old = json.loads((cv2.OUT / "result.json").read_text())
    report = {"seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        model = more.PriorMatched(prior).fit(fit[keep].reset_index(drop=True), pd.DataFrame(), xf[keep])
        entry = {}
        for name, q in sets.items():
            p = model.probabilities(xe, ev.game_index.to_numpy(), q)
            truth, k = q.rank_id.to_numpy(), q.k.to_numpy()
            a = official_scores(score_rank(p), truth)
            b = official_scores(p.argmax(axis=1), truth)
            assert hf.by_k(a, k) == old["seeds"][str(seed)][name]["v2"]
            entry[name] = {"score_rank": hf.by_k(a, k), "argmax": hf.by_k(b, k),
                           "delta": hf.by_k(b - a, k),
                           "exact_score_rank": hf.by_k((score_rank(p) == truth).astype(float), k),
                           "exact_argmax": hf.by_k((p.argmax(axis=1) == truth).astype(float), k)}
            if seed == hf.SEED:
                entry[name]["player_bootstrap_95"] = hf.bootstrap(a, b, q)
        report["seeds"][str(seed)] = entry
        print(seed, json.dumps(entry), flush=True)
    report["mean"] = {name: {arm: {k: round(float(np.mean([
        report["seeds"][str(s)][name][arm][k] for s in hf.SEEDS])), 4)
        for k in ("5", "10", "15", "20", "equal_k")}
        for arm in ("score_rank", "argmax", "delta", "exact_score_rank", "exact_argmax")} for name in sets}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["mean"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "evaluate"])
    parser.add_argument("--job", default="1113")
    args = parser.parse_args()
    {"export": lambda: export(args.job), "evaluate": evaluate}[args.command]()
