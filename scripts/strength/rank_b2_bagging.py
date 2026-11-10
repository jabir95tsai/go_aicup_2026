"""Cheap B2 variants on the primary P1+P2+P3 model (RESULTS 2026-11-08), CPU only.

bag   average the probabilities of B2 fitted on BAG synthetic-draw seeds (the draw alone
      moves Equal-k by about 0.005, so one seed carries sampling noise)
syn   one seed with 5x the synthetic questions (another way to cut the same noise)
noA   drop the 10 A-024 columns: how much A adds (decides whether H+C on training-split
      games, where A is in-sample, is worth computing)
allp  final fit on Valid-S + Valid-U (HD<=1) players; no local score (E1/E2 are Valid-U)

    python -m scripts.strength.rank_b2_bagging local          # E2 / E1, paired with the primary
    python -m scripts.strength.rank_b2_bagging export --job ID [--arms noA-allp noA-bag10 ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3

OUT = ROOT / "outputs" / "rank" / "humansl" / "b2_bagging"
BAG = tuple(range(2027, 2037))   # 10 seeds; 2027 is the primary's
PRIMARY = ROOT / "submissions" / "rank" / "ahc-p12-komihc_725150_rank.csv"
NOA = ROOT / "submissions" / "rank" / "ahc-p3-noA_725504_rank.csv"   # Public 0.7271, best on 11/09
ARMS = ("bag10", "allp", "allpbag10", "noA", "noA-allp", "noA-bag10", "noA-allpbag10")


def data():
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    both = pd.concat([fit, eval_games], ignore_index=True)
    x, _ = p3.replaced(both, np.vstack([x_fit, x_eval]), use_c=True)
    x = np.hstack([x, hf.hd1_columns(both)])
    return games, fit, eval_games, x[:len(fit)], x[len(fit):]


def fit_b2(games: pd.DataFrame, x: np.ndarray, seed: int, synthetic: int = hs.SYNTHETIC) -> RankModel:
    keep = (games["handicap"] <= 1).to_numpy()
    hs.SEED, old = seed, hs.SYNTHETIC
    hs.SYNTHETIC = synthetic
    try:
        return RankModel("B2").fit(games[keep].reset_index(drop=True), pd.DataFrame(), x[keep])
    finally:
        hs.SYNTHETIC = old


def scores(p: np.ndarray, q: pd.DataFrame) -> np.ndarray:
    return official_scores(score_rank(p), q["rank_id"].to_numpy())


def local() -> None:
    games, fit, eval_games, x_fit, x_eval = data()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    index = eval_games["game_index"].to_numpy()
    proba = {name: {} for name in sets}
    for seed in BAG:
        model = fit_b2(fit, x_fit, seed)
        for name, q in sets.items():
            proba[name][seed] = model.probabilities(x_eval, index, q)
        print("seed", seed, flush=True)
    syn = fit_b2(fit, x_fit, hf.SEED, 5 * hs.SYNTHETIC)
    no_a = {seed: fit_b2(fit, x_fit[:, 10:], seed) for seed in hf.SEEDS}
    result = {}
    for name, q in sets.items():
        k = q["k"].to_numpy()
        primary = scores(proba[name][hf.SEED], q)
        singles = [hf.equal_k(scores(proba[name][s], q), k) for s in hf.SEEDS]
        arms = {"bag10": scores(np.mean([proba[name][s] for s in BAG], axis=0), q),
                "syn5x": scores(syn.probabilities(x_eval, index, q), q),
                "noA_2027": scores(no_a[hf.SEED].probabilities(x_eval[:, 10:], index, q), q)}
        no_a_singles = [hf.equal_k(scores(no_a[s].probabilities(x_eval[:, 10:], index, q), q), k) for s in hf.SEEDS]
        entry = {"primary_2027": hf.by_k(primary, k),
                 "single_seed_equal_k": [round(v, 4) for v in singles],
                 "noA_single_seed_equal_k": [round(v, 4) for v in no_a_singles],
                 "noA_mean_diff": round(float(np.mean(no_a_singles) - np.mean(singles)), 4)}
        for arm, s in arms.items():
            entry[arm] = hf.by_k(s, k) | {"diff_vs_primary": round(hf.equal_k(s, k) - hf.equal_k(primary, k), 4),
                                          "diff_vs_single_mean": round(hf.equal_k(s, k) - float(np.mean(singles)), 4),
                                          "bootstrap_95": hf.bootstrap(primary, s, q)}
        result[name] = entry
        print(name, json.dumps(entry), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local.json").write_text(json.dumps(result, indent=1))


def export(job: str, arms: list[str]) -> None:
    games, fit, eval_games, x_fit, x_eval = data()
    both = pd.concat([fit, eval_games], ignore_index=True)
    x_both = np.vstack([x_fit, x_eval])
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    builders = {"bag10": lambda: [fit_b2(fit, x_fit, s) for s in BAG],
                "allp": lambda: [fit_b2(both, x_both, hf.SEED)],
                "allpbag10": lambda: [fit_b2(both, x_both, s) for s in BAG],
                "noA": lambda: [fit_b2(fit, x_fit[:, 10:], hf.SEED)],
                "noA-allp": lambda: [fit_b2(both, x_both[:, 10:], hf.SEED)],
                "noA-bag10": lambda: [fit_b2(fit, x_fit[:, 10:], s) for s in BAG],
                "noA-allpbag10": lambda: [fit_b2(both, x_both[:, 10:], s) for s in BAG]}
    models = {"primary": [fit_b2(fit, x_fit, hf.SEED)]} | {name: builders[name]() for name in arms}
    print("fitted", flush=True)
    template = pd.read_csv(hf.ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {name: [] for name in models}
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for name, ms in models.items():
            x = table.to_numpy()[:, 10:] if name.startswith("noA") else table.to_numpy()
            p = np.mean([m.probabilities(x, table.index.to_numpy(), questions) for m in ms], axis=0)
            rows[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                            "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    references = {"vs_primary": PRIMARY, "vs_noA": NOA}
    report = {}
    for name, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        report[name] = {}
        for label, path in references.items():
            changed = result != pd.read_csv(path, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
            report[name][label] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                                   "private": int(changed[changed.index.str.startswith("priv")].sum())}
            if label == "vs_primary":
                diff = changed
        if name == "primary":
            if diff.any():
                raise ValueError("primary refit does not reproduce the P1+P2+P3 submission")
            continue
        output = ROOT / "submissions" / "rank" / f"ahc-p3-{name}_{job}_rank.csv"
        result.reset_index().to_csv(output, index=False, lineterminator="\r\n")
        print("wrote", output, flush=True)
    print(json.dumps({"changed": report}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["local", "export"])
    parser.add_argument("--job", default="local")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=["bag10", "allp", "allpbag10", "noA"])
    args = parser.parse_args()
    local() if args.command == "local" else export(args.job, args.arms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
