"""Rank-prior LB probes on the P1+P2 model (RESULTS 2026-11-08).

B2 is fitted on synthetic questions that draw players uniformly, so its implied
rank prior is the Valid-S player share per rank (per k: players with >= k games),
not uniform. If the test set holds equal questions per rank, re-weighting toward a
uniform prior should help; if it does not, it can hurt. Three decodings, each a
single change from the P1+P2 submission:

  uprior    p'(r) ~ p(r) * (1/10) / prior_k(r), then score_rank
  em        Saerens EM: estimate the set's rank prior from the predictions, re-weight
  balanced  equal question count per rank within the set (assignment on expected score)

    python -m scripts.strength.rank_prior_probes local     # E1 / E2 local check (E2 is rank-balanced)
    python -m scripts.strength.rank_prior_probes export    # submission files from the test predictions
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.v2.aggregate import official_scores
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import K_BUCKETS, RankModel, k_bucket

import scripts.strength.humansl_hd_filter as hf

E = np.e
NEIGHBOUR = np.eye(10) + (np.eye(10, k=1) + np.eye(10, k=-1)) / E  # expected score of guess r given truth
WORK = ROOT / "outputs" / "rank_test_ahc_713770" / "rank"
SAMPLE = ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv"
OUT = ROOT / "outputs" / "rank" / "humansl" / "prior_probes"


def train_priors(fit: pd.DataFrame) -> dict[int, np.ndarray]:
    """Per k: share of B2-eligible players (HD<=1 games >= k) per rank, as the synthetic draws see it."""
    sub = fit[fit["handicap"] <= 1]
    counts = sub.groupby("player_id").agg(rank_id=("rank_id", "first"), n=("game_index", "size"))
    out = {}
    for k in K_BUCKETS:
        share = np.bincount(counts.loc[counts["n"] >= k, "rank_id"], minlength=10).astype(float)
        out[k] = share / share.sum()
    return out


def reweight(p: np.ndarray, k: np.ndarray, priors: dict, target: np.ndarray) -> np.ndarray:
    out = np.empty_like(p)
    for b in K_BUCKETS:
        m = k == b
        w = p[m] * (target / priors[b])
        out[m] = w / w.sum(axis=1, keepdims=True)
    return out


def em(p: np.ndarray, k: np.ndarray, priors: dict, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    target = np.full(10, 0.1)
    for _ in range(iters):
        q = reweight(p, k, priors, target)
        new = q.mean(axis=0)
        if np.abs(new - target).max() < 1e-7:
            break
        target = new
    return reweight(p, k, priors, target), target


def balanced(p: np.ndarray) -> np.ndarray:
    """Max total expected score with every rank used n/10 times (n divisible by 10)."""
    n = len(p)
    if n % 10:
        raise ValueError("balanced decoding needs a multiple of 10 questions")
    gain = p @ NEIGHBOUR  # [n, 10]: expected score of each guess
    cost = -np.repeat(gain, n // 10, axis=1)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(n, dtype=int)
    out[rows] = cols // (n // 10)
    return out


def decode_all(p, k, priors) -> dict[str, np.ndarray]:
    em_p, em_prior = em(p, k, priors)
    return {"base": score_rank(p),
            "uprior": score_rank(reweight(p, k, priors, np.full(10, 0.1))),
            "em": score_rank(em_p),
            "balanced": balanced(p)}, em_prior


def local() -> None:
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    priors = train_priors(fit)
    model, _ = RankModel.load(hf.OUT / "model_p12.pkl")
    x = np.hstack([x_eval, hf.hd1_columns(eval_games)])
    sets = hf.question_sets(games, rebuild=False)
    result = {"train_priors": {str(b): priors[b].round(4).tolist() for b in K_BUCKETS}}
    for name in ("E1", "E2"):
        q = sets[name]
        k = q["k"].to_numpy()
        y = q["rank_id"].to_numpy()
        p = model.probabilities(x, eval_games["game_index"].to_numpy(), q)
        decoded, em_prior = decode_all(p, k, priors)
        base = official_scores(decoded["base"], y)
        result[name] = {"true_rank_share": (np.bincount(y, minlength=10) / len(y)).round(4).tolist(),
                        "em_prior": em_prior.round(4).tolist()}
        for arm, pred in decoded.items():
            s = official_scores(pred, y)
            result[name][arm] = hf.by_k(s, k) | {"diff": round(hf.equal_k(s, k) - hf.equal_k(base, k), 4),
                                                 "bootstrap_95": hf.bootstrap(base, s, q) if arm != "base" else None,
                                                 "changed": int((pred != decoded["base"]).sum())}
        print(name, json.dumps(result[name]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local.json").write_text(json.dumps(result, indent=1))


def export(job: str) -> None:
    games, fit, *_ = hf.setup()
    priors = train_priors(fit)
    template = pd.read_csv(SAMPLE, dtype=str)
    preds = {arm: [] for arm in ("base", "uprior", "em", "balanced")}
    report = {}
    for part in ("public", "private"):
        table = pd.read_parquet(WORK / f"rank_{part}" / "rankvec_p12_predictions.parquet")
        if table.fallback_v2.any() or table.fallback_other.any() or (table.humansl_games != table.num_games).any():
            raise ValueError(f"{part}: incomplete features or fallback")
        p = table[[f"p_{r}" for r in RANKS]].to_numpy()
        k = table["num_games"].astype(int).map(k_bucket).to_numpy()
        decoded, em_prior = decode_all(p, k, priors)
        if not (np.array([RANKS.index(r) for r in table["rank"]]) == decoded["base"]).all():
            raise ValueError(f"{part}: base decoding does not reproduce the P1+P2 submission")
        report[part] = {"em_prior": em_prior.round(4).tolist(),
                        "base_share": (np.bincount(decoded["base"], minlength=10) / len(p)).round(4).tolist()}
        for arm, pred in decoded.items():
            report[part][f"changed_{arm}"] = int((pred != decoded["base"]).sum())
            preds[arm].append(pd.DataFrame({"question_id": table["question_id"], "pred_rank": [RANKS[i] for i in pred]}))
    for arm in ("uprior", "em", "balanced"):
        result = pd.concat(preds[arm], ignore_index=True).set_index("question_id").loc[template.question_id].reset_index()
        if len(result) != 800 or result.pred_rank.isna().any() or not result.pred_rank.isin(RANKS).all():
            raise ValueError(arm)
        output = ROOT / "submissions" / "rank" / f"ahc-p12-{arm}_{job}_rank.csv"
        result.to_csv(output, index=False, lineterminator="\r\n")
        print("wrote", output)
    print(json.dumps(report))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"export_{job}.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["local", "export"])
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    local() if args.command == "local" else export(args.job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
