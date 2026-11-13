"""Learned aggregation over a question's games (user proposal, RESULTS 2026-11-12).

B2 averages the per-game features of a question with equal weights. Here a small scorer gives
every game a weight, w_g = softmax_g(s(x_g)), so games that carry clear rank evidence can count
more; the pooled vector goes to the same multinomial LR as B2. With the scorer at zero the model
is B2 (equal weights), which is also the control arm fitted through the same code path.

  mean     equal weights (control; the sklearn base is reported as a reference)
  att_lin  s(x) = u . x            (52 parameters)
  att_mlp  s(x) = v . tanh(U x)    (52 -> 8 -> 1)
Data and prior as the cvec2 base: fit Valid-S + train20 HD<=1 (C v2), prior-matched synthetic
questions (the same draws as PriorMatched), one model per k. Valid-U E1 / E2, five seeds.
Loss = mean CE + ||W||^2 / (2 N) (sklearn C = 1) + the same L2 on the scorer; L-BFGS.
Gate (pre-registered, as G1): best attention arm minus mean, five-seed mean Equal-k >= +0.005 on
E1 or E2, the other >= 0, every E2 k >= -0.005.

    python -m scripts.strength.rank_attn_pool [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs" / "rank" / "humansl" / "attn_pool"
ARMS = ("mean", "att_lin", "att_mlp")
HIDDEN, ITERATIONS = 8, 300


def synthetic(games: pd.DataFrame, y: np.ndarray, prior: dict, k: int, seed: int, count: int):
    """Row indices [count, k] and labels, drawn exactly like PriorMatched._fit_synthetic for bucket k."""
    rng = np.random.default_rng(seed)
    by_player = list(games.groupby(games["player_id"].astype(str)).indices.values())
    out = None
    for bucket in K_BUCKETS:            # consume the generator in the same order as the base
        players = [rows for rows in by_player if len(rows) >= bucket]
        by_rank = [[p for p in players if y[p[0]] == r] for r in range(10)]
        ranks = rng.choice(10, count, p=prior[bucket])
        rows = np.stack([rng.choice(by_rank[r][rng.integers(len(by_rank[r]))], bucket, replace=False) for r in ranks])
        if bucket == k:
            out = (rows, ranks)
    return out


def fit_predict(arm: str, x_q: np.ndarray, labels: np.ndarray, x_eval: list[np.ndarray], seed: int) -> list[np.ndarray]:
    """x_q [N, k, D] standardised game features; returns probabilities for each eval tensor [Q, k, D]."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    x = torch.as_tensor(x_q, dtype=torch.float32)
    target = torch.as_tensor(labels)
    n, _, d = x.shape
    mean_pool = x.mean(1)
    centre, spread = mean_pool.mean(0), mean_pool.std(0).clamp(min=1e-6)
    weight = torch.zeros((10, d), requires_grad=True)
    bias = torch.zeros(10, requires_grad=True)
    params = [weight, bias]
    if arm == "att_lin":
        scorer = [torch.zeros(d, requires_grad=True)]
    elif arm == "att_mlp":
        scorer = [(0.1 * torch.randn(HIDDEN, d)).requires_grad_(), torch.zeros(HIDDEN, requires_grad=True)]
    else:
        scorer = []
    params += scorer

    def pooled(t):
        if arm == "mean":
            p = t.mean(1)
        else:
            s = t @ scorer[0] if arm == "att_lin" else torch.tanh(t @ scorer[0].T).matmul(scorer[1])
            p = (torch.softmax(s, dim=1)[:, :, None] * t).sum(1)
        return (p - centre) / spread

    optimizer = torch.optim.LBFGS(params, max_iter=ITERATIONS, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(pooled(x) @ weight.T + bias, target)
        loss = loss + (weight.pow(2).sum() + sum(p.pow(2).sum() for p in scorer)) / (2 * n)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return [torch.softmax(pooled(torch.as_tensor(e, dtype=torch.float32)) @ weight.T + bias, dim=1).numpy()
                for e in x_eval]


def task(seed, k, arm, fit, x_fit, y, prior, eval_tensors, count):
    rows, labels = synthetic(fit, y, prior, k, seed, count)
    return fit_predict(arm, x_fit[rows], labels, eval_tensors, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    from joblib import Parallel, delayed
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_more_players as more

    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    y = fit["rank_id"].to_numpy()
    centre, spread = x_fit.mean(0), x_fit.std(0)
    spread[spread == 0] = 1.0
    z_fit, z_eval = (x_fit - centre) / spread, (x_eval - centre) / spread
    position = pd.Series(np.arange(len(eval_games)), index=eval_games["game_index"].to_numpy())
    sets = {n: q.reset_index(drop=True) for n, q in hf.question_sets(games, rebuild=False).items() if n in ("E1", "E2")}
    names = list(sets)
    tensors = {k: [np.stack([z_eval[position.loc[list(g)].to_numpy()] for g in q.loc[q["k"] == k, "games"]])
                   for q in sets.values()] for k in K_BUCKETS}

    tasks = [(seed, k, arm) for seed in hf.SEEDS for k in K_BUCKETS for arm in ARMS]
    done = Parallel(n_jobs=args.jobs)(delayed(task)(s, k, a, fit, z_fit, y, prior, tensors[k], hs.SYNTHETIC)
                                      for s, k, a in tasks)
    result = {"fit_games": int(len(fit)), "seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        reference = more.PriorMatched(prior).fit(fit, pd.DataFrame(), x_fit)
        entry = {}
        for i, (name, q) in enumerate(sets.items()):
            k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
            scores = {"sklearn": hf.predict(reference, eval_games, x_eval, q)}
            for arm in ARMS:
                p = np.empty((len(q), 10))
                for (s, bucket, a), probs in zip(tasks, done):
                    if s == seed and a == arm:
                        p[k == bucket] = probs[i]
                scores[arm] = official_scores(score_rank(p), truth)
            cell = {a: hf.by_k(s, k) for a, s in scores.items()}
            for arm in ARMS[1:]:
                cell[f"{arm}-mean"] = {
                    "by_k": {str(b): round(float(scores[arm][k == b].mean() - scores["mean"][k == b].mean()), 4)
                             for b in K_BUCKETS},
                    "equal_k": round(hf.equal_k(scores[arm], k) - hf.equal_k(scores["mean"], k), 4)}
                if seed == hf.SEED:
                    cell[f"{arm}-mean"]["bootstrap_95"] = hf.bootstrap(scores["mean"], scores[arm], q)
            entry[name] = cell
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {a: e[a]["equal_k"] for a in e} for n, e in entry.items()}), flush=True)
    diffs = [f"{a}-mean" for a in ARMS[1:]]
    keys = ["sklearn", *ARMS, *diffs]
    result["mean"] = {n: {a: round(float(np.mean([result["seeds"][str(s)][n][a]["equal_k"] for s in hf.SEEDS])), 4)
                          for a in keys} for n in names}
    result["mean_by_k"] = {n: {d: {str(b): round(float(np.mean([result["seeds"][str(s)][n][d]["by_k"][str(b)]
                                                                 for s in hf.SEEDS])), 4) for b in K_BUCKETS}
                               for d in diffs} for n in names}
    gate = {}
    for d in diffs:
        e1, e2 = result["mean"]["E1"][d], result["mean"]["E2"][d]
        worst = min(result["mean_by_k"]["E2"][d].values())
        gate[d] = {"E1": e1, "E2": e2, "min_E2_k": worst,
                   "pass": bool(max(e1, e2) >= 0.005 and min(e1, e2) >= 0 and worst >= -0.005)}
    result["gate"] = gate
    print(json.dumps({"mean": result["mean"], "mean_by_k": result["mean_by_k"], "gate": gate}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
