"""HD-aware opponent term in B2 (RESULTS 2026-11-13).

In an HD1 game the opponent is about one rank stronger (player black) or weaker (player white):
on the fitting pool the opponent feature block of HD1 games sits +0.4..+1.4 / -0.5..-0.8 ranks
from the HD0 games of the same owner rank, the own block does not move. B2 averages the opponent
features of all games and only gets the two HD1-share flags to correct for it. Here the question
logit of rank r reads the opponent features of each game type with the weights of the opponent's
rank: logit_r = b_r + W_own[r] . own + W_flag[r] . flags + sum_d W_opp[r + d] . opp_d, with
opp_d = (1/k) * sum of opponent features over the games of type d (0: HD0, +1: HD1 as black,
-1: HD1 as white) and W_opp of 12 rows (opponent ranks -1..10).

  mean    B2 (control through the same code path; the sklearn base is reported as a reference)
  tied    the model above (530 + 50 parameters against 530)
  untied  a separate 10-row W_opp per game type (diagnostic: is the tie the right constraint?)
Data, prior, synthetic draws, loss and gate as rank_attn_pool (cvec2 base; Valid-U E1 / E2, five
seeds; G1: best arm minus mean >= +0.005 on E1 or E2, the other >= 0, every E2 k >= -0.005).

    python -m scripts.strength.rank_hd_shift [--jobs 12]
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

OUT = ROOT / "outputs" / "rank" / "humansl" / "hd_shift"
ARMS = ("mean", "tied", "untied")
ITERATIONS = 300
OWN = np.r_[0:15, 30:40]
OPP = np.r_[15:30, 40:50]
FLAGS = np.r_[50:52]
SHIFT = (0, 1, -1)              # opponent rank minus player rank for game types 0 / 1 / 2


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


def game_types(games: pd.DataFrame) -> np.ndarray:
    """0: HD0, 1: HD1 and player black (stronger opponent), 2: HD1 and player white."""
    hd1 = (games["handicap"] == 1).to_numpy()
    black = (games["player_color"] == 0).to_numpy()
    return np.where(~hd1, 0, np.where(black, 1, 2))


def fit_predict(arm: str, x_q, t_q, labels, x_eval, t_eval, seed: int) -> list[np.ndarray]:
    """x [N, k, D] standardised game features, t [N, k] game types; probabilities for each eval set."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    as_x = lambda v: torch.as_tensor(v, dtype=torch.float32)
    x, t, target = as_x(x_q), torch.as_tensor(t_q), torch.as_tensor(labels)
    n, _, d = x.shape
    own_i, opp_i, flag_i = (torch.as_tensor(v) for v in (OWN, OPP, FLAGS))
    mean_pool = x.mean(1)
    centre, spread = mean_pool.mean(0), mean_pool.std(0).clamp(min=1e-6)
    bias = torch.zeros(10, requires_grad=True)
    if arm == "mean":
        params = [torch.zeros((10, d), requires_grad=True)]
    else:
        rows = 12 if arm == "tied" else 30
        params = [torch.zeros((10, len(OWN) + len(FLAGS)), requires_grad=True),
                  torch.zeros((rows, len(OPP)), requires_grad=True)]

    def logits(v, types):
        pooled = (v.mean(1) - centre) / spread
        if arm == "mean":
            return pooled @ params[0].T + bias
        out = pooled[:, torch.cat([own_i, flag_i])] @ params[0].T + bias
        for code, shift in enumerate(SHIFT):
            part = (v[:, :, opp_i] * (types == code)[:, :, None]).sum(1) / v.shape[1] / spread[opp_i]
            share = (types == code).float().mean(1, keepdim=True)
            part = part - share * centre[opp_i] / spread[opp_i]
            w = params[1][1 + shift:11 + shift] if arm == "tied" else params[1][10 * code:10 * code + 10]
            out = out + part @ w.T
        return out

    optimizer = torch.optim.LBFGS([*params, bias], max_iter=ITERATIONS, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits(x, t), target) + sum(p.pow(2).sum() for p in params) / (2 * n)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return [torch.softmax(logits(as_x(e), torch.as_tensor(te)), dim=1).numpy() for e, te in zip(x_eval, t_eval)]


def task(seed, k, arm, fit, x_fit, t_fit, y, prior, eval_tensors, eval_types, count):
    rows, labels = synthetic(fit, y, prior, k, seed, count)
    return fit_predict(arm, x_fit[rows], t_fit[rows], labels, eval_tensors, eval_types, seed)


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
    t_fit, t_eval = game_types(fit), game_types(eval_games)
    centre, spread = x_fit.mean(0), x_fit.std(0)
    spread[spread == 0] = 1.0
    z_fit, z_eval = (x_fit - centre) / spread, (x_eval - centre) / spread
    position = pd.Series(np.arange(len(eval_games)), index=eval_games["game_index"].to_numpy())
    sets = {n: q.reset_index(drop=True) for n, q in hf.question_sets(games, rebuild=False).items() if n in ("E1", "E2")}
    names = list(sets)
    tensors = {k: [np.stack([z_eval[position.loc[list(g)].to_numpy()] for g in q.loc[q["k"] == k, "games"]])
                   for q in sets.values()] for k in K_BUCKETS}

    types = {k: [np.stack([t_eval[position.loc[list(g)].to_numpy()] for g in q.loc[q["k"] == k, "games"]])
                 for q in sets.values()] for k in K_BUCKETS}
    tasks = [(seed, k, arm) for seed in hf.SEEDS for k in K_BUCKETS for arm in ARMS]
    done = Parallel(n_jobs=args.jobs)(delayed(task)(s, k, a, fit, z_fit, t_fit, y, prior, tensors[k], types[k], hs.SYNTHETIC)
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
