"""Lower-capacity B2 heads (RESULTS 2026-11-14).

B2 is one full multinomial LR per k (52 x 10 weights, C = 1). Every head with MORE capacity lost
(MLP, attention, colour interactions, extra columns); the other direction was tried once
(C = 0.3, +0.0013, old setting). The rank is ordinal and the middle ranks have few players, so a
head with fewer free directions may generalise better. Same features (cvec2 base, 52 columns),
same prior-matched synthetic questions, only the head changes:
  base            LogisticRegression(C=1)                   (must reproduce PriorMatched)
  l2_03/01/003    C = 0.3 / 0.1 / 0.03
  rr2, rr3        reduced-rank multinomial: logits = (x A) B + b, rank 2 / 3
  ord             cumulative logit (proportional odds): one direction, 9 ordered cut points
  smooth          full LR fitted to neighbour-smoothed targets (0.15 / 0.70 / 0.15)

select    head choice inside the fitting pool only (as rank_permove.select: rank-stratified
          player halves, rank-balanced questions of the other half, three seeds).
evaluate  the chosen head against base on Valid-U E1 / E2, five seeds.
Pre-registered: only the head with the highest selection gain is scored on Valid-U, and only if
that gain is >= +0.003; G1 promote = five-seed mean Equal-k >= +0.005 on E1 or E2, the other
>= 0, every E2 k >= -0.005; both >= 0 -> LB probe; else stop.

    python -m scripts.strength.rank_b2_lowcap auto [--jobs 12]
    python -m scripts.strength.rank_b2_lowcap inout          # diagnostic, see inout()
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import go_aicup.tasks.strength.humansl as hs
import scripts.strength.rank_more_players as more

OUT = ROOT / "outputs" / "rank" / "humansl" / "b2_lowcap"
L2 = {"base": 1.0, "l2_03": 0.3, "l2_01": 0.1, "l2_003": 0.03}
ARMS = (*L2, "rr2", "rr3", "ord", "smooth")
SELECT_SEEDS, SELECT_MIN = (2027, 2028, 2029), 0.003
SMOOTH = 0.15


class TorchHead:
    """rr<r> / ord / smooth fitted by L-BFGS; loss = mean CE + ||weights||^2 / (2 N) (sklearn C = 1)."""

    def __init__(self, arm: str, seed: int):
        self.arm, self.seed = arm, seed

    def _logp(self, x, p):
        import torch
        if self.arm == "ord":
            cuts = p["first"] + torch.cat([torch.zeros(1), torch.cumsum(torch.nn.functional.softplus(p["gaps"]), 0)])
            cdf = torch.sigmoid(cuts[None, :] - (x @ p["w"])[:, None])
            edges = torch.cat([torch.zeros(len(x), 1), cdf, torch.ones(len(x), 1)], dim=1)
            return torch.log((edges[:, 1:] - edges[:, :-1]).clamp_min(1e-9))
        logits = (x @ p["a"]) @ p["b"] + p["bias"] if self.arm.startswith("rr") else x @ p["w"] + p["bias"]
        return torch.log_softmax(logits, dim=1)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchHead":
        import torch
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(self.seed)
        x_t, n, d = torch.as_tensor(x, dtype=torch.float64), len(x), x.shape[1]
        target = torch.zeros(n, 10, dtype=torch.float64)
        target[torch.arange(n), torch.as_tensor(y)] = 1.0
        if self.arm == "smooth":
            soft = torch.zeros_like(target)
            soft[:, 1:] += SMOOTH * target[:, :-1]
            soft[:, :-1] += SMOOTH * target[:, 1:]
            target = soft + target * (1 - soft.sum(dim=1, keepdim=True))      # edge ranks keep the missing share
        rand = lambda *shape: (0.1 * torch.randn(*shape, generator=generator, dtype=torch.float64)).requires_grad_()
        zeros = lambda *shape: torch.zeros(*shape, dtype=torch.float64, requires_grad=True)
        if self.arm == "ord":
            share = torch.as_tensor(np.cumsum(np.bincount(y, minlength=10) / n)[:-1]).clamp(1e-3, 1 - 1e-3)
            cuts = torch.log(share / (1 - share))
            gaps = (cuts[1:] - cuts[:-1]).clamp_min(1e-3)
            p = {"w": zeros(d), "first": cuts[:1].clone().requires_grad_(),
                 "gaps": torch.log(torch.expm1(gaps)).requires_grad_()}
            penalised = ("w",)
        elif self.arm.startswith("rr"):
            r = int(self.arm[2:])
            p = {"a": rand(d, r), "b": rand(r, 10), "bias": zeros(10)}
            penalised = ("a", "b")
        else:
            p = {"w": zeros(d, 10), "bias": zeros(10)}
            penalised = ("w",)
        optimiser = torch.optim.LBFGS(list(p.values()), max_iter=1000, tolerance_grad=1e-7, tolerance_change=1e-11,
                                      history_size=20, line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = -(target * self._logp(x_t, p)).sum(dim=1).mean() + sum((p[k] ** 2).sum() for k in penalised) / (2 * n)
            loss.backward()
            return loss

        optimiser.step(closure)
        self.params = {k: v.detach() for k, v in p.items()}
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        import torch
        return torch.exp(self._logp(torch.as_tensor(x, dtype=torch.float64), self.params)).numpy()


class LowCap(more.PriorMatched):
    """PriorMatched with the same synthetic draws and a different head."""

    def __init__(self, prior: dict, arm: str):
        super().__init__(prior)
        self.arm = arm

    def _fit_synthetic(self, games, x, y) -> None:
        rng = np.random.default_rng(hs.SEED)
        by_player = list(games.groupby(games["player_id"].astype(str)).indices.values())
        self.per_k = {}
        for k in K_BUCKETS:
            players = [rows for rows in by_player if len(rows) >= k]
            by_rank = [[p for p in players if y[p[0]] == r] for r in range(10)]
            ranks = rng.choice(10, hs.SYNTHETIC, p=self.prior[k])
            features = np.stack([x[rng.choice(by_rank[r][rng.integers(len(by_rank[r]))], k, replace=False)].mean(axis=0)
                                 for r in ranks])
            scaler = StandardScaler().fit(features)
            head = (LogisticRegression(C=L2[self.arm], max_iter=5000) if self.arm in L2
                    else TorchHead(self.arm, hs.SEED + k))
            self.per_k[k] = (scaler, head.fit(scaler.transform(features), np.asarray(ranks)))


def _fit_pool():
    import scripts.strength.rank_cvec_v2 as cv2
    games, fit, eval_games, xs = cv2.data()
    return games, fit, eval_games, xs["v2"], (fit["handicap"] <= 1).to_numpy()


def _select_task(seed, side, arm, fit, x, half, questions):
    hs.SEED = seed
    train = half == side
    model = LowCap(more.player_prior(fit[train]), arm).fit(fit[train].reset_index(drop=True), pd.DataFrame(), x[train])
    q = questions[side]
    return official_scores(score_rank(model.probabilities(x, fit["game_index"].to_numpy(), q)), q["rank_id"].to_numpy())


def select(jobs: int) -> None:
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_permove as rp
    _, fit, _, (x_fit, _), hd01 = _fit_pool()
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    half = rp._halves(fit)
    questions = {side: rp._questions(fit[half != side], 2027 + int(side)) for side in (True, False)}
    tasks = [(seed, side, arm) for seed in SELECT_SEEDS for side in (True, False) for arm in ARMS]
    scores = Parallel(n_jobs=jobs)(delayed(_select_task)(s, side, a, fit, x_fit, half, questions) for s, side, a in tasks)
    k = np.concatenate([questions[True]["k"].to_numpy(), questions[False]["k"].to_numpy()])
    table = {}
    for arm in ARMS:
        per_seed = []
        for seed in SELECT_SEEDS:
            got = {side: s for (sd, side, a), s in zip(tasks, scores) if sd == seed and a == arm}
            per_seed.append(np.concatenate([got[True], got[False]]))
        table[arm] = np.mean(per_seed, axis=0)
    base = table["base"]
    result = {"fit_games": int(len(fit)), "players": int(fit["player_id"].nunique()), "questions": int(len(k)), "arms": {}}
    for arm, s in table.items():
        result["arms"][arm] = hf.by_k(s, k) | {
            "diff_equal_k": round(hf.equal_k(s, k) - hf.equal_k(base, k), 4),
            "diff_by_k": {str(b): round(float(s[k == b].mean() - base[k == b].mean()), 4) for b in K_BUCKETS}}
        print(arm, json.dumps(result["arms"][arm]), flush=True)
    best = max((a for a in ARMS if a != "base"), key=lambda a: result["arms"][a]["diff_equal_k"])
    result["chosen"], result["proceed"] = best, bool(result["arms"][best]["diff_equal_k"] >= SELECT_MIN)
    print(json.dumps({"chosen": best, "gain": result["arms"][best]["diff_equal_k"], "proceed": result["proceed"]}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "select.json").write_text(json.dumps(result, indent=1))


def _eval_task(seed, arm, fit, x_fit, prior):
    hs.SEED = seed
    return LowCap(prior, arm).fit(fit, pd.DataFrame(), x_fit)


def evaluate(arm: str, jobs: int) -> None:
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    games, fit, eval_games, (x_fit, x_eval), hd01 = _fit_pool()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit_rows = fit[hd01].reset_index(drop=True)
    tasks = [(seed, a) for seed in hf.SEEDS for a in ("base", arm)]
    models = dict(zip(tasks, Parallel(n_jobs=jobs)(delayed(_eval_task)(s, a, fit_rows, x_fit[hd01], prior) for s, a in tasks)))
    hs.SEED = hf.SEED
    reference = more.PriorMatched(prior).fit(fit_rows, pd.DataFrame(), x_fit[hd01])
    result = {"arm": arm, "seeds": {}}
    for seed in hf.SEEDS:
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            a = hf.predict(models[seed, "base"], eval_games, x_eval, q)
            b = hf.predict(models[seed, arm], eval_games, x_eval, q)
            entry[name] = {"base": hf.by_k(a, k), arm: hf.by_k(b, k),
                           "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4) for bk in K_BUCKETS},
                           "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
            if seed == hf.SEED:
                entry[name]["bootstrap_95"] = hf.bootstrap(a, b, q)
                entry[name]["base_vs_PriorMatched_max_abs"] = float(np.abs(a - hf.predict(reference, eval_games, x_eval, q)).max())
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    mean = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"] for s in hf.SEEDS])), 4) for n in sets}
    by_k = {n: {str(bk): round(float(np.mean([result["seeds"][str(s)][n]["diff_by_k"][str(bk)] for s in hf.SEEDS])), 4)
                for bk in K_BUCKETS} for n in sets}
    promote = max(mean.values()) >= 0.005 and min(mean.values()) >= 0 and min(by_k["E2"].values()) >= -0.005
    result["G1"] = {"mean_diff_equal_k": mean, "mean_diff_by_k": by_k,
                    "decision": "promote" if promote else "probe" if min(mean.values()) >= 0 else "stop"}
    print(json.dumps(result["G1"]))
    (OUT / f"result_{arm}.json").write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ in-pool vs Valid-U diagnostic
INOUT_ARMS = ("base", "ord", "smooth", "rr2", "l2_003")
FEATURES = {"HC": None, "H": slice(30, 50)}      # column block removed from [H 30 | C 20 | HD1 flags 2]


def _view(x: np.ndarray, drop) -> np.ndarray:
    return x if drop is None else np.delete(x, np.arange(x.shape[1])[drop], axis=1)


def _inout_task(seed, side, arm, fit, x, half, q_pool, prior, eval_games, x_eval, sets):
    """One model scored on Valid-U and, when fitted on one half of the pool, on the other half too."""
    import scripts.strength.humansl_hd_filter as hf
    hs.SEED = seed
    out = {}
    if side is None:
        model = LowCap(prior, arm).fit(fit, pd.DataFrame(), x)
    else:
        train = half == side
        model = LowCap(more.player_prior(fit[train]), arm).fit(fit[train].reset_index(drop=True), pd.DataFrame(), x[train])
        out["P"] = official_scores(score_rank(model.probabilities(x, fit["game_index"].to_numpy(), q_pool[side])),
                                   q_pool[side]["rank_id"].to_numpy())
    for name, q in sets.items():
        out[name] = hf.predict(model, eval_games, x_eval, q)
    return out


def inout(jobs: int) -> None:
    """Diagnostic only (no selection, no promotion): why did the in-pool choice reverse on Valid-U?
    half -> P   models fitted on one pool half, scored on the other half   (= select)
    half -> U   the SAME models scored on Valid-U E1 / E2                  (players change, nothing else)
    full -> U   models fitted on the whole pool                            (= evaluate; fit size changes)
    each for HC (52 columns) and H only (C removed: no column was learned on pool players)."""
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_permove as rp
    games, fit, eval_games, (x_fit, x_eval), hd01 = _fit_pool()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    half = rp._halves(fit)
    q_pool = {side: rp._questions(fit[half != side], 2027 + int(side)) for side in (True, False)}
    pool_q = pd.concat([q_pool[True], q_pool[False]], ignore_index=True)
    role = fit.groupby(fit["player_id"].astype(str))["role"].first().loc[pool_q["player_id"].astype(str)].to_numpy()
    tasks = [(f, seed, side, arm) for f in FEATURES for arm in INOUT_ARMS
             for seed, side in [(s, sd) for s in SELECT_SEEDS for sd in (True, False)] + [(s, None) for s in hf.SEEDS]]
    got = dict(zip(tasks, Parallel(n_jobs=jobs)(
        delayed(_inout_task)(seed, side, arm, fit, _view(x_fit, FEATURES[f]), half, q_pool, prior, eval_games,
                             _view(x_eval, FEATURES[f]), sets) for f, seed, side, arm in tasks)))

    def scores(f, arm):
        pool = np.mean([np.concatenate([got[f, s, True, arm]["P"], got[f, s, False, arm]["P"]]) for s in SELECT_SEEDS], axis=0)
        out = {"half_P": (pool, pool_q)}
        for name, q in sets.items():
            out[f"half_{name}"] = (np.mean([got[f, s, sd, arm][name] for s in SELECT_SEEDS for sd in (True, False)], axis=0), q)
            out[f"full_{name}"] = (np.mean([got[f, s, None, arm][name] for s in hf.SEEDS], axis=0), q)
        return out

    result = {"fit_games": int(len(fit)), "players": int(fit["player_id"].nunique()),
              "pool_question_roles": {str(r): int((role == r).sum()) for r in np.unique(role)}, "features": {}}
    for f in FEATURES:
        base = scores(f, "base")
        result["features"][f] = {"base": {cell: hf.by_k(s, q["k"].to_numpy()) for cell, (s, q) in base.items()}}
        for arm in INOUT_ARMS[1:]:
            entry = {}
            for cell, (s, q) in scores(f, arm).items():
                k, a = q["k"].to_numpy(), base[cell][0]
                entry[cell] = {"diff_equal_k": round(hf.equal_k(s, k) - hf.equal_k(a, k), 4),
                               "diff_by_k": {str(b): round(float(s[k == b].mean() - a[k == b].mean()), 4) for b in K_BUCKETS},
                               "bootstrap_95": hf.bootstrap(a, s, q)}
                if cell == "half_P":
                    entry[cell]["diff_by_role"] = {str(r): round(float((s - a)[role == r].mean()), 4) for r in np.unique(role)}
            result["features"][f][arm] = entry
            print(f, arm, json.dumps({c: (e["diff_equal_k"], e["bootstrap_95"]) for c, e in entry.items()}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inout.json").write_text(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["select", "evaluate", "auto", "inout"])
    parser.add_argument("--arm")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if args.command in ("select", "auto"):
        select(args.jobs)
    if args.command == "inout":
        inout(args.jobs)
    if args.command == "evaluate":
        evaluate(args.arm, args.jobs)
    if args.command == "auto":      # Valid-U is read only when the pre-registered selection rule says so
        chosen = json.loads((OUT / "select.json").read_text())
        if chosen["proceed"]:
            evaluate(chosen["chosen"], args.jobs)


if __name__ == "__main__":
    main()
