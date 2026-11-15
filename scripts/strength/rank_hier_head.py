"""Plan step A: a shared hierarchical Gaussian head instead of four independent B2 (RESULTS 2026-11-15).

Only the head changes. Same 52 columns [H 30 | C v2 20 | HD1 flags 2], per-question game mean,
Valid-S player prior per k, score_rank decoding, fit pool (Valid-S + train20, HD<=1).

Head (every fit uses its training players only, in this order):
  1 standardise columns with player-equal weights (each game 1/n_p)
  2 rank means mu_r = equal-weight mean of the player means m_p
  3 within: Sigma_W = equal-weight mean over ranks of the player-equal mean of unbiased S_p (n_p >= 2)
  4 between: Sigma_B = equal-weight mean over ranks of T_r - mean_p(1/n_p) * Sigma_W_r, T_r the unbiased
    covariance of the m_p. The correction assumes one within covariance per rank; if a player's game
    variance differs by player and correlates with n_p it is not guaranteed unbiased.
  5 PSD: clip negative eigenvalues of Sigma_B (report count and sum|clipped| / sum|all eigenvalues|)
  6 Sigma(k) = Sigma_B + Sigma_W / k  (shared by all ranks -> linear decision form)
  7 isotropic shrinkage (1-g) Sigma + g tr(Sigma)/d I, g in GAMMAS
  8 Cholesky with escalating jitter; logit_r = log prior_k(r) + loglik_r / T_k
Colour is not modelled: within covariance mixes both colours while questions are single-colour.
The conclusion is limited to this colour-agnostic shared head.

Selection inside each outer fold: 5 inner player folds; T_k minimises the NLL of inner out-of-fold
questions on TEMPERATURES (ties -> larger T; a minimum on the grid edge voids that gamma); gamma by
inner Equal-k official score (within TIE of the best -> larger gamma). No valid gamma -> status
"calibration_out_of_range" (stop; not a statistical rejection, the grid is not widened).

    python -m scripts.strength.rank_hier_head run        # pool nested CV -> lock -> Valid-U once
Gates: pool Equal-k >= +0.003, player bootstrap P >= 0.90, every k >= -0.005; then Valid-U E1 or E2
>= +0.005, the other >= 0, every E2 k >= -0.005. The pool CV screens the head on fixed H+C features
(the C vectors saw some pool players); it is not an unseen-player test of the pipeline.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs" / "rank" / "humansl" / "hier_head"
SEED, SEEDS = 2027, (2027, 2028, 2029, 2030, 2031)
GAMMAS = (0.05, 0.1, 0.2, 0.4)
TEMPERATURES = np.geomspace(0.5, 32.0, 25)
FOLDS, PER_CELL, BOOTSTRAP, TIE = 5, 100, 2000, 0.001
POOL_GAIN, POOL_PROB, MAX_K_DROP, U_GAIN = 0.003, 0.90, 0.005, 0.005
RANKS = 10


class CalibrationOutOfRange(RuntimeError):
    pass


# ------------------------------------------------------------------ data
class Data:
    """Per-player sufficient statistics of the game features (raw scale)."""

    def __init__(self, x, player, rank, color, is_s):
        self.x, self.player, self.color = np.asarray(x, float), np.asarray(player), np.asarray(color)
        if not np.isfinite(self.x).all():
            raise ValueError("non-finite features")
        count = int(self.player.max()) + 1
        d = self.x.shape[1]
        self.n = np.bincount(self.player, minlength=count).astype(float)
        self.n_s = np.bincount(self.player[np.asarray(is_s, bool)], minlength=count)
        self.rank = np.zeros(count, int)
        self.rank[self.player] = rank
        self.mean, self.square, self.cov = np.zeros((count, d)), np.zeros((count, d)), np.zeros((count, d, d))
        order = np.argsort(self.player, kind="stable")
        bounds = np.searchsorted(self.player[order], np.arange(count + 1))
        self.rows = [order[bounds[p]:bounds[p + 1]] for p in range(count)]
        self.groups = []                                  # (player, game rows of one colour)
        for p, rows in enumerate(self.rows):
            v = self.x[rows]
            self.mean[p], self.square[p] = v.mean(0), (v * v).mean(0)
            if len(rows) >= 2:
                c = v - self.mean[p]
                self.cov[p] = c.T @ c / (len(rows) - 1)
            for value in np.unique(self.color[rows]):
                self.groups.append((p, rows[self.color[rows] == value]))

    def prior(self, players: np.ndarray) -> dict:
        """Per k, the rank share of the given players with >= k role-S games (more.player_prior)."""
        out = {}
        for k in K_BUCKETS:
            counts = np.bincount(self.rank[players][self.n_s[players] >= k], minlength=RANKS).astype(float)
            out[k] = counts / counts.sum()
        return out

    def questions(self, players: np.ndarray, rng, per_cell: int = PER_CELL) -> dict:
        """E2-style: per k x rank, per_cell single-colour questions of the given players."""
        allowed = np.zeros(len(self.n), bool)
        allowed[players] = True
        groups = [(p, rows) for p, rows in self.groups if allowed[p]]
        mean, rank, k_col, who = [], [], [], []
        for k in K_BUCKETS:
            for r in range(RANKS):
                eligible = [(p, rows) for p, rows in groups if self.rank[p] == r and len(rows) >= k]
                if not eligible:
                    raise ValueError(f"no eligible player for k={k} rank={r}")
                for pick in rng.integers(0, len(eligible), per_cell):
                    p, rows = eligible[pick]
                    mean.append(self.x[rng.choice(rows, k, replace=False)].mean(0))
                    rank.append(r), k_col.append(k), who.append(p)
        return {"mean": np.stack(mean), "rank": np.array(rank), "k": np.array(k_col), "player": np.array(who)}


def stratified_folds(data: Data, players: np.ndarray, rng) -> np.ndarray:
    fold = np.zeros(len(players), int)
    for r in range(RANKS):
        rows = np.flatnonzero(data.rank[players] == r)
        fold[rows[rng.permutation(len(rows))]] = np.arange(len(rows)) % FOLDS
    return fold


# ------------------------------------------------------------------ head
def fit_components(data: Data, players: np.ndarray) -> dict:
    """Steps 1-5 on the given training players."""
    center = data.mean[players].mean(0)
    variance = data.square[players].mean(0) - center ** 2
    if (variance < 1e-16).any():
        raise ValueError("constant column in the training players")
    scale = np.sqrt(variance)
    mean = (data.mean[players] - center) / scale
    cov = data.cov[players] / np.outer(scale, scale)
    n, rank = data.n[players], data.rank[players]
    d = len(center)
    mu, within, between = np.zeros((RANKS, d)), np.zeros((d, d)), np.zeros((d, d))
    for r in range(RANKS):
        rows = rank == r
        mu[r] = mean[rows].mean(0)
        w_r = cov[rows & (n >= 2)].mean(0)
        c = mean[rows] - mu[r]
        between += (c.T @ c / (rows.sum() - 1) - (1.0 / n[rows]).mean() * w_r) / RANKS
        within += w_r / RANKS
    between = (between + between.T) / 2
    values, vectors = np.linalg.eigh(between)
    clipped = values < 0
    info = {"clipped": int(clipped.sum()),
            "clipped_share": float(np.abs(values[clipped]).sum() / np.abs(values).sum())}
    between = (vectors * np.maximum(values, 0)) @ vectors.T
    return {"center": center, "scale": scale, "mu": mu, "within": (within + within.T) / 2, "between": between,
            "info": info}


def covariance(parts: dict, k: int, gamma: float) -> tuple[np.ndarray, dict]:
    """Steps 6-8: the Cholesky factor of the shrunk Sigma(k)."""
    sigma = parts["between"] + parts["within"] / k
    level = np.trace(sigma) / len(sigma)
    sigma = (1 - gamma) * sigma + gamma * level * np.eye(len(sigma))
    jitter = 0.0
    while True:
        try:
            chol = np.linalg.cholesky(sigma + jitter * level * np.eye(len(sigma)))
            break
        except np.linalg.LinAlgError:
            jitter = 1e-8 if jitter == 0 else jitter * 10
            if jitter > 1e-4:
                raise
    return chol, {"jitter": jitter, "condition": float(np.linalg.cond(sigma))}


def loglik(parts: dict, gamma: float, mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """[Q, 10] Gaussian log-likelihood of each question mean under each rank."""
    z = (mean - parts["center"]) / parts["scale"]
    out = np.zeros((len(z), RANKS))
    for bucket in K_BUCKETS:
        rows = np.flatnonzero(k == bucket)
        if not len(rows):
            continue
        chol, _ = covariance(parts, bucket, gamma)
        half_logdet = np.log(np.diag(chol)).sum()
        for r in range(RANKS):
            w = np.linalg.solve(chol, (z[rows] - parts["mu"][r]).T)
            out[rows, r] = -0.5 * (w * w).sum(0) - half_logdet
    return out


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def log_prior(prior: dict, k: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        table = {bucket: np.log(prior[bucket]) for bucket in K_BUCKETS}
    return np.stack([table[int(v)] for v in k])


def fit_temperatures(ll: np.ndarray, lp: np.ndarray, k: np.ndarray, truth: np.ndarray) -> dict | None:
    """Per k the NLL-minimising T (ties -> larger T); None if any minimum is on the grid edge."""
    temps = {}
    for bucket in K_BUCKETS:
        rows = k == bucket
        nll = np.array([-log_softmax(lp[rows] + ll[rows] / t)[np.arange(rows.sum()), truth[rows]].mean()
                        for t in TEMPERATURES])
        best = int(np.flatnonzero(nll == nll.min())[-1])
        if best in (0, len(TEMPERATURES) - 1):
            return None
        temps[bucket] = float(TEMPERATURES[best])
    return temps


def question_scores(ll, lp, k, truth, temps: dict) -> np.ndarray:
    t = np.array([temps[int(v)] for v in k])[:, None]
    return official_scores(score_rank(np.exp(log_softmax(lp + ll / t))), truth)


def equal_k(scores: np.ndarray, k: np.ndarray) -> float:
    return float(np.mean([scores[k == bucket].mean() for bucket in K_BUCKETS]))


def by_k(scores: np.ndarray, k: np.ndarray) -> dict:
    return {str(b): round(float(scores[k == b].mean()), 4) for b in K_BUCKETS} | {"equal_k": round(equal_k(scores, k), 4)}


def select(data: Data, players: np.ndarray, splits: list) -> dict:
    """Choose gamma and T_k from held-out predictions of `splits` = [(held players, questions)], all inside
    `players`; then fit on all of `players`."""
    ll = {g: [] for g in GAMMAS}
    lp, k, truth = [], [], []
    for held, questions in splits:
        if not np.isin(held, players).all() or not np.isin(questions["player"], held).all():
            raise AssertionError("calibration players outside the training players")
        train = np.setdiff1d(players, held)
        parts = fit_components(data, train)
        for g in GAMMAS:
            ll[g].append(loglik(parts, g, questions["mean"], questions["k"]))
        lp.append(log_prior(data.prior(train), questions["k"]))
        k.append(questions["k"]), truth.append(questions["rank"])
    lp, k, truth = np.concatenate(lp), np.concatenate(k), np.concatenate(truth)
    table = {}
    for g in GAMMAS:
        values = np.concatenate(ll[g])
        temps = fit_temperatures(values, lp, k, truth)
        table[g] = {"temps": temps,
                    "equal_k": None if temps is None else equal_k(question_scores(values, lp, k, truth, temps), k)}
    valid = [g for g in GAMMAS if table[g]["temps"] is not None]
    if not valid:
        raise CalibrationOutOfRange(json.dumps({str(g): v for g, v in table.items()}))
    best = max(table[g]["equal_k"] for g in valid)
    gamma = max(g for g in valid if table[g]["equal_k"] >= best - TIE)
    parts = fit_components(data, players)
    numeric = {str(b): covariance(parts, b, gamma)[1] for b in K_BUCKETS}
    return {"parts": parts, "gamma": gamma, "temps": table[gamma]["temps"], "prior": data.prior(players),
            "report": {"gamma": gamma, "temps": table[gamma]["temps"], "psd": parts["info"], "numeric": numeric,
                       "candidates": {str(g): {"equal_k": None if v["equal_k"] is None else round(v["equal_k"], 4),
                                               "temps": v["temps"]} for g, v in table.items()}}}


def fold_model(data: Data, players: np.ndarray, tag: tuple) -> dict:
    """Nested selection for one set of training players (inner folds and questions seeded by tag)."""
    inner = stratified_folds(data, players, np.random.default_rng([SEED, 1, *tag]))
    splits = [(players[inner == i], data.questions(players[inner == i], np.random.default_rng([SEED, 2, *tag, i])))
              for i in range(FOLDS)]
    return select(data, players, splits)


def predict(model: dict, questions: dict) -> np.ndarray:
    ll = loglik(model["parts"], model["gamma"], questions["mean"], questions["k"])
    return question_scores(ll, log_prior(model["prior"], questions["k"]), questions["k"], questions["rank"],
                           model["temps"])


def bootstrap(diff: np.ndarray, k: np.ndarray, player: np.ndarray, rng) -> dict:
    """Players resampled with all their questions; per draw the four k means, then Equal-k."""
    ids, inverse = np.unique(player, return_inverse=True)
    sums, counts = np.zeros((len(ids), len(K_BUCKETS))), np.zeros((len(ids), len(K_BUCKETS)))
    for j, bucket in enumerate(K_BUCKETS):
        rows = k == bucket
        np.add.at(sums[:, j], inverse[rows], diff[rows])
        np.add.at(counts[:, j], inverse[rows], 1)
    draws = rng.integers(0, len(ids), (BOOTSTRAP, len(ids)))
    values = (sums[draws].sum(1) / counts[draws].sum(1)).mean(1)
    return {"bootstrap_95": [round(float(v), 4) for v in np.quantile(values, [0.025, 0.975])],
            "prob_positive": round(float((values > 0).mean()), 3)}


def difference(a: np.ndarray, b: np.ndarray, k, player, rng) -> dict:
    d = a - b
    return {"equal_k": round(equal_k(d, k), 4), "by_k": {str(bk): round(float(d[k == bk].mean()), 4) for bk in K_BUCKETS}
            } | bootstrap(d, k, player, rng)


# ------------------------------------------------------------------ real data and the B2 reference
_STATE = {}


def _b2(job):
    """One PriorMatched B2 fit on the given game rows; official scores of the questions."""
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    name, seed, rows, prior, mean, k, truth = job
    fit, x = _STATE["fit"], _STATE["x"]
    hs.SEED = seed
    model = more.PriorMatched(prior).fit(fit.iloc[rows].reset_index(drop=True), pd.DataFrame(), x[rows])
    out = np.zeros(len(k))
    for bucket in K_BUCKETS:
        mask = k == bucket
        scaler, lr = model.per_k[bucket]
        out[mask] = official_scores(score_rank(lr.predict_proba(scaler.transform(mean[mask]))), truth[mask])
    return name, seed, out


def load_pool():
    import scripts.strength.rank_cvec_v2 as cv2
    games, fit, eval_games, xs = cv2.data()
    keep = (fit["handicap"] <= 1).to_numpy()
    fit, x = fit[keep].reset_index(drop=True), xs["v2"][0][keep]
    ids, player = np.unique(fit["player_id"].astype(str), return_inverse=True)
    data = Data(x, player, fit["rank_id"].to_numpy(), fit["player_color"].to_numpy(), (fit["role"] == "S").to_numpy())
    return games, fit, x, eval_games, xs["v2"][1], data, ids


def run(workers: int) -> None:
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_more_players as more
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    games, fit, x, eval_games, x_eval, data, ids = load_pool()
    _STATE.update(fit=fit, x=x)
    spec = {"gammas": GAMMAS, "temperatures": [round(float(t), 4) for t in TEMPERATURES], "folds": FOLDS,
            "per_cell": PER_CELL, "seeds": SEEDS, "bootstrap": BOOTSTRAP, "tie": TIE, "structure": "shared",
            "pool_gate": {"equal_k": POOL_GAIN, "prob": POOL_PROB, "max_k_drop": MAX_K_DROP},
            "valid_u_gate": {"gain": U_GAIN, "other": 0.0, "max_e2_k_drop": MAX_K_DROP},
            "pool": {"games": int(len(fit)), "players": int(len(ids))}}
    (OUT / "spec.json").write_text(json.dumps(spec, indent=1))
    everyone = np.arange(len(ids))
    outer = stratified_folds(data, everyone, np.random.default_rng([SEED, 0]))
    game_fold = outer[data.player]
    result = {"status": "running", "folds": {}}
    questions, jobs, a_scores = {}, [], {}
    try:
        for f in range(FOLDS):
            held, train = everyone[outer == f], everyone[outer != f]
            questions[f] = data.questions(held, np.random.default_rng([SEED, 3, f]))
            if np.isin(questions[f]["player"], train).any():
                raise AssertionError("outer test players in the training set")
            model = fold_model(data, train, (f,))
            a_scores[f] = predict(model, questions[f])
            result["folds"][str(f)] = model["report"]
            rows = np.flatnonzero(game_fold != f)
            prior = more.player_prior(fit.iloc[rows][fit["role"].to_numpy()[rows] == "S"])
            if any(not np.allclose(prior[k], model["prior"][k]) for k in K_BUCKETS):
                raise AssertionError("prior mismatch between the two arms")
            jobs += [(f, seed, rows, prior, questions[f]["mean"], questions[f]["k"], questions[f]["rank"])
                     for seed in SEEDS]
            print(f"fold {f}: A done, gamma {model['gamma']}, temps {model['temps']}", flush=True)
    except CalibrationOutOfRange as error:
        result |= {"status": "calibration_out_of_range", "detail": json.loads(str(error))}
        (OUT / "pool_cv.json").write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))
        return
    with mp.get_context("fork").Pool(workers) as pool:
        b2 = {(name, seed): scores for name, seed, scores in pool.imap_unordered(_b2, jobs)}
    k = np.concatenate([questions[f]["k"] for f in range(FOLDS)])
    player = np.concatenate([questions[f]["player"] for f in range(FOLDS)])
    a = np.concatenate([a_scores[f] for f in range(FOLDS)])
    per_seed = {seed: np.concatenate([b2[(f, seed)] for f in range(FOLDS)]) for seed in SEEDS}
    b = np.mean([per_seed[seed] for seed in SEEDS], axis=0)                 # per question, scored per seed
    table = pd.DataFrame({"fold": np.repeat(np.arange(FOLDS), [len(questions[f]["k"]) for f in range(FOLDS)]),
                          "player_id": ids[player], "k": k,
                          "rank_id": np.concatenate([questions[f]["rank"] for f in range(FOLDS)]), "hier": a,
                          "b2_mean": b} | {f"b2_{seed}": per_seed[seed] for seed in SEEDS})
    table.to_parquet(OUT / "pool_scores.parquet")
    diff = difference(a, b, k, player, np.random.default_rng([SEED, 4]))
    passed = bool(diff["equal_k"] >= POOL_GAIN and diff["prob_positive"] >= POOL_PROB
                  and min(diff["by_k"].values()) >= -MAX_K_DROP)
    result |= {"status": "pool_pass" if passed else "pool_gate_fail", "questions": int(len(k)),
               "hier": by_k(a, k), "b2_five_seed_mean": by_k(b, k),
               "b2_by_seed": {str(seed): by_k(per_seed[seed], k) for seed in SEEDS}, "hier-b2": diff,
               "seconds": round(time.time() - started)}
    (OUT / "pool_cv.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({key: result[key] for key in ("status", "hier", "b2_five_seed_mean", "hier-b2")}, indent=1), flush=True)
    if not passed:
        return

    # lock: the outer folds are the selection folds on the whole pool, then one Valid-U evaluation
    try:
        locked = select(data, everyone, [(everyone[outer == f], questions[f]) for f in range(FOLDS)])
    except CalibrationOutOfRange as error:
        (OUT / "locked.json").write_text(json.dumps({"status": "calibration_out_of_range",
                                                     "detail": json.loads(str(error))}, indent=1))
        print("lock: calibration out of range; stop")
        return
    (OUT / "locked.json").write_text(json.dumps({"status": "locked"} | locked["report"], indent=1))
    prior = more.player_prior(fit[fit["role"] == "S"])
    sets = {name: q for name, q in hf.question_sets(games, rebuild=False).items() if name in ("E1", "E2")}
    u_ids = np.unique(np.concatenate([q["player_id"].astype(str).to_numpy() for q in sets.values()]))
    final, jobs, packed = {"locked": locked["report"]}, [], {}
    for name, q in sets.items():
        mean = hs.question_mean(x_eval, eval_games["game_index"].to_numpy(), q)
        if not np.isfinite(mean).all():
            raise ValueError(f"{name}: non-finite question features")
        packed[name] = {"mean": mean, "k": q["k"].to_numpy(), "rank": q["rank_id"].to_numpy(),
                        "player": np.searchsorted(u_ids, q["player_id"].astype(str).to_numpy())}
        jobs += [(name, seed, np.arange(len(fit)), prior, mean, packed[name]["k"], packed[name]["rank"])
                 for seed in SEEDS]
    with mp.get_context("fork").Pool(workers) as pool:
        b2 = {(name, seed): scores for name, seed, scores in pool.imap_unordered(_b2, jobs)}
    for name, q in packed.items():
        a = predict(locked, q)
        b = np.mean([b2[(name, seed)] for seed in SEEDS], axis=0)
        final[name] = {"hier": by_k(a, q["k"]), "b2_five_seed_mean": by_k(b, q["k"]),
                       "b2_by_seed": {str(seed): by_k(b2[(name, seed)], q["k"]) for seed in SEEDS},
                       "hier-b2": difference(a, b, q["k"], q["player"], np.random.default_rng([SEED, 5]))}
    gains = [final[name]["hier-b2"]["equal_k"] for name in ("E1", "E2")]
    final["pass"] = bool(max(gains) >= U_GAIN and min(gains) >= 0
                         and min(final["E2"]["hier-b2"]["by_k"].values()) >= -MAX_K_DROP)
    final["seconds"] = round(time.time() - started)
    (OUT / "valid_u.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
