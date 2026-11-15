"""Question features as a move-weighted mean of the game rows (2026-11-15, user: "do it").

The base (cvec2, PriorMatched B2) averages the per-game features of a question with equal game
weights. Each feature is a per-move mean log p, so if moves are independent evidence the
question log-likelihood is the sum over all its moves: long games should count more. Here the
question row is sum_g n_g x_g / sum_g n_g, n_g the game's length (own and opponent move counts
differ by at most one). Applied the same way to the synthetic fitting questions and to the
evaluated / test questions; everything else (fit set, prior, seeds, C) is the cvec2 base.

    python -m scripts.strength.rank_lenweight evaluate     # Valid-U E1 / E2, five seeds
    python -m scripts.strength.rank_lenweight export --job ID

Pre-registered (before reading Valid-U): promote = five-seed mean Equal-k >= +0.005 on E1 or E2,
the other >= 0, every E2 k >= -0.005; LB probe if both >= -0.01 (user's near-miss rule); else stop.
Expected gain: small (< +0.003); game lengths vary little (median 228) and B2 has a scale per k.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as more

OUT = ROOT / "outputs" / "rank" / "humansl" / "lenweight"
TRAIN_GAMES = ROOT / "data" / "processed" / "boards" / "strength_v1" / "games.parquet"


def weighted_question_mean(x: np.ndarray, w: np.ndarray, game_index, questions: pd.DataFrame) -> np.ndarray:
    position = pd.Series(np.arange(len(game_index)), index=np.asarray(game_index))
    rows = [position.loc[list(g)].to_numpy() for g in questions["games"]]
    return np.stack([np.average(x[r], axis=0, weights=w[r]) for r in rows])


class LengthWeighted(more.PriorMatched):
    """PriorMatched B2 with move-weighted question means; `weights` aligned with the x rows of
    fit() and passed again (aligned with x) to probabilities()."""

    def __init__(self, prior: dict, fit_weights: np.ndarray):
        super().__init__(prior)
        self.fit_weights = fit_weights

    def _fit_synthetic(self, games, x, y) -> None:
        w = self.fit_weights
        if len(w) != len(x):
            raise ValueError("weights and x are not aligned")
        rng = np.random.default_rng(hs.SEED)               # same draws as PriorMatched
        by_player = list(games.groupby(games["player_id"].astype(str)).indices.values())
        self.per_k = {}
        for k in K_BUCKETS:
            players = [rows for rows in by_player if len(rows) >= k]
            by_rank = [[p for p in players if y[p[0]] == r] for r in range(10)]
            ranks = rng.choice(10, hs.SYNTHETIC, p=self.prior[k])
            features, labels = [], []
            for r in ranks:
                rows = by_rank[r][rng.integers(len(by_rank[r]))]
                pick = rng.choice(rows, k, replace=False)
                features.append(np.average(x[pick], axis=0, weights=w[pick]))
                labels.append(r)
            features = np.stack(features)
            scaler = StandardScaler().fit(features)
            model = LogisticRegression(C=1.0, max_iter=5000)
            self.per_k[k] = (scaler, model.fit(scaler.transform(features), np.array(labels)))

    def probabilities_w(self, x, w, game_index, questions) -> np.ndarray:
        k = (questions["k"] if "k" in questions else questions["num_games"].map(hs.k_bucket)).to_numpy()
        mean = weighted_question_mean(x, w, game_index, questions)
        out = np.empty((len(questions), 10))
        for bucket in K_BUCKETS:
            mask = k == bucket
            if mask.any():
                scaler, model = self.per_k[bucket]
                out[mask] = model.predict_proba(scaler.transform(mean[mask]))
        return out


def train_lengths(games: pd.DataFrame) -> np.ndarray:
    table = pd.read_parquet(TRAIN_GAMES, columns=["game_index", "rank", "game_id", "length"]).set_index("game_index")
    rows = table.loc[games["game_index"].to_numpy()]
    if not (rows["rank"].to_numpy() == games["rank"].to_numpy()).all() \
            or not (rows["game_id"].astype(str).to_numpy() == games["game_id"].astype(str).to_numpy()).all():
        raise ValueError("game_index does not match the strength_v1 cache")
    return rows["length"].to_numpy().astype(np.float64)


def setup():
    import scripts.strength.rank_cvec_v2 as cv2
    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    return games, fit, eval_games, x_fit, x_eval, hd01, train_lengths(fit), train_lengths(eval_games)


def evaluate() -> None:
    from go_aicup.metrics import score_rank
    from go_aicup.tasks.strength.v2.aggregate import official_scores
    games, fit, eval_games, x_fit, x_eval, hd01, w_fit, w_eval = setup()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit01 = fit[hd01].reset_index(drop=True)
    gi = eval_games["game_index"].to_numpy()
    result = {"length_cv_within_question": None, "seeds": {}}
    q2 = sets["E2"]
    pos = pd.Series(np.arange(len(gi)), index=gi)
    cvs = [w_eval[pos.loc[list(g)].to_numpy()].std() / w_eval[pos.loc[list(g)].to_numpy()].mean() for g in q2["games"]]
    result["length_cv_within_question"] = round(float(np.mean(cvs)), 4)
    for seed in hf.SEEDS:
        hs.SEED = seed
        base = more.PriorMatched(prior).fit(fit01, pd.DataFrame(), x_fit[hd01])
        arm = LengthWeighted(prior, w_fit[hd01]).fit(fit01, pd.DataFrame(), x_fit[hd01])
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            a = hf.predict(base, eval_games, x_eval, q)
            b = official_scores(score_rank(arm.probabilities_w(x_eval, w_eval, gi, q)), q["rank_id"].to_numpy())
            entry[name] = {"base": hf.by_k(a, k), "lenw": hf.by_k(b, k),
                           "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4) for bk in K_BUCKETS},
                           "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
            if seed == hf.SEED:
                entry[name]["bootstrap_95"] = hf.bootstrap(a, b, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    mean = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"] for s in hf.SEEDS])), 4) for n in sets}
    by_k = {n: {str(bk): round(float(np.mean([result["seeds"][str(s)][n]["diff_by_k"][str(bk)] for s in hf.SEEDS])), 4)
                for bk in K_BUCKETS} for n in sets}
    promote = max(mean.values()) >= 0.005 and min(mean.values()) >= 0 and min(by_k["E2"].values()) >= -0.005
    probe = min(mean.values()) >= -0.01
    result["G1"] = {"mean_diff_equal_k": mean, "mean_diff_by_k": by_k,
                    "decision": "promote" if promote else "probe" if probe else "stop"}
    print(json.dumps(result["G1"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


def export(job: str) -> None:
    """cvec2 export (Valid-S + train20 + Valid-U HD<=1, seed 2027) with move-weighted question means.
    PriorMatched on the same data must reproduce 751959-cvec2.csv."""
    import pickle
    from go_aicup.data.board_cache import RANKS
    from go_aicup.metrics import score_rank
    from go_aicup.tasks.strength import humansl_predict as hp
    from go_aicup.tasks.strength.submission_files import write_rank_csv
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_cvec_v2 as cv2
    games, fit, eval_games, x_fit, x_eval, hd01, w_fit, w_eval = setup()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    x_all, w_all = np.vstack([x_fit[hd01], x_eval[u]]), np.concatenate([w_fit[hd01], w_eval[u]])
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    base = more.PriorMatched(prior).fit(stacked, pd.DataFrame(), x_all)
    arm = LengthWeighted(prior, w_all).fit(stacked, pd.DataFrame(), x_all)
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    out = {"cvec2": [], "lenw": []}
    for part in ("public", "private"):
        _, x2, index = cv2.test_x(part, columns)
        cache = pd.read_parquet(p3.WORK / f"rank_{part}" / "cache" / "games.parquet").set_index("game_index")
        w = cache.loc[index, "length"].to_numpy().astype(np.float64)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for name, p in (("cvec2", base.probabilities(x2, index, questions)),
                        ("lenw", arm.probabilities_w(x2, w, index, questions))):
            out[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                           "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    result = {a: pd.concat(f, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
              for a, f in out.items()}
    ref = pd.read_csv(ROOT / "submissions" / "rank" / "751959-cvec2.csv", dtype=str).set_index(
        "question_id")["pred_rank"].loc[template.question_id]
    if (result["cvec2"] != ref).any():
        raise ValueError(f"cvec2 refit does not reproduce 751959-cvec2.csv ({int((result['cvec2'] != ref).sum())})")
    changed = result["lenw"] != ref
    order = {r: i for i, r in enumerate(RANKS)}
    step = result["lenw"][changed].map(order) - ref[changed].map(order)
    report = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
              "private": int(changed[changed.index.str.startswith("priv")].sum()),
              "step_counts": {str(int(s)): int(n) for s, n in step.value_counts().sort_index().items()},
              "file": write_rank_csv(result["lenw"].reset_index(), "lenw", job).name}
    print(json.dumps(report), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"export_{job}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["evaluate", "export"])
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    evaluate() if args.command == "evaluate" else export(args.job)


if __name__ == "__main__":
    main()
