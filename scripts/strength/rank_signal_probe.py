"""Player-level probe for signals that H + C do not carry (RESULTS 2026-11-13).

The base saturates on player-level bias: with all of a player's games the own-side likelihood
features still point one rank off for ~30% of A-3D players. A new source is useful only if its
player-level error is not the same error. Probe: one row per player (mean of the 52 base features
over the player's HD<=1 games in the fitting pool or Valid-U, >= MIN_GAMES games), 5 folds by player
stratified by rank, multinomial LR (as B2) on the base features with and without a candidate block;
official score of the held-out players, rank-balanced. Every block is built without the held-out
players' labels (in-fold).

Blocks
  style    PCA (in-fold, COMPONENTS) of the player's GE2E centroid (rank_opp_link gallery vectors)
  knn      rank histogram of the NEIGHBOURS nearest training-fold players in GE2E space (cosine,
           softmax temperature TEMPERATURE)
  opening  mean per game of the opening-model log-likelihood (pp+slot, D4) of the player's games
           under each rank, the ten rank "galleries" being the training-fold players' HD<=1 games
           (black and white separately; 20 columns, centred per colour)
The opening block of a training player is scored by galleries of other inner folds (first run,
770109, left the player's own games in the gallery: the block looked perfect on the fit side and
the stacked model collapsed, -0.31; that number is void).
Caveat: the GE2E encoder was trained on most of these players (not on pid splits 0-2); the result
is also reported on the players it never saw. This is a k = infinity screen; a block that passes
(>= +0.01 with a player-bootstrap interval above 0) goes to a question-level test, others stop.

    python -m scripts.strength.rank_signal_probe [--jobs 5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT

OUT = ROOT / "outputs" / "rank" / "humansl" / "signal_probe"
MIN_GAMES, FOLDS, SEED = 10, 5, 2027
COMPONENTS, NEIGHBOURS, TEMPERATURE = 24, 100, 0.02
OPENING_CAP = 40
E = float(np.exp(-1))


def lr_predict(x_train, y_train, x_test) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(x_train), y_train)
    p = np.zeros((len(x_test), 10))
    p[:, model.classes_] = model.predict_proba(scaler.transform(x_test))
    return score_rank(p)


def fold_blocks(fold, train, test, players, centroids, style_ok, opening_fn):
    from sklearn.decomposition import PCA
    y = players["rank_id"].to_numpy()
    out = {}
    pca = PCA(COMPONENTS, random_state=SEED).fit(centroids[train])
    out["style"] = (pca.transform(centroids[train]), pca.transform(centroids[test]))

    def knn(rows, exclude_self):
        s = centroids[rows] @ centroids[train].T
        if exclude_self:
            s[np.arange(len(rows)), np.arange(len(rows))] = -np.inf
        idx = np.argpartition(-s, NEIGHBOURS - 1, axis=1)[:, :NEIGHBOURS]
        top = np.take_along_axis(s, idx, 1)
        w = np.exp((top - top.max(1, keepdims=True)) / TEMPERATURE)
        hist = np.stack([(w * (y[train][idx] == r)).sum(1) for r in range(10)], axis=1)
        return hist / hist.sum(1, keepdims=True)
    out["knn"] = (knn(train, True), knn(test, False))
    out["opening"] = opening_fn(train, test)
    return fold, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_opp_link as ol
    import scripts.strength.rank_own_link as wl
    from go_aicup.tasks.identity import data, opening
    from go_aicup.tasks.identity.predict import PLIES, load_params
    from scripts.strength.l10_check import score_questions

    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    pool = pd.concat([fit, eval_games], ignore_index=True)
    x = np.vstack([x_fit, x_eval])
    keep = (pool["handicap"] <= 1).to_numpy() & np.isfinite(x).all(axis=1)
    pool, x = pool[keep].reset_index(drop=True), x[keep]
    pid = pool["player_id"].astype(str).to_numpy()
    codes, names = pd.factorize(pid, sort=True)
    count = np.bincount(codes)
    means = np.stack([np.bincount(codes, weights=x[:, j]) for j in range(x.shape[1])], axis=1) / count[:, None]
    players = pd.DataFrame({"player_id": names, "games": count,
                            "rank_id": pd.Series(pool["rank_id"].to_numpy()).groupby(codes).first().to_numpy(),
                            "valid_u": pd.Series((pool["role"] == "U").to_numpy()).groupby(codes).first().to_numpy()})
    big = (players["games"] >= MIN_GAMES).to_numpy()
    players, means = players[big].reset_index(drop=True), means[big]

    meta = ol.load_train()
    candidates, column, centroids_all, _, gallery_size = wl.centroids_and_ranks(meta)
    centroids = centroids_all[column.loc[players["player_id"]].to_numpy()]
    gallery_table, _ = data.pid_splits(data.load_meta())
    unseen = set(gallery_table.loc[gallery_table["split"].isin(ol.EXCLUDED_SPLITS), "player_id"].astype(str))
    players["ge2e_unseen"] = players["player_id"].isin(unseen)
    y = players["rank_id"].to_numpy()
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(players), int)
    for r in range(10):
        rows = np.flatnonzero(y == r)
        fold[rows[rng.permutation(len(rows))]] = np.arange(len(rows)) % FOLDS
    print(f"players {len(players)} (Valid-U {int(players['valid_u'].sum())}, GE2E-unseen {int(players['ge2e_unseen'].sum())})",
          flush=True)

    # opening by rank: every HD<=1 game of the probe players (capped per colour) against rank galleries
    params = load_params("d4")
    prefix = data.opening_prefix_d4(meta, PLIES)
    hashes = opening.prefix_hashes(prefix)
    m_player = meta["player_id"].astype(str).to_numpy()
    m_color = meta["player_color"].to_numpy()
    m_ok = (meta["handicap"].to_numpy() <= 1) & np.isin(m_player, players["player_id"])
    row_of = pd.Series(np.arange(len(players)), index=players["player_id"])
    by_color = {}
    for color in (0, 1):
        idx = np.flatnonzero(m_ok & (m_color == color))
        groups = pd.Series(idx).groupby(row_of.loc[m_player[idx]].to_numpy()).apply(np.asarray)
        by_color[color] = {int(p): g for p, g in groups.items()}

    def opening_fn(train, test):
        """Train rows are scored by rank galleries of the other inner folds (their own games never
        count for them); test rows by the galleries of all training players."""
        blocks = {"train": np.zeros((len(train), 20)), "test": np.zeros((len(test), 20))}
        pick = np.random.default_rng(SEED)
        inner = pick.permutation(len(train)) % 4
        for color in (0, 1):
            have = by_color[color]

            def score(gallery_rows, rows):
                galleries = [np.concatenate([have[p] for p in gallery_rows if p in have and y[p] == r]) for r in range(10)]
                present = [i for i, p in enumerate(rows) if p in have]
                queries = [have[rows[i]] if len(have[rows[i]]) <= OPENING_CAP
                           else pick.choice(have[rows[i]], OPENING_CAP, replace=False) for i in present]
                s = score_questions(prefix, hashes, galleries, queries, np.concatenate(galleries), color, params)
                s = s / np.array([len(q) for q in queries])[:, None]
                return present, s - s.mean(1, keepdims=True)

            present, s = score(train, test)
            blocks["test"][present, 10 * color:10 * color + 10] = s
            for j in range(4):
                part = np.flatnonzero(inner == j)
                present, s = score(train[inner != j], train[part])
                blocks["train"][part[present], 10 * color:10 * color + 10] = s
        return blocks["train"], blocks["test"]

    from joblib import Parallel, delayed
    folds = [(f, np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(FOLDS)]
    built = dict(Parallel(n_jobs=args.jobs)(delayed(fold_blocks)(f, tr, te, players, centroids, None, opening_fn)
                                            for f, tr, te in folds))
    arms = {"base": [], "style": ["style"], "knn": ["knn"], "opening": ["opening"], "all": ["style", "knn", "opening"],
            "style_only": ["style"], "knn_only": ["knn"], "opening_only": ["opening"]}
    pred = {a: np.zeros(len(players), int) for a in arms}
    for f, tr, te in folds:
        for arm, blocks in arms.items():
            base = [] if arm.endswith("_only") else [(means[tr], means[te])]
            parts = base + [built[f][b] for b in blocks]
            pred[arm][te] = lr_predict(np.hstack([p[0] for p in parts]), y[tr], np.hstack([p[1] for p in parts]))
    score = {a: ((p == y) + E * (np.abs(p - y) == 1)).astype(float) for a, p in pred.items()}

    def balanced(s, mask):
        return float(np.mean([s[mask & (y == r)].mean() for r in range(10) if (mask & (y == r)).any()]))

    subsets = {"all": np.ones(len(players), bool), "valid_u": players["valid_u"].to_numpy(),
               "ge2e_unseen": players["ge2e_unseen"].to_numpy(), "A-3D": (y >= 3) & (y <= 6),
               "games>=40": (players["games"] >= 40).to_numpy()}
    result = {"players": int(len(players)), "subsets": {k: int(v.sum()) for k, v in subsets.items()}, "score": {}, "diff": {}}
    boot = np.random.default_rng(SEED)
    for name, mask in subsets.items():
        result["score"][name] = {a: round(balanced(s, mask), 4) for a, s in score.items()}
        rows = np.flatnonzero(mask)
        result["diff"][name] = {}
        for arm in ("style", "knn", "opening", "all"):
            draws = []
            for _ in range(1000):
                pick = rows[boot.integers(0, len(rows), len(rows))]
                m = np.zeros(len(players), bool)
                w = np.bincount(pick, minlength=len(players))
                draws.append(float(np.mean([np.average(score[arm][y == r] - score["base"][y == r], weights=w[y == r])
                                            for r in range(10) if w[y == r].sum() > 0])))
            result["diff"][name][arm] = {"mean": round(balanced(score[arm], mask) - balanced(score["base"], mask), 4),
                                         "boot95": [round(float(q), 4) for q in np.quantile(draws, [0.025, 0.975])]}
    OUT.mkdir(parents=True, exist_ok=True)
    players.assign(**{f"pred_{a}": p for a, p in pred.items()}).to_parquet(OUT / "players.parquet", index=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
