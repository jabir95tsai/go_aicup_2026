"""k=infinity ceiling of the competition-2 base (RESULTS 2026-11-11).

Base = noA-t20-prior: PriorMatched B2 on HD<=1 games of Valid-S + train20, Valid-S prior,
x = [H 30 | C 20 | HD1 flags 2]. Evaluation units are Valid-U (player, colour) groups with
>= 20 HD<=1 games (never used in any fit).

Per group, five B2 seeds:
  curve   expected official score at k = 5/10/15/20 (DRAWS random k-subsets, model per_k[k])
  more    with the k=20 model, n = 20/30/40 games on groups with >= 40 games
  inf     mean of all the group's games through the k=20 model (the k -> infinity limit)
Oracles (fit on the evaluation groups themselves, an upper bound for any B2 refit with
these features, e.g. dropping outlier fit players):
  oracle_inf     LR on the group all-game means, in-sample and 5-fold by player
  oracle_k20     LR on the k=20 draws of the evaluation groups, in-sample

    python -m scripts.strength.rank_ceiling
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as mp

OUT = mp.OUT / "ceiling.json"
DRAWS, MIN_GAMES, MORE_GAMES = 40, 20, 40


def summary(scores: np.ndarray, ranks: np.ndarray) -> dict:
    """Group-equal and rank-balanced (E2-like) mean."""
    per_rank = [float(scores[ranks == r].mean()) for r in range(10) if (ranks == r).any()]
    return {"groups": round(float(scores.mean()), 4), "rank_balanced": round(float(np.mean(per_rank)), 4)}


def lr_scores(features, labels, eval_features, eval_labels) -> np.ndarray:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(features), labels)
    p = np.zeros((len(eval_features), 10))
    p[:, model.classes_] = model.predict_proba(scaler.transform(eval_features))
    return official_scores(score_rank(p), eval_labels)


def main() -> None:
    games, fit, eval_games, x_fit, x_eval, _ = mp.data(with_ex=False)
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    prior = mp.player_prior(fit[valid_s])
    fit_rows = fit[hd01].reset_index(drop=True)

    u = (eval_games["handicap"] <= 1).to_numpy()
    ug, xu = eval_games[u].reset_index(drop=True), x_eval[u]
    groups = [rows for rows in ug.groupby(["player_id", "player_color"]).indices.values() if len(rows) >= MIN_GAMES]
    ranks = np.array([ug["rank_id"].iat[rows[0]] for rows in groups])
    players = np.array([str(ug["player_id"].iat[rows[0]]) for rows in groups])
    sizes = np.array([len(rows) for rows in groups])
    big = sizes >= MORE_GAMES
    result = {"groups": len(groups), "players": int(len(set(players))),
              "groups_per_rank": np.bincount(ranks, minlength=10).tolist(),
              "games_per_group_median": int(np.median(sizes)), "groups_ge40": int(big.sum()), "seeds": {}}
    print(json.dumps(result), flush=True)

    rng = np.random.default_rng(hf.SEED)
    draws = {n: np.stack([np.stack([xu[rng.choice(rows, n, replace=False)].mean(axis=0) for _ in range(DRAWS)])
                          for rows in groups]) for n in K_BUCKETS}            # [G, DRAWS, 52]
    more = {n: np.stack([np.stack([xu[rng.choice(groups[g], n, replace=False)].mean(axis=0) for _ in range(DRAWS)])
                         for g in np.flatnonzero(big)]) for n in (30, MORE_GAMES)}
    full = np.stack([xu[rows].mean(axis=0) for rows in groups])
    rep = np.repeat(ranks, DRAWS)

    def expected(scaler_model, feats, labels):
        scaler, model = scaler_model
        g, d, f = feats.shape
        s = official_scores(score_rank(model.predict_proba(scaler.transform(feats.reshape(-1, f)))),
                            np.repeat(labels, d))
        return s.reshape(g, d).mean(axis=1)

    per_seed_inf, per_seed_k20 = [], []
    for seed in hf.SEEDS:
        hs.SEED = seed
        model = mp.PriorMatched(prior).fit(fit_rows, pd.DataFrame(), x_fit[hd01])
        entry = {"curve": {}}
        for n in K_BUCKETS:
            s = expected(model.per_k[n], draws[n], ranks)
            entry["curve"][str(n)] = summary(s, ranks)
            if n == 20:
                k20 = s
        scaler, lr = model.per_k[20]
        pred_inf = score_rank(lr.predict_proba(scaler.transform(full)))
        inf = official_scores(pred_inf, ranks)
        entry["inf"] = summary(inf, ranks)
        dist = np.abs(pred_inf - ranks)
        entry["inf_error"] = {"exact": round(float((dist == 0).mean()), 4), "off1": round(float((dist == 1).mean()), 4),
                              "off2plus": round(float((dist >= 2).mean()), 4),
                              "exact_by_rank": [round(float((dist[ranks == r] == 0).mean()), 3) for r in range(10)],
                              "mean_signed_error_by_rank": [round(float((pred_inf - ranks)[ranks == r].mean()), 3)
                                                            for r in range(10)]}
        entry["k20_by_inf_outcome"] = {name: {"share": round(float(m.mean()), 4),
                                              "k20_mean": round(float(k20[m].mean()), 4),
                                              "inf_mean": round(float(inf[m].mean()), 4)}
                                       for name, m in (("inf_exact", dist == 0), ("inf_off1", dist == 1),
                                                       ("inf_off2plus", dist >= 2))}
        bk = big
        entry["more_games_ge40"] = {"20": summary(k20[bk], ranks[bk])} | {
            str(n): summary(expected(model.per_k[20], more[n], ranks[bk]), ranks[bk]) for n in (30, MORE_GAMES)} | {
            "inf": summary(inf[bk], ranks[bk])}
        result["seeds"][str(seed)] = entry
        per_seed_inf.append(inf)
        per_seed_k20.append(k20)
        print(seed, json.dumps({"curve": entry["curve"], "inf": entry["inf"], "inf_error": entry["inf_error"],
                                "more": entry["more_games_ge40"]}), flush=True)

    # oracles: LR fit on the evaluation groups themselves
    oracle = {"inf_in_sample": summary(lr_scores(full, ranks, full, ranks), ranks)}
    cv = np.zeros(len(groups))
    for train, test in GroupKFold(5).split(full, ranks, players):
        cv[test] = lr_scores(full[train], ranks[train], full[test], ranks[test])
    oracle["inf_cv5_by_player"] = summary(cv, ranks)
    flat = draws[20].reshape(-1, draws[20].shape[-1])
    oracle["k20_in_sample"] = summary(lr_scores(flat, rep, flat, rep).reshape(len(groups), DRAWS).mean(axis=1), ranks)
    result["oracle"] = oracle

    mean = lambda key: {m: round(float(np.mean([key(result["seeds"][str(s)])[m] for s in hf.SEEDS])), 4)
                        for m in ("groups", "rank_balanced")}
    result["mean"] = {"curve": {str(n): mean(lambda e, n=n: e["curve"][str(n)]) for n in K_BUCKETS},
                      "inf": mean(lambda e: e["inf"]),
                      "more_games_ge40": {n: mean(lambda e, n=n: e["more_games_ge40"][n]) for n in ("20", "30", "40", "inf")},
                      "inf_error": {m: round(float(np.mean([result["seeds"][str(s)]["inf_error"][m] for s in hf.SEEDS])), 4)
                                    for m in ("exact", "off1", "off2plus")}}
    print(json.dumps({"mean": result["mean"], "oracle": oracle}), flush=True)
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
