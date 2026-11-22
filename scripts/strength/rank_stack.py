"""Competition 2 LB probes (2026-11-22, user "這兩個都加看看 並上傳 LB 驗證"): the current base
839169-ygbm-l10 (H 30 | C v2 20 | HD1 flags 2 | Yugen nonlinear H regressor 2) plus
  ogs   the OGS rank regressor g(own, opp), g(opp, own)            (rank_ogs; local vs hc E1 -0.0011 / E2 -0.0009)
  eng   the 14 strong-engine move-quality features (A-048 / 11/15)  (local vs hc E1 -0.0062 / E2 -0.0057)
  both  ogs + eng
Export recipe = 839169: B2 refit on Valid-S + train20 + Valid-U HD<=1, prior-matched, seed 2027; the Yugen
regressor features are player-grouped out-of-fold on that pool; then the 826812 question merge groups and
L10. The base arm must reproduce 839169-ygbm-l10.csv row for row. Afterwards a local Valid-U E1/E2 check
(five seeds; Yugen features out-of-fold on the local fit pool) of each arm against the base, for the record.

    python -m scripts.strength.rank_stack --job <jobid>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import k_bucket
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_engine_full as ef
import scripts.strength.rank_more_players as more
import scripts.strength.rank_ogs as ro
import scripts.strength.rank_prod_a0_fusion as prod
import scripts.strength.rank_qmerge as qm

OUT = ROOT / "outputs" / "rank" / "stack"
CURRENT = ROOT / "submissions" / "rank" / "839169-ygbm-l10.csv"
TAGS = {"ogs": "ygo-l10", "eng": "yge-l10", "both": "ygoe-l10"}


def ogs_model():
    rows = ro.ogs_rows()
    cols = [f"own_{j}" for j in range(15)] + [f"opp_{j}" for j in range(15)]
    return ro.regressor().fit(rows[cols].to_numpy(), rows["rank"].to_numpy())


def eng_block(game_index: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    cols = ef.columns(table, ("eng",))
    if not np.isin(game_index, table.index).all():
        raise ValueError("games without engine features")
    return table.loc[game_index, cols].to_numpy(np.float64)


def arms(base_x, h, g_model, eng) -> dict:
    ogs = ro.g_features(g_model, h)
    return {"base": base_x, "ogs": np.hstack([base_x, ogs]), "eng": np.hstack([base_x, eng]),
            "both": np.hstack([base_x, ogs, eng])}


def predict(model, x, gi, questions, groups, by_id, l10) -> pd.Series:
    sep = pd.Series([RANKS[r] for r in score_rank(model.probabilities(x, gi, questions))], index=questions["question_id"])
    members = [[np.asarray(by_id.loc[q, "games"]) for q in g] for g in groups]
    pm = qm.merged_probabilities(model, x, gi, members, [k_bucket(sum(len(m) for m in ms)) for ms in members])
    out = sep.copy()
    for g, prob in zip(groups, pm):
        for q in g:
            out[q] = RANKS[int(score_rank(prob[None])[0])]
    out.loc[l10.index] = l10
    return out


def export(fit, eval_games, x_fit, x_eval, g_model, eng_train, job) -> dict:
    u = (eval_games["handicap"] <= 1).to_numpy()
    pool = pd.concat([fit, eval_games[u]], ignore_index=True)
    x_pool = np.vstack([x_fit, x_eval[u]])
    oof, _, yfull = ro.yugen_oof(pool, x_pool, x_pool[:1])
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    pool_arms = arms(np.hstack([x_pool, oof]), x_pool[:, :30], g_model, eng_block(pool["game_index"].to_numpy(), eng_train))
    models = {a: prod._fit(pool, xa, prior, 2027) for a, xa in pool_arms.items()}

    xs, gis, qs, engs, shift = [], [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        engs.append(eng_block(gi, ef.engine(f"test_{part}")))
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        xs.append(x), gis.append(gi + shift), qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True)
    test_arms = arms(np.hstack([x, ro.g_features(yfull, x[:, :30])]), x[:, :30], g_model, np.vstack(engs))
    groups = qm.groups_from_pairs(pd.read_csv(qm.PAIRS))
    by_id = questions.set_index("question_id")
    l10 = pd.read_csv(qm.L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    current = pd.read_csv(CURRENT, dtype=str).set_index("question_id")["pred_rank"]
    preds = {a: predict(models[a], test_arms[a], gi, questions, groups, by_id, l10).loc[current.index] for a in models}
    if (preds["base"] != current).any():
        raise ValueError(f"base arm does not reproduce 839169-ygbm-l10.csv ({int((preds['base'] != current).sum())})")
    order = {r: i for i, r in enumerate(RANKS)}
    k = by_id["num_games"]
    report = {}
    for a, tag in TAGS.items():
        new = preds[a]
        diff = new != current
        pub = diff & diff.index.str.startswith("pub")
        report[a] = {"public": int(pub.sum()), "private": int((diff & diff.index.str.startswith("priv")).sum()),
                     "public_by_k": {str(b): int((pub & (k.loc[diff.index] == b).to_numpy()).sum()) for b in K_BUCKETS},
                     "steps": {str(int(s)): int(n) for s, n in
                               (new[diff].map(order) - current[diff].map(order)).value_counts().sort_index().items()},
                     "file": write_rank_csv(new.rename("pred_rank").reset_index(), tag, job).name}
    return report


def local(fit, eval_games, x_fit, x_eval, g_model, eng_train) -> dict:
    oof, y_eval, _ = ro.yugen_oof(fit, x_fit, x_eval)
    fit_arms = arms(np.hstack([x_fit, oof]), x_fit[:, :30], g_model, eng_block(fit["game_index"].to_numpy(), eng_train))
    eval_arms = arms(np.hstack([x_eval, y_eval]), x_eval[:, :30], g_model,
                     eng_block(eval_games["game_index"].to_numpy(), eng_train))
    sets = prod.question_sets()
    prior = more.player_prior(fit[fit["role"] == "S"])
    seeds = {}
    for seed in prod.SEEDS:
        entry = {}
        for name, q in sets.items():
            kk = q["k"].to_numpy()
            scores = {a: hf.predict(prod._fit(fit, fit_arms[a], prior, seed), eval_games, eval_arms[a], q) for a in fit_arms}
            entry[name] = {a: round(hf.equal_k(scores[a], kk) - hf.equal_k(scores["base"], kk), 4) for a in TAGS}
            entry[name]["by_k"] = {a: {str(b): round(float(scores[a][kk == b].mean() - scores["base"][kk == b].mean()), 4)
                                       for b in K_BUCKETS} for a in TAGS}
        seeds[str(seed)] = entry
        print(seed, json.dumps({n: {a: e[a] for a in TAGS} for n, e in entry.items()}), flush=True)
    mean = {a: {n: round(float(np.mean([seeds[str(s)][n][a] for s in prod.SEEDS])), 4) for n in sets} for a in TAGS}
    return {"seeds": seeds, "mean_equal_k_vs_base": mean}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    g_model = ogs_model()
    eng_train = ef.engine("train")
    fit, eval_games, x_fit, x_eval = prod.train_data()
    report = {"export": export(fit, eval_games, x_fit, x_eval, g_model, eng_train, args.job)}
    print(json.dumps(report["export"], indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))
    report["local"] = local(fit, eval_games, x_fit, x_eval, g_model, eng_train)
    print(json.dumps(report["local"]["mean_equal_k_vs_base"], indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
