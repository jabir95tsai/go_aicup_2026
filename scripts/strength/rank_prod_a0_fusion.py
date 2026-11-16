"""HC calibration and HC + A0 posterior fusion on the production pipeline (RESULTS 2026-11-16).

Step 1 rebuilds the current base's features from data that still exist after the 11/16 cleanup
(no A-024 categories, no C v1, no test rankvec, missing outputs/rank_test_ahc_713770 links) and must
reproduce (a) cvec_v2/result.json seed-2027 v2 E1/E2 scores and (b) submissions/rank/751959-cvec2.csv
question by question. Step 2 runs only if both hold.

    python -m scripts.strength.rank_prod_a0_fusion run        # sbatch slurm/rank_prod_a0_fusion.slurm

Arms: HC (PriorMatched B2, Valid-S + train20 HD<=1, Valid-S prior), HC_cal (alpha only), HC+A0
(alpha log p_HC + beta log p_A0). (alpha, beta) per k and seed from inner player folds of the fit pool;
calibration questions exclude the 1,200 players A0 was trained / selected on.
Gate (five-seed means): HC+A0 - HC_cal: E1 or E2 >= +0.005, the other >= 0, every E2 k >= -0.005;
the same rule reported for HC_cal - HC.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

import scripts.strength.rank_hc_s_fusion as fu

OUT = ROOT / "outputs" / "rank" / "humansl" / "prod_a0_fusion"
RELEASE = ROOT / "data" / "processed" / "release" / "rank_test_ahc_713770" / "rank"
BASE = ROOT / "submissions" / "rank" / "751959-cvec2.csv"
SEEDS = (2027, 2028, 2029, 2030, 2031)
FOLDS, CAL_CELL = 5, 200
GAIN, MAX_K_DROP = 0.005, 0.005


# ------------------------------------------------------------------ step 1: features
def _h_frame(path) -> pd.DataFrame:
    return fu._h_frame("x", path)


def train_data():
    """(fit, eval_games, x_fit, x_eval): fit = Valid-S HD<=1 then train20 (standard-array order), eval = Valid-U;
    x = [H 30 | C v2 20 | HD1 flags 2]."""
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_train20_hc as t20
    meta = t20.metadata()
    folds = pd.read_parquet(ROOT / "data" / "processed" / "cv" / "strength_folds_seed2027.parquet").set_index("player_id")["fold"]
    valid_ids = set(pd.concat([pd.read_parquet(cv2.OUT / f"ext_valid_hd{h}.parquet") for h in (0, 1)])["game_index"])
    std_index = np.load(hf.STANDARD / "game_index.npy")
    std_h = pd.DataFrame(np.load(hf.STANDARD / "features.npy"), index=std_index)
    order = std_index[np.isin(std_index, list(valid_ids))]
    if len(order) != len(valid_ids):
        raise ValueError("Valid games missing from the H standard array")
    valid = meta.set_index("game_index").loc[order].reset_index()
    valid["role"] = np.where(folds.loc[valid["player_id"]].to_numpy() == 0, "U", "S")
    if ((valid["role"] == "S") & (valid["handicap"] > 1)).any():
        raise ValueError("Valid-S HD>=2 game in the C v2 table")
    t20g = t20.train20_games()
    h_valid = std_h.loc[valid["game_index"]].to_numpy()
    h1 = _h_frame(p3.OUT / "h_train")
    rows = np.flatnonzero((valid["handicap"] == 1).to_numpy())
    if not np.isin(valid["game_index"].to_numpy()[rows], h1.index).all():
        raise ValueError("Valid HD1 game without komi-0.5 H")
    h_valid[rows] = h1.loc[valid["game_index"].to_numpy()[rows]].to_numpy()
    h_t20 = np.full((len(t20g), 30), np.nan)
    hd = t20g["handicap"].to_numpy()
    h_t20[hd == 0] = std_h.loc[t20g["game_index"].to_numpy()[hd == 0]].to_numpy()
    t1 = _h_frame(t20.OUT / "h_hd1")
    h_t20[hd == 1] = t1.loc[t20g["game_index"].to_numpy()[hd == 1]].to_numpy()
    c = cv2.new_c("train")
    games = pd.concat([valid, t20g[valid.columns]], ignore_index=True)
    h = np.vstack([h_valid, h_t20])
    if not set(games["game_index"]) <= set(c.index):
        raise ValueError("games without C v2")
    x = np.hstack([h, c.loc[games["game_index"]].to_numpy(), hf.hd1_columns(games)])
    if not np.isfinite(x).all():
        raise ValueError("non-finite training features")
    is_fit = ((games["role"] != "U") & (games["handicap"] <= 1)).to_numpy()
    is_eval = (games["role"] == "U").to_numpy()
    fit, eval_games = games[is_fit].reset_index(drop=True), games[is_eval].reset_index(drop=True)
    return fit, eval_games, x[is_fit], x[is_eval]


def test_data(part: str):
    """(x, game_index, questions) of one test part, C v2 and komi-0.5 H for HD1 games."""
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_cvec_v2 as cv2
    from go_aicup.tasks.strength import humansl_predict as hp
    work = RELEASE / f"rank_{part}"
    a024 = hp.a024_per_game(np.load(work / "category.npy", allow_pickle=False))
    human = hp.release_features(work / "humansl").set_index("game_index")
    hcols = [c for c in human.columns if c.startswith(("own_", "opp_"))]
    if [c.split("_", 1)[0] for c in hcols] != ["own"] * 15 + ["opp"] * 15:
        raise ValueError("unexpected test H column order")
    table = a024.join(human[hcols], how="inner").join(
        hp.hd1_features(work, ROOT / "data" / "raw" / "tests" / f"rank_prediction_test_{part}.csv"), how="inner")
    hd1 = table.index[table[hp.HD1_COLUMNS].sum(axis=1) > 0]
    h1 = _h_frame(p3.OUT / f"h_test_{part}")
    if not set(hd1) <= set(h1.index):
        raise ValueError(f"{part}: missing komi-0.5 H")
    table.loc[hd1, hcols] = h1.loc[hd1].to_numpy()
    c = cv2.new_c(f"test_{part}")
    if not set(table.index) <= set(c.index):
        raise ValueError(f"{part}: games without C v2")
    x = np.hstack([table[hcols].to_numpy(), c.loc[table.index].to_numpy(), table[hp.HD1_COLUMNS].to_numpy()])
    return x, table.index.to_numpy(), hp.question_table(work)


def question_sets() -> dict:
    import scripts.strength.humansl_hd_filter as hf
    e1 = pd.read_parquet(hf.Q1000)
    return {"E1": e1[e1["role"] == "U"].reset_index(drop=True),
            "E2": pd.read_parquet(hf.OUT / "questions_e2_hd01.parquet")}


def _fit(fit: pd.DataFrame, x: np.ndarray, prior: dict, seed: int):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    hs.SEED = seed
    return more.PriorMatched(prior).fit(fit.reset_index(drop=True), pd.DataFrame(), x)


def step1(fit, eval_games, x_fit, x_eval, sets) -> dict:
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_more_players as more
    report = {"fit_games": int(len(fit)), "fit_players": int(fit["player_id"].nunique()),
              "eval_games": int(len(eval_games))}
    prior = more.player_prior(fit[fit["role"] == "S"])
    model = _fit(fit, x_fit, prior, 2027)
    reference = json.loads((cv2.OUT / "result.json").read_text())["seeds"]["2027"]
    local = {}
    for name, q in sets.items():
        local[name] = hf.by_k(hf.predict(model, eval_games, x_eval, q), q["k"].to_numpy())
    report["local"] = local
    report["local_matches"] = all(local[n] == reference[n]["v2"] for n in sets)
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit, eval_games[u]], ignore_index=True)
    prior_all = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    model_all = _fit(stacked, np.vstack([x_fit, x_eval[u]]), prior_all, 2027)
    rows = []
    for part in ("public", "private"):
        x, index, questions = test_data(part)
        p = model_all.probabilities(x, index, questions)
        rows.append(pd.DataFrame({"question_id": questions["question_id"], "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    ours = pd.concat(rows).set_index("question_id")["pred_rank"]
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")["pred_rank"]
    ours = ours.loc[base.index]
    report["export_questions"] = int(len(ours))
    report["export_changed"] = int((ours != base).sum())
    report["pass"] = bool(report["local_matches"] and report["export_changed"] == 0 and len(ours) == 800)
    return report


# ------------------------------------------------------------------ step 2
_STATE: dict = {}


def _job(job):
    """fold -1: full fit pool (Valid-S prior), Valid-U E1 / E2 posteriors. fold i: fit without inner fold i
    (prior from its Valid-S players), posteriors of the calibration questions of fold i."""
    import scripts.strength.rank_more_players as more
    seed, fold, train_players, kind, q_rows = job
    fit, x = _STATE["fit"], _STATE["x_fit"]
    if kind == "eval":
        model = _fit(fit, x, _STATE["prior"], seed)
        index = _STATE["eval_games"]["game_index"].to_numpy()
        return seed, fold, kind, None, {name: model.probabilities(_STATE["x_eval"], index, q)
                                        for name, q in _STATE["sets"].items()}
    rows = np.flatnonzero(fit["player_id"].astype(str).isin(train_players).to_numpy())
    sub = fit.iloc[rows]
    model = _fit(sub, x[rows], more.player_prior(sub[sub["role"] == "S"]), seed)
    return seed, fold, kind, q_rows, model.probabilities(x, fit["game_index"].to_numpy(), _STATE["cal"].iloc[q_rows])


def calibration_questions(fit: pd.DataFrame, exclude: set) -> tuple[pd.DataFrame, pd.Series]:
    """Single-colour HD<=1 questions of the selection players (fit pool minus `exclude`), per k x rank
    CAL_CELL, plus the inner fold of every fit-pool player (stratified by rank)."""
    players = fit.groupby(fit["player_id"].astype(str))["rank_id"].first().sort_index()
    fold = fu._stratified(players, np.random.default_rng(2027))
    pool = fit[~fit["player_id"].astype(str).isin(exclude)]
    groups = pool.groupby(["rank_id", pool["player_id"].astype(str), "player_color"])["game_index"].apply(np.sort)
    rng = np.random.default_rng(2028)
    rows = []
    for k in K_BUCKETS:
        for rank in range(10):
            eligible = [(key, g) for key, g in groups.loc[rank].items() if len(g) >= k]
            for pick in rng.integers(0, len(eligible), CAL_CELL):
                (player, _), g = eligible[pick]
                rows.append({"player_id": player, "rank_id": rank, "k": k,
                             "games": np.sort(rng.choice(g, k, replace=False)).tolist()})
    q = pd.DataFrame(rows)
    q["inner"] = fold.loc[q["player_id"]].to_numpy()
    return q, fold


def gate(diff_by_seed: dict) -> dict:
    """Five-seed means of E1 / E2 Equal-k differences and E2 per-k differences."""
    mean = {n: round(float(np.mean([d[n]["equal_k"] for d in diff_by_seed.values()])), 4) for n in ("E1", "E2")}
    e2k = {str(k): round(float(np.mean([d["E2"]["by_k"][str(k)] for d in diff_by_seed.values()])), 4) for k in K_BUCKETS}
    ok = max(mean.values()) >= GAIN and min(mean.values()) >= 0 and min(e2k.values()) >= -MAX_K_DROP
    return {"mean_equal_k": mean, "E2_mean_by_k": e2k, "pass": bool(ok)}


def run(workers: int) -> None:
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_more_players as more
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fit, eval_games, x_fit, x_eval = train_data()
    sets = question_sets()
    report = step1(fit, eval_games, x_fit, x_eval, sets)
    (OUT / "step1.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)
    if not report["pass"]:
        raise SystemExit("step 1 did not reproduce the base; stop")
    roles = pd.read_parquet(fu.RUN / "games.parquet")
    exclude = set(roles.loc[roles["role"].isin(("train", "monitor")), "player_id"].astype(str))
    cal, fold = calibration_questions(fit, exclude)
    cal.to_parquet(OUT / "questions_cal.parquet")
    fold.rename("inner").to_frame().to_parquet(OUT / "inner_folds.parquet")
    _STATE.update(fit=fit, x_fit=x_fit, eval_games=eval_games, x_eval=x_eval, sets=sets, cal=cal,
                  prior=more.player_prior(fit[fit["role"] == "S"]))
    jobs = []
    for seed in SEEDS:
        jobs.append((seed, -1, None, "eval", None))
        for i in range(FOLDS):
            jobs.append((seed, i, set(fold.index[fold != i]), "cal", np.flatnonzero(cal["inner"].to_numpy() == i)))
    hc_cal = {s: np.zeros((len(cal), 10)) for s in SEEDS}
    hc_eval = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for n, (seed, f, kind, rows, p) in enumerate(pool.imap_unordered(_job, jobs), 1):
            if kind == "cal":
                hc_cal[seed][rows] = p
            else:
                hc_eval[seed] = p
            print(f"{n}/{len(jobs)} B2 fits, {time.time() - started:.0f}s", flush=True)
    # A0 posteriors (C v2 = columns 30:50)
    a0_cal = fu._logp(fu.frozen_posterior("A0", x_fit[:, 30:50], fit["game_index"].to_numpy(), cal))
    a0_eval = {n: fu._logp(fu.frozen_posterior("A0", x_eval[:, 30:50], eval_games["game_index"].to_numpy(), q))
               for n, q in sets.items()}
    truth_c, k_c = cal["rank_id"].to_numpy(), cal["k"].to_numpy()
    scores = {arm: {n: {} for n in sets} for arm in ("HC", "HC_cal", "HC+A0")}
    chosen = []
    for seed in SEEDS:
        lp_c = fu._logp(hc_cal[seed])
        for arm, betas in (("HC_cal", (0.0,)), ("HC+A0", fu.BETAS)):
            weights = {}
            for k in K_BUCKETS:
                rows = k_c == k
                weights[k] = fu._select(lp_c[rows], a0_cal[rows], truth_c[rows], betas)
                chosen.append({"arm": arm, "seed": seed, "k": k, "alpha": weights[k][0], "beta": weights[k][1]})
            for n, q in sets.items():
                lp_e, ke = fu._logp(hc_eval[seed][n]), q["k"].to_numpy()
                out = np.zeros(len(q))
                for k in K_BUCKETS:
                    r = ke == k
                    out[r] = fu._score(fu._fuse(lp_e[r], a0_eval[n][r], *weights[k]), q["rank_id"].to_numpy()[r])
                scores[arm][n][seed] = out
        for n, q in sets.items():
            scores["HC"][n][seed] = fu._score(hc_eval[seed][n], q["rank_id"].to_numpy())
    result = {"step1": report, "arms": {}, "comparisons": {}, "A0_alone": {}}
    for n, q in sets.items():
        ke = q["k"].to_numpy()
        result["A0_alone"][n] = hf.by_k(fu._score(np.exp(a0_eval[n]), q["rank_id"].to_numpy()), ke)
        for arm in scores:
            result["arms"].setdefault(arm, {})[n] = {
                "mean": hf.by_k(np.mean([scores[arm][n][s] for s in SEEDS], axis=0), ke),
                "by_seed": {str(s): hf.by_k(scores[arm][n][s], ke)["equal_k"] for s in SEEDS}}
    for a, b in (("HC_cal", "HC+A0"), ("HC", "HC_cal"), ("HC", "HC+A0")):
        per_seed = {}
        for s in SEEDS:
            per_seed[s] = {}
            for n, q in sets.items():
                ke, d = q["k"].to_numpy(), scores[b][n][s] - scores[a][n][s]
                per_seed[s][n] = {"equal_k": float(np.mean([d[ke == k].mean() for k in K_BUCKETS])),
                                  "by_k": {str(k): float(d[ke == k].mean()) for k in K_BUCKETS}}
        entry = gate(per_seed)
        entry["E2_bootstrap_95_seed2027"] = hf.bootstrap(scores[a]["E2"][2027], scores[b]["E2"][2027], sets["E2"])
        result["comparisons"][f"{b} - {a}"] = entry
    selected = pd.DataFrame(chosen)
    selected.to_parquet(OUT / "selected_weights.parquet")
    result["alpha_counts"] = {arm: {str(a): int(n) for a, n in v["alpha"].value_counts().sort_index().items()}
                              for arm, v in selected.groupby("arm")}
    result["beta_counts"] = {str(b): int(n) for b, n in
                             selected.loc[selected["arm"] == "HC+A0", "beta"].value_counts().sort_index().items()}
    result["calibration_questions"] = int(len(cal))
    result["excluded_players"] = int(len(exclude))
    result["seconds"] = round(time.time() - started)
    for n, q in sets.items():
        table = q[["player_id", "rank_id", "k"]].copy()
        for arm in scores:
            for s in SEEDS:
                table[f"{arm}_{s}"] = scores[arm][n][s]
        table.to_parquet(OUT / f"scores_{n}.parquet")
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("arms", "comparisons", "A0_alone", "alpha_counts", "beta_counts")},
                     indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["step1", "run"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    if args.command == "step1":
        fit, eval_games, x_fit, x_eval = train_data()
        report = step1(fit, eval_games, x_fit, x_eval, question_sets())
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "step1.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
