"""P1 (HANDICAP_V11.md): fit and validate the AH+C B2 model without HD>=2 games.

Mechanism: in HD>=2 games the SGF move labels are reversed while the `color`
column is the real colour, so own/opponent features of those games are swapped
and their boards lack the handicap stones (Q&A Q13, scripts/audit_handicap_semantics.py).
The rank test files hold no HD>=2 game.

Arms (only the fitting games change; features, B2, C=1 and decoding are fixed):
  all   B2 on every Valid-S game with features (65,223, as the production model)
  hd01  the same games minus HD>=2
Evaluation (Valid-U, never fitted):
  E1      the 1,000 A-044 Valid-U questions (reference; the production model
          scored through this feature assembly must give 0.6934)
  E1_hd01 the E1 questions without an HD>=2 game (subset, biased toward
          players without handicap games; diagnostic only)
  E2      new HD<=1 questions, the gate set: per k 500 questions, 50 per rank;
          a player-colour group with >= k HD<=1 games drawn uniformly within the
          rank, k distinct games, seed 2027. Saved for later comparisons.
Pre-registered gate (HANDICAP_V11.md): on E2, Equal-k(hd01) - Equal-k(all)
>= +0.005 and every k >= -0.005; player-grouped paired bootstrap 95% reported,
an interval spanning zero only makes a candidate. B2 draws synthetic questions
at random, so both arms are also refitted with 5 synthetic seeds to show that noise.

P2 (same sets, gate and seeds; E2 is read back, not rebuilt): every Valid-S game,
AH+C vs AH+C plus [HD1 & player black, HD1 & player white]; HD is part of the
test SGF, so it is an input, not a leak. fit_hd01 saves the P1 hd01 arm (seed
2027) as an LB-probe model in the production pickle format.

P12 combines both changes (HD<=1 fitting games plus the HD1 columns) against the
production setup, same sets, gate and seeds.

fit_p12 saves the P1+P2 arm (seed 2027) as a submission model; aggregate it with
humansl_predict aggregate --raw-csv <official question CSV>.

    python scripts/humansl_hd_filter.py p1 | p2 | p12 | fit_hd01 | fit_p12
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

STANDARD = PROCESSED / "strength_train_hc" / "standard"
CANONICAL = PROCESSED / "reproduce_paper" / "seed2026" / "games.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
A024 = ROOT / "outputs" / "rank" / "v2" / "v2_strength"
Q1000 = PROCESSED / "strength_train_hc" / "legacy" / "humansl_q1000" / "questions.parquet"
PRODUCTION = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec_model" / "model.pkl"
OUT = ROOT / "outputs" / "rank" / "humansl" / "hd_filter"
SEED, PER_RANK, BOOTSTRAP, SEEDS = 2027, 50, 2000, (2027, 2028, 2029, 2030, 2031)
MIN_GAIN, MAX_DROP = 0.005, 0.005


def equal_k(scores, k) -> float:
    return float(np.mean([scores[k == b].mean() for b in K_BUCKETS]))


def by_k(scores, k) -> dict:
    return {str(b): round(float(scores[k == b].mean()), 4) for b in K_BUCKETS} | {"equal_k": round(equal_k(scores, k), 4)}


def load_games() -> tuple[pd.DataFrame, np.ndarray]:
    """Games with A (A-024 mean log-prob), H and C features; x = [A | H | C]."""
    meta = pd.read_parquet(CANONICAL, columns=["game_index", "rank", "game_id", "rank_id", "player_id", "player_color"])
    meta = meta.merge(pd.read_csv(HANDICAP, usecols=["rank", "game_id", "handicap"]), on=["rank", "game_id"],
                      validate="one_to_one")
    index = np.load(STANDARD / "game_index.npy")
    hc = np.load(STANDARD / "features.npy")
    parts = []
    for role, name in (("U", "validation_unseen_category.npy"), ("S", "validation_seen_category.npy")):
        category = np.load(A024 / name)
        parts.append(pd.DataFrame(category["log_probability_sum"] / category["num_moves"][:, None],
                                  index=category["game_index"].astype(np.int64)).assign(role=role))
    a = pd.concat(parts)
    keep = np.isin(index, a.index) & np.isfinite(hc).all(axis=1)
    index, hc = index[keep], hc[keep]
    games = meta.set_index("game_index").loc[index].reset_index()
    games["role"] = a.loc[index, "role"].to_numpy()
    x = np.hstack([a.loc[index].drop(columns="role").to_numpy(), hc])
    return games, x


def build_e2(games: pd.DataFrame) -> pd.DataFrame:
    pool = games[(games["role"] == "U") & (games["handicap"] <= 1)]
    groups = pool.groupby(["rank_id", "player_id", "player_color"])["game_index"].apply(np.sort)
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_BUCKETS:
        for rank in range(10):
            eligible = [(key, g) for key, g in groups.loc[rank].items() if len(g) >= k]
            for pick in rng.integers(0, len(eligible), PER_RANK):
                (player, color), g = eligible[pick]
                rows.append({"player_id": player, "rank_id": rank, "k": k, "color": color,
                             "games": np.sort(rng.choice(g, k, replace=False)).tolist()})
    return pd.DataFrame(rows).rename_axis("question_id").reset_index()


def predict(model, games, x, questions) -> np.ndarray:
    return official_scores(score_rank(model.probabilities(x, games["game_index"].to_numpy(), questions)),
                           questions["rank_id"].to_numpy())


def bootstrap(a, b, questions) -> list[float]:
    k = questions["k"].to_numpy()
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(b[rows], k[rows]) - equal_k(a[rows], k[rows]))
    return [round(float(q), 4) for q in np.quantile(draws, [0.025, 0.975])]


def hd1_columns(games: pd.DataFrame) -> np.ndarray:
    """P2: per game [HD1 and player black, HD1 and player white]; a question mean is
    the HD1 share on the player's colour (questions are single-colour)."""
    hd1 = (games["handicap"] == 1).to_numpy()
    black = (games["player_color"] == 0).to_numpy()
    return np.stack([hd1 & black, hd1 & ~black], axis=1).astype(float)


def setup():
    games, x = load_games()
    fit_rows = np.flatnonzero(games["role"] == "S")
    eval_rows = np.flatnonzero(games["role"] == "U")
    fit, eval_games = games.iloc[fit_rows].reset_index(drop=True), games.iloc[eval_rows].reset_index(drop=True)
    return games, fit, eval_games, x[fit_rows], x[eval_rows]


def question_sets(games: pd.DataFrame, rebuild: bool) -> dict:
    hd = games.set_index("game_index")["handicap"]
    e1 = pd.read_parquet(Q1000)
    e1 = e1[e1["role"] == "U"].reset_index(drop=True)
    e1_clean = e1[e1["games"].map(lambda g: bool((hd.loc[list(g)] <= 1).all()))].reset_index(drop=True)
    if rebuild:
        e2 = build_e2(games)
        OUT.mkdir(parents=True, exist_ok=True)
        e2.to_parquet(OUT / "questions_e2_hd01.parquet")
    else:
        e2 = pd.read_parquet(OUT / "questions_e2_hd01.parquet")
    return {"E1": e1, "E1_hd01": e1_clean, "E2": e2}


def compare(fit, eval_games, sets, arms: dict, treatment: str, result: dict, out_name: str) -> None:
    """arms: name -> (fit row mask, x_fit, x_eval); 'all' is the reference arm."""
    for seed in SEEDS:
        hs.SEED = seed  # B2 synthetic-question draws; everything else is unchanged
        models = {arm: (RankModel("B2").fit(fit[mask].reset_index(drop=True), pd.DataFrame(), x_fit[mask]), x_eval)
                  for arm, (mask, x_fit, x_eval) in arms.items()}
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {arm: predict(m, eval_games, x_eval, q) for arm, (m, x_eval) in models.items()}
            entry[name] = {arm: by_k(s, k) for arm, s in scores.items()}
            entry[name]["diff_by_k"] = {str(b): round(float(scores[treatment][k == b].mean() - scores["all"][k == b].mean()), 4)
                                       for b in K_BUCKETS}
            entry[name]["diff_equal_k"] = round(equal_k(scores[treatment], k) - equal_k(scores["all"], k), 4)
            if seed == SEED:
                entry[name]["bootstrap_95"] = bootstrap(scores["all"], scores[treatment], q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)

    primary = result["seeds"][str(SEED)]["E2"]
    mean_diff = float(np.mean([result["seeds"][str(s)]["E2"]["diff_equal_k"] for s in SEEDS]))
    result["gate"] = {"set": "E2", "seed": SEED, "diff_equal_k": primary["diff_equal_k"],
                      "min_diff_by_k": min(primary["diff_by_k"].values()), "bootstrap_95": primary["bootstrap_95"],
                      "mean_diff_equal_k_over_seeds": round(mean_diff, 4),
                      "pass": bool(primary["diff_equal_k"] >= MIN_GAIN and min(primary["diff_by_k"].values()) >= -MAX_DROP)}
    print(json.dumps(result["gate"]))
    (OUT / out_name).write_text(json.dumps(result, indent=1))


def p1() -> None:
    games, fit, eval_games, x_fit, x_eval = setup()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    sets = question_sets(games, rebuild=True)
    with PRODUCTION.open("rb") as handle:
        production = pickle.load(handle)["model"]
    check = by_k(predict(production, eval_games, x_eval, sets["E1"]), sets["E1"]["k"].to_numpy())
    print("production model on E1", check, flush=True)
    result = {"fit_games": {"all": int(len(fit)), "hd01": int(hd01.sum())},
              "fit_players": {"all": int(fit["player_id"].nunique()), "hd01": int(fit.loc[hd01, "player_id"].nunique())},
              "questions": {name: {str(b): int((q["k"] == b).sum()) for b in K_BUCKETS} for name, q in sets.items()},
              "production_on_E1": check, "seeds": {}}
    every = np.ones(len(fit), bool)
    compare(fit, eval_games, sets, {"all": (every, x_fit, x_eval), "hd01": (hd01, x_fit, x_eval)}, "hd01",
            result, "result.json")


def p2() -> None:
    """P2: all Valid-S games; AH+C vs AH+C + the two HD1 columns."""
    games, fit, eval_games, x_fit, x_eval = setup()
    sets = question_sets(games, rebuild=False)
    every = np.ones(len(fit), bool)
    arms = {"all": (every, x_fit, x_eval),
            "hd1feat": (every, np.hstack([x_fit, hd1_columns(fit)]), np.hstack([x_eval, hd1_columns(eval_games)]))}
    share = {f"{role}_{c}": round(float(v), 4) for role, frame in (("S", fit), ("U", eval_games))
             for c, v in zip(("hd1_black", "hd1_white"), hd1_columns(frame).mean(axis=0))}
    result = {"fit_games": int(len(fit)), "hd1_share": share,
              "questions": {name: {str(b): int((q["k"] == b).sum()) for b in K_BUCKETS} for name, q in sets.items()},
              "seeds": {}}
    compare(fit, eval_games, sets, arms, "hd1feat", result, "result_p2.json")


def p12() -> None:
    """P1+P2: Valid-S HD<=1 games and the two HD1 columns vs the production setup."""
    games, fit, eval_games, x_fit, x_eval = setup()
    sets = question_sets(games, rebuild=False)
    hd01 = (fit["handicap"] <= 1).to_numpy()
    arms = {"all": (np.ones(len(fit), bool), x_fit, x_eval),
            "hd01_hd1feat": (hd01, np.hstack([x_fit, hd1_columns(fit)]), np.hstack([x_eval, hd1_columns(eval_games)]))}
    result = {"fit_games": {"all": int(len(fit)), "hd01_hd1feat": int(hd01.sum())},
              "questions": {name: {str(b): int((q["k"] == b).sum()) for b in K_BUCKETS} for name, q in sets.items()},
              "seeds": {}}
    compare(fit, eval_games, sets, arms, "hd01_hd1feat", result, "result_p12.json")


def fit_hd01() -> None:
    """LB-probe model: the P1 hd01 arm at seed 2027 (production columns and format)."""
    games, fit, eval_games, x_fit, x_eval = setup()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    with PRODUCTION.open("rb") as handle:
        info = pickle.load(handle)["info"]
    hs.SEED = SEED
    model = RankModel("B2").fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01])
    e2 = pd.read_parquet(OUT / "questions_e2_hd01.parquet")
    score = by_k(predict(model, eval_games, x_eval, e2), e2["k"].to_numpy())
    print("hd01 model on E2", score, flush=True)
    model.save(OUT / "model_hd01.pkl", method="B2", columns=info["columns"], fit_games=int(hd01.sum()),
               fit_players=int(fit.loc[hd01, "player_id"].nunique()), fit_rule="Valid-S games with HD<=1 (P1)",
               a024_run=str(A024), features=str(STANDARD), seed=SEED, e2=score,
               rankvec_vectors=info.get("rankvec_vectors"))
    print(f"saved {OUT / 'model_hd01.pkl'}")


def fit_p12() -> None:
    """Submission model: the P1+P2 arm at seed 2027; columns = production + HD1 flags."""
    from go_aicup.tasks.strength.humansl_predict import HD1_COLUMNS
    games, fit, eval_games, x_fit, x_eval = setup()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    with PRODUCTION.open("rb") as handle:
        info = pickle.load(handle)["info"]
    hs.SEED = SEED
    model = RankModel("B2").fit(fit[hd01].reset_index(drop=True), pd.DataFrame(),
                                np.hstack([x_fit, hd1_columns(fit)])[hd01])
    e2 = pd.read_parquet(OUT / "questions_e2_hd01.parquet")
    score = by_k(predict(model, eval_games, np.hstack([x_eval, hd1_columns(eval_games)]), e2), e2["k"].to_numpy())
    print("P1+P2 model on E2", score, flush=True)
    model.save(OUT / "model_p12.pkl", method="B2", columns=info["columns"] + HD1_COLUMNS, fit_games=int(hd01.sum()),
               fit_players=int(fit.loc[hd01, "player_id"].nunique()),
               fit_rule="Valid-S games with HD<=1, plus HD1 flags by player colour (P1+P2)",
               a024_run=str(A024), features=str(STANDARD), seed=SEED, e2=score,
               rankvec_vectors=info.get("rankvec_vectors"))
    print(f"saved {OUT / 'model_p12.pkl'}")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("p1", "p2", "p12", "fit_hd01", "fit_p12"))
    {"p1": p1, "p2": p2, "p12": p12, "fit_hd01": fit_hd01, "fit_p12": fit_p12}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
