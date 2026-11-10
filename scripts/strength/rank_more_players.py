"""Two follow-ups to noA-t20-allp (Public 0.7405, RESULTS 2026-11-09).

ex     the 798 fold -1 holdout players of the V2 split (never used by any model, fit or
       evaluation): up to 40 random HD<=1 games each, H and C (HD0 komi 6.5, HD1 komi 0.5),
       added to the B2 fit. Same kind of data as allp (+1,442 unseen players, +0.0116 LB).
prior  B2 synthetic questions draw the rank first from a reference prior, then a player of
       that rank uniformly (plain B2 draws players uniformly, so its implied prior is the
       eligible-player share per rank of the fit set). Adding train20 moved that prior
       (fewer 1D, more A); matching the noA-allp prior separates data from prior.

    python -m scripts.strength.rank_more_players tables            # excluded-player game tables
    sbatch slurm/rank_more_players.slurm                  # 8 GPUs, H and C of those games
    python -m scripts.strength.rank_more_players local             # prior arm on E2 / E1, five seeds (CPU)
    python -m scripts.strength.rank_more_players local-ex          # + excluded players, prior / plain, five seeds
    python -m scripts.strength.rank_more_players export --job ID [--no-ex]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.features import humansl as hfeat
from go_aicup.features import humansl_rankvec as cfeat
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_train20_hc as t20

OUT = ROOT / "outputs" / "rank" / "humansl" / "more_players"
BEST = ROOT / "submissions" / "rank" / "ahc-p3-noA-t20-allp_733179_rank.csv"   # Public 0.7405
PER_PLAYER = 40
TAGS = {"noA-t20-allp-prior": "prior", "noA-t20-allp-ex": "ex", "noA-t20-allp-prior-ex": "priorex"}


class PriorMatched(RankModel):
    """B2 whose synthetic draws follow prior[k] (10 rank shares) instead of the player share."""

    def __init__(self, prior: dict):
        super().__init__("B2")
        self.prior = prior

    def _fit_synthetic(self, games, x, y) -> None:
        rng = np.random.default_rng(hs.SEED)
        by_player = list(games.groupby(games["player_id"].astype(str)).indices.values())
        self.per_k = {}
        for k in K_BUCKETS:
            players = [rows for rows in by_player if len(rows) >= k]
            by_rank = [[p for p in players if y[p[0]] == r] for r in range(10)]
            ranks = rng.choice(10, hs.SYNTHETIC, p=self.prior[k])
            features, labels = [], []
            for r in ranks:
                rows = by_rank[r][rng.integers(len(by_rank[r]))]
                features.append(x[rng.choice(rows, k, replace=False)].mean(axis=0))
                labels.append(r)
            features = np.stack(features)
            scaler = StandardScaler().fit(features)
            model = LogisticRegression(C=1.0, max_iter=5000)
            self.per_k[k] = (scaler, model.fit(scaler.transform(features), np.array(labels)))


def player_prior(games: pd.DataFrame) -> dict:
    """Per k, the rank share of players with >= k games (plain B2's implied prior)."""
    n = games.groupby(games["player_id"].astype(str)).agg(games=("game_index", "size"), rank=("rank_id", "first"))
    out = {}
    for k in K_BUCKETS:
        counts = np.bincount(n.loc[n["games"] >= k, "rank"], minlength=10).astype(float)
        out[k] = counts / counts.sum()
    return out


def excluded_games() -> pd.DataFrame:
    """fold -1 players: everyone not in Valid-U, Valid-S or train20; up to PER_PLAYER HD<=1 games."""
    meta = t20.metadata()
    used = np.concatenate([np.load(hf.A024 / n)["game_index"] for n in
                           ("validation_unseen_category.npy", "validation_seen_category.npy")])
    seen = set(meta.loc[meta["game_index"].isin(used), "player_id"])
    seen |= set(meta.loc[meta["game_index"].isin(pd.read_parquet(t20.TRAIN20)["game_index"]), "player_id"])
    games = meta[~meta["player_id"].isin(seen) & (meta["handicap"] <= 1)]
    games = games.sample(frac=1.0, random_state=hf.SEED).groupby("player_id").head(PER_PLAYER)
    return games.sort_values("game_index").reset_index(drop=True).assign(role="X")


def tables() -> None:
    games = excluded_games()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"players": int(games["player_id"].nunique()),
              "players_per_rank": games.groupby("rank_id")["player_id"].nunique().tolist()}
    for hd in (0, 1):
        table = games.loc[games["handicap"] == hd, ["game_index", "player_color", "role"]].reset_index(drop=True)
        table.to_parquet(OUT / f"ex_hd{hd}.parquet")
        report[f"hd{hd}"] = len(table)
    (OUT / "tables.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def excluded_x(games: pd.DataFrame) -> np.ndarray:
    """[H 30 | C 20 | HD1 flags 2]; HD0 at komi 6.5, HD1 at komi 0.5."""
    x = np.full((len(games), 50), np.nan)
    gi, hd = games["game_index"].to_numpy(), games["handicap"].to_numpy()
    for value in (0, 1):
        h = hfeat.run_features(OUT / f"h_hd{value}").set_index("game_index")
        h = h[[c for c in h.columns if c.startswith(("own_", "opp_"))]]
        c = cfeat.run_features(OUT / f"c_hd{value}").set_index("game_index")
        rows = np.flatnonzero(hd == value)
        for source, cols in ((h, slice(0, 30)), (c, slice(30, 50))):
            have = np.isin(gi[rows], source.index)
            x[rows[have], cols] = source.loc[gi[rows[have]]].to_numpy()
    return np.hstack([x, hf.hd1_columns(games)])


def data(with_ex: bool):
    """noA rows: fit = Valid-S + train20 (+ excluded, role X); eval = Valid-U."""
    games, fit, eval_games, x_fit, x_eval, _ = t20.data()
    coverage = {}
    if with_ex:
        ex = excluded_games()
        x_ex = excluded_x(ex)
        complete = np.isfinite(x_ex).all(axis=1)
        coverage = {"excluded_games": int(len(ex)), "complete": int(complete.sum())}
        if complete.mean() < 0.99:
            raise ValueError(f"excluded features incomplete: {coverage}")
        fit = pd.concat([fit, ex.loc[complete, fit.columns]], ignore_index=True)
        x_fit = np.vstack([x_fit, x_ex[complete]])
    return games, fit, eval_games, x_fit, x_eval, coverage


def local() -> None:
    """Valid-S fit vs Valid-S + train20 fit with the Valid-S prior (plain t20 was -0.0102 on E2)."""
    games, fit, eval_games, x_fit, x_eval, _ = data(with_ex=False)
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    prior = player_prior(fit[valid_s])
    result = {"prior_valid_s": {str(k): v.round(4).tolist() for k, v in prior.items()},
              "prior_t20": {str(k): v.round(4).tolist() for k, v in player_prior(fit[hd01]).items()}, "seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        ref = RankModel("B2").fit(fit[valid_s].reset_index(drop=True), pd.DataFrame(), x_fit[valid_s])
        arm = PriorMatched(prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), x_fit[hd01])
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            a, b = hf.predict(ref, eval_games, x_eval, q), hf.predict(arm, eval_games, x_eval, q)
            entry[name] = {"all": hf.by_k(a, k), "t20prior": hf.by_k(b, k),
                           "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4) for bk in K_BUCKETS},
                           "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
            if seed == hf.SEED:
                entry[name]["bootstrap_95"] = hf.bootstrap(a, b, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    result["mean_diff_equal_k"] = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"]
                                                            for s in hf.SEEDS])), 4) for n in sets}
    print(json.dumps(result["mean_diff_equal_k"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_prior.json").write_text(json.dumps(result, indent=1))


def local_ex() -> None:
    """Value of the excluded players on E2 / E1, five seeds: + excluded vs Valid-S + train20,
    once with the Valid-S prior (matches -prior-ex) and once plain (matches -ex)."""
    games, fit, eval_games, x_fit, x_eval, coverage = data(with_ex=True)
    print(json.dumps(coverage), flush=True)
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    base = hd01 & (fit["role"] != "X").to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    prior = player_prior(fit[valid_s])
    result = {"coverage": coverage, "seeds": {}}

    def fitted(model, mask):
        return model.fit(fit[mask].reset_index(drop=True), pd.DataFrame(), x_fit[mask])

    for seed in hf.SEEDS:
        hs.SEED = seed
        arms = {"prior": (fitted(PriorMatched(prior), base), fitted(PriorMatched(prior), hd01)),
                "plain": (fitted(RankModel("B2"), base), fitted(RankModel("B2"), hd01))}
        entry = {}
        for arm, (ref, with_x) in arms.items():
            for name, q in sets.items():
                k = q["k"].to_numpy()
                a, b = hf.predict(ref, eval_games, x_eval, q), hf.predict(with_x, eval_games, x_eval, q)
                cell = {"t20": hf.by_k(a, k), "t20_ex": hf.by_k(b, k),
                        "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4)
                                      for bk in K_BUCKETS},
                        "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
                if seed == hf.SEED:
                    cell["bootstrap_95"] = hf.bootstrap(a, b, q)
                entry[f"{arm}/{name}"] = cell
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    names = next(iter(result["seeds"].values())).keys()
    result["mean_diff_equal_k"] = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"]
                                                            for s in hf.SEEDS])), 4) for n in names}
    print(json.dumps(result["mean_diff_equal_k"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "local_ex.json").write_text(json.dumps(result, indent=1))


def export(job: str, with_ex: bool) -> None:
    """Arms (seed 2027, all on HD<=1 of fit + Valid-U):
    noA-t20-allp            reproduces the current best (check)
    noA-t20-allp-prior      same data, synthetic ranks drawn from the noA-allp prior
    noA-t20-allp-ex         + the 798 excluded players (with_ex)"""
    games, fit, eval_games, x_fit, x_eval, coverage = data(with_ex)
    print(json.dumps(coverage), flush=True)
    u = (eval_games["handicap"] <= 1).to_numpy()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    base = hd01 & (fit["role"] != "X").to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()

    def stack(mask):
        return pd.concat([fit[mask], eval_games[u]], ignore_index=True), np.vstack([x_fit[mask], x_eval[u]])

    hs.SEED = hf.SEED
    allp_prior = player_prior(stack(valid_s)[0])
    models = {"noA-t20-allp": RankModel("B2").fit(stack(base)[0], pd.DataFrame(), stack(base)[1]),
              "noA-t20-allp-prior": PriorMatched(allp_prior).fit(stack(base)[0], pd.DataFrame(), stack(base)[1])}
    if with_ex:
        models["noA-t20-allp-ex"] = RankModel("B2").fit(stack(hd01)[0], pd.DataFrame(), stack(hd01)[1])
        models["noA-t20-allp-prior-ex"] = PriorMatched(allp_prior).fit(stack(hd01)[0], pd.DataFrame(),
                                                                       stack(hd01)[1])
    print("fitted", flush=True)
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {name: [] for name in models}
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for name, model in models.items():
            p = model.probabilities(table.to_numpy()[:, 10:], table.index.to_numpy(), questions)
            rows[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                            "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    reference = pd.read_csv(BEST, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {}
    for name, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = result != reference
        report[name] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                        "private": int(changed[changed.index.str.startswith("priv")].sum()),
                        "distribution": result.value_counts().reindex(RANKS).fillna(0).astype(int).tolist()}
        if name == "noA-t20-allp":
            if changed.any():
                raise ValueError("refit does not reproduce ahc-p3-noA-t20-allp_733179")
            continue
        report[name]["file"] = write_rank_csv(result.reset_index(), TAGS[name], job).name
    report["allp_prior"] = {str(k): v.round(4).tolist() for k, v in allp_prior.items()}
    print(json.dumps({"changed_vs_best": report}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tables", "local", "local-ex", "export"])
    parser.add_argument("--job", default="local")
    parser.add_argument("--no-ex", action="store_true", help="export without the excluded players")
    args = parser.parse_args()
    if args.command == "tables":
        tables()
    elif args.command == "local":
        local()
    elif args.command == "local-ex":
        local_ex()
    else:
        export(args.job, not args.no_ex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
