"""C4g follow-up: does the fine-tuned C keep its gain once H is in the head? (RESULTS 2026-11-15)

C4g G0 compared 20-column C features only: few - base = +0.0166 Equal-k on the dev players.
Production uses [H 30 | C 20 | HD1 flags 2], and earlier full-network fine-tunes lost their gain
once H was added. This repeats G0 on the same dev players / folds / seeds with H joined in.

H source: the fold -1 extraction of rank_more_players (HD0 komi 6.5, HD1 komi 0.5, every move),
plus OUT/h_dev_hd{0,1} for dev games it does not cover (`tables` writes their extraction lists).
C: the dev npz of each arm (50 own + 50 opponent sampled positions per game, same for all arms).

    python -m scripts.strength.rank_c4g_hc audit            # coverage by rank x k, alignment checks
    python -m scripts.strength.rank_c4g_hc tables           # dev games lacking H -> h_dev_hd*.parquet
    python -m scripts.strength.rank_c4g_hc g0 [--partial]   # paired comparison (CPU)

Arms: H, C_base / C_few / C_many (as G0), HC_base / HC_few / HC_many (52 columns).
Gate (pre-registered), HC_few - HC_base on the dev CV:
  Equal-k >= +0.005, player bootstrap (2,000) P(diff > 0) >= 0.90, every k bucket >= -0.005.
Coverage below MIN_COVERAGE of the dev games => "undetermined", not a model failure; --partial
runs the comparison on the covered games anyway and labels it provisional.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_c4g as g
import scripts.strength.rank_more_players as more

OUT = g.OUT
MIN_COVERAGE, MIN_CELL_PLAYERS = 0.99, 5
MIN_GAIN, MIN_PROB, MAX_K_DROP = 0.005, 0.90, 0.005
PAIRS = (("HC_base", "HC_few"), ("HC_base", "HC_many"), ("HC_few", "HC_many"), ("C_base", "C_few"),
         ("H", "HC_base"), ("H", "HC_few"))


def h_features(games: pd.DataFrame) -> np.ndarray:
    """[games, 30] H (own 15 | opp 15); NaN where no extraction covers the game."""
    x = np.full((len(games), 30), np.nan)
    gi, hd = games["game_index"].to_numpy(), games["handicap"].to_numpy()
    for value in (0, 1):
        for run in (more.OUT / f"h_hd{value}", OUT / f"h_dev_hd{value}"):
            if not run.exists():
                continue
            h = more.hfeat.run_features(run).set_index("game_index")
            h = h[[c for c in h.columns if c.startswith(("own_", "opp_"))]]
            if [c.split("_", 1)[0] for c in h.columns] != ["own"] * 15 + ["opp"] * 15:
                raise ValueError(f"{run}: unexpected H column order")
            rows = np.flatnonzero((hd == value) & np.isin(gi, h.index))
            x[rows] = h.loc[gi[rows]].to_numpy()
    return x


def cell_players(games: pd.DataFrame) -> pd.DataFrame:
    n = games.groupby(games["player_id"].astype(str)).agg(n=("game_index", "size"), rank=("rank_id", "first"))
    return pd.DataFrame({k: np.bincount(n.loc[n["n"] >= k, "rank"], minlength=10) for k in K_BUCKETS}).T


def load():
    games = pd.read_parquet(OUT / "dev_games.parquet")
    h = h_features(games)
    return games, h, np.isfinite(h).all(axis=1)


def audit() -> dict:
    games, h, have = load()
    meta = g._meta().set_index("game_index").loc[games["game_index"]]
    checks = {"player": bool((meta["player_id"].to_numpy() == games["player_id"].to_numpy()).all()),
              "color": bool((meta["player_color"].to_numpy() == games["player_color"].to_numpy()).all()),
              "rank": bool((meta["rank_id"].to_numpy() == games["rank_id"].to_numpy()).all()),
              "handicap": bool((meta["handicap"].to_numpy() == games["handicap"].to_numpy()).all()),
              "one_color_per_player": bool((games.groupby("player_id")["player_color"].nunique() == 1).all())}
    z = np.load(OUT / "few" / "A" / "dev_step0.npz")
    checks["c_denominator_50_per_side"] = bool((z["moves"] == g.DEV_POSITIONS).all())
    checks["c_arms_same_games"] = all(
        np.array_equal(np.load(OUT / a / "A" / f)["game_index"], z["game_index"])
        for a, f in (("few", "dev_final.npz"), ("many", "dev_final.npz")))
    checks["no_duplicate_games"] = bool(games["game_index"].is_unique)
    checks["h_all_finite"] = bool(np.isfinite(h).all())
    sources, seen = {}, []
    for value, komi in ((0, 6.5), (1, 0.5)):
        for run in (more.OUT / f"h_hd{value}", OUT / f"h_dev_hd{value}"):
            if not run.exists():
                continue
            logs = [json.loads(path.read_text().strip().splitlines()[-1]) for path in sorted(run.glob("shard_*.log"))]
            index = more.hfeat.run_features(run)["game_index"]
            seen.append(index[index.isin(games.loc[games["handicap"] == value, "game_index"])])
            sources[str(run.relative_to(OUT.parent))] = {
                "komi": sorted({log["komi"] for log in logs}), "model": sorted({log["model"] for log in logs}),
                "games": int(sum(log["games_complete"] for log in logs)),
                "complete": all(log["games"] == log["games_complete"] and log["katago_returncode"] == 0
                                and log["rejected_queries"] == 0 for log in logs),
                "profiles_same": all(log["profiles"] == logs[0]["profiles"] for log in logs),
                "komi_ok": all(log["komi"] == komi for log in logs)}
            sources[str(run.relative_to(OUT.parent))]["profiles"] = logs[0]["profiles"]
    checks["h_single_source_per_game"] = bool(pd.concat(seen).is_unique)
    checks["h_sources_complete_and_komi"] = all(v["complete"] and v["komi_ok"] and v["profiles_same"] for v in sources.values())
    checks["h_one_model_and_profile_order"] = (len({tuple(v["model"]) for v in sources.values()}) == 1
                                                and len({tuple(v["profiles"]) for v in sources.values()}) == 1)
    before, after = cell_players(games), cell_players(games[have])
    report = {"dev_games": int(len(games)), "with_h": int(have.sum()), "coverage": round(float(have.mean()), 4),
              "by_handicap": {str(hd): [int(have[games["handicap"] == hd].sum()), int((games["handicap"] == hd).sum())]
                              for hd in (0, 1)},
              "players": int(games["player_id"].nunique()),
              "players_with_any_h": int(games.loc[have, "player_id"].nunique()),
              "checks": checks, "h_sources": sources,
              "players_rank_by_k_before": {str(k): before.loc[k].tolist() for k in K_BUCKETS},
              "players_rank_by_k_after": {str(k): after.loc[k].tolist() for k in K_BUCKETS}}
    (OUT / "hc_audit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report


def tables() -> None:
    games, _, have = load()
    missing = games[~have]
    for hd in (0, 1):
        table = missing.loc[missing["handicap"] == hd, ["game_index", "player_color"]].assign(role="dev")
        table.reset_index(drop=True).to_parquet(OUT / f"h_dev_hd{hd}.parquet")
        print(f"hd{hd}: {len(table)} games to extract")


def g0(partial: bool) -> None:
    games, h, have = load()
    coverage = float(have.mean())
    if coverage < MIN_COVERAGE and not partial:
        raise SystemExit(f"undetermined: H covers {coverage:.1%} of the dev games (< {MIN_COVERAGE:.0%}); "
                         "extract the rest (tables) or pass --partial")
    c = {"base": g._features(OUT / "few" / "A" / "dev_step0.npz", games)[0],
         "few": g._features(OUT / "few" / "A" / "dev_final.npz", games)[0],
         "many": g._features(OUT / "many" / "A" / "dev_final.npz", games)[0]}
    flags = hf.hd1_columns(games)
    games, h, flags = games[have].reset_index(drop=True), h[have], flags[have]
    xs = {"H": np.hstack([h, flags])}
    for arm, x in c.items():
        xs[f"C_{arm}"] = x[have]
        xs[f"HC_{arm}"] = np.hstack([h, x[have], flags])
    cells = cell_players(games)
    thin = {str(k): int(cells.loc[k].min()) for k in K_BUCKETS if cells.loc[k].min() < MIN_CELL_PLAYERS}
    if thin:
        raise SystemExit(f"undetermined: rank x k cells with < {MIN_CELL_PLAYERS} covered players (min per k): {thin}")
    result = {"provisional": coverage < MIN_COVERAGE, "coverage": round(coverage, 4), "games": int(len(games)),
              "players": int(games["player_id"].nunique()), "thin_cells_min_players": thin, "arms": {}}
    fold = g._fold_of(pd.read_parquet(OUT / "dev_games.parquet"))          # same player folds as G0
    game_fold = fold.loc[games["player_id"].astype(str)].to_numpy()
    questions = g._questions(games, np.random.default_rng(g.SEED + 1))
    q_fold = fold.loc[questions["player_id"]].to_numpy()
    k, truth = questions["k"].to_numpy(), questions["rank_id"].to_numpy()
    gi = games["game_index"].to_numpy()
    scores = {arm: np.zeros(len(questions)) for arm in xs}
    hs.SEED = g.SEED
    for f in range(g.FOLDS):
        train, test = game_fold != f, q_fold == f
        prior = more.player_prior(games[train])
        for arm, x in xs.items():
            model = more.PriorMatched(prior).fit(games[train].reset_index(drop=True), pd.DataFrame(), x[train])
            scores[arm][test] = official_scores(score_rank(model.probabilities(x, gi, questions[test])), truth[test])
        print(f"fold {f} done", flush=True)
    for arm in xs:
        result["arms"][arm] = hf.by_k(scores[arm], k)
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].to_numpy()).apply(np.asarray).to_list()
    rng = np.random.default_rng(g.SEED + 2)
    draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
             for _ in range(g.BOOTSTRAP)]
    result["diffs"] = {}
    for a, b in PAIRS:
        boot = np.array([hf.equal_k(scores[b][rows], k[rows]) - hf.equal_k(scores[a][rows], k[rows]) for rows in draws])
        result["diffs"][f"{b}-{a}"] = {
            "equal_k": round(hf.equal_k(scores[b], k) - hf.equal_k(scores[a], k), 4),
            "by_k": {str(bk): round(float((scores[b] - scores[a])[k == bk].mean()), 4) for bk in K_BUCKETS},
            "bootstrap_95": [round(float(v), 4) for v in np.quantile(boot, [0.025, 0.975])],
            "prob_positive": round(float((boot > 0).mean()), 3)}
    main_diff = result["diffs"]["HC_few-HC_base"]
    passed = bool(main_diff["equal_k"] >= MIN_GAIN and main_diff["prob_positive"] >= MIN_PROB
                  and min(main_diff["by_k"].values()) >= -MAX_K_DROP)
    result["verdict"] = "undetermined" if result["provisional"] else ("pass" if passed else "fail")
    result["gate_numbers_met"] = passed
    name = "g0_hc_partial.json" if result["provisional"] else "g0_hc.json"
    (OUT / name).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "tables", "g0"])
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    {"audit": audit, "tables": tables, "g0": lambda: g0(args.partial)}[args.command]()


if __name__ == "__main__":
    main()
