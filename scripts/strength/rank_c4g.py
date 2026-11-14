"""C4g: does game coverage matter for a generative (pure NLL) HumanSL fine-tune? (RESULTS 2026-11-14)

Three arms, same model / optimiser / steps / position budget as C4x adj without its rank CE:
  base  HumanSL + C v2 vectors, not fine-tuned (step 0 of any run)
  few   pure NLL on the C4x learning set: <= 20 games per player, every position (learn_<half>)
  many  pure NLL on the SAME number of positions drawn from every HD<=1 game of the same players
What is held equal between few and many (per half): the players; each player's number of own-side
positions; the number of opponent-side positions per rank; hence positions per rank x side, the rank
weights, the step count. What changes: the positions come from ~6x as many games (few per game).
The number of distinct opponents is unknown (the data has no opponent id).

Stage 1 (this file): half A only. Dev set = fold -1 players (in no half, no B2 fitting or evaluation
set), one colour per player, <= DEV_GAMES games, DEV_POSITIONS own + opponent positions per game.
The fine-tune job saves per-game sums of log p under the 10 rank hypotheses (dev_step0 / dev_final).
g0: 20 columns per game (own / opp mean log p), 5-fold player-grouped CV inside the dev players with
the production head (prior-matched B2 on synthetic questions), rank-balanced single-colour questions,
official score by k, player bootstrap of the paired differences.

    python -m scripts.strength.rank_c4g tables
    sbatch slurm/rank_c4g_pack.slurm
    ARM=few sbatch slurm/rank_c4g_train.slurm ; ARM=many sbatch --dependency=afterok:<pack> slurm/rank_c4g_train.slurm
    python -m scripts.strength.rank_c4g g0

Gate (pre-registered; all on the dev CV, Equal-k of the official score):
  many - few  >= +0.005 with bootstrap P(diff > 0) >= 0.90, and
  many - base >= +0.005 with bootstrap P(diff > 0) >= 0.90, no k bucket < -0.005, and
  dev NLL(many) <= dev NLL(base).
Otherwise stop: no half B, no extraction.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_c4x as c4
import scripts.strength.rank_more_players as more
import scripts.strength.rank_train20_hc as t20

OUT = ROOT / "outputs" / "rank" / "humansl" / "c4g"
SEED = 2027
DEV_GAMES, DEV_MIN_GAMES, DEV_POSITIONS = 20, 5, 50      # per player / per player / per side per game
FOLDS, PER_CELL, BOOTSTRAP = 5, 200, 2000
MIN_GAIN, MIN_PROB, MAX_K_DROP = 0.005, 0.90, 0.005
COLUMNS = ["game_index", "player_color", "rank_id", "role"]


def _meta() -> pd.DataFrame:
    meta = t20.metadata().merge(pd.read_parquet(hf.CANONICAL, columns=["game_index", "length"]), on="game_index")
    return meta[meta["handicap"] <= 1].reset_index(drop=True)


def _side_counts(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(own, opp) position counts per game: black moves on even plies (A-002), HD<=1 only."""
    length, color = games["length"].to_numpy(), games["player_color"].to_numpy()
    black = (length + 1) // 2
    own = np.where(color == 0, black, length - black)
    return own, length - own


def _draw(counts: np.ndarray, budget: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """budget distinct (game row, offset) pairs, uniform over all positions of the given per-game counts."""
    ends = np.cumsum(counts)
    picks = np.sort(rng.choice(int(ends[-1]), budget, replace=False))
    game = np.searchsorted(ends, picks, side="right")
    return game, picks - (ends[game] - counts[game])


def many_games(meta: pd.DataFrame, few: pd.DataFrame, half_players: np.ndarray, rng) -> pd.DataFrame:
    """Every HD<=1 game of the half's players with the plies drawn for the many arm."""
    pool = meta[meta["player_id"].isin(half_players)].reset_index(drop=True)
    own, opp = _side_counts(pool)
    few_own, few_opp = _side_counts(few)
    color = pool["player_color"].to_numpy()
    plies = [[] for _ in range(len(pool))]
    budget = pd.Series(few_own).groupby(few["player_id"].to_numpy()).sum()
    for player, rows in pool.groupby("player_id").indices.items():       # own side: the player's few-arm count
        game, offset = _draw(own[rows], int(budget[player]), rng)
        for g, o in zip(rows[game], offset):
            plies[g].append(int(color[g] + 2 * o))
    budget = pd.Series(few_opp).groupby(few["rank_id"].to_numpy()).sum()
    for rank, rows in pool.groupby("rank_id").indices.items():           # opponent side: the rank's few-arm count
        game, offset = _draw(opp[rows], int(budget[rank]), rng)
        for g, o in zip(rows[game], offset):
            plies[g].append(int(1 - color[g] + 2 * o))
    pool = pool.assign(plies=[sorted(p) for p in plies], role="learn")
    return pool[pool["plies"].map(len) > 0].reset_index(drop=True)


def dev_games(meta: pd.DataFrame, rng) -> pd.DataFrame:
    folds = pd.read_parquet(c4.FOLDS)
    pool = meta[meta["player_id"].isin(set(folds.loc[folds["fold"] == -1, "player_id"]))]
    picks = []
    for _, group in pool.groupby("player_id", sort=True):
        size = group.groupby("player_color").size()
        group = group[group["player_color"] == size.idxmax()]            # one colour per player (ties: black)
        if len(group) >= DEV_MIN_GAMES:
            picks.append(group.iloc[rng.permutation(len(group))[:DEV_GAMES]])
    games = pd.concat(picks).sort_values("game_index").reset_index(drop=True)
    own, opp = _side_counts(games)
    color = games["player_color"].to_numpy()
    plies = []
    for g in range(len(games)):
        a = color[g] + 2 * np.sort(rng.choice(own[g], min(DEV_POSITIONS, own[g]), replace=False))
        b = 1 - color[g] + 2 * np.sort(rng.choice(opp[g], min(DEV_POSITIONS, opp[g]), replace=False))
        plies.append(sorted(int(p) for p in np.concatenate([a, b])))
    return games.assign(plies=plies, role="dev")


def tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _meta()
    halves = pd.read_parquet(c4.OUT / "halves.parquet")
    rng = np.random.default_rng(SEED)
    report = {"learn": {}}
    for half in c4.HALVES:
        few_index = pd.concat([pd.read_parquet(c4.OUT / f"learn_{half}_hd{hd}.parquet") for hd in (0, 1)])["game_index"]
        few = meta[meta["game_index"].isin(few_index)].reset_index(drop=True)
        if len(few) != len(few_index):
            raise ValueError(f"half {half}: few-arm games missing from the metadata")
        many = many_games(meta, few, halves.loc[halves["half"] == half, "player_id"].to_numpy(), rng)
        cells = {}
        for name, games, n in (("few", few, None), ("many", many, many["plies"].map(len).to_numpy())):
            own, opp = _side_counts(games)
            if n is not None:     # drawn plies: own = plies of the player's colour
                own = np.array([sum(p % 2 == c for p in pl) for pl, c in zip(games["plies"], games["player_color"])])
                opp = n - own
            w = pd.Series(own).groupby(games["player_id"].to_numpy()).sum().astype(float)
            cells[name] = {"games": int(len(games)), "players": int(games["player_id"].nunique()),
                           "positions": int(own.sum() + opp.sum()), "own": int(own.sum()), "opp": int(opp.sum()),
                           "own_by_rank": pd.Series(own).groupby(games["rank_id"].to_numpy()).sum().tolist(),
                           "opp_by_rank": pd.Series(opp).groupby(games["rank_id"].to_numpy()).sum().tolist(),
                           "games_by_rank": games.groupby("rank_id").size().tolist(),
                           "effective_players_own": round(float(w.sum() ** 2 / (w ** 2).sum())),
                           "positions_per_game_quantiles": np.quantile(own + opp, [0.05, 0.5, 0.95]).round(1).tolist(),
                           "own_per_player": w.sort_index().to_numpy()}
        if not np.array_equal(cells["few"].pop("own_per_player"), cells["many"].pop("own_per_player")):
            raise ValueError("own-side positions per player differ between the arms")
        for key in ("own_by_rank", "opp_by_rank", "players"):
            if cells["few"][key] != cells["many"][key]:
                raise ValueError(f"{key} differs between the arms")
        cells["distinct_opponents"] = "unknown (no opponent id in the data)"
        report["learn"][half] = cells
        (OUT / f"learn_{half}").mkdir(exist_ok=True)
        for hd in (0, 1):
            many.loc[many["handicap"] == hd, COLUMNS + ["plies"]].reset_index(drop=True) \
                .to_parquet(OUT / f"learn_{half}_hd{hd}.parquet")
    dev = dev_games(meta, rng)
    if set(dev["player_id"]) & set(halves["player_id"]):
        raise ValueError("dev players in a half")
    dev[["game_index", "player_id", "player_color", "rank_id", "handicap"]].to_parquet(OUT / "dev_games.parquet")
    (OUT / "dev").mkdir(exist_ok=True)
    for hd in (0, 1):
        dev.loc[dev["handicap"] == hd, COLUMNS + ["plies"]].reset_index(drop=True).to_parquet(OUT / f"dev_hd{hd}.parquet")
    per_player = dev.groupby("player_id").size()
    report["dev"] = {"games": int(len(dev)), "players": int(dev["player_id"].nunique()),
                     "positions": int(dev["plies"].map(len).sum()),
                     "players_per_rank": dev.groupby("rank_id")["player_id"].nunique().tolist(),
                     "players_with_k_games": {str(k): int((per_player >= k).sum()) for k in K_BUCKETS}}
    (OUT / "tables.json").write_text(json.dumps(report, indent=1, default=int))
    print(json.dumps(report, indent=1, default=int))


# ------------------------------------------------------------------ gate
def _features(path, games: pd.DataFrame) -> tuple[np.ndarray, float]:
    """[games, 20] own / opp mean log p aligned with games, and the dev NLL under the true rank."""
    z = np.load(path)
    order = pd.Series(np.arange(len(z["game_index"])), index=z["game_index"]).loc[games["game_index"]].to_numpy()
    sides, moves, truth = z["sides"][order], z["moves"][order], z["truth"][order]
    if not np.array_equal(truth, games["rank_id"].to_numpy()):
        raise ValueError(f"{path}: rank mismatch")
    x = (sides / np.maximum(moves[:, :, None], 1)).reshape(len(games), 20)
    nll = -sides[np.arange(len(games)), :, truth].sum() / moves.sum()
    return x, float(nll)


def _questions(games: pd.DataFrame, rng) -> pd.DataFrame:
    groups = games.groupby(["rank_id", games["player_id"].astype(str)])["game_index"].apply(np.sort)
    rows = []
    for k in K_BUCKETS:
        for rank in range(10):
            eligible = [(key, g) for key, g in groups.loc[rank].items() if len(g) >= k]
            for pick in rng.integers(0, len(eligible), PER_CELL):
                player, g = eligible[pick]
                rows.append({"player_id": player, "rank_id": rank, "k": k,
                             "games": np.sort(rng.choice(g, k, replace=False)).tolist()})
    return pd.DataFrame(rows)


def _fold_of(games: pd.DataFrame) -> pd.Series:
    players = games.groupby(games["player_id"].astype(str))["rank_id"].first().sort_index()
    rng = np.random.default_rng(SEED)
    fold = pd.Series(0, index=players.index)
    for _, ids in players.groupby(players):
        fold.loc[ids.index[rng.permutation(len(ids))]] = np.arange(len(ids)) % FOLDS
    return fold


def _mid_dprime(x: np.ndarray, truth: np.ndarray) -> float:
    per_move = (x[:, :10] + x[:, 10:]) / 2
    out = []
    for r in (4, 5, 6):
        s = per_move[:, r + 1] - per_move[:, r]
        a, b = s[truth == r], s[truth == r + 1]
        out.append((b.mean() - a.mean()) / np.sqrt((a.var() + b.var()) / 2))
    return round(float(np.mean(out)), 4)


def g0(half: str) -> None:
    games = pd.read_parquet(OUT / "dev_games.parquet")
    paths = {"base": OUT / "few" / half / "dev_step0.npz", "few": OUT / "few" / half / "dev_final.npz",
             "many": OUT / "many" / half / "dev_final.npz"}
    xs, result = {}, {"half": half, "dev_games": int(len(games)), "dev_players": int(games["player_id"].nunique()), "arms": {}}
    for arm, path in paths.items():
        xs[arm], nll = _features(path, games)
        result["arms"][arm] = {"dev_nll": round(nll, 5)}
    fold = _fold_of(games)
    game_fold = fold.loc[games["player_id"].astype(str)].to_numpy()
    questions = _questions(games, np.random.default_rng(SEED + 1))
    q_fold = fold.loc[questions["player_id"]].to_numpy()
    k, truth = questions["k"].to_numpy(), questions["rank_id"].to_numpy()
    gi = games["game_index"].to_numpy()
    scores = {arm: np.zeros(len(questions)) for arm in xs}
    hs.SEED = SEED
    for f in range(FOLDS):
        train, test = game_fold != f, q_fold == f
        prior = more.player_prior(games[train])
        for arm, x in xs.items():
            model = more.PriorMatched(prior).fit(games[train].reset_index(drop=True), pd.DataFrame(), x[train])
            scores[arm][test] = official_scores(score_rank(model.probabilities(x, gi, questions[test])), truth[test])
    player_truth = games.groupby(games["player_id"].astype(str))["rank_id"].first()
    for arm, x in xs.items():
        pooled = pd.DataFrame(x).groupby(games["player_id"].astype(str).to_numpy()).mean()
        result["arms"][arm] |= hf.by_k(scores[arm], k) | {
            "game_mid_d": _mid_dprime(x, games["rank_id"].to_numpy()),
            "player_mid_d": _mid_dprime(pooled.to_numpy(), player_truth.loc[pooled.index].to_numpy())}
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].to_numpy()).apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED + 2)
    draws = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(BOOTSTRAP)]
    result["diffs"] = {}
    for a, b in (("few", "many"), ("base", "many"), ("base", "few")):
        boot = np.array([hf.equal_k(scores[b][rows], k[rows]) - hf.equal_k(scores[a][rows], k[rows]) for rows in draws])
        result["diffs"][f"{b}-{a}"] = {
            "equal_k": round(hf.equal_k(scores[b], k) - hf.equal_k(scores[a], k), 4),
            "by_k": {str(bk): round(float((scores[b] - scores[a])[k == bk].mean()), 4) for bk in K_BUCKETS},
            "bootstrap_95": [round(float(v), 4) for v in np.quantile(boot, [0.025, 0.975])],
            "prob_positive": round(float((boot > 0).mean()), 3)}
    mf, mb = result["diffs"]["many-few"], result["diffs"]["many-base"]
    result["pass"] = bool(mf["equal_k"] >= MIN_GAIN and mf["prob_positive"] >= MIN_PROB
                          and mb["equal_k"] >= MIN_GAIN and mb["prob_positive"] >= MIN_PROB
                          and min(mb["by_k"].values()) >= -MAX_K_DROP
                          and result["arms"]["many"]["dev_nll"] <= result["arms"]["base"]["dev_nll"])
    (OUT / f"g0_{half}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tables", "g0"])
    parser.add_argument("--half", default="A", choices=c4.HALVES)
    args = parser.parse_args()
    tables() if args.command == "tables" else g0(args.half)


if __name__ == "__main__":
    main()
