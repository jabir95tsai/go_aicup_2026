"""C vectors v2: relearn the HumanSL rank vectors (C, A-055) without the known flaws (RESULTS 2026-11-11).

The C vectors in use (outputs/rank/humansl/humansl_rankvec/vectors.pt, 2026-10-25) predate the
handicap findings of 11/07:
  1. HD>=2 games were in the learning set: their SGF colour labels are reversed, so own / opponent
     positions are swapped (and the board lacks the handicap stones); e[r, own] and e[r, opp]
     each learned a little of the other side.
  2. HD1 (讓先, no komi) games were encoded at komi 6.5. P3 extracts C of HD1 games at komi 0.5,
     with vectors that never saw komi 0.5.
  3. Only 13,500 learning games (from the train20 draw).
v2 learns the same 10 x 2 x 384 vectors (same model, loss, optimiser) on HD<=1 games only, HD1
games encoded at komi 0.5, from 36,000 games of training-split players that no fitting or
evaluation set uses; then recomputes C for every fitting, evaluation and test game and refits the
current base (Valid-S + train20 + Valid-U, prior-matched B2) with C swapped.

Learning pool: games of players outside Valid-U and fold -1, HD<=1, not a Valid-S / Valid-U game
and not a train20 fitting game. Per rank, 10% of pool players (seed 2027) are monitor players
(MONITOR games per rank); the others give LEARN games per rank, at most PER_PLAYER per player.

    python -m scripts.strength.rank_cvec_v2 tables
    sbatch slurm/rank_cvec_v2_pack.slurm                      # 8 GPUs (CPU-bound encode), ~20 min
    sbatch --dependency=afterok:<pack> slurm/rank_cvec_v2_learn.slurm   # 1 GPU: train + G0
    sbatch slurm/rank_cvec_v2_extract.slurm                   # only if G0 passes; 8 GPUs, ~1.5 h
    python -m scripts.strength.rank_cvec_v2 evaluate          # G1 (CPU)
    python -m scripts.strength.rank_cvec_v2 export --job ID   # LB file <job>-cvec2.csv

Gates (pre-registered, RESULTS 2026-11-11):
  G0 (monitor players, before extraction): with both vector sets on the same monitor positions /
     games (HD1 at komi 0.5), NLL(v2) <= NLL(v1) and single-game score_rank official
     score(v2) >= score(v1) - 0.01. Fail -> stop, no extraction.
  G1 (Valid-U, five B2 seeds, base config: fit Valid-S + train20, Valid-S prior): C v2 minus C v1.
     promote  : five-seed mean Equal-k >= +0.005 on E1 or E2, the other >= 0, and no k bucket of E2
                with five-seed mean < -0.005
     probe    : both means >= 0 but below promote
     stop     : otherwise
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.features import humansl_rankvec as cfeat
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_more_players as more
import scripts.strength.rank_train20_hc as t20

OUT = ROOT / "outputs" / "rank" / "humansl" / "cvec_v2"
LEARN_RUN = OUT / "learn"
OLD_VECTORS = ROOT / "outputs" / "rank" / "humansl" / "humansl_rankvec" / "vectors.pt"
BASE = ROOT / "submissions" / "rank" / "ahc-p3-noA-t20-allp-prior_736589_rank.csv"   # Public 0.7509
LEARN, MONITOR, PER_PLAYER, MONITOR_PLAYERS = 3600, 150, 15, 0.1
C_COLS = slice(30, 50)          # noA x = [H 30 | C 20 | HD1 flags 2]
G0_MAX_DROP = 0.01
G1_PROMOTE, G1_MAX_K_DROP = 0.005, 0.005
# extraction tables: name -> board cache (None = training cache)
TEST_CACHE = {part: p3.WORK / f"rank_{part}" / "cache" for part in ("public", "private")}


def learning_pool() -> pd.DataFrame:
    meta = t20.metadata().merge(pd.read_parquet(hf.CANONICAL, columns=["game_index", "length"]), on="game_index")
    games, _ = hf.load_games()                       # every Valid-S / Valid-U game (role S / U)
    fit_t20 = t20.train20_games()
    u_players = set(games.loc[games["role"] == "U", "player_id"])
    seen = set(games["player_id"]) | set(meta.loc[meta["game_index"].isin(
        pd.read_parquet(t20.TRAIN20)["game_index"]), "player_id"])          # fold -1 players are unseen
    # v1's own learning / monitor games are left out too, so G0 does not favour v1 on them
    used = set(games["game_index"]) | set(fit_t20["game_index"]) | set(pd.read_parquet(t20.LEARNED)["game_index"])
    keep = (meta["player_id"].isin(seen) & ~meta["player_id"].isin(u_players) & (meta["handicap"] <= 1)
            & ~meta["game_index"].isin(used))
    return meta[keep].reset_index(drop=True)


def learning_games() -> pd.DataFrame:
    pool = learning_pool()
    rng = np.random.default_rng(cfeat.SEED)
    picks = []
    for rank, games in pool.groupby("rank_id", sort=True):
        players = rng.permutation(np.sort(games["player_id"].unique()))
        held = set(players[:int(round(MONITOR_PLAYERS * len(players)))])
        monitor = games[games["player_id"].isin(held)]
        learn = games[~games["player_id"].isin(held)]
        learn = learn.iloc[rng.permutation(len(learn))].groupby("player_id").head(PER_PLAYER)
        if len(learn) < LEARN:
            raise ValueError(f"rank {rank}: only {len(learn)} learning games")
        picks.append(learn.iloc[rng.permutation(len(learn))[:LEARN]].assign(role="learn"))
        picks.append(monitor.iloc[rng.permutation(len(monitor))[:MONITOR]].assign(role="monitor"))
    return pd.concat(picks).sort_values("game_index").reset_index(drop=True)


def extraction_games() -> dict[str, pd.DataFrame]:
    """Every game whose C the evaluation / export reads: Valid-S HD<=1, all Valid-U (E1 holds HD>=2
    games; those are encoded at komi 6.5 like before), train20 fitting games, both test parts."""
    games, _ = hf.load_games()
    valid = games[(games["role"] == "U") | (games["handicap"] <= 1)]
    out = {"valid": valid, "t20": t20.train20_games()}
    for part, cache in TEST_CACHE.items():
        table = pd.read_parquet(cache / "games.parquet")[["game_index", "player_color"]]
        hd1 = set(pd.read_parquet(p3.OUT / f"test_{part}_hd1.parquet")["game_index"])
        out[f"test_{part}"] = table.assign(handicap=table["game_index"].isin(hd1).astype(int))
    return out


def tables() -> None:
    learn = learning_games()
    old_players = set(t20.metadata().set_index("game_index").loc[
        pd.read_parquet(t20.LEARNED)["game_index"], "player_id"])
    LEARN_RUN.mkdir(parents=True, exist_ok=True)
    columns = ["game_index", "player_color", "rank_id", "role"]
    report = {"pool_games": int(len(learning_pool())), "learn": {}, "extract": {}}
    for hd in (0, 1):
        table = learn.loc[learn["handicap"] == hd, columns].reset_index(drop=True)
        table.to_parquet(LEARN_RUN / f"learn_hd{hd}.parquet")
        report["learn"][f"hd{hd}"] = len(table)
    report["learn"] |= {
        "by_role": learn.groupby("role").agg(games=("game_index", "size"), players=("player_id", "nunique"),
                                             positions=("length", "sum")).reset_index().to_dict("records"),
        "players_per_rank": learn[learn["role"] == "learn"].groupby("rank_id")["player_id"].nunique().tolist(),
        "monitor_players_in_v1_learning": int(learn.loc[learn["role"] == "monitor", "player_id"]
                                              .drop_duplicates().isin(old_players).sum())}
    for name, table in extraction_games().items():
        for hd in (0, 1):     # HD>=2 (Valid-U only) goes with HD0: komi 6.5, as v1
            part = table[(table["handicap"] == 1) == bool(hd)]
            part[["game_index", "player_color"]].reset_index(drop=True).to_parquet(OUT / f"ext_{name}_hd{hd}.parquet")
            report["extract"][f"{name}_hd{hd}"] = len(part)
    report["extract"]["total"] = int(sum(report["extract"].values()))
    (OUT / "tables.json").write_text(json.dumps(report, indent=1, default=int))
    print(json.dumps(report, indent=1, default=int))


def g0() -> None:
    """Old vs new vectors on the v2 monitor positions and games (GPU)."""
    import torch
    device = torch.device("cuda")
    model, _ = cfeat.load_frozen(device)
    data, _, monitor = cfeat._load_packs(LEARN_RUN, device)
    probe = torch.as_tensor(np.random.default_rng(cfeat.SEED).permutation(monitor)[:50_000], device=device)
    result = {}
    for name, path in (("v1", OLD_VECTORS), ("v2", LEARN_RUN / "vectors.pt")):
        vectors = torch.load(path, map_location=device)["vectors"]
        with torch.no_grad():
            losses = []
            for s in range(0, len(probe), 2048):
                index = probe[s:s + 2048]
                meta = vectors[data["rank"][index], data["side"][index]]
                losses.append(-cfeat.policy_logp(model, cfeat.unpack(data["bits"][index]), data["globals"][index],
                                                 meta, data["target"][index]))
        result[name] = {"monitor_nll": round(float(torch.cat(losses).mean()), 5),
                        **cfeat._monitor_games(LEARN_RUN, model, vectors, data, monitor)}
        print(name, json.dumps(result[name]), flush=True)
    result["pass"] = bool(result["v2"]["monitor_nll"] <= result["v1"]["monitor_nll"]
                          and result["v2"]["score_rank_official"] >= result["v1"]["score_rank_official"] - G0_MAX_DROP)
    print(json.dumps({"G0_pass": result["pass"]}))
    (OUT / "g0.json").write_text(json.dumps(result, indent=1))


def new_c(prefix: str) -> pd.DataFrame:
    """C v2 of one game-index space: 'train' (c_valid_*, c_t20_*: training cache) or 'test_<part>'
    (each test part has its own cache, so its game_index values overlap the others)."""
    names = ("valid", "t20") if prefix == "train" else (prefix,)
    parts = [cfeat.run_features(OUT / f"c_{n}_hd{hd}") for n in names for hd in (0, 1)]
    c = pd.concat(parts, ignore_index=True)
    if c["game_index"].duplicated().any():
        raise ValueError(f"{prefix}: a game has C v2 twice")
    return c.set_index("game_index")


def swap_c(games: pd.DataFrame, x: np.ndarray, c: pd.DataFrame, needed: np.ndarray) -> np.ndarray:
    """x with C replaced by v2; rows not needed (HD>=2 fitting rows, masked out) become NaN."""
    gi = games["game_index"].to_numpy()
    missing = needed & ~np.isin(gi, c.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} needed games without C v2")
    x = x.copy()
    x[:, C_COLS] = np.nan
    have = np.isin(gi, c.index)
    x[have, C_COLS] = c.loc[gi[have]].to_numpy()
    return x


def data():
    games, fit, eval_games, x_fit, x_eval, _ = more.data(with_ex=False)
    c = new_c("train")
    hd01 = (fit["handicap"] <= 1).to_numpy()
    x_fit2 = swap_c(fit, x_fit, c, hd01)
    x_eval2 = swap_c(eval_games, x_eval, c, np.ones(len(eval_games), bool))
    return games, fit, eval_games, {"v1": (x_fit, x_eval), "v2": (x_fit2, x_eval2)}


def evaluate() -> None:
    games, fit, eval_games, xs = data()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    both = np.vstack([xs["v1"][0][hd01], xs["v1"][1]])
    shift = np.vstack([xs["v2"][0][hd01], xs["v2"][1]])[:, C_COLS] - both[:, C_COLS]
    result = {"fit_games": int(hd01.sum()), "c_shift_mean_abs": round(float(np.abs(shift).mean()), 4), "seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        models = {v: more.PriorMatched(prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), xf[hd01])
                  for v, (xf, _) in xs.items()}
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            a = hf.predict(models["v1"], eval_games, xs["v1"][1], q)
            b = hf.predict(models["v2"], eval_games, xs["v2"][1], q)
            entry[name] = {"v1": hf.by_k(a, k), "v2": hf.by_k(b, k),
                           "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4)
                                         for bk in K_BUCKETS},
                           "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
            if seed == hf.SEED:
                entry[name]["bootstrap_95"] = hf.bootstrap(a, b, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    mean = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"] for s in hf.SEEDS])), 4)
            for n in sets}
    e2_k = {str(bk): round(float(np.mean([result["seeds"][str(s)]["E2"]["diff_by_k"][str(bk)] for s in hf.SEEDS])), 4)
            for bk in K_BUCKETS}
    promote = (max(mean.values()) >= G1_PROMOTE and min(mean.values()) >= 0
               and min(e2_k.values()) >= -G1_MAX_K_DROP)
    result["G1"] = {"mean_diff_equal_k": mean, "E2_mean_diff_by_k": e2_k,
                    "decision": "promote" if promote else "probe" if min(mean.values()) >= 0 else "stop"}
    print(json.dumps(result["G1"]))
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


def test_x(part: str, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x with C v1, x with C v2, game_index) of one test part."""
    table, _ = p3.test_table(part, "hc", columns)
    x = table.to_numpy()[:, 10:]
    games = pd.DataFrame({"game_index": table.index.to_numpy()})
    return x, swap_c(games, x, new_c(f"test_{part}"), np.ones(len(table), bool)), table.index.to_numpy()


def export(job: str) -> None:
    """Base (-prior_736589) refitted with C v1 (must reproduce it) and with C v2, seed 2027."""
    games, fit, eval_games, xs = data()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    models = {v: more.PriorMatched(prior).fit(stacked, pd.DataFrame(), np.vstack([xf[hd01], xe[u]]))
              for v, (xf, xe) in xs.items()}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {v: [] for v in models}
    for part in ("public", "private"):
        x1, x2, index = test_x(part, columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for v, x in (("v1", x1), ("v2", x2)):
            p = models[v].probabilities(x, index, questions)
            rows[v].append(pd.DataFrame({"question_id": questions["question_id"],
                                         "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {}
    for v, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = result != base
        report[v] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                     "private": int(changed[changed.index.str.startswith("priv")].sum()),
                     "distribution": result.value_counts().reindex(RANKS).fillna(0).astype(int).tolist()}
        if v == "v1":
            if changed.any():
                raise ValueError("C v1 refit does not reproduce -prior_736589")
            continue
        report[v]["file"] = write_rank_csv(result.reset_index(), "cvec2", job).name
    print(json.dumps({"changed_vs_base": report}))
    (OUT / f"export_{job}.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tables", "g0", "evaluate", "export"])
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    if args.command == "export":
        export(args.job)
    else:
        {"tables": tables, "g0": g0, "evaluate": evaluate}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
