"""Strong-engine move quality and life-and-death features on the full fit pool (RESULTS 2026-11-15).

A-048 and the 11/11 visits pilot tested the 14 engine features on 18,772 pilot games (760
players) only; every result was negative, but with few fitting players. This run extracts them
for every fitting / evaluation / test game of the current base (the cvec_v2 extraction tables,
1 visit, HD1 games at komi 0.5) and adds 10 life-and-death features from KataGo's ownership:
per side, how many of the mover's stones become lost over its move (dead on the board or
captured), how many lost opponent stones come back to life, and the mover's unsettled stones.

    sbatch slurm/rank_engine_full.slurm                      # 8 GPUs
    python -m scripts.strength.rank_engine_full evaluate     # G1 (CPU)
    python -m scripts.strength.rank_engine_full export --job ID --arm HC_L

Arms (base = cvec2: fit Valid-S + train20 HD<=1, Valid-S prior, prior-matched B2, 52 columns):
  HC      base
  HC_E    + 14 engine move-quality features
  HC_L    + 10 life-and-death features
  HC_EL   + both

G1 per arm (pre-registered; same rule as cvec_v2), arm minus HC on Valid-U, five B2 seeds:
  promote : five-seed mean Equal-k >= +0.005 on E1 or E2, the other >= 0, and no k bucket of E2
            with five-seed mean < -0.005
  probe   : both means >= 0 but below promote
  stop    : otherwise
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
from go_aicup.features.katago_engine import run_features
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_cvec_v2 as cv2
import scripts.strength.rank_more_players as more

OUT = ROOT / "outputs" / "rank" / "humansl" / "engine_full"
BASE = ROOT / "submissions" / "rank" / "751959-cvec2.csv"
ARMS = {"HC": (), "HC_E": ("eng",), "HC_L": ("life",), "HC_EL": ("eng", "life")}
G1_PROMOTE, G1_MAX_K_DROP = 0.005, 0.005


def engine(prefix: str) -> pd.DataFrame:
    """Engine + life features of one game-index space ('train' or 'test_<part>'), cached."""
    path = OUT / f"features_{prefix}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    names = ("valid", "t20") if prefix == "train" else (prefix,)
    parts = []
    for name in names:
        for hd in (0, 1):
            run = OUT / f"e_{name}_hd{hd}"
            shards = [p for p in run.glob("shard_*") if (p / "meta.json").exists()]
            if len(shards) != 8:
                raise ValueError(f"{run}: {len(shards)} of 8 shards complete")
            parts.append(run_features(run))
    table = pd.concat(parts, ignore_index=True)
    if table["game_index"].duplicated().any():
        raise ValueError(f"{prefix}: a game has engine features twice")
    table = table.set_index("game_index")
    table.to_parquet(path)
    return table


def columns(table: pd.DataFrame, groups) -> list[str]:
    return [c for c in table.columns if any(f"_{g}_" in c for g in groups)]


def add(games: pd.DataFrame, x: np.ndarray, table: pd.DataFrame, groups, needed: np.ndarray) -> np.ndarray:
    cols = columns(table, groups)
    if not cols:
        return x
    gi = games["game_index"].to_numpy()
    have = np.isin(gi, table.index)
    if (needed & ~have).any():
        raise ValueError(f"{int((needed & ~have).sum())} needed games without engine features")
    extra = np.full((len(gi), len(cols)), np.nan, dtype=np.float64)
    extra[have] = table.loc[gi[have], cols].to_numpy()
    return np.hstack([x, extra])


def data():
    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    table = engine("train")
    hd01 = (fit["handicap"] <= 1).to_numpy()
    arms = {arm: (add(fit, x_fit, table, groups, hd01),
                  add(eval_games, x_eval, table, groups, np.ones(len(eval_games), bool)))
            for arm, groups in ARMS.items()}
    return games, fit, eval_games, arms, table


def evaluate() -> None:
    games, fit, eval_games, arms, table = data()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit01 = fit[hd01].reset_index(drop=True)
    result = {"fit_games": int(hd01.sum()), "fit_players": int(fit01["player_id"].nunique()),
              "columns": {g: columns(table, (g,)) for g in ("eng", "life")}, "seeds": {}}
    scores = {}
    for seed in hf.SEEDS:
        hs.SEED = seed
        entry = {name: {} for name in sets}
        for arm, (xf, xe) in arms.items():
            model = more.PriorMatched(prior).fit(fit01, pd.DataFrame(), xf[hd01])
            for name, q in sets.items():
                scores[(seed, arm, name)] = hf.predict(model, eval_games, xe, q)
                entry[name][arm] = hf.by_k(scores[(seed, arm, name)], q["k"].to_numpy())
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {a: e[a]["equal_k"] for a in arms} for n, e in entry.items()}), flush=True)
    result["G1"] = {}
    for arm in list(arms)[1:]:
        mean, by_k, boot = {}, {}, {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            mean[name] = round(float(np.mean([hf.equal_k(scores[(s, arm, name)], k) - hf.equal_k(scores[(s, "HC", name)], k)
                                              for s in hf.SEEDS])), 4)
            by_k[name] = {str(b): round(float(np.mean([scores[(s, arm, name)][k == b].mean()
                                                       - scores[(s, "HC", name)][k == b].mean() for s in hf.SEEDS])), 4)
                          for b in K_BUCKETS}
            boot[name] = hf.bootstrap(scores[(hf.SEED, "HC", name)], scores[(hf.SEED, arm, name)], q)
        promote = (max(mean.values()) >= G1_PROMOTE and min(mean.values()) >= 0
                   and min(by_k["E2"].values()) >= -G1_MAX_K_DROP)
        result["G1"][arm] = {"mean_diff_equal_k": mean, "mean_diff_by_k": by_k, "bootstrap_95_seed2027": boot,
                             "decision": "promote" if promote else "probe" if min(mean.values()) >= 0 else "stop"}
        print(arm, json.dumps(result["G1"][arm]), flush=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


def export(job: str, arm: str) -> None:
    """cvec2 base refitted (must reproduce 751959-cvec2.csv) and with the arm's features, seed 2027."""
    games, fit, eval_games, arms, _ = data()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    models = {a: more.PriorMatched(prior).fit(stacked, pd.DataFrame(), np.vstack([arms[a][0][hd01], arms[a][1][u]]))
              for a in ("HC", arm)}
    with hf.PRODUCTION.open("rb") as handle:
        cols = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {a: [] for a in models}
    for part in ("public", "private"):
        _, x, index = cv2.test_x(part, cols)
        test_games = pd.DataFrame({"game_index": index})
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        table = engine(f"test_{part}")
        for a in models:
            xa = add(test_games, x, table, ARMS[a], np.ones(len(index), bool))
            p = models[a].probabilities(xa, index, questions)
            rows[a].append(pd.DataFrame({"question_id": questions["question_id"],
                                         "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {}
    for a, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = result != base
        report[a] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                     "private": int(changed[changed.index.str.startswith("priv")].sum()),
                     "distribution": result.value_counts().reindex(RANKS).fillna(0).astype(int).tolist()}
        if a == "HC":
            if changed.any():
                raise ValueError("HC refit does not reproduce 751959-cvec2.csv")
            continue
        report[a]["file"] = write_rank_csv(result.reset_index(), arm.lower().replace("_", "-"), job).name
    print(json.dumps({"changed_vs_cvec2": report}))
    (OUT / f"export_{job}_{arm}.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["evaluate", "export"])
    parser.add_argument("--job", default="local")
    parser.add_argument("--arm", choices=list(ARMS)[1:], default="HC_L")
    args = parser.parse_args()
    if args.command == "export":
        export(args.job, args.arm)
    else:
        evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
