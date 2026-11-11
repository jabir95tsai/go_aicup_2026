"""C4x: HumanSL fine-tuning retried with player-balanced data, cross-fitted (RESULTS 2026-11-11).

The 10/26 C4 / C4b runs (A-057) put every game of 2,880 F players in the pool with a per-position
loss, so a player weighed by game count: ~960 effective players (~60 for 1D), and C4b learned the
training players. C4x changes the data, not the model / loss / optimiser:
  * players: every fold 1-4 player (5,767; all Valid-S / train20 fitting players) is split per
    rank (seed 2027) into halves A and B; model M_A is fine-tuned on half A, M_B on half B
  * at most PER_PLAYER HD<=1 games per player, HD1 encoded at komi 0.5; positions weighted so
    every rank weighs the same (--rank-balance)
  * monitor: fold -1 players (in no fitting / evaluation set and no half), MONITOR games per rank,
    at most MONITOR_PER_PLAYER per player
  * cross-fitting: a fitting game (Valid-S, train20) gets C4 from the model that never saw its
    player; Valid-U and test games get C4 from both models ("views"), and their probabilities are
    the mean of the B2 probabilities of the two views. In the final export, each Valid-U player
    (used as fitting data there) is assigned one view at random (seed 2027).
Arms (separate fine-tunes; each gated by G0 before extraction):
  mmi : NLL + 1.0 x rank CE over all 10 hypotheses, 1 epoch          (C4b objective; reference)
  adj : NLL + 1.0 x rank CE over {r-1, r, r+1} only, 1 epoch          (aims at neighbouring ranks)
  gen : generative NLL, 2 epochs (C4 objective; kept in code, not in the 11/11 plan: C4 showed the
        generative objective does not separate ranks)
B2 arms: base (current H + C), swap (C replaced by C4, primary), add (H + C + C4, diagnostic).
Mid-range diagnostics (goal: resolution of 1D/2D, 2D/3D, 3D/4D): the monitor logs the single-game
d' of the per-move log-likelihood ratio (own / opponent / all moves); evaluate reports, on Valid-U
questions of k = 1 / 5 / 20, the d' of an LDA fitted on fitting-pool questions (own / opp / all),
for base and swap.

    python -m scripts.strength.rank_c4x tables
    sbatch slurm/rank_c4x_pack.slurm                                    # 8 GPUs (CPU encode)
    ARM=mmi HALF=A sbatch slurm/rank_c4x_train.slurm                    # x {mmi, adj} x {A, B}
    python -m scripts.strength.rank_c4x g0 --arm mmi                     # CPU, reads the logs
    ARM=mmi sbatch slurm/rank_c4x_extract.slurm                          # only if G0 passes
    python -m scripts.strength.rank_c4x evaluate --arm mmi               # G1 (CPU)
    python -m scripts.strength.rank_c4x export --arm mmi --job ID        # LB file <job>-c4x-mmi.csv

Gates (pre-registered, RESULTS 2026-11-11):
  G0 (fold -1 monitor, per arm, before extraction): no harm (each model's final single-game
     official score >= its step 0) and at least one of: official gain (final - step 0, mean over
     M_A and M_B) >= +0.01, or mid-range d' gain (all moves, mean of the three pairs, mean over
     the two models) >= +0.05. Fail -> no extraction for that arm.
  G1 (Valid-U, five B2 seeds, base config: fit Valid-S + train20, Valid-S prior): swap minus base.
     promote : five-seed mean Equal-k >= +0.005 on E1 or E2, the other >= 0, and no E2 k bucket
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
from go_aicup.features import humansl_rankvec as cfeat
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.submission_files import write_rank_csv
from go_aicup.tasks.strength.v2.aggregate import official_scores

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.humansl_p3_komi as p3
import scripts.strength.rank_cvec_v2 as cv2
import scripts.strength.rank_more_players as more
import scripts.strength.rank_train20_hc as t20

OUT = ROOT / "outputs" / "rank" / "humansl" / "c4x"
FOLDS = ROOT / "data" / "processed" / "cv" / "strength_folds_seed2027.parquet"
SEED = 2027
HALVES = ("A", "B")
PER_PLAYER = 20
MONITOR, MONITOR_PER_PLAYER = 300, 8
C_COLS = cv2.C_COLS                 # noA x = [H 30 | C 20 | HD1 flags 2]
G0_MIN_GAIN, G0_MIN_MID_GAIN = 0.01, 0.05
G1_PROMOTE, G1_MAX_K_DROP = 0.005, 0.005
PRIMARY = "swap"


def other(half: str) -> str:
    return "B" if half == "A" else "A"


# ------------------------------------------------------------------ tables
def player_halves() -> pd.DataFrame:
    """fold 1-4 players split per rank into halves A / B (seed 2027)."""
    folds = pd.read_parquet(FOLDS)
    players = folds[folds["fold"].isin([1, 2, 3, 4])]
    rng = np.random.default_rng(SEED)
    rows = []
    for rank, group in players.groupby("rank_id", sort=True):
        ids = rng.permutation(np.sort(group["player_id"].to_numpy()))
        cut = len(ids) // 2
        rows.append(pd.DataFrame({"player_id": ids, "rank_id": rank, "half": ["A"] * cut + ["B"] * (len(ids) - cut)}))
    return pd.concat(rows, ignore_index=True)


def learning_games(halves: pd.DataFrame) -> pd.DataFrame:
    meta = t20.metadata().merge(pd.read_parquet(hf.CANONICAL, columns=["game_index", "length"]), on="game_index")
    games = meta[(meta["handicap"] <= 1)].merge(halves[["player_id", "half"]], on="player_id")
    games = games.sample(frac=1.0, random_state=SEED).groupby("player_id").head(PER_PLAYER)
    return games.sort_values("game_index").reset_index(drop=True).assign(role="learn")


def monitor_games() -> pd.DataFrame:
    folds = pd.read_parquet(FOLDS)
    unused = set(folds.loc[folds["fold"] == -1, "player_id"])
    meta = t20.metadata().merge(pd.read_parquet(hf.CANONICAL, columns=["game_index", "length"]), on="game_index")
    pool = meta[meta["player_id"].isin(unused) & (meta["handicap"] <= 1)]
    pool = pool.sample(frac=1.0, random_state=SEED).groupby("player_id").head(MONITOR_PER_PLAYER)
    picks = []
    for rank, group in pool.groupby("rank_id", sort=True):
        if len(group) < MONITOR:
            raise ValueError(f"rank {rank}: only {len(group)} monitor games")
        picks.append(group.iloc[:MONITOR])
    return pd.concat(picks).sort_values("game_index").reset_index(drop=True).assign(role="monitor")


def extraction_tables(halves: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """(model half, table name) -> games that model extracts. A fitting game (Valid-S / train20) goes
    to the model of the other half; Valid-U and test games go to both."""
    half_of = halves.set_index("player_id")["half"]
    out = {}
    for name, table in cv2.extraction_games().items():
        if name.startswith("test_"):
            for model in HALVES:
                out[(model, name)] = table
            continue
        fitting = table["role"] != "U" if "role" in table else np.ones(len(table), bool)
        owner = table["player_id"].map(half_of)
        if owner[fitting].isna().any():
            raise ValueError(f"{name}: {int(owner[fitting].isna().sum())} fitting games of players in no half")
        for model in HALVES:
            out[(model, name)] = table[~fitting | (owner == other(model))]
    return out


def tables() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    halves = player_halves()
    halves.to_parquet(OUT / "halves.parquet")
    learn, monitor = learning_games(halves), monitor_games()
    if set(monitor["player_id"]) & set(halves["player_id"]):
        raise ValueError("monitor players in a half")
    columns = ["game_index", "player_color", "rank_id", "role"]
    report = {"halves": halves.groupby(["rank_id", "half"]).size().unstack().to_dict("list"), "learn": {},
              "monitor": {"games": len(monitor), "players": int(monitor["player_id"].nunique()),
                          "players_per_rank": monitor.groupby("rank_id")["player_id"].nunique().tolist(),
                          "positions": int(monitor["length"].sum())}, "extract": {}}
    for half in HALVES:
        part = learn[learn["half"] == half]
        (OUT / f"learn_{half}").mkdir(exist_ok=True)
        for hd in (0, 1):
            part.loc[part["handicap"] == hd, columns].reset_index(drop=True).to_parquet(OUT / f"learn_{half}_hd{hd}.parquet")
        w = part.groupby("player_id").size().astype(float)
        report["learn"][half] = {"games": len(part), "players": int(part["player_id"].nunique()),
                                 "positions": int(part["length"].sum()),
                                 "effective_players": round(float(w.sum() ** 2 / (w ** 2).sum())),
                                 "games_per_rank": part.groupby("rank_id").size().tolist(),
                                 "hd1": int((part["handicap"] == 1).sum())}
    (OUT / "monitor").mkdir(exist_ok=True)
    for hd in (0, 1):
        monitor.loc[monitor["handicap"] == hd, columns].reset_index(drop=True).to_parquet(OUT / f"monitor_hd{hd}.parquet")
    for (model, name), table in extraction_tables(halves).items():
        for hd in (0, 1):     # HD>=2 (Valid-U only) goes with HD0: komi 6.5, as v1 / v2
            part = table[(table["handicap"] == 1) == bool(hd)]
            part[["game_index", "player_color"]].reset_index(drop=True).to_parquet(OUT / f"ext_{model}_{name}_hd{hd}.parquet")
            report["extract"][f"{model}_{name}_hd{hd}"] = len(part)
    report["extract"]["total"] = int(sum(report["extract"].values()))
    (OUT / "tables.json").write_text(json.dumps(report, indent=1, default=int))
    print(json.dumps(report, indent=1, default=int))


# ------------------------------------------------------------------ G0
def g0(arm: str) -> dict:
    result = {"arm": arm, "models": {}}
    for half in HALVES:
        run = OUT / arm / half
        if not (run / "done.json").exists():
            raise FileNotFoundError(f"{run} has not finished")
        log = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()]
        mon = [e for e in log if "monitor_official" in e]
        result["models"][half] = {"step0": mon[0]["monitor_official"], "final": mon[-1]["monitor_official"],
                                  "final_step": mon[-1]["step"], "min": min(e["monitor_official"] for e in mon),
                                  "nll_step0": mon[0]["monitor_nll"], "nll_final": mon[-1]["monitor_nll"],
                                  **{f"{k}_{when}": e.get(k) for k in ("mid_d_all", "mid_d_own", "mid_d_opp")
                                     for when, e in (("step0", mon[0]), ("final", mon[-1]))},
                                  "mid_pairs_step0": mon[0].get("mid_d_pairs"), "mid_pairs_final": mon[-1].get("mid_d_pairs"),
                                  "curve": [(e["step"], e["monitor_official"], e.get("mid_d_all")) for e in mon]}
    models = result["models"].values()
    gains = [m["final"] - m["step0"] for m in models]
    result["mean_gain"] = round(float(np.mean(gains)), 4)
    result["mean_mid_gain"] = round(float(np.mean([m["mid_d_all_final"] - m["mid_d_all_step0"] for m in models])), 4)
    result["pass"] = bool(min(gains) >= 0 and (result["mean_gain"] >= G0_MIN_GAIN
                                               or result["mean_mid_gain"] >= G0_MIN_MID_GAIN))
    (OUT / f"g0_{arm}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "models"} |
                     {h: {k: v for k, v in m.items() if k != "curve"} for h, m in result["models"].items()}))
    return result


# ------------------------------------------------------------------ features
def c4_frame(arm: str, model: str, names) -> pd.DataFrame:
    parts = [cfeat.run_features(OUT / arm / f"c_{model}_{n}_hd{hd}") for n in names for hd in (0, 1)]
    c = pd.concat(parts, ignore_index=True)
    if c["game_index"].duplicated().any():
        raise ValueError(f"{arm}/{model}: a game has C4 twice")
    return c.set_index("game_index")


def lookup(c: pd.DataFrame, gi: np.ndarray, needed: np.ndarray) -> np.ndarray:
    have = np.isin(gi, c.index)
    if (needed & ~have).any():
        raise ValueError(f"{int((needed & ~have).sum())} needed games without C4")
    out = np.full((len(gi), 20), np.nan)
    out[have] = c.loc[gi[have]].to_numpy()
    return out


def arm_x(x: np.ndarray, c4: np.ndarray, kind: str) -> np.ndarray:
    if kind == "swap":
        x = x.copy()
        x[:, C_COLS] = c4
        return x
    return np.hstack([x, c4])


def base_data(base: str):
    """(games, fit, eval_games, x_fit, x_eval) of the base: C v1 (more.data) or C v2 (cvec2)."""
    if base == "v1":
        games, fit, eval_games, x_fit, x_eval, _ = more.data(with_ex=False)
        return games, fit, eval_games, x_fit, x_eval
    games, fit, eval_games, xs = cv2.data()
    return games, fit, eval_games, *xs["v2"]


def c4_data(arm: str, fit: pd.DataFrame, eval_games: pd.DataFrame):
    """C4 of the fitting rows (out-of-half model) and the two views of the Valid-U rows."""
    half_of = pd.read_parquet(OUT / "halves.parquet").set_index("player_id")["half"]
    frames = {m: c4_frame(arm, m, ("valid", "t20")) for m in HALVES}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    owner = fit["player_id"].map(half_of).to_numpy()
    if pd.isna(owner[hd01]).any():
        raise ValueError("fitting rows of players in no half")
    gi = fit["game_index"].to_numpy()
    c_fit = np.full((len(fit), 20), np.nan)
    for model in HALVES:
        rows = owner == other(model)
        c_fit[rows] = lookup(frames[model], gi[rows], hd01[rows])
    egi = eval_games["game_index"].to_numpy()
    views = {m: lookup(frames[m], egi, np.ones(len(egi), bool)) for m in HALVES}
    return c_fit, views


def view_scores(model, eval_games, xs: list, q) -> np.ndarray:
    gi = eval_games["game_index"].to_numpy()
    p = np.mean([model.probabilities(x, gi, q) for x in xs], axis=0)
    return official_scores(score_rank(p), q["rank_id"].to_numpy())


# ------------------------------------------------------------------ mid-range diagnostic
GROUPS = {"own": list(range(0, 15)) + list(range(30, 40)), "opp": list(range(15, 30)) + list(range(40, 50)),
          "all": list(range(50))}     # x = [H own 15 | H opp 15 | C own 10 | C opp 10 | HD1 flags 2]
MID_PAIRS, DIAG_K, DIAG_DRAWS, DIAG_MIN_GAMES = (4, 5, 6), (1, 5, 20), 3, 20


def _questions(games: pd.DataFrame, x: np.ndarray, k: int, seed: int):
    """DIAG_DRAWS mean-of-k questions per player with >= DIAG_MIN_GAMES games (same players for every k)."""
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for rows in games.groupby(games["player_id"].astype(str)).indices.values():
        if len(rows) < DIAG_MIN_GAMES:
            continue
        for _ in range(DIAG_DRAWS):
            feats.append(x[rng.choice(rows, k, replace=False)].mean(axis=0))
            labels.append(int(games["rank_id"].iloc[rows[0]]))
    return np.array(feats), np.array(labels)


def mid_scaling(fit: pd.DataFrame, x_fit: np.ndarray, eval_games: pd.DataFrame, x_eval: np.ndarray) -> dict:
    """Mid-range question-level d' on Valid-U (HD<=1): per pair an LDA fitted on fitting-pool questions
    of the same k, applied to Valid-U questions; mean over the three pairs, per feature group and k."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    f_rows = (fit["handicap"] <= 1).to_numpy()
    e_rows = (eval_games["handicap"] <= 1).to_numpy()
    out = {}
    for group, cols in GROUPS.items():
        out[group] = {}
        for k in DIAG_K:
            xa, ya = _questions(fit[f_rows].reset_index(drop=True), x_fit[f_rows][:, cols], k, SEED)
            xb, yb = _questions(eval_games[e_rows].reset_index(drop=True), x_eval[e_rows][:, cols], k, SEED + 1)
            ds = []
            for r in MID_PAIRS:
                a, b = np.isin(ya, (r, r + 1)), np.isin(yb, (r, r + 1))
                z = LinearDiscriminantAnalysis().fit(xa[a], ya[a]).transform(xb[b])[:, 0]
                lo, hi = z[yb[b] == r], z[yb[b] == r + 1]
                ds.append(abs(hi.mean() - lo.mean()) / np.sqrt((hi.var() + lo.var()) / 2))
            out[group][str(k)] = round(float(np.mean(ds)), 3)
        out[group]["k20_over_k1"] = round(out[group]["20"] / out[group]["1"], 2)
    return out


# ------------------------------------------------------------------ G1
def evaluate(arm: str, base: str) -> None:
    games, fit, eval_games, x_fit, x_eval = base_data(base)
    c_fit, views = c4_data(arm, fit, eval_games)
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    xs = {"base": (x_fit, [x_eval])}
    for kind in ("swap", "add"):
        xs[kind] = (arm_x(x_fit, c_fit, kind), [arm_x(x_eval, views[m], kind) for m in HALVES])
    shift = np.abs(c_fit[hd01] - x_fit[hd01][:, C_COLS]).mean()
    view_gap = np.abs(views["A"] - views["B"]).mean()
    result = {"arm": arm, "base": base, "fit_games": int(hd01.sum()),
              "c4_minus_c_mean_abs": round(float(shift), 4), "view_a_minus_b_mean_abs": round(float(view_gap), 4),
              "mid_range": {"base": mid_scaling(fit, x_fit, eval_games, x_eval),
                            "swap": mid_scaling(fit, xs["swap"][0], eval_games, np.mean(xs["swap"][1], axis=0))},
              "seeds": {}}
    print(json.dumps({"mid_range": result["mid_range"]}), flush=True)
    for seed in hf.SEEDS:
        hs.SEED = seed
        models = {kind: more.PriorMatched(prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), xf[hd01])
                  for kind, (xf, _) in xs.items()}
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {kind: view_scores(models[kind], eval_games, xe, q) for kind, (_, xe) in xs.items()}
            for half in HALVES:      # diagnostic: one view only
                scores[f"{PRIMARY}_view{half}"] = view_scores(models[PRIMARY], eval_games,
                                                              [xs[PRIMARY][1][HALVES.index(half)]], q)
            a = scores["base"]
            entry[name] = {kind: hf.by_k(s, k) for kind, s in scores.items()}
            for kind in ("swap", "add"):
                b = scores[kind]
                entry[name][f"{kind}_diff_by_k"] = {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4)
                                                    for bk in K_BUCKETS}
                entry[name][f"{kind}_diff_equal_k"] = round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)
                if seed == hf.SEED:
                    entry[name][f"{kind}_bootstrap_95"] = hf.bootstrap(a, b, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {kind: e[f"{kind}_diff_equal_k"] for kind in ("swap", "add")}
                                for n, e in entry.items()}), flush=True)
    summary = {}
    for kind in ("swap", "add"):
        mean = {n: round(float(np.mean([result["seeds"][str(s)][n][f"{kind}_diff_equal_k"] for s in hf.SEEDS])), 4)
                for n in sets}
        e2_k = {str(bk): round(float(np.mean([result["seeds"][str(s)]["E2"][f"{kind}_diff_by_k"][str(bk)]
                                              for s in hf.SEEDS])), 4) for bk in K_BUCKETS}
        promote = max(mean.values()) >= G1_PROMOTE and min(mean.values()) >= 0 and min(e2_k.values()) >= -G1_MAX_K_DROP
        summary[kind] = {"mean_diff_equal_k": mean, "E2_mean_diff_by_k": e2_k,
                         "decision": "promote" if promote else "probe" if min(mean.values()) >= 0 else "stop"}
    result["G1"] = summary[PRIMARY] | {"arm": PRIMARY}
    result["diagnostic_add"] = summary["add"]
    print(json.dumps({"G1": result["G1"], "add": summary["add"]}))
    (OUT / f"result_{arm}_{base}.json").write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ export
def export(arm: str, base: str, kind: str, job: str) -> None:
    """Base config (Valid-S + train20 + Valid-U, prior matched, seed 2027) with C4 (kind), test
    probabilities averaged over the two views. Base v1 must reproduce -prior_736589."""
    games, fit, eval_games, x_fit, x_eval = base_data(base)
    c_fit, views = c4_data(arm, fit, eval_games)
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    rng = np.random.default_rng(SEED)
    u_players = np.sort(eval_games.loc[u, "player_id"].astype(str).unique())
    pick = dict(zip(u_players, rng.choice(HALVES, len(u_players))))
    u_view = eval_games.loc[u, "player_id"].astype(str).map(pick).to_numpy()
    c_u = np.where((u_view == "A")[:, None], views["A"][u], views["B"][u])
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    x_base = np.vstack([x_fit[hd01], x_eval[u]])
    x_c4 = np.vstack([arm_x(x_fit[hd01], c_fit[hd01], kind), arm_x(x_eval[u], c_u, kind)])
    models = {"base": more.PriorMatched(prior).fit(stacked, pd.DataFrame(), x_base),
              "c4": more.PriorMatched(prior).fit(stacked, pd.DataFrame(), x_c4)}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {v: [] for v in models}
    for part in ("public", "private"):
        if base == "v1":       # cv2.test_x would also need the C v2 test features
            table, _ = p3.test_table(part, "hc", columns)
            xb, index = table.to_numpy()[:, 10:], table.index.to_numpy()
        else:
            _, xb, index = cv2.test_x(part, columns)
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        test_views = [arm_x(xb, lookup(c4_frame(arm, m, (f"test_{part}",)), index, np.ones(len(index), bool)), kind)
                      for m in HALVES]
        probs = {"base": models["base"].probabilities(xb, index, questions),
                 "c4": np.mean([models["c4"].probabilities(x, index, questions) for x in test_views], axis=0)}
        for v, p in probs.items():
            rows[v].append(pd.DataFrame({"question_id": questions["question_id"],
                                         "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    reference = pd.read_csv(cv2.BASE, dtype=str).set_index("question_id")["pred_rank"].loc[template.question_id]
    report = {"arm": arm, "base": base, "kind": kind}
    for v, frames in rows.items():
        result = pd.concat(frames, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
        changed = result != reference
        report[v] = {"public": int(changed[changed.index.str.startswith("pub")].sum()),
                     "private": int(changed[changed.index.str.startswith("priv")].sum()),
                     "distribution": result.value_counts().reindex(RANKS).fillna(0).astype(int).tolist()}
        if v == "base":
            if base == "v1" and changed.any():
                raise ValueError("base refit does not reproduce -prior_736589")
            continue
        report[v]["file"] = write_rank_csv(result.reset_index(), f"c4x-{arm}", job).name
    print(json.dumps({"changed_vs_prior_736589": report}))
    (OUT / f"export_{arm}_{job}.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["tables", "g0", "evaluate", "export"])
    parser.add_argument("--arm", choices=["gen", "mmi", "adj"])
    parser.add_argument("--base", choices=["v1", "v2"], default="v1", help="C of the base (v2 = cvec2 vectors)")
    parser.add_argument("--kind", choices=["swap", "add"], default=PRIMARY, help="export: how C4 enters x")
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    if args.command == "tables":
        tables()
    elif args.arm is None:
        parser.error("--arm is required")
    elif args.command == "g0":
        g0(args.arm)
    elif args.command == "evaluate":
        evaluate(args.arm, args.base)
    else:
        export(args.arm, args.base, args.kind, args.job)
    return 0


if __name__ == "__main__":
    sys.exit(main())
