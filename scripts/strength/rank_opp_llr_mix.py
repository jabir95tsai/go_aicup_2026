"""Compare fixed 27D, 50D and equal-weight opponent LLR on the same B2 draws."""
from __future__ import annotations

import argparse
import json
import re
from time import monotonic

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS
from scripts.strength import rank_opp_next as n
from scripts.strength.rank_opp_calibrate import interval

DRAW_SEED = 20261124
TAG = "omix-l10"


def mix_llr(base, rich, width):
    keys = ("qrows", "masked_rows", "seen_rows")
    if any(not np.array_equal(base[k], rich[k]) for k in keys):
        raise ValueError("27D and 50D LLR query rows differ")
    mixed = {k: base[k] for k in keys}
    for label in ("masked", "seen"):
        key = f"{label}_llr"
        left, right = base[key], rich[key]
        shape = (len(base[f"{label}_rows"]), width)
        if left.shape != shape or right.shape != shape:
            raise ValueError(f"{label} LLR matrices must have shape {shape}")
        if not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError(f"{label} LLR matrices contain non-finite scores")
        mixed[key] = 0.5 * left.astype(np.float64) + 0.5 * right
    return mixed


def validation_questions(sets, games, fit_players):
    meta = games.set_index("game_index", verify_integrity=True)
    excluded = set(fit_players) | set(sets["E1"]["player_id"]) | set(sets["E2"]["player_id"])
    clean = games.loc[~games["player_id"].isin(excluded) & games["handicap"].le(1)]
    rng = np.random.default_rng(DRAW_SEED)
    rows = []
    for pid, group in clean.sort_values("game_index").groupby("player_id", sort=True):
        colours = [g for _, g in group.groupby("player_color", sort=True) if len(g) >= 5]
        if colours:
            g = colours[rng.integers(len(colours))]
            rows.append({"question_id": len(rows), "player_id": pid, "rank_id": int(g["rank_id"].iat[0]),
                         "k": 5, "color": int(g["player_color"].iat[0]),
                         "games": np.sort(rng.choice(g["game_index"].to_numpy(), size=5, replace=False))})
    independent = pd.DataFrame(rows)
    if independent.empty:
        raise ValueError("no independent Valid-U players are eligible")
    questions = []
    for split, table in (("E1", sets["E1"]), ("E2", sets["E2"]), ("independent_k5", independent)):
        table = table.reset_index(drop=True).assign(split=split)
        # E1 stores color as B/W, while E2 and the production metadata use 0/1.
        if split == "E1":
            table["color"] = table["color"].map({"B": 0, "W": 1}).astype(int)
        table["has_hd2plus"] = [meta.loc[g, "handicap"].gt(1).any() for g in table["games"]]
        for row in table.itertuples(index=False):
            if len(set(row.games)) != row.k:
                raise ValueError(f"{split} question {row.question_id} has repeated or missing games")
            used = meta.loc[row.games]
            if not (used["player_id"].eq(row.player_id).all() and used["rank_id"].eq(row.rank_id).all()
                    and used["player_color"].eq(row.color).all()):
                raise ValueError(f"{split} question {row.question_id} has inconsistent game metadata")
        questions.append(table)
    result = pd.concat(questions, ignore_index=True).rename_axis("question_row").reset_index()
    if result.loc[result["split"].ne("E1"), "has_hd2plus"].any():
        raise ValueError("E2 or independent questions contain HD>=2 games")
    return result


def summarize(predictions, buckets):
    scores = predictions.groupby(["arm", "seed", "k"])["score"].mean().unstack("k").loc[:, list(buckets)]
    scores["equal_k"] = scores.mean(axis=1)
    deltas = scores.subtract(scores.loc["opp7"], level="seed").drop(index="opp7")
    return {"absolute": scores.groupby("arm").mean().to_dict("index"),
            "mean_delta": deltas.groupby("arm").mean().to_dict("index"),
            "delta_by_seed": {a: d.droplevel("arm").to_dict("index") for a, d in deltas.groupby(level="arm")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.job):
        parser.error("invalid job identifier")
    start = monotonic()
    path = n.llr.OUT / "next" / f"{args.job}_llr_mix.json"
    path.parent.mkdir(exist_ok=True)
    report = {"reference": n.BASE.name, "draw_seed": DRAW_SEED, "b2_seeds": n.prod.SEEDS,
              "mu": n.MU, "tau": n.TAU, "alpha": n.ALPHA, "mix": [0.5, 0.5],
              "gate": {"E1_equal_k": 0, "E2_equal_k": 0.002, "E2_each_k": -0.005, "independent_k5": 0}}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(path)

    q = pd.read_parquet(n.link.OUT / "queries.parquet")
    arm = json.loads((n.link.OUT / "result.json").read_text())["selected_arm"]
    with np.load(n.llr.OUT / "llr_g40.npz") as b, np.load(n.llr.OUT / "llr_g40_rich.npz") as r:
        caches = {"opp7": dict(b), "rich": dict(r), "mix": mix_llr(b, r, n.llr.o2.TOP)}
    fit, games, xf, xe = n.prod.train_data()
    questions = validation_questions(n.prod.question_sets(), games, fit["player_id"])
    questions.drop(columns="role", errors="ignore").to_parquet(path.with_name(f"{path.stem}_questions.parquet"), index=False)
    ids = questions["split"].eq("independent_k5")
    counts = questions.loc[ids].groupby("rank_id")["player_id"].nunique()
    report["independent_players_per_rank"] = counts.reindex(range(10), fill_value=0).tolist()
    report["E1_k5_hd01_questions"] = int((questions["split"].eq("E1") & questions["k"].eq(5)
                                         & ~questions["has_hd2plus"]).sum())
    print(f"Independent k=5 players by rank: {report['independent_players_per_rank']}", flush=True)

    links, evidence = {}, {}
    report["ref_null_edges"] = {}
    with np.load(n.link.OUT / "scores.npz") as z:
        for name, cache in caches.items():
            ranked = n.llr.reranked(n.MU, z, arm, cache)
            rows, top, edges = n.link_table(q, ranked, z, cache)
            s, u = (rows[rows["set"].eq(split)] for split in ("s_opp", "u_opp"))
            if len(s) != 20_000 or not s["game_index"].is_unique:
                raise ValueError("expected 20,000 distinct Valid-S DELTA calibration games")
            links[name] = rows, top
            evidence[name] = n.question_evidence(questions, u, top, n.rf._delta_prob(s))
            report["ref_null_edges"][name] = {str(c): e.tolist() for c, e in edges.items()}
    save()
    oof, (val,) = n.nl.oof("yH", fit, xf, [xe])
    xf, xe = np.column_stack([xf, oof]), np.column_stack([xe, val])
    prior = n.more.player_prior(fit[fit["role"].eq("S")])
    truth = questions["rank_id"].to_numpy()
    labels = questions[["question_row", "question_id", "split", "player_id", "k", "has_hd2plus"]]
    frames = []
    for seed in n.prod.SEEDS:
        p = n.prod._fit(fit, xf, prior, seed).probabilities(xe, games["game_index"].to_numpy(), questions)
        pred = {name: n.predict(p, ll) for name, ll in evidence.items()}
        base = n.qm.official(pred["opp7"], truth)
        for name, rank in pred.items():
            score = n.qm.official(rank, truth)
            frames.append(labels.assign(seed=seed, arm=name, truth=truth, rank=rank, score=score,
                                        opp7_rank=pred["opp7"], delta=score - base))
        print(f"Evaluated B2 seed {seed}", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(path.with_name(f"{path.stem}_predictions.parquet"), index=False)
    report["scores"] = {}
    for split in ("E1", "E2", "E1_k5_hd01", "E1_k5_hd2plus", "independent_k5"):
        if split in ("E1", "E2", "independent_k5"):
            sub = predictions[predictions["split"].eq(split)]
        else:
            sub = predictions[predictions["split"].eq("E1") & predictions["k"].eq(5)
                              & predictions["has_hd2plus"].eq(split == "E1_k5_hd2plus")]
        buckets = K_BUCKETS if split in ("E1", "E2") else (5,)
        result = summarize(sub, buckets)
        if buckets == (5,):
            result["paired_player_bootstrap"] = {}
            result["delta_by_rank"] = {}
            for name in ("rich", "mix"):
                pairs = sub[sub["arm"].eq(name)]
                d = pairs.pivot(index="seed", columns="question_row", values="delta")
                paired = questions.loc[d.columns]
                result["paired_player_bootstrap"][name] = interval(paired, d.to_numpy(), buckets=buckets)
                result["delta_by_rank"][name] = pairs.groupby("truth")["delta"].mean().to_dict()
        report["scores"][split] = result
        print(split, json.dumps(result["mean_delta"]), flush=True)

    gate = report["gate"]
    e1, e2, hold = (report["scores"][s]["mean_delta"]["mix"] for s in ("E1", "E2", "independent_k5"))
    report["pass"] = bool(e1["equal_k"] >= gate["E1_equal_k"] and e2["equal_k"] >= gate["E2_equal_k"]
                          and all(e2[k] >= gate["E2_each_k"] for k in K_BUCKETS) and hold[5] >= gate["independent_k5"])
    report["evaluation_seconds"] = round(monotonic() - start, 1)
    save()
    if report["pass"]:
        report["export"] = n.exports(args.job, "mix", *links["mix"], False, None, tag=TAG)
        save()
    print(f"Saved {path}, gate passed: {report['pass']}", flush=True)


if __name__ == "__main__":
    main()
