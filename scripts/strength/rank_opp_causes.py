"""Diagnose opp7 calibration on Valid-S before opening E1 or E2."""
from __future__ import annotations

import argparse
import json
import re
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import log_softmax

from go_aicup.metrics import K_BUCKETS, score_rank
from scripts.strength import rank_hc_s_fusion as fusion
from scripts.strength import rank_opp_next as n

TAGS = {"soft": "ofit-l10", "length": "olen-l10", "short": "osh-l10", "dampen": "odmp-l10", "long": "olong10"}
SHORT_PLIES, LONG_PLIES, SHRINK_GAMES, SHORT_SCALE = 93, 131, 100, 0.93


def positions(cache):
    jobs = pd.read_parquet(n.llr.OUT / "jobs.parquet")
    jobs = jobs[jobs["id"] >= n.llr.QUERY_ID].sort_values("id")
    if not np.array_equal(jobs["id"].to_numpy() - n.llr.QUERY_ID, cache["masked_rows"]):
        raise ValueError("LLR cache and policy jobs have different game rows")
    games = n.llr.job_games(jobs)
    lengths = [np.count_nonzero(colours == keep) for (_, colours), keep in zip(games, jobs["keep"])]
    if not all(lengths):
        raise ValueError("policy job without any opponent positions")
    return pd.Series(lengths, index=cache["masked_rows"])


def scale_llr(cache, plies):
    scale = np.sqrt(115 / plies).clip(0.8, 1.2)
    return cache | {f"{label}_llr": cache[f"{label}_llr"] * scale.loc[cache[f"{label}_rows"]].to_numpy()[:, None]
                    for label in ("masked", "seen")}


def kernel(t, top, *, soft=False):
    return n.rf._delta_prob(t, soft_top=top if soft else None, tau=n.TAU)


def length_evidence(fit, top, *, plies, arm):
    """Apply the chosen evidence adjustment only to its length group."""
    pooled = kernel(fit, top)
    if arm == "short":
        short = fit[plies.loc[fit.index].to_numpy() <= SHORT_PLIES]
        groups = ["hd", "oc", "bin"]
        shape = (2, 2, n.fu.N_BIN)
        counts = short.groupby(groups).size().reindex(pd.MultiIndex.from_product(map(range, shape)), fill_value=0)
        count = counts.to_numpy().reshape(shape)
        weight = count / (count + SHRINK_GAMES)
        adjusted = pooled + weight[:, :, None, :, None] * (kernel(short, top) - pooled)

    def predict(target):
        ll = n.rf.soft_ll(pooled, target, top, n.TAU)
        length = plies.loc[target.index].to_numpy()
        if arm == "long":
            ll[length > LONG_PLIES] /= SHORT_SCALE
        elif arm == "dampen":
            ll[length <= SHORT_PLIES] *= SHORT_SCALE
        else:
            use = length <= SHORT_PLIES
            ll[use] = n.rf.soft_ll(adjusted, target.iloc[use], top, n.TAU)
        return ll

    return predict


def gap_calibration(fit, held):
    """Bin the gap in the fitting players only and estimate adjacent-grade match rates."""
    cuts = np.quantile(fit["gap"], np.arange(1, 10) / 10)
    bins = np.searchsorted(cuts, fit["gap"], side="right")
    count = np.bincount(bins, minlength=10)
    hits = np.bincount(bins, weights=fit["hit"], minlength=10)
    return ((hits + 0.5) / (count + 1))[np.searchsorted(cuts, held["gap"], side="right")]


def diagnose(rows, top, plies, *, length_arm=None):
    s = rows[rows["set"] == "s_opp"].copy()
    s["plies"] = plies.loc[s.index]
    s["hit"] = (s["owner_rank"] - s["rj"]).abs().le(1)
    players = s.groupby("owner")["owner_rank"].first().sort_index()
    folds = s["owner"].map(fusion._stratified(players, np.random.default_rng(2027)))
    lls, soft_groups, length_lls = [], [], []
    estimated = pd.Series(np.nan, index=s.index)
    for f in sorted(folds.unique()):
        fit, held = s[folds != f], s[folds == f]
        low = held[held["bin"] == 0]
        truth = low["owner_rank"].to_numpy(dtype=int)
        losses = []
        for soft in (False, True):
            ll = n.rf.soft_ll(kernel(fit, top, soft=soft), low, top, n.TAU)
            losses.append(-log_softmax(ll, axis=1)[np.arange(len(low)), truth])
        lls.append(np.column_stack(losses))
        if length_arm:
            rank, weight = n.rf._soft_candidates(low, top, n.TAU)
            mass = (weight[:, :, None] * (rank[:, :, None] == np.arange(10))).sum(axis=1)
            soft_groups.append(mass.argmax(axis=1) == rank[:, 0])
            use = held["plies"] > LONG_PLIES if length_arm == "long" else held["plies"] <= SHORT_PLIES
            target = held[use]
            labels = target["owner_rank"].to_numpy(dtype=int)
            orig = n.rf.soft_ll(kernel(fit, top), target, top, n.TAU)
            adjusted = length_evidence(fit, top, plies=plies, arm=length_arm)(target)
            length_lls.append(np.column_stack([-log_softmax(ll, axis=1)[np.arange(len(labels)), labels]
                                               for ll in (orig, adjusted)]))
        for (hd, c), g in held.groupby(["hd", "qcolor"]):
            other = fit[(fit["hd"] == hd) & (fit["qcolor"] == c)]
            estimated.loc[g.index] = gap_calibration(other, g)
    if estimated.isna().any():
        raise ValueError("incomplete cross-fitted gap predictions")
    soft_losses = np.vstack(lls)
    base, soft = soft_losses.mean(axis=0)
    stages = {"soft": {"games": len(soft_losses), "players": len(players), "hard_log_loss": float(base),
                       "soft_log_loss": float(soft), "delta": float(soft - base), "proceed": bool(soft < base)}}
    s["expected_hit"] = estimated
    weights = s.groupby(["hd", "qcolor"]).size().div(len(s))
    length = {}
    for label, g in (("short", s[s["plies"] <= SHORT_PLIES]), ("long", s[s["plies"] > LONG_PLIES])):
        by_group = g.groupby(["hd", "qcolor"]).agg(gap=("gap", "median"), hit=("hit", "mean"),
                                                     expected_hit=("expected_hit", "mean"))
        summary = by_group.reindex(weights.index).mul(weights, axis=0).sum(min_count=len(weights))
        if not np.isfinite(summary.to_numpy()).all():
            raise ValueError(f"{label}: missing HD and colour stratum")
        length[label] = {**summary.to_dict(), "games": len(g), "residual": float(summary.hit - summary.expected_hit)}
    short, long = length["short"], length["long"]
    length["proceed"] = bool(long["gap"] > short["gap"] and long["residual"] < min(0, short["residual"]))
    stages["length"] = length
    if length_arm:
        agree = np.concatenate(soft_groups)
        diff = soft_losses[:, 1] - soft_losses[:, 0]
        stages["soft"]["by_rank_agreement"] = {
            name: {"games": int(use.sum()), "delta": float(diff[use].mean())}
            for name, use in (("agree", agree), ("disagree", ~agree)) if use.any()}
        loss = np.vstack(length_lls)
        before, after = loss.mean(axis=0)
        stages[length_arm] = {"games": len(loss), "base_log_loss": float(before), "candidate_log_loss": float(after),
                           "delta": float(after - before), "proceed": bool(after < before)}
    return stages


def score(questions, split, models, xe, eval_index, variants, plies):
    truth = questions["rank_id"].to_numpy()
    evidence = {}
    for name, (rows, top) in variants.items():
        fit, held = (rows[rows["set"] == subset] for subset in ("s_opp", "u_opp"))
        if name in ("short", "dampen", "long"):
            ll = length_evidence(fit, top, plies=plies, arm=name)(held)
            per_game = pd.DataFrame(ll, index=held["game_index"].to_numpy())
            evidence[name] = n.fu.question_ll(questions["games"], per_game)
        else:
            evidence[name] = n.question_evidence(questions, held, top, kernel(fit, top, soft=name == "soft"))
    results = []
    bounds = {"split": split, "question_row": np.arange(len(questions)), "k": questions["k"].to_numpy(),
              "truth": truth}
    for seed, model in models.items():
        p = model.probabilities(xe, eval_index, questions)
        b2 = score_rank(p)
        base = n.predict(p, evidence["opp7"])
        base_score = n.qm.official(base, truth)
        for name, ll in evidence.items():
            if name == "opp7":
                continue
            pred = n.predict(p, ll)
            value = n.qm.official(pred, truth)
            results.append(pd.DataFrame(bounds | {"arm": name, "seed": seed, "b2_rank": b2, "opp7_rank": base,
                                                  "rank": pred, "opp7_score": base_score, "score": value,
                                                  "delta": value - base_score}))
    return pd.concat(results, ignore_index=True)


def summarize(predictions, arm):
    per_seed = predictions[predictions["arm"] == arm].groupby(["seed", "k"])["delta"].mean().unstack("k")
    by_k = per_seed.loc[:, list(K_BUCKETS)].mean()
    return {str(k): float(by_k[k]) for k in K_BUCKETS} | {"equal_k": float(by_k.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--short-fix", dest="length_arm", action="store_const", const="short")
    modes.add_argument("--short-dampen", dest="length_arm", action="store_const", const="dampen")
    modes.add_argument("--long-boost", dest="length_arm", action="store_const", const="long")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.job):
        parser.error("invalid job identifier")
    out = n.llr.OUT / "next"
    out.mkdir(exist_ok=True)
    path = out / f"{args.job}_causes.json"

    def save(report):
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(path)

    q = pd.read_parquet(n.link.OUT / "queries.parquet")
    arm = json.loads((n.link.OUT / "result.json").read_text())["selected_arm"]
    with np.load(n.link.OUT / "scores.npz") as z, np.load(n.llr.OUT / "llr_g40.npz") as source:
        cache = dict(source)
        rows, top, _ = n.link_table(q, n.llr.reranked(n.MU, z, arm, cache), z, cache)
        plies = positions(cache)
        diagnostics = diagnose(rows, top, plies, length_arm=args.length_arm)
        report = {"reference": n.BASE.name, "diagnostics": diagnostics, "scores": {}}
        save(report)
        print(json.dumps(report), flush=True)
        variants = {"opp7": (rows, top)}
        if args.length_arm:
            if diagnostics[args.length_arm]["proceed"]:
                variants[args.length_arm] = (rows, top)
        else:
            if diagnostics["soft"]["proceed"]:
                variants["soft"] = (rows, top)
            if diagnostics["length"]["proceed"]:
                rr = n.llr.reranked(n.MU, z, arm, scale_llr(cache, plies))
                t, v, _ = n.link_table(q, rr, z, cache)
                variants["length"] = (t, v)
    if len(variants) == 1:
        return

    fit, eval_games, x_fit, x_eval = n.prod.train_data()
    o, (e,) = n.nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.column_stack([x_fit, o]), np.column_stack([x_eval, e])
    prior = n.more.player_prior(fit[fit["role"] == "S"])
    models = {}
    for seed in n.prod.SEEDS:
        models[seed] = n.prod._fit(fit, xf, prior, seed)
        print(f"fitted seed {seed}", flush=True)
    eval_index = eval_games["game_index"].to_numpy()
    e1 = pd.read_parquet(n.hf.Q1000)
    e1 = score(e1[e1["role"] == "U"].reset_index(drop=True), "E1", models, xe, eval_index, variants, plies)
    e1.to_parquet(out / f"{args.job}_causes_E1.parquet", index=False)
    for name in variants.keys() - {"opp7"}:
        report["scores"][name] = {"E1": summarize(e1, name), "pass": False}
    save(report)
    eligible = {name: variants[name] for name, s in report["scores"].items() if s["E1"]["equal_k"] >= 0}
    if eligible:
        e2 = pd.read_parquet(n.hf.OUT / "questions_e2_hd01.parquet")
        e2 = score(e2, "E2", models, xe, eval_index, {"opp7": variants["opp7"], **eligible}, plies)
        e2.to_parquet(out / f"{args.job}_causes_E2.parquet", index=False)
        for name in eligible:
            s = report["scores"][name]
            s["E2"] = summarize(e2, name)
            s["pass"] = s["E2"]["equal_k"] >= 0.002 and all(s["E2"][str(k)] >= -0.005 for k in K_BUCKETS)
        save(report)
        for name, (t, v) in eligible.items():
            if report["scores"][name]["pass"]:
                game_evidence = (partial(length_evidence, plies=plies, arm=name)
                                 if name in ("short", "dampen", "long") else None)
                report["scores"][name]["export"] = n.exports(
                    args.job, name, t, v, False, None, soft_counts=name == "soft", tag=TAGS[name],
                    game_evidence=game_evidence)
                save(report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
