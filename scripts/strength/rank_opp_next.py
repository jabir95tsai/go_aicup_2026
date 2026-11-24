"""Four independent opponent-link comparisons against opp7 (tau 3.5, alpha 0.55)."""
from __future__ import annotations

import argparse
import json
import re
from functools import partial

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, rank_gains, score_rank
from go_aicup.paths import ROOT

from scripts.strength import humansl_hd_filter as hf
from scripts.strength import rank_more_players as more
from scripts.strength import rank_nonlin as nl
from scripts.strength import rank_opp_fuse as fu
from scripts.strength import rank_opp_link2 as link
from scripts.strength import rank_opp_llr as llr
from scripts.strength import rank_opp_refine as rf
from scripts.strength import rank_prod_a0_fusion as prod
from scripts.strength import rank_qmerge as qm

TAU, ALPHA, MU = 3.5, 0.55, 2.0
TAGS = {"colour": "oclr-l10", "delta": "odlt-l10", "confidence": "ocon-l10", "rich": "orich10"}
BASE = ROOT / "submissions" / "rank" / "849173-opp7-l10.csv"


def reranks(q, z, arm, cache, *, colourwise=False):
    ref = q.index.isin(cache["seen_rows"])
    colour = q["qcolor"].to_numpy()
    players = np.array(sorted(q.loc[ref, "owner"].unique()))
    half_a = set(np.random.default_rng(llr.SEED).permutation(players)[:len(players) // 2])
    a = ref & q["owner"].isin(half_a).to_numpy()
    b = ref & ~a
    ranked = {mu: llr.reranked(mu, z, arm, cache) for mu in (1.0, MU)}

    def stats(rows, null_rows, rr):
        mi, mv = rr["masked"]
        si, sv = rr["seen"]
        null_gap = mv[:, 0] - mv[:, 1]
        seen_gap = sv[:, 0] - sv[:, 1]
        cut = {c: np.quantile(null_gap[null_rows & (colour == c)], 0.99)
               for c in np.unique(colour[rows])}
        hit = si[rows, 0] == z["owner_col"][rows]
        tail = hit & (seen_gap[rows] > np.array([cut[c] for c in colour[rows]]))
        return {"top1": float(hit.mean()), "hit_q99": float(tail.mean())}

    candidates = {}
    chosen = {}
    for c in (0, 1) if colourwise else (None,):
        use = a if c is None else a & (colour == c)
        candidates[str(c)] = {mu: stats(use, use, rr) for mu, rr in ranked.items()}
        chosen[c] = max((MU, 1.0), key=lambda mu: candidates[str(c)][mu]["hit_q99"])
    if colourwise:
        left = (colour == 0)[:, None]
        rr = {key: tuple(np.where(left, x, y) for x, y in zip(ranked[chosen[0]][key], ranked[chosen[1]][key]))
              for key in ("masked", "seen")}
    else:
        rr = ranked[chosen[None]]
    report = {"mu": {str(c): mu for c, mu in chosen.items()}, "ref_a": candidates,
              "ref_b": stats(b, b, rr), "opp7_ref_b": stats(b, b, ranked[MU]),
              "ref_b_by_colour": {str(c): stats(b & (colour == c), b, rr) for c in (0, 1)}}
    return ranked[MU], rr, report


def link_table(q, rr, z, cache):
    idx, val = rr["masked"]
    gap = val[:, 0] - val[:, 1]
    ref = q.index.isin(cache["seen_rows"])
    colour = q["qcolor"].to_numpy()
    edges = {c: np.r_[np.quantile(gap[ref & (colour == c)], fu.QS), gap[ref & (colour == c)].max()]
             for c in (0, 1)}
    rows = llr.link_rows(q, idx, val, z["cand_rank"], edges).loc[cache["masked_rows"]]
    return rows, {"idx": idx, "val": val, "cand_rank": z["cand_rank"]}, edges


def add_calibration_games(q, rows, top, edges, arm, cache):
    extra = pd.read_parquet(link.OUT / "queries_extra.parquet")
    extra.index = np.arange(len(q), len(q) + len(extra))
    with np.load(llr.OUT / "llr_g40_expanded.npz") as expanded, np.load(link.OUT / "scores_extra.npz") as z:
        if (not np.array_equal(expanded["qrows"], cache["qrows"])
                or not np.array_equal(extra["game_index"], expanded["extra_game_index"])
                or not np.array_equal(z["cand_rank"], top["cand_rank"])):
            raise ValueError("extra Valid-S queries or candidate ranks do not match the opp7 cache")
        val = z[f"{arm}_val"].astype(np.float64) + MU * expanded["extra_games_llr"]
        order = np.argsort(-val, axis=1)
        idx = np.take_along_axis(z[f"{arm}_idx"], order, axis=1)
        val = np.take_along_axis(val, order, axis=1)
        add = llr.link_rows(extra, idx, val, top["cand_rank"], edges)
    combined = pd.concat([rows[rows["set"] == "s_opp"], add], verify_integrity=True)
    if combined["game_index"].duplicated().any():
        raise ValueError("duplicate Valid-S calibration game")
    return combined, {"idx": np.vstack([top["idx"], idx]), "val": np.vstack([top["val"], val]),
                      "cand_rank": top["cand_rank"]}


def question_evidence(questions, rows, top, prob):
    game_ll = pd.DataFrame(rf.soft_ll(prob, rows, top, TAU), index=rows["game_index"].to_numpy())
    return fu.question_ll(questions["games"], game_ll)


def margin(p):
    two = np.partition(rank_gains(p), -2, axis=1)[:, -2:]
    return two[:, 1] - two[:, 0]


def question_alpha(p, cut, low, high):
    return np.where(margin(p) <= cut, low, high)


def predict(p, evidence, rule=None):
    alpha = ALPHA if rule is None else rule(p)[:, None]
    return score_rank(fu.posterior(p, evidence, alpha))


def calibrate(fit, x_fit, cal_rows, cal_top, delta_rows):
    eligible = fit[(fit["role"] == "S") & fit["game_index"].isin(cal_rows["game_index"])]
    questions, fold = prod.calibration_questions(eligible, set())
    p = np.full((len(questions), 10), np.nan)
    evidence = np.full_like(p, np.nan)
    for i in sorted(fold.unique()):
        held = set(fold.index[fold == i])
        train = ~fit["player_id"].astype(str).isin(held).to_numpy()
        val = ~train
        sub = fit.loc[train]
        o, (v,) = nl.oof("yH", sub, x_fit[train], [x_fit[val]])
        model = prod._fit(sub, np.column_stack([x_fit[train], o]), more.player_prior(sub[sub["role"] == "S"]), 2027)
        select = questions["inner"] == i
        q = questions[select]
        p[select] = model.probabilities(np.column_stack([x_fit[val], v]), fit.loc[val, "game_index"].to_numpy(), q)
        kernel = rf._delta_prob(delta_rows[~delta_rows["owner"].astype(str).isin(held)])
        games = cal_rows[cal_rows["game_index"].isin(fit.loc[val, "game_index"])]
        evidence[select] = question_evidence(q, games, cal_top, kernel)
        print(f"confidence OOF fold {i}: {len(q)} questions, {len(held)} players", flush=True)
    if not (np.isfinite(p).all() and np.isfinite(evidence).all()):
        raise ValueError("incomplete out-of-fold predictions")
    cut = float(np.median(margin(p)))
    unsure = margin(p) <= cut
    k, truth = questions["k"].to_numpy(), questions["rank_id"].to_numpy()
    weights = np.array([1 / (len(K_BUCKETS) * (k == v).sum()) for v in k])
    grid = (ALPHA, 0.35, 0.75)
    scores = {a: weights * qm.official(score_rank(fu.posterior(p, evidence, a)), truth) for a in grid}
    low = max(grid, key=lambda a: scores[a][unsure].sum())
    high = max(grid, key=lambda a: scores[a][~unsure].sum())
    report = {"questions": len(questions), "players": questions["player_id"].nunique(),
              "margin_threshold": cut, "alpha_low_margin": low, "alpha_high_margin": high,
              "oof_delta_equal_k": float(scores[low][unsure].sum() + scores[high][~unsure].sum() - scores[ALPHA].sum())}
    return partial(question_alpha, cut=cut, low=low, high=high), report


def evaluate(sets, fit, eval_games, x_fit, x_eval, evidence, rules):
    o, (e,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.column_stack([x_fit, o]), np.column_stack([x_eval, e])
    prior = more.player_prior(fit[fit["role"] == "S"])
    scores = {name: {split: {} for split in sets} for name in evidence}
    for seed in prod.SEEDS:
        model = prod._fit(fit, xf, prior, seed)
        for split, q in sets.items():
            p = model.probabilities(xe, eval_games["game_index"].to_numpy(), q)
            for name, ll in evidence.items():
                score = qm.official(predict(p, ll[split], rules.get(name)), q["rank_id"].to_numpy())
                k = q["k"].to_numpy()
                scores[name][split][str(seed)] = {**{str(v): float(score[k == v].mean()) for v in K_BUCKETS},
                                                  "equal_k": float(hf.equal_k(score, k))}
        print(f"evaluated seed {seed}", flush=True)
    return scores


def deltas(scores, base):
    by_seed, mean = {}, {}
    for split, seeds in scores.items():
        by_seed[split] = {s: {k: v - base[split][s][k] for k, v in entry.items()} for s, entry in seeds.items()}
        mean[split] = {k: float(np.mean([v[k] for v in by_seed[split].values()])) for k in next(iter(seeds.values()))}
    return by_seed, mean


def exports(job, name, rows, top, smooth, rule):
    pool = rows[rows["set"].isin(["s_opp", "u_opp"])]
    kernel = rf._delta_prob(pool, smooth=smooth)
    test = rows[rows["set"] == "test"].reset_index(names="query_row").set_index(["part", "store_row"])

    def evidence(games, sub, _table, key):
        index = pd.MultiIndex.from_frame(sub[["part", "store_row"]])
        aligned = sub.set_index(test.loc[index, "query_row"].to_numpy())
        values = rf.soft_ll(kernel, aligned, top, TAU)
        return fu.question_ll(games, pd.DataFrame(values, index=sub[key].to_numpy()))

    report = fu.export(rows, f"DELTA-ALL_{ALPHA}", job, evidence=evidence, alpha_rule=rule, tag=TAGS[name])
    pred = pd.read_csv(BASE.with_name(report["file"]), dtype=str)
    base = pd.read_csv(BASE, dtype=str)
    if len(pred) != 800 or not pred["question_id"].equals(base["question_id"]):
        raise ValueError("export questions differ from opp7")
    changed = pred["pred_rank"].ne(base["pred_rank"])
    report["changed_vs_opp7"] = {part: int((changed & pred["question_id"].str.startswith(prefix)).sum())
                                 for part, prefix in (("public", "pub"), ("private", "priv"))}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs="+", choices=tuple(TAGS))
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.job) or len(set(args.arms)) != len(args.arms):
        parser.error("invalid job identifier or repeated arm")
    out = llr.OUT / "next"
    out.mkdir(exist_ok=True)
    q = pd.read_parquet(link.OUT / "queries.parquet")
    with np.load(link.OUT / "scores.npz") as source:
        z = dict(source)
    arm = json.loads((link.OUT / "result.json").read_text())["selected_arm"]
    with np.load(llr.OUT / "llr_g40.npz") as source:
        cache = dict(source)
    original = llr.reranked(MU, z, arm, cache)
    rows, top, edges = link_table(q, original, z, cache)
    variants = {"opp7": (rows, top, False)}
    reports, rules = {}, {}
    if "colour" in args.arms:
        _, adjusted, reports["colour"] = reranks(q, z, arm, cache, colourwise=True)
        t, v, _ = link_table(q, adjusted, z, cache)
        variants["colour"] = (t, v, False)
    if "delta" in args.arms:
        variants["delta"] = (rows, top, True)
        counts = rows[rows["set"] == "s_opp"].groupby(["hd", "oc", "bin"]).size().reset_index(name="games")
        reports["delta"] = {"prior_games": 9.5, "bin_counts": counts.to_dict("records")}
    if "rich" in args.arms:
        with np.load(llr.OUT / "llr_g40_rich.npz") as source:
            rich = dict(source)
        if not all(np.array_equal(rich[k], cache[k]) for k in ("masked_rows", "seen_rows", "qrows")):
            raise ValueError("rich LLR queries do not match opp7")
        baseline, adjusted, reports["rich"] = reranks(q, z, arm, rich)
        reports["rich"]["ref_b_rich_mu2"] = reports["rich"].pop("opp7_ref_b")
        _, _, original_report = reranks(q, z, arm, cache)
        reports["rich"]["opp7_ref_b"] = original_report["opp7_ref_b"]
        t, v, _ = link_table(q, adjusted, z, rich)
        variants["rich"] = (t, v, False)
    fit, eval_games, x_fit, x_eval = prod.train_data()
    if "confidence" in args.arms:
        cal_rows, cal_top = add_calibration_games(q, rows, top, edges, arm, cache)
        rules["confidence"], reports["confidence"] = calibrate(fit, x_fit, cal_rows, cal_top, rows[rows["set"] == "s_opp"])
        variants["confidence"] = (rows, top, False)
    sets = prod.question_sets()
    evidence = {}
    for name, (t, v, smooth) in variants.items():
        kernel = rf._delta_prob(t[t["set"] == "s_opp"], smooth=smooth)
        evidence[name] = {split: question_evidence(qs, t[t["set"] == "u_opp"], v, kernel) for split, qs in sets.items()}
    scores = evaluate(sets, fit, eval_games, x_fit, x_eval, evidence, rules)
    for name in args.arms:
        by_seed, mean = deltas(scores[name], scores["opp7"])
        passed = (mean["E1"]["equal_k"] >= 0 and mean["E2"]["equal_k"] >= 0.002
                  and all(mean["E2"][str(k)] >= -0.005 for k in K_BUCKETS))
        report = {"arm": name, "reference": BASE.name, "baseline_scores": scores["opp7"], "details": reports[name],
                  "delta_by_seed": by_seed, "mean_delta": mean, "pass": passed}
        path = out / f"{args.job}_{name}.json"
        if passed:
            try:
                report["export"] = exports(args.job, name, *variants[name], rules.get(name))
            except Exception as error:
                report["export_error"] = str(error)
                raise
            finally:
                path.write_text(json.dumps(report, indent=2, default=int) + "\n")
        else:
            path.write_text(json.dumps(report, indent=2, default=int) + "\n")
        print(json.dumps({"report": str(path), "mean_delta": mean, "pass": passed}), flush=True)


if __name__ == "__main__":
    main()
