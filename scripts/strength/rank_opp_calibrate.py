"""Calibrate cached opponent evidence against DELTA-ALL alpha 0.25."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

from scripts.strength import rank_more_players as more
from scripts.strength import rank_nonlin as nl
from scripts.strength import rank_opp_fuse as fuse
from scripts.strength import rank_prod_a0_fusion as prod
from scripts.strength import rank_qmerge as qm

REFERENCE = "delta_0.25"
HISTORY = fuse.OUT / "fuse-843387.out"
OPP_CSV = ROOT / "submissions/rank/843411-opp-l10.csv"
FOLDS = ROOT / "data/processed/cv/strength_folds_seed2027.parquet"


def grouped_evidence(games, rows, key, likelihood, candidates=None):
    index = pd.Index(rows[key])
    assert index.is_unique
    out = np.zeros((len(games), 10))
    for i, ids in enumerate(games):
        loc = index.get_indexer(ids)
        loc = loc[loc >= 0]
        values = likelihood[loc]
        if candidates is not None and len(loc):
            _, inverse, counts = np.unique(candidates[loc], return_inverse=True, return_counts=True)
            values = values / counts[inverse, None]
        out[i] = values.sum(axis=0)
    return out


def match_features(scores, color, hd):
    return np.column_stack((scores[:, 0], scores[:, 0] - scores[:, 1],
                            scores[:, 0] - scores[:, -1], color, hd))


def calibrate(t, scores, arm):
    ids, val = (scores[f"{arm}_{name}"] for name in ("idx", "val"))
    candidate_ranks = scores["cand_rank"]
    assert ids.shape == val.shape == (len(t), 10)
    assert ((ids >= 0) & (ids < len(candidate_ranks))).all() and np.isfinite(val).all()
    assert np.all(val[:, :-1] >= val[:, 1:])
    owner_col = scores["owner_col"]
    folds = pd.read_parquet(FOLDS).set_index("player_id")["fold"]
    ref = t[(t["set"] == "ref") & t["owner"].map(folds).gt(0)]
    players = sorted(ref["owner"].unique())
    rng = np.random.default_rng(2037)
    fit_players = set(rng.permutation(players)[:len(players) // 2])
    ref_rows = ref.index.to_numpy()
    is_fit = ref["owner"].isin(fit_players).to_numpy()
    color, hd = ref["qcolor"], ref["hd"]
    seen_ids, seen = (scores[f"seen_{arm}_{name}"][ref_rows] for name in ("idx", "val"))
    true_in_top = seen_ids == owner_col[ref_rows, None]
    assert np.all(true_in_top.sum(axis=1) <= 1) and np.isfinite(seen).all()
    hit = true_in_top.any(axis=1)
    xy = np.vstack([match_features(seen, color, hd), match_features(val[ref_rows], color, hd)])
    target = np.r_[hit, np.zeros(len(hit), dtype=bool)]
    fit = np.tile(is_fit, 2)
    classifier = make_pipeline(StandardScaler(), LogisticRegression()).fit(xy[fit], target[fit])

    def loss(log_temp, mask):
        p = softmax(seen[mask] / np.exp(log_temp), axis=1)[true_in_top[mask]]
        return float(-np.log(np.maximum(1e-12, p)).mean())

    optimum = minimize_scalar(loss, args=(is_fit & hit,), bounds=(-3, 7), method="bounded")
    if not optimum.success:
        raise RuntimeError("candidate temperature fit failed")
    log_temp = optimum.x
    q = classifier.predict_proba(xy[~fit])[:, 1]
    py = target[~fit]
    calibration = {
        "fit_players": len(fit_players), "held_players": len(players) - len(fit_players),
        "fit_examples": int(fit.sum()), "held_examples": int((~fit).sum()),
        "held_prevalence": float(py.mean()),
        "held_log_loss": log_loss(py, q),
        "held_constant_log_loss": log_loss(py, np.full(len(py), target[fit].mean())),
        "held_conditional_log_loss": loss(log_temp, ~is_fit & hit),
        "held_conditional_uniform_log_loss": float(np.log(ids.shape[1])),
        "temperature": float(np.exp(log_temp)),
        "held_reliability": [{"n": len(ix), "predicted": float(q[ix].mean()), "actual": float(py[ix].mean())}
                             for ix in np.array_split(np.argsort(q), 5)],
    }

    def evidence(games, rows, table, key, mode):
        source = rows["link_row"].to_numpy()
        candidates = ids[source, 0] if mode in ("repeat", "combined") else None
        if mode in ("multi", "combined"):
            strength = val[source]
            confidence = classifier.predict_proba(match_features(strength, rows["qcolor"], rows["hd"]))[:, 1]
            weights = softmax(strength / np.exp(log_temp), axis=1)
            rank = candidate_ranks[ids[source]]
            likelihood_by_rank = table[rows["hd"], rows["oc"], :, rows["bin"], :]
            selected = np.take_along_axis(likelihood_by_rank, rank[:, None, :], axis=2)
            matched = np.einsum("ng,nrg->nr", weights, np.exp(selected))
            ll = np.log1p(confidence[:, None] * (matched - 1.0))
        else:
            ll = fuse.game_ll(table, rows, high=False)
        assert np.isfinite(ll).all()
        return grouped_evidence(games, rows, key, ll, candidates)

    return evidence, calibration


def score(questions, posteriors, prior):
    truth = questions["rank_id"].to_numpy()
    k = questions["k"].to_numpy()
    decoded = np.stack([score_rank(p) for p in posteriors])
    scores = qm.official(decoded, truth)
    bases = qm.official(np.stack([score_rank(p) for p in prior]), truth)
    delta = scores - bases
    dist = np.abs(decoded - truth)
    return delta, {
        "bucket": {str(b): float(delta[:, k == b].mean()) for b in K_BUCKETS},
        "equal_k": float(np.mean([delta[:, k == b].mean() for b in K_BUCKETS])),
        "seeds": {str(seed): {str(b): float(delta[i, k == b].mean()) for b in K_BUCKETS}
                  for i, seed in enumerate(prod.SEEDS)},
        "counts_by_seed": {str(seed): {str(b): {
            "exact": int((dist[i, k == b] == 0).sum()), "adjacent": int((dist[i, k == b] == 1).sum())}
            for b in K_BUCKETS} for i, seed in enumerate(prod.SEEDS)},
    }


def interval(questions, delta, *, buckets=K_BUCKETS):
    _, owner = np.unique(questions["player_id"], return_inverse=True)
    k = questions["k"].to_numpy()
    n = len(np.unique(owner))
    count = np.zeros((n, len(buckets)))
    totals = np.zeros_like(count)
    for i, b in enumerate(buckets):
        mask = k == b
        np.add.at(count[:, i], owner[mask], 1)
        np.add.at(totals[:, i], owner[mask], delta[:, mask].mean(axis=0))
    assert (count.sum(axis=0) > 0).all()
    weights = np.random.default_rng(2037).multinomial(n, np.full(n, 1 / n), size=2000)
    counts = weights @ count
    valid = (counts > 0).all(axis=1)
    draws = ((weights @ totals)[valid] / counts[valid]).mean(axis=1)
    return {"players": n, "questions_by_k": count.sum(axis=0).astype(int).tolist(),
            "draws": int(valid.sum()), "ci95": np.quantile(draws, (0.025, 0.975)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    out = fuse.OUT / "calibration" / args.job
    out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    t = fuse.link_table()
    t["link_row"] = np.arange(len(t))
    arm = t.attrs["arm"]
    assert arm == "f8" and t.groupby("set").size().to_dict() == {
        "ref": 23640, "s_opp": 20000, "test": 10000, "u_opp": 33656}
    with np.load(fuse.OUT / "scores.npz") as z:
        evidence, calibration = calibrate(t, z, arm)
    table = fuse.fit_delta(t[t["set"] == "s_opp"])
    evaluate = t[t["set"] == "u_opp"]

    def likelihood(q, mode):
        return evidence(q["games"], evaluate, table, "game_index", mode)

    fit, eval_games, x_fit, x_eval = prod.train_data()
    oof, (eval_h,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.column_stack([x_fit, oof]), np.column_stack([x_eval, eval_h])
    prior = more.player_prior(fit[fit["role"] == "S"])
    sets = prod.question_sets()
    assert not set(fit.player_id) & (set(sets["E1"].player_id) | set(sets["E2"].player_id))
    raw = {name: [] for name in sets}
    for seed in prod.SEEDS:
        model = prod._fit(fit, xf, prior, seed)
        for name, q in sets.items():
            raw[name].append(model.probabilities(xe, eval_games["game_index"].to_numpy(), q))
        print(seed, "baseline posteriors saved in memory", round(monotonic() - started, 1), flush=True)

    def blend(p, ll, alpha):
        return np.stack([fuse.posterior(current, ll, alpha) for current in p])

    e1, e2 = sets["E1"], sets["E2"]
    assert len(e1) == 1000 and len(e2) == 2000
    delta1 = likelihood(e1, "delta")
    post1 = {f"delta_{a:g}": blend(raw["E1"], delta1, a) for a in (0.25, 0.1875, 0.125, 0.0625)}
    post1.update({mode: blend(raw["E1"], likelihood(e1, mode), 0.25) for mode in ("repeat", "multi")})
    e1_summary = {name: score(e1, p, post1[REFERENCE])[1] for name, p in post1.items()}
    if all(e1_summary[mode]["equal_k"] > 0 for mode in ("repeat", "multi")):
        post1["combined"] = blend(raw["E1"], likelihood(e1, "combined"), 0.25)
        e1_summary["combined"] = score(e1, post1["combined"], post1[REFERENCE])[1]
    chosen = max(e1_summary, key=lambda name: e1_summary[name]["equal_k"])
    print("E1 selection", chosen, e1_summary[chosen]["equal_k"], flush=True)

    mode, alpha = (("delta", float(chosen.split("_")[1])) if chosen.startswith("delta_") else (chosen, 0.25))
    post2_ref = blend(raw["E2"], likelihood(e2, "delta"), 0.25)
    post2 = post2_ref if chosen == REFERENCE else blend(raw["E2"], likelihood(e2, mode), alpha)
    disjoint = ~e2["player_id"].isin(e1["player_id"]).to_numpy()
    assert int(disjoint.sum()) == 496
    dfull, full = score(e2, post2, post2_ref)
    dsub, subset = score(e2[disjoint], post2[:, disjoint], post2_ref[:, disjoint])
    history_lines = HISTORY.read_text().splitlines(keepends=True)
    legacy = json.JSONDecoder().raw_decode("".join(history_lines[5:]).lstrip())[0]
    reference = {}
    for name, p in (("E1", post1[REFERENCE]), ("E2", post2_ref)):
        differences, report = score(sets[name], p, raw[name])
        for i, seed in enumerate(prod.SEEDS):
            line = history_lines[i].split(" ", 1)
            assert int(line[0]) == seed
            previous = json.loads(line[1])[name]["DELTA-ALL_0.25"]
            k = sets[name]["k"].to_numpy()
            now = np.mean([differences[i, k == b].mean() for b in K_BUCKETS])
            assert abs(now - previous) <= 0.00005001, (name, seed, now, previous)
        for k, gain in report["bucket"].items():
            assert abs(gain - legacy["mean_vs_base"][name]["DELTA-ALL_0.25"][k]) <= 0.00011
        reference[name] = report

    passed = full["equal_k"] >= 0.002 and min(full["bucket"].values()) >= -0.005 and subset["equal_k"] > 0
    report = {"chosen": chosen, "passed": passed, "calibration": calibration,
              "reference_vs_b2": reference, "E1_vs_reference": e1_summary,
              "E2_vs_reference": {**full, **interval(e2, dfull)},
              "E2_without_E1_players": {**subset, **interval(e2[disjoint], dsub)},
              "chosen_vs_b2": {"E1": score(e1, post1[chosen], raw["E1"])[1],
                               "E2": score(e2, post2, raw["E2"])[1]}}
    for name, q in sets.items():
        q.to_parquet(out / f"{name}.parquet", index=False)
    np.savez_compressed(out / "posteriors.npz", E1_base=raw["E1"], E2_base=raw["E2"],
                        E2_reference=post2_ref, E2_chosen=post2, E2_disjoint=disjoint,
                        **{f"E1_{name}": p for name, p in post1.items()})
    for path in (fuse.OUT / "queries.parquet", fuse.OUT / "scores.npz", fuse.OUT / "result.json", HISTORY, OPP_CSV):
        with path.open("rb") as f:
            report.setdefault("sources", {})[str(path.relative_to(ROOT))] = hashlib.file_digest(f, "sha256").hexdigest()
    (out / "local.json").write_text(json.dumps(report, indent=2) + "\n")
    print("E2", full["equal_k"], full["bucket"], "disjoint", subset["equal_k"], "passed", passed, flush=True)

    repro = fuse.export(t, "DELTA-ALL_0.25", args.job, directory=out)
    regenerated = pd.read_csv(out / repro["file"], dtype=str)
    pd.testing.assert_frame_equal(regenerated, pd.read_csv(OPP_CSV, dtype=str))
    report["export_reproduction"] = {"rows": len(regenerated), "alpha_zero_matches_839169": True,
                                     "alpha_025_matches_843411": True}
    if passed:
        def new_evidence(games, rows, table, key):
            return evidence(games, rows, table, key, mode)

        report["export"] = fuse.export(t, f"DELTA-ALL_{alpha:g}", args.job, evidence=new_evidence, tag="opcl-l10")
        candidate = OPP_CSV.parent / report["export"]["file"]
        report["export"]["sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    report["seconds"] = monotonic() - started
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Completed", out, report["seconds"], flush=True)


if __name__ == "__main__":
    main()
