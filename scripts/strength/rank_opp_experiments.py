"""Independent opponent-link experiments sharing the production B2 and opp7 control."""
from __future__ import annotations

import argparse
import json
import re
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import softmax

from go_aicup.metrics import K_BUCKETS, rank_gains, score_rank
from scripts.strength import rank_ogs
from scripts.strength import rank_opp_next as n

TAGS = {"ablation": "oown-l10", "consensus": "ocns-l10", "cross": "oxg-l10"}
CANDIDATE = {"ablation": "own_opp", "consensus": "consensus", "cross": "cross"}


def own_features(fit, x_fit, x_eval):
    oof, val, _ = rank_ogs.yugen_oof(fit, x_fit, x_eval, own_only=True)
    def combine(x, y):
        return np.column_stack([x[:, :15], x[:, 30:40], x[:, 50:52], y])
    return combine(x_fit, oof), combine(x_eval, val)


def group_evidence(games, per_game, rows, top):
    """Return [all log evidence, majority voters' evidence, majority rank, distinct votes]."""
    if len(per_game) != len(rows) or not per_game.index.is_unique:
        raise ValueError("question game evidence is not one-to-one")
    ll = per_game.to_numpy()
    identity = top["idx"][rows.index.to_numpy(), 0]
    confident = rows["bin"].to_numpy() >= 1
    result = np.zeros((len(games), 22))
    for i, g in enumerate(games):
        pos = per_game.index.get_indexer(g)
        pos = pos[pos >= 0]
        result[i, :10] = ll[pos].sum(axis=0)
        pos = np.unique(pos[confident[pos]])
        unique, inverse, counts = np.unique(identity[pos], return_inverse=True, return_counts=True)
        if len(unique) < 2:
            continue
        by_identity = np.zeros((len(unique), 10))
        np.add.at(by_identity, inverse, ll[pos])
        by_identity /= counts[:, None]
        votes = score_rank(softmax(by_identity, axis=1))
        rank = np.bincount(votes, minlength=10).argmax()
        support = votes == rank
        if support.sum() >= 2 and 2 * support.sum() > len(votes):
            result[i, 10:20] = by_identity[support].sum(axis=0)
            result[i, 20:] = [rank, support.sum()]
    return result


def consensus_predict(p, evidence, *, tolerance):
    baseline = n.predict(p, evidence[:, :10])
    candidate = n.predict(p, evidence[:, 10:20])
    gain = rank_gains(p)
    j = np.arange(len(p))
    regret = gain[j, baseline] - gain[j, candidate]
    accepted = (evidence[:, 21] >= 2) & (candidate == evidence[:, 20]) & (regret <= tolerance)
    return np.where(accepted, candidate, baseline)


def game_evidence(questions, rows, top, kernel, *, consensus=False):
    game_ll = pd.DataFrame(n.rf.soft_ll(kernel, rows, top, n.TAU), index=rows["game_index"].to_numpy())
    if consensus:
        return group_evidence(questions["games"], game_ll, rows, top)
    return n.fu.question_ll(questions["games"], game_ll)


def fit_consensus(fit, x_fit, cal_rows, cal_top, delta_rows):
    eligible = fit[(fit["role"] == "S") & fit["game_index"].isin(cal_rows["game_index"])]
    q, fold = n.prod.calibration_questions(eligible, set())
    p = np.full((len(q), 10), np.nan)
    ev = np.full((len(q), 22), np.nan)
    for i in sorted(fold.unique()):
        held = set(fold.index[fold == i])
        valid = fit["player_id"].astype(str).isin(held).to_numpy()
        train = fit.loc[~valid]
        oof, (seen,) = n.nl.oof("yH", train, x_fit[~valid], [x_fit[valid]])
        model = n.prod._fit(train, np.column_stack([x_fit[~valid], oof]), n.more.player_prior(train[train["role"] == "S"]), 2027)
        mask = q["inner"] == i
        games = fit.loc[valid, "game_index"].to_numpy()
        p[mask] = model.probabilities(np.column_stack([x_fit[valid], seen]), games, q[mask])
        kernel = n.rf._delta_prob(delta_rows[~delta_rows["owner"].astype(str).isin(held)])
        ev[mask] = game_evidence(q[mask], cal_rows[cal_rows["game_index"].isin(games)], cal_top, kernel, consensus=True)
        print(f"consensus OOF fold {i}: {mask.sum()} questions", flush=True)
    if not (np.isfinite(p).all() and np.isfinite(ev).all()):
        raise ValueError("incomplete out-of-fold consensus features")
    truth, k = q["rank_id"].to_numpy(), q["k"].to_numpy()
    base = n.predict(p, ev[:, :10])
    base_score = n.qm.official(base, truth)
    reports = {}
    for tolerance in (0.01, 0.03, 0.05):
        pred = consensus_predict(p, ev, tolerance=tolerance)
        diff = n.qm.official(pred, truth) - base_score
        reports[tolerance] = {"equal_k": float(n.hf.equal_k(diff, k)), "changed": int((pred != base).sum())}
    selected = max(reports, key=lambda t: reports[t]["equal_k"])
    return {"questions": len(q), "players": int(q["player_id"].nunique()), "threshold": selected, "grid": reports}


def score(questions, models, x, eval_index, rows, top, *, mode, candidate_data=None, tolerance=None):
    s = rows[rows["set"] == "s_opp"]
    u = rows[rows["set"] == "u_opp"]
    ll = game_evidence(questions, u, top, n.rf._delta_prob(s), consensus=mode == "consensus")
    if candidate_data is not None:
        r, t = candidate_data
        alt = game_evidence(questions, r[r["set"] == "u_opp"], t, n.rf._delta_prob(r[r["set"] == "s_opp"]))
    result = []
    truth = questions["rank_id"].to_numpy()
    for seed, (full, own) in models.items():
        p = full.probabilities(x[0], eval_index, questions)
        original = n.predict(p, ll[:, :10])
        preds = {"b2": score_rank(p), "opp7": original}
        if mode == "ablation":
            pp = own.probabilities(x[1], eval_index, questions)
            preds.update(own=score_rank(pp), own_opp=n.predict(pp, ll))
        elif mode == "consensus":
            preds["consensus"] = consensus_predict(p, ll, tolerance=tolerance)
        else:
            preds["cross"] = n.predict(p, alt)
        for arm, pred in preds.items():
            result.append(pd.DataFrame({"seed": seed, "arm": arm, "question_row": np.arange(len(questions)),
                                        "k": questions["k"].to_numpy(), "truth": truth, "rank": pred,
                                        "score": n.qm.official(pred, truth), "opp7_rank": original,
                                        "delta": n.qm.official(pred, truth) - n.qm.official(original, truth)}))
    return pd.concat(result, ignore_index=True)


def summarize(frame, mode):
    raw = frame.groupby(["arm", "seed", "k"])["score"].mean().groupby(level=["arm", "k"]).mean().unstack("k")
    raw = raw.loc[:, list(K_BUCKETS)]
    raw["equal_k"] = raw.mean(axis=1)
    absolute = {a: {str(k): float(v) for k, v in r.items()} for a, r in raw.iterrows()}
    candidate = CANDIDATE[mode]
    diff = raw.loc[candidate] - raw.loc["opp7"]
    result = {"absolute": absolute, "candidate_delta": {str(k): float(v) for k,v in diff.items()}}
    if mode == "ablation":
        for name, a, b in (("full_opp_gain", "opp7", "b2"), ("own_opp_gain", "own_opp", "own")):
            result[name] = {str(k): float(v) for k,v in (raw.loc[a] - raw.loc[b]).items()}
        result["gain_difference"] = {k: result["own_opp_gain"][k] - v for k,v in result["full_opp_gain"].items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=tuple(TAGS))
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.job):
        parser.error("invalid job identifier")
    mode = args.mode
    path = n.llr.OUT / "next" / f"{args.job}_{mode}.json"
    path.parent.mkdir(exist_ok=True)
    report = {"mode": mode, "reference": n.BASE.name}

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(path)

    q = pd.read_parquet(n.link.OUT / "queries.parquet")
    arm = json.loads((n.link.OUT / "result.json").read_text())["selected_arm"]
    candidate_data = None
    with np.load(n.link.OUT / "scores.npz") as z, np.load(n.llr.OUT / "llr_g40.npz") as f:
        cache = dict(f)
        rr = n.llr.reranked(n.MU, z, arm, cache)
        rows, top, edges = n.link_table(q, rr, z, cache)
        if mode == "cross":
            from scripts.strength import rank_opp_cross_game as cross

            updated, report["pairs"] = cross.run(q, z, rr, rows, edges, cache)
            save()
            print(json.dumps(report), flush=True)
            if updated is None:
                return
            t, v, _ = n.link_table(q, updated, z, cache)
            candidate_data = (t, v)
    fit, eval_games, xf, xe = n.prod.train_data()
    if mode == "consensus":
        cal_rows, cal_top = n.add_calibration_games(q, rows, top, edges, arm, cache)
        report["oof"] = fit_consensus(fit, xf, cal_rows, cal_top, rows[rows["set"] == "s_opp"])
        save()
        print(json.dumps(report["oof"]), flush=True)
    oof, (val,) = n.nl.oof("yH", fit, xf, [xe])
    xx = np.column_stack([xf, oof]), np.column_stack([xe, val])
    xo = own_features(fit, xf, xe) if mode == "ablation" else (None, None)
    prior = n.more.player_prior(fit[fit["role"] == "S"])
    models = {}
    for seed in n.prod.SEEDS:
        models[seed] = (n.prod._fit(fit, xx[0], prior, seed),
                        n.prod._fit(fit, xo[0], prior, seed) if mode == "ablation" else None)
        print(f"fitted seed {seed}", flush=True)
    pred = partial(score, models=models, x=(xx[1], xo[1]), eval_index=eval_games["game_index"].to_numpy(),
                   rows=rows, top=top, mode=mode, candidate_data=candidate_data,
                   tolerance=report.get("oof", {}).get("threshold"))

    def run(split, questions):
        frame = pred(questions)
        frame.to_parquet(path.with_name(f"{path.stem}_{split}.parquet"), index=False)
        report[split] = summarize(frame, mode)
        save()
        print(split, json.dumps(report[split]), flush=True)
        return report[split]["candidate_delta"]

    e1 = pd.read_parquet(n.hf.Q1000)
    report["pass"] = False
    if run("E1", e1[e1["role"] == "U"].reset_index(drop=True))["equal_k"] >= 0:
        e2 = run("E2", pd.read_parquet(n.hf.OUT / "questions_e2_hd01.parquet"))
        report["pass"] = e2["equal_k"] >= 0.002 and all(e2[str(k)] >= -0.005 for k in K_BUCKETS)
        save()
        if report["pass"]:
            t, v = (rows, top) if candidate_data is None else candidate_data
            options = {}
            if mode == "ablation":
                options["feature_builder"] = own_features
            if mode == "consensus":
                options |= {"group_evidence": group_evidence,
                            "decoder": partial(consensus_predict, tolerance=report["oof"]["threshold"])}
            report["export"] = n.exports(args.job, mode, t, v, False, None, tag=TAGS[mode], **options)
            save()


if __name__ == "__main__":
    main()
