"""Opponent identity link v2, stage 1: fuse the per-game opponent links into the 839169 B2 posterior.
(2026-11-23; runs only if stage 0 passes; RESULTS.md, registered before the E1 / E2 numbers are seen)

Per queried game g (opponent side, HD<=1) stage 0 gives the top-1 training player j(g) under the selected arm and
the top-1 minus top-2 gap. The evidence is the pair (gap bin b(g), linked rank r_j(g)), binned on the reference
null quantiles of the gap (per queried colour). Its likelihood given the owner's rank r is a lookup table estimated
on Valid-S owners only (s_opp):
    P(b, r_j | r, hd, owner colour) = (n + a * m) / (N + a),  m = the same table pooled over r,  a = SMOOTH
    l_q(r) = sum_g [log P(b_g, r_j | r, hd_g, oc_g) - log m(b_g, r_j | hd_g, oc_g)]
    P(r | q) ~ P_B2(r | q) * exp(alpha * l_q(r)),  decoded with score_rank.
Arms: TABLE (the lookup above) or DELTA (pooled histogram of r_j - r per bin / hd / oc, normalised per r; added
before E1 / E2 because the high bins of TABLE are sparse), each with ALL (every bin) or HIGH (only bins at or above
the null q99). alpha in ALPHAS. Selection on E1 (five-seed
mean Equal-k), report E2 at the selected arm / alpha; gate: E1 and E2 five-seed mean Equal-k both >= +0.005 over
alpha = 0 -> export on the 839169 recipe (base arm must reproduce 839169-ygbm-l10.csv; merged groups add the
member questions' l_q; L10 last), table re-estimated on s_opp + u_opp for the export.

    python -m scripts.strength.rank_opp_fuse --job <jobid>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import k_bucket
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as more
import scripts.strength.rank_nonlin as nl
import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_prod_a0_fusion as prod
import scripts.strength.rank_qmerge as qm
import scripts.strength.rank_stack as rs

OUT = ROOT / "outputs" / "rank" / "opp_link2"
SMOOTH = 20.0
ALPHAS = (0.25, 0.5, 0.75, 1.0)
QS = (0.9, 0.99, 0.999)          # bin edges on the reference null gap; last bin = above the null max
GAIN = 0.005


def link_table() -> pd.DataFrame:
    q = pd.read_parquet(OUT / "queries.parquet")
    z = np.load(OUT / "scores.npz")
    arm = json.loads((OUT / "result.json").read_text())["selected_arm"]
    val, idx = z[f"{arm}_val"], z[f"{arm}_idx"]
    gap = val[:, 0] - val[:, 1]
    colour = q["qcolor"].to_numpy()
    ref = (q["set"] == "ref").to_numpy()
    edges = {c: [float(np.quantile(gap[ref & (colour == c)], x)) for x in QS] + [float(gap[ref & (colour == c)].max())]
             for c in (0, 1)}
    b = np.zeros(len(q), np.int64)
    for c in (0, 1):
        m = colour == c
        b[m] = np.searchsorted(np.array(edges[c]), gap[m], side="right")
    out = q.assign(bin=b, rj=z["cand_rank"][idx[:, 0]], gap=gap, oc=(1 - colour).astype(np.int64))
    out.attrs["edges"], out.attrs["arm"] = edges, arm
    return out


N_BIN = len(QS) + 2


def fit_table(t: pd.DataFrame) -> np.ndarray:
    """log P(bin, r_j | r, hd, oc) - log m(bin, r_j | hd, oc): [2 hd, 2 oc, 10 r, N_BIN, 10 r_j]."""
    n = np.zeros((2, 2, 10, N_BIN, 10))
    np.add.at(n, (t["hd"].to_numpy(), t["oc"].to_numpy(), t["owner_rank"].to_numpy(), t["bin"].to_numpy(),
                  t["rj"].to_numpy()), 1.0)
    pooled = n.sum(axis=2, keepdims=True)
    m = (pooled + 1e-3) / (pooled + 1e-3).sum(axis=(3, 4), keepdims=True)
    p = (n + SMOOTH * m) / (n.sum(axis=(3, 4), keepdims=True) + SMOOTH)
    return np.log(p) - np.log(m)


def fit_delta(t: pd.DataFrame) -> np.ndarray:
    """translation-invariant version: P(r_j | r, bin, hd, oc) from the pooled histogram of r_j - r (+0.5 each
    offset), normalised over r_j = 0..9 for each r; same layout and r-free offset as fit_table."""
    c = np.zeros((2, 2, N_BIN, 19))
    np.add.at(c, (t["hd"].to_numpy(), t["oc"].to_numpy(), t["bin"].to_numpy(),
                  t["rj"].to_numpy() - t["owner_rank"].to_numpy() + 9), 1.0)
    c += 0.5
    r, rj = np.arange(10)[:, None], np.arange(10)[None, :]
    p = c[..., rj - r + 9]                                  # [2, 2, bin, r, rj]
    p = p / p.sum(-1, keepdims=True)
    p = np.moveaxis(p, 3, 2)                                # [2, 2, r, bin, rj]
    return np.log(p) - np.log(p.mean(axis=2, keepdims=True))


def game_ll(table: np.ndarray, t: pd.DataFrame, high: bool) -> np.ndarray:
    """[n, 10] per-game evidence; rows below the q99 bin are zero for HIGH."""
    ll = table[t["hd"].to_numpy(), t["oc"].to_numpy(), :, t["bin"].to_numpy(), t["rj"].to_numpy()]
    if high:
        ll = np.where((t["bin"].to_numpy() >= 2)[:, None], ll, 0.0)
    return ll


def question_ll(games: pd.Series, per_game: pd.DataFrame) -> np.ndarray:
    """sum of the per-game rows (indexed by game key) over each question's games; missing games add 0."""
    out = np.zeros((len(games), 10))
    for i, g in enumerate(games):
        keys = [x for x in g if x in per_game.index]
        if keys:
            out[i] = per_game.loc[keys].to_numpy().sum(0)
    return out


def posterior(p: np.ndarray, ll: np.ndarray, alpha: float) -> np.ndarray:
    logit = np.log(np.clip(p, 1e-12, 1)) + alpha * ll
    logit -= logit.max(1, keepdims=True)
    e = np.exp(logit)
    return e / e.sum(1, keepdims=True)


def local(t: pd.DataFrame) -> dict:
    fit, eval_games, x_fit, x_eval = prod.train_data()
    o, (e,) = nl.oof("yH", fit, x_fit, [x_eval])
    xf, xe = np.hstack([x_fit, o]), np.hstack([x_eval, e])
    prior = more.player_prior(fit[fit["role"] == "S"])
    tables = {"TABLE": fit_table(t[t["set"] == "s_opp"]), "DELTA": fit_delta(t[t["set"] == "s_opp"])}
    u = t[t["set"] == "u_opp"]
    sets = prod.question_sets()
    arms = {f"{m}-{b}": (m, b == "HIGH") for m in tables for b in ("ALL", "HIGH")}
    per_game = {a: pd.DataFrame(game_ll(tables[m], u, h), index=u["game_index"].to_numpy()) for a, (m, h) in arms.items()}
    qll = {a: {n: question_ll(q["games"], per_game[a]) for n, q in sets.items()} for a in arms}
    coverage = {n: {str(b): float((np.abs(qll["TABLE-HIGH"][n][q["k"].to_numpy() == b]).sum(1) > 0).mean()) for b in K_BUCKETS}
                for n, q in sets.items()}
    seeds = {}
    for seed in prod.SEEDS:
        model = prod._fit(fit, xf, prior, seed)
        entry = {}
        for name, q in sets.items():
            k, truth = q["k"].to_numpy(), q["rank_id"].to_numpy()
            p = model.probabilities(xe, eval_games["game_index"].to_numpy(), q)
            base = qm.official(score_rank(p), truth)
            row = {"base": hf.by_k(base, k)}
            for a in arms:
                for alpha in ALPHAS:
                    s = qm.official(score_rank(posterior(p, qll[a][name], alpha)), truth)
                    row[f"{a}_{alpha:g}"] = {**{str(b): round(float(s[k == b].mean() - base[k == b].mean()), 4)
                                                for b in K_BUCKETS},
                                             "equal_k": round(hf.equal_k(s, k) - hf.equal_k(base, k), 4)}
            entry[name] = row
        seeds[str(seed)] = entry
        print(seed, json.dumps({n: {c: v["equal_k"] for c, v in r.items()} for n, r in entry.items()}), flush=True)
    cells = [c for c in seeds["2027"]["E1"] if c != "base"]
    mean = {n: {c: {key: round(float(np.mean([seeds[str(s)][n][c][key] for s in prod.SEEDS])), 4)
                    for key in [str(b) for b in K_BUCKETS] + ["equal_k"]} for c in cells} for n in sets}
    chosen = max(cells, key=lambda c: mean["E1"][c]["equal_k"])
    passed = mean["E1"][chosen]["equal_k"] >= GAIN and mean["E2"][chosen]["equal_k"] >= GAIN
    return {"arm": t.attrs["arm"], "edges": t.attrs["edges"], "coverage_high": coverage, "seeds": seeds,
            "mean_vs_base": mean, "chosen": chosen, "chosen_E1": mean["E1"][chosen], "chosen_E2": mean["E2"][chosen],
            "pass": bool(passed)}


def export(t: pd.DataFrame, cell: str, job: str, *, evidence=None, directory=None, tag="opp-l10",
           alpha_rule=None, feature_builder=None, decoder=None) -> dict:
    from go_aicup.data.preprocess import load_store
    arm, alpha = cell.split("_")
    alpha = float(alpha)
    model_name, bins = arm.split("-")
    fit, eval_games, x_fit, x_eval = prod.train_data()
    u = (eval_games["handicap"] <= 1).to_numpy()
    pool = pd.concat([fit, eval_games[u]], ignore_index=True)
    x_pool = np.vstack([x_fit, x_eval[u]])
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    table = (fit_table if model_name == "TABLE" else fit_delta)(t[t["set"].isin(["s_opp", "u_opp"])])
    test = t[t["set"] == "test"].copy()
    xs, gis, qs, keys, shift = [], [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        work = prod.RELEASE / f"rank_{part}"
        smeta = pd.read_parquet(work / "store_meta.parquet")
        cache = pd.read_parquet(work / "cache" / "games.parquet").set_index("game_index")
        rel = smeta.iloc[cache.loc[gi, "store_row"].to_numpy()][["question_id", "slot", "color"]].reset_index(drop=True)
        _, _, om = load_store(f"rank_{part}", ol.STORE, mmap=False)
        om = om.reset_index(drop=True).assign(store_row=np.arange(len(om)))
        j = rel.merge(om[["question_id", "slot", "color", "store_row"]], on=["question_id", "slot"], how="left",
                      validate="one_to_one", suffixes=("", "_o"))
        if j["store_row"].isna().any() or (j["color"].astype(str) != j["color_o"].astype(str)).any():
            raise ValueError(f"{part}: release games do not map onto the opp_link store")
        sub = test[test["part"] == f"rank_{part}"].set_index("store_row").loc[j["store_row"].to_numpy()].reset_index()
        sub["hd"] = (x[:, 50:52].sum(1) > 0).astype(np.int64)        # HD1 flags of the test x
        sub["key"] = gi + shift
        keys.append(sub)
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        xs.append(x), gis.append(gi + shift), qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True)
    sub = pd.concat(keys, ignore_index=True)
    if evidence is None:
        per_game = pd.DataFrame(game_ll(table, sub, bins == "HIGH"), index=sub["key"].to_numpy())
        values = question_ll(questions["games"], per_game)
    else:
        values = evidence(questions["games"], sub, table, "key")
    qll = pd.DataFrame(values, index=questions["question_id"].to_numpy())

    o, (tt,) = nl.oof("yH", pool, x_pool, [x])
    model = prod._fit(pool, np.hstack([x_pool, o]), prior, 2027)
    xt = np.hstack([x, tt])
    groups = qm.groups_from_pairs(pd.read_csv(qm.PAIRS))
    by_id = questions.set_index("question_id")
    l10 = pd.read_csv(qm.L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    current = pd.read_csv(rs.CURRENT, dtype=str).set_index("question_id")["pred_rank"]

    def apply(p, ll, a):
        if a and decoder is not None:
            return decoder(p, ll)
        weight = alpha_rule(p)[:, None] if a and alpha_rule is not None else a
        return score_rank(posterior(p, ll[:, :len(RANKS)], weight))

    def predict(a: float, model, xt) -> pd.Series:
        pred = apply(model.probabilities(xt, gi, questions), qll.loc[questions["question_id"]].to_numpy(), a)
        out = pd.Series([RANKS[r] for r in pred], index=questions["question_id"])
        members = [[np.asarray(by_id.loc[qq, "games"]) for qq in g] for g in groups]
        pm = qm.merged_probabilities(model, xt, gi, members, [k_bucket(sum(len(m) for m in ms)) for ms in members])
        if evidence is None:
            merged_ll = [qll.loc[g].to_numpy().sum(0) for g in groups]
        else:
            merged_games = pd.Series([np.concatenate(m) for m in members])
            merged_ll = evidence(merged_games, sub, table, "key")
        for g, prob, ll in zip(groups, pm, merged_ll):
            pg = apply(prob[None], ll[None], a)
            for qq in g:
                out[qq] = RANKS[int(pg[0])]
        out.loc[l10.index] = l10
        return out.loc[current.index]

    base = predict(0.0, model, xt)
    if (base != current).any():
        raise ValueError(f"alpha 0 does not reproduce 839169-ygbm-l10.csv ({int((base != current).sum())})")
    if feature_builder is not None:
        xf, xt = feature_builder(pool, x_pool, x)
        model = prod._fit(pool, xf, prior, 2027)
    pred = predict(alpha, model, xt)
    diff = pred != current
    kq = questions.set_index("question_id")["num_games"].loc[current.index]
    report = {"cell": cell, "public": int((diff & diff.index.str.startswith("pub")).sum()),
              "private": int((diff & diff.index.str.startswith("priv")).sum()),
              "public_by_k": {str(b): int((diff & diff.index.str.startswith("pub") & (kq == b)).sum()) for b in K_BUCKETS},
              "private_by_k": {str(b): int((diff & diff.index.str.startswith("priv") & (kq == b)).sum()) for b in K_BUCKETS},
              "test_high_games": int((sub["bin"] >= 2).sum()), "test_games": int(len(sub))}
    kwargs = {} if directory is None else {"directory": directory}
    report["file"] = write_rank_csv(pred.rename("pred_rank").reset_index(), tag, job, **kwargs).name
    print(json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    parser.add_argument("--export", help="skip the local check and export this cell (e.g. HIGH_0.5)")
    args = parser.parse_args()
    t = link_table()
    if args.export:
        report = {"export": export(t, args.export, args.job)}
    else:
        report = {"local": local(t)}
        print(json.dumps({k: v for k, v in report["local"].items() if k != "seeds"}, indent=1), flush=True)
        if report["local"]["pass"]:
            report["export"] = export(t, report["local"]["chosen"], args.job)
        else:
            print("gate not passed: no file", flush=True)
    (OUT / f"fuse_{args.job}.json").write_text(json.dumps(report, indent=1, default=int))


if __name__ == "__main__":
    main()
