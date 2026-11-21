"""Competition 2: soft merge of same-player test questions with two fingerprints (2026-11-21, RANK_QMERGE_PLAN.md §7)

Same-colour question pairs get a same-player probability w from the style score s (rank_link2 vectors) and the
opening score m (rank_identity_merge mutual likelihood ratio): a logistic regression on reference pairs of the
1,182 encoder-unseen players gives a likelihood ratio LR(s, m, k); the share pi of same-player pairs among the
test pairs comes from EM, and w = pi LR / (pi LR + 1 - pi). Decision (expected-score mixture, score_rank):
  pairs with w >= W_HARD form hard groups; a member predicts (1 - wbar) p_self + wbar p_group
  (wbar = its largest w inside the group); a question outside groups with 0.05 <= w < 0.5 to some question
  predicts (1 - w) p_self + w p_pair with its strongest partner.

  pairs   CPU: reference and test pair tables (style + opening)
  run     calibration check on held-out reference players, Valid-U B2 decision simulation, export

    python -m scripts.strength.rank_softmerge pairs
    python -m scripts.strength.rank_softmerge run --job <jobid>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from go_aicup.data.board_cache import RANKS
from go_aicup.data.preprocess import load_store
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix
from go_aicup.tasks.strength.humansl import k_bucket
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.rank_link2 as l2
import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_qmerge as qm
from scripts.strength.l10_check import score_questions
from scripts.strength.rank_identity_merge import mutual

OUT = ROOT / "outputs" / "rank" / "softmerge"
SEED = 2033
W_HARD, W_MIN = 0.5, 0.05
W_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95)
PAIR_CELLS = ((5, 5), (5, 10), (10, 10), (10, 15), (15, 20))
SIM_SETS, SIM_PLAYERS, SIM_TWO = 50, 400, 20
W_BINS = (0.05, 0.2, 0.5, 0.8, 1.000001)


# ---------------------------------------------------------------- pair tables
def reference_questions(meta: pd.DataFrame, rng) -> pd.DataFrame:
    """Two disjoint single-colour questions per reference player (rank_link2 reference games + gallery games)."""
    ref = pd.read_parquet(l2.OUT / "ref.parquet")
    work = pd.read_parquet(ol.OUT / "plan.parquet")
    gal = work.loc[work["set"] == "gallery", "game_index"].to_numpy()
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    gal_by = pd.Series(gal).groupby(player[gal]).apply(np.asarray)
    rows = []
    for pid, g in ref.groupby("player_id", sort=True):
        colour = int(g["color"].iloc[0])
        mine = gal_by.get(pid, np.array([], np.int64))
        pool = rng.permutation(np.unique(np.concatenate([g["game_index"].to_numpy(), mine[pcolor[mine] == colour]])))
        k1 = int(rng.choice([k for k in K_BUCKETS if k <= len(pool) - 5]))
        k2 = int(rng.choice([k for k in K_BUCKETS if k <= len(pool) - k1]))
        for j, (part, k) in enumerate(((pool[:k1], k1), (pool[k1:k1 + k2], k2))):
            rows.append({"qid": f"{pid}_{j}", "player": pid, "colour": colour, "k": k, "games": part})
    return pd.DataFrame(rows)


def upper_pairs(ids, ks, s, m, players=None) -> pd.DataFrame:
    iu = np.triu_indices(len(ids), 1)
    out = pd.DataFrame({"a": np.asarray(ids)[iu[0]], "b": np.asarray(ids)[iu[1]], "s": s[iu], "m": m[iu],
                        "k1": np.asarray(ks)[iu[0]], "k2": np.asarray(ks)[iu[1]]})
    if players is not None:
        out["pa"], out["pb"] = np.asarray(players)[iu[0]], np.asarray(players)[iu[1]]
        out["same"] = out["pa"] == out["pb"]
    return out


def pairs(_args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    meta = ol.load_train()
    rng = np.random.default_rng(SEED)
    refq = reference_questions(meta, rng)
    refq.drop(columns="games").assign(games=refq["games"].map(list)).to_parquet(OUT / "ref_questions.parquet")

    train_games = np.load(l2.OUT / "train_games.npy")
    train_vec = l2.load_vectors("train", 4, len(train_games), None)
    pos = lambda g: np.searchsorted(train_games, g)
    if not all(np.isin(g, train_games).all() for g in refq["games"]):
        raise ValueError("reference games without style vectors")

    train_pre = data.opening_prefix_d4(meta, PLIES)
    blocks, offsets, size, tests = [train_pre], {}, len(train_pre), {}
    for name in ol.PARTS:
        moves, colors, q_meta = load_store(name, ol.STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        cache_dir, games, _, _, rows = l2.part_inputs(name, meta)
        tests[name] = (q_meta, ok, l2.load_vectors(name, 4, len(games), rows))
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
    prefix = np.concatenate(blocks)
    hashes = opening.prefix_hashes(prefix)
    hd01 = meta["handicap"].to_numpy() <= 1
    ref_players = meta["player_id"].astype(str).isin(set(refq["player"])).to_numpy()
    empty = np.array([], np.int64)
    ref_tables, test_tables = [], []
    for colour in (0, 1):
        own = (meta["player_color"].to_numpy() == colour) & hd01
        part = refq[refq["colour"] == colour].reset_index(drop=True)
        queries = list(part["games"])
        sc = score_questions(prefix, hashes, queries + [empty], queries, np.flatnonzero(own & ~ref_players), colour, params)
        m = mutual(sc)
        qv = l2.question_vectors(train_vec, [pos(g) for g in queries])
        s = l2.pair_scores(qv, qv)
        ref_tables.append(upper_pairs(part["qid"], part["k"], s, m, part["player"]).assign(colour=colour))
        print(f"colour {colour}: {len(part)} reference questions", flush=True)

        ids, ks, op_q, st_q = [], [], [], {n: [] for n in l2.ENCODERS}
        for name, (q_meta, ok, vec) in tests.items():
            index = np.flatnonzero(q_meta["color"].astype(str).to_numpy() == ("W" if colour else "B"))
            groups = q_meta.iloc[index].groupby("question_id", sort=False).indices
            for qid, r in groups.items():
                rows = index[r]
                ids.append(qid)
                ks.append(int(q_meta["num_games"].iloc[rows[0]]))
                op_q.append(offsets[name] + rows[ok[rows]])
                for n in l2.ENCODERS:
                    st_q[n].append(vec[n][rows].sum(0))
        qv = {n: l2.unit_rows(np.stack(v)) for n, v in st_q.items()}
        s = sum(l2.WEIGHTS[n] * (qv[n] @ qv[n].T) for n in l2.ENCODERS)
        sc = score_questions(prefix, hashes, op_q + [empty], op_q, np.flatnonzero(own), colour, params)
        test_tables.append(upper_pairs(ids, ks, s, mutual(sc)).assign(colour=colour))
        print(f"colour {colour}: {len(ids)} test questions", flush=True)
    pd.concat(ref_tables, ignore_index=True).to_parquet(OUT / "ref_pairs.parquet", index=False)
    pd.concat(test_tables, ignore_index=True).to_parquet(OUT / "test_pairs.parquet", index=False)
    print("pairs written", flush=True)


# ---------------------------------------------------------------- likelihood ratio and EM
def features(t: pd.DataFrame) -> np.ndarray:
    s = t["s"].to_numpy()
    m = np.clip(t["m"].to_numpy(), -200.0, 600.0) / 100.0
    lk = np.log(np.minimum(t["k1"], t["k2"]).to_numpy())
    lK = np.log(np.maximum(t["k1"], t["k2"]).to_numpy())
    return np.column_stack([s, s * s, m, s * lk, m * lk, lk, lK])


class PairModel:
    def fit(self, t: pd.DataFrame):
        x, y = features(t), t["same"].to_numpy()
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(C=1.0, max_iter=5000).fit(self.scaler.transform(x), y)
        self.offset = float(np.log(y.sum() / (~y).sum()))
        return self

    def log_lr(self, t: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self.scaler.transform(features(t))) - self.offset


def em_pi(log_lr: np.ndarray) -> float:
    """argmax_pi sum log(pi LR + 1 - pi) on [0, 0.5] (concave; bisection on the derivative)."""
    lr = np.exp(np.clip(log_lr, -50, 50))
    d = lambda p: float(np.sum((lr - 1) / (1 + p * (lr - 1))))
    if d(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if d(mid) > 0 else (lo, mid)
    return lo


def posterior(log_lr: np.ndarray, pi: float) -> np.ndarray:
    if pi <= 0:
        return np.zeros_like(log_lr)
    z = np.log(pi) - np.log1p(-pi) + log_lr
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def calibration(ref: pd.DataFrame) -> dict:
    """Fit on half of the reference players, simulate test-like sets on the other half."""
    rng = np.random.default_rng(SEED + 1)
    players = np.array(sorted(set(ref["pa"]) | set(ref["pb"])))
    half = set(players[rng.permutation(len(players))[:len(players) // 2]])
    fit = ref[ref["pa"].isin(half) & ref["pb"].isin(half)]
    held = ref[~ref["pa"].isin(half) & ~ref["pb"].isin(half)].reset_index(drop=True)
    model = PairModel().fit(fit)
    held["llr"] = model.log_lr(held)
    held_players = np.array(sorted(set(held["pa"])))
    ratios, rows = [], []
    for _ in range(SIM_SETS):
        chosen = rng.permutation(held_players)[:SIM_PLAYERS]
        two = set(chosen[:SIM_TWO])
        keep_q = {f"{p}_0" for p in chosen} | {f"{p}_1" for p in two}
        t = held[held["a"].isin(keep_q) & held["b"].isin(keep_q)]
        true_pi = float(t["same"].mean())
        pi = em_pi(t["llr"].to_numpy())
        ratios.append(pi / true_pi if true_pi > 0 else np.nan)
        w = posterior(t["llr"].to_numpy(), pi)
        keep = w >= W_MIN
        rows.append(pd.DataFrame({"w": w[keep], "same": t["same"].to_numpy()[keep]}))
    pooled = pd.concat(rows, ignore_index=True)
    bins = {}
    for lo, hi in zip(W_BINS[:-1], W_BINS[1:]):
        b = pooled[(pooled["w"] >= lo) & (pooled["w"] < hi)]
        bins[f"{lo}-{min(hi, 1.0)}"] = {"pairs": int(len(b)), "mean_w": float(b["w"].mean()) if len(b) else None,
                                         "same_rate": float(b["same"].mean()) if len(b) else None}
    positives = held[held["same"]]
    recall = {"llr_median_same": float(positives["llr"].median()), "llr_q999_diff": float(held.loc[~held["same"], "llr"].quantile(0.999))}
    ok_bins = all(abs(v["mean_w"] - v["same_rate"]) <= 0.15 for v in bins.values() if v["pairs"] >= 20)
    ratio = float(np.nanmedian(ratios))
    return {"fit_pairs": int(len(fit)), "held_pairs": int(len(held)), "pi_ratio_median": ratio,
            "pi_ratio_q10_q90": [float(np.nanquantile(ratios, 0.1)), float(np.nanquantile(ratios, 0.9))],
            "bins": bins, "separation": recall, "pass": bool(0.5 <= ratio <= 2 and ok_bins)}


# ---------------------------------------------------------------- recalibration (plan §7.1)
RECAL_SEED, RECAL_PLAYERS, RECAL_TWO, RAW_FLOOR = 2034, 250, 12, 1e-3


def logit(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 1e-12, 1 - 1e-12)
    return np.log(w) - np.log1p(-w)


def sim_sets(table: pd.DataFrame, players: np.ndarray, rng, sets: int = SIM_SETS):
    """Yield (pair rows, raw w) of test-like sets drawn from `players` (both questions of RECAL_TWO of them)."""
    for _ in range(sets):
        chosen = rng.permutation(players)[:RECAL_PLAYERS]
        keep_q = {f"{p}_0" for p in chosen} | {f"{p}_1" for p in chosen[:RECAL_TWO]}
        t = table[table["a"].isin(keep_q) & table["b"].isin(keep_q)]
        yield t, posterior(t["llr"].to_numpy(), em_pi(t["llr"].to_numpy()))


class Platt:
    def fit(self, w: np.ndarray, same: np.ndarray):
        keep = w >= RAW_FLOOR
        self.model = LogisticRegression(C=1e6, max_iter=5000).fit(logit(w[keep])[:, None], same[keep])
        self.a, self.b = float(self.model.intercept_[0]), float(self.model.coef_[0, 0])
        return self

    def __call__(self, w: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(self.a + self.b * logit(w), -50, 50)))


def bin_table(w: np.ndarray, same: np.ndarray) -> tuple[dict, bool]:
    bins = {}
    for lo, hi in zip(W_BINS[:-1], W_BINS[1:]):
        m = (w >= lo) & (w < hi)
        bins[f"{lo}-{min(hi, 1.0)}"] = {"pairs": int(m.sum()), "mean_w": float(w[m].mean()) if m.any() else None,
                                         "same_rate": float(same[m].mean()) if m.any() else None}
    ok = all(abs(v["mean_w"] - v["same_rate"]) <= 0.15 for v in bins.values() if v["pairs"] >= 20)
    return bins, ok


def recalibrate(ref: pd.DataFrame) -> tuple[dict, list]:
    """Two cross-fitted directions: (pair model on one half, Platt map on a quarter, check on the other quarter)."""
    rng = np.random.default_rng(RECAL_SEED)
    players = np.array(sorted(set(ref["pa"]) | set(ref["pb"])))
    perm = rng.permutation(players)
    halves = (perm[:len(perm) // 2], perm[len(perm) // 2:])
    report, fitted = {}, []
    for name, (fit_p, other) in (("A_to_B", halves), ("B_to_A", halves[::-1])):
        fit_set = set(fit_p)
        model = PairModel().fit(ref[ref["pa"].isin(fit_set) & ref["pb"].isin(fit_set)])
        other = rng.permutation(other)
        cal_p, val_p = other[:len(other) // 2], other[len(other) // 2:]
        entry = {}
        for role, ps in (("calibration", cal_p), ("validation", val_p)):
            group = set(ps)
            t = ref[ref["pa"].isin(group) & ref["pb"].isin(group)].reset_index(drop=True)
            t["llr"] = model.log_lr(t)
            ws, ys = [], []
            for rows, w in sim_sets(t, np.array(sorted(group)), rng):
                ws.append(w), ys.append(rows["same"].to_numpy())
            entry[role] = (np.concatenate(ws), np.concatenate(ys))
        platt = Platt().fit(*entry["calibration"])
        w_val, y_val = entry["validation"]
        raw_bins, raw_ok = bin_table(w_val[w_val >= W_MIN], y_val[w_val >= W_MIN])
        cal = platt(w_val)
        cal_bins, cal_ok = bin_table(cal[cal >= W_MIN], y_val[cal >= W_MIN])
        report[name] = {"platt": {"a": platt.a, "b": platt.b}, "validation_raw_bins": raw_bins,
                        "validation_calibrated_bins": cal_bins, "pass": bool(cal_ok),
                        "validation_same_pairs": int(y_val.sum()),
                        "validation_recall_at_0.5": float(((cal >= 0.5) & y_val).sum() / max(y_val.sum(), 1))}
        fitted.append((model, platt))
        print(name, json.dumps(report[name], default=float), flush=True)
    report["pass"] = bool(all(report[n]["pass"] for n in ("A_to_B", "B_to_A")))
    return report, fitted


def calibrated_test_w(test: pd.DataFrame, fitted: list) -> tuple[np.ndarray, list]:
    ws, pis = [], []
    for model, platt in fitted:
        llr = model.log_lr(test)
        pi = em_pi(llr)
        ws.append(platt(posterior(llr, pi)))
        pis.append(pi)
    return np.mean(ws, axis=0), pis


def run_recal(args) -> None:
    import scripts.strength.rank_prod_a0_fusion as prod
    ref = pd.read_parquet(OUT / "ref_pairs.parquet")
    test = pd.read_parquet(OUT / "test_pairs.parquet")
    report = {"recalibration": None}
    report["recalibration"], fitted = recalibrate(ref)
    (OUT / f"recal_{args.job}.json").write_text(json.dumps(report, indent=1, default=float))
    if not report["recalibration"]["pass"]:
        print("recalibration gate failed: no submission file", flush=True)
        return
    test = test.copy()
    test["w"], pis = calibrated_test_w(test, fitted)
    report["test_pi_raw"] = pis
    fit, eval_games, x_fit, x_eval = prod.train_data()
    report["export"] = export(test, fit, eval_games, x_fit, x_eval, args.job, pi=float(np.mean(pis)))
    print(json.dumps(report["export"], indent=1, default=int), flush=True)
    (OUT / f"recal_{args.job}.json").write_text(json.dumps(report, indent=1, default=float))


# ---------------------------------------------------------------- decision value on Valid-U
def decision(fit, eval_games, x_fit, x_eval) -> dict:
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_prod_a0_fusion as prod
    saved = qm.CELLS
    qm.CELLS = PAIR_CELLS
    try:
        groups = qm.draw_cells(eval_games, np.random.default_rng(SEED + 2))
    finally:
        qm.CELLS = saved
    gi = eval_games["game_index"].to_numpy()
    prior = more.player_prior(fit[fit["role"] == "S"])
    per_seed = {}
    for seed in prod.SEEDS:
        model = prod._fit(fit, x_fit, prior, seed)
        p_sep = [model.probabilities(x_eval, gi, pd.DataFrame({"games": [list(q) for q in g["questions"]],
                                                               "k": [k_bucket(len(q)) for q in g["questions"]]}))
                 for g in groups]
        p_m = qm.merged_probabilities(model, x_eval, gi, [g["questions"] for g in groups],
                                      [k_bucket(sum(len(q) for q in g["questions"])) for g in groups])
        # false links: consecutive groups of the same cell from different players, first question of each
        first = np.array([score_rank(p[:1])[0] for p in p_sep])
        cells = np.array([str(g["cell"]) for g in groups])
        false = []
        for c in sorted(set(cells)):
            idx = np.flatnonzero(cells == c)
            for a, b in zip(idx[0::2], idx[1::2]):
                if abs(int(first[a]) - int(first[b])) <= 1:
                    false.append((a, b))
        f_qs = [[groups[a]["questions"][0], groups[b]["questions"][1]] for a, b in false]
        f_sep = [np.vstack([p_sep[a][:1], p_sep[b][1:2]]) for a, b in false]
        f_m = qm.merged_probabilities(model, x_eval, gi, f_qs, [k_bucket(sum(len(q) for q in g)) for g in f_qs])
        f_truth = [np.array([groups[a]["rank"], groups[b]["rank"]]) for a, b in false]
        t_truth = [np.array([g["rank"]] * 2) for g in groups]
        entry = {}
        for w in W_GRID:
            def delta(seps, merged, truths, weight):
                d = []
                for p, pm, tr in zip(seps, merged, truths):
                    mix = (1 - weight) * p + weight * pm[None]
                    d.append(qm.official(score_rank(mix), tr) - qm.official(score_rank(p), tr))
                return float(np.mean(np.concatenate(d)))
            hard = 1.0 if w >= W_HARD else 0.0
            dt, df_ = delta(p_sep, p_m, t_truth, w), delta(f_sep, f_m, f_truth, w)
            ht, hf_ = delta(p_sep, p_m, t_truth, hard), delta(f_sep, f_m, f_truth, hard)
            entry[str(w)] = {"soft": w * dt + (1 - w) * df_, "hard": w * ht + (1 - w) * hf_,
                             "soft_true": dt, "soft_false": df_}
        per_seed[str(seed)] = entry
        print(seed, json.dumps({w: round(e["soft"], 4) for w, e in entry.items()}), flush=True)
    mean = {w: {key: round(float(np.mean([per_seed[s][w][key] for s in per_seed])), 4)
                for key in ("soft", "hard", "soft_true", "soft_false")} for w in map(str, W_GRID)}
    ok = all(v["soft"] >= 0 for v in mean.values()) and \
        np.mean([v["soft"] for v in mean.values()]) >= np.mean([v["hard"] for v in mean.values()])
    return {"groups": len(groups), "per_seed": per_seed, "mean": mean, "pass": bool(ok)}


# ---------------------------------------------------------------- export
def export(test: pd.DataFrame, fit, eval_games, x_fit, x_eval, job: str, pi: float) -> dict:
    """`test` carries the same-player probability w of every test pair."""
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_prod_a0_fusion as prod
    links = test[test["w"] >= W_MIN].sort_values("w", ascending=False)

    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit, eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    b2 = prod._fit(stacked, np.vstack([x_fit, x_eval[u]]), prior, 2027)
    xs, gis, qs, shift = [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        xs.append(x), gis.append(gi + shift), qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True).set_index("question_id")
    p = pd.DataFrame(b2.probabilities(x, gi, questions.reset_index()), index=questions.index)
    base = pd.read_csv(prod.BASE, dtype=str).set_index("question_id")["pred_rank"]
    sep = pd.Series([RANKS[r] for r in score_rank(p.to_numpy())], index=p.index).loc[base.index]
    if (sep != base).any():
        raise ValueError(f"refit does not reproduce 751959-cvec2.csv ({int((sep != base).sum())})")

    def merged(ids):
        games = [np.asarray(questions.loc[q, "games"]) for q in ids]
        total = int(sum(len(g) for g in games))
        return qm.merged_probabilities(b2, x, gi, [games], [k_bucket(total)])[0]

    hard = links[links["w"] >= W_HARD]
    hard_groups = qm.groups_from_pairs(hard.rename(columns={"a": "q1", "b": "q2"})) if len(hard) else []
    member = {q: i for i, g in enumerate(hard_groups) for q in g}
    final = p.copy()
    detail = []
    for i, g in enumerate(hard_groups):
        pg = merged(g)
        inside = hard[hard["a"].isin(g) & hard["b"].isin(g)]
        for q in g:
            wbar = float(inside.loc[(inside["a"] == q) | (inside["b"] == q), "w"].max())
            final.loc[q] = (1 - wbar) * p.loc[q].to_numpy() + wbar * pg
            detail.append({"question_id": q, "group": i, "w": round(wbar, 3), "partners": [x for x in g if x != q]})
    for q in p.index:
        if q in member:
            continue
        cand = links[((links["a"] == q) | (links["b"] == q)) & (links["w"] < W_HARD)]
        if not len(cand):
            continue
        best = cand.iloc[0]
        other = best["b"] if best["a"] == q else best["a"]
        w = float(best["w"])
        final.loc[q] = (1 - w) * p.loc[q].to_numpy() + w * merged([q, other])
        detail.append({"question_id": q, "group": None, "w": round(w, 3), "partners": [other]})
    soft = pd.Series([RANKS[r] for r in score_rank(final.to_numpy())], index=final.index).loc[base.index]

    l10 = pd.read_csv(qm.L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    with_l10 = soft.copy()
    with_l10.loc[l10.index] = l10
    current = pd.read_csv(ROOT / "submissions" / "rank" / "826812-qm-l10.csv", dtype=str).set_index("question_id")["pred_rank"]
    diff = with_l10 != current
    detail = pd.DataFrame(detail)
    if len(detail):
        detail["separate"] = base.loc[detail["question_id"]].to_numpy()
        detail["soft"] = soft.loc[detail["question_id"]].to_numpy()
    detail.to_csv(OUT / f"links_{job}.csv", index=False)
    out = {"pi": pi, "test_pairs": int(len(test)), "pairs_w_ge_0.05": int(len(links)), "pairs_w_ge_0.5": int(len(hard)),
           "hard_groups": hard_groups, "questions_touched": int(len(detail)),
           "w_histogram": {f"{lo}-{min(hi, 1.0)}": int(((test["w"] >= lo) & (test["w"] < hi)).sum())
                           for lo, hi in zip(W_BINS[:-1], W_BINS[1:])},
           "changed_vs_separate": {"public": int((soft != base)[soft.index.str.startswith("pub")].sum()),
                                   "private": int((soft != base)[soft.index.str.startswith("priv")].sum())},
           "vs_826812_qm_l10": {"public": int(diff[diff.index.str.startswith("pub")].sum()),
                                "private": int(diff[diff.index.str.startswith("priv")].sum()),
                                "questions": sorted(diff[diff].index)}}
    out["file"] = write_rank_csv(with_l10.rename("pred_rank").reset_index(), "qsoft-l10", job).name
    return out


def run(args) -> None:
    import scripts.strength.rank_prod_a0_fusion as prod
    ref = pd.read_parquet(OUT / "ref_pairs.parquet")
    test = pd.read_parquet(OUT / "test_pairs.parquet")
    report = {"reference": {"pairs": int(len(ref)), "same_pairs": int(ref["same"].sum())}}
    report["calibration"] = calibration(ref)
    print(json.dumps(report["calibration"], indent=1), flush=True)
    fit, eval_games, x_fit, x_eval = prod.train_data()
    report["decision"] = decision(fit, eval_games, x_fit, x_eval)
    print(json.dumps({"mean": report["decision"]["mean"], "pass": report["decision"]["pass"]}, indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1, default=int))
    if report["calibration"]["pass"] and report["decision"]["pass"]:
        model = PairModel().fit(ref)
        test = test.copy()
        llr = model.log_lr(test)
        pi = em_pi(llr)
        test["w"] = posterior(llr, pi)
        report["export"] = export(test, fit, eval_games, x_fit, x_eval, args.job, pi)
        print(json.dumps(report["export"], indent=1, default=int), flush=True)
    else:
        print("gate failed: no submission file", flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1, default=int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pairs").set_defaults(fn=pairs)
    p = sub.add_parser("recal")
    p.add_argument("--job", default="local")
    p.set_defaults(fn=run_recal)
    p = sub.add_parser("run")
    p.add_argument("--job", default="local")
    p.set_defaults(fn=run)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
