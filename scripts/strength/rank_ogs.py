"""Competition 2: OGS rank supervision as an extra B2 feature (2026-11-21, user "賭一把"; registered in RESULTS.md)

The rank label is a player attribute and Yugen has 8,007 players; OGS has ~82,000 ranked players with a
per-game rank (BR/WR). Stage 1 only: HumanSL H features (15 KGS profiles, own / opponent mean log p, the
same extraction as the Yugen H) on a rank-stratified OGS sample; a nonlinear per-game regressor
g(own 15, opp 15) -> OGS rank trained on OGS only; on Yugen games g_own = g(own, opp) and g_opp = g(opp, own)
are added to the 52 B2 features (a linear map of H would add nothing to the linear B2, hence nonlinear).
Control arm: the same regressor trained on Yugen fit-pool games (player-grouped out-of-fold) -> separates
"nonlinearity" from "OGS players".

  sample                     CPU: OGS games (ranked, even, not correspondence, 100+ moves), stratified by rank
  extract --shard I --num-shards N   1 GPU each: KataGo HumanSL per-move log p (go_aicup.features.humansl.run)
  evaluate --job ID          CPU: regressors, Valid-U E1/E2 seeds 2027-2031, export if the gate passes

    python -m scripts.strength.rank_ogs sample
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

OGS = Path("/home/jason1122/ogs2025/filtered")
OUT = ROOT / "outputs" / "rank" / "ogs"
DATA = ROOT / "data" / "processed" / "strength_features" / "ogs_h"
SEED = 2035
RANK_LO, RANK_HI = 10.0, 38.0          # 20k .. 9d (OGS scale: 30k = 0, 1d = 30)
PER_BIN, PER_PLAYER, MIN_MOVES = 1000, 3, 100
ARMS = ("hc", "ogs", "yugen")


# ---------------------------------------------------------------- sample
def sample(_args) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    metas = []
    for path in sorted(OGS.glob("meta_*.npy")):
        m = pd.DataFrame(np.load(path))
        m["shard"] = int(path.stem.split("_")[1])
        metas.append(m)
    meta = pd.concat(metas, ignore_index=True)
    ok = ((meta["ranked"] == 1) & (meta["ha"] == 0) & (meta["speed"] != 2) & (meta["length"] >= MIN_MOVES)
          & meta["br"].between(RANK_LO, RANK_HI) & meta["wr"].between(RANK_LO, RANK_HI)
          & ((meta["br"] - meta["wr"]).abs() <= 3))
    pool = meta[ok].sample(frac=1.0, random_state=SEED)
    pool["bin"] = np.floor(pool["br"]).astype(int)
    pool = pool.groupby("black").head(PER_PLAYER)
    chosen = pool.groupby("bin").head(PER_BIN).sort_values(["shard", "offset"]).reset_index(drop=True)
    chosen.to_parquet(DATA / "games.parquet", index=False)
    print(json.dumps({"eligible": int(ok.sum()), "chosen": int(len(chosen)),
                      "black_players": int(chosen["black"].nunique()),
                      "players": int(pd.concat([chosen["black"], chosen["white"]]).nunique()),
                      "per_bin": chosen.groupby("bin").size().to_dict()}, indent=1))


def load_games(table: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    out, cache = [], {}
    for row in table.itertuples():
        if row.shard not in cache:
            cache = {row.shard: np.load(OGS / f"moves_{row.shard:05d}.npy", mmap_mode="r")}
        moves = np.asarray(cache[row.shard][row.offset:row.offset + row.length]).astype(np.int64)
        passes = np.flatnonzero(moves == 361)
        if len(passes):
            moves = moves[:passes[0]]
        out.append((moves, (np.arange(len(moves)) % 2).astype(np.int8)))
    return out


def extract(args) -> None:
    from go_aicup.features import humansl as hs
    table = pd.read_parquet(DATA / "games.parquet").iloc[args.shard::args.num_shards].reset_index(drop=True)
    out = DATA / f"shard_{args.shard}"
    if (out / "meta.json").exists():
        print(f"{out} done", flush=True)
        return
    table.to_parquet(DATA / f"games_{args.shard}.parquet", index=False)
    out.mkdir(parents=True, exist_ok=True)
    # 8 engines per node: fewer analysis threads and one log file each (as cvec2_policy/played.py)
    meta = hs.run(load_games(table), list(hs.PROFILES), out, komi=hs.KOMI, allow_failures=True,
                  overrides=f"numAnalysisThreads=64,logFile={out / 'katago.log'}")
    print(json.dumps(meta), flush=True)


# ---------------------------------------------------------------- features
def ogs_rows() -> pd.DataFrame:
    """Two rows per complete OGS game (black and white as the own side): own 15, opp 15, own / opp rank."""
    rows = []
    for shard in sorted(DATA.glob("shard_*")):
        i = int(shard.name.split("_")[1])
        table = pd.read_parquet(DATA / f"games_{i}.parquet")
        logp, offsets, done = (np.load(shard / f"{n}.npy") for n in ("logp", "offsets", "done"))
        for g, row in enumerate(table.itertuples()):
            v = logp[offsets[g]:offsets[g + 1]]
            if not done[g].all() or not np.isfinite(v).all() or len(v) < MIN_MOVES:
                continue
            black, white = v[0::2].mean(axis=0), v[1::2].mean(axis=0)
            for own, opp, r_own, r_opp, pid in ((black, white, row.br, row.wr, row.black),
                                                (white, black, row.wr, row.br, row.white)):
                rows.append(np.r_[own, opp, r_own, r_opp, pid])
    arr = np.array(rows)
    cols = [f"own_{j}" for j in range(15)] + [f"opp_{j}" for j in range(15)] + ["rank", "opp_rank", "player"]
    return pd.DataFrame(arr, columns=cols)


def regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=SEED)


def swap(h: np.ndarray) -> np.ndarray:
    return np.hstack([h[:, 15:30], h[:, :15]])


def g_features(model, h: np.ndarray) -> np.ndarray:
    """[n, 2] = (g(own, opp), g(opp, own)) for H rows [own 15 | opp 15]."""
    return np.column_stack([model.predict(h), model.predict(swap(h))])


def yugen_oof(fit: pd.DataFrame, x_fit: np.ndarray, x_eval: np.ndarray, folds: int = 5):
    """Control: the regressor on Yugen fit games (target rank_id), player-grouped out-of-fold on the fit pool."""
    players = fit["player_id"].astype(str).to_numpy()
    uniq = np.array(sorted(set(players)))
    fold_of = pd.Series(np.random.default_rng(SEED).integers(folds, size=len(uniq)), index=uniq).loc[players].to_numpy()
    y = fit["rank_id"].to_numpy().astype(float)
    oof = np.zeros((len(fit), 2))
    for f in range(folds):
        m = regressor().fit(x_fit[fold_of != f, :30], y[fold_of != f])
        oof[fold_of == f] = g_features(m, x_fit[fold_of == f, :30])
    full = regressor().fit(x_fit[:, :30], y)
    return oof, g_features(full, x_eval[:, :30]), full


# ---------------------------------------------------------------- evaluate
def evaluate(args) -> None:
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_prod_a0_fusion as prod

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    rows = ogs_rows()
    players = rows["player"].to_numpy()
    uniq = np.unique(players)
    held = set(np.random.default_rng(SEED).permutation(uniq)[:len(uniq) // 5])
    test_mask = np.isin(players, list(held))
    h_cols = [c for c in rows.columns if c.startswith(("own_", "opp_"))]
    sanity = regressor().fit(rows.loc[~test_mask, h_cols].to_numpy(), rows.loc[~test_mask, "rank"].to_numpy())
    pred = sanity.predict(rows.loc[test_mask, h_cols].to_numpy())
    truth = rows.loc[test_mask, "rank"].to_numpy()
    report["ogs"] = {"rows": int(len(rows)), "players": int(len(uniq)),
                     "holdout_mae": float(np.abs(pred - truth).mean()),
                     "holdout_r2": float(1 - ((pred - truth) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()),
                     "rank_mean": float(rows["rank"].mean())}
    g_model = regressor().fit(rows[h_cols].to_numpy(), rows["rank"].to_numpy())
    print(json.dumps(report["ogs"]), flush=True)

    fit, eval_games, x_fit, x_eval = prod.train_data()
    g_fit, g_eval = g_features(g_model, x_fit[:, :30]), g_features(g_model, x_eval[:, :30])
    y_oof, y_eval, y_model = yugen_oof(fit, x_fit, x_eval)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    report["yugen_check"] = {"corr_g_own_rank_fit": corr(g_fit[:, 0], fit["rank_id"]),
                             "corr_g_own_rank_eval": corr(g_eval[:, 0], eval_games["rank_id"]),
                             "corr_yugen_oof_rank_fit": corr(y_oof[:, 0], fit["rank_id"]),
                             "g_own_by_rank_eval": pd.Series(g_eval[:, 0]).groupby(eval_games["rank_id"].to_numpy())
                             .mean().round(2).tolist()}
    print(json.dumps(report["yugen_check"]), flush=True)
    if report["yugen_check"]["corr_g_own_rank_eval"] < 0.2:
        raise SystemExit("OGS regressor barely tracks Yugen rank (H column order or transfer broken)")

    xs_fit = {"hc": x_fit, "ogs": np.hstack([x_fit, g_fit]), "yugen": np.hstack([x_fit, y_oof])}
    xs_eval = {"hc": x_eval, "ogs": np.hstack([x_eval, g_eval]), "yugen": np.hstack([x_eval, y_eval])}
    sets = prod.question_sets()
    prior = more.player_prior(fit[fit["role"] == "S"])
    seeds = {}
    for seed in prod.SEEDS:
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {a: hf.predict(prod._fit(fit, xs_fit[a], prior, seed), eval_games, xs_eval[a], q) for a in ARMS}
            entry[name] = {a: hf.by_k(s, k) for a, s in scores.items()}
            for a in ("ogs", "yugen"):
                d = {str(b): round(float(scores[a][k == b].mean() - scores["hc"][k == b].mean()), 4) for b in K_BUCKETS}
                d["equal_k"] = round(hf.equal_k(scores[a], k) - hf.equal_k(scores["hc"], k), 4)
                entry[name][f"diff_{a}"] = d
        seeds[str(seed)] = entry
        print(seed, json.dumps({n: {a: e[f"diff_{a}"]["equal_k"] for a in ("ogs", "yugen")} for n, e in entry.items()}),
              flush=True)
    mean = {a: {n: {key: round(float(np.mean([seeds[str(s)][n][f"diff_{a}"][key] for s in prod.SEEDS])), 4)
                    for key in [str(b) for b in K_BUCKETS] + ["equal_k"]} for n in sets} for a in ("ogs", "yugen")}
    report["local"] = {"seeds": seeds, "mean_diff": mean}
    m = mean["ogs"]
    best, other = max(m["E1"]["equal_k"], m["E2"]["equal_k"]), min(m["E1"]["equal_k"], m["E2"]["equal_k"])
    report["gate"] = {"pass": bool(best >= 0.005 and other >= 0 and all(m["E2"][str(b)] >= -0.005 for b in K_BUCKETS))}
    print(json.dumps({"mean_diff": mean, "gate": report["gate"]}, indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))
    if report["gate"]["pass"] or args.force_export:
        report["export"] = export(g_model, fit, eval_games, x_fit, x_eval, g_fit, g_eval, args.job)
        print(json.dumps(report["export"], indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))


def export(g_model, fit, eval_games, x_fit, x_eval, g_fit, g_eval, job: str) -> dict:
    """Refit on Valid-S + train20 + Valid-U HD<=1 (the base recipe, H+C arm must reproduce 751959-cvec2),
    then the 826812 question merge groups and the L10 overlay."""
    from go_aicup.data.board_cache import RANKS
    from go_aicup.tasks.strength.humansl import k_bucket
    from go_aicup.tasks.strength.submission_files import write_rank_csv
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_prod_a0_fusion as prod
    import scripts.strength.rank_qmerge as qm

    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit, eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    models = {"hc": prod._fit(stacked, np.vstack([x_fit, x_eval[u]]), prior, 2027),
              "ogs": prod._fit(stacked, np.vstack([np.hstack([x_fit, g_fit]), np.hstack([x_eval, g_eval])[u]]), prior, 2027)}
    xs, gis, qs, shift = [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        xs.append(x), gis.append(gi + shift), qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True)
    x_ogs = np.hstack([x, g_features(g_model, x[:, :30])])
    base = pd.read_csv(prod.BASE, dtype=str).set_index("question_id")["pred_rank"]
    preds = {}
    for arm, xx in (("hc", x), ("ogs", x_ogs)):
        sep = pd.Series([RANKS[r] for r in score_rank(models[arm].probabilities(xx, gi, questions))],
                        index=questions["question_id"]).loc[base.index]
        groups = qm.groups_from_pairs(pd.read_csv(qm.PAIRS))
        by_id = questions.set_index("question_id")
        members = [[np.asarray(by_id.loc[q, "games"]) for q in g] for g in groups]
        pm = qm.merged_probabilities(models[arm], xx, gi, members, [k_bucket(sum(len(m) for m in ms)) for ms in members])
        merged = sep.copy()
        for g, prob in zip(groups, pm):
            for q in g:
                merged[q] = RANKS[int(score_rank(prob[None])[0])]
        l10 = pd.read_csv(qm.L10)
        l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
        merged.loc[l10.index] = l10
        preds[arm] = (sep, merged)
    if (preds["hc"][0] != base).any():
        raise ValueError("H+C refit does not reproduce 751959-cvec2.csv")
    current = pd.read_csv(ROOT / "submissions" / "rank" / "826812-qm-l10.csv", dtype=str).set_index("question_id")["pred_rank"]
    if (preds["hc"][1] != current).any():
        raise ValueError("H+C merge + L10 does not reproduce 826812-qm-l10.csv")
    new = preds["ogs"][1]
    diff = new != current
    order = {r: i for i, r in enumerate(RANKS)}
    return {"public": int(diff[diff.index.str.startswith("pub")].sum()),
            "private": int(diff[diff.index.str.startswith("priv")].sum()),
            "steps": {str(int(s)): int(n) for s, n in (new[diff].map(order) - current[diff].map(order)).value_counts().sort_index().items()},
            "distribution": new.value_counts().reindex(RANKS, fill_value=0).tolist(),
            "file": write_rank_csv(new.rename("pred_rank").reset_index(), "ogs-qm-l10", job).name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sample").set_defaults(fn=sample)
    p = sub.add_parser("extract")
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.set_defaults(fn=extract)
    p = sub.add_parser("evaluate")
    p.add_argument("--job", default="local")
    p.add_argument("--force-export", action="store_true", help="write the file even if the gate fails (LB probe)")
    p.set_defaults(fn=evaluate)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
