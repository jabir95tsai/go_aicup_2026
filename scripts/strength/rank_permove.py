"""S2: re-aggregate the per-move C log-likelihoods instead of the plain game mean (RESULTS 2026-11-12).

C (cvec2 base) is, per game and side, the mean over moves of log p(played move | rank r). A few
very unlikely moves dominate that mean, and k=20 is not yet at the per-game noise limit (k=inf
diagnostic, 11/11). slurm/rank_permove_extract.slurm stores every position's 10 log p values
(C v2 vectors, same tables as cvec_v2). Here each variant turns them into a replacement for the
20 C columns (own 10 | opp 10); H and the HD1 flags are unchanged:
  mean            plain mean (must reproduce the cvec2 C features)
  clip4/6/8       mean of max(log p, -c)
  trim5/10        drop the 5% / 10% of a side's moves with the lowest rank-average log p
  vote            mean over moves of softmax_r(log p)   (bounded per move)
  p20, p40        plies >= 20 / >= 40 only
  u200, mid       plies < 200 only / 40 <= ply < 200
  mean+vote       40 columns (the only variant that adds dimensions)

features  per-move shards -> outputs/rank/humansl/permove/agg_<space>.npz
select    variant choice inside the fitting pool only (Valid-S + train20, HD<=1): players split in
          two rank-stratified halves, B2 fitted on one half and scored on rank-balanced questions
          of the other (per k and rank 100 questions), three B2 seeds. Valid-U is not read.
evaluate  the chosen variant against mean on Valid-U E1 / E2, five seeds (G1 as cvec2).
Pre-registered: the variant with the highest selection gain is the only one scored on Valid-U,
and only if its selection gain is >= +0.003; G1 promote = five-seed mean Equal-k >= +0.005 on E1
or E2, the other >= 0, every E2 k >= -0.005; both >= 0 -> LB probe; else stop.

    python -m scripts.strength.rank_permove features
    python -m scripts.strength.rank_permove select [--jobs 12]
    python -m scripts.strength.rank_permove evaluate --variant NAME
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs" / "rank" / "humansl" / "permove"
SPACES = {"train": ("valid", "t20"), "test_public": ("test_public",), "test_private": ("test_private",)}
CLIPS, TRIMS = (4, 6, 8), (5, 10)
WINDOWS = {"p20": (20, 10_000), "p40": (40, 10_000), "u200": (0, 200), "mid": (40, 200)}
SELECT_SEEDS, SELECT_PER_CELL, SELECT_MIN = (2027, 2028, 2029), 100, 0.003
C_COLS = slice(30, 50)


def _side_mean(values: np.ndarray, slot: np.ndarray, weight: np.ndarray | None, slots: int) -> np.ndarray:
    """[slots, R] weighted mean of values per slot; NaN where the weight sum is 0."""
    total = np.bincount(slot, weights=weight, minlength=slots)
    out = np.empty((slots, values.shape[1]))
    for r in range(values.shape[1]):
        column = values[:, r] if weight is None else values[:, r] * weight
        out[:, r] = np.bincount(slot, weights=column, minlength=slots)
    with np.errstate(invalid="ignore", divide="ignore"):
        return out / total[:, None]


def shard_variants(shard) -> tuple[np.ndarray, dict]:
    games = pd.read_parquet(shard / "games.parquet")
    done = np.load(shard / "done.npy")
    slot = np.load(shard / "move_slot.npy").astype(np.int64)
    ply = np.load(shard / "move_ply.npy")
    logp = np.load(shard / "move_logp.npy").astype(np.float32)
    slots = 2 * len(games)
    out = {"mean": _side_mean(logp, slot, None, slots)}
    for c in CLIPS:
        out[f"clip{c}"] = _side_mean(np.maximum(logp, -c), slot, None, slots)
    common = logp.mean(axis=1)
    order = np.lexsort((common, slot))
    counts = np.bincount(slot, minlength=slots)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    position = np.arange(len(slot)) - start[slot[order]]
    for t in TRIMS:
        keep = np.empty(len(slot))
        keep[order] = position >= np.ceil(t / 100 * counts[slot[order]])
        out[f"trim{t}"] = _side_mean(logp, slot, keep, slots)
    z = logp - logp.max(axis=1, keepdims=True)
    post = np.exp(z)
    post /= post.sum(axis=1, keepdims=True)
    out["vote"] = _side_mean(post, slot, None, slots)
    for name, (low, high) in WINDOWS.items():
        window = _side_mean(logp, slot, ((ply >= low) & (ply < high)).astype(float), slots)
        out[name] = np.where(np.isfinite(window), window, out["mean"])
    out = {name: value.reshape(len(games), -1)[done] for name, value in out.items()}
    reference = np.load(shard / "feature.npy")[done]
    worst = float(np.abs(out["mean"] - reference).max()) if done.any() else 0.0
    if worst > 5e-3:          # float16 storage of log p
        raise ValueError(f"{shard}: per-move mean differs from feature.npy by {worst}")
    return games["game_index"].to_numpy()[done], out


def features() -> None:
    from go_aicup.features import humansl_rankvec as cfeat
    import scripts.strength.rank_cvec_v2 as cv2
    report = {}
    for space, names in SPACES.items():
        index, parts = [], {}
        for name in names:
            for hd in (0, 1):
                run = OUT / f"c_{name}_hd{hd}"
                shards = sorted(p for p in run.glob("shard_*") if (p / "meta.json").exists())
                if len(shards) != 8:
                    raise ValueError(f"{run}: {len(shards)} of 8 shards complete")
                for shard in shards:
                    gi, variants = shard_variants(shard)
                    index.append(gi)
                    for key, value in variants.items():
                        parts.setdefault(key, []).append(value.astype(np.float32))
        index = np.concatenate(index)
        if len(np.unique(index)) != len(index):
            raise ValueError(f"{space}: duplicate games")
        arrays = {key: np.concatenate(value) for key, value in parts.items()}
        # the new run must agree with the cvec2 run the base was fitted on
        old = cv2.new_c(space if space != "train" else "train")
        common = np.isin(index, old.index)
        diff = float(np.abs(arrays["mean"][common] - old.loc[index[common]].to_numpy()).max())
        report[space] = {"games": int(len(index)), "in_cvec2": int(common.sum()), "cvec2_games": int(len(old)),
                         "max_abs_diff_vs_cvec2": round(diff, 5)}
        if common.sum() != len(old) or diff > 1e-2:
            raise ValueError(f"{space}: per-move run does not match cvec2: {report[space]}")
        np.savez(OUT / f"agg_{space}.npz", game_index=index, **arrays)
        print(space, json.dumps(report[space]), flush=True)
    (OUT / "features.json").write_text(json.dumps(report, indent=1))


def variants(space: str) -> tuple[pd.Series, dict]:
    """(game_index -> row, {variant: [games, 20 or 40]})."""
    data = np.load(OUT / f"agg_{space}.npz")
    arrays = {key: data[key] for key in data.files if key != "game_index"}
    arrays["mean+vote"] = np.hstack([arrays["mean"], arrays["vote"]])
    return pd.Series(np.arange(len(data["game_index"])), index=data["game_index"]), arrays


def with_variant(x: np.ndarray, game_index: np.ndarray, rows: pd.Series, block: np.ndarray) -> np.ndarray:
    """x = [H 30 | C 20 | flags 2] with the C block replaced; games without per-move data get NaN."""
    have = np.isin(game_index, rows.index)
    c = np.full((len(x), block.shape[1]), np.nan)
    c[have] = block[rows.loc[game_index[have]].to_numpy()]
    return np.hstack([x[:, :30], c, x[:, 50:]])


def _halves(fit: pd.DataFrame) -> np.ndarray:
    players = fit.groupby(fit["player_id"].astype(str))["rank_id"].first().sort_index()
    rng = np.random.default_rng(2027)
    half = pd.Series(False, index=players.index)
    for _, ids in players.groupby(players):
        order = rng.permutation(len(ids))
        half.loc[ids.index[order[: len(ids) // 2]]] = True
    return half.loc[fit["player_id"].astype(str)].to_numpy()


def _questions(games: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rank-balanced single-colour questions (as E2) from the given games."""
    groups = games.groupby(["rank_id", games["player_id"].astype(str), "player_color"])["game_index"].apply(np.sort)
    rng = np.random.default_rng(seed)
    rows = []
    for k in K_BUCKETS:
        for rank in range(10):
            eligible = [(key, g) for key, g in groups.loc[rank].items() if len(g) >= k]
            for pick in rng.integers(0, len(eligible), SELECT_PER_CELL):
                (player, _), g = eligible[pick]
                rows.append({"player_id": player, "rank_id": rank, "k": k,
                             "games": np.sort(rng.choice(g, k, replace=False)).tolist()})
    return pd.DataFrame(rows)


def _select_task(seed, side, name, fit, x, half, questions):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    hs.SEED = seed
    train = half == side
    model = more.PriorMatched(more.player_prior(fit[train])).fit(fit[train].reset_index(drop=True), pd.DataFrame(),
                                                                 x[train])
    q = questions[side]
    p = model.probabilities(x, fit["game_index"].to_numpy(), q)
    return official_scores(score_rank(p), q["rank_id"].to_numpy())


def _fit_pool():
    import scripts.strength.rank_cvec_v2 as cv2
    games, fit, eval_games, xs = cv2.data()
    hd01 = (fit["handicap"] <= 1).to_numpy()
    return games, fit, eval_games, xs["v2"], hd01


def select(jobs: int) -> None:
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    _, fit, _, (x_fit, _), hd01 = _fit_pool()
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    rows, arrays = variants("train")
    gi = fit["game_index"].to_numpy()
    xs = {name: with_variant(x_fit, gi, rows, block) for name, block in arrays.items()}
    if not np.isfinite(xs["mean"]).all():
        raise ValueError("fitting games without per-move data")
    check = float(np.abs(xs["mean"] - x_fit).max())
    half = _halves(fit)
    # questions of the half that is NOT fitted: side True fits half==True, scores half==False
    questions = {side: _questions(fit[half != side], 2027 + int(side)) for side in (True, False)}
    tasks = [(seed, side, name) for seed in SELECT_SEEDS for side in (True, False) for name in xs]
    scores = Parallel(n_jobs=jobs)(delayed(_select_task)(s, side, n, fit, xs[n], half, questions) for s, side, n in tasks)
    k = np.concatenate([questions[True]["k"].to_numpy(), questions[False]["k"].to_numpy()])
    table = {}
    for name in xs:
        per_seed = []
        for seed in SELECT_SEEDS:
            got = {side: s for (sd, side, n), s in zip(tasks, scores) if sd == seed and n == name}
            per_seed.append(np.concatenate([got[True], got[False]]))
        table[name] = np.mean(per_seed, axis=0)
    base = table["mean"]
    result = {"fit_games": int(len(fit)), "players": int(fit["player_id"].nunique()),
              "mean_vs_cvec2_max_abs": round(check, 5), "questions": int(len(k)), "variants": {}}
    for name, s in table.items():
        result["variants"][name] = hf.by_k(s, k) | {
            "diff_equal_k": round(hf.equal_k(s, k) - hf.equal_k(base, k), 4),
            "diff_by_k": {str(b): round(float(s[k == b].mean() - base[k == b].mean()), 4) for b in K_BUCKETS}}
        print(name, json.dumps(result["variants"][name]), flush=True)
    best = max((n for n in table if n != "mean"), key=lambda n: result["variants"][n]["diff_equal_k"])
    result["chosen"] = best
    result["proceed"] = bool(result["variants"][best]["diff_equal_k"] >= SELECT_MIN)
    print(json.dumps({"chosen": best, "gain": result["variants"][best]["diff_equal_k"], "proceed": result["proceed"]}))
    (OUT / "select.json").write_text(json.dumps(result, indent=1))


def evaluate(variant: str) -> None:
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_more_players as more
    games, fit, eval_games, (x_fit, x_eval), hd01 = _fit_pool()
    rows, arrays = variants("train")
    arms = {"mean": (x_fit, x_eval),
            variant: (with_variant(x_fit, fit["game_index"].to_numpy(), rows, arrays[variant]),
                      with_variant(x_eval, eval_games["game_index"].to_numpy(), rows, arrays[variant]))}
    if not (np.isfinite(arms[variant][0][hd01]).all() and np.isfinite(arms[variant][1]).all()):
        raise ValueError("games without per-move data")
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    result = {"variant": variant, "seeds": {}}
    for seed in hf.SEEDS:
        hs.SEED = seed
        models = {a: more.PriorMatched(prior).fit(fit[hd01].reset_index(drop=True), pd.DataFrame(), xf[hd01])
                  for a, (xf, _) in arms.items()}
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            a = hf.predict(models["mean"], eval_games, arms["mean"][1], q)
            b = hf.predict(models[variant], eval_games, arms[variant][1], q)
            entry[name] = {"mean": hf.by_k(a, k), variant: hf.by_k(b, k),
                           "diff_by_k": {str(bk): round(float(b[k == bk].mean() - a[k == bk].mean()), 4)
                                         for bk in K_BUCKETS},
                           "diff_equal_k": round(hf.equal_k(b, k) - hf.equal_k(a, k), 4)}
            if seed == hf.SEED:
                entry[name]["bootstrap_95"] = hf.bootstrap(a, b, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: (e["diff_equal_k"], e["diff_by_k"]) for n, e in entry.items()}), flush=True)
    mean = {n: round(float(np.mean([result["seeds"][str(s)][n]["diff_equal_k"] for s in hf.SEEDS])), 4) for n in sets}
    by_k = {n: {str(bk): round(float(np.mean([result["seeds"][str(s)][n]["diff_by_k"][str(bk)] for s in hf.SEEDS])), 4)
                for bk in K_BUCKETS} for n in sets}
    promote = max(mean.values()) >= 0.005 and min(mean.values()) >= 0 and min(by_k["E2"].values()) >= -0.005
    result["G1"] = {"mean_diff_equal_k": mean, "mean_diff_by_k": by_k,
                    "decision": "promote" if promote else "probe" if min(mean.values()) >= 0 else "stop"}
    print(json.dumps(result["G1"]))
    (OUT / f"result_{variant}.json").write_text(json.dumps(result, indent=1))


def export(variant: str, job: str) -> None:
    """LB probe (11/14, user: submit near-miss arms): the cvec2 export (Valid-S + train20 + Valid-U
    HD<=1, prior-matched B2, seed 2027) with the C block replaced by `variant`. The unchanged arm
    must reproduce 751959-cvec2.csv exactly; only the C columns differ between the two fits."""
    import go_aicup.tasks.strength.humansl as hs
    import pickle
    from go_aicup.data.board_cache import RANKS
    from go_aicup.tasks.strength import humansl_predict as hp
    from go_aicup.tasks.strength.submission_files import write_rank_csv
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_more_players as more
    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    rows, arrays = variants("train")
    arms = {"cvec2": (x_fit, x_eval),
            variant: (with_variant(x_fit, fit["game_index"].to_numpy(), rows, arrays[variant]),
                      with_variant(x_eval, eval_games["game_index"].to_numpy(), rows, arrays[variant]))}
    hd01 = (fit["handicap"] <= 1).to_numpy()
    valid_s = hd01 & (fit["role"] == "S").to_numpy()
    u = (eval_games["handicap"] <= 1).to_numpy()
    for name, (xf, xe) in arms.items():
        if not (np.isfinite(xf[hd01]).all() and np.isfinite(xe[u]).all()):
            raise ValueError(f"{name}: fitting games without features")
    stacked = pd.concat([fit[hd01], eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[valid_s], eval_games[u]], ignore_index=True))
    hs.SEED = hf.SEED
    models = {a: more.PriorMatched(prior).fit(stacked, pd.DataFrame(), np.vstack([xf[hd01], xe[u]]))
              for a, (xf, xe) in arms.items()}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    out = {a: [] for a in arms}
    for part in ("public", "private"):
        _, x2, index = cv2.test_x(part, columns)
        t_rows, t_arrays = variants(f"test_{part}")
        xt = {"cvec2": x2, variant: with_variant(x2, index, t_rows, t_arrays[variant])}
        if not np.isfinite(xt[variant]).all():
            raise ValueError(f"{part}: test games without per-move data")
        questions = hp.question_table(p3.WORK / f"rank_{part}")
        for a, x in xt.items():
            p = models[a].probabilities(x, index, questions)
            out[a].append(pd.DataFrame({"question_id": questions["question_id"],
                                        "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    result = {a: pd.concat(f, ignore_index=True).set_index("question_id").loc[template.question_id]["pred_rank"]
              for a, f in out.items()}
    base = pd.read_csv(ROOT / "submissions" / "rank" / "751959-cvec2.csv", dtype=str).set_index(
        "question_id")["pred_rank"].loc[template.question_id]
    if (result["cvec2"] != base).any():
        raise ValueError(f"cvec2 refit does not reproduce 751959-cvec2.csv ({int((result['cvec2'] != base).sum())})")
    changed = result[variant] != base
    order = {r: i for i, r in enumerate(RANKS)}
    step = result[variant][changed].map(order) - base[changed].map(order)
    report = {"variant": variant, "public": int(changed[changed.index.str.startswith("pub")].sum()),
              "private": int(changed[changed.index.str.startswith("priv")].sum()),
              "step_counts": {str(int(s)): int(n) for s, n in step.value_counts().sort_index().items()},
              "distribution": result[variant].value_counts().reindex(RANKS).fillna(0).astype(int).tolist(),
              "file": write_rank_csv(result[variant].reset_index(), variant[:8], job).name}
    print(json.dumps(report))
    (OUT / f"export_{variant}_{job}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["features", "select", "evaluate", "auto", "export"])
    parser.add_argument("--variant")
    parser.add_argument("--job", default="local")
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if args.command in ("features", "auto"):
        features()
    if args.command in ("select", "auto"):
        select(args.jobs)
    if args.command == "evaluate":
        evaluate(args.variant)
    if args.command == "export":
        export(args.variant, args.job)
    if args.command == "auto":      # Valid-U is read only when the pre-registered selection rule says so
        chosen = json.loads((OUT / "select.json").read_text())
        if chosen["proceed"]:
            evaluate(chosen["chosen"])


if __name__ == "__main__":
    main()
