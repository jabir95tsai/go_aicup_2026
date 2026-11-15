"""Added-set diagnostics, phase 1 (2026-11-15, CPU): where the 9,600-step fused model
loses points and whether re-weighting the fusion per k is worth anything. Nothing is
retrained and no existing file is changed.

    S_full = P + L + lambda * w * S + mu * C      (lambda, mu) = (16, 1)

P prefix back-off, L slot naive Bayes (P + L = twocolor.opening_scores), S cosine of the
mean style vectors, C the HD-and-colour histogram.

step 0  identity of the inputs (vectors, bank, player split, query/gallery exclusion) and
        the base score on pool 0 draws 100-119 and pool 1 draws 100-129, checked against
        the paired CSV of the 9,600-step run.
D1      leave-one-part-out and single-part scores on both pools, paired with the full
        score per draw; prefix hit counts on pool 0 (true / wrong candidate x colour x
        number of the candidate's gallery games of that colour x own move j = 0..7).
D2      rank gap of a wrong top-1 against the other candidates of the same questions
        (style alone and the full score), and the score when only the candidates of the
        true rank are kept.
D3-a    per-k fusion weights chosen on pool 0: arm 1 lambda only, arm 2 lambda and mu.
        Pool 1 is scored only if an arm clears the pool 0 threshold.

    python -m scripts.identity.pid2_diag_d123a --run pid_ge2e_v4_d4_hd_9600 [--limit 2]
"""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import time

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening, twocolor
from scripts.identity.pid2_paired import BENCH, points
from scripts.identity.pid2_paired import OUT as PAIRED
from scripts.identity.pid2_transductive import load

OUT = ROOT / "outputs" / "pid" / "pid2_diag"
FIRST = 100
DRAWS = {0: 20, 1: 30}
BASE = (16.0, 1.0)
LAMBDAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MUS = (0.0, 0.5, 1.0, 2.0)
COLS = [str(k) for k in cv.PID2_K] + ["all"]
GATE = {"pool0": 0.003, "pool1": 0.005, "z": 2.5, "bucket": -0.002}
GAPS = ("0", "1", "2", "3+")


def opening_parts(env, gallery_games, query_games):
    """twocolor.opening_scores with its two terms kept apart -> (prefix, slot)."""
    prefix, hashes, game_color, params = env["prefix"], env["hashes"], env["color"], env["summary"]["opening"]
    shape = (len(query_games), len(gallery_games))
    out_p, out_l = np.zeros(shape), np.zeros(shape)
    for color in (0, 1):
        gallery = [g[game_color[g] == color] for g in gallery_games]
        query = [q[game_color[q] == color] for q in query_games]
        has = np.array([len(q) > 0 for q in query])
        if not has.any():
            continue
        query = [q for q in query if len(q)]
        population = env["populations"][color]
        plies = twocolor.color_plies(params["moves"], color)
        pop = opening.population_prior(prefix, population, plies)
        base = opening.population_conditional(hashes, prefix, population, np.concatenate(query), plies, pop)
        out_p[has] += opening.prefix_backoff(prefix, gallery, query, plies, pop, beta=params["beta"],
                                             hashes=hashes, base=base)
        if params["slot_moves"]:
            slot_plies = twocolor.color_plies(params["slot_moves"], color)
            slot_pop = opening.population_prior(prefix, population, slot_plies)
            out_l[has] += opening.slot_nb(prefix, gallery, query, slot_plies, slot_pop, params["alpha"])
    return out_p, out_l


def draw_parts(env, pool: int, draw: int, check: bool):
    candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
    query = list(q["games"])
    games = np.concatenate(query)
    assert len(np.unique(games)) == len(games), "a game sits in two questions"
    assert not np.isin(games, np.concatenate(gallery)).any(), "query game in a gallery"
    assert all(len(g) == cv.PID2_GALLERY for g in gallery)
    p, l = opening_parts(env, gallery, query)
    if check:
        whole = twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                        env["populations"], env["summary"]["opening"])
        assert np.allclose(p + l, whole), "prefix + slot does not give the opening score"
    pos = env["position"]
    g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
    assert all((x >= 0).all() for x in g + qq), "game without an embedding"
    s = env["summary"]["w"] * twocolor.style_scores(env["vectors"], g, qq)
    c = twocolor.category_scores(env["codes"], gallery, query, 4)
    return dict(candidates=candidates, gallery=gallery, q=q, P=p, L=l, S=s, C=c)


def bucket_means(pts: np.ndarray, ks: np.ndarray) -> dict:
    return {**{str(k): float(pts[ks == k].mean()) for k in cv.PID2_K}, "all": float(pts.mean())}


def paired(frame: pd.DataFrame, base: str, key: str = "arm") -> pd.DataFrame:
    """mean per arm, and the per-draw difference to `base` with its standard error over draws."""
    b = frame[frame[key] == base].set_index("draw")[COLS]
    out = []
    for arm, part in frame.groupby(key, sort=False):
        d = part.set_index("draw")[COLS] - b
        n = len(d)
        row = {key: arm, "draws": n, **{c: part[c].mean() for c in COLS}}
        row.update({f"d_{c}": d[c].mean() for c in COLS})
        row["se_all"] = d["all"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        row["pos"] = (d["all"] > 0).mean()
        out.append(row)
    return pd.DataFrame(out)


def step0(env, run: str) -> dict:
    meta, bank, summary = env["meta"], env["bank"], env["summary"]
    saved = np.load(BENCH / run / "embeddings.npz")
    games = saved["game_index"]
    bank_games = np.sort(bank["game"].to_numpy())
    pid = meta["player_id"].astype(str).to_numpy()
    gallery, _ = data.pid_splits(meta)
    config = json.loads((ROOT / summary["run_dir"] / "config.json").read_text())
    held = set(gallery.loc[gallery["split"].isin(config["exclude_splits"]), "player_id"])
    pools = [set(bank.loc[bank["pool"] == p, "player_id"]) for p in (0, 1)]
    per_player = bank.groupby("player_id")["game"].size()
    report = {
        "vectors": list(saved["swap"].shape),
        "vectors_cover_bank": bool(np.array_equal(games, bank_games)),
        "vectors_finite": bool(np.isfinite(saved["swap"]).all()),
        "w_matches_summary": bool(float(saved["w"]) == summary["w"]),
        "bank_games_unique": bool(len(np.unique(bank_games)) == len(bank_games)),
        "bank_owner_matches_meta": bool((pid[bank["game"].to_numpy()] == bank["player_id"].to_numpy()).all()),
        "bank_handicap_max": int(env["handicap"][bank["game"].to_numpy()].max()),
        "bank_games_per_player": [int(per_player.min()), int(per_player.max())],
        "pool_players": [len(pools[0]), len(pools[1])],
        "pools_overlap": len(pools[0] & pools[1]),
        "bank_players_not_excluded_from_encoder": len((pools[0] | pools[1]) - held),
        "encoder_exclude_splits": config["exclude_splits"],
        "population_rows_of_bank_players": int(np.isin(pid[np.concatenate(env["populations"])],
                                                       bank["player_id"].unique()).sum()),
        "opening": summary["opening"], "style": summary["style"], "w": summary["w"],
        "summary_weights": [summary["lambda"], summary["mu"]], "lambda_per_k": summary["lambda_per_k"],
    }
    bad = [k for k, v in report.items() if v is False]
    bad += [k for k in ("pools_overlap", "bank_players_not_excluded_from_encoder",
                        "population_rows_of_bank_players") if report[k] != 0]
    if report["bank_handicap_max"] > 1:
        bad.append("bank_handicap_max")
    report["failed"] = bad
    return report


def hit_counts(env, item) -> pd.DataFrame:
    """Per colour, own move j and number n of the candidate's gallery games of that colour:
    (query game, candidate) pairs whose candidate has a gallery game sharing the line before
    the move (before) and through it (after), for the true and for the wrong candidates."""
    hashes, game_color = env["hashes"], env["color"]
    moves = env["summary"]["opening"]["moves"]
    truth_q = np.searchsorted(item["candidates"], item["q"]["player_id"].to_numpy())
    n_cand = len(item["candidates"])
    cand = np.arange(n_cand)
    rows = []
    for color in (0, 1):
        gallery = [g[game_color[g] == color] for g in item["gallery"]]
        n_same = np.array([len(g) for g in gallery])
        flat = np.concatenate([g[game_color[g] == color] for g in item["q"]["games"]])
        truth = np.concatenate([np.full(int((game_color[g] == color).sum()), t)
                                for g, t in zip(item["q"]["games"], truth_q)])
        if not len(flat):
            continue
        plies = twocolor.color_plies(moves, color)
        counter = opening._Counter(hashes, gallery, int(plies.max()) + 1)
        size = cv.PID2_GALLERY + 1
        pairs_true = np.bincount(n_same[truth], minlength=size)
        pairs_all = np.bincount(n_same, minlength=size) * len(flat)
        for j, t in enumerate(plies):
            hit = {"before": counter(opening._keyed(hashes[flat, t][:, None], cand[None, :])) > 0,
                   "after": counter(opening._keyed(hashes[flat, t + 1][:, None], cand[None, :])) > 0}
            true = {name: np.bincount(n_same[truth], weights=h[np.arange(len(flat)), truth], minlength=size)
                    for name, h in hit.items()}
            total = {name: np.bincount(n_same, weights=h.sum(axis=0), minlength=size) for name, h in hit.items()}
            for n in range(size):
                rows.append({"candidate": "true", "color": "BW"[color], "n_same": n, "j": j, "ply": int(t),
                             "pairs": int(pairs_true[n]), "before": int(true["before"][n]),
                             "after": int(true["after"][n])})
                rows.append({"candidate": "wrong", "color": "BW"[color], "n_same": n, "j": j, "ply": int(t),
                             "pairs": int(pairs_all[n] - pairs_true[n]),
                             "before": int(total["before"][n] - true["before"][n]),
                             "after": int(total["after"][n] - true["after"][n])})
    return pd.DataFrame(rows)


def rank_gaps(scores, item, rank_of) -> tuple[pd.DataFrame, np.ndarray]:
    """Counts per k of the rank gap of a wrong top-1 and of every other candidate of those
    questions; and the points when only candidates of the true rank stay."""
    candidates, q = item["candidates"], item["q"]
    ranks = rank_of[candidates].to_numpy()
    truth = np.searchsorted(candidates, q["player_id"].to_numpy())
    gap = np.minimum(np.abs(ranks[None, :] - ranks[truth][:, None]), 3)
    top = scores.argmax(axis=1)
    wrong = top != truth
    ks = q["k"].to_numpy()
    rows = []
    for k in cv.PID2_K:
        sel = wrong & (ks == k)
        model = np.bincount(gap[sel, top[sel]], minlength=4)
        others = gap[sel].copy()
        others[np.arange(len(others)), truth[sel]] = 4             # the true candidate is not background
        back = np.bincount(others.ravel(), minlength=5)[:4]
        rows.append({"k": k, "questions": int((ks == k).sum()), "wrong": int(sel.sum()),
                     **{f"model_{g}": int(v) for g, v in zip(GAPS, model)},
                     **{f"back_{g}": int(v) for g, v in zip(GAPS, back)}})
    kept = np.where(gap == 0, scores, -np.inf)
    return pd.DataFrame(rows), points(kept, candidates, q)


def gap_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Shares and enrichment per (pool, arm, k) plus the k-pooled row; NaN = not applicable."""
    total = counts.groupby(["pool", "arm"], as_index=False, sort=False).sum(numeric_only=True).assign(k="all")
    both = pd.concat([counts.groupby(["pool", "arm", "k"], as_index=False, sort=False).sum(numeric_only=True)
                      .astype({"k": str}), total], ignore_index=True)
    model = both[[f"model_{g}" for g in GAPS]].to_numpy(dtype=float)
    back = both[[f"back_{g}" for g in GAPS]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ms = model / model.sum(axis=1, keepdims=True)
        bs = back / back.sum(axis=1, keepdims=True)
        enrich = np.where(bs > 0, ms / bs, np.nan)
    out = both[["pool", "arm", "k", "questions", "wrong"]].copy()
    for i, g in enumerate(GAPS):
        out[f"model_{g}"], out[f"back_{g}"], out[f"x_{g}"] = ms[:, i], bs[:, i], enrich[:, i]
    return out


def choose(mean: np.ndarray, mus) -> tuple[float, float]:
    """Best (lambda, mu) of a [lambda, mu] table restricted to `mus`; ties go to the weights
    nearest the base (|log2(lambda / 16)|, then |mu - 1|), then the smaller lambda and mu."""
    options = [(-mean[i, j], abs(np.log2(lam / BASE[0])), abs(mu - BASE[1]), lam, mu)
               for (i, lam), (j, mu) in itertools.product(enumerate(LAMBDAS), enumerate(MUS)) if mu in mus]
    best = min(options)
    return best[3], best[4]


def d3a(grid: dict, ks_cols=cv.PID2_K) -> dict:
    """grid[pool] = [draws, lambda, mu, k] bucket means. Selection uses pool 0 only."""
    li, mi = LAMBDAS.index(BASE[0]), MUS.index(BASE[1])
    mean0 = grid[0].mean(axis=0)
    arms = {"arm1": {k: choose(mean0[:, :, i], (BASE[1],)) for i, k in enumerate(ks_cols)},
            "arm2": {k: choose(mean0[:, :, i], MUS) for i, k in enumerate(ks_cols)}}

    def per_draw(pool: int, weights: dict) -> np.ndarray:        # [draws, k]
        return np.stack([grid[pool][:, LAMBDAS.index(weights[k][0]), MUS.index(weights[k][1]), i]
                         for i, k in enumerate(ks_cols)], axis=1)

    def stats(pool: int, a: np.ndarray, b: np.ndarray) -> dict:
        d = a - b
        overall = d.mean(axis=1)
        se = overall.std(ddof=1) / np.sqrt(len(overall)) if len(overall) > 1 else float("nan")
        return {"pool": pool, "draws": len(overall), "score": float(a.mean()), "base": float(b.mean()),
                "delta": float(overall.mean()), "se": float(se),
                "z": float(overall.mean() / se) if se > 0 else float("nan"),
                "pos": float((overall > 0).mean()),
                "delta_k": {str(k): float(d[:, i].mean()) for i, k in enumerate(ks_cols)}}

    result = {"gate": GATE, "base": list(BASE), "arms": {}}
    for name, weights in arms.items():
        base0 = grid[0][:, li, mi, :]
        entry = {"weights": {str(k): list(v) for k, v in weights.items()},
                 "pool0": stats(0, per_draw(0, weights), base0)}
        entry["pool0_pass"] = bool(entry["pool0"]["delta"] >= GATE["pool0"])
        result["arms"][name] = entry
    result["arm2_minus_arm1_pool0"] = stats(0, per_draw(0, arms["arm2"]), per_draw(0, arms["arm1"]))
    locked = [n for n, e in result["arms"].items() if e["pool0_pass"]]
    result["pool1_scored"] = bool(locked) and 1 in grid
    if result["pool1_scored"]:
        base1 = grid[1][:, li, mi, :]
        for name, weights in arms.items():
            entry = result["arms"][name]
            entry["pool1"] = s = stats(1, per_draw(1, weights), base1)
            entry["pool1_reference_only"] = not entry["pool0_pass"]
            entry["pass"] = bool(entry["pool0_pass"] and s["delta"] >= GATE["pool1"] and s["z"] >= GATE["z"]
                                 and min(s["delta_k"].values()) >= GATE["bucket"])
        result["arm2_minus_arm1_pool1"] = stats(1, per_draw(1, arms["arm2"]), per_draw(1, arms["arm1"]))
    passed = [n for n, e in result["arms"].items() if e.get("pass")]
    result["candidate"] = "arm1" if "arm1" in passed else ("arm2" if passed else None)
    return result


def self_check() -> None:
    mean = np.zeros((len(LAMBDAS), len(MUS)))
    assert choose(mean, MUS) == BASE and choose(mean, (BASE[1],)) == BASE          # all tied -> base
    mean[LAMBDAS.index(8.0), MUS.index(1.0)] = mean[LAMBDAS.index(32.0), MUS.index(1.0)] = 1.0
    assert choose(mean, MUS) == (8.0, 1.0)                                         # same distance -> smaller
    mean[LAMBDAS.index(4.0), MUS.index(2.0)] = 2.0
    assert choose(mean, MUS) == (4.0, 2.0) and choose(mean, (1.0,)) == (8.0, 1.0)
    rng = np.random.default_rng(0)
    grid = {p: rng.normal(size=(6, len(LAMBDAS), len(MUS), 4)) * 0.001 for p in (0, 1)}
    grid[0][:, 0, MUS.index(1.0), :] += 1.0
    grid[1][:, 0, MUS.index(1.0), :] += 1.0
    out = d3a(grid)
    assert out["arms"]["arm1"]["weights"]["1"] == [1.0, 1.0] and out["candidate"] == "arm1"
    assert abs(out["arm2_minus_arm1_pool0"]["delta"]) < 1e-9
    flat = {0: np.zeros((6, len(LAMBDAS), len(MUS), 4))}
    out = d3a(flat)
    assert not out["pool1_scored"] and out["candidate"] is None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="pid_ge2e_v4_d4_hd_9600")
    parser.add_argument("--limit", type=int, default=None, help="draws per pool, for a timing run")
    args = parser.parse_args()
    self_check()
    out_dir = OUT / (args.run if args.limit is None else f"{args.run}_limit{args.limit}")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    env = load(args.run)
    meta = data.load_meta()
    table = pd.read_csv(ROOT / "data" / "raw" / "training" / "handicap.csv",
                        usecols=["rank", "source_row", "handicap"])
    env["meta"] = meta
    env["handicap"] = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                         validate="1:1")["handicap"].to_numpy().astype(np.int64)
    saved = np.load(BENCH / args.run / "embeddings.npz")
    env["vectors"] = saved[env["summary"]["style"].split("-")[1]]
    rank_of = pd.Series(meta["rank_id"].to_numpy().astype(np.int64),
                        index=meta["player_id"].astype(str).to_numpy())
    rank_of = rank_of[~rank_of.index.duplicated()]
    print(f"loaded in {time.time() - started:.0f}s", flush=True)

    check = step0(env, args.run)
    print(json.dumps(check, indent=1), flush=True)
    (out_dir / "step0.json").write_text(json.dumps(check, indent=1))
    if check["failed"]:
        raise SystemExit(f"input identity check failed: {check['failed']}")

    summary = env["summary"]
    per_k = {int(k): v for k, v in summary["lambda_per_k"].items()}
    d1_rows, d2_counts, d2_rows, hits, grid = [], [], [], [], {}
    for pool in (0, 1):
        n = DRAWS[pool] if args.limit is None else args.limit
        grid[pool] = np.empty((n, len(LAMBDAS), len(MUS), len(cv.PID2_K)))
        for i, draw in enumerate(range(FIRST, FIRST + n)):
            t0 = time.time()
            it = draw_parts(env, pool, draw, check=i == 0)
            p, l, s, c, q = it["P"], it["L"], it["S"], it["C"], it["q"]
            ks = q["k"].to_numpy()
            lam, mu = BASE
            full = p + l + lam * s + mu * c
            arms = {
                "full": full,
                "summary weights": p + l + summary["lambda"] * s + summary["mu"] * c,
                "summary per-k lambda": p + l + np.array([per_k[k] for k in ks])[:, None] * s + summary["mu"] * c,
                "no slot": p + lam * s + mu * c,
                "no prefix": l + lam * s + mu * c,
                "no opening": lam * s + mu * c,
                "opening only": p + l, "prefix only": p, "slot only": l, "style only": s,
            }
            for name, scores in arms.items():
                d1_rows.append({"pool": pool, "draw": draw, "arm": name,
                                **bucket_means(points(scores, it["candidates"], q), ks)})
            for name in ("style only", "full"):
                counts, kept = rank_gaps(arms[name], it, rank_of)
                d2_counts.append(counts.assign(pool=pool, draw=draw, arm=name))
                d2_rows.append({"pool": pool, "draw": draw, "arm": f"{name}, true rank kept",
                                **bucket_means(kept, ks)})
                d2_rows.append({"pool": pool, "draw": draw, "arm": name,
                                **bucket_means(points(arms[name], it["candidates"], q), ks)})
            if pool == 0:
                hits.append(hit_counts(env, it))
            for (a, la), (b, m) in itertools.product(enumerate(LAMBDAS), enumerate(MUS)):
                pts = points(p + l + la * s + m * c, it["candidates"], q)
                grid[pool][i, a, b] = [pts[ks == k].mean() for k in cv.PID2_K]
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
            print(f"pool {pool} draw {draw}: full {d1_rows[-10]['all']:.4f}  "
                  f"{time.time() - t0:.1f}s  peak RSS {rss:.1f} GiB", flush=True)

    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    d1 = pd.DataFrame(d1_rows)
    d1.to_csv(out_dir / "d1_per_draw.csv", index=False)
    report = {"run": args.run, "draws": {str(p): len(grid[p]) for p in grid}, "step0": check}

    print("\n== step 0: base score ==")
    for pool in (0, 1):
        part = d1[d1["pool"] == pool]
        t = paired(part[part["arm"].isin(["full", "summary weights", "summary per-k lambda"])], "full")
        print(f"pool {pool}\n{t.round(4).to_string(index=False)}")
    old_path = PAIRED / "pid_ge2e_v4_d4_hd_4800__pid_ge2e_v4_d4_hd_9600_pool1.csv"
    if args.run == "pid_ge2e_v4_d4_hd_9600" and old_path.exists():
        old = pd.read_csv(old_path)
        old = old[old["run"] == 1].set_index("draw")[COLS]
        new = d1[(d1["pool"] == 1) & (d1["arm"] == "summary weights")].set_index("draw")[COLS]
        both = old.index.intersection(new.index)
        diff = float(np.abs(old.loc[both].to_numpy() - new.loc[both].to_numpy()).max())
        report["paired_csv"] = {"draws": len(both), "max_abs_diff": diff,
                                "old_mean_all": float(old.loc[both, "all"].mean()),
                                "new_mean_all": float(new.loc[both, "all"].mean())}
        print(f"paired CSV, {len(both)} draws: max |diff| {diff:.2e}; "
              f"mean {old.loc[both, 'all'].mean():.6f} -> {new.loc[both, 'all'].mean():.6f}")

    print("\n== D1: parts (difference to full, se over draws) ==")
    report["d1"] = {}
    for pool in (0, 1):
        t = paired(d1[(d1["pool"] == pool) & ~d1["arm"].str.startswith("summary")], "full")
        t.to_csv(out_dir / f"d1_pool{pool}.csv", index=False)
        report["d1"][str(pool)] = t.to_dict(orient="records")
        print(f"pool {pool}\n{t.round(4).to_string(index=False)}")
    hit = pd.concat(hits).groupby(["candidate", "color", "n_same", "j", "ply"], as_index=False).sum()
    hit["before_rate"] = hit["before"] / hit["pairs"].where(hit["pairs"] > 0)
    hit["after_rate"] = hit["after"] / hit["pairs"].where(hit["pairs"] > 0)
    hit.to_csv(out_dir / "d1_prefix_hits_pool0.csv", index=False)
    pooled = hit.groupby(["candidate", "color", "j", "ply"], as_index=False)[["pairs", "before", "after"]].sum()
    pooled["before_rate"], pooled["after_rate"] = pooled["before"] / pooled["pairs"], pooled["after"] / pooled["pairs"]
    print("prefix hits, pool 0, all gallery sizes (full table in d1_prefix_hits_pool0.csv):")
    print(pooled.round(4).to_string(index=False))
    by_n = hit[hit["j"].isin([1, 3])].pivot_table(index=["candidate", "color", "j"], columns="n_same",
                                                  values="after_rate")
    print("after-rate by number of same-colour gallery games (j = 1, 3):")
    print(by_n.round(3).to_string())
    report["d1_prefix_hits"] = pooled.to_dict(orient="records")

    print("\n== D2: rank gap of a wrong top-1 (share; x = model share / background share) ==")
    gaps = gap_table(pd.concat(d2_counts).drop(columns="draw"))
    gaps.to_csv(out_dir / "d2_rank_gap.csv", index=False)
    print(gaps.round(3).to_string(index=False))
    d2 = pd.DataFrame(d2_rows)
    d2.to_csv(out_dir / "d2_true_rank_kept_per_draw.csv", index=False)
    report["d2_gap"] = json.loads(gaps.to_json(orient="records"))
    report["d2_true_rank_kept"] = {}
    for pool, name in itertools.product((0, 1), ("style only", "full")):
        part = d2[(d2["pool"] == pool) & d2["arm"].str.startswith(name)]
        t = paired(part, name).assign(pool=pool)
        report["d2_true_rank_kept"][f"{pool}/{name}"] = t.to_dict(orient="records")
        print(t.round(4).to_string(index=False))

    print("\n== D3-a: per-k fusion weights ==")
    np.savez(out_dir / "d3a_grid.npz", lambdas=LAMBDAS, mus=MUS, ks=cv.PID2_K,
             **{f"pool{p}": g for p, g in grid.items()})
    mean0 = grid[0].mean(axis=0)
    for i, k in enumerate(cv.PID2_K):
        print(f"pool 0, k={k} (rows lambda, columns mu):")
        print(pd.DataFrame(mean0[:, :, i], index=LAMBDAS, columns=MUS).round(4).to_string())
    report["d3a"] = d3a(grid)
    print(json.dumps(report["d3a"], indent=1))
    report["seconds"] = time.time() - started
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"\ndone in {report['seconds']:.0f}s, peak RSS {report['peak_rss_gib']:.1f} GiB -> {out_dir}")


if __name__ == "__main__":
    main()
