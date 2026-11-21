"""Diagnostic (2026-11-21, plan B): are added-set candidates (pool2_*, 10 games, either colour) the
same people as original-set candidates (pool_*, 100 black games)?

Registered in RESULTS.md before the run. CPU only, opening model of the original set (d4, pp+slot,
parameters from predict.load_params). Each added candidate's black games form one query, scored
against the 400 original candidates of a part. Two references with the same number m of games:

  control  original-set questions of the same part, truncated to m random games: their player is in
           that pool (official: the answer is always among the part's candidates)
  null     original-set questions of the other part: different people (official), so not in the pool

Statistic: margin = top-1 minus top-2 opening score. Per m bucket, the threshold is the null's 99th
percentile; overlap share estimate = (test rate - null rate) / (control rate - null rate).

    python -m scripts.identity.pid2_overlap_orig
    python -m scripts.identity.pid2_overlap_orig --null-reps 20 --quantile 0.995   # robustness (post hoc)

--null-reps R draws R null (and control) truncations per added candidate, so the threshold rests on
R x n samples instead of n; the Poisson p-value tests the test exceedances against the null rate.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity.predict import load_params, opening_scores, store_prefix, train_prefix
from scripts.identity.pid2_style_export import STORE

OUT = ROOT / "outputs" / "pid" / "overlap_orig"
PARTS = ("public", "private")
BUCKETS = ((1, 2), (3, 4), (5, 6), (7, 10))
SEED = 2026


def main() -> None:
    from scipy.stats import poisson
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-reps", type=int, default=1)
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()
    reps, qu = args.null_reps, args.quantile
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    train_meta, train_pre = train_prefix("d4")
    population = np.flatnonzero(train_meta["color"].astype(str).to_numpy() == "B")
    blocks, offset = {"pop": (0, len(population))}, len(population)
    prefixes, metas = [train_pre[population]], {}
    for name in [f"pool_{p}" for p in PARTS] + [f"pid_{p}" for p in PARTS] + [f"pool2_{p}" for p in PARTS]:
        m, c, meta = load_store(name, STORE, mmap=False)
        pre, ok = store_prefix(m, c, meta, "d4", min_plies)
        meta = meta.assign(ok=ok, row=offset + np.arange(len(meta)))
        prefixes.append(pre)
        metas[name] = meta
        blocks[name] = (offset, len(meta))
        offset += len(meta)
    prefix = np.concatenate(prefixes)
    rng = np.random.default_rng(SEED)
    report = {"params": params, "population_games": int(len(population)), "combos": {}}

    # added candidates' black games (HD <= 1, usable), grouped per candidate
    tests = {}
    for p in PARTS:
        meta = metas[f"pool2_{p}"]
        ok = meta["ok"] & (meta["color"].astype(str) == "B") & (meta["hd"].to_numpy() <= 1)
        groups = meta[ok].groupby(meta.loc[ok, "player_id"].astype(str))["row"].apply(np.array)
        tests[p] = [np.sort(g) for g in groups.to_numpy()]
        report[f"added_{p}_candidates_with_black"] = int(len(tests[p]))
        report[f"added_{p}_black_count_hist"] = np.bincount([len(g) for g in tests[p]], minlength=11).tolist()

    def matched(question_part: str, ms: list[int]) -> list[np.ndarray]:
        """one truncated original question per requested size m (random question with >= m usable games)."""
        meta = metas[f"pid_{question_part}"]
        rows = meta[meta["ok"]].groupby("question_id")["row"].apply(np.array)
        pools = {m: [r for r in rows.to_numpy() if len(r) >= m] for m in set(ms)}
        return [np.sort(rng.choice(pools[m][rng.integers(len(pools[m]))], size=m, replace=False)) for m in ms]

    def margins(pool_part: str, queries: list[np.ndarray]) -> np.ndarray:
        meta = metas[f"pool_{pool_part}"]
        players = meta["player_id"].astype(str).to_numpy()
        cands = np.array(sorted(set(players)))
        gallery = [meta["row"].to_numpy()[(players == c) & meta["ok"].to_numpy()] for c in cands]
        s = opening_scores(prefix, gallery, queries, np.arange(len(population)), params)
        top2 = -np.sort(-s, axis=1)[:, :2]
        return top2[:, 0] - top2[:, 1]

    rows = []
    for pool_part in PARTS:
        other = PARTS[1 - PARTS.index(pool_part)]
        for added_part in PARTS:
            q = tests[added_part]
            ms = [len(g) for g in q]
            mt = margins(pool_part, q)
            mc = margins(pool_part, matched(pool_part, ms * reps))
            mn = margins(pool_part, matched(other, ms * reps))
            ms_r = ms * reps
            for lo, hi in BUCKETS:
                sel = np.array([lo <= m <= hi for m in ms])
                sel_r = np.array([lo <= m <= hi for m in ms_r])
                if sel.sum() < 10:
                    continue
                t = float(np.quantile(mn[sel_r], qu))
                rt, rc, rn = float((mt[sel] > t).mean()), float((mc[sel_r] > t).mean()), float((mn[sel_r] > t).mean())
                share = (rt - rn) / (rc - rn) if rc > rn else float("nan")
                hits = int((mt[sel] > t).sum())
                pval = float(poisson.sf(hits - 1, rn * sel.sum())) if hits else 1.0
                rows.append({"pool": pool_part, "added": added_part, "m": f"{lo}-{hi}", "n": int(sel.sum()),
                             "threshold": t, "test_rate": rt, "control_rate": rc, "null_rate": rn,
                             "overlap_share": share, "hits": hits, "poisson_p": pval,
                             "test_median": float(np.median(mt[sel])), "control_median": float(np.median(mc[sel_r])),
                             "null_median": float(np.median(mn[sel_r]))})
    frame = pd.DataFrame(rows)
    tag = "" if reps == 1 and qu == 0.99 else f"_r{reps}_q{qu:g}"
    report.update({"null_reps": reps, "quantile": qu})
    frame.to_csv(OUT / f"margins{tag}.csv", index=False)
    report["table"] = frame.to_dict(orient="records")
    (OUT / f"report{tag}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "table"}, indent=1, default=float))
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
