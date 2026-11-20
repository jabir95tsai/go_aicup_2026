"""LB probe (11/20, user): the 751959-cvec2 base with the H block dropped (C v2 + HD1 flags only).

Features come from rank_prod_a0_fusion (the post-11/16 rebuild that reproduces 751959-cvec2):
x = [H 30 | C v2 20 | HD1 flags 2]; the C-only arm keeps columns 30:52. Local: Valid-U E1 / E2,
seeds 2027-2031, B2 on Valid-S + train20 HD<=1 (= the 11/11 ablation, now with C v2). Export: fit on
Valid-S + train20 + Valid-U HD<=1, prior-matched, seed 2027; the full arm must reproduce 751959-cvec2.csv.

    python -m scripts.strength.rank_c_only --job <jobid>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.tasks.strength.submission_files import write_rank_csv

import scripts.strength.humansl_hd_filter as hf
import scripts.strength.rank_more_players as more
import scripts.strength.rank_prod_a0_fusion as prod

ARMS = {"hc": np.arange(52), "conly": np.arange(30, 52)}
OUT = prod.OUT.parent / "c_only"


def local(fit, eval_games, x_fit, x_eval) -> dict:
    sets = prod.question_sets()
    prior = more.player_prior(fit[fit["role"] == "S"])
    seeds = {}
    for seed in prod.SEEDS:
        entry = {}
        for name, q in sets.items():
            k = q["k"].to_numpy()
            scores = {a: hf.predict(prod._fit(fit, x_fit[:, c], prior, seed), eval_games, x_eval[:, c], q)
                      for a, c in ARMS.items()}
            entry[name] = {a: hf.by_k(s, k) for a, s in scores.items()}
            entry[name]["diff"] = {str(b): round(float(scores["conly"][k == b].mean() - scores["hc"][k == b].mean()), 4)
                                   for b in K_BUCKETS}
            entry[name]["diff"]["equal_k"] = round(hf.equal_k(scores["conly"], k) - hf.equal_k(scores["hc"], k), 4)
        seeds[str(seed)] = entry
        print(seed, json.dumps({n: e["diff"] for n, e in entry.items()}), flush=True)
    mean = {n: {key: round(float(np.mean([seeds[str(s)][n]["diff"][key] for s in prod.SEEDS])), 4)
                for key in [str(b) for b in K_BUCKETS] + ["equal_k"]} for n in sets}
    hc_ref = json.loads((prod.OUT / "step1.json").read_text())["local"] if (prod.OUT / "step1.json").exists() else None
    if hc_ref and any(seeds["2027"][n]["hc"] != hc_ref[n] for n in sets):
        raise ValueError("H+C seed-2027 local scores differ from prod_a0_fusion step1")
    return {"seeds": seeds, "mean_diff_conly_minus_hc": mean}


def export(fit, eval_games, x_fit, x_eval, job: str) -> dict:
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit, eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    x_all = np.vstack([x_fit, x_eval[u]])
    models = {a: prod._fit(stacked, x_all[:, c], prior, 2027) for a, c in ARMS.items()}
    out = {a: [] for a in ARMS}
    k = {}
    for part in ("public", "private"):
        x, index, questions = prod.test_data(part)
        k |= dict(zip(questions["question_id"], questions["num_games"]))
        for a, c in ARMS.items():
            p = models[a].probabilities(x[:, c], index, questions)
            out[a].append(pd.DataFrame({"question_id": questions["question_id"],
                                        "pred_rank": [RANKS[r] for r in score_rank(p)]}))
    base = pd.read_csv(prod.BASE, dtype=str).set_index("question_id")["pred_rank"]
    result = {a: pd.concat(f).set_index("question_id")["pred_rank"].loc[base.index] for a, f in out.items()}
    if len(result["hc"]) != 800 or (result["hc"] != base).any():
        raise ValueError(f"H+C refit does not reproduce 751959-cvec2.csv ({int((result['hc'] != base).sum())})")
    changed = result["conly"] != base
    order = {r: i for i, r in enumerate(RANKS)}
    step = result["conly"][changed].map(order) - base[changed].map(order)
    k = pd.Series(k)
    return {"public": int(changed[changed.index.str.startswith("pub")].sum()),
            "private": int(changed[changed.index.str.startswith("priv")].sum()),
            "public_by_k": {str(b): int((changed & base.index.str.startswith("pub") & (k.loc[base.index] == b).to_numpy()).sum())
                            for b in K_BUCKETS},
            "step_counts": {str(int(s)): int(n) for s, n in step.value_counts().sort_index().items()},
            "distribution": result["conly"].value_counts().reindex(RANKS, fill_value=0).tolist(),
            "file": write_rank_csv(result["conly"].reset_index(), "conly", job).name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    job = parser.parse_args().job
    OUT.mkdir(parents=True, exist_ok=True)
    fit, eval_games, x_fit, x_eval = prod.train_data()
    assert x_fit.shape[1] == 52 and x_eval.shape[1] == 52, (x_fit.shape, x_eval.shape)
    report = {"export": export(fit, eval_games, x_fit, x_eval, job)}
    print(json.dumps(report["export"]), flush=True)
    report["local"] = local(fit, eval_games, x_fit, x_eval)
    print(json.dumps(report["local"]["mean_diff_conly_minus_hc"]), flush=True)
    (OUT / f"result_{job}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
