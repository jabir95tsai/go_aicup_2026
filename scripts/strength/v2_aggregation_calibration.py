"""Pre-registered V2 question-aggregation / temperature experiment (CPU only).

Single variable: how per-game results become a question distribution; the
decoder is fixed to metrics.score_rank. Valid-S players are split in half:
S-fit fits per-k temperatures (question NLL), S-select ranks candidates.
Valid-U is never used for fitting or selection, only reported.

Candidates
    C0 mean_probability, T=1            (current official rule)
    C1 mean_log_probability, T=1        (U score already seen in A-024)
    C2 mean_log_probability, T_k fitted on S-fit
    C3 mean_probability, T_k fitted on S-fit
Gate: adopt the best S-select candidate only if it beats C0 on S-select
Equal-k by >= 0.01 and none of its temperatures sits on the grid edge.

    python -m scripts.strength.v2_aggregation_calibration --run-dir outputs/v2_strength
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from go_aicup.metrics import K_BUCKETS, rank_report
from go_aicup.tasks.strength.v2.aggregate import (decode_questions, fit_temperature,
                                                  question_logits, tempered_probabilities)
from go_aicup.tasks.strength.v2.evaluate import load_run

CANDIDATES = {"C0": ("mean_probability", False), "C1": ("mean_log_probability", False),
              "C2": ("mean_log_probability", True), "C3": ("mean_probability", True)}
GATE = 0.01


def bucket_scores(probabilities, questions) -> dict:
    report = rank_report(decode_questions(probabilities, "score_rank"),
                         questions["rank_id"].to_numpy(), questions["k"].to_numpy())
    scores = {str(k): float(report.loc[str(k), "score"]) for k in K_BUCKETS}
    return scores | {"equal_k": float(np.mean(list(scores.values())))}


def tempered_by_k(logits, questions, temperatures) -> np.ndarray:
    result = np.empty_like(logits)
    for k in K_BUCKETS:
        mask = questions["k"].to_numpy() == k
        result[mask] = tempered_probabilities(logits[mask], temperatures[str(k)])
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/v2_strength"))
    parser.add_argument("--output", type=Path, default=Path("outputs/v2_agg_calibration/result.json"))
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    run = load_run(args.run_dir)
    seen = run["seen_questions"]
    players = np.sort(seen["player_id"].astype(str).unique())
    fit_players = set(np.random.default_rng(args.seed).permutation(players)[:len(players) // 2])
    in_fit = seen["player_id"].astype(str).isin(fit_players).to_numpy()
    parts = {"S-fit": seen[in_fit].reset_index(drop=True), "S-select": seen[~in_fit].reset_index(drop=True),
             "U": run["unseen_questions"]}
    sources = {"S-fit": run["seen"], "S-select": run["seen"], "U": run["unseen"]}
    for name in ("S-fit", "S-select"):
        if sorted(set(parts[name]["k"])) != list(K_BUCKETS):
            raise ValueError(f"{name} lacks a k bucket")
    result = {"split": {name: {"questions": len(frame), "players": int(frame["player_id"].nunique()),
                               "per_k": frame["k"].value_counts().sort_index().to_dict()}
                        for name, frame in parts.items()}, "candidates": {}}
    for candidate, (method, fitted) in CANDIDATES.items():
        logits = {name: question_logits(sources[name], parts[name]["games"], method) for name in parts}
        temperatures, edges = {str(k): 1.0 for k in K_BUCKETS}, {}
        if fitted:
            for k in K_BUCKETS:
                mask = parts["S-fit"]["k"].to_numpy() == k
                temperatures[str(k)], edges[str(k)] = fit_temperature(
                    logits["S-fit"][mask], parts["S-fit"]["rank_id"].to_numpy()[mask])
        result["candidates"][candidate] = {
            "method": method, "temperatures": temperatures, "grid_edge": edges,
            **{name: bucket_scores(tempered_by_k(logits[name], parts[name], temperatures), parts[name])
               for name in ("S-select", "U")}}
    select = {name: value["S-select"]["equal_k"] for name, value in result["candidates"].items()}
    best = max(select, key=select.get)
    adopt = (best != "C0" and select[best] - select["C0"] >= GATE
             and not any(result["candidates"][best]["grid_edge"].values()))
    result["decision"] = {"best_on_S_select": best, "margin_over_C0": select[best] - select["C0"],
                          "gate": GATE, "adopted": best if adopt else "C0",
                          "U_equal_k_adopted": result["candidates"][best if adopt else "C0"]["U"]["equal_k"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
