"""Competition 2: merge test questions that are the same player (2026-11-21, plan RANK_QMERGE_PLAN.md)

rank_link2 (c) found 15 test-question pairs whose style similarity is above the maximum of 1.4 million
different-player reference pairs; their base ranks agree far more often than chance. Connected pairs form
groups; a group is predicted as one question holding all its games (B2 of 751959-cvec2, k bucket of the
total game count, score_rank) and every member gets that rank.

  export  refit the 751959-cvec2 model (must reproduce its 800 rows), merged predictions,
          write <job>-qmerge.csv and the L10 overlay <job>-qm-l10.csv
  local   Valid-U simulation with the local model (Valid-S + train20 fit, seeds 2027-2031): per member
          question, official score of the merged prediction minus the separate one, for true links (same
          player) and false links (two players whose separate predictions are within one rank)

    python -m scripts.strength.rank_qmerge --job <jobid>
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
import scripts.strength.rank_prod_a0_fusion as prod

PAIRS = ROOT / "outputs" / "rank" / "link2" / "c_pairs.csv"
L10 = ROOT / "outputs" / "rank" / "l10_real_test.questions.csv"
OUT = ROOT / "outputs" / "rank" / "qmerge"
SIM_SEED, SIM_PER_CELL = 2032, 600
CELLS = ((5, 5), (5, 10), (10, 10), (10, 15), (15, 20), (10, 15, 20), (5, 5, 5, 5))


def groups_from_pairs(pairs: pd.DataFrame) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(pairs["q1"], pairs["q2"]):
        parent[find(a)] = find(b)
    out: dict[str, list[str]] = {}
    for q in parent:
        out.setdefault(find(q), []).append(q)
    return sorted((sorted(g) for g in out.values()), key=lambda g: g[0])


def merged_probabilities(model, x, game_index, members: list[list], k: list[int]) -> np.ndarray:
    questions = pd.DataFrame({"games": [list(np.concatenate(m)) for m in members], "k": k})
    return model.probabilities(x, game_index, questions)


def official(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    d = np.abs(pred - truth)
    return np.where(d == 0, 1.0, np.where(d == 1, np.exp(-1.0), 0.0))


def export(fit, eval_games, x_fit, x_eval, job: str) -> dict:
    u = (eval_games["handicap"] <= 1).to_numpy()
    stacked = pd.concat([fit, eval_games[u]], ignore_index=True)
    prior = more.player_prior(pd.concat([fit[fit["role"] == "S"], eval_games[u]], ignore_index=True))
    model = prod._fit(stacked, np.vstack([x_fit, x_eval[u]]), prior, 2027)

    xs, gis, qs, shift = [], [], [], 0
    for part in ("public", "private"):
        x, gi, questions = prod.test_data(part)
        xs.append(x)
        gis.append(gi + shift)
        q = questions.copy()
        q["games"] = [list(np.asarray(g) + shift) for g in q["games"]]
        qs.append(q)
        shift += int(gi.max()) + 1
    x, gi, questions = np.vstack(xs), np.concatenate(gis), pd.concat(qs, ignore_index=True)
    p = model.probabilities(x, gi, questions)
    sep = pd.Series([RANKS[r] for r in score_rank(p)], index=questions["question_id"])
    base = pd.read_csv(prod.BASE, dtype=str).set_index("question_id")["pred_rank"]
    if len(base) != 800 or (sep.loc[base.index] != base).any():
        raise ValueError(f"refit does not reproduce 751959-cvec2.csv ({int((sep.loc[base.index] != base).sum())})")

    groups = groups_from_pairs(pd.read_csv(PAIRS))
    by_id = questions.set_index("question_id")
    members = [[np.asarray(by_id.loc[q, "games"]) for q in g] for g in groups]
    totals = [int(sum(len(m) for m in ms)) for ms in members]
    pm = merged_probabilities(model, x, gi, members, [k_bucket(t) for t in totals])
    merged = sep.loc[base.index].copy()
    detail = []
    for g, prob, total in zip(groups, pm, totals):
        rank = RANKS[int(score_rank(prob[None])[0])]
        for q in g:
            merged[q] = rank
        detail.append({"group": g, "games": total, "k_bucket": k_bucket(total), "merged": rank,
                       "separate": [base[q] for q in g], "k": [int(by_id.loc[q, "num_games"]) for q in g],
                       "colour": [str(c) for c in by_id.loc[g, "color"]] if "color" in by_id else None,
                       "p_top": round(float(prob.max()), 3)})
    changed = merged != base
    l10 = pd.read_csv(L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    overlap = sorted(set(l10.index) & {q for g in groups for q in g})
    with_l10 = merged.copy()
    with_l10.loc[l10.index] = l10
    current = pd.read_csv(ROOT / "submissions" / "rank" / "751959-cv2-l10.csv", dtype=str).set_index("question_id")["pred_rank"]
    diff = with_l10 != current
    order = {r: i for i, r in enumerate(RANKS)}
    out = {"groups": detail, "questions_in_groups": int(sum(len(g) for g in groups)), "l10_overlap": overlap,
           "changed_public": int(changed[changed.index.str.startswith("pub")].sum()),
           "changed_private": int(changed[changed.index.str.startswith("priv")].sum()),
           "vs_current": {"public": int(diff[diff.index.str.startswith("pub")].sum()),
                          "private": int(diff[diff.index.str.startswith("priv")].sum()),
                          "steps": {str(int(s)): int(n) for s, n in
                                    (with_l10[diff].map(order) - current[diff].map(order)).value_counts().sort_index().items()}},
           "file": write_rank_csv(merged.rename("pred_rank").reset_index(), "qmerge", job).name,
           "file_l10": write_rank_csv(with_l10.rename("pred_rank").reset_index(), "qm-l10", job).name}
    return out


def draw_cells(eval_games: pd.DataFrame, rng) -> list[dict]:
    """True groups: one Valid-U player, disjoint single-colour questions with the cell's k values."""
    games = eval_games[eval_games["handicap"] <= 1]
    by = {pid: {c: g.loc[g["player_color"] == c, "game_index"].to_numpy() for c in (0, 1)}
          for pid, g in games.groupby(games["player_id"].astype(str))}
    rank = games.groupby(games["player_id"].astype(str))["rank_id"].first()
    out = []
    for cell in CELLS:
        players = [p for p, c in by.items() if len(c[0]) + len(c[1]) >= sum(cell) + 5]
        made = 0
        for pid in rng.permutation(players):
            left = {c: list(rng.permutation(by[pid][c])) for c in (0, 1)}
            qs = []
            for k in cell:
                colours = [c for c in (0, 1) if len(left[c]) >= k]
                if not colours:
                    break
                c = colours[rng.integers(len(colours))]
                qs.append(np.array(left[c][:k]))
                left[c] = left[c][k:]
            if len(qs) == len(cell):
                out.append({"cell": cell, "player": pid, "rank": int(rank[pid]), "questions": qs})
                made += 1
            if made == SIM_PER_CELL:
                break
    return out


def score_groups(model, x, gi, groups: list[dict], truth_of) -> tuple[np.ndarray, np.ndarray]:
    """(separate, merged) official scores per member question, in group order."""
    flat = [q for g in groups for q in g["questions"]]
    sep_p = model.probabilities(x, gi, pd.DataFrame({"games": [list(q) for q in flat],
                                                      "k": [k_bucket(len(q)) for q in flat]}))
    sep = score_rank(sep_p)
    merged_p = merged_probabilities(model, x, gi, [g["questions"] for g in groups],
                                    [k_bucket(sum(len(q) for q in g["questions"])) for g in groups])
    merged = np.repeat(score_rank(merged_p), [len(g["questions"]) for g in groups])
    truth = np.array([t for g in groups for t in truth_of(g)])
    return official(sep, truth), official(merged, truth), sep


def local(fit, eval_games, x_fit, x_eval) -> dict:
    rng = np.random.default_rng(SIM_SEED)
    true_groups = draw_cells(eval_games, rng)
    gi = eval_games["game_index"].to_numpy()
    prior = more.player_prior(fit[fit["role"] == "S"])
    report = {}
    for seed in prod.SEEDS:
        model = prod._fit(fit, x_fit, prior, seed)
        s_sep, s_mrg, sep_rank = score_groups(model, x_eval, gi, true_groups, lambda g: [g["rank"]] * len(g["questions"]))
        # false links: pair groups of the same cell from different players whose separate predictions of the
        # first members are within one rank (the observed links all agree within one rank)
        first = np.cumsum([0] + [len(g["questions"]) for g in true_groups])[:-1]
        false_groups, cursor = [], 0
        for cell in CELLS:
            idx = [i for i, g in enumerate(true_groups) if g["cell"] == cell]
            for a, b in zip(idx[0::2], idx[1::2]):
                if abs(int(sep_rank[first[a]]) - int(sep_rank[first[b]])) <= 1:
                    ga, gb = true_groups[a], true_groups[b]
                    half = len(cell) // 2 or 1
                    qs = ga["questions"][:half] + gb["questions"][half:] if len(cell) > 1 else ga["questions"]
                    ranks = [ga["rank"]] * half + [gb["rank"]] * (len(cell) - half)
                    false_groups.append({"cell": cell, "questions": qs, "ranks": ranks})
        f_sep, f_mrg, _ = score_groups(model, x_eval, gi, false_groups, lambda g: g["ranks"])
        entry = {}
        for name, groups, sep_s, mrg_s in (("true", true_groups, s_sep, s_mrg), ("false", false_groups, f_sep, f_mrg)):
            cells = np.array([str(g["cell"]) for g in groups for _ in g["questions"]])
            kq = np.array([len(q) for g in groups for q in g["questions"]])
            entry[name] = {"questions": int(len(sep_s)), "separate": round(float(sep_s.mean()), 4),
                           "merged": round(float(mrg_s.mean()), 4), "delta": round(float((mrg_s - sep_s).mean()), 4),
                           "delta_by_cell": {c: round(float((mrg_s - sep_s)[cells == c].mean()), 4) for c in sorted(set(cells))},
                           "delta_by_member_k": {str(k): round(float((mrg_s - sep_s)[kq == k].mean()), 4)
                                                 for k in K_BUCKETS if (kq == k).any()}}
        t, f = entry["true"]["delta"], entry["false"]["delta"]
        entry["break_even_precision"] = round(-f / (t - f), 3) if t > 0 > f else None
        report[str(seed)] = entry
        print(seed, json.dumps({n: {k: v for k, v in e.items() if k in ("questions", "delta")} if isinstance(e, dict) else e
                                for n, e in entry.items()}), flush=True)
    keys = ("true", "false")
    report["mean"] = {n: round(float(np.mean([report[str(s)][n]["delta"] for s in prod.SEEDS])), 4) for n in keys}
    report["mean_by_cell"] = {n: {c: round(float(np.mean([report[str(s)][n]["delta_by_cell"][c] for s in prod.SEEDS])), 4)
                                  for c in report[str(prod.SEEDS[0])][n]["delta_by_cell"]} for n in keys}
    t, f = report["mean"]["true"], report["mean"]["false"]
    report["break_even_precision"] = round(-f / (t - f), 3) if t > 0 > f else None
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", default="local")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fit, eval_games, x_fit, x_eval = prod.train_data()
    report = {"export": export(fit, eval_games, x_fit, x_eval, args.job)}
    print(json.dumps(report["export"], indent=1), flush=True)
    (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))
    if not args.skip_local:
        report["local"] = local(fit, eval_games, x_fit, x_eval)
        print(json.dumps({k: report["local"][k] for k in ("mean", "mean_by_cell", "break_even_precision")}, indent=1))
        (OUT / f"result_{args.job}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
