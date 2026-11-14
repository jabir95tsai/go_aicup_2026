"""L10 with two fingerprints: test questions whose player is a labelled training player (RESULTS 2026-11-13)

l10_check (opening model) and rank_own_link (k-game GE2E vectors) each recognise a training player
in a competition-2 test question; 27 questions get the same player from both. Here both scores are
calibrated together on reference questions and the detected questions take the matched player's rank.

Scores per question against the 8,007 training players:
  o  opening pp+slot log-likelihood (D4, A-029 parameters) on the question colour's plies; galleries
     are each player's HD<=1 games of that colour, reference games removed
  g  cosine of the question's mean GE2E vector with the player centroid (rank_opp_link gallery)
  f  o + lam * w * g  (w = GE2E scale, as the competition-1 fusion)
Reference: REF training players per colour and k with a question of games outside the GE2E gallery
("seen"); the same rows with the own column removed in both scores are the unseen null.
Rules: fused gap (top-1 minus top-2 of f) above the per-k null maximum, for lam in LAMBDAS; and
"agree" (o and g have the same top-1 player). Selection uses reference half A (players split at
random): lam with the highest seen recall; agree is added when its null rate on A is <= AGREE_FP.
Half B reports false-positive rate, recall and precision of the chosen rule (thresholds from A).
The test questions use thresholds from the whole reference null.

    python -m scripts.strength.rank_link plan
    python -m scripts.strength.rank_link embed                  # 1 GPU
    python -m scripts.strength.rank_link analyze
    python -m scripts.strength.rank_link export --job ID [--base 751959-cvec2.csv]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

import scripts.strength.rank_opp_link as ol
import scripts.strength.rank_own_link as wl
from scripts.strength.l10_check import score_questions

OUT = ol.OUT / "link"
wl.OUT, wl.REF, wl.SEED = OUT, 500, 2029
LAMBDAS = (0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
AGREE_FP = 0.001
SUMMARY = ROOT / "outputs" / "pid" / "pid2_bench" / ol.RUN.name / "summary.json"


def reduce(o: np.ndarray, g: np.ndarray, w: float) -> dict:
    """Per question: top-1 of o and g, and for each lam the top-1 and gap of the fused score."""
    out = {"arg_o": o.argmax(1), "arg_g": g.argmax(1)}
    for lam in LAMBDAS:
        arg, _, gap = ol.top2(o + lam * w * g)
        out[f"arg_{lam}"], out[f"gap_{lam}"] = arg, gap
    return out


def analyze(_args) -> None:
    params = load_params("d4")
    w = json.loads(SUMMARY.read_text())["w"]
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    meta = ol.load_train()
    candidates, column, centroids, cand_rank, _ = wl.centroids_and_ranks(meta)
    player = meta["player_id"].astype(str).to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd01 = meta["handicap"].to_numpy() <= 1
    train_pre = data.opening_prefix_d4(meta, PLIES)

    ref = pd.read_parquet(OUT / "ref.parquet")
    packed = np.load(OUT / "ref_vectors.npz")
    position = pd.Series(np.arange(len(packed["game_index"])), index=packed["game_index"])
    held = ref["game_index"].to_numpy()

    tests, blocks, offsets, size = {}, [train_pre], {}, len(train_pre)
    for name in ol.PARTS:
        moves, colors, q_meta = load_store(name, ol.STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        if not ok.all():
            raise ValueError(f"{name}: {int((~ok).sum())} games too short for the opening model")
        tests[name] = (q_meta, np.load(ol.OUT / f"{name}_own.npy"))
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
    prefix = np.concatenate(blocks)
    hashes = opening.prefix_hashes(prefix)

    ref_rows, test_rows = [], []
    for color in (0, 1):
        own = (pcolor == color) & hd01
        keep = own.copy()
        keep[held] = False
        galleries = [np.array([], np.int64)] * len(candidates)
        for p, rows in pd.Series(np.flatnonzero(keep)).groupby(player[keep]):
            galleries[column[p]] = rows.to_numpy()
        population = np.flatnonzero(own)

        part = ref[ref["color"] == color]
        groups = part.groupby("question", sort=False)
        names = list(groups.groups)
        by_name = groups.indices
        ref_games = part["game_index"].to_numpy()
        queries = [ref_games[by_name[q]] for q in names]
        info = groups[["k", "player_id"]].first().loc[names]
        truth = column.loc[info["player_id"].astype(str)].to_numpy()
        o = score_questions(prefix, hashes, galleries, queries, population, color, params)
        _, vectors = wl.question_vectors(packed["vectors"][position.loc[part["game_index"]].to_numpy()],
                                         part["question"].to_numpy())
        g = vectors @ centroids.T
        seen = reduce(o, g, w)
        rows = np.arange(len(truth))
        o[rows, truth] = -np.inf
        g[rows, truth] = -np.inf
        null = reduce(o, g, w)
        frame = pd.DataFrame({"question": names, "k": info["k"].to_numpy(), "color": color,
                              "player_id": info["player_id"].astype(str).to_numpy(), "truth": truth})
        for key, value in seen.items():
            frame[f"seen_{key}"] = value
        for key, value in null.items():
            frame[f"null_{key}"] = value
        ref_rows.append(frame)
        print(f"colour {color}: {len(names)} reference questions", flush=True)

        for name, (q_meta, own_vectors) in tests.items():
            in_color = (q_meta["color"].astype(str).to_numpy() == ("W" if color else "B"))
            index = np.flatnonzero(in_color)
            if not len(index):
                continue
            groups = q_meta.iloc[index].groupby("question_id", sort=False)
            q_names = list(groups.groups)
            by_name = groups.indices
            queries = [offsets[name] + index[by_name[q]] for q in q_names]
            o = score_questions(prefix, hashes, galleries, queries, population, color, params)
            v_names, vectors = wl.question_vectors(own_vectors[index], q_meta["question_id"].to_numpy()[index])
            if list(v_names) != q_names:
                raise ValueError("question order differs between the two scores")
            g = vectors @ centroids.T
            frame = pd.DataFrame({"part": name, "question_id": q_names, "color": color,
                                  "k": groups["num_games"].first().loc[q_names].to_numpy()})
            for key, value in reduce(o, g, w).items():
                frame[key] = value
            test_rows.append(frame)
            print(f"colour {color}: {name} {len(q_names)} questions", flush=True)

    ref_table, test_table = pd.concat(ref_rows, ignore_index=True), pd.concat(test_rows, ignore_index=True)
    ref_table.to_parquet(OUT / "ref_scores.parquet", index=False)
    test_table.to_parquet(OUT / "test_scores.parquet", index=False)
    decide(ref_table, test_table, candidates, cand_rank)


def flags(table: pd.DataFrame, prefix: str, lam: float, thresholds: dict, agree: bool):
    """(flag, matched candidate) under the rule; the fused top-1 wins when both parts fire."""
    k = table["k"].to_numpy()
    limit = np.array([thresholds[int(x)] for x in k])
    fused = table[f"{prefix}gap_{lam}"].to_numpy() > limit
    same = table[f"{prefix}arg_o"].to_numpy() == table[f"{prefix}arg_g"].to_numpy()
    match = np.where(fused, table[f"{prefix}arg_{lam}"].to_numpy(), table[f"{prefix}arg_g"].to_numpy())
    return (fused | same) if agree else fused, match, fused, same


def decide(ref: pd.DataFrame, test: pd.DataFrame, candidates, cand_rank) -> None:
    rng = np.random.default_rng(wl.SEED)
    players = np.array(sorted(ref["player_id"].unique()))
    half_a = set(players[rng.permutation(len(players))[:len(players) // 2]])
    a = ref[ref["player_id"].isin(half_a)].reset_index(drop=True)
    b = ref[~ref["player_id"].isin(half_a)].reset_index(drop=True)
    result = {"reference": {"questions": int(len(ref)), "half_a": int(len(a)), "half_b": int(len(b)),
                            "per_k": ref.groupby("k").size().to_dict()}, "selection": {}}
    result["reference"]["seen_top1"] = {
        name: {str(k): float((g[f"seen_{col}"] == g["truth"]).mean()) for k, g in ref.groupby("k")}
        for name, col in (("opening", "arg_o"), ("ge2e", "arg_g"), *((f"fused_{lam}", f"arg_{lam}") for lam in LAMBDAS))}

    def thresholds(table, lam):
        return {int(k): float(g[f"null_gap_{lam}"].max()) for k, g in table.groupby("k")}

    best = None
    for lam in LAMBDAS:
        flag, match, _, _ = flags(a, "seen_", lam, thresholds(a, lam), False)
        recall = float((flag & (match == a["truth"].to_numpy())).mean())
        result["selection"][f"lam_{lam}"] = {"recall_a": recall}
        if best is None or recall > best[1]:
            best = (lam, recall)
    lam = best[0]
    agree_fp_a = float((a["null_arg_o"] == a["null_arg_g"]).mean())
    use_agree = agree_fp_a <= AGREE_FP
    result["selection"].update({"lam": lam, "agree_null_rate_a": agree_fp_a, "use_agree": bool(use_agree)})

    t_a = thresholds(a, lam)
    null_flag, _, null_fused, null_same = flags(b, "null_", lam, t_a, use_agree)
    seen_flag, seen_match, seen_fused, seen_same = flags(b, "seen_", lam, t_a, use_agree)
    right = seen_match == b["truth"].to_numpy()
    result["half_b"] = {
        "false_positive_rate": float(null_flag.mean()), "false_positives": int(null_flag.sum()),
        "false_positives_fused": int(null_fused.sum()), "false_positives_agree": int(null_same.sum()),
        "recall": float((seen_flag & right).mean()), "flagged_seen": float(seen_flag.mean()),
        "precision_seen": float(right[seen_flag].mean()) if seen_flag.any() else None,
        "recall_by_k": {str(k): float((seen_flag & right)[b["k"].to_numpy() == k].mean()) for k in K_BUCKETS},
        "recall_fused_only": float((seen_fused & (b[f"seen_arg_{lam}"].to_numpy() == b["truth"].to_numpy())).mean()),
        "recall_agree_only": float((seen_same & (b["seen_arg_g"].to_numpy() == b["truth"].to_numpy())).mean())}
    result["half_b"]["expected_false_positives_in_800"] = round(800 * result["half_b"]["false_positive_rate"], 2)

    t_all = thresholds(ref, lam)
    flag, match, fused, same = flags(test, "", lam, t_all, use_agree)
    out = test.loc[flag, ["part", "question_id", "k", "color"]].copy()
    out["player"] = candidates[match[flag]]
    out["matched_rank"] = [wl.RANKS[r] for r in cand_rank[match[flag]]]
    out["fused"], out["agree"] = fused[flag], same[flag]
    out["gap"] = test.loc[flag, f"gap_{lam}"].to_numpy()
    out.to_csv(OUT / "links.csv", index=False)
    result["test"] = {"flagged": out.groupby("part").size().to_dict(), "by_k": out.groupby("k").size().to_dict(),
                      "fused": int(fused.sum()), "agree": int(same.sum()), "thresholds": t_all,
                      "implied_training_share": round(float(flag.mean()) / max(result["half_b"]["recall"], 1e-9), 4)}
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=int))
    print(json.dumps(result, indent=1, default=int))


def export(args) -> None:
    from go_aicup.tasks.strength.submission_files import write_rank_csv

    base = pd.read_csv(ROOT / "submissions" / "rank" / args.base, dtype=str).set_index("question_id")["pred_rank"]
    links = pd.read_csv(OUT / "links.csv").set_index("question_id")
    result = base.copy()
    result.loc[links.index] = links["matched_rank"]
    changed = result != base
    detail = links.assign(before=base.loc[links.index], changed=changed.loc[links.index]).reset_index()
    detail["rank_diff"] = (detail["matched_rank"].map(wl.RANKS.index) - detail["before"].map(wl.RANKS.index)).abs()
    out = {"base": args.base, "links": int(len(links)), "same_as_base": float((~detail["changed"]).mean()),
           "within1_of_base": float((detail["rank_diff"] <= 1).mean()),
           "changed_public": int(changed[changed.index.str.startswith("pub")].sum()),
           "changed_private": int(changed[changed.index.str.startswith("priv")].sum())}
    if changed.any():
        out["file"] = write_rank_csv(result.rename("pred_rank").reset_index(), args.tag, args.job).name
    detail.to_csv(OUT / f"export_{args.job}-{args.tag}.csv", index=False)
    (OUT / f"export_{args.job}-{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(detail[["question_id", "k", "matched_rank", "before", "fused", "agree"]].to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(fn=wl.plan)
    sub.add_parser("embed").set_defaults(fn=wl.embed)
    sub.add_parser("analyze").set_defaults(fn=analyze)
    sub.add_parser("decide").set_defaults(fn=lambda _a: decide(
        pd.read_parquet(OUT / "ref_scores.parquet"), pd.read_parquet(OUT / "test_scores.parquet"),
        *(lambda c: (c[0], c[3]))(wl.centroids_and_ranks(ol.load_train()))))
    p = sub.add_parser("export")
    p.add_argument("--job", required=True)
    p.add_argument("--base", default="751959-cvec2.csv")
    p.add_argument("--tag", default="cv2-link")
    p.set_defaults(fn=export)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
