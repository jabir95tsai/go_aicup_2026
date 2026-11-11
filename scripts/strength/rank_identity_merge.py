"""Identity-based follow-ups on the competition-2 base (RESULTS 2026-11-11).

pairs   Same-player pairs among the 800 test questions, competition-1 opening model
        (pp+slot, D4, A-029 parameters) on the question colour's own plies. Every question
        is a gallery; LR(i|j) = score of question i against gallery j minus its score against
        an empty gallery (population only); M_ij = LR(i|j) + LR(j|i). Null and recall come
        from synthetic Valid-U questions (some players hold two disjoint questions), built
        like the test (HD<=1, k in 5/10/15/20); their games are removed from the population.
export  Refit the base (-prior_736589: PriorMatched B2 on Valid-S + train20 + Valid-U) and
        check it reproduces the base file, then write
          l10        questions above the L10 unseen-null max take the matched player's rank
          pairs      each detected same-player component is scored as one merged question
          pairs-l10  both (an L10 match sets its whole component)

    python -m scripts.strength.rank_identity_merge pairs
    python -m scripts.strength.rank_identity_merge export --job ID
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

from scripts.strength.l10_check import score_questions

STORE = ROOT / "outputs" / "rank_release_20261106" / "store"
REFERENCE = ROOT / "data" / "processed" / "humansl_residual_b2" / "seed2027" / "questions_U.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
L10 = ROOT / "outputs" / "rank" / "l10_real_test.questions.csv"
BASE = ROOT / "submissions" / "rank" / "ahc-p3-noA-t20-allp-prior_736589_rank.csv"
OUT = ROOT / "outputs" / "rank" / "identity_merge"
PARTS = ("rank_public", "rank_private")
K = (5, 10, 15, 20)
PAIRED, SINGLE = 150, 250   # synthetic players per colour with two / one questions
SEED = 2027


def mutual(scores: np.ndarray) -> np.ndarray:
    """[Q, Q+1] scores (last column = empty gallery) -> symmetric M with -inf diagonal."""
    q = len(scores)
    lr = scores[:, :q] - scores[:, q:q + 1]
    m = lr + lr.T
    m[np.arange(q), np.arange(q)] = -np.inf
    return m


def reference_questions(meta: pd.DataFrame, color: int, rng) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Synthetic questions of Valid-U players: (games, player label, k)."""
    players = pd.read_parquet(REFERENCE, columns=["player_id"])["player_id"].astype(str).unique()
    own = meta[(meta["player_color"] == color) & (meta["handicap"] <= 1) & meta["player_id"].isin(players)]
    by_player = {p: g.to_numpy() for p, g in own.groupby("player_id")["game_index"]}
    two = [p for p, g in by_player.items() if len(g) >= 10]
    chosen = rng.permutation(two)[:PAIRED]
    rest = [p for p in by_player if p not in set(chosen) and len(by_player[p]) >= 5]
    singles = rng.permutation(rest)[:SINGLE]
    games, label, ks = [], [], []
    for p in chosen:
        g = rng.permutation(by_player[p])
        k1 = int(rng.choice([k for k in K if k <= len(g) - 5]))
        k2 = int(rng.choice([k for k in K if k <= len(g) - k1]))
        for part, k in ((g[:k1], k1), (g[k1:k1 + k2], k2)):
            games.append(part), label.append(p), ks.append(k)
    for p in singles:
        g = by_player[p]
        k = int(rng.choice([k for k in K if k <= len(g)]))
        games.append(rng.permutation(g)[:k]), label.append(p), ks.append(k)
    return games, np.array(label), np.array(ks)


def pairs() -> None:
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    meta = data.load_meta()
    meta = meta.merge(pd.read_csv(HANDICAP, usecols=["rank", "game_id", "handicap"]), on=["rank", "game_id"],
                      how="left", validate="one_to_one")
    if not np.array_equal(meta["game_index"].to_numpy(), np.arange(len(meta))):
        raise ValueError("meta order changed by the handicap join")
    train_pre = data.opening_prefix_d4(meta, PLIES)
    stores = {}
    for name in PARTS:
        moves, colors, q_meta = load_store(name, STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        stores[name] = (pre, ok, q_meta)
    prefix = np.concatenate([train_pre] + [stores[n][0] for n in PARTS])
    offsets = {PARTS[0]: len(train_pre), PARTS[1]: len(train_pre) + len(stores[PARTS[0]][0])}
    hashes = opening.prefix_hashes(prefix)
    rng = np.random.default_rng(SEED)
    empty = np.array([], np.int64)
    report, null_all, rows = {"params": params}, [], []
    for color in (0, 1):
        own = (meta["player_color"] == color).to_numpy()
        # reference: synthetic Valid-U questions; their players' games leave the population
        games, label, ks = reference_questions(meta, color, rng)
        ref_players = meta["player_id"].isin(set(label)).to_numpy()
        population = np.flatnonzero(own & ~ref_players)
        s = score_questions(prefix, hashes, list(games) + [empty], list(games), population, color, params)
        m = mutual(s)
        iu = np.triu_indices(len(games), 1)
        same = label[iu[0]] == label[iu[1]]
        null_all.append(m[iu][~same])
        pd.DataFrame({"m": m[iu][~same], "ka": ks[iu[0]][~same], "kb": ks[iu[1]][~same], "color": color}).to_parquet(
            OUT / f"reference_null_color{color}.parquet")
        true = pd.DataFrame({"m": m[iu][same], "k1": ks[iu[0]][same], "k2": ks[iu[1]][same]})
        report[f"reference_color{color}"] = {"questions": len(games), "true_pairs": int(same.sum()),
                                             "null_pairs": int((~same).sum())}
        true.assign(color=color).to_csv(OUT / f"reference_true_color{color}.csv", index=False)
        print(f"color {color}: reference {len(games)} questions scored", flush=True)

        # test: every test question of this colour (both parts) is a gallery
        population = np.flatnonzero(own)
        queries, ids = [], []
        for name in PARTS:
            pre, ok, q_meta = stores[name]
            in_color = q_meta["color"].astype(str).to_numpy() == ("W" if color else "B")
            index = np.flatnonzero(in_color & ok)
            for qid, r in q_meta.iloc[index].groupby("question_id", sort=False).indices.items():
                queries.append(offsets[name] + index[r])
                ids.append(qid)
        s = score_questions(prefix, hashes, queries + [empty], queries, population, color, params)
        m = mutual(s)
        iu = np.triu_indices(len(ids), 1)
        rows.append(pd.DataFrame({"a": np.array(ids)[iu[0]], "b": np.array(ids)[iu[1]], "m": m[iu],
                                  "ka": [len(queries[i]) for i in iu[0]], "kb": [len(queries[j]) for j in iu[1]],
                                  "color": color}))
        print(f"color {color}: test {len(ids)} questions scored", flush=True)

    null = np.concatenate(null_all)
    true = pd.concat([pd.read_csv(OUT / f"reference_true_color{c}.csv") for c in (0, 1)])
    test = pd.concat(rows, ignore_index=True)
    test = test[np.isfinite(test["m"])].sort_values("m", ascending=False)
    test.to_csv(OUT / "test_pairs_top.csv", index=False)
    np.save(OUT / "null_m.npy", null.astype(np.float32))
    qs = (0.5, 0.9, 0.99, 0.995, 0.999)
    report["quantiles"] = {str(q): {"null": float(np.quantile(null, q)), "test": float(np.quantile(test["m"], q))}
                           for q in qs}
    thresholds = {"null_q999": float(np.quantile(null, 0.999)), "null_q9999": float(np.quantile(null, 0.9999)),
                  "null_max": float(null.max())}
    table = {}
    for name, t in thresholds.items():
        rate = float((null > t).mean())
        table[name] = {"threshold": t, "null_rate": rate, "reference_recall": float((true["m"] > t).mean()),
                       "recall_by_min_k": true.assign(kmin=true[["k1", "k2"]].min(axis=1)).groupby("kmin")["m"]
                       .apply(lambda v: float((v > t).mean())).to_dict(),
                       "test_pairs_above": int((test["m"] > t).sum()),
                       "expected_false": rate * len(test)}
    report.update({"null_pairs": int(len(null)), "true_pairs": int(len(true)), "test_pairs": int(len(test)),
                   "true_m_median": float(true["m"].median()), "null_m_median": float(np.median(null)),
                   "thresholds": table})
    (OUT / "pairs.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


def components(pairs_frame: pd.DataFrame, threshold: float, ids: list[str]) -> dict[str, int]:
    keep = pairs_frame[pairs_frame["m"] > threshold]
    pos = {q: i for i, q in enumerate(ids)}
    graph = coo_matrix((np.ones(len(keep)), (keep["a"].map(pos), keep["b"].map(pos))), shape=(len(ids),) * 2)
    _, comp = connected_components(graph, directed=False)
    return dict(zip(ids, comp))


def export(job: str, threshold_name: str, tags: list[str]) -> None:
    import go_aicup.tasks.strength.humansl as hs
    from go_aicup.data.board_cache import RANKS
    from go_aicup.metrics import score_rank
    from go_aicup.tasks.strength import humansl_predict as hp
    from go_aicup.tasks.strength.humansl import RankModel
    from go_aicup.tasks.strength.submission_files import write_rank_csv
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_more_players as mp

    games, fit, eval_games, x_fit, x_eval, _ = mp.data(False)
    u = (eval_games["handicap"] <= 1).to_numpy()
    base = ((fit["handicap"] <= 1) & (fit["role"] != "X")).to_numpy()
    valid_s = base & (fit["role"] == "S").to_numpy()

    def stack(mask):
        return pd.concat([fit[mask], eval_games[u]], ignore_index=True), np.vstack([x_fit[mask], x_eval[u]])

    hs.SEED = hf.SEED
    model = mp.PriorMatched(mp.player_prior(stack(valid_s)[0])).fit(stack(base)[0], pd.DataFrame(),
                                                                   stack(base)[1])
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    # one feature table over both parts; private game indices shifted past the public ones
    xs, index, questions = [], [], []
    shift = 0
    for part in ("public", "private"):
        table, _ = p3.test_table(part, "hc", columns)
        q = hp.question_table(p3.WORK / f"rank_{part}")
        q["games"] = [g + shift for g in q["games"]]
        xs.append(table.to_numpy()[:, 10:]), index.append(table.index.to_numpy() + shift), questions.append(q)
        shift += int(table.index.max()) + 1
    x, index, questions = np.vstack(xs), np.concatenate(index), pd.concat(questions, ignore_index=True)
    p = model.probabilities(x, index, questions)
    pred = pd.Series([RANKS[r] for r in score_rank(p)], index=questions["question_id"])
    base_file = pd.read_csv(BASE, dtype=str).set_index("question_id")["pred_rank"]
    if (pred.loc[base_file.index] != base_file).any():
        raise ValueError("refit does not reproduce the base")
    print("base reproduced", flush=True)
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)

    report = json.loads((OUT / "pairs.json").read_text())
    threshold = report["thresholds"][threshold_name]["threshold"]
    comp = components(pd.read_csv(OUT / "test_pairs_top.csv"), threshold, list(questions["question_id"]))
    merged = pred.copy()
    groups = pd.Series(comp).groupby(pd.Series(comp)).groups
    details = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sub = questions[questions["question_id"].isin(members)]
        union = np.sort(np.concatenate(list(sub["games"])))
        mq = pd.DataFrame({"question_id": ["merged"], "games": [union], "num_games": [len(union)]})
        r = RANKS[int(score_rank(model.probabilities(x, index, mq))[0])]
        details.append({"members": list(members), "games": int(len(union)), "merged": r,
                        "before": [pred[m] for m in members]})
        merged.loc[list(members)] = r

    l10 = pd.read_csv(L10)
    l10 = l10[l10["above_null_max"]].set_index("question_id")["top1_rank"]
    with_l10 = pred.copy()
    with_l10.loc[l10.index] = l10
    both = merged.copy()
    for q, r in l10.items():
        both.loc[[m for m, c in comp.items() if c == comp[q]]] = r
    out = {"threshold": threshold_name, "components": details, "files": {}}
    for tag, series in (("l10", with_l10), ("pairs", merged), ("pairs-l10", both)):
        if tag not in tags:
            continue
        result = series.loc[template["question_id"]].rename("pred_rank").reset_index()
        changed = result["pred_rank"].to_numpy() != base_file.loc[template["question_id"]].to_numpy()
        entry = {"public": int(changed[result["question_id"].str.startswith("pub")].sum()),
                 "private": int(changed[result["question_id"].str.startswith("priv")].sum())}
        if changed.any():
            entry["file"] = write_rank_csv(result, tag, job).name
        out["files"][tag] = entry
    (OUT / "export.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["pairs", "export"])
    parser.add_argument("--job", default="local")
    parser.add_argument("--threshold", default="null_max", choices=["null_q999", "null_q9999", "null_max"])
    parser.add_argument("--tags", nargs="+", default=["l10"], choices=["l10", "pairs", "pairs-l10"],
                        help="files to write (pairs: rank agreement says the detected pairs are mostly different players)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "pairs":
        pairs()
    else:
        export(args.job, args.threshold, args.tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
