"""Hard-negative premise check (2026-11-20, registered in RESULTS.md before the run).

Question: is "true player A is ranked below candidate B" a stable player pair, or noise of one query
game / one 10-game gallery? No training; teacher = current ogsa12w added-set form
    full  = rest + 16 w 1.5 (0.25 cos_seqB + 0.75 mean(cos_A1, cos_A2)),   style = the bracket alone.
Bench pool 0 (600 players never seen by an encoder, 40 games each); draws 2100-2107; every player gets
10 single-game (k=1) queries per draw from its non-gallery games.

Mining (PID_HARD_NEGATIVE_PLAN.md section 2): draws 2100-2103, first query of each player only.
A hit A->B = B in the full top 5, B above A in full, and B above A in style. Pair = hits >= 2,
at most 3 B per A (hits desc, mean style margin desc, player id).
Check: (i) new gallery = draws 2104-2107, queries exclude A's 4 mining games;
(ii) same gallery = the other 9 queries of a mining draw in which the pair hit.
Report P(B above A) and P(B top 1) by hit count, the base rate, coverage, rank gaps and hubs.

  python -m scripts.identity.pid2_hardneg_diag            # GPU: scores + analysis
  python -m scripts.identity.pid2_hardneg_diag --analyse  # CPU: analysis of saved scores only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv

POOL = 0
MINE = tuple(range(2100, 2104))
HOLD = tuple(range(2104, 2108))
PER_DRAW = 10
MIN_HITS, MAX_PAIRS = 2, 3
OUT = Path("outputs/pid/hard_negative/diag")
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
WEIGHTS = np.exp(-np.arange(5.0))


def queries(bank: pd.DataFrame, draw: int, exclude: dict[str, set[int]]):
    """candidates, gallery and one row per (player, slot) single-game query."""
    candidates, gallery, _ = cv.draw_pid2(bank, POOL, draw)
    part = bank[bank["pool"] == POOL]
    games = {p: np.sort(g["game"].to_numpy()) for p, g in part.groupby("player_id")}
    rng = np.random.default_rng([draw, POOL, 77])
    truth, slot, game = [], [], []
    for i, p in enumerate(candidates):
        left = np.setdiff1d(games[p], gallery[i])
        left = left[~np.isin(left, list(exclude.get(p, ())))]
        chosen = rng.permutation(left)[:PER_DRAW]
        assert len(chosen) == PER_DRAW, p
        truth += [i] * PER_DRAW
        slot += list(range(PER_DRAW))
        game += chosen.tolist()
    return candidates, gallery, np.array(truth), np.array(slot), np.array(game)


def compute() -> None:
    import torch

    from go_aicup.tasks.identity import data, twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
    from scripts.identity.pid2_ogs_a2_eval import ENC, RUN
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    OUT.mkdir(parents=True, exist_ok=True)
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda")
    vec = {}
    for name, (cache, head_file) in ENC.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, _ = load_head(Path(head_file), device)
        vec[name] = game_vectors(head, store, np.arange(len(store.games)), device)
        del store
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda games: b1["keys"][row[games]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popf["keys"], color[popf["game_index"]])
    mined: dict[str, set[int]] = {}
    for draw in MINE + HOLD:
        candidates, gallery, truth, slot, game = queries(env["bank"], draw, mined if draw in HOLD else {})
        if draw in MINE:
            for p, g in zip(candidates[truth[slot == 0]], game[slot == 0]):
                mined.setdefault(p, set()).add(int(g))
        query = [np.array([g]) for g in game]
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4)
                + s1["lambda1"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]]
                + s1["lambda2"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]])
        g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
        c = {k: twocolor.style_scores(v, g, qq) for k, v in vec.items()}
        style = 0.25 * c["B"] + 0.75 * 0.5 * (c["A1"] + c["A2"])
        np.savez(OUT / f"draw{draw}.npz", full=(rest + scale * 1.5 * style).astype(np.float32),
                 style=style.astype(np.float32), truth=truth, slot=slot, game=game, candidates=candidates)
        print(f"draw {draw} done", flush=True)


def above(scores: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return scores > scores[np.arange(len(truth)), truth][:, None]


def analyse() -> dict:
    from go_aicup.tasks.identity import data

    d = {draw: dict(np.load(OUT / f"draw{draw}.npz", allow_pickle=True)) for draw in MINE + HOLD}
    candidates = d[MINE[0]]["candidates"]
    assert all(np.array_equal(x["candidates"], candidates) for x in d.values())
    n = len(candidates)
    report: dict = {"accuracy": {}}
    for draw, x in d.items():
        for name in ("full", "style"):
            order = np.argsort(-x[name], axis=1, kind="stable")[:, :5]
            hit = order == x["truth"][:, None]
            report["accuracy"].setdefault(name, []).append(
                {"draw": draw, "top1": float(hit[:, 0].mean()), "score": float((hit * WEIGHTS).sum(1).mean())})

    hits = np.zeros((n, n), dtype=np.int64)
    margin = np.zeros((n, n))
    hit_draw = {}
    for draw in MINE:
        x = d[draw]
        m = x["slot"] == 0
        full, style, truth = x["full"][m], x["style"][m], x["truth"][m]
        top5 = np.zeros_like(full, dtype=bool)
        np.put_along_axis(top5, np.argsort(-full, axis=1, kind="stable")[:, :5], True, axis=1)
        h = top5 & above(full, truth) & above(style, truth)
        per = np.zeros((n, n), dtype=bool)
        per[truth] = h
        hit_draw[draw] = per
        hits += per
        margin[truth] += np.where(h, style - style[np.arange(len(truth)), truth][:, None], 0.0)
    np.fill_diagonal(hits, 0)

    pairs = []
    for a in range(n):
        bs = np.flatnonzero(hits[a] >= MIN_HITS)
        bs = sorted(bs, key=lambda b: (-hits[a, b], -margin[a, b] / hits[a, b], candidates[b]))[:MAX_PAIRS]
        pairs += [(a, b) for b in bs]
    selected = np.zeros((n, n), dtype=bool)
    for a, b in pairs:
        selected[a, b] = True

    def rates(draws, mask_fn):
        num_above, num_top1, count = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n)
        for draw in draws:
            x = d[draw]
            m = mask_fn(x)
            full, truth = x["full"][m], x["truth"][m]
            np.add.at(num_above, truth, above(full, truth))
            top1 = np.zeros_like(full)
            top1[np.arange(len(truth)), full.argmax(1)] = 1.0
            np.add.at(num_top1, truth, top1)
            np.add.at(count, truth, 1)
        return num_above / count[:, None], num_top1 / count[:, None]

    new_above, new_top1 = rates(HOLD, lambda x: np.ones(len(x["truth"]), dtype=bool))
    off = ~np.eye(n, dtype=bool)
    # same gallery: other 9 queries of a mining draw, only for pairs that hit in that draw
    same_num, same_cnt = np.zeros((n, n)), np.zeros((n, n))
    same_all = []
    for draw in MINE:
        x = d[draw]
        m = x["slot"] > 0
        num =np.zeros((n, n))
        cnt = np.zeros(n)
        np.add.at(num, x["truth"][m], above(x["full"][m], x["truth"][m]))
        np.add.at(cnt, x["truth"][m], 1)
        rate = num / cnt[:, None]
        same_all.append(rate[off].mean())
        same_num += np.where(hit_draw[draw], rate, 0.0)
        same_cnt += hit_draw[draw]

    table = []
    groups = [("0", off & (hits == 0)), ("1", off & (hits == 1)), ("2", off & (hits == 2)),
              ("3", off & (hits == 3)), ("4", off & (hits == 4)), (">=2", off & (hits >= 2)),
              ("selected", selected)]
    for name, g in groups:
        sg = g & (same_cnt > 0)
        table.append({"hits": name, "pairs": int(g.sum()),
                      "new_gallery_above": float(new_above[g].mean()) if g.any() else None,
                      "new_gallery_top1": float(new_top1[g].mean()) if g.any() else None,
                      "same_gallery_above": float((same_num[sg] / same_cnt[sg]).mean()) if sg.any() else None})
    report["by_hits"] = table
    base = float(new_above[off].mean())
    report["base_new_gallery_above"] = base
    report["base_same_gallery_above"] = float(np.mean(same_all))
    report["base_new_gallery_top1"] = float(new_top1[off].mean())

    meta = data.load_meta()
    prank = meta.groupby(meta["player_id"].astype(str))["rank"].first()
    level = np.array([RANKS.index(str(prank[str(p)])) for p in candidates])
    gap = np.abs(level[:, None] - level[None, :])
    dist = lambda m: {k: float(v) for k, v in zip(("same", "adjacent", ">=2"),
                                                     [(gap[m] == 0).mean(), (gap[m] == 1).mean(), (gap[m] >= 2).mean()])}
    report["rank_gap"] = {"selected": dist(selected), "all_pairs": dist(off)}
    by_b = selected.sum(axis=0)
    top = np.argsort(-by_b, kind="stable")[:10]
    report["hubs"] = {"top10": [{"player": str(candidates[b]), "rank": RANKS[level[b]], "as_B": int(by_b[b])}
                                for b in top],
                      "share_top10": float(by_b[top].sum() / max(by_b.sum(), 1)),
                      "distinct_B": int((by_b > 0).sum())}
    report["coverage_A"] = float(selected.any(axis=1).mean())
    report["pairs_selected"] = int(selected.sum())

    sel = table[-1]["new_gallery_above"]
    one = table[1]["new_gallery_above"]
    same_sel = table[-1]["same_gallery_above"]
    report["verdict"] = {
        "ratio_selected_vs_one_hit": sel / one if sel is not None and one else None,
        "ratio_selected_vs_base": sel / base if sel is not None else None,
        "ratio_same_vs_new_gallery_selected": same_sel / sel if sel and same_sel is not None else None,
    }
    v = report["verdict"]
    report["verdict"]["premise_holds"] = bool(
        v["ratio_selected_vs_one_hit"] is not None and v["ratio_selected_vs_one_hit"] >= 1.5
        and v["ratio_selected_vs_base"] >= 5 and report["coverage_A"] >= 0.20)
    report["verdict"]["gallery_noise_dominant"] = bool(
        v["ratio_same_vs_new_gallery_selected"] is not None and v["ratio_same_vs_new_gallery_selected"] >= 2)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    pd.DataFrame([{"A": candidates[a], "B": candidates[b], "hits": int(hits[a, b]),
                   "mean_style_margin": margin[a, b] / hits[a, b],
                   "rank_A": RANKS[level[a]], "rank_B": RANKS[level[b]],
                   "new_gallery_above": new_above[a, b], "new_gallery_top1": new_top1[a, b]}
                  for a, b in pairs]).to_csv(OUT / "pairs.csv", index=False)
    print(json.dumps(report, indent=1), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analyse", action="store_true", help="skip scoring; analyse saved draws")
    args = parser.parse_args()
    if not args.analyse:
        compute()
    analyse()


if __name__ == "__main__":
    main()
