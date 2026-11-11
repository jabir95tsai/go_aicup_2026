"""Are the opponents in competition-2 test games training players? (RESULTS 2026-11-11)

Each game is one query: the opponent's plies (the colour opposite the question colour)
are scored with the competition-1 opening model (pp+slot, D4, A-029 parameters) against
every training player's games in that colour. Statistic: gap = best minus second-best
candidate log-likelihood (as l10_check, but single games).

Null: single games of Valid-U players on their own plies, with their own column removed
("unseen") or kept ("seen"); those games leave every gallery. A training sample of games
is scored the same way on the opponent side, to see how often training opponents are
pool players and, when identified, how close the identified opponent's rank is to the
game owner's rank (HD0 / HD1 split) -- the size of the possible lever.

    python -m scripts.strength.rank_opponent_check
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, opening
from go_aicup.tasks.identity.predict import PLIES, load_params, store_prefix

from scripts.strength.l10_check import gap, score_questions

STORE = ROOT / "outputs" / "rank_release_20261106" / "store"
REFERENCE = ROOT / "data" / "processed" / "humansl_residual_b2" / "seed2027" / "questions_U.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
OUT = ROOT / "outputs" / "rank" / "opponent_check"
PARTS = ("rank_public", "rank_private")
REF_PER_COLOR, TRAIN_PER_COLOR, SEED = 2000, 6000, 2027


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    params = load_params("d4")
    min_plies = 2 * max(params["moves"], params["slot_moves"]) + 1
    meta = data.load_meta()
    meta = meta.merge(pd.read_csv(HANDICAP, usecols=["rank", "game_id", "handicap"]), on=["rank", "game_id"],
                      how="left", validate="one_to_one")
    if not np.array_equal(meta["game_index"].to_numpy(), np.arange(len(meta))):
        raise ValueError("meta order changed by the handicap join")
    train_pre = data.opening_prefix_d4(meta, PLIES)
    player = meta["player_id"].astype(str).to_numpy()
    rank_id = meta["rank_id"].to_numpy()
    pcolor = meta["player_color"].to_numpy()
    hd = meta["handicap"].to_numpy()
    candidates = np.array(sorted(set(player)))
    cand_index = {p: i for i, p in enumerate(candidates)}
    cand_rank = pd.Series(rank_id, index=player).groupby(level=0).first().loc[candidates].to_numpy()
    rng = np.random.default_rng(SEED)

    stores, blocks, offsets, size = {}, [train_pre], {}, len(train_pre)
    for name in PARTS:
        moves, colors, q_meta = load_store(name, STORE, mmap=False)
        pre, ok = store_prefix(moves, colors, q_meta, "d4", min_plies)
        stores[name] = (ok, q_meta)
        offsets[name] = size
        blocks.append(pre)
        size += len(pre)
    prefix = np.concatenate(blocks)
    hashes = opening.prefix_hashes(prefix)

    vu = set(pd.read_parquet(REFERENCE, columns=["player_id"])["player_id"].astype(str))
    ok_train = (hd <= 1) & (meta["length"].to_numpy() >= min_plies)
    ref = {c: rng.permutation(np.flatnonzero(ok_train & (pcolor == c) & np.isin(player, list(vu))))[:REF_PER_COLOR]
           for c in (0, 1)}
    held_out = np.concatenate(list(ref.values()))
    # training sample, grouped by the opponent colour (= 1 - owner colour); disjoint from the reference
    pool = ok_train.copy()
    pool[held_out] = False
    sample = {c: rng.permutation(np.flatnonzero(pool & (pcolor == 1 - c)))[:TRAIN_PER_COLOR] for c in (0, 1)}

    null_rows, train_rows, test_rows = [], [], []
    for color in (0, 1):            # colour of the plies being scored
        own = pcolor == color
        keep = own.copy()
        keep[held_out] = False
        galleries = [np.array([], np.int64)] * len(candidates)
        frame = pd.DataFrame({"p": player[keep], "g": np.flatnonzero(keep)})
        for p, rows in frame.groupby("p")["g"]:
            galleries[cand_index[p]] = rows.to_numpy()
        population = np.flatnonzero(own)

        games = ref[color]
        s = score_questions(prefix, hashes, galleries, [np.array([g]) for g in games], population, color, params)
        truth = np.array([cand_index[p] for p in player[games]])
        seen = gap(s)
        s[np.arange(len(s)), truth] = -np.inf
        null_rows.append(pd.DataFrame({"color": color, "seen": seen, "unseen": gap(s)}))
        print(f"colour {color}: reference {len(games)} games", flush=True)

        games = sample[color]
        s = score_questions(prefix, hashes, galleries, [np.array([g]) for g in games], population, color, params)
        top = np.argmax(s, axis=1)
        train_rows.append(pd.DataFrame({"color": color, "game_index": games, "gap": gap(s),
                                        "owner_rank": rank_id[games], "hd": hd[games],
                                        "top1_rank": cand_rank[top], "top1_is_owner": candidates[top] == player[games]}))
        print(f"colour {color}: training sample {len(games)} games", flush=True)

        for name in PARTS:
            ok, q_meta = stores[name]
            # opponent plies of colour `color` = questions of the other colour
            mask = ok & (q_meta["color"].astype(str).to_numpy() == ("B" if color else "W"))
            index = np.flatnonzero(mask)
            s = score_questions(prefix, hashes, galleries, [np.array([offsets[name] + i]) for i in index],
                                population, color, params)
            top = np.argmax(s, axis=1)
            test_rows.append(pd.DataFrame({"part": name, "question_id": q_meta["question_id"].to_numpy()[index],
                                           "color": color, "gap": gap(s), "top1_rank": cand_rank[top]}))
            print(f"colour {color}: {name} {len(index)} games", flush=True)

    null, train, test = (pd.concat(r, ignore_index=True) for r in (null_rows, train_rows, test_rows))
    train.to_parquet(OUT / "train_sample.parquet")
    test.to_parquet(OUT / "test_games.parquet")
    null.to_parquet(OUT / "null.parquet")
    q95, qmax = float(np.quantile(null["unseen"], 0.95)), float(null["unseen"].max())
    q999 = float(np.quantile(null["unseen"], 0.999))

    def share(v):
        return {"q95": float((v > q95).mean()), "q999": float((v > q999).mean()), "max": float((v > qmax).mean()),
                "median": float(np.median(v))}

    result = {"params": params, "null_games": len(null), "unseen_q95": q95, "unseen_q999": q999, "unseen_max": qmax,
              "reference_seen": share(null["seen"]), "reference_unseen": share(null["unseen"]),
              "train_opponents": share(train["gap"]), "train_top1_is_owner": float(train["top1_is_owner"].mean()),
              "test_opponents": {p: share(test.loc[test["part"] == p, "gap"]) for p in PARTS}}
    # lever size: identified opponent's rank vs the owner's rank, confident identifications only
    lever = {}
    for name, t in (("q999", q999), ("max", qmax), ("all", -np.inf)):
        for h in (0, 1):
            sub = train[(train["gap"] > t) & (train["hd"] == h) & ~train["top1_is_owner"]]
            d = sub["top1_rank"].to_numpy().astype(int) - sub["owner_rank"].to_numpy().astype(int)
            lever[f"{name}_hd{h}"] = {"games": int(len(sub)), "same": float(np.mean(d == 0)) if len(d) else None,
                                      "within1": float(np.mean(np.abs(d) <= 1)) if len(d) else None,
                                      "mean_diff": float(d.mean()) if len(d) else None,
                                      "mean_abs": float(np.abs(d).mean()) if len(d) else None}
    result["lever"] = lever
    result["test_questions_with_confident_opponent"] = {
        p: int(test[(test["part"] == p) & (test["gap"] > qmax)]["question_id"].nunique()) for p in PARTS}
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
