"""Official added-set export of the multi-level opening terms (stage 1 passed, 2026-11-17).

Adds lambda1 * S_L1 + lambda2 * S_L2 (setting from outputs/pid/pid2_multilevel/stage1/dev_choice.json,
population tables from the same 200,000-game sample) to the frozen hung-seqB score terms of each
part (outputs/pid/day0_1112/{part}2_d4_9600_seqB), writes the per-part artifacts to
outputs/pid/day0_1112/{part}2_d4_9600_seqB_ml/ and submissions/pid/hung-seqB-ml.csv
(original test set rows copied unchanged from 772748-hung.csv; refuses to overwrite).

    python -m scripts.identity.pid2_multilevel_export
"""
from __future__ import annotations

import hashlib
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT, PROCESSED
from go_aicup.tasks.identity import data, multilevel
from go_aicup.tasks.identity.assign import top5
from scripts.identity.pid2_multilevel import N_OWN
from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, PopTable, level_scores

STORE = PROCESSED / "identity" / "artifacts" / "day0_1112" / "store"
DAY0 = ROOT / "outputs" / "pid" / "day0_1112"
BASE = ROOT / "submissions" / "pid" / "772748-hung.csv"
SEQB = ROOT / "submissions" / "pid" / "hung-seqB.csv"
TARGET = ROOT / "submissions" / "pid" / "hung-seqB-ml.csv"
COLS = [f"top_{i}" for i in range(1, 6)]

_GAMES = None


def _init(games):
    global _GAMES
    _GAMES = games


def _keys_of(i: int) -> np.ndarray:
    moves, own = _GAMES[i]
    return multilevel.game_keys(moves, own, N_OWN)


def game_list(moves, colors, meta):
    out = []
    for offset, length, col in zip(meta["offset"], meta["length"], meta["color"].astype(str)):
        m = np.asarray(moves[offset:offset + length], np.int64)
        c = np.asarray(colors[offset:offset + length])
        own = int(col == "W")
        if length < 2 * N_OWN + own or not np.array_equal(c[:2 * N_OWN + 1] % 2, np.arange(2 * N_OWN + 1) % 2):
            raise ValueError("game too short or not alternating from black")
        if (m[:2 * N_OWN + 1] >= 361).any():
            raise ValueError("pass or invalid point in the first moves")
        out.append((m, own))
    return out


def main() -> None:
    assert not TARGET.exists(), TARGET
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    report = json.loads((STAGE1 / "report.json").read_text())
    assert report["gate"]["pass"] is True
    n, beta, l1, l2 = choice["n"], choice["beta"], choice["lambda1"], choice["lambda2"]
    meta_train = data.load_meta()
    color_train = meta_train["player_color"].to_numpy().astype(np.int64)
    popfile = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popfile["keys"], color_train[popfile["game_index"]])
    d4 = data.d4_tables()

    base = pd.read_csv(BASE, dtype=str).set_index("question_id")
    seqb = pd.read_csv(SEQB, dtype=str).set_index("question_id")
    out = base.copy()
    summary = {"choice": choice, "parts": {}}
    for part, prefix in (("public", "pub2"), ("private", "priv2")):
        src = DAY0 / f"{part}2_d4_9600_seqB"
        pool_moves, pool_colors, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
        q_moves, q_colors, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
        games = game_list(pool_moves, pool_colors, pool_meta) + game_list(q_moves, q_colors, q_meta)
        with Pool(12, initializer=_init, initargs=(games,)) as p:
            keys = np.stack(p.map(_keys_of, range(len(games)), chunksize=64))
        # symmetry check: keys of a few games are unchanged under every board symmetry
        for i in range(0, len(games), max(1, len(games) // 20)):
            for t in range(8):
                moved = d4[t][games[i][0]].astype(np.int64)
                assert np.array_equal(multilevel.game_keys(moved, games[i][1], N_OWN), keys[i]), "D4 invariance"
        color = np.array([g[1] for g in games], np.int64)
        n_pool = len(pool_meta)
        players = pool_meta["player_id"].astype(str).to_numpy()
        candidates = np.array(sorted(set(players)))
        gallery = [np.flatnonzero(players == c) for c in candidates]
        qids = list(dict.fromkeys(q_meta["question_id"]))
        rows_by_q = q_meta.groupby("question_id", sort=False).indices
        query = [n_pool + np.sort(rows_by_q[q]) for q in qids]
        ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[qids].to_numpy()

        pred = pd.read_parquet(src / "predictions.parquet")
        assert pred.question_id.astype(str).tolist() == qids
        assert np.array_equal(np.load(src / "candidates.npy", allow_pickle=True).astype(str), candidates)
        terms = dict(np.load(src / "parts.npz"))
        info = json.loads((src / "info.json").read_text())
        s = info["summary"]
        seqb_scores = terms["opening"] + s["lambda"] * terms["style"] + s["mu"] * terms["category"]
        np.testing.assert_allclose(seqb_scores, np.load(src / "scores.npy"))
        np.testing.assert_array_equal(candidates[top5(seqb_scores)], seqb.loc[qids, COLS].to_numpy())

        keys_of = lambda g: keys[g]
        ml1 = level_scores(keys_of, color, pop, gallery, query, n, "L1")[beta]
        ml2 = level_scores(keys_of, color, pop, gallery, query, n, "L2")[beta]
        scores = seqb_scores + l1 * ml1 + l2 * ml2
        assert np.isfinite(scores).all() and scores.shape == (400, 600)
        ranked = candidates[top5(scores)]
        out.loc[qids, COLS] = ranked
        ch = ranked[:, 0] != seqb.loc[qids, "top_1"].to_numpy()
        dst = DAY0 / f"{part}2_d4_9600_seqB_ml"
        dst.mkdir(parents=True, exist_ok=False)
        np.savez(dst / "parts.npz", **terms, ml_L1=ml1, ml_L2=ml2)
        np.save(dst / "scores.npy", scores)
        np.save(dst / "candidates.npy", candidates)
        table = pd.DataFrame({"question_id": qids, "num_games": ks})
        for j in range(5):
            table[f"top{j + 1}"] = ranked[:, j]
        table.to_parquet(dst / "predictions.parquet", index=False)
        stats = {"top1_changed_vs_seqB": int(ch.sum()),
                 "by_k": {int(k): int(ch[ks == k].sum()) for k in (1, 2, 3, 5)},
                 "L1_none_share": float((keys[:, :n, 1] == 0).mean()),
                 "L2_none_share": float((keys[:, :n, 3] == 0).mean()),
                 "ml_spread_per_question": float(np.mean((l1 * ml1 + l2 * ml2).std(axis=1)))}
        (dst / "info.json").write_text(json.dumps({**info, "multilevel": {**choice, **stats}}, indent=2))
        summary["parts"][prefix] = stats
        print(prefix, json.dumps(stats), flush=True)

    orig = base.index[base.index.str.startswith(("pub_", "priv_"))]
    pd.testing.assert_frame_equal(out.loc[orig], base.loc[orig])
    for q, row in out.iterrows():
        assert len(set(row)) == 5 and all(v.split("_")[0] == q.split("_")[0] for v in row)
    out.to_csv(TARGET)
    pd.testing.assert_frame_equal(pd.read_csv(TARGET, dtype=str).set_index("question_id"), out)
    summary["path"] = str(TARGET.relative_to(ROOT))
    summary["sha256"] = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    (STAGE1 / "export_report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
