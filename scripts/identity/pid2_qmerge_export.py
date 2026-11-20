"""E1 question merging on the official added set (LB probe, 2026-11-20; switch rule NOT passed).

The user asked to submit E1 although the sealed evaluation of pid2_qmerge (job 816715) failed the
switch rule (pool 1 +0.0002, z 0.47; pool 0 -0.0008). Nothing is refitted on the official files:
  * base S per part = the ogsa12w fusion, rebuilt from the frozen day0 parts and checked to give
    exactly the ogsa12w top-5;
  * verifier = the logistic regression of pid2_qmerge fitted on the same FIT draws (pool 0
    2200-2219), arm (tau, beta, theta) read from outputs/pid/qmerge/report.json;
  * official query-query pair features follow pid2_qmerge.compute with the official stores:
    question-centre cosines of seqB / A1 / A2 (vectors checked to reproduce each stored style
    term), opening / category / L1 / L2 of q against q' as a one-person gallery (official
    populations and parameters, as in predict2 and pid2_multilevel_export).
Merging is done within each part (pub2 / priv2 have separate candidate pools). Original test-set
rows are copied from ogsa12w. Writes submissions/pid/<SLURM_JOB_ID>-qmerge.csv; refuses to overwrite.

    python -m scripts.identity.pid2_qmerge_export
"""
from __future__ import annotations

import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import load_store
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, multilevel, opening, twocolor
from go_aicup.tasks.identity.assign import top5
from go_aicup.tasks.identity.predict import PLIES, store_prefix, train_prefix
from scripts.identity import pid2_qmerge as qm
from scripts.identity.pid2_multilevel import N_OWN
from scripts.identity.pid2_multilevel_export import _init, _keys_of, game_list
from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, PopTable, level_scores
from scripts.identity.pid2_style_export import REF_HEAD, REF_RUN, STORE, head_vectors

DAY0 = ROOT / "outputs" / "pid" / "day0_1112"
SUB = ROOT / "submissions" / "pid"
BASE = SUB / "ogsa12w.csv"
COLS = [f"top_{i}" for i in range(1, 6)]
OUT = ROOT / "outputs" / "pid" / "qmerge" / "official"
# encoder name -> (day0 suffix, weight inside 16 * style) as in pid2_combine_ml3 "ogsa12w"
ENC = {"B": ("_seqB", 0.375), "A1": ("_e2eogs9600", 0.5625), "A2": ("_e2eogsA2", 0.5625)}


def pair_features(part: str, parts: dict, info: dict, get_vec) -> dict:
    s = info["summary"]
    _, _, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
    q_moves, q_colors, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
    pool_moves, pool_colors, _ = load_store(f"pool2_{part}", STORE, mmap=False)
    n_pool = len(pool_meta)
    qids = list(dict.fromkeys(q_meta["question_id"]))
    rows_by_q = q_meta.groupby("question_id", sort=False).indices
    qrows = [np.sort(rows_by_q[q]) for q in qids]
    players = pool_meta["player_id"].astype(str).to_numpy()
    candidates = np.array(sorted(set(players)))
    gallery = [np.flatnonzero(players == c) for c in candidates]
    query = [n_pool + r for r in qrows]
    pair = {}

    # opening and category, official populations and parameters (predict2)
    params = s["opening"]
    min_plies = 2 * max(params["moves"], params["slot_moves"])
    train_meta, train_pre = train_prefix("d4")
    pool_pre, pool_ok = store_prefix(pool_moves, pool_colors, pool_meta, "d4", min_plies)
    q_pre, q_ok = store_prefix(q_moves, q_colors, q_meta, "d4", min_plies)
    assert pool_ok.all() and q_ok.all()
    train_color = train_meta["player_color"].to_numpy().astype(np.int64)
    populations = (np.flatnonzero(train_color == 0), np.flatnonzero(train_color == 1))
    n_train = len(train_meta)
    prefix = np.concatenate([train_pre[:, :PLIES], pool_pre, q_pre])
    hashes = opening.prefix_hashes(prefix)
    is_white = lambda meta: (meta["color"].astype(str).to_numpy() == "W").astype(np.int64)
    color = np.concatenate([train_color, is_white(pool_meta), is_white(q_meta)])
    codes = np.concatenate([np.zeros(n_train, np.int64),
                            np.minimum(pool_meta["hd"].to_numpy(), 1) + 2 * is_white(pool_meta),
                            np.minimum(q_meta["hd"].to_numpy(), 1) + 2 * is_white(q_meta)])
    gal_t = [n_train + g for g in gallery]
    qry_t = [n_train + n_pool + r for r in qrows]
    op = twocolor.opening_scores(prefix, hashes, gal_t, qry_t, color, populations, params)
    diff = float(np.abs(op - parts["B"]["opening"]).max())
    print(f"{part}: opening reproduction max abs diff {diff:.2e}", flush=True)
    assert diff <= 1e-6, diff
    pair["open"] = twocolor.opening_scores(prefix, hashes, qry_t, qry_t, color, populations, params)
    pair["cat"] = twocolor.category_scores(codes, qry_t, qry_t, 4)

    # L1 / L2 (pid2_multilevel_export)
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    meta_train = data.load_meta()
    color_train = meta_train["player_color"].to_numpy().astype(np.int64)
    popfile = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popfile["keys"], color_train[popfile["game_index"]])
    games = game_list(pool_moves, pool_colors, pool_meta) + game_list(q_moves, q_colors, q_meta)
    with Pool(12, initializer=_init, initargs=(games,)) as p:
        keys = np.stack(p.map(_keys_of, range(len(games)), chunksize=64))
    gcol = np.array([g[1] for g in games], np.int64)
    keys_of = lambda g: keys[g]
    for lev in ("L1", "L2"):
        ref = level_scores(keys_of, gcol, pop, gallery, query, choice["n"], lev)[choice["beta"]]
        diff = float(np.abs(ref - parts["ml"][f"ml_{lev}"]).max())
        print(f"{part}: {lev} reproduction max abs diff {diff:.2e}", flush=True)
        assert diff <= 1e-6, diff
        pair[lev] = level_scores(keys_of, gcol, pop, query, query, choice["n"], lev)[choice["beta"]]
    # cosines: vectors must reproduce the stored style term of each encoder
    vec = get_vec()
    for name, v in vec.items():
        style = s["w"] * twocolor.style_scores(v, gallery, query)
        diff = float(np.abs(style - parts[name]["style"]).max())
        print(f"{part} {name}: style reproduction max abs diff {diff:.2e}", flush=True)
        assert diff <= 1e-4, (part, name, diff)
        u = twocolor.normalise(np.stack([twocolor.normalise(twocolor.normalise(v[x].astype(np.float64)).mean(0))
                                         for x in query]))
        pair[f"cos_{name}"] = u @ u.T

    ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[qids].to_numpy()
    return qids, ks, candidates, pair


def main() -> None:
    job = os.environ.get("SLURM_JOB_ID", "local")
    target = SUB / f"{job}-qmerge.csv"
    assert not target.exists(), target
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((qm.OUT / "report.json").read_text())
    arm = rep["params"]
    assert rep["chosen_verifier"] == "lr"
    verifier = qm.fit_verifiers(qm.FIT)["lr"]
    choice = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    base = pd.read_csv(BASE, dtype=str).set_index("question_id")
    out = base.copy()
    report = {"arm": arm, "base": str(BASE.relative_to(ROOT)), "parts": {}}
    for part, prefix in (("public", "pub2"), ("private", "priv2")):
        parts, info = {}, None
        for name, (suffix, _) in ENC.items():
            d = DAY0 / f"{part}2_d4_9600{suffix}"
            parts[name] = dict(np.load(d / "parts.npz"))
            if name == "B":
                info = json.loads((d / "info.json").read_text())
                cand0 = np.load(d / "candidates.npy", allow_pickle=True).astype(str)
                qids0 = pd.read_parquet(d / "predictions.parquet").question_id.astype(str).tolist()
        parts["ml"] = dict(np.load(DAY0 / f"{part}2_d4_9600_seqB_ml" / "parts.npz"))
        rest = parts["B"]["opening"] + parts["B"]["category"]
        ml = choice["lambda1"] * parts["ml"]["ml_L1"] + choice["lambda2"] * parts["ml"]["ml_L2"]
        s = rest + 16 * sum(w * parts[n]["style"] for n, (_, w) in ENC.items()) + ml
        np.testing.assert_array_equal(cand0[top5(s)], base.loc[qids0, COLS].to_numpy())

        runs = {"B": (REF_RUN, REF_HEAD)}
        for name in ("A1", "A2"):
            inf = json.loads((DAY0 / f"{part}2_d4_9600{ENC[name][0]}" / "info.json").read_text())
            runs[name] = (inf["run"], ROOT / inf["head"])
        get_vec = lambda: {name: head_vectors(r, Path(h), part) for name, (r, h) in runs.items()}
        qids, ks, candidates, pair = pair_features(part, parts, info, get_vec)
        assert qids == qids0 and np.array_equal(candidates, cand0)
        d = {"S": s, "k": ks, **{k: pair[k].astype(np.float32) for k in qm.PAIRS}}
        np.savez(OUT / f"{prefix}.npz", **d)
        m = qm.pair_m(verifier, d)
        s2 = s + arm["beta"] * qm.merge_term(s, m, arm["tau"], arm["theta"])
        assert np.isfinite(s2).all()
        ranked = candidates[top5(s2)]
        out.loc[qids, COLS] = ranked
        ch = ranked[:, 0] != base.loc[qids, "top_1"].to_numpy()
        off = ~np.eye(len(m), dtype=bool)
        report["parts"][prefix] = {
            "top1_changed": int(ch.sum()), "by_k": {int(k): int(ch[ks == k].sum()) for k in (1, 2, 3, 5)},
            "pairs_over_theta": int((m[off] >= arm["theta"]).sum() // 2),
            "mass_per_question": float(np.where(m >= arm["theta"], m, 0).sum(1).mean())}
        print(prefix, json.dumps(report["parts"][prefix]), flush=True)

    orig = base.index[base.index.str.startswith(("pub_", "priv_"))]
    pd.testing.assert_frame_equal(out.loc[orig], base.loc[orig])
    for q, row in out.iterrows():
        assert len(set(row)) == 5 and all(v.split("_")[0] == q.split("_")[0] for v in row)
    out.to_csv(target)
    pd.testing.assert_frame_equal(pd.read_csv(target, dtype=str).set_index("question_id"), out)
    report["path"] = str(target.relative_to(ROOT))
    report["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT / f"report_{job}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
