"""Three cheap changes to the added-set style term of the current fusion (2026-11-20,
registered in RESULTS.md before the run).

Base (= ogsa12w added-set form): S = rest + 16 w m (0.25 cos_B + 0.75 mean(cos_A1, cos_A2)), m = 1.5,
rest = opening + category + L1 + L2.
  A-a  per-k style multiplier m_k in MULTS, chosen per k bucket on the development draws.
  A-b  candidate-centre EM (as pid2_b3em) on the three encoders, posterior softmax(S / tau) from the
       full fused score; (tau, alpha, iters) chosen on the development draws.
  A-c  D4 test-time augmentation: per-move vectors recomputed under all 8 D4 transforms (one
       transform per game), passed through each encoder's head; (i) cosine averaged over the 8
       transforms or (ii) unit game vectors averaged over the 8 transforms; chosen on development.
Draws: development pool 0 2300-2319; sealed S1 pool 0 2400-2419, pool 1 2400-2429; sealed S2
(combination only, opened when >= 2 items pass) pool 0 2500-2519, pool 1 2500-2529.
Switch rule (11/20): sealed pool 1 delta > 0 and z >= 2.5, every k >= -0.005, sealed pool 0 delta > 0.

  python -m scripts.identity.pid2_cheap tta --enc B|A1|A2   # GPU: 8-transform game vectors
  python -m scripts.identity.pid2_cheap evaluate            # selection, sealed S1, optional S2
  python -m scripts.identity.pid2_cheap selfcheck           # CPU checks on synthetic data
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs/pid/cheap")
RUN = "pid_ge2e_v4_d4_hd_9600"
ENC_RUN = {"B": ("pid_ge2e_v4_d4_hd_9600", "outputs/pid/pid2_seqagg/head_B/best.pt"),
           "A1": ("pid_e2e_ogs_A1_YO9600/step9600", "outputs/pid/ge2e/pid_e2e_ogs_A1_YO9600/step9600/best.pt"),
           "A2": ("pid_e2e_ogs_A2_YO9600/step9600", "outputs/pid/ge2e/pid_e2e_ogs_A2_YO9600/step9600/best.pt")}
WEIGHTS = {"B": 0.25, "A1": 0.375, "A2": 0.375}
BASE_M = 1.5
MULTS = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
EM_GRID = list(itertools.product((1.0, 2.0, 4.0, 8.0), (0.25, 0.5, 1.0, 2.0), (1, 2)))
DEV = [(0, d) for d in range(2300, 2320)]
S1 = [(0, d) for d in range(2400, 2420)] + [(1, d) for d in range(2400, 2430)]
S2 = [(0, d) for d in range(2500, 2520)] + [(1, d) for d in range(2500, 2530)]
CHECK = (1, 2000)
CHUNK_GAMES = 4000


# ---------------------------------------------------------------- scoring helpers (pure numpy)
def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def unit_sums(unit: np.ndarray, groups) -> np.ndarray:
    return np.stack([unit[g].sum(axis=0) for g in groups])


def cosines(unit: np.ndarray, g, qq) -> np.ndarray:
    """= twocolor.style_scores for unit-norm rows (normalised mean == normalised sum)."""
    return normalise(unit_sums(unit, qq)) @ normalise(unit_sums(unit, g)).T


def softmax(s: np.ndarray, tau: float) -> np.ndarray:
    z = (s - s.max(axis=1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def style(sets, g, qq) -> np.ndarray:
    """sets = [(weight, unit array)]; weighted sum of cosines."""
    return sum(w * cosines(u, g, qq) for w, u in sets)


def style_em(sets, g, qq, rest, scale_rows, tau, alpha, iters) -> np.ndarray:
    """candidate-centre expansion per vector set (leave-one-out), shared posterior from the full
    fused score rest + scale_rows * style, re-estimated every round (as pid2_b3em.expanded_multi)."""
    a = [unit_sums(u, g) for _, u in sets]
    qs = [unit_sums(u, qq) for _, u in sets]
    cos = [normalise(q) @ normalise(x).T for q, x in zip(qs, a)]
    w = [wt for wt, _ in sets]
    for _ in range(iters):
        p = softmax(rest + scale_rows * sum(wi * c for wi, c in zip(w, cos)), tau)
        new = []
        for x, qsum in zip(a, qs):
            qmean = normalise(qsum)
            qdot = (qmean * qsum).sum(axis=1, keepdims=True)
            qsq = (qsum ** 2).sum(axis=1, keepdims=True)
            full = x + alpha * (p.T @ qsum)
            ap = alpha * p
            num = qmean @ full.T - ap * qdot
            norm2 = (full ** 2).sum(axis=1)[None, :] - 2 * ap * (qsum @ full.T) + ap ** 2 * qsq
            new.append(num / np.sqrt(np.maximum(norm2, 1e-12)))
        cos = new
    return sum(wi * c for wi, c in zip(w, cos))


def official(scores: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """official points per question: e^-(r-1) for the owner's rank r <= 5 (same sort as assign.top5)."""
    from go_aicup.metrics import PID_WEIGHTS
    top = np.argsort(-scores, axis=1)[:, :5]
    hit = top == owner[:, None]
    return (hit * PID_WEIGHTS).sum(axis=1)


def buckets(pts: np.ndarray, ks: np.ndarray) -> dict:
    return {**{str(k): float(pts[ks == k].mean()) for k in (1, 2, 3, 5)}, "all": float(pts.mean())}


COLS = ["1", "2", "3", "5", "all"]


def paired(frame: pd.DataFrame, base: str) -> pd.DataFrame:
    b = frame[frame["arm"] == base].set_index("draw")[COLS]
    out = []
    for arm, part in frame.groupby("arm", sort=False):
        d = part.set_index("draw")[COLS] - b
        row = {"arm": arm, "draws": len(d), **{c: part[c].mean() for c in COLS},
               **{f"d_{c}": d[c].mean() for c in COLS}}
        row["se_all"] = d["all"].std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        row["pos"] = float((d["all"] > 0).mean())
        out.append(row)
    return pd.DataFrame(out)


def verdict(t1: pd.DataFrame, t0: pd.DataFrame, arm: str) -> dict:
    r1, r0 = t1.set_index("arm").loc[arm], t0.set_index("arm").loc[arm]
    z = float(r1["d_all"] / r1["se_all"]) if r1["se_all"] > 0 else 0.0
    ks = {k: float(r1[f"d_{k}"]) for k in ("1", "2", "3", "5")}
    ok = bool(r1["d_all"] > 0 and z >= 2.5 and min(ks.values()) >= -0.005 and r0["d_all"] > 0)
    return {"pool1_delta": float(r1["d_all"]), "se": float(r1["se_all"]), "z": z, "buckets": ks,
            "pool0_delta": float(r0["d_all"]), "pass": ok}


# ---------------------------------------------------------------- GPU stage: TTA vectors
def d4_torch(x, t: int):
    import torch
    if t >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, t % 4, dims=(-2, -1))


def tta(args) -> None:
    import torch

    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity import data
    from go_aicup.tasks.identity.seqagg import WORKERS, MoveStore, _Planes, game_vectors, load_head, own_plies
    from go_aicup.tasks.strength.shared.dataset import apply_d4
    from go_aicup.tasks.strength.shared.runtime import autocast_context
    from scripts.identity.pid2_seqagg import BENCH, load_encoder

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"tta_{args.enc}.npy"
    if target.exists() and not args.limit:
        print(f"{target} exists", flush=True)
        return
    run, head_file = ENC_RUN[args.enc]
    model, config, device = load_encoder(run)
    assert config.get("sampling", {}).get("min_ply", 20) == 0, "bench caches use min_ply 0"
    model.eval()
    head, step = load_head(Path(head_file), device)
    head.eval()
    print(f"{args.enc}: encoder {run}, head step {step}", flush=True)
    # torch D4 must equal the training-time numpy D4
    probe = np.random.default_rng(0).normal(size=(2, 3, 19, 19)).astype(np.float32)
    for t in range(8):
        assert np.array_equal(d4_torch(torch.from_numpy(probe), t).numpy(), apply_d4(probe, t)), t
    meta = data.load_meta()
    colors = meta["player_color"].to_numpy().astype(np.int8)
    lengths = meta["length"].to_numpy()
    games = np.load(BENCH / RUN / "embeddings.npz")["game_index"]
    if args.limit:
        games = games[:args.limit]
    cache = BoardCache(data.CACHE_DIR)
    out = np.empty((8, len(games), 256), np.float32)
    batch = 8192
    for s in range(0, len(games), CHUNK_GAMES):
        part = games[s:s + CHUNK_GAMES]
        plies = own_plies(colors[part], lengths[part], 0)
        sizes = np.array([len(p) for p in plies])
        owner = np.repeat(np.arange(len(part)), sizes)
        vec = np.empty((8, len(owner), 256), np.float32)
        dataset = _Planes(cache.directory, part[owner], np.concatenate(plies), colors[part][owner], batch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=WORKERS, prefetch_factor=4,
                                             multiprocessing_context="spawn", pin_memory=True)
        with torch.no_grad():
            for x, i in loader:
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                for t in range(8):
                    with autocast_context(config, device):
                        v = model(d4_torch(x, t).contiguous())
                    vec[t, i * batch:(i + 1) * batch] = v.float().cpu().numpy()
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        for t in range(8):
            store = object.__new__(MoveStore)
            store.vectors, store.offsets = vec[t], offsets
            store.first_ply = colors[part].astype(np.int64)
            out[t, s:s + len(part)] = game_vectors(head, store, np.arange(len(part)), device)
            head.eval()
        print(f"{args.enc}: games {s + len(part)}/{len(games)}", flush=True)
    assert np.isfinite(out).all()
    if args.limit:                               # smoke: compare t = 0 with the cached vectors, no output
        from scripts.identity.pid2_ogs_a2_eval import ENC
        store = MoveStore(Path(ENC[args.enc][0]) / "bench", colors)
        cached = game_vectors(head, store, np.arange(len(games)), device)
        cos = (normalise(cached.astype(np.float64)) * normalise(out[0].astype(np.float64))).sum(axis=1)
        other = [float((normalise(out[0].astype(np.float64)) * normalise(out[t].astype(np.float64))).sum(1).mean())
                 for t in range(1, 8)]
        print(f"smoke {args.enc}: t=0 vs cache min cosine {cos.min():.6f}, mean {cos.mean():.6f}; "
              f"mean cosine t=0 vs t=1..7 {np.round(other, 4).tolist()}", flush=True)
        return
    np.save(target.with_suffix(".tmp.npy"), out)
    target.with_suffix(".tmp.npy").replace(target)
    print(f"wrote {target}", flush=True)


# ---------------------------------------------------------------- evaluation
def evaluate(args) -> None:
    import torch

    from go_aicup.data import cv
    from go_aicup.tasks.identity import data, twocolor
    from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head
    from scripts.identity.pid2_diag_d123a import bucket_means
    from scripts.identity.pid2_multilevel_fuse import OUT as STAGE1, STAGE0, PopTable, level_scores
    from scripts.identity.pid2_ogs_a2_eval import ENC, OUT as A2OUT
    from scripts.identity.pid2_paired import points
    from scripts.identity.pid2_seqagg import BENCH
    from scripts.identity.pid2_transductive import load

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    env = load(RUN)
    color, pos = env["color"], env["position"]
    s1 = json.loads((STAGE1 / "dev_choice.json").read_text())["chosen"]
    saved = np.load(BENCH / RUN / "embeddings.npz")
    meta = data.load_meta()
    pcol = meta["player_color"].to_numpy().astype(np.int8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit, tta_unit = {}, {}
    for name, (cache, head_file) in ENC.items():
        store = MoveStore(Path(cache) / "bench", pcol)
        assert np.array_equal(store.games, saved["game_index"]) and store.finite, name
        head, _ = load_head(Path(head_file), device)
        unit[name] = normalise(game_vectors(head, store, np.arange(len(store.games)), device).astype(np.float64))
        del store
        tta_unit[name] = normalise(np.load(OUT / f"tta_{name}.npy").astype(np.float64))
    # engineering gate 1: t = 0 recomputation against the cached vectors
    eng = {n: float((unit[n] * tta_unit[n][0]).sum(axis=1).min()) for n in unit}
    report["tta_t0_min_cosine"] = eng
    print("t=0 min cosine:", eng, flush=True)
    scale = 16.0 * float(saved["w"])
    b1 = np.load(STAGE0 / "keys.npz")
    row = np.full(len(meta), -1)
    row[b1["game_index"]] = np.arange(len(b1["game_index"]))
    keys1 = lambda games: b1["keys"][row[games]]
    popf = np.load(STAGE1 / "pop_keys.npz")
    pop = PopTable(popf["keys"], color[popf["game_index"]])
    base_sets = [(WEIGHTS[n], unit[n]) for n in WEIGHTS]
    t0_sets = [(WEIGHTS[n], tta_unit[n][0]) for n in WEIGHTS]
    score_sets = [(WEIGHTS[n] / 8, tta_unit[n][t]) for n in WEIGHTS for t in range(8)]
    vec_sets = [(WEIGHTS[n], normalise(tta_unit[n].mean(axis=0))) for n in WEIGHTS]
    tta_sets = {"tta_score": score_sets, "tta_vector": vec_sets}

    def draw(pool, d):
        candidates, gallery, q = cv.draw_pid2(env["bank"], pool, d)
        query = list(q["games"])
        rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, color,
                                        env["populations"], env["summary"]["opening"])
                + 1.0 * twocolor.category_scores(env["codes"], gallery, query, 4)
                + s1["lambda1"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L1")[s1["beta"]]
                + s1["lambda2"] * level_scores(keys1, color, pop, gallery, query, s1["n"], "L2")[s1["beta"]])
        owner = np.searchsorted(candidates, q["player_id"].to_numpy())
        assert (candidates[owner] == q["player_id"].to_numpy()).all()
        return dict(candidates=candidates, q=q, owner=owner, ks=q["k"].to_numpy(), rest=rest,
                    g=[pos[x] for x in gallery], qq=[pos[x] for x in query])

    def rows_scale(ks, mk):
        return scale * np.array([mk[int(k)] for k in ks], np.float64)[:, None]

    # reproduction gate: A12w of the A2 evaluation (via the official points function)
    x = draw(*CHECK)
    s = x["rest"] + scale * BASE_M * style(base_sets, x["g"], x["qq"])
    got = bucket_means(points(s, x["candidates"], x["q"]), x["ks"])
    mine = buckets(official(s, x["owner"]), x["ks"])
    assert max(abs(mine[c] - got[c]) for c in got) < 1e-12, (mine, got)
    ref = pd.read_csv(A2OUT / "sealed_per_draw.csv")
    ref = ref[(ref["pool"] == CHECK[0]) & (ref["draw"] == CHECK[1]) & (ref["arm"] == "A12w")].iloc[0]
    diff = max(abs(got[c] - ref[c]) for c in got)
    report["reproduction_max_diff"] = diff
    print(f"reproduction of A12w pool {CHECK[0]} draw {CHECK[1]}: max diff {diff:.2e}", flush=True)
    assert diff <= 1e-9, (got, ref.to_dict())

    # ------------------------------------------------ development selection (pool 0 2300-2319)
    base_mk = {k: BASE_M for k in cv.PID2_K}
    dev_mult, dev_em, dev_tta = [], [], []
    for pool, d in DEV:
        x = draw(pool, d)
        c = style(base_sets, x["g"], x["qq"])
        for m in MULTS:
            dev_mult.append({"draw": d, "m": m, **buckets(official(x["rest"] + scale * m * c, x["owner"]), x["ks"])})
        sr = rows_scale(x["ks"], base_mk)
        for tau, alpha, iters in EM_GRID:
            ce = style_em(base_sets, x["g"], x["qq"], x["rest"], sr, tau, alpha, iters)
            dev_em.append({"draw": d, "arm": f"t{tau:g}_a{alpha:g}_i{iters}",
                           **buckets(official(x["rest"] + sr * ce, x["owner"]), x["ks"])})
        dev_em.append({"draw": d, "arm": "base", **buckets(official(x["rest"] + sr * c, x["owner"]), x["ks"])})
        for name, sets in {"base": base_sets, "tta_t0": t0_sets, **tta_sets}.items():
            ct = style(sets, x["g"], x["qq"])
            dev_tta.append({"draw": d, "arm": name, **buckets(official(x["rest"] + sr * ct, x["owner"]), x["ks"])})
        print(f"dev draw {d}", flush=True)
    dm = pd.DataFrame(dev_mult)
    mean_m = dm.groupby("m")[COLS].mean()
    chosen_mk = {}
    for k in cv.PID2_K:
        col = mean_m[str(k)]
        best = col.max()
        ties = [m for m in MULTS if col[m] == best]
        chosen_mk[k] = min(ties, key=lambda m: abs(np.log(m / BASE_M)))
    report["perk"] = {"dev_table": mean_m.reset_index().to_dict(orient="records"), "chosen": chosen_mk,
                      "grid_edge": [k for k, m in chosen_mk.items() if m in (MULTS[0], MULTS[-1])]}
    print("per-k dev means:\n", mean_m.round(4).to_string(), "\nchosen m_k:", chosen_mk, flush=True)
    em = paired(pd.DataFrame(dev_em), "base").set_index("arm").drop(index="base")
    best_em = em["d_all"].idxmax()
    tau, alpha, iters = next(g for g in EM_GRID if f"t{g[0]:g}_a{g[1]:g}_i{g[2]}" == best_em)
    report["em"] = {"chosen": {"tau": tau, "alpha": alpha, "iters": iters, "dev_delta": float(em.loc[best_em, "d_all"])},
                    "dev_top": em["d_all"].sort_values(ascending=False).head(8).to_dict(),
                    "grid_edge": bool(tau in (1.0, 8.0) or alpha in (0.25, 2.0))}
    print("EM dev top:\n", em["d_all"].sort_values(ascending=False).head(8).round(4).to_string(), flush=True)
    tt = paired(pd.DataFrame(dev_tta), "base").set_index("arm")
    t0_delta = float(tt.loc["tta_t0", "d_all"])
    tta_ok = min(eng.values()) >= 0.9995 and abs(t0_delta) <= 0.002
    tta_choice = tt.loc[["tta_score", "tta_vector"], "d_all"].idxmax()
    report["tta"] = {"dev": tt[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].to_dict(orient="index"),
                     "t0_dev_delta": t0_delta, "engineering_pass": bool(tta_ok), "chosen": tta_choice}
    print("TTA dev:\n", tt[["d_1", "d_2", "d_3", "d_5", "d_all", "se_all"]].round(4).to_string(),
          f"\nengineering pass {tta_ok}, chosen {tta_choice}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))

    # ------------------------------------------------ arms
    def arms_for(x, which):
        out = {}
        sr = rows_scale(x["ks"], base_mk)
        if "base" in which:
            out["base"] = x["rest"] + sr * style(base_sets, x["g"], x["qq"])
        if "perk" in which:
            out["perk"] = x["rest"] + rows_scale(x["ks"], chosen_mk) * style(base_sets, x["g"], x["qq"])
        if "em" in which:
            out["em"] = x["rest"] + sr * style_em(base_sets, x["g"], x["qq"], x["rest"], sr, tau, alpha, iters)
        if "tta" in which:
            out["tta"] = x["rest"] + sr * style(tta_sets[tta_choice], x["g"], x["qq"])
        if "combo" in which:
            sets = tta_sets[tta_choice] if "tta" in passed else base_sets
            sc = rows_scale(x["ks"], chosen_mk) if "perk" in passed else sr
            c = (style_em(sets, x["g"], x["qq"], x["rest"], sc, tau, alpha, iters) if "em" in passed
                 else style(sets, x["g"], x["qq"]))
            out["combo"] = x["rest"] + sc * c
        return out

    def run_sealed(draws, which, tag):
        rows = []
        for pool, d in draws:
            x = draw(pool, d)
            for name, s in arms_for(x, which).items():
                rows.append({"pool": pool, "draw": d, "arm": name, **buckets(official(s, x["owner"]), x["ks"])})
            print(f"{tag} pool {pool} draw {d}", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / f"{tag}_per_draw.csv", index=False)
        return frame

    passed: list[str] = []
    which = ["base", "perk", "em"] + (["tta"] if tta_ok else [])
    f1 = run_sealed(S1, which, "s1")
    t1, t0 = paired(f1[f1.pool == 1], "base"), paired(f1[f1.pool == 0], "base")
    report["s1"] = {"pool1": t1.to_dict(orient="records"), "pool0": t0.to_dict(orient="records"),
                    "verdict": {a: verdict(t1, t0, a) for a in which if a != "base"}}
    print("S1 pool 1:\n", t1.round(4).to_string(index=False), "\nS1 pool 0:\n", t0.round(4).to_string(index=False),
          flush=True)
    passed = [a for a, v in report["s1"]["verdict"].items() if v["pass"]]
    report["passed_s1"] = passed
    print("S1 verdicts:", json.dumps(report["s1"]["verdict"]), flush=True)
    if len(passed) >= 2:
        f2 = run_sealed(S2, ["base", "combo"] + passed, "s2")
        u1, u0 = paired(f2[f2.pool == 1], "base"), paired(f2[f2.pool == 0], "base")
        best_single = max(passed, key=lambda a: report["s1"]["verdict"][a]["pool1_delta"])
        v1 = paired(f2[f2.pool == 1], best_single).set_index("arm").loc["combo"]
        report["s2"] = {"pool1": u1.to_dict(orient="records"), "pool0": u0.to_dict(orient="records"),
                        "verdict_combo": verdict(u1, u0, "combo"), "combo_vs_best_single": {
                            "arm": best_single, "delta": float(v1["d_all"]), "se": float(v1["se_all"])}}
        print("S2 pool 1:\n", u1.round(4).to_string(index=False), "\nS2 pool 0:\n", u0.round(4).to_string(index=False),
              "\ncombo verdict:", json.dumps(report["s2"]["verdict_combo"]), flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("done", flush=True)


# ---------------------------------------------------------------- CPU self-checks
def selfcheck(_args) -> None:
    rng = np.random.default_rng(0)
    n, dim = 300, 16
    u = normalise(rng.normal(size=(n, dim)))
    g = [np.arange(i * 10, i * 10 + 10) for i in range(20)]
    qq = [rng.choice(np.arange(200, 300), size=k, replace=False) for k in (1, 2, 3, 5) * 5]
    rest = rng.normal(size=(len(qq), len(g)))
    sr = np.full((len(qq), 1), 3.0)
    sets = [(0.25, u), (0.75, u)]
    # EM with alpha 0 is the plain style term
    assert np.allclose(style_em(sets, g, qq, rest, sr, 2.0, 0.0, 2), style(sets, g, qq))
    # 8 identical transforms with weight / 8 equal the single set
    assert np.allclose(style([(0.25 / 8, u)] * 8, g, qq), style([(0.25, u)], g, qq))
    # cosines match twocolor.style_scores semantics (normalised mean of unit vectors)
    ref = np.stack([normalise(u[q].mean(0)) for q in qq]) @ np.stack([normalise(u[x].mean(0)) for x in g]).T
    assert np.allclose(cosines(u, g, qq), ref)
    # official points
    s = np.array([[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]] * 3)
    assert np.allclose(official(s, np.array([0, 2, 5])), [1.0, np.exp(-2), 0.0])
    # verdict rule
    t1 = pd.DataFrame([{"arm": "x", "d_all": 0.003, "se_all": 0.001, "d_1": 0.0, "d_2": 0.01, "d_3": -0.004, "d_5": 0.0}])
    t0 = pd.DataFrame([{"arm": "x", "d_all": 0.001}])
    assert verdict(t1, t0, "x")["pass"]
    t1.loc[0, "d_3"] = -0.006
    assert not verdict(t1, t0, "x")["pass"]
    print("selfcheck ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tta")
    p.add_argument("--enc", choices=list(ENC_RUN), required=True)
    p.add_argument("--limit", type=int, default=0, help="smoke test on the first N bench games (no output)")
    sub.add_parser("evaluate")
    sub.add_parser("selfcheck")
    args = parser.parse_args()
    {"tta": tta, "evaluate": evaluate, "selfcheck": selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
