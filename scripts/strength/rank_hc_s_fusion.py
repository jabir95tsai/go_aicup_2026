"""Does the policy_residual S posterior add to H+C? Log-linear posterior fusion on the dev players
(RESULTS 2026-11-16, S fusion spec and its revision).

Stage 1 (H for the dev games no H source covers):
    python -m scripts.strength.rank_hc_s_fusion htables     # missing list = dev games minus H sources
    sbatch slurm/rank_hc_s_hdev.slurm                       # HumanSL engine, HD0 komi 6.5 / HD1 komi 0.5
    python -m scripts.strength.rank_hc_s_fusion haudit      # coverage, identity, model, profiles, komi ...
Stage 2 (CPU):
    python -m scripts.strength.rank_hc_s_fusion prepare     # folds + question files, minimum checks
    sbatch slurm/rank_hc_s_fusion.slurm                     # 150 B2 fits, fusion, gates

Arms on the same saved folds / questions / B2 seeds: HC, HC_cal (alpha only), HC+S, HC+A0; S and A0
alone (frozen heads) as diagnostics. Fusion z = alpha_k log p_HC + beta_k log p_X, softmax, score_rank;
(alpha, beta) per k from inner-fold OOF calibration questions (official mean score), ties within 1e-6:
smaller beta, alpha closer to 1, smaller alpha.
Gates: utility HC+S - HC_cal (Equal-k >= +0.005, P >= 0.90, every k >= -0.005, all five seeds > 0);
new information HC+S - HC+A0 (mean > 0, P >= 0.90).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

RUN = ROOT / "outputs" / "rank" / "humansl" / "policy_residual" / "seed2027_t1000_m200" / "785715"
IDENTITY = "1b79d5cc9b7a88442fe6cddfa366cf32239722621a6b7bda6b96165ccd16540a"
HDIR = ROOT / "data" / "processed" / "strength_features" / "humansl" / "dev_fold_m1_h" / "v1"
OUT = ROOT / "outputs" / "rank" / "humansl" / "hc_s_fusion"
SEEDS = (2027, 2028, 2029, 2030, 2031)
FOLDS, EVAL_CELL, CAL_CELL, MIN_PLAYERS = 5, 200, 100, 5
ALPHAS = (0.6, 0.8, 1.0, 1.2, 1.4)
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TIE = 1e-6
GAIN, PROB, MAX_K_DROP = 0.005, 0.90, 0.005
MODEL, KOMI = "b18c384nbt-humanv0.bin.gz", {0: 6.5, 1: 0.5}


def dev_games() -> pd.DataFrame:
    if json.loads((RUN / "identity.json").read_text())["identity"] != IDENTITY:
        raise ValueError("785715 identity changed")
    games = pd.read_parquet(RUN / "games.parquet")
    return games[games["role"] == "dev"].sort_values("game_index").reset_index(drop=True)


def h_sources() -> dict:
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.humansl_p3_komi as p3
    import scripts.strength.rank_more_players as more
    import scripts.strength.rank_train20_hc as t20
    return {0: {"standard": hf.STANDARD, "more_players/h_hd0": more.OUT / "h_hd0", "dev/h_hd0": HDIR / "h_hd0"},
            1: {"more_players/h_hd1": more.OUT / "h_hd1", "train20_hc/h_hd1": t20.OUT / "h_hd1",
                "p3_komi/h_train": p3.OUT / "h_train", "dev/h_hd1": HDIR / "h_hd1"}}


def _h_frame(name: str, path: Path) -> pd.DataFrame:
    import scripts.strength.rank_more_players as more
    if name == "standard":
        index, x = np.load(path / "game_index.npy"), np.load(path / "features.npy")
        keep = np.isfinite(x).all(axis=1)
        return pd.DataFrame(x[keep], index=index[keep])
    h = more.hfeat.run_features(path).set_index("game_index")
    h = h[[c for c in h.columns if c.startswith(("own_", "opp_"))]]
    if [c.split("_", 1)[0] for c in h.columns] != ["own"] * 15 + ["opp"] * 15:
        raise ValueError(f"{path}: unexpected H column order")
    return h


def h_features(games: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """[games, 30] H and the number of sources covering each game (komi-consistent sources only)."""
    x, count = np.full((len(games), 30), np.nan), np.zeros(len(games), int)
    gi, hd = games["game_index"].to_numpy(), games["handicap"].to_numpy()
    for value, sources in h_sources().items():
        for name, path in sources.items():
            if not path.exists():
                continue
            h = _h_frame(name, path)
            rows = np.flatnonzero((hd == value) & np.isin(gi, h.index))
            x[rows] = h.loc[gi[rows]].to_numpy()
            count[rows] += 1
    return x, count


# ------------------------------------------------------------------ stage 1
def htables() -> None:
    games = dev_games()
    hidden = HDIR / "h_hd0", HDIR / "h_hd1"
    if any(p.exists() for p in hidden):
        raise SystemExit(f"{HDIR} already has extraction output; not overwriting")
    x, count = h_features(games)
    missing = games[count == 0]
    HDIR.mkdir(parents=True, exist_ok=True)
    report = {"dev_games": int(len(games)), "missing": int(len(missing))}
    for hd in (0, 1):
        table = missing.loc[missing["handicap"] == hd, ["game_index", "player_color"]].assign(role="dev")
        table.reset_index(drop=True).to_parquet(HDIR / f"missing_hd{hd}.parquet")
        report[f"missing_hd{hd}"] = int(len(table))
    (HDIR / "tables.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def haudit() -> dict:
    import scripts.strength.rank_train20_hc as t20
    games = dev_games()
    x, count = h_features(games)
    meta = t20.metadata().set_index("game_index").loc[games["game_index"]]
    checks = {name: bool((meta[col].to_numpy() == games[col].to_numpy()).all())
              for name, col in (("player", "player_id"), ("rank", "rank_id"), ("color", "player_color"),
                                ("handicap", "handicap"))}
    checks |= {"coverage_100": bool((count >= 1).all()), "single_source": bool((count <= 1).all()),
               "h_finite": bool(np.isfinite(x).all()), "no_duplicate_games": bool(games["game_index"].is_unique)}
    logs = {}
    for value, sources in h_sources().items():
        for name, path in sources.items():
            if name == "standard" or not path.exists():
                continue
            entries = [json.loads(p.read_text().strip().splitlines()[-1]) for p in sorted(path.glob("shard_*.log"))]
            logs[name] = {"model": sorted({e["model"] for e in entries}), "komi": sorted({e["komi"] for e in entries}),
                          "profiles": entries[0]["profiles"], "games": int(sum(e["games_complete"] for e in entries)),
                          "complete": all(e["games"] == e["games_complete"] and e["katago_returncode"] == 0
                                          and e["rejected_queries"] == 0 for e in entries),
                          "same_profiles": all(e["profiles"] == entries[0]["profiles"] for e in entries),
                          "komi_ok": all(e["komi"] == KOMI[value] for e in entries)}
    checks["sources_complete_komi_profiles"] = all(v["complete"] and v["komi_ok"] and v["same_profiles"]
                                                   for v in logs.values())
    checks["one_model_and_profile_order"] = (all(v["model"] == [MODEL] for v in logs.values())
                                             and len({tuple(v["profiles"]) for v in logs.values()}) == 1)
    new = {}
    for hd in (0, 1):
        path = HDIR / f"h_hd{hd}"
        listed = set(pd.read_parquet(HDIR / f"missing_hd{hd}.parquet")["game_index"])
        got = set(_h_frame("x", path).index)
        checks[f"new_hd{hd}_equals_missing_list"] = got == listed
        new[f"h_hd{hd}"] = {str(p.relative_to(HDIR)): _sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
    report = {"pass": all(checks.values()), "checks": checks, "dev_games": int(len(games)), "sources": logs}
    (HDIR / "audit.json").write_text(json.dumps(report, indent=1))
    manifest = {"format": "dev_fold_m1_h_v1", "source_games": str(RUN / "games.parquet"), "identity": IDENTITY,
                "model": MODEL, "komi": {"hd0": 6.5, "hd1": 0.5},
                "lists": {f"missing_hd{hd}.parquet": _sha(HDIR / f"missing_hd{hd}.parquet") for hd in (0, 1)},
                "files": new}
    (HDIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: report[k] for k in ("pass", "checks")}, indent=1))
    return report


# ------------------------------------------------------------------ stage 2: inputs
def _stratified(players: pd.Series, rng) -> pd.Series:
    fold = pd.Series(0, index=players.index)
    for _, ids in players.groupby(players):
        fold.loc[ids.index[rng.permutation(len(ids))]] = np.arange(len(ids)) % FOLDS
    return fold


def _questions(games: pd.DataFrame, per_cell: int, rng) -> pd.DataFrame:
    """Single-colour (one colour per dev player) rank x k cells, as rank_c4g._questions."""
    groups = games.groupby(["rank_id", games["player_id"].astype(str)])["game_index"].apply(np.sort)
    rows = []
    for k in K_BUCKETS:
        for rank in range(10):
            eligible = [(key, g) for key, g in groups.loc[rank].items() if len(g) >= k]
            for pick in rng.integers(0, len(eligible), per_cell):
                player, g = eligible[pick]
                rows.append({"player_id": player, "rank_id": rank, "k": k,
                             "games": np.sort(rng.choice(g, k, replace=False)).tolist()})
    return pd.DataFrame(rows)


def _cells(games: pd.DataFrame) -> np.ndarray:
    n = games.groupby(games["player_id"].astype(str)).agg(n=("game_index", "size"), r=("rank_id", "first"))
    return np.array([np.bincount(n["r"][n["n"] >= k], minlength=10) for k in K_BUCKETS])


def prepare() -> None:
    import scripts.strength.rank_c4g as g4
    if not json.loads((HDIR / "audit.json").read_text())["pass"]:
        raise SystemExit("stage-1 H audit did not pass")
    games = dev_games()
    OUT.mkdir(parents=True, exist_ok=True)
    outer = g4._fold_of(games)
    players = games.groupby(games["player_id"].astype(str))["rank_id"].first().sort_index()
    folds = pd.DataFrame({"player_id": players.index, "rank_id": players.to_numpy(),
                          "outer": outer.loc[players.index].to_numpy(), "inner": -1})
    checks = {"overall_min": _cells(games).min(axis=1).tolist()}
    ok = bool(_cells(games).min() >= MIN_PLAYERS)
    cal = []
    for o in range(FOLDS):
        fit_players = folds.loc[folds["outer"] != o, ["player_id", "rank_id"]].set_index("player_id")["rank_id"]
        inner = _stratified(fit_players, np.random.default_rng([2027, o]))
        folds.loc[folds["outer"] != o, f"inner_{o}"] = inner.loc[folds.loc[folds["outer"] != o, "player_id"]].to_numpy()
        fit = games[games["player_id"].astype(str).isin(fit_players.index)]
        checks[f"outer{o}_fit_min"] = _cells(fit).min(axis=1).tolist()
        checks[f"outer{o}_test_min"] = _cells(games[~games["player_id"].astype(str).isin(fit_players.index)]).min(axis=1).tolist()
        inner_fit = []
        for i in range(FOLDS):
            held = set(inner.index[inner == i])
            inner_fit.append(int(_cells(fit[~fit["player_id"].astype(str).isin(held)]).min()))
        checks[f"outer{o}_inner_fit_min"] = min(inner_fit)
        ok &= bool(_cells(fit).min() >= MIN_PLAYERS and min(inner_fit) >= MIN_PLAYERS)
        q = _questions(fit, CAL_CELL, np.random.default_rng([2028, o]))
        cal.append(q.assign(outer=o, inner=inner.loc[q["player_id"]].to_numpy()))
    folds = folds.drop(columns="inner")
    evaluation = _questions(games, EVAL_CELL, np.random.default_rng(2028))
    evaluation["outer"] = outer.loc[evaluation["player_id"]].to_numpy()
    folds.to_parquet(OUT / "folds.parquet")
    evaluation.rename_axis("question_id").reset_index().to_parquet(OUT / "questions_eval.parquet")
    pd.concat(cal, ignore_index=True).rename_axis("question_id").reset_index().to_parquet(OUT / "questions_cal.parquet")
    checks["eval_questions_per_outer"] = evaluation["outer"].value_counts().sort_index().tolist()
    checks["pass"] = ok
    (OUT / "prepare.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps(checks, indent=1))
    if not ok:
        raise SystemExit("undetermined: minimum players per rank x k not met")


def features(games: pd.DataFrame) -> dict:
    import scripts.strength.humansl_hd_filter as hf
    h, count = h_features(games)
    if not ((count == 1).all() and np.isfinite(h).all()):
        raise SystemExit("undetermined: H incomplete or multiply sourced")
    raw = np.load(RUN / "raw_c.npz")
    c = pd.DataFrame(raw["c"], index=raw["game_index"]).loc[games["game_index"]].to_numpy().astype(float)
    out = {"HC": np.hstack([h, c, hf.hd1_columns(games)])}
    for arm in ("S", "A0"):
        z = np.load(RUN / arm / "features.npz")
        if set(z["game_index"]) != set(games["game_index"]):
            raise SystemExit(f"undetermined: {arm} features cover different games")
        out[arm] = pd.DataFrame(z["c"], index=z["game_index"]).loc[games["game_index"]].to_numpy().astype(float)
    if not all(np.isfinite(v).all() for v in out.values()):
        raise SystemExit("undetermined: non-finite features")
    return out


def frozen_posterior(arm: str, x: np.ndarray, game_index: np.ndarray, questions: pd.DataFrame) -> np.ndarray:
    import torch
    state = torch.load(RUN / arm / "model.pt", map_location="cpu", weights_only=False)
    if state["identity"] != IDENTITY:
        raise ValueError(f"{arm}: model identity mismatch")
    head = state["head"]
    position = pd.Series(np.arange(len(game_index)), index=game_index)
    mean = np.stack([x[position.loc[list(g)].to_numpy()].mean(axis=0) for g in questions["games"]])
    w, b = head["linear.weight"].double().numpy(), head["linear.bias"].double().numpy()
    z = ((mean - head["mean"].double().numpy()) / head["scale"].double().numpy()) @ w.T + b
    z -= z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ stage 2: B2 fits
_STATE: dict = {}


def _fit(job):
    """One PriorMatched B2 fit; HC posteriors of the given questions plus coefficients and n_iter_."""
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    seed, outer, inner, fit_players, qkind, q_rows = job
    games, x = _STATE["games"], _STATE["x"]
    rows = np.flatnonzero(games["player_id"].astype(str).isin(fit_players).to_numpy())
    fit = games.iloc[rows].reset_index(drop=True)
    hs.SEED = seed
    model = more.PriorMatched(more.player_prior(fit)).fit(fit, pd.DataFrame(), x[rows])
    q = _STATE[qkind].iloc[q_rows]
    p = model.probabilities(x, games["game_index"].to_numpy(), q)
    coef = {str(k): {"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
                     "n_iter": np.asarray(lr.n_iter_).tolist()} for k, (_, lr) in model.per_k.items()}
    return seed, outer, inner, qkind, q_rows, p, coef


def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-300, None))


def _fuse(lp_hc, lp_x, alpha, beta):
    z = alpha * lp_hc + beta * lp_x
    z -= z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def _score(p, truth):
    from go_aicup.tasks.strength.v2.aggregate import official_scores
    return official_scores(score_rank(p), truth)


def _select(lp_hc, lp_x, truth, betas):
    """(alpha, beta) maximising the mean official score; ties within TIE: smaller beta, alpha nearer 1,
    smaller alpha."""
    best = None
    for beta in betas:
        for alpha in ALPHAS:
            value = float(_score(_fuse(lp_hc, lp_x, alpha, beta), truth).mean())
            key = (-value, beta, abs(alpha - 1), alpha)
            if best is None or value > best[0] + TIE or (abs(value - best[0]) <= TIE and key[1:] < best[1][1:]):
                best = (value, key, alpha, beta)
    return best[2], best[3]


def run(workers: int) -> None:
    import scripts.strength.rank_hier_head as hh
    started = time.time()
    games = dev_games()
    xs = features(games)
    folds = pd.read_parquet(OUT / "folds.parquet")
    qe, qc = pd.read_parquet(OUT / "questions_eval.parquet"), pd.read_parquet(OUT / "questions_cal.parquet")
    _STATE.update(games=games, x=xs["HC"], eval=qe, cal=qc)
    gi = games["game_index"].to_numpy()
    frozen = {(arm, kind): _logp(frozen_posterior(arm, xs[arm], gi, q))
              for arm in ("S", "A0") for kind, q in (("eval", qe), ("cal", qc))}
    jobs = []
    for seed in SEEDS:
        for o in range(FOLDS):
            fit_players = folds.loc[folds["outer"] != o, "player_id"]
            jobs.append((seed, o, -1, set(fit_players), "eval", np.flatnonzero(qe["outer"].to_numpy() == o)))
            inner = folds.loc[folds["outer"] != o].set_index("player_id")[f"inner_{o}"]
            for i in range(FOLDS):
                rows = np.flatnonzero((qc["outer"].to_numpy() == o) & (qc["inner"].to_numpy() == i))
                jobs.append((seed, o, i, set(inner.index[inner != i]), "cal", rows))
    hc = {seed: {"eval": np.zeros((len(qe), 10)), "cal": np.zeros((len(qc), 10))} for seed in SEEDS}
    coefficients = []
    with mp.get_context("fork").Pool(workers) as pool:
        for n, (seed, o, i, kind, rows, p, coef) in enumerate(pool.imap_unordered(_fit, jobs), 1):
            hc[seed][kind][rows] = p
            coefficients.append({"seed": seed, "outer": o, "inner": i, "per_k": coef})
            if n % 10 == 0:
                print(f"{n}/{len(jobs)} B2 fits, {time.time() - started:.0f}s", flush=True)
    for seed in SEEDS:
        for kind in ("eval", "cal"):
            if not np.isfinite(hc[seed][kind]).all() or (hc[seed][kind].sum(axis=1) == 0).any():
                raise ValueError("missing HC posteriors")
    truth_e, truth_c = qe["rank_id"].to_numpy(), qc["rank_id"].to_numpy()
    ke, kc = qe["k"].to_numpy(), qc["k"].to_numpy()
    arms = ("HC", "HC_cal", "HC+S", "HC+A0")
    scores = {arm: {} for arm in arms}
    chosen = []
    for seed in SEEDS:
        lp_e, lp_c = _logp(hc[seed]["eval"]), _logp(hc[seed]["cal"])
        scores["HC"][seed] = _score(hc[seed]["eval"], truth_e)
        for arm, partner, betas in (("HC_cal", "S", (0.0,)), ("HC+S", "S", BETAS), ("HC+A0", "A0", BETAS)):
            out = np.zeros(len(qe))
            for o in range(FOLDS):
                for k in K_BUCKETS:
                    c_rows = (qc["outer"].to_numpy() == o) & (kc == k)
                    e_rows = (qe["outer"].to_numpy() == o) & (ke == k)
                    alpha, beta = _select(lp_c[c_rows], frozen[(partner, "cal")][c_rows], truth_c[c_rows], betas)
                    out[e_rows] = _score(_fuse(lp_e[e_rows], frozen[(partner, "eval")][e_rows], alpha, beta),
                                         truth_e[e_rows])
                    chosen.append({"arm": arm, "seed": seed, "outer": o, "k": k, "alpha": alpha, "beta": beta})
            scores[arm][seed] = out
    alone = {arm: _score(np.exp(frozen[(arm, "eval")]), truth_e) for arm in ("S", "A0")}
    mean = {arm: np.mean([scores[arm][s] for s in SEEDS], axis=0) for arm in arms}
    player = pd.factorize(qe["player_id"])[0]
    rng = np.random.default_rng(2029)
    result = {"questions": int(len(qe)), "arms": {arm: hh.by_k(mean[arm], ke) for arm in arms},
              "arms_by_seed": {arm: {str(s): hh.by_k(scores[arm][s], ke) for s in SEEDS} for arm in arms},
              "alone": {arm: hh.by_k(v, ke) for arm, v in alone.items()}, "diffs": {}}
    for a, b in (("HC_cal", "HC+S"), ("HC+A0", "HC+S"), ("HC", "HC+S"), ("HC", "HC_cal"), ("HC_cal", "HC+A0")):
        entry = hh.difference(mean[b], mean[a], ke, player, rng)
        entry["seed_equal_k"] = {str(s): round(hh.equal_k(scores[b][s] - scores[a][s], ke), 4) for s in SEEDS}
        result["diffs"][f"{b} - {a}"] = entry
    selected = pd.DataFrame(chosen)
    result["beta_selected"] = {
        arm: {str(k): {str(b): int(n) for b, n in v["beta"].value_counts().sort_index().items()}
              for k, v in selected[selected["arm"] == arm].groupby("k")} for arm in ("HC+S", "HC+A0")}
    result["alpha_on_edge"] = {arm: int(selected[(selected["arm"] == arm) & selected["alpha"].isin(
        (ALPHAS[0], ALPHAS[-1]))].shape[0]) for arm in ("HC_cal", "HC+S", "HC+A0")}
    u, n = result["diffs"]["HC+S - HC_cal"], result["diffs"]["HC+S - HC+A0"]
    utility = bool(u["equal_k"] >= GAIN and u["prob_positive"] >= PROB and min(u["by_k"].values()) >= -MAX_K_DROP
                   and all(v > 0 for v in u["seed_equal_k"].values()))
    information = bool(n["equal_k"] > 0 and n["prob_positive"] >= PROB)
    never_s = bool((selected.loc[selected["arm"] == "HC+S", "beta"] == 0).all())
    result["gates"] = {"utility": utility, "new_information": information, "S_never_selected": never_s}
    result["seconds"] = round(time.time() - started)
    table = qe[["question_id", "player_id", "rank_id", "k", "outer"]].copy()
    for arm in arms:
        for s in SEEDS:
            table[f"{arm}_{s}"] = scores[arm][s]
    for arm, v in alone.items():
        table[f"{arm}_alone"] = v
    table.to_parquet(OUT / "scores.parquet")
    selected.to_parquet(OUT / "selected_weights.parquet")
    np.savez_compressed(OUT / "hc_posteriors.npz", **{f"{kind}_{s}": hc[s][kind] for s in SEEDS for kind in ("eval", "cal")})
    (OUT / "b2_coefficients.json").write_text(json.dumps(coefficients))
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["htables", "haudit", "prepare", "run"])
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    {"htables": htables, "haudit": haudit, "prepare": prepare, "run": lambda: run(args.workers)}[args.command]()


if __name__ == "__main__":
    main()
