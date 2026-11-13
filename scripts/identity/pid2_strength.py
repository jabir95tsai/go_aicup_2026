"""Cross-competition strength term for the added competition-1 set (pool2_* / pid2_*).

Hypothesis: a question and its true candidate share one Yugen rank, so a candidate whose
rank posterior disagrees with the question's should drop. The term added to the current
fused score (opening + lambda * w * style + mu * HD-and-colour, pid2_bench summary) is

    nu * log sum_r P_question(r) * P_candidate(r)

where P_set = softmax(T_n * mean over the set's games of the V2 (A-024) per-game mean
log-prob), n = number of games in the set; T_n is fitted on Valid-S synthetic sets of n
games (never on the bench or the official files).

    infer-bench     GPU: V2 per-game log-probs of every pid2 bench game (training cache)
    infer-official  GPU: the same for pool2_* / pid2_* stores of --store-dir
    eval            CPU: temperatures, nu chosen on pool 0, reported on pool 1, for
                    (a) the full bench and (b) the bench restricted to players V2 never
                    trained on (strength fold 0 or -1). About 2/3 of bench players are in
                    V2's training folds, so (a) overstates the gain; (b) is the gate.
    predict         CPU: add nu * term to predict2's scores of --base-dir and write
                    predictions.parquet for pid_submission merge.

Outputs: outputs/pid/pid2_strength/. One GPU, a few minutes per infer step:

    sbatch -p dev -A GOV115094 --gres=gpu:1 -c 12 --mem=120G -t 01:00:00 \\
      -o outputs/pid/pid2_strength/infer-%j.out --chdir=/work/jason1122/go_aicup_2026 \\
      --wrap "PYTHONPATH=src ~/.conda/envs/go/bin/python -m scripts.identity.pid2_strength infer-bench \\
              && PYTHONPATH=src ~/.conda/envs/go/bin/python -m scripts.identity.pid2_strength infer-official"
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data import cv
from go_aicup.metrics import PID_WEIGHTS
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import assign, data, opening, twocolor
from go_aicup.tasks.strength.v2.aggregate import N_RANKS, fit_temperature

OUT = ROOT / "outputs" / "pid" / "pid2_strength"
V2_RUN = ROOT / "outputs" / "rank" / "v2" / "v2_strength"
VALID_S = V2_RUN / "validation_seen_category.npy"
BENCH_DIR = ROOT / "outputs" / "pid" / "pid2_bench" / "pid_ge2e_v1_short"
BANK = PROCESSED / "identity" / "pid2_bank.parquet"
FOLDS = PROCESSED / "cv" / "strength_folds_seed2027.parquet"
HANDICAP = ROOT / "data" / "raw" / "training" / "handicap.csv"
STORE_DIR = ROOT / "outputs" / "pid" / "day0_1112" / "store"
STORES = ("pool2_public", "pool2_private", "pid2_public", "pid2_private")
DRAWS = 5
SET_SIZES = (1, 2, 3, 5, 10)
NUS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
PREFIX = "test"


# ---------------------------------------------------------------- GPU inference
def _infer(work: Path, cache_dir: Path) -> None:
    """V2 model.pt over every sample of work/test_{game,ply}.npy -> work/category.npy."""
    import torch
    from go_aicup.tasks.strength.shared.training import configure_torch, seed_process
    from go_aicup.tasks.strength.v2.infer import infer_games
    from go_aicup.tasks.strength.v2.model import StrengthResNet

    out = work / "category.npy"
    if out.exists():
        print(f"{out} exists, skipping")
        return
    payload = torch.load(V2_RUN / "model.pt", map_location="cpu", weights_only=False)
    if payload.get("artifact_format") != "strength_v2_1":
        raise ValueError("not a Strength V2 model.pt")
    config = json.loads(json.dumps(payload["config"]))
    config["paths"].update(dataset_dir=str(work), cache_dir=str(cache_dir))
    config["evaluation"]["global_batch_size"] = 8192
    config["loader"]["num_workers"] = 4
    configure_torch(config)
    seed_process(config["seed"])
    device = torch.device("cuda", 0)
    model = StrengthResNet(**config["model"])
    model.load_state_dict(payload["model"])
    model.to(device)
    category = infer_games(model, config, PREFIX, device)
    np.save(out, category, allow_pickle=False)
    print(f"{work.name}: {len(category):,} games, {int(category['num_moves'].sum()):,} moves")


def infer_bench(_) -> None:
    meta = data.load_meta()
    games = np.sort(pd.read_parquet(BANK)["game"].to_numpy())
    color = meta["player_color"].to_numpy()[games].astype(np.int64)
    length = meta["length"].to_numpy()[games]
    plies = [np.arange(c, n, 2) for c, n in zip(color, length)]   # bank games are HD <= 1: plain alternation
    work = OUT / "bench"
    work.mkdir(parents=True, exist_ok=True)
    np.save(work / f"{PREFIX}_game.npy", np.concatenate([np.full(len(p), g, np.int64) for g, p in zip(games, plies)]))
    np.save(work / f"{PREFIX}_ply.npy", np.concatenate(plies).astype(np.int64))
    _infer(work, data.CACHE_DIR)


def infer_official(args) -> None:
    from go_aicup.tasks.strength.v2.predict import prepare

    for name in STORES:
        work = OUT / "official" / name
        prepare(args.store_dir, name, work)
        _infer(work, work / "cache")


# ---------------------------------------------------------------- posteriors
def per_game_logits(category: np.ndarray) -> dict[int, np.ndarray]:
    mean = category["log_probability_sum"] / category["num_moves"][:, None]
    return dict(zip(category["game_index"].astype(np.int64).tolist(), mean))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def fit_temperatures(meta: pd.DataFrame, seed: int = 2026) -> dict[int, float]:
    """T_n per set size n on Valid-S: disjoint n-game sets of one player, NLL of its rank."""
    category = np.load(VALID_S, allow_pickle=False)
    logits = category["log_probability_sum"] / category["num_moves"][:, None]
    players = meta["player_id"].astype(str).to_numpy()[category["game_index"].astype(np.int64)]
    labels = category["rank_id"].astype(np.int64)
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(category))).groupby(players).indices
    temps = {}
    for n in SET_SIZES:
        x, y = [], []
        for rows in groups.values():
            rows = rng.permutation(rows)
            for start in range(0, len(rows) - n + 1, n):
                take = rows[start:start + n]
                x.append(logits[take].mean(axis=0))
                y.append(labels[take[0]])
        t, edge = fit_temperature(np.array(x), np.array(y))
        temps[n] = t
        print(f"T_{n} = {t:.3f} from {len(y):,} Valid-S sets{' (grid edge!)' if edge else ''}")
    return temps


def set_posteriors(sets, logits: dict[int, np.ndarray], temps: dict[int, float]) -> np.ndarray:
    """[len(sets), 10]; a set of n games uses the temperature of the nearest fitted size."""
    out = np.empty((len(sets), N_RANKS))
    for i, games in enumerate(sets):
        n = min(temps, key=lambda m: (abs(m - len(games)), m))
        out[i] = softmax(temps[n] * np.mean([logits[int(g)] for g in games], axis=0))
    return out


def strength_term(p_query: np.ndarray, p_cand: np.ndarray) -> np.ndarray:
    return np.log(p_query @ p_cand.T + 1e-12)


# ---------------------------------------------------------------- bench evaluation
def score_table(scores, candidates, q) -> pd.Series:
    ranked = candidates[assign.top5(scores)]
    points = ((ranked == q["player_id"].to_numpy()[:, None]) * PID_WEIGHTS).sum(axis=1)
    ks = q["k"].to_numpy()
    return pd.Series({**{str(k): float(points[ks == k].mean()) for k in cv.PID2_K}, "all": float(points.mean())})


def run_out(bench_dir: Path) -> Path:
    """Eval outputs: OUT for the original encoder, OUT/<run> for any other."""
    return OUT if bench_dir.name == BENCH_DIR.name else OUT / bench_dir.name


def evaluate(args) -> None:
    from go_aicup.tasks.identity.predict import train_prefix

    bench_dir = args.bench_dir
    summary = json.loads((bench_dir / "summary.json").read_text())
    meta, prefix = train_prefix("d4")
    bank = pd.read_parquet(BANK)
    table = pd.read_csv(HANDICAP, usecols=["rank", "source_row", "handicap"])
    handicap = meta[["rank", "source_row"]].merge(table, on=["rank", "source_row"], how="left",
                                                  validate="1:1")["handicap"].to_numpy().astype(np.int64)
    color = meta["player_color"].to_numpy().astype(np.int64)
    pid = meta["player_id"].astype(str).to_numpy()
    rank_id = meta["rank_id"].to_numpy().astype(np.int64)
    saved = np.load(bench_dir / "embeddings.npz")
    position = np.full(len(meta), -1)
    position[saved["game_index"]] = np.arange(len(saved["game_index"]))
    hashes = opening.prefix_hashes(prefix)
    codes = np.minimum(handicap, 1) + 2 * color
    outside = ~np.isin(pid, bank["player_id"].unique())
    populations = (np.flatnonzero(outside & (color == 0)), np.flatnonzero(outside & (color == 1)))

    temps = fit_temperatures(meta)
    logits = per_game_logits(np.load(OUT / "bench" / "category.npy", allow_pickle=False))
    folds = pd.read_parquet(FOLDS).set_index("player_id")["fold"]
    unseen = set(folds.index[folds.isin([0, -1])])
    benches = {"full": bank, "unseen": bank[bank["player_id"].isin(unseen)]}
    print({name: b.groupby("pool")["player_id"].nunique().to_dict() for name, b in benches.items()})

    report, chosen = [], {}
    for name, part_bank in benches.items():
        items = {}
        for pool, draw in itertools.product((0, 1), range(DRAWS)):
            candidates, gallery, q = cv.draw_pid2(part_bank, pool, draw)
            query = list(q["games"])
            base = (twocolor.opening_scores(prefix, hashes, gallery, query, color, populations, summary["opening"])
                    + summary["lambda"] * summary["w"]
                    * twocolor.style_scores(saved["swap"], [position[g] for g in gallery], [position[g] for g in query])
                    + summary["mu"] * twocolor.category_scores(codes, gallery, query, 4))
            p_q, p_c = set_posteriors(query, logits, temps), set_posteriors(gallery, logits, temps)
            cand_rank = np.array([rank_id[g[0]] for g in gallery])
            q_rank = np.array([rank_id[g[0]] for g in query])
            items[pool, draw] = {
                "candidates": candidates, "q": q, "base": base, "term": strength_term(p_q, p_c),
                # diagnostics: perfect candidate ranks; the query/candidate posterior hit rates
                "oracle": np.log(p_q @ np.eye(N_RANKS)[cand_rank].T + 1e-12),
                "acc_q": float((p_q.argmax(1) == q_rank).mean()), "acc_c": float((p_c.argmax(1) == cand_rank).mean())}

        def mean_score(pool, fn):
            return pd.concat([score_table(fn(items[pool, d]), items[pool, d]["candidates"], items[pool, d]["q"])
                              for d in range(DRAWS)], axis=1).mean(axis=1)

        curve = {nu: mean_score(0, lambda it, nu=nu: it["base"] + nu * it["term"]) for nu in NUS}
        nu = max(NUS, key=lambda v: float(curve[v]["all"]))
        oracle_curve = {v: float(mean_score(0, lambda it, v=v: it["base"] + v * it["oracle"])["all"]) for v in NUS}
        nu_o = max(oracle_curve, key=oracle_curve.get)
        chosen[name] = nu
        print(f"\n[{name}] pool 0 curve (all):", {v: round(float(s["all"]), 4) for v, s in curve.items()})
        print(f"[{name}] pool 0 per-k curve:\n" + pd.DataFrame(curve).T.round(4).to_string())
        print(f"[{name}] nu={nu}; oracle curve:", {v: round(s, 4) for v, s in oracle_curve.items()})
        print(f"[{name}] posterior top-1 acc pool 1: question "
              f"{np.mean([items[1, d]['acc_q'] for d in range(DRAWS)]):.3f}, candidate "
              f"{np.mean([items[1, d]['acc_c'] for d in range(DRAWS)]):.3f}")
        variants = {"current": lambda it: it["base"],
                    f"current + {nu} x strength": lambda it: it["base"] + nu * it["term"],
                    "strength alone": lambda it: it["term"],
                    f"current + {nu_o} x oracle-candidate-rank": lambda it: it["base"] + nu_o * it["oracle"]}
        for pool in (0, 1):
            for vname, fn in variants.items():
                report.append({"bench": name, "pool": pool, "variant": vname,
                               **mean_score(pool, fn).round(4).to_dict()})
        gain = [float(score_table(variants[f"current + {nu} x strength"](items[1, d]), items[1, d]["candidates"],
                                  items[1, d]["q"])["all"]
                      - score_table(items[1, d]["base"], items[1, d]["candidates"], items[1, d]["q"])["all"])
                for d in range(DRAWS)]
        print(f"[{name}] pool 1 gain per draw:", np.round(gain, 4).tolist())
    frame = pd.DataFrame(report)
    print("\n" + frame.to_string(index=False))
    out = run_out(bench_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "eval.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"temperatures": {str(k): v for k, v in temps.items()}, "nu": chosen, "v2_run": str(V2_RUN),
         "base_summary": str(bench_dir / "summary.json")}, indent=1))


# ---------------------------------------------------------------- official files
def predict(args) -> None:
    """predict2's scores (--base-dir) + nu * strength term -> predictions.parquet in --out-dir."""
    summary = json.loads((run_out(args.bench_dir) / "summary.json").read_text())
    temps = {int(k): v for k, v in summary["temperatures"].items()}
    nu = summary["nu"][args.bench] if args.nu is None else args.nu
    scores = np.load(args.base_dir / "scores.npy")
    candidates = np.load(args.base_dir / "candidates.npy", allow_pickle=True).astype(str)
    old = pd.read_parquet(args.base_dir / "predictions.parquet")

    def store_sets(name, key, order):
        work = OUT / "official" / name
        games = pd.read_parquet(work / "cache" / "games.parquet")
        meta = pd.read_parquet(work / "store_meta.parquet")
        if len(games) != len(meta):
            raise ValueError(f"{name}: games dropped by the cache")
        cache_game = np.empty(len(meta), np.int64)
        cache_game[games["store_row"].to_numpy()] = games["game_index"].to_numpy()
        by = pd.Series(cache_game).groupby(meta[key].astype(str).to_numpy()).apply(np.asarray)
        return [by[k] for k in order], per_game_logits(np.load(work / "category.npy", allow_pickle=False))

    gal, gal_logits = store_sets(f"pool2_{args.part}", "player_id", candidates)
    qry, qry_logits = store_sets(f"pid2_{args.part}", "question_id", old["question_id"].astype(str))
    term = strength_term(set_posteriors(qry, qry_logits, temps), set_posteriors(gal, gal_logits, temps))
    fused = scores + nu * term
    ranked = assign.top5(fused)
    table = old[["question_id", "num_games"]].copy()
    for j in range(5):
        table[f"top{j + 1}"] = candidates[ranked[:, j]]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out_dir / "predictions.parquet", index=False)
    np.save(args.out_dir / "scores.npy", fused)
    np.save(args.out_dir / "term.npy", term)
    np.save(args.out_dir / "candidates.npy", candidates)
    changed = (table["top1"].to_numpy() != old["top1"].to_numpy())
    per_k = pd.Series(changed).groupby(table["num_games"].to_numpy()).sum().to_dict()
    info = {"nu": nu, "base_dir": str(args.base_dir), "top1_changed": int(changed.sum()), "per_k": per_k,
            "distinct_top1": int(table["top1"].nunique())}
    (args.out_dir / "info.json").write_text(json.dumps(info, indent=1, default=int))
    print(json.dumps(info, default=int))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("infer-bench").set_defaults(func=infer_bench)
    p = sub.add_parser("infer-official")
    p.add_argument("--store-dir", type=Path, default=STORE_DIR)
    p.set_defaults(func=infer_official)
    p = sub.add_parser("eval")
    p.add_argument("--bench-dir", type=Path, default=BENCH_DIR, help="pid2_bench run dir (summary + embeddings)")
    p.set_defaults(func=evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--bench-dir", type=Path, default=BENCH_DIR, help="the pid2_bench run the base scores came from")
    p.add_argument("--part", choices=("public", "private"), required=True)
    p.add_argument("--base-dir", type=Path, required=True, help="predict2 output dir (scores.npy, candidates.npy)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bench", choices=("full", "unseen"), default="unseen", help="which bench's nu to use")
    p.add_argument("--nu", type=float)
    p.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
