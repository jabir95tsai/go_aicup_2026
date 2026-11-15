"""Plan step B: board-baselined C on the C4g dev set (RESULTS 2026-11-15).

Upstream fixed: HumanSL + C v2 vectors, not fine-tuned. Per position and rank hypothesis r (vector
of the position's role), p_r = softmax over the positions the current C mask allows (empty, not
ko-banned, plus pass; suicide / superko are not excluded). Reference q = equal-weight arithmetic
mean of the ten p_r.
    raw_r       = log p_r(played)
    baselined_r = log p_r(played) - sum_a q(a) log p_r(a)      (= raw_r + H(q) + KL(q || p_r))
Both come from the same forward pass; per game and side the mean over its 50 sampled moves.

    SHARDS=0 sbatch slurm/rank_c4g_baseline.slurm           # short test: pack_hd{0,1}_0
    python -m scripts.strength.rank_c4g_baseline check      # vs few/A/dev_step0.npz, diagnostics
    SHARDS=1-7 sbatch slurm/rank_c4g_baseline.slurm         # the other 14 packs
    MODE=g0 sbatch slurm/rank_c4g_baseline.slurm            # paired comparison (CPU)

Engineering gate: max |new raw - dev_step0| per game feature <= RAW_TOLERANCE, same games / move
counts / ranks, all finite; otherwise stop. The comparison uses only this pass's raw and baselined.
Main test: [H | baselined C | HD] - [H | raw C | HD] (52 columns each), same folds / questions / B2 as
rank_c4g_hc. Gate: Equal-k >= +0.005, player bootstrap P >= 0.90, every k >= -0.005.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

OUT = ROOT / "outputs" / "rank" / "humansl" / "c4g" / "baseline"
DEV = ROOT / "outputs" / "rank" / "humansl" / "c4g" / "dev"
OLD = ROOT / "outputs" / "rank" / "humansl" / "c4g" / "few" / "A" / "dev_step0.npz"
VECTORS = ROOT / "outputs" / "rank" / "humansl" / "cvec_v2" / "learn" / "vectors.pt"
RAW_TOLERANCE, SUM_TOLERANCE, MOVES = 2e-3, 1e-3, 50
MIN_GAIN, MIN_PROB, MAX_K_DROP = 0.005, 0.90, 0.005
PACKS = [f"pack_hd{hd}_{i}" for i in range(8) for hd in (0, 1)]


def baseline_terms(logits, legal, target, ranks: int):
    """logits [n * ranks, A] (rank fastest), legal [n * ranks, A] bool, target [n * ranks].
    Returns raw [n, ranks], baseline [n, ranks] and the numeric checks of this batch."""
    import torch
    logp = torch.log_softmax(logits.double().masked_fill(~legal, float("-inf")), dim=-1)
    safe = torch.where(legal, logp, torch.zeros_like(logp))            # no 0 * -inf at masked points
    p = torch.where(legal, logp.exp(), torch.zeros_like(logp))
    n = len(logits) // ranks
    p, safe = p.reshape(n, ranks, -1), safe.reshape(n, ranks, -1)
    q = p.mean(1)
    baseline = (q[:, None, :] * safe).sum(-1)
    raw = logp.gather(1, target[:, None])[:, 0].reshape(n, ranks)
    centred = (q[:, None, :] * (safe - baseline[:, :, None]) * legal.reshape(n, ranks, -1)).sum(-1)
    checks = {"policy_sum_error": float((p.sum(-1) - 1).abs().max()), "q_sum_error": float((q.sum(-1) - 1).abs().max()),
              "centred_expectation": float(centred.abs().max()),
              "finite": bool(torch.isfinite(raw).all() and torch.isfinite(baseline).all())}
    return raw, baseline, checks


def extract(shards: list[int], compile_model: bool) -> None:
    import torch
    import go_aicup.features.humansl_rankvec as rv
    from go_aicup.tasks.strength import humansl_finetune as ft
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    tic = time.monotonic()
    model = ft.RankPolicy(device, torch.load(VECTORS, map_location=device)["vectors"]).eval()
    forward = torch.compile(model, dynamic=False) if compile_model else model
    load_seconds, first = time.monotonic() - tic, True
    size = ft.EVAL_POSITIONS
    ranks = torch.arange(rv.RANKS, device=device).repeat(size)
    for name in [p for p in PACKS if int(p.rsplit("_", 1)[1]) in shards]:
        if (OUT / f"{name}.npz").exists():
            print(json.dumps({"pack": name, "skipped": "exists"}), flush=True)
            continue
        data = ft.load_shards([DEV / name], device)
        games = data["games"]
        totals = {key: torch.zeros((games * 2, rv.RANKS), device=device, dtype=torch.float64) for key in ("raw", "base")}
        moves = torch.zeros(games * 2, device=device, dtype=torch.float64)
        stats = {"offset_sum": 0.0, "offset_sq": 0.0, "spread_sum": 0.0, "positions": 0}
        checks = {"policy_sum_error": 0.0, "q_sum_error": 0.0, "centred_expectation": 0.0, "finite": True}
        positions = len(data["game"])
        steady_positions, steady_seconds, started = 0, 0.0, time.monotonic()
        with torch.no_grad():
            for start in range(0, positions, size):
                tick = time.monotonic()
                index = torch.arange(start, min(positions, start + size), device=device)
                n = len(index)
                if n < size:                                           # fixed shape for the compiled forward
                    index = torch.cat([index, index[-1:].repeat(size - n)])
                x = rv.unpack(data["bits"][index]).repeat_interleave(rv.RANKS, 0)
                logits = forward(x, data["globals"][index].repeat_interleave(rv.RANKS, 0), ranks,
                                 data["side"][index].repeat_interleave(rv.RANKS, 0))
                legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).reshape(len(x), rv.AREA) > 0
                legal = torch.cat([legal, torch.ones_like(legal[:, :1])], dim=1)
                raw, base, batch = baseline_terms(logits, legal, data["target"][index].repeat_interleave(rv.RANKS, 0),
                                                  rv.RANKS)
                raw, base = raw[:n], base[:n]
                slot = data["game"][index[:n]] * 2 + data["side"][index[:n]]
                totals["raw"].index_add_(0, slot, raw)
                totals["base"].index_add_(0, slot, base)
                moves.index_add_(0, slot, torch.ones(n, device=device, dtype=torch.float64))
                offset = base.mean(1)
                stats["offset_sum"] += float(offset.sum())
                stats["offset_sq"] += float((offset * offset).sum())
                stats["spread_sum"] += float(base.std(1).sum())
                stats["positions"] += n
                for key in ("policy_sum_error", "q_sum_error", "centred_expectation"):
                    checks[key] = max(checks[key], batch[key])
                checks["finite"] &= batch["finite"]
                torch.cuda.synchronize()
                if first:
                    first_seconds, first = time.monotonic() - tick, False
                else:
                    steady_positions, steady_seconds = steady_positions + n, steady_seconds + time.monotonic() - tick
        truth = torch.zeros(games, dtype=torch.long, device=device)
        truth.index_put_((data["game"],), data["rank"])
        if not checks["finite"] or max(checks["policy_sum_error"], checks["q_sum_error"],
                                       checks["centred_expectation"]) > SUM_TOLERANCE:
            raise ValueError(f"{name}: numeric self-check failed: {checks}")
        np.savez(OUT / f"{name}.npz", raw=totals["raw"].reshape(-1, 2, rv.RANKS).cpu().numpy(),
                 base=totals["base"].reshape(-1, 2, rv.RANKS).cpu().numpy(), moves=moves.reshape(-1, 2).cpu().numpy(),
                 truth=truth.cpu().numpy(), game_index=data["game_index"])
        report = {"pack": name, "games": games, "positions": positions, "seconds": round(time.monotonic() - started, 1),
                  "steady_positions_per_second": round(steady_positions / max(steady_seconds, 1e-9), 1),
                  "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "checks": checks,
                  "position_stats": stats}
        if "first_seconds" in locals():
            report |= {"load_seconds": round(load_seconds, 1), "first_batch_seconds": round(first_seconds, 1)}
            del first_seconds
        (OUT / f"{name}.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report), flush=True)
        del data


def load_done() -> dict:
    """Concatenated per-game sums of the finished packs."""
    parts = [np.load(OUT / f"{name}.npz") for name in PACKS if (OUT / f"{name}.npz").exists()]
    if not parts:
        raise SystemExit("no finished pack")
    out = {key: np.concatenate([p[key] for p in parts]) for key in ("raw", "base", "moves", "truth", "game_index")}
    out["packs"] = len(parts)
    return out


def check() -> dict:
    new, old = load_done(), np.load(OLD)
    if len(np.unique(new["game_index"])) != len(new["game_index"]):
        raise ValueError("duplicate games across packs")
    order = pd.Series(np.arange(len(old["game_index"])), index=old["game_index"]).loc[new["game_index"]].to_numpy()
    same = {"moves_all_50": bool((new["moves"] == MOVES).all()),
            "moves_equal_old": bool(np.array_equal(new["moves"], old["moves"][order])),
            "truth_equal_old": bool(np.array_equal(new["truth"], old["truth"][order])),
            "finite": bool(np.isfinite(new["raw"]).all() and np.isfinite(new["base"]).all())}
    diff = np.abs(new["raw"] / new["moves"][:, :, None] - old["sides"][order] / old["moves"][order][:, :, None]).ravel()
    reports = [json.loads((OUT / f"{name}.json").read_text()) for name in PACKS if (OUT / f"{name}.json").exists()]
    positions = sum(r["position_stats"]["positions"] for r in reports)
    offset_mean = sum(r["position_stats"]["offset_sum"] for r in reports) / positions
    offset_sq = sum(r["position_stats"]["offset_sq"] for r in reports) / positions
    game_base = new["base"] / new["moves"][:, :, None]                       # [games, 2, 10] mean baseline term
    game_raw = new["raw"] / new["moves"][:, :, None]
    result = {"packs": new["packs"], "games": int(len(new["game_index"])), "same": same,
              "raw_abs_diff": {"median": float(np.median(diff)), "p99": float(np.quantile(diff, 0.99)),
                               "max": float(diff.max()), "tolerance": RAW_TOLERANCE},
              "numeric_max": {key: max(r["checks"][key] for r in reports)
                              for key in ("policy_sum_error", "q_sum_error", "centred_expectation")},
              "baseline_diagnostics": {
                  "position_common_offset_mean": round(offset_mean, 4),
                  "position_common_offset_sd": round(float(np.sqrt(offset_sq - offset_mean ** 2)), 4),
                  "position_between_rank_sd_mean": round(
                      sum(r["position_stats"]["spread_sum"] for r in reports) / positions, 4),
                  "game_common_offset_sd": [round(float(v), 4) for v in game_base.mean(2).std(0)],
                  "game_between_rank_sd_mean": [round(float(v), 4) for v in game_base.std(2).mean(0)],
                  "game_raw_common_sd": [round(float(v), 4) for v in game_raw.mean(2).std(0)],
                  "game_raw_between_rank_sd_mean": [round(float(v), 4) for v in game_raw.std(2).mean(0)]},
              "throughput": [{k: r.get(k) for k in ("pack", "positions", "seconds", "steady_positions_per_second",
                                                    "peak_memory_gib", "load_seconds", "first_batch_seconds")}
                             for r in reports]}
    result["pass"] = bool(all(same.values()) and diff.max() <= RAW_TOLERANCE)
    (OUT / "check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


def _variance_share(x: np.ndarray, player: np.ndarray) -> float:
    """Mean over columns of var(player means) / total var (descriptive)."""
    frame = pd.DataFrame(x)
    return round(float((frame.groupby(player).mean().var() / frame.var()).mean()), 4)


def g0() -> None:
    import go_aicup.tasks.strength.humansl as hs
    from go_aicup.tasks.strength.v2.aggregate import official_scores
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_c4g as g
    import scripts.strength.rank_c4g_hc as hc
    import scripts.strength.rank_hier_head as hh
    import scripts.strength.rank_more_players as more
    verdict = check()
    if verdict["packs"] != len(PACKS) or not verdict["pass"]:
        raise SystemExit("engineering check not passed on all 16 packs; stop")
    games, h, have = hc.load()
    if not have.all():
        raise SystemExit("H incomplete")
    new = load_done()
    order = pd.Series(np.arange(len(new["game_index"])), index=new["game_index"]).loc[games["game_index"]].to_numpy()
    raw = (new["raw"] / new["moves"][:, :, None])[order].reshape(len(games), 20)
    bl = ((new["raw"] - new["base"]) / new["moves"][:, :, None])[order].reshape(len(games), 20)
    flags = hf.hd1_columns(games)
    xs = {"H": np.hstack([h, flags]), "C_raw": raw, "C_bl": bl, "HC_raw": np.hstack([h, raw, flags]),
          "HC_bl": np.hstack([h, bl, flags])}
    fold = g._fold_of(games)
    game_fold = fold.loc[games["player_id"].astype(str)].to_numpy()
    questions = g._questions(games, np.random.default_rng(g.SEED + 1))
    q_fold = fold.loc[questions["player_id"]].to_numpy()
    k, truth = questions["k"].to_numpy(), questions["rank_id"].to_numpy()
    gi = games["game_index"].to_numpy()
    scores = {arm: np.zeros(len(questions)) for arm in xs}
    hs.SEED = g.SEED
    for f in range(g.FOLDS):
        train, test = game_fold != f, q_fold == f
        prior = more.player_prior(games[train])
        for arm, x in xs.items():
            model = more.PriorMatched(prior).fit(games[train].reset_index(drop=True), pd.DataFrame(), x[train])
            scores[arm][test] = official_scores(score_rank(model.probabilities(x, gi, questions[test])), truth[test])
        print(f"fold {f} done", flush=True)
    player = pd.factorize(questions["player_id"])[0]
    game_player = games["player_id"].astype(str).to_numpy()
    player_truth = games.groupby(game_player)["rank_id"].first()
    result = {"games": int(len(games)), "players": int(games["player_id"].nunique()), "arms": {}, "diffs": {}}
    for arm, x in xs.items():
        result["arms"][arm] = hf.by_k(scores[arm], k)
        if arm.startswith("C_"):
            pooled = pd.DataFrame(x).groupby(game_player).mean()
            result["arms"][arm] |= {"game_mid_d": g._mid_dprime(x, games["rank_id"].to_numpy()),
                                    "player_mid_d": g._mid_dprime(pooled.to_numpy(),
                                                                  player_truth.loc[pooled.index].to_numpy()),
                                    "between_player_variance_share": _variance_share(x, game_player)}
    rng = np.random.default_rng(g.SEED + 2)
    for a, b in (("HC_raw", "HC_bl"), ("C_raw", "C_bl"), ("H", "HC_raw"), ("H", "HC_bl")):
        result["diffs"][f"{b}-{a}"] = hh.difference(scores[b], scores[a], k, player, rng)
    main_diff = result["diffs"]["HC_bl-HC_raw"]
    result["verdict"] = "pass" if (main_diff["equal_k"] >= MIN_GAIN and main_diff["prob_positive"] >= MIN_PROB
                                   and min(main_diff["by_k"].values()) >= -MAX_K_DROP) else "fail"
    (OUT / "g0_baseline.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def _shards(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["extract", "check", "g0"])
    parser.add_argument("--shards", type=_shards, default=list(range(8)))
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.shards, not args.no_compile)
    else:
        {"check": check, "g0": g0}[args.command]()


if __name__ == "__main__":
    main()
