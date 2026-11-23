"""Extract paired KataRank ratings and evaluate the locked local three-arm comparison."""

from __future__ import annotations

import argparse
import heapq
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from scripts.strength import rank_katarank_pilot as pilot
from scripts.strength import rank_opp_calibrate as opp
from scripts.strength import rank_opp_fuse as fuse
from scripts.strength import rank_prod_a0_fusion as prod
from scripts.strength import rank_more_players as more
from scripts.strength import rank_nonlin as nl
from scripts.strength.rank_qmerge import official

SMOKE = ROOT / "outputs/rank/katarank/845543"
NETWORK = ROOT / "data/external/katarank_nano4/kata1-b28c512nbt-s12763923712-d5805955894.bin.gz"
ARMS = ("lite", "full")
SHARDS = 8


def fit_sample(fit: pd.DataFrame) -> pd.DataFrame:
    eligible = defaultdict(lambda: defaultdict(dict))
    for (rank, player, color), games in fit.groupby(["rank_id", "player_id", "player_color"], sort=True):
        if len(games) >= 20:
            eligible[rank][player][color] = np.sort(games["game_index"].to_numpy())
    rng = np.random.default_rng(2027)
    chosen = []
    for rank in range(10):
        players = sorted(eligible[rank])
        for pi in rng.choice(len(players), size=60, replace=False):
            colors = eligible[rank][players[pi]]
            games = colors[sorted(colors)[rng.integers(len(colors))]]
            chosen.extend(games[rng.choice(len(games), size=20, replace=False)])
    picked = fit.set_index("game_index").loc[chosen].reset_index()
    assert len(picked) == picked.game_index.nunique() == 12000
    assert picked.player_id.nunique() == 600 and int(picked.length.sum()) == 2663143
    return picked


def plan(out: Path) -> None:
    out.mkdir()
    fit = fit_sample(pd.read_parquet(pilot.FIT))
    u = pd.concat([pd.read_parquet(pilot.FIT.with_name(f"{part}.parquet"))
                   for part in ("e2", "reserve")], ignore_index=True).set_index("game_index", verify_integrity=True)
    sets = prod.question_sets()
    e1 = sets["E1"]
    allowed = set(u.index)
    sets["E1"] = e1[e1.games.map(lambda games: set(games) <= allowed)].reset_index(drop=True)
    e2_disjoint = ~sets["E2"].player_id.isin(sets["E1"].player_id)
    assert (len(sets["E1"]), len(sets["E2"]), int(e2_disjoint.sum())) == (840, 2000, 651)
    assert not set(fit.player_id) & (set(sets["E1"].player_id) | set(sets["E2"].player_id))
    for name, q in sets.items():
        assert all(len(games) == k for games, k in zip(q.games, q.k))
        q.to_parquet(out / f"{name}.parquet", index=False)
    ids = sorted({g for q in sets.values() for games in q.games for g in games})
    evaluate = u.loc[ids].reset_index()
    assert len(evaluate) == 18785 and int(evaluate.length.sum()) == 4183127
    fields = ["game_index", "player_color", "rank_id", "handicap", "length"]
    sample = pd.concat([fit[fields].assign(pool="fit"), evaluate[fields].assign(pool="eval")], ignore_index=True)
    assert sample.game_index.is_unique and sample.handicap.isin([0, 1]).all()
    ordered = sample.sort_values(["length", "game_index"], ascending=[False, True])
    heap = [(0, i) for i in range(SHARDS)]
    assignment = {}
    for game, length in zip(ordered.game_index, ordered.length):
        count, shard = heapq.heappop(heap)
        assignment[game] = shard
        heapq.heappush(heap, (count + length, shard))
    sample["shard"] = sample.game_index.map(assignment)
    sample.to_parquet(out / "manifest.parquet", index=False)
    identity = json.loads((SMOKE / "result.json").read_text())
    paths = {"engine_sha256": SMOKE / "build/katago", "network_sha256": NETWORK}
    assert identity["shared_scan"]["pass"] and all(pilot.digest(p) == identity[key] for key, p in paths.items())
    for name, p in checkpoint_paths().items():
        assert pilot.digest(p) == identity["checkpoints"][name]
    report = {"pilot": SMOKE.name, "network_sha256": identity["network_sha256"],
              "engine_sha256": identity["engine_sha256"], "checkpoints": identity["checkpoints"],
              "manifest_sha256": pilot.digest(out / "manifest.parquet"),
              "positions_by_shard": [int(n) for n in sample.groupby("shard").length.sum()]}
    (out / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def checkpoint_paths():
    return {"full": SMOKE / "rank/nets/katarank/best.pt", "lite": SMOKE / "rank/nets/katarank_lite/best_lite.pt"}


def extract(out: Path, shard: int, scratch: Path, threads: int):
    import torch
    from katarank import InferenceWorkflow
    from katarank.model import KataRankModel

    start = monotonic()
    work = out / f"shard-{shard}"
    work.mkdir()
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    if len(devices) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected an isolated single-GPU step, nvidia-smi sees {devices}")
    gpu = devices[0].strip()
    print(f"shard {shard} device {gpu}", flush=True)
    tuning = SMOKE / "engine-home/opencltuning"
    (work / "opencl.cfg").write_text(
        f"homeDataDir = {scratch}\n"
        f"openclTunerFile = {next(tuning.glob('tune*_x19_y19_c512_*.txt'))}\n"
    )
    games = pd.read_parquet(out / "manifest.parquet", filters=[("shard", "=", shard)])
    sample = pilot.prepare(work, games, scratch=scratch)
    heads = {}
    for name, path in checkpoint_paths().items():
        model = KataRankModel.load(str(path))
        assert model.get_config()["input_dim"] == (1034 if name == "full" else 10)
        heads[name] = InferenceWorkflow(model, engine=None, device="cuda")
    buffer, parts = [], []
    checkpoints = work / "checkpoints"
    checkpoints.mkdir()

    def flush():
        if not buffer:
            return
        ids, values, black, white = zip(*buffer)
        meta = sample.loc[list(ids)]
        arrays = {"game_index": meta.game_index.to_numpy(), "player_color": meta.player_color.to_numpy(),
                  "length": meta.length.to_numpy(), "positions_by_color": np.column_stack([black, white])}
        for name in ARMS:
            ratings, probabilities = zip(*(value[name] for value in values))
            arrays[f"{name}_rating"] = np.stack(ratings)
            arrays[f"{name}_probabilities"] = np.stack(probabilities)
        path = checkpoints / f"{len(parts):04d}.npz"
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
        parts.append(path)
        buffer.clear()

    def checkpoint(game, values, black, white):
        buffer.append((game, values, black, white))
        if len(buffer) >= 256:
            flush()

    _, _, report = pilot.scan(work, sample, SMOKE / "build/katago", NETWORK, heads, 1034,
                              "full", threads, keep_scalars=False, progress_every=250,
                              allow_prefix=True, checkpoint=checkpoint)
    flush()
    blocks = []
    for path in parts:
        with np.load(path, allow_pickle=False) as saved:
            blocks.append(dict(saved))
    arrays = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    for key in ("game_index", "player_color", "length"):
        np.testing.assert_array_equal(arrays[key], sample[key].to_numpy())
    temp = work / "predictions.tmp.npz"
    np.savez_compressed(temp, **arrays)
    temp.replace(work / "predictions.npz")
    report.update({"gpu": gpu, "elapsed": monotonic() - start,
                   "sha256": pilot.digest(work / "predictions.npz")})
    (work / "done.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def read_features(out, manifest):
    reports, tables = [], []
    for shard in range(SHARDS):
        work = out / f"shard-{shard}"
        report = json.loads((work / "done.json").read_text())
        path = work / "predictions.npz"
        assert pilot.digest(path) == report["sha256"]
        expected = manifest[manifest.shard == shard].sort_values("game_index")
        with np.load(path, allow_pickle=False) as a:
            for key in ("game_index", "player_color", "length"):
                np.testing.assert_array_equal(a[key], expected[key].to_numpy())
            counts = a["positions_by_color"]
            assert counts.shape == (len(expected), 2) and (counts >= 5).all()
            positions = counts.sum(axis=1)
            assert (positions <= a["length"]).all() and positions.sum() == report["positions"]
            assert int((positions < a["length"]).sum()) == report["truncated_games"]
            table = pd.DataFrame(index=a["game_index"])
            for name in ARMS:
                rating, prob = a[f"{name}_rating"], a[f"{name}_probabilities"]
                assert rating.shape == (len(expected), 2) and np.isfinite(rating).all()
                assert prob.shape == (len(expected), 2, 29) and np.isfinite(prob).all() and (prob >= 0).all()
                np.testing.assert_allclose(prob.sum(-1), 1, rtol=0, atol=1e-5)
                table[[f"{name}_own", f"{name}_opp"]] = rating
            tables.append(table)
        reports.append(report)
    rates = pd.concat(tables, verify_integrity=True)
    assert len(rates) == len(manifest)
    if len({r["gpu"] for r in reports}) != SHARDS:
        raise ValueError("extraction did not use eight distinct GPU UUIDs")
    return rates, reports


def evaluate(out: Path):
    start = monotonic()
    manifest = pd.read_parquet(out / "manifest.parquet")
    rates, shards = read_features(out, manifest)
    sets = {name: pd.read_parquet(out / f"{name}.parquet") for name in ("E1", "E2")}
    e1, e2 = sets["E1"], sets["E2"]
    disjoint = ~e2.player_id.isin(e1.player_id).to_numpy()
    fit, ev, xf, xe = prod.train_data()
    prior = more.player_prior(fit[fit.role == "S"])
    fi = np.flatnonzero(fit.game_index.isin(manifest.loc[manifest.pool == "fit", "game_index"]))
    ei = np.flatnonzero(ev.game_index.isin(manifest.loc[manifest.pool == "eval", "game_index"]))
    fit, ev, xf, xe = fit.iloc[fi].reset_index(drop=True), ev.iloc[ei].reset_index(drop=True), xf[fi], xe[ei]
    assert (len(fit), len(ev), int(disjoint.sum())) == (12000, 18785, 651)
    assert not set(fit.player_id) & set(ev.player_id)
    yf, (ye,) = nl.oof("yH", fit, xf, [xe])
    xf, xe = np.column_stack([xf, yf]), np.column_stack([xe, ye])
    probs = {key: {arm: [] for arm in ("base", *ARMS)} for key in sets}
    for seed in prod.SEEDS:
        for arm in ("base", *ARMS):
            cols = [f"{arm}_own", f"{arm}_opp"]
            fa = xf if arm == "base" else np.column_stack([xf, rates.loc[fit.game_index, cols]])
            ea = xe if arm == "base" else np.column_stack([xe, rates.loc[ev.game_index, cols]])
            model = prod._fit(fit, fa, prior, seed)
            for name, q in sets.items():
                probs[name][arm].append(model.probabilities(ea, ev.game_index.to_numpy(), q))
        print(seed, "B2 three arms", round(monotonic() - start, 1), flush=True)
    probs = {name: {arm: np.stack(p) for arm, p in arms.items()} for name, arms in probs.items()}

    def paired(q, p, ref):
        delta, result = opp.score(q, p, ref)
        return {**result, **opp.interval(q, delta)}

    dev = {arm: paired(e1, probs["E1"][arm], probs["E1"]["base"]) for arm in ARMS}
    selected = max(ARMS, key=lambda arm: dev[arm]["equal_k"])
    print("E1 frozen choice", selected, dev[selected]["equal_k"], flush=True)
    holdout = {arm: paired(e2, probs["E2"][arm], probs["E2"]["base"]) for arm in ARMS}
    separate = {arm: paired(e2[disjoint], probs["E2"][arm][:, disjoint], probs["E2"]["base"][:, disjoint])
                for arm in ARMS}
    arms_pass = (dev[selected]["equal_k"] >= 0.003 and holdout[selected]["equal_k"] >= 0.003
                 and min(holdout[selected]["bucket"].values()) >= -0.005
                 and separate[selected]["equal_k"] > 0)
    summary = {"selected": selected, "single_gate": arms_pass, "E1_vs_base": dev, "E2_vs_base": holdout,
               "E2_without_E1_players": separate, "shards": shards}
    for name, q in sets.items():
        base = probs[name]["base"]
        scores = np.stack([official(score_rank(p), q.rank_id.to_numpy()) for p in base])
        summary[f"{name}_base"] = {str(k): float(scores[:, q.k.to_numpy() == k].mean()) for k in K_BUCKETS}
    if arms_pass:
        t = fuse.link_table()
        table = fuse.fit_delta(t[t["set"] == "s_opp"])
        u = t[t["set"] == "u_opp"]
        for name, q in sets.items():
            ll = opp.grouped_evidence(q.games, u, "game_index", fuse.game_ll(table, u, high=False))
            probs[name]["opp"] = np.stack([fuse.posterior(p, ll, 0.25) for p in probs[name]["base"]])
            probs[name]["combined"] = np.stack([fuse.posterior(p, ll, 0.25) for p in probs[name][selected]])
        opponent_dev = paired(e1, probs["E1"]["opp"], probs["E1"]["base"])
        preferred = "opp" if opponent_dev["equal_k"] > dev[selected]["equal_k"] else selected
        combo = {name: paired(q, probs[name]["combined"], probs[name][preferred]) for name, q in sets.items()}
        summary["opponent_vs_base"] = {"E1": opponent_dev, "E2": paired(e2, probs["E2"]["opp"], probs["E2"]["base"])}
        summary["combined"] = {"stronger_on_E1": preferred, "delta": combo,
                               "gate": combo["E2"]["equal_k"] >= 0.002
                               and min(combo["E2"]["bucket"].values()) >= -0.005}
    output = out / "local_posteriors.npz"
    np.savez_compressed(output, **{f"{name}_{arm}": p for name, arms in probs.items() for arm, p in arms.items()},
                        E2_without_E1=disjoint)
    summary["posteriors_sha256"] = pilot.digest(output)
    summary["cpu_seconds"] = monotonic() - start
    (out / "local_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Final choice", selected, "E2", holdout[selected]["equal_k"], "gate", arms_pass, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "extract", "evaluate"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    if args.stage == "plan":
        plan(args.out)
    elif args.stage == "extract":
        if args.shard is None or args.scratch is None:
            parser.error("extract requires --shard and --scratch")
        extract(args.out, args.shard, args.scratch, args.threads)
    else:
        evaluate(args.out)


if __name__ == "__main__":
    main()
