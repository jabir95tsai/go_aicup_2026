"""Run the same HD0/HD1 games through the pinned KataRank Lite and Full heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from go_aicup.data import sgf
from go_aicup.paths import ROOT
from go_aicup.tasks.identity.data import CACHE_DIR

RANK_COMMIT = "d60f17477c25461ca9e3c73563d029407d014adf"
ENGINE_COMMIT = "6cd855a91543d5480bd3423133976a6951a50f14"
FIT = ROOT / "outputs/rank/humansl/cvec2_policy/all/821286-mask-reuse/cpu/fit.parquet"
NET18 = ROOT / ("data/external/kata1-b18c384nbt-s9996604416/"
                "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select_games() -> pd.DataFrame:
    fit = pd.read_parquet(FIT, columns=["game_index", "player_color", "rank_id", "handicap", "length"])
    selected = []
    for (rank, color, hd), group in fit.groupby(["rank_id", "player_color", "handicap"]):
        order = group.sort_values(["length", "game_index"])
        fractions = [0.2, 0.8] + ([0.5] if (rank + color) % 2 == hd else [])
        selected.append(order.iloc[[round((len(order) - 1) * f) for f in fractions]])
    sample = pd.concat(selected, ignore_index=True)
    assert len(sample) == sample.game_index.nunique() == 100
    assert len(sample.groupby(["rank_id", "player_color", "handicap"])) == 40
    return sample


def to_sgf(moves: np.ndarray, colors: np.ndarray, hd: int) -> str:
    if hd not in (0, 1) or len(moves) != len(colors) or not np.array_equal(colors, np.arange(len(moves)) % 2):
        raise ValueError("expected alternating HD0/HD1 play starting with Black")
    komi = 6.5 if hd == 0 else 0.5
    return f"(;GM[1]FF[4]SZ[19]RU[Japanese]KM[{komi}]" + "".join(
        f";{'BW'[c]}[{sgf.index_to_point(int(m))}]" for m, c in zip(moves, colors)
    ) + ")"


def prepare(out: Path, sample=None, *, scratch: Path | None = None) -> pd.DataFrame:
    sample = select_games() if sample is None else sample
    size = len(sample)
    metadata = pd.read_parquet(
        CACHE_DIR / "games.parquet", columns=["game_index", "shard_id", "local_offset", "length", "player_color"],
        filters=[("game_index", "in", sample.game_index.tolist())],
    )
    sample = sample.merge(metadata, validate="one_to_one", on=["game_index", "length", "player_color"])
    if len(sample) != size:
        raise ValueError("selected games do not align with the board cache")
    sgf_dir = (scratch or out) / "sgf"
    sgf_dir.mkdir()
    rows = []
    for shard, games in sample.groupby("shard_id", sort=True):
        prefix = CACHE_DIR / f"shard_{shard:05d}"
        moves, colors = (np.load(str(prefix) + f"_{kind}.npy", mmap_mode="r", allow_pickle=False)
                         for kind in ("moves", "colors"))
        for row in games.itertuples():
            span = slice(row.local_offset, row.local_offset + row.length)
            sgf_path = sgf_dir / f"p{row.game_index:07d}.sgf"
            sgf_path.write_text(to_sgf(moves[span], colors[span], row.handicap))
            rows.append({"id": sgf_path.stem, "game_index": row.game_index, "rank_id": row.rank_id,
                         "handicap": row.handicap, "player_color": row.player_color, "length": row.length,
                         "black": int((colors[span] == 0).sum()), "white": int((colors[span] == 1).sum()),
                         "path": str(sgf_path), "sgf_sha256": digest(sgf_path)})
    sample = pd.DataFrame(rows).sort_values("id")
    sample.to_parquet(out / "sample.parquet", index=False)
    (out / "input.csv").write_text("sgf_path,black,white,score,black_elo,white_elo,set\n"
                                  + "".join(f"{path},,,,,,I\n" for path in sample.path))
    return sample.set_index("id")


def prediction(result, color: int) -> tuple[np.ndarray, np.ndarray]:
    probs = np.stack([result["b_rank_probs"].numpy(), result["w_rank_probs"].numpy()]).astype("float32")
    rating = np.array([result["b_rating"], result["w_rating"]], dtype="float32")
    assert probs.shape == (2, 29) and np.isfinite(probs).all() and np.isfinite(rating).all()
    assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5, rtol=0)
    order = [color, 1 - color]
    return rating[order], probs[order]


def scan(out, sample, binary, network, models, width, label, threads, *,
         keep_scalars=True, progress_every=25, allow_prefix=False, checkpoint=None):
    from katarank import KataGoEngine

    engine = KataGoEngine(model=str(network), config=str(out / "opencl.cfg"), katago_bin=str(binary))
    cmd = engine._base_cmd() + ["-list", str(out / "input.csv"), "-stream",
                               "-threads", str(threads), "-batch-size", str(threads)]
    if width == 10:
        cmd.append("-no-trunk")
    started = monotonic()
    first_game = None
    pending, rows, features = {}, {}, {}
    positions, truncated = 0, 0
    with (out / f"{label}.engine.log").open("wb") as log:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log) as proc:
            try:
                for side, moves, info in engine._read_stream(proc.stdout):
                    game = info["game_id"]
                    if side not in ("B", "W") or game not in sample.index or game in rows or side in pending:
                        raise ValueError(f"unexpected frame {game}/{side}")
                    if pending and game not in {entry[0] for entry in pending.values()}:
                        raise ValueError(f"missing side before {game}")
                    row = sample.loc[game]
                    count = row["black" if side == "B" else "white"]
                    lengths_ok = 5 <= len(moves) <= count if allow_prefix else len(moves) == count
                    if moves.shape != (len(moves), width) or not lengths_ok or not np.isfinite(moves).all():
                        raise ValueError(f"{game}/{side} expected {(count, width)}, got {moves.shape}")
                    if width == 1034 and any(not np.any(moves[:, sl]) for sl in (slice(10, 522), slice(522, 1034))):
                        raise ValueError(f"{game}/{side} produced empty trunk or pick")
                    pending[side] = (game, moves)
                    if len(pending) != 2:
                        continue
                    black, white = (pending[c][1] for c in ("B", "W"))
                    pending.clear()
                    if keep_scalars:
                        features[game] = np.concatenate([black[:, :10], white[:, :10]], axis=0)
                    rows[game] = {key: prediction(model.rank_frames(black, white, game), int(row.player_color))
                                  for key, model in models.items()}
                    actual_positions = len(black) + len(white)
                    positions += actual_positions
                    if actual_positions < row.length:
                        truncated += 1
                        print(f"{label} prefix {game} {actual_positions}/{row.length}", flush=True)
                    if checkpoint is not None:
                        checkpoint(game, rows[game], len(black), len(white))
                    if first_game is None:
                        first_game = monotonic()
                        first_positions = actual_positions
                    if len(rows) % progress_every == 0:
                        print(label, len(rows), "games", round(monotonic() - started, 2), "s", flush=True)
            except BaseException:
                proc.kill()
                raise
            if proc.wait() != 0:
                raise RuntimeError(f"{label} exited {proc.returncode}: {(out / f'{label}.engine.log').read_text()[-2000:]}")
    if pending or set(rows) != set(sample.index):
        raise ValueError(f"{label} completed {len(rows)}/{len(sample)} games")
    end = monotonic()
    return rows, features, {
        "games": len(rows), "positions": positions, "expected_positions": int(sample.length.sum()),
        "truncated_games": truncated, "width": width,
        "first_game_seconds": first_game - started, "seconds": end - started,
        "positions_per_second": (positions - first_positions) / (end - first_game),
    }


def main() -> None:
    import torch
    from katarank import InferenceWorkflow
    from katarank.model import KataRankModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank-source", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    out = args.out.resolve()
    (out / "engine-home").mkdir(exist_ok=True)
    (out / "opencl.cfg").write_text(
        f"homeDataDir = {out / 'engine-home'}\n"
        "openclReTunePerBoardSize = true\n"
    )
    sample = prepare(out)
    assert torch.cuda.is_available(), "H200 not visible to PyTorch"
    checkpoints = {"full": args.rank_source / "nets/katarank/best.pt",
                   "lite": args.rank_source / "nets/katarank_lite/best_lite.pt"}
    heads = {}
    for name, width in (("full", 1034), ("lite", 10)):
        model = KataRankModel.load(str(checkpoints[name]))
        if model.get_config()["input_dim"] != width:
            raise ValueError(f"unexpected {name} model dimension")
        heads[name] = InferenceWorkflow(model, engine=None, device="cuda")

    report = {"rank_commit": RANK_COMMIT, "engine_commit": ENGINE_COMMIT,
              "engine_sha256": digest(args.engine), "sample_sha256": digest(out / "sample.parquet"),
              "checkpoints": {name: digest(path) for name, path in checkpoints.items()},
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(),
              "network": args.network.name, "network_sha256": digest(args.network), "threads": args.threads}
    readings = {}
    full_data = full_features = None
    try:
        full_data, full_features, report["full"] = scan(
            out, sample, args.engine, args.network, {"full": heads["full"], "lite_from_full": heads["lite"]},
            1034, "full", args.threads)
        readings.update(full_data)
    except (RuntimeError, ValueError, AssertionError) as e:
        report["full_error"] = str(e)
        print("Full:", e, flush=True)

    for network, label in ((args.network, "lite"), (NET18, "lite_b18")):
        try:
            lite_data, lite_features, report[label] = scan(
                out, sample, args.engine, network, {"lite": heads["lite"]}, 10, label, args.threads)
        except (RuntimeError, ValueError, AssertionError) as e:
            report[f"{label}_error"] = str(e)
            print(label, e, flush=True)
            continue
        for game, values in lite_data.items():
            readings.setdefault(game, {}).update(values)
        report["lite_network"] = network.name
        report["lite_network_sha256"] = digest(network)
        if label == "lite" and full_data:
            max_feature = max(float(np.max(np.abs(full_features[g] - lite_features[g]))) for g in sample.index)
            max_rating = max(float(np.max(np.abs(readings[g]["lite"][0] - readings[g]["lite_from_full"][0])))
                             for g in sample.index)
            max_probability = max(float(np.max(np.abs(readings[g]["lite"][1] - readings[g]["lite_from_full"][1])))
                                  for g in sample.index)
            report["shared_scan"] = {"max_abs_scalar": max_feature, "max_abs_rating": max_rating,
                                     "max_abs_probability": max_probability,
                                     "pass": max_rating <= 1e-3 and max_probability <= 1e-3}
        break
    report["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
    keys = sorted({key for values in readings.values() for key in values})
    arrays = {"game_index": sample.game_index.to_numpy(), "player_color": sample.player_color.to_numpy()}
    for key in keys:
        rating, probabilities = zip(*(readings[g][key] for g in sample.index))
        arrays[f"{key}_rating"] = np.stack(rating)
        arrays[f"{key}_probabilities"] = np.stack(probabilities)
    np.savez_compressed(out / "predictions.npz", **arrays)
    report["predictions_sha256"] = digest(out / "predictions.npz")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if "lite_network" not in report or not full_data:
        raise SystemExit("partial KataRank short test; see result.json")


if __name__ == "__main__":
    main()
