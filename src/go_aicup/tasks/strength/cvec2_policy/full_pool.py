"""Train the four full-pool arms and export paired Public LB candidates."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from threading import Event
from time import monotonic

from filelock import FileLock
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength.policy_residual.epoch import epoch_batches
from scripts.strength.rank_more_players import PriorMatched, player_prior

from . import all_data, all_report, all_run, full_cache, hc5, joint, official, planes


DRAW_SEED = 2027
HEAD_SEED = 2028
GROUPS = ("fit", "e2", "reserve")
FILES = ("full_pool.py", "hc5.py", "official.py", "full_cache.py", "joint.py",
         "all_data.py", "all_report.py", "encoder.py", "planes.py", "train.py")


def read_json(path):
    return json.loads(path.read_text())


def source_data(source):
    config = read_json(source / "config.json")
    packed = Path(config["planes"])
    all_report.verify_files(packed, config["plane_files"])
    all_report.verify_files(source, {"cpu/complete.json": config["cpu_manifest"]})
    prepared = read_json(source / "cpu/complete.json")["files"]
    all_report.verify_files(source / "cpu", {f"{group}.parquet": prepared[f"{group}.parquet"] for group in GROUPS})
    stage = Path(config["stage1"])
    all_data._check_files(stage, read_json(stage / "complete.json")["files"], all_data.STAGE1_FILES)
    original = pd.read_parquet(stage / "games.parquet")
    games = pd.concat([pd.read_parquet(source / "cpu" / f"{g}.parquet") for g in GROUPS], ignore_index=True)
    if (not games.game_index.is_unique or len(games) != 183271 or games.player_id.nunique() != 7161
            or int(games.length.sum()) != 40866906 or not games.handicap.isin([0, 1]).all()):
        raise ValueError("full-pool game, player or move coverage differs")
    planes.PlaneReader(packed, games, groups=GROUPS)
    index = pd.Index(original.game_index)
    if not index.is_unique:
        raise ValueError("duplicate Stage 1 game index")
    rows = index.get_indexer(games.game_index)
    metadata = ["role", "player_id", "rank_id", "handicap", "player_color"]
    if (rows < 0).any() or not original.iloc[rows][metadata].reset_index(drop=True).equals(games[metadata]):
        raise ValueError("full-pool A feature order differs")
    with np.load(stage / "features.npz", allow_pickle=False) as stored:
        base = stored["A"][rows].astype(np.float32)
    if base.shape != (len(games), 52) or not np.isfinite(base).all():
        raise ValueError("invalid full-pool A feature coverage")
    return config, packed, games, base


def question_draws(games):
    prior = player_prior(games[games.role.isin(["S", "U"])])
    draws = list(PriorMatched(prior).synthetic_rows(games, games.rank_id.to_numpy(), DRAW_SEED))
    if tuple(k for k, _, _ in draws) != K_BUCKETS or any(len(y) != 20000 for _, _, y in draws):
        raise ValueError("full-pool B2 needs 20,000 questions per k")
    return draws


def fit_head(features, draws):
    fields = {key: [] for key in all_data.HEAD_FIELDS}
    rng = np.random.default_rng(HEAD_SEED)
    for _, rows, y in draws:
        x = np.concatenate([features[part].mean(axis=1) for part in np.array_split(rows, 10)]).astype(np.float64)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1, warm_start=True, max_iter=5000, tol=1e-8)
        model.coef_ = rng.normal(0, .01, (10, x.shape[1]))
        model.intercept_ = rng.normal(0, .01, 10)
        model.fit(scaler.transform(x), y)
        np.testing.assert_array_equal(model.classes_, np.arange(10))
        for key, a in zip(fields, (scaler.mean_, scaler.scale_, model.coef_, model.intercept_), strict=True):
            fields[key].append(a)
    return {key: np.stack(values) for key, values in fields.items()}


def save_head(path, head):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as handle:
        np.savez(handle, **head)
    temp.replace(path)


def train(source, out, updates, hc5_updates):
    started = monotonic()
    config, packed, games, base = source_data(source)
    code = {n: file_hash(Path(__file__).with_name(n)) for n in FILES}
    teacher = all_run.teacher_sources()
    if teacher != config["teacher_sources"]:
        raise ValueError("full-pool teacher identity differs from frozen source")
    devices = hc5.DEVICES
    cpu_threads = max(1, min(len(os.sched_getaffinity(0)), int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
                      - len(devices))
    settings = {"source": str(source.resolve()), "code": code, "teacher": teacher,
                "updates": updates, "hc5_updates": hc5_updates, "hc5_weights": hc5.WEIGHTS,
                "r10_devices": devices, "cnn_device": "cpu", "cnn_cpu_threads": cpu_threads,
                "hc5_devices": devices,
                "hc5_window_starts": hc5.WINDOW_STARTS, "hc5_window_length": hc5.WINDOW_LENGTH,
                "hc5_alpha": hc5.ALPHA, "draw_seed": DRAW_SEED,
                "head_seed": HEAD_SEED, "planes": str(packed.resolve()),
                "cpu_manifest": config["cpu_manifest"], "stage1": config["stage1"],
                "stage1_manifest": file_hash(Path(config["stage1"]) / "complete.json")}
    identity = object_hash(settings)
    spec = out / "full_pool.json"
    if spec.exists() and read_json(spec)["identity"] != identity:
        raise ValueError("full-pool run identity differs")
    write_json(spec, {"identity": identity, "settings": settings})
    completed = out / "complete.json"
    if completed.exists():
        marker = read_json(completed)
        if marker["identity"] != identity:
            raise ValueError("completed full-pool run identity differs")
        all_report.verify_files(out, marker["files"])
        return
    draws = question_draws(games)
    labels = np.concatenate([y for _, _, y in draws])
    k = np.concatenate([np.full(len(y), bucket) for bucket, _, y in draws])
    indices = [row for _, rows, _ in draws for row in rows]
    epochs = (max(updates, hc5_updates) + 2499) // 2500
    table = pd.DataFrame({"k": k})
    batches = np.concatenate([epoch_batches(table, DRAW_SEED, epoch, 8) for epoch in range(1, epochs + 1)])
    if len(batches) < max(updates, hc5_updates) or any(np.any((k[batches] == bucket).sum(axis=1) != 8)
                                                     for bucket in K_BUCKETS):
        raise ValueError("expected eight questions from each k per update")
    draw_path = out / "draws.npz"
    save_head(draw_path, {f"games_{bucket}": games.game_index.to_numpy()[rows] for bucket, rows, _ in draws}
              | {"labels": labels, "k": k, "batches": batches})
    save_head(out / "a-head.npz", fit_head(base, draws))
    fitted = (out / "hc5.pt").exists() and all((out / arm / "head.npz").is_file() for arm in official.ARMS)
    if not fitted:
        teacher_out = out / "teacher"
        teacher_out.mkdir(exist_ok=True)
        split = 165075
        teacher_config = config["teacher_config"] | {"cache_fraction": .90}
        cache = full_cache.overlap(games, split, teacher_out, packed, teacher_config,
                                   games.length.to_numpy()[:split], games.length.to_numpy()[split:],
                                   lambda _: None, phase_started=started, groups=(GROUPS[:2], GROUPS[2:]))
        all_run.teacher_parity(cache, base[:, :50], teacher_out / "teacher_parity.json")

        abort = Event()

        def small(arm, device, replicas=()):
            started = monotonic()
            destination = out / arm / "head.npz"
            if destination.exists():
                all_report.verify_files(out / arm, read_json(out / arm / "features.json")["files"])
                return monotonic() - started
            initial = fit_head(base, draws)
            joint.train_one(arm, cache, base, indices, labels, k, batches[:updates], initial,
                            device, out, identity, abort, replicas=replicas)
            joint.extract_one(arm, cache, base, device, out, identity, len(games), abort,
                              replicas=replicas)
            features = np.load(out / arm / "fit.npy", mmap_mode="r")
            save_head(destination, fit_head(features, draws))
            return monotonic() - started

        def cpu_cnn():
            try:
                torch.set_num_threads(cpu_threads)
                return small(official.ARMS[1], "cpu")
            except BaseException:
                abort.set()
                raise

        small_started = monotonic()
        with ThreadPoolExecutor(max_workers=1) as pool:
            cnn = pool.submit(cpu_cnn)
            try:
                r10_seconds = small(official.ARMS[0], devices[0], devices[1:])
                r10_path = out / official.ARMS[0]
                mother = joint.load_checkpoint(r10_path / "model.pt", official.ARMS[0], identity, None)
                parent = {name: file_hash(r10_path / name) for name in ("model.pt", "head.npz", "fit.npy")}
                hc5_identity = object_hash({"run": identity, "r10": parent})
                timing = {"r10_seconds": r10_seconds, "hc5_start_seconds": monotonic() - small_started}
                write_json(out / "small-stage-timing.json", timing)
                with np.load(r10_path / "head.npz", allow_pickle=False) as head:
                    hc5.train(cache, planes.PlaneReader(packed, games, groups=GROUPS), games,
                              np.load(r10_path / "fit.npy", mmap_mode="r"), mother["model"], dict(head),
                              indices, labels, k, batches[:hc5_updates], out, hc5_identity, DRAW_SEED,
                              abort=abort)
            except BaseException:
                if abort.is_set():
                    cnn.result()
                abort.set()
                raise
            write_json(out / "small-stage-timing.json", timing | {"cnn_seconds": cnn.result()})
    parent_path = out / official.ARMS[0]
    parent = {name: file_hash(parent_path / name) for name in ("model.pt", "head.npz", "fit.npy")}
    final = torch.load(out / "hc5.pt", map_location="cpu", weights_only=True)
    if final["identity"] != object_hash({"run": identity, "r10": parent}) or final["steps"] != hc5_updates:
        raise ValueError("full-pool HC5 final checkpoint differs")
    files = ["full_pool.json", "draws.npz", "a-head.npz", "hc5.pt"]
    files.extend(f"{arm}/{name}" for arm in official.ARMS
                 for name in ("model.pt", "head.npz", "fit.npy", "features.json"))
    write_json(completed, {"identity": identity, "files": all_report.fingerprint(out, files),
                           "games": len(games), "players": int(games.player_id.nunique()),
                           "fit_questions": len(labels), "seconds": monotonic() - started})


def export(out, job):
    marker = read_json(out / "complete.json")
    all_report.verify_files(out, marker["files"])
    spec = read_json(out / "full_pool.json")
    code = {n: file_hash(Path(__file__).with_name(n)) for n in FILES}
    if (spec["identity"] != marker["identity"] or spec["settings"]["teacher"] != all_run.teacher_sources()
            or spec["settings"]["code"] != code):
        raise ValueError("full-pool release model or teacher identity differs")
    source = Path(spec["settings"]["source"])
    config = read_json(source / "config.json")
    packed = Path(spec["settings"]["planes"])
    all_report.verify_files(packed, config["plane_files"])
    release = out / "official"
    release.mkdir(exist_ok=True)
    games, questions, base, reader = official.dataset(packed)
    predictions = {}
    state = torch.load(out / "hc5.pt", map_location="cpu", weights_only=True)["model"]
    r10 = joint.load_checkpoint(out / official.ARMS[0] / "model.pt", official.ARMS[0], marker["identity"], None)
    with np.load(out / "a-head.npz", allow_pickle=False) as head:
        predictions["A"] = [official.predict(questions, base, head)]
    cache = full_cache.infer(games, release, packed, config["teacher_config"], reader=reader)
    all_run.teacher_parity(cache, base[:, :50], release / "teacher_parity.json")
    official.extract(cache, base, release, out, marker["identity"])
    for arm in official.ARMS:
        with np.load(out / arm / "head.npz", allow_pickle=False) as head:
            predictions[arm] = [official.predict(questions, np.load(release / arm / "fit.npy"), head)]
    corrected = hc5.extract(cache, reader, games, np.load(release / official.ARMS[0] / "fit.npy"),
                            r10["model"], state)
    with np.load(out / official.ARMS[0] / "head.npz", allow_pickle=False) as head:
        predictions[hc5.ARM] = [official.predict(questions, corrected, head)]
    official.publish(predictions, release, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "export"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--hc5-steps", type=int, default=2500)
    parser.add_argument("--job", default=os.environ.get("SLURM_JOB_ID"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(args.out / "full_pool.lock", timeout=0):
        if args.phase == "train":
            if args.source is None or min(args.steps, args.hc5_steps) <= 0:
                parser.error("train requires --source and positive step counts")
            train(args.source, args.out, args.steps, args.hc5_steps)
        else:
            if not args.job:
                parser.error("export requires --job outside Slurm")
            export(args.out, args.job)


if __name__ == "__main__":
    main()
