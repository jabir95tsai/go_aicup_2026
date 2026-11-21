"""Run historical controls or the locked full cvec2 policy experiment."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, write_json
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores
from scripts.strength import humansl_hd_filter as hf, rank_prod_a0_fusion as prod
from scripts.strength.rank_more_players import player_prior
from scripts.strength.rank_train20_hc import TRAIN20


BASE_SHA256 = "5471b8be92494c5a43762526bed269364b03328807f58179ab81b4f8092930a9"
LEGACY_ZIP = ROOT / "data/processed/strength_train_hc.zip"
RAW_ZIP = ROOT / "data/raw/raw.zip"
STAGE1_ARMS = ("A", "B", "C")


def materialize(path, archive, member, scratch):
    if path.is_file():
        return path
    target = scratch / member
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped, zipped.open(member) as source, target.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    return target


@contextmanager
def source_paths(legacy_zip):
    if not legacy_zip.is_file() and not all(p.is_file() for p in (
        hf.Q1000, TRAIN20, hf.STANDARD / "game_index.npy", hf.STANDARD / "features.npy",
    )):
        raise FileNotFoundError(f"restore the original strength_train_hc archive at {legacy_zip}")
    with tempfile.TemporaryDirectory(prefix="cvec2-policy-") as temporary:
        scratch = Path(temporary)
        def legacy(path, member):
            return materialize(path, legacy_zip, f"strength_train_hc/{member}", scratch)
        def raw(path, member):
            archive = RAW_ZIP if RAW_ZIP.is_file() else ROOT / "data/raw.zip"
            return materialize(path, archive, f"raw/{member}", scratch)
        standard = [legacy(hf.STANDARD / name, f"standard/{name}")
                    for name in ("game_index.npy", "features.npy")]
        # Keep the expected pair together even if only one unpacked member was available.
        if standard[0].parent != standard[1].parent:
            target = scratch / "strength_train_hc/standard"
            for path in standard:
                if path.parent != target:
                    (target / path.name).symlink_to(path)
            standard = [target / p.name for p in standard]
        train20 = legacy(TRAIN20, "legacy/humansl_train20/games.parquet")
        e1 = legacy(hf.Q1000, "legacy/humansl_q1000/questions.parquet")
        handicap = raw(hf.HANDICAP, "training/handicap.csv")
        test_paths = [raw(ROOT / "data/raw/tests" / name, f"tests/{name}") for name in (
            "rank_prediction_test_public.csv", "rank_prediction_test_private.csv",
        )]
        if test_paths[0].parent != test_paths[1].parent:
            target = scratch / "raw/tests"
            for path in test_paths:
                if path.parent != target:
                    (target / path.name).symlink_to(path)
            test_paths = [target / p.name for p in test_paths]
        files = [*standard, train20, e1, handicap, *test_paths]
        yield standard[0].parent, train20, e1, handicap, test_paths[0].parent, {
            str(i) + "_" + p.name: file_hash(p) for i, p in enumerate(files)
        }


def prepare(legacy_zip):
    if file_hash(prod.BASE) != BASE_SHA256:
        raise ValueError("751959-cvec2.csv differs from the registered reference")
    with source_paths(legacy_zip) as (standard, train20, e1, handicap, raw_dir, sources):
        fit, valid, x_fit, x_valid = prod.train_data(
            standard=standard, handicap_path=handicap, train20_table=train20,
        )
        sets = prod.question_sets(e1)
        if (len(fit), fit.player_id.nunique(), len(valid)) != (149615, 5728, 35083):
            raise ValueError("wrong cvec2 fit/evaluation pool")
        if set(fit.player_id) & set(valid.player_id):
            raise ValueError("fit and Valid-U players overlap")
        parity = prod.step1(fit, valid, x_fit, x_valid, sets, raw_dir=raw_dir)
    if not parity["pass"]:
        raise ValueError(f"cvec2 reproduction failed: {parity}")
    games = pd.concat([fit, valid], ignore_index=True)
    if not games.game_index.is_unique:
        raise ValueError("duplicate cvec2 game indices")
    return games, np.vstack((x_fit, x_valid)), sets, parity, sources


def _fit_scores(games, features, sets, prior, seed):
    fit = games.role.ne("U").to_numpy()
    model = prod._fit(games.loc[fit], features[fit], prior, seed)
    rows = {}
    for name, q in sets.items():
        p = model.probabilities(features[~fit], games.loc[~fit, "game_index"], q)
        rows[name] = official_scores(score_rank(p), q.rank_id.to_numpy())
    return rows


def evaluate(games, arms, sets, pairs, workers):
    prior = player_prior(games[games.role.eq("S")])
    tasks = [(arm, seed) for arm in arms for seed in prod.SEEDS]
    evaluated = Parallel(n_jobs=workers)(
        delayed(_fit_scores)(games, arms[arm], sets, prior, seed) for arm, seed in tasks
    )
    scores = dict(zip(tasks, evaluated, strict=True))
    per_seed, means, comparisons, frames = {}, {}, {}, []
    for name, q in sets.items():
        k = q.k.to_numpy()
        average = {arm: np.mean([scores[arm, seed][name] for seed in prod.SEEDS], axis=0) for arm in arms}
        means[name] = {arm: hf.by_k(value, k) for arm, value in average.items()}
        per_seed[name] = {str(seed): {arm: hf.by_k(scores[arm, seed][name], k) for arm in arms}
                          for seed in prod.SEEDS}
        comparisons[name] = {
            label: {
                "delta": hf.by_k(average[b] - average[a], k),
                "equal_k_delta": hf.equal_k(average[b] - average[a], k),
                "min_k_delta": min(float((average[b] - average[a])[k == bucket].mean()) for bucket in K_BUCKETS),
                "player_bootstrap_95": hf.bootstrap(average[a], average[b], q),
            } for label, (a, b) in pairs.items()
        }
        for seed in prod.SEEDS:
            frame = q[["question_id", "player_id", "rank_id", "k"]].assign(split=name, seed=seed)
            for arm in arms:
                frame[arm] = scores[arm, seed][name]
            frames.append(frame)
    return {"mean": means, "per_seed": per_seed, "comparisons": comparisons}, pd.concat(frames, ignore_index=True)


def stage1_features(base, h_columns, h_offset, c_offset):
    h = base[:, h_columns]
    c = (base[:, 30:50].reshape(-1, 2, 10) - c_offset[:, :, None]).reshape(-1, 20)
    h_normalized = (h.reshape(-1, 2, 9) - h_offset[:, :, None]).reshape(-1, 18)
    return {"A": base, "B": np.hstack((h, c, base[:, 50:])),
            "C": np.hstack((h_normalized, c, base[:, 50:]))}


def stage1(args):
    from . import played

    started = time.monotonic()
    games, base, sets, parity, sources = prepare(args.legacy_zip)
    print(json.dumps({"stage": "cvec2_parity", **parity}), flush=True)
    if args.check_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "complete.json").exists():
        raise FileExistsError(args.out / "complete.json")
    h_offset, c_offset = played.normalized_offsets(games, base, args.out)
    arms = stage1_features(base, played.H_COLUMNS, h_offset, c_offset)
    report, rows = evaluate(games, arms, sets, {"B-A": ("A", "B"), "C-A": ("A", "C"),
                                               "C-B": ("B", "C")}, args.workers)
    e1, e2 = (report["comparisons"][s]["C-B"] for s in ("E1", "E2"))
    selected = "C" if e2["equal_k_delta"] > 0 and e1["equal_k_delta"] >= 0 and e2["min_k_delta"] >= -.005 else "B"
    games.to_parquet(args.out / "games.parquet", index=False)
    for name, table in sets.items():
        table.to_parquet(args.out / f"{name}.parquet", index=False)
    np.savez(args.out / "features.npz", **arms)
    rows.to_parquet(args.out / "question_scores.parquet", index=False)
    write_json(args.out / "result.json", report | {"suggested_anchor": selected, "parity": parity,
               "sources": sources, "seconds": time.monotonic() - started})
    files = ["games.parquet", "E1.parquet", "E2.parquet", "features.npz", "result.json", "question_scores.parquet"]
    write_json(args.out / "complete.json", {"stage": 1, "files": {p: file_hash(args.out / p) for p in files}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="stage", required=True)
    first = tasks.add_parser("stage1")
    first.add_argument("--legacy-zip", type=Path, default=LEGACY_ZIP)
    first.add_argument("--check-only", action="store_true")
    second = tasks.add_parser("stage2")
    second.add_argument("--stage1-out", type=Path, required=True)
    second.add_argument("--anchor", choices=("B", "C"), required=True)
    for command in (first, second):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--workers", type=int, default=8)
    full = tasks.add_parser("all")
    full.add_argument("--phase", choices=("preflight", "prepare", "train", "score"), required=True)
    full.add_argument("--out", type=Path)
    full.add_argument("--stage1-out", type=Path,
                      default=ROOT / "outputs/rank/humansl/cvec2_policy/stage1/812279")
    full.add_argument("--folds", type=Path, default=PROCESSED / "cv/strength_folds_seed2027.parquet")
    full.add_argument("--planes", type=Path,
                      default=PROCESSED / "strength_features/humansl/cvec2_policy/input_planes")
    full.add_argument("--reuse-planes", type=Path, help="reuse a verified input-plane directory during prepare")
    full.add_argument("--workers", type=int, default=88)
    full.add_argument("--smoke-steps", type=int, choices=range(2, 6), metavar="2..5")
    args = parser.parse_args()
    if args.stage == "stage1":
        stage1(args)
    elif args.stage == "stage2":
        from .train import stage2
        stage2(args)
    else:
        from .all_run import execute
        execute(args)


if __name__ == "__main__":
    main()
