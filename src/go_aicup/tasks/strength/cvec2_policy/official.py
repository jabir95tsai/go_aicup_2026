"""Score frozen cvec2 policy arms on the official Public and Private questions."""

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.submission_files import RANK_DIR, write_rank_csv
from scripts.strength import rank_identity_merge, rank_prod_a0_fusion

from . import all_report, full_cache, joint, planes
from .all_run import teacher_parity
from .run import materialize


ARMS = ("A/R10-D", "A/CNN-R10-D")
TAGS = {"A": "cva-l10", ARMS[0]: "r10d-l10", ARMS[1]: "c10d-l10", "H5-R10-D": "h5d-l10"}
REFERENCES = (
    "839620-chc-l10.csv", "839232-ygoe-l10.csv", "839232-ygo-l10.csv",
    "839232-yge-l10.csv", "839169-ygbm-l10.csv", "826812-qm-l10.csv", "751959-cv2-l10.csv",
)
PARTS = ("public", "private")


def inputs(directory, part):
    manifest = planes._verified_manifest(directory)
    path = directory / "questions.parquet"
    if file_hash(path) != manifest["tables"][path.name]:
        raise ValueError("official question table checksum differs")
    q = pd.read_parquet(path)
    q = q.loc[q.part.eq(part)].reset_index(drop=True)
    games = pd.read_parquet(directory / f"{part}.parquet")
    planes.PlaneReader(directory, games, groups=(part,))
    with TemporaryDirectory() as temporary:
        name = f"rank_prediction_test_{part}.csv"
        raw = materialize(ROOT / "data/raw/tests" / name, planes.RAW_ZIP, f"raw/tests/{name}", Path(temporary))
        x, index, original = rank_prod_a0_fusion.test_data(part, raw.parent)
    original = original.set_index("question_id").loc[q.question_id]
    if len(q) != 400 or not q.question_id.is_unique or not np.array_equal(original.num_games, q.k):
        raise ValueError("official question identities or k differ")
    if any(len(new) != size or len(old) != size for new, old, size in zip(q.games, original.games, q.k, strict=True)):
        raise ValueError("official question has incomplete games")
    by_game = np.concatenate(q.games.to_numpy())
    old_game = np.concatenate(original.games.to_numpy())
    rows = pd.Index(index).get_indexer(old_game)
    if (rows < 0).any() or not np.array_equal(by_game, games.game_index) or len(np.unique(rows)) != len(x):
        raise ValueError("official game order differs from release H/C coverage")
    work = rank_prod_a0_fusion.RELEASE / f"rank_{part}"
    indexed = pd.read_parquet(work / "cache/games.parquet").set_index("game_index").loc[old_game]
    slots = pd.read_parquet(work / "store_meta.parquet").iloc[indexed.store_row.to_numpy()]
    if (not np.array_equal(indexed.length, games.length)
            or not np.array_equal(slots.question_id, np.repeat(q.question_id.to_numpy(), q.k))
            or not np.array_equal(slots.slot, np.concatenate([np.arange(1, k + 1) for k in q.k]))):
        raise ValueError("official game lengths or question slots differ")
    base = x[rows].astype(np.float32)
    hd = games.handicap.to_numpy() == 1
    color = games.player_color.to_numpy()
    np.testing.assert_array_equal(base[:, -2:], np.column_stack((hd & (color == 0), hd & (color == 1))))
    return games, q, base


def predict(questions, features, head):
    k = questions.k.to_numpy()
    positions = {bucket: np.stack(questions.loc[k == bucket, "games"]) for bucket in K_BUCKETS}
    p = all_report.head_probabilities(head, features, positions, k)
    return questions[["question_id", "part", "k"]].assign(
        probabilities=p.tolist(), pred_rank=np.asarray(RANKS)[score_rank(p)])


def dataset(packed):
    public, private = [inputs(packed, part) for part in PARTS]
    canonical = pd.concat([public[0], private[0]], ignore_index=True)
    reader = planes.PlaneReader(packed, canonical, groups=PARTS)
    games = canonical.assign(game_index=np.arange(len(canonical)))
    second = private[1].assign(games=[ids + len(public[0]) for ids in private[1].games])
    questions = pd.concat([public[1], second], ignore_index=True)
    base = np.concatenate((public[2], private[2]))
    return games, questions, base, reader


def extract(cache, base, out, model_root, identity):
    for i, arm in enumerate(ARMS):
        dest = out / arm
        dest.mkdir(parents=True, exist_ok=True)
        link = dest / "model.pt"
        source = (model_root / arm / "model.pt").resolve()
        if not link.exists():
            link.symlink_to(source)
        if link.resolve() != source:
            raise ValueError("official extraction checkpoint differs")
        joint.extract_one(arm, cache, base, f"cuda:{4 + i}", out, identity, len(base))


def changes(pred, reference):
    changed = pred.pred_rank.ne(reference.loc[pred.question_id].to_numpy())
    return {part: {"total": int((changed & pred.part.eq(part)).sum()),
                   "by_k": {str(k): int((changed & pred.part.eq(part) & pred.k.eq(k)).sum()) for k in K_BUCKETS}}
            for part in PARTS}


def publish(predictions, out, job):
    l10 = pd.read_csv(rank_identity_merge.L10)
    l10 = l10.loc[l10.above_null_max].set_index("question_id").top1_rank
    if not l10.index.is_unique or not l10.isin(RANKS).all():
        raise ValueError("invalid L10 overrides")
    template = pd.read_csv(RANK_DIR / "751959-cv2-l10.csv", dtype=str).set_index("question_id")
    references = {name: pd.read_csv(RANK_DIR / name, dtype=str).set_index("question_id").pred_rank
                  for name in REFERENCES if (RANK_DIR / name).is_file()}
    report, final = {}, {}
    for arm, parts in predictions.items():
        pred = pd.concat(parts, ignore_index=True)
        if (len(pred) != 800 or not pred.question_id.is_unique or set(pred.question_id) != set(template.index)
                or not set(l10.index) <= set(pred.question_id)):
            raise ValueError("invalid official question or L10 coverage")
        pred = pred.set_index("question_id").loc[template.index].reset_index()
        raw = pred.set_index("question_id").pred_rank.copy()
        replacements = pred.question_id.map(l10)
        pred["pred_rank"] = replacements.fillna(pred.pred_rank)
        tag = TAGS[arm]
        all_report.write_table(out / f"{tag}.parquet", pred.assign(pred_before_l10=raw.to_numpy()))
        csv = write_rank_csv(pred[["question_id", "pred_rank"]], tag, job)
        report[arm] = {"file": csv.name, "sha256": file_hash(csv), "l10": changes(pred, raw),
                       "comparisons": {name: changes(pred, ref) for name, ref in references.items()}}
        final[arm] = pred
    if len(final) > 1:
        first = next(iter(final))
        ref = final[first].set_index("question_id").pred_rank
        for arm, pred in final.items():
            if arm != first:
                report[arm]["comparisons"][report[first]["file"]] = changes(pred, ref)
    write_json(out / "submissions.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def current(run, refit, out, job):
    config = json.loads((run / "config.json").read_text())
    packed = Path(config["planes"])
    gpu = json.loads((run / "gpu_complete.json").read_text())
    arms = json.loads((refit / "result.json").read_text())["arms"]
    if config["teacher_sources"] != {name: file_hash(ROOT / name) for name in config["teacher_sources"]}:
        raise ValueError("official teacher sources differ from the frozen encoders")
    all_report.verify_files(packed, config["plane_files"])
    for arm in ARMS:
        all_report.verify_files(run / arm, arms[arm]["source_features"])
        relative = f"{arm}/model.pt"
        all_report.verify_files(run, {relative: gpu["files"][relative]})
    out.mkdir(parents=True, exist_ok=True)
    games, questions, base, reader = dataset(packed)
    if not all((out / arm / "features.json").is_file() for arm in ARMS):
        cache = full_cache.infer(games, out, packed, config["teacher_config"], reader=reader)
        teacher_parity(cache, base[:, :50], out / "teacher_parity.json")
        extract(cache, base, out, run, gpu["identity"])
    predictions = {}
    for arm in ARMS:
        work_arm = out / arm
        record = json.loads((work_arm / "features.json").read_text())
        if (record["identity"] != gpu["identity"]
                or record["model_sha256"] != gpu["files"][f"{arm}/model.pt"]):
            raise ValueError("cached official features have different model identity")
        all_report.verify_files(work_arm, record["files"])
        with np.load(refit / arm.split("/")[-1] / "head.npz", allow_pickle=False) as head:
            predictions[arm] = [predict(questions, np.load(work_arm / "fit.npy"), head)]
    write_json(out / "sources.json", {"model_manifest": file_hash(run / "gpu_complete.json"),
                                      "refit_report": file_hash(refit / "result.json"),
                                      "heads": {arm: file_hash(refit / arm.split("/")[-1] / "head.npz") for arm in ARMS},
                                      "official_planes": object_hash(config["plane_files"])})
    return publish(predictions, out, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--refit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--job", default=os.environ.get("SLURM_JOB_ID"))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        packed = Path(json.loads((args.run / "config.json").read_text())["planes"])
        for part in PARTS:
            games, questions, _ = inputs(packed, part)
            print(part, len(games), len(questions), int(games.length.sum()))
        return
    if not args.job:
        parser.error("--job is required outside Slurm")
    current(args.run, args.refit, args.out, args.job)


if __name__ == "__main__":
    main()
