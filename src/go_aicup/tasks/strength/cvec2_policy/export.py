"""Release CSVs for the stage-1 arms, fitted exactly like 751959-cvec2 (seed 2027, Valid-S + train20 + Valid-U).

Arm A must reproduce 751959-cvec2.csv question by question before B or C are written.
Test normalizers come from the existing per-move caches: C from ``permove/c_test_<part>_hd<h>``,
H9 from the release HumanSL shards (komi 6.5) with HD1 games replaced by ``p3_komi/h_test_<part>``.

    python -m go_aicup.tasks.strength.cvec2_policy.export --stage1-out <dir> --job <id>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash, write_json
from go_aicup.data.board_cache import RANKS
from go_aicup.metrics import score_rank
from go_aicup.tasks.strength.submission_files import write_rank_csv
from scripts.strength import humansl_p3_komi as p3, rank_prod_a0_fusion as prod
from scripts.strength.rank_more_players import player_prior
from scripts.strength.rank_permove import OUT as C_CACHE

from . import played
from .run import LEGACY_ZIP, source_paths, stage1_features

TAGS = {"A": "cv2s1-a", "B": "cv2s1-b", "C": "cv2s1-c"}


def _place(offsets, errors, target, ids, norm, error):
    rows = target.get_indexer(ids)
    if (rows < 0).any():
        raise ValueError("per-move shard has games outside the test part")
    offsets[rows] = norm
    errors[rows] = error


def test_offsets(part, x, index):
    """H9 and C10 normalizers [game, own/opp] for one test part, aligned with ``index``."""
    target = pd.Index(index)
    hd1 = x[:, 50:52].sum(axis=1) > 0
    c_off = np.full((len(index), 2), np.nan)
    c_err = np.full(len(index), np.nan)
    for hd in (0, 1):
        for marker in sorted((C_CACHE / f"c_test_{part}_hd{hd}").glob("shard_*/meta.json")):
            shard = marker.parent
            meta = json.loads(marker.read_text())
            if (not meta["per_move"] or meta["finetuned"] is not None or meta["komi"] != (.5 if hd else 6.5)
                    or not meta["vectors"].endswith("/cvec_v2/learn/vectors.pt")):
                raise ValueError(f"unexpected C teacher identity: {shard}")
            table = pd.read_parquet(shard / "games.parquet")
            if not np.load(shard / "done.npy").all():
                raise ValueError(f"incomplete C shard {shard}")
            mean, norm = played.means_and_offsets(np.load(shard / "move_logp.npy").astype(np.float32),
                                                  np.load(shard / "move_slot.npy"), len(table))
            rows = target.get_indexer(table.game_index)
            if (rows < 0).any() or (hd1[rows] != bool(hd)).any():
                raise ValueError(f"C shard games/handicap mismatch: {shard}")
            _place(c_off, c_err, target, table.game_index, norm,
                   np.abs(mean - x[rows, 30:50]).max(axis=1))
    h_off = np.full((len(index), 2), np.nan)
    h_err = np.full(len(index), np.nan)
    release = prod.RELEASE / f"rank_{part}" / "humansl"
    for shard, handicap in ([(m.parent, 0) for m in sorted(release.glob("shard_*/meta.json"))]
                            + [(m.parent, 1) for m in sorted((p3.OUT / f"h_test_{part}").glob("shard_*/meta.json"))]):
        table = pd.read_parquet(shard / "games.parquet").assign(handicap=handicap)
        mean, norm = played.h_means(shard, table)
        keep = hd1[target.get_indexer(table.game_index)] == bool(handicap)
        rows = target.get_indexer(table.game_index[keep])
        _place(h_off, h_err, target, table.game_index[keep], norm[keep],
               np.abs(mean[keep] - x[rows][:, played.H_COLUMNS]).max(axis=1))
    for name, off, err, tol in (("C", c_off, c_err, played.C_MEAN_TOLERANCE),
                                ("H", h_off, h_err, played.H_MEAN_TOLERANCE)):
        if not np.isfinite(off).all():
            raise ValueError(f"{part}: missing {name} test normalizers")
        if not err.max() <= tol:
            raise ValueError(f"{part}: {name} per-move means differ from the release features: {err.max():.6g}")
    return h_off, c_off, {"h_max_error": float(h_err.max()), "c_max_error": float(c_err.max())}


def export(stage1_out, job, legacy_zip=LEGACY_ZIP):
    stage1_out = Path(stage1_out)
    complete = json.loads((stage1_out / "complete.json").read_text())
    for name, digest in complete["files"].items():
        if file_hash(stage1_out / name) != digest:
            raise ValueError(f"stage-1 file changed: {name}")
    games = pd.read_parquet(stage1_out / "games.parquet")
    with np.load(stage1_out / "features.npz") as data:
        train = {arm: data[arm] for arm in ("A", "B", "C")}
    stacked = (games.role.ne("U") | games.handicap.le(1)).to_numpy()
    prior = player_prior(games[games.role.eq("S") | (games.role.eq("U") & games.handicap.le(1))])
    with source_paths(legacy_zip) as (_, _, _, _, raw_dir, _):
        tests = {part: prod.test_data(part, raw_dir) for part in ("public", "private")}
    audit, test_arms = {}, {}
    for part, (x, index, _) in tests.items():
        h_off, c_off, audit[part] = test_offsets(part, x, index)
        test_arms[part] = stage1_features(x, played.H_COLUMNS, h_off, c_off)
    base = pd.read_csv(prod.BASE, dtype=str).set_index("question_id")["pred_rank"]
    report = {"stage1_out": str(stage1_out), "normalizer_audit": audit, "arms": {}}
    predictions = {}
    for arm in ("A", "B", "C"):
        model = prod._fit(games[stacked], train[arm][stacked], prior, 2027)
        rows = []
        for part, (_, index, questions) in tests.items():
            p = model.probabilities(test_arms[part][arm], index, questions)
            rows.append(pd.DataFrame({"question_id": questions["question_id"],
                                      "pred_rank": [RANKS[r] for r in score_rank(p)]}))
        ours = pd.concat(rows).set_index("question_id")["pred_rank"].loc[base.index]
        predictions[arm] = ours
        report["arms"][arm] = {"changed_vs_base": int((ours != base).sum()),
                               "pred_counts": ours.value_counts().reindex(RANKS, fill_value=0).tolist()}
        if arm == "A" and report["arms"]["A"]["changed_vs_base"]:
            write_json(stage1_out / "export.json", report)
            raise ValueError(f"arm A does not reproduce 751959-cvec2: {report['arms']['A']}")
    for arm in ("B", "C"):
        path = write_rank_csv(predictions[arm].reset_index(), TAGS[arm], str(job))
        report["arms"][arm] |= {"file": path.name, "sha256": file_hash(path)}
        report["arms"][arm]["changed_vs_b"] = int((predictions[arm] != predictions["B"]).sum())
    write_json(stage1_out / "export.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-out", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--legacy-zip", type=Path, default=LEGACY_ZIP)
    args = parser.parse_args()
    export(args.stage1_out, args.job, args.legacy_zip)


if __name__ == "__main__":
    main()
