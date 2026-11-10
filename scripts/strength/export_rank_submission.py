"""Export Public/Private AH+C predictions in official sample order."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import RANKS


def export(work_dir: Path, sample: Path, output: Path,
           predictions: str = "rankvec_predictions.parquet") -> pd.DataFrame:
    template = pd.read_csv(sample, dtype=str)
    if list(template) != ["question_id", "pred_rank"] or not template.question_id.is_unique:
        raise ValueError("sample must contain unique question_id and pred_rank columns")
    parts = []
    for part in ("public", "private"):
        table = pd.read_parquet(work_dir / f"rank_{part}" / predictions)
        if (table.fallback_v2.any() or table.fallback_other.any()
                or (table.humansl_games != table.num_games).any()):
            raise ValueError(f"{part}: incomplete AH+C features or fallback predictions")
        proba = table[[f"p_{rank}" for rank in RANKS]].to_numpy()
        if (not np.isfinite(proba).all() or (proba < 0).any()
                or not np.allclose(proba.sum(axis=1), 1, atol=1e-6)):
            raise ValueError(f"{part}: invalid probabilities")
        parts.append(table[["question_id", "rank"]].rename(columns={"rank": "pred_rank"}))
    result = pd.concat(parts, ignore_index=True)
    if (not result.question_id.is_unique
            or set(result.question_id) != set(template.question_id)
            or not result.pred_rank.isin(RANKS).all()):
        raise ValueError("predictions must match all sample IDs exactly and contain legal ranks")
    result = result.set_index("question_id").loc[template.question_id].reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    result.to_csv(temporary, index=False, lineterminator="\r\n")
    pd.testing.assert_frame_equal(pd.read_csv(temporary, dtype=str), result)
    temporary.replace(output)
    print(f"submission: {output} ({len(result)} rows; sample order; complete AH+C; no fallback)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path, help="contains rank_public/ and rank_private/")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions", default="rankvec_predictions.parquet",
                        help="AH+C predictions file inside each rank_<part>/ (e.g. an LB-probe model's)")
    args = parser.parse_args()
    export(args.work_dir, args.sample, args.output, args.predictions)


if __name__ == "__main__":
    main()
