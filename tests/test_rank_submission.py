"""Official rank CSV boundary: sample ordering, IDs and full AH+C coverage."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from go_aicup.data.board_cache import RANKS

spec = importlib.util.spec_from_file_location("export_rank_submission", Path(__file__).parents[1] / "scripts/export_rank_submission.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def release(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"question_id": ["pub_0002", "pub_0001", "priv_0001"],
                  "pred_rank": ["1D"] * 3}).to_csv(sample, index=False)
    for part, ids in (("public", ["pub_0001", "pub_0002"]), ("private", ["priv_0001"])):
        directory = tmp_path / f"rank_{part}"
        directory.mkdir()
        table = pd.DataFrame({"question_id": ids, "rank": ["D"] * len(ids), "num_games": 5,
                              "humansl_games": 5, "fallback_v2": False, "fallback_other": False,
                              **{f"p_{r}": 0.1 for r in RANKS}})
        table.to_parquet(directory / "rankvec_predictions.parquet", index=False)
    return tmp_path, sample, tmp_path / "submission.csv"


def test_export_follows_sample_order(release):
    work, sample, output = release
    result = module.export(work, sample, output)
    assert output.read_bytes().splitlines()[0] == b"question_id,pred_rank"
    assert result.question_id.tolist() == ["pub_0002", "pub_0001", "priv_0001"]
    assert pd.read_csv(output).pred_rank.tolist() == ["D"] * 3


@pytest.mark.parametrize("column,value", [("question_id", "pub_0001"), ("rank", "9D"),
                                          ("fallback_v2", True), ("humansl_games", 4), ("p_D", float("nan"))])
def test_export_rejects_incomplete_or_invalid_predictions(release, column, value):
    work, sample, output = release
    path = work / "rank_private/rankvec_predictions.parquet"
    table = pd.read_parquet(path)
    table.loc[0, column] = value
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        module.export(work, sample, output)
    assert not output.exists()
