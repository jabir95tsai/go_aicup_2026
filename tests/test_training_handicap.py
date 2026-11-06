"""Reject source drift before installing the official HD sidecar."""
import csv
import importlib.util
import io

import pytest

from go_aicup.paths import ROOT

spec = importlib.util.spec_from_file_location('extract_hd', ROOT / 'scripts/extract_training_handicap.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def test_handicap_sidecar_alignment_and_source_drift(tmp_path):
    old, new = tmp_path / 'old.csv', tmp_path / 'new.csv'
    row = ['p1', 'g1', 'A', 'W', '(;SZ[19];B[pd];W[dp])']
    for path, content in [(old, row), (new, row[:-1] + [row[-1].replace('SZ[19]', 'SZ[19]HD[1]')])]:
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(extract.FIELDS)
            writer.writerow(content)
    output = io.StringIO()
    result = extract.validate_rank(new, old, 'A', csv.writer(output))
    assert result['handicap_counts'] == {1: 1}
    assert list(csv.reader(io.StringIO(output.getvalue()))) == [['A', 'g1', 'p1', 'W', '0', '1']]
    assert extract.validate_rank(new, new, 'A', csv.writer(io.StringIO())) == result
    new.write_text(new.read_text().replace('W[dp]', 'W[dd]'))
    with pytest.raises(ValueError, match='changes beyond HD'):
        extract.validate_rank(new, old, 'A', csv.writer(io.StringIO()))
