import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax, logsumexp

from go_aicup.tasks.strength.cvec2_policy import all_data


def test_prepare_smoke_keeps_complete_batches_and_fixed_e2_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(all_data, "QUESTIONS_PER_K", 16)
    fit_ids = np.arange(100, 131)
    groups = {
        "fit": pd.DataFrame({"game_index": fit_ids, "rank_id": fit_ids % 10}),
        "e2": pd.DataFrame({"game_index": np.arange(200, 300)}, index=np.arange(31, 131)),
        "reserve": pd.DataFrame({"game_index": [301]}, index=[131]),
    }
    questions = pd.DataFrame(
        [{"k": k, "games": np.arange(start, start + k)}
         for k, start in zip(all_data.K_BUCKETS, (200, 210, 230, 260)) for _ in range(2)]
    )
    scores = pd.DataFrame({"A": np.arange(len(questions))})
    labels = np.tile(np.arange(16) % 10, 4)
    k = np.repeat(all_data.K_BUCKETS, 16)
    batches = np.asarray(all_data.epoch_batches(pd.DataFrame({"k": k}), 2027, 1, 8))
    draws = {
        size: (fit_ids[(np.arange(16)[:, None] + np.arange(size)) % len(fit_ids)], labels[16*i:16*(i+1)])
        for i, size in enumerate(all_data.K_BUCKETS)
    }
    identity = {"seed": 2027, "draw_sha256": all_data.DRAW_SHA256}
    source = all_data.Data(
        groups, questions, scores, {"A": np.zeros((132, 52), np.float32)},
        draws, batches, labels, k, {size: np.full(10, .1) for size in all_data.K_BUCKETS},
        {"A": {name: np.zeros(1, np.float32) for name in all_data.HEAD_FIELDS}},
        identity, [], np.arange(131),
    )
    monkeypatch.setattr(all_data, "preflight", lambda *_: source)
    monkeypatch.setattr(all_data, "_source_identity", lambda *_: {"seed": 2027})
    data = all_data.prepare(tmp_path, smoke_steps=1)

    assert data.scores.A.tolist() == [0, 2, 4, 6]
    assert data.groups["reserve"].empty
    assert data.batches.shape == (1, 32)
    assert np.array_equal(data.labels[data.batches[0]], labels[batches[0]])
    for local, original in zip(data.batches[0], batches[0], strict=True):
        expected = draws[k[original]][0][original % 16]
        assert np.array_equal(data.games.game_index.iloc[data.indices[local]], expected)
    assert np.array_equal(data.nominal_lengths, source.nominal_lengths)
    assert all(np.count_nonzero(counts) == 10 for counts in data.identity["normalization_fit_ranks"].values())
    assert np.load(tmp_path / "baseline_fit_40.npy").shape == (len(fit_ids), 40)

    rng = np.random.default_rng(2027)
    h, c = (rng.normal(size=(len(data.games), 2, 3, channels)).astype(np.float32)
            for channels in (15, 10))
    offsets = {name: logsumexp(q, axis=-1).mean(axis=-1) for name, q in (("H", h), ("C", c))}
    raw = data.features["A"]
    raw[:, :30], raw[:, 30:50] = (q.mean(axis=2).reshape(len(raw), -1) for q in (h, c))
    raw[:, -2:] = rng.integers(0, 2, size=(len(raw), 2))
    controls = all_data.fit_normalized(data, offsets)
    c_only, both = (controls[arm][0] for arm in all_data.NORMALIZED_ARMS)
    expected_h, expected_c = (log_softmax(q, axis=-1).mean(axis=2).reshape(len(raw), -1) for q in (h, c))
    np.testing.assert_allclose(both[:, :30], expected_h, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(both[:, 30:50], expected_c, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(both[:, 30:], c_only[:, 30:])
    np.testing.assert_array_equal(c_only[:, :30], raw[:, :30])
    np.testing.assert_array_equal(c_only[:, -2:], raw[:, -2:])
    for _, head in controls.values():
        assert head["weights"].shape == (4, 10, 52)
    offsets["H"][0, 0] = np.nan
    with pytest.raises(ValueError, match="teacher offsets"):
        all_data.fit_normalized(data, offsets)

    reloaded = all_data.prepare(tmp_path, smoke_steps=1)
    assert np.array_equal(reloaded.batches, data.batches)
    with pytest.raises(ValueError, match="configuration changed"):
        all_data.prepare(tmp_path, smoke_steps=2)
    with (tmp_path / "initial_b2.npz").open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="checksums failed"):
        all_data.load(tmp_path)
