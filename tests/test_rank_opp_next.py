import numpy as np
import pandas as pd

from go_aicup.metrics import score_rank
from scripts.strength import rank_opp_next as next_run
from scripts.strength import rank_opp_refine as refine


def test_delta_smooths_only_neighbour_bins_of_the_same_hd_and_colour():
    rows = pd.DataFrame({"hd": [1] * 20, "oc": [0] * 20, "bin": [3] * 20,
                         "rj": [6] * 20, "owner_rank": [5] * 20})
    base = refine._delta_prob(rows)
    smooth = refine._delta_prob(rows, smooth=True)
    np.testing.assert_allclose(smooth.sum(axis=-1), 1)
    assert smooth[1, 0, 5, 4, 6] > base[1, 0, 5, 4, 6]
    np.testing.assert_array_equal(smooth[0], base[0])
    np.testing.assert_array_equal(smooth[1, 1], base[1, 1])
    np.testing.assert_array_equal(smooth[1, 0, :, :2], base[1, 0, :, :2])


def test_question_confidence_uses_the_official_score_margin_and_broadcasts_weight():
    p = np.zeros((2, 10))
    p[0, [4, 5]] = 0.5
    p[1, 4] = 1
    weights = next_run.question_alpha(p, 0.05, 0.75, 0.35)
    np.testing.assert_array_equal(weights, [0.75, 0.35])
    ll = np.zeros((2, 10))
    ll[:, 5] = 1.0
    for i, weight in enumerate(weights):
        expected = score_rank(next_run.fu.posterior(p[i:i + 1], ll[i:i + 1], weight))
        np.testing.assert_array_equal(next_run.predict(p, ll, lambda _: weights)[i:i + 1], expected)


def test_colour_mu_uses_one_value_for_each_query_colour():
    q = pd.DataFrame({"owner": np.repeat(np.arange(4), 2), "qcolor": np.tile([0, 1], 4)})
    cand = np.tile(np.arange(10), (len(q), 1))
    val = np.tile(np.arange(10, 0, -1, dtype=float), (len(q), 1))
    seen = cand.copy()
    seen[:, 0], seen[:, 1] = cand[:, 1], cand[:, 0]
    seen_llr = np.zeros_like(val)
    seen_llr[::2, 0] = 4.0
    seen_llr[::2, 1] = 3.0
    seen_llr[::2, 2] = 4.8
    seen_llr[1::2, 0] = 0.6
    z = {"f8_idx": cand, "f8_val": val, "seen_f8_idx": seen, "seen_f8_val": val,
         "owner_col": np.ones(len(q), dtype=int)}
    cache = {"masked_rows": q.index.to_numpy(), "seen_rows": q.index.to_numpy(),
             "masked_llr": np.zeros_like(val), "seen_llr": seen_llr}
    baseline, adjusted, result = next_run.reranks(q, z, "f8", cache, colourwise=True)
    assert result["mu"] == {"0": 1.0, "1": 2.0}
    np.testing.assert_array_equal(adjusted["seen"][0][1::2], baseline["seen"][0][1::2])
    assert np.all(adjusted["seen"][0][::2, 0] == 1)


def test_rich_gpu_builds_multishard_features_and_fits_eb_prior(tmp_path):
    import json

    import pytest
    import torch
    from scripts.identity.pid2_likelihood import DIM_RICH, OPTIONS, fit

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    device = torch.device("cuda")
    moves = np.array([0, 20, 40, 60], dtype=np.int16)
    colors = np.arange(len(moves), dtype=np.int8) % 2
    games = {g: (moves, colors) for g in (42, 84)}
    shards = []
    for g in games:
        shard = tmp_path / str(g)
        shard.mkdir()
        top = np.tile(np.arange(32, dtype=np.int16), (2, 1))
        logp = np.full((2, 4, OPTIONS), -np.log(OPTIONS), dtype=np.float16)
        arr = {"game": np.full(2, g), "ply": np.array([2, 0]), "top": top, "lp": logp, "slot": np.array([32, 0])}
        for name, value in arr.items():
            np.save(shard / f"{name}.npy", value)
        (shard / "meta.json").write_text(json.dumps({"positions": 2}))
        shards.append(shard)
    b, phi, slot, span = next_run.llr.rich_tensors(shards, pd.DataFrame({"id": list(games)}), games, device)
    assert phi.shape == (4, OPTIONS, DIM_RICH)
    assert span == {42: (0, 2), 84: (2, 4)}
    assert slot.tolist() == [0, 32, 0, 32]
    assert torch.isfinite(phi).all()
    prior = torch.zeros(DIM_RICH, dtype=torch.float64, device=device)
    pos = torch.arange(4, device=device)
    owner = torch.tensor([0, 0, 1, 1], device=device)
    theta = fit(phi, b, slot, pos, owner, 2, prior, 3.0, iters=3, precision=torch.eye(
        DIM_RICH, dtype=torch.float64, device=device))
    assert torch.isfinite(theta).all() and theta[0].norm() > 0
    torch.testing.assert_close(theta[0], theta[1])


def test_soft_delta_counts_only_change_lowest_gap_bucket():
    rows = pd.DataFrame({"hd": [0] * 4, "oc": [1] * 4, "bin": [0, 0, 1, 1],
                         "rj": [5] * 4, "owner_rank": [5] * 4})
    top = {"idx": np.tile([0, 1], (4, 1)), "val": np.zeros((4, 2)), "cand_rank": np.array([5, 6])}
    original = refine._delta_prob(rows)
    updated = refine._delta_prob(rows, soft_top=top)
    np.testing.assert_array_equal(updated[:, :, :, 1:, :], original[:, :, :, 1:, :])
    np.testing.assert_array_equal(updated[:, 0], original[:, 0])
    np.testing.assert_allclose(updated.sum(axis=-1), 1)
    assert updated[0, 1, 5, 0, 6] > original[0, 1, 5, 0, 6]
    assert updated[0, 1, 5, 0, 5] < original[0, 1, 5, 0, 5]


def test_length_calibration_bounds_and_preserves_cache_alignment():
    from scripts.strength import rank_opp_causes as causes

    cache = {"masked_rows": np.array([5, 7, 9]), "seen_rows": np.array([9, 5]),
             "masked_llr": np.ones((3, 2)), "seen_llr": np.ones((2, 2))}
    changed = causes.scale_llr(cache, pd.Series([50, 115, 208], index=[5, 7, 9]))
    np.testing.assert_allclose(changed["masked_llr"][:, 0], [1.2, 1.0, 0.8])
    np.testing.assert_allclose(changed["seen_llr"][:, 0], [0.8, 1.2])
    np.testing.assert_array_equal(cache["masked_llr"], np.ones((3, 2)))


def test_short_delta_updates_only_short_games_in_observed_strata():
    from scripts.strength import rank_opp_causes as causes

    rows = pd.DataFrame({"hd": [0] * 4, "oc": [1, 1, 1, 0], "bin": [0] * 4,
                         "rj": [7, 5, 5, 5], "owner_rank": [5] * 4})
    top = {"idx": np.array([[1], [0], [0], [0]]), "val": np.zeros((4, 1)), "cand_rank": np.array([5, 7])}
    plies = pd.Series([93, 131, 132, 90])
    fit = rows.iloc[:2]
    original = refine.soft_ll(refine._delta_prob(fit), rows, top, 3.5)
    modified = causes.length_evidence(fit, top, plies=plies, arm="short")(rows)
    assert not np.array_equal(original[0], modified[0])
    np.testing.assert_array_equal(original[1:], modified[1:])
    dampened = causes.length_evidence(fit, top, plies=plies, arm="dampen")(rows)
    np.testing.assert_allclose(dampened[[0, 3]], causes.SHORT_SCALE * original[[0, 3]])
    np.testing.assert_array_equal(dampened[1:3], original[1:3])
    boosted = causes.length_evidence(fit, top, plies=plies, arm="long")(rows)
    np.testing.assert_allclose(boosted[2], original[2] / causes.SHORT_SCALE)
    np.testing.assert_array_equal(boosted[[0, 1, 3]], original[[0, 1, 3]])


def test_length_export_aligns_opponent_plies_by_part_and_store_row(tmp_path, monkeypatch):
    from functools import partial

    from scripts.strength import rank_opp_causes as causes

    rows = pd.DataFrame({"set": ["s_opp", "s_opp", "test", "test"], "part": ["train", "train", "rank_public", "rank_private"],
                         "store_row": [-1, -1, 0, 0], "hd": [0] * 4, "oc": [1] * 4, "bin": [0] * 4,
                         "rj": [7, 5, 7, 5], "owner_rank": [5] * 4})
    top = {"idx": np.array([[1], [0], [1], [0]]), "val": np.zeros((4, 1)), "cand_rank": np.array([5, 7])}
    plies = pd.Series([93, 131, 132, 90])
    params = partial(causes.length_evidence, plies=plies, arm="long")
    expected = params(rows.iloc[:2], top)(rows.iloc[2:])
    base = tmp_path / 'opp7.csv'
    predictions = pd.DataFrame({"question_id": [f"pub_{i}" for i in range(800)], "pred_rank": "A"})
    predictions.to_csv(base, index=False)
    monkeypatch.setattr(next_run, "BASE", base)

    def export(_rows, _cell, _job, *, evidence, **_):
        sub = rows.iloc[[3, 2]].assign(key=[10, 20])
        scores = evidence(pd.Series([[20], [10], [10, 20]]), sub, None, "key")
        np.testing.assert_allclose(scores[:2], expected)
        np.testing.assert_allclose(scores[2], expected.sum(axis=0))
        return {"file": base.name}

    monkeypatch.setattr(next_run.fu, "export", export)
    output = next_run.exports("test", "long", rows, top, False, None, tag="olong10", game_evidence=params)
    assert output["changed_vs_opp7"] == {"public": 0, "private": 0}


def test_consensus_requires_distinct_opponents_and_respects_b2_regret():
    from scripts.strength import rank_opp_experiments as exp

    rows = pd.DataFrame({"bin": [1, 1, 1, 0]})
    top = {"idx": np.array([[0], [0], [1], [2]])}
    ll = np.zeros((4, 10))
    ll[:3, 5] = 3
    ll[3, 4] = 9
    per_game = pd.DataFrame(ll, index=[10, 11, 12, 13])
    evidence = exp.group_evidence(pd.Series([[10, 11, 13], [10, 11, 12, 13]]), per_game, rows, top)
    np.testing.assert_array_equal(evidence[:, 21], [0, 2])
    assert evidence[1, 15] == 6
    np.testing.assert_array_equal(evidence[:, :10], [ll[[0, 1, 3]].sum(axis=0), ll.sum(axis=0)])
    p = np.zeros((2, 10))
    p[:, [4, 5]] = [0.52, 0.48]
    np.testing.assert_array_equal(exp.consensus_predict(p, evidence, tolerance=0.01), [4, 4])
    np.testing.assert_array_equal(exp.consensus_predict(p, evidence, tolerance=0.05), [4, 5])


def test_own_ablation_keeps_no_opponent_feature_columns(monkeypatch):
    from scripts.strength import rank_opp_experiments as exp

    def own(_fit, x, y, *, own_only):
        assert own_only
        return x[:, :1], y[:, :1], None

    monkeypatch.setattr(exp.rank_ogs, "yugen_oof", own)
    features = np.arange(104).reshape(2, 52)
    fit, evaluation = exp.own_features(None, features, features)
    expected = np.r_[0:15, 30:40, 50:52, 0]
    np.testing.assert_array_equal(fit, features[:, expected])
    np.testing.assert_array_equal(evaluation, fit)


def test_cross_game_pairs_require_shared_opening_and_exclude_duplicate_games():
    import torch
    from scripts.strength import rank_opp_cross_game as cross

    x = np.array([[1, 0], [.8, .6], [.6, .8], [.8, .6], [.8, .6]], dtype=np.float32)
    result = cross.mutual_pairs(x, [8, 8, 8, 9, -1], [1, 2, 1, 4, 5], torch.device("cpu"))
    assert result[['left', 'right']].to_numpy().tolist() == [[1, 2]]
    pairs = pd.DataFrame({"left": [0, 2], "right": [1, 3], "similarity": [.8, .9]})
    idx = np.array([[4, 6], [5, 4], [7, 8], [5, 4]])
    val = np.array([[8., 0.], [3., 2.], [8., 0.], [3., 2.]])
    accepted = cross.proposals(pairs, (idx, val), np.zeros(4, dtype=int), {0: [0, 2], 1: [0, 2]})
    assert accepted.target.to_list() == [1]
    updated, _ = cross.boost((idx, val), accepted)
    np.testing.assert_array_equal(updated[:, 0], [4, 4, 7, 5])
    np.testing.assert_array_equal(idx[:, 0], [4, 5, 7, 5])
