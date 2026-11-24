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
