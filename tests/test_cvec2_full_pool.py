from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.cvec2_policy import encoder, full_pool, hc5


@pytest.mark.parametrize(("length", "starts"), [(100, (50, 85)), (229, (115, 195)), (445, (223, 379))])
def test_fixed_relative_windows(length, starts):
    side = np.arange(length) % 2
    source = (np.zeros((length, 993), dtype=np.uint8), np.zeros((length, 19), dtype=np.float32),
              np.arange(length) % 362, side)
    reader = SimpleNamespace(batches=lambda *args: [([0], [source])])
    games = pd.DataFrame({"length": [length], "player_color": [1]})
    moves = hc5.Moves(reader, games, sample=True)
    expected = np.concatenate([np.arange(start, start + 16) for start in starts])
    np.testing.assert_array_equal(moves.positions + 1, expected)
    np.testing.assert_array_equal(moves.counts, [[16, 16]])


def test_fixed_relative_window_rejects_short_board():
    with pytest.raises(ValueError, match="windows"):
        hc5.sampled_moves(99)


def test_policy_shift_preserves_mass_and_support():
    p = torch.arange(1, 363).float().view(1, 1, -1).expand(2, 25, -1).clone()
    p[..., 7] = 0
    p[:, :15, 8] = 0
    p /= p.sum(dim=-1, keepdim=True)
    raw = p.log()
    target = torch.tensor([33, 361])
    q = raw[torch.arange(2), :, target]
    zero = torch.zeros_like(raw)
    logp, got, played = hc5.policy_shift(raw, q, target, zero)
    torch.testing.assert_close(logp, raw, rtol=0, atol=0)
    torch.testing.assert_close(got, q, rtol=0, atol=0)
    assert not played.any()

    delta = torch.randn_like(zero, requires_grad=True)
    logp, got, played = hc5.policy_shift(raw, q, target, delta)
    torch.testing.assert_close(logp.exp().sum(dim=-1), torch.ones(2, 25))
    assert not logp.exp()[..., 7].any()
    assert not logp.exp()[:, :15, 8].any() and logp.exp()[:, 15:, 8].gt(0).all()
    torch.testing.assert_close(logp[torch.arange(2), :, target], got)
    played.sum().backward()
    assert delta.grad[:, :15].abs().sum() > 0
    assert delta.grad[:, 15:].abs().sum() > 0


def test_weighted_k_loss():
    k = np.repeat((5, 10, 15, 20), 8)
    x = torch.arange(32, dtype=torch.float32)
    logits = torch.stack([x / 10, torch.zeros_like(x)], dim=1)
    y = torch.ones(32, dtype=torch.long)
    value, ce = hc5.balanced_loss(logits, y, k)
    torch.testing.assert_close(value, (ce[0] + ce[1] + 2 * ce[2] + 2 * ce[3]) / 6)


def test_refit_initializes_and_fits_all_heads():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(10, 52)).astype(np.float32)
    labels = np.tile(np.arange(10), 3)
    draws = [(k, np.repeat(labels[:, None], k, axis=1), labels) for k in (5, 10, 15, 20)]
    model = full_pool.fit_head(features, draws)
    assert model["weights"].shape == (4, 10, 52)
    assert np.isfinite(model["weights"]).all() and (model["scales"] > 0).all()
    x = (features - model["means"][0]) / model["scales"][0]
    np.testing.assert_array_equal((x @ model["weights"][0].T + model["biases"][0]).argmax(-1), np.arange(10))


@pytest.mark.skipif(not hc5.CKPT.is_file(), reason="requires the HumanSL checkpoint config")
def test_reduced_trunk_has_same_primary_policy_forward():
    # Use the real HumanSL configuration and nonzero output weights so a missing path is observable.
    vectors = torch.randn(10, 2, 384)
    net = hc5.Policy(hc5.model_config(), vectors).eval()
    x = torch.zeros(1, 22, 19, 19)
    x[:, 0] = 1
    x[:, 1, 5, 5] = 1
    g = torch.randn(1, 19)
    c = torch.tensor([1])
    side = torch.tensor([0])
    with torch.no_grad():
        net.net.policy_head.conv2p.weight.normal_(std=.01)
        net.net.policy_head.linear_pass.weight.normal_(std=.01)
        result = net(x, g, c, side).flatten(0, 1)
        expected_h = net.net(x.repeat(15, 1, 1, 1), g.repeat(15, 1), input_meta=net.profiles[1])[0][0][:, 0]
        net.net.metadata_encoder.forward = lambda metadata, extra_outputs=None: metadata
        expected_c = net.net(x.repeat(10, 1, 1, 1), g.repeat(10, 1), input_meta=vectors[:, 0])[0][0][:, 0]
    assert len(net.net.blocks) == 5
    torch.testing.assert_close(result, torch.cat((expected_h, expected_c)))


@pytest.mark.skipif(torch.cuda.device_count() != 8 or not hc5.CKPT.is_file(),
                    reason="requires the HumanSL config and production shard layout")
def test_hc5_peer_training_and_extraction(tmp_path, monkeypatch):
    torch.manual_seed(3)
    vectors = torch.randn(10, 2, 384)
    monkeypatch.setattr(hc5, "c_vectors", lambda: vectors)
    n = 100
    games = pd.DataFrame({"length": [n] * 4, "player_color": [0, 1, 0, 1]})
    b = np.zeros((n, 22, 19, 19), dtype=np.uint8)
    b[:, 0] = 1
    bits = np.packbits(b.reshape(n, -1), axis=1)
    targets = np.tile([17, 361], n // 2)
    sample = (bits, np.zeros((n, 19), dtype=np.float32), targets, np.arange(n) % 2)
    reader = SimpleNamespace(batches=lambda *args: [(range(4), [sample] * 4)])
    tensors = []
    for family, width in ((0, 15), (4, 10)):
        shards = []
        for i in range(4):
            device = f"cuda:{family+i}"
            policy = torch.randn(n, width, 362, device=device).log_softmax(dim=-1)
            q = policy[torch.arange(n, device=device), :, torch.as_tensor(targets, device=device)]
            shards.append(SimpleNamespace(first=i, last=i+1, logp=policy.half(), q=q))
        tensors.append(SimpleNamespace(shards=shards, bounds=np.arange(5) * n))
    cache = SimpleNamespace(h=tensors[0], c=tensors[1])
    r10 = encoder.build("R10", True, 2027, raw_a=True).state_dict()
    head = {"means": np.zeros((4, 52)), "scales": np.ones((4, 52)),
            "weights": np.random.default_rng(3).normal(size=(4, 10, 52)),
            "biases": np.zeros((4, 10))}
    base = np.zeros((4, 52), dtype=np.float32)
    indices = [np.arange(4)] * 4
    labels, k, batches = np.arange(4), np.array([5, 10, 15, 20]), np.tile(np.arange(4), (2, 1))
    fitted = hc5.train(cache, reader, games, base, r10, head, indices, labels, k, batches, tmp_path, "test", 2027)
    result = hc5.extract(cache, reader, games, base, r10, fitted["model"])
    assert fitted["steps"] == 2 and len(fitted["history"]) == 2
    assert np.isfinite(result).all() and np.any(result[:, :30] != 0)
    assert np.any(result[:, 30:50] != 0)
    np.testing.assert_array_equal(result[:, 50:], base[:, 50:])
