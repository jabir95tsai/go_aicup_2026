import copy
from threading import Event

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.artifacts import file_hash
from go_aicup.tasks.strength.cvec2_policy import encoder, joint
from go_aicup.tasks.strength.cvec2_policy.train import B2


class Cache:
    device = torch.device("cpu")
    games = pd.DataFrame({"length": [3, 2, 3]})
    bounds = [0, 3, 5, 8]
    sides = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0])
    progress = torch.tensor([1/3, 2/3, 1, 1/2, 1, 1/3, 2/3, 1])

    def __init__(self):
        generator = torch.Generator().manual_seed(9)
        self.logp = torch.randn(8, 25, 362, generator=generator)
        self.logp[:, :, 27] = -torch.inf
        self.logp = self.logp.log_softmax(-1)
        self.target = torch.tensor([47, 302, 361, 1, 3, 2, 18, 1])
        self.stones = torch.zeros((8, 2, 361), dtype=torch.uint8)
        self.stones[1, 1, 47] = 1  # White to move after Black 47.
        self.stones[2, 0, 47] = 1  # Black to move after White 302.
        self.stones[2, 1, 302] = 1
        self.packed = torch.from_numpy(np.packbits(self.stones.flatten(1), axis=1))
        self.q = self.logp.gather(
            2, self.target[:, None, None].expand(-1, 25, 1),
        ).squeeze(-1)

    def for_device(self, device):
        assert device == self.device
        return self

    def counts(self, rows):
        return torch.cat([self.sides[self.bounds[row]:self.bounds[row + 1]].bincount(minlength=2)
                          for row in rows]).float()

    def chunks(self, rows, chunk, *, board=False):
        positions, slots = [], []
        for ordinal, row in enumerate(rows):
            a, b = self.bounds[row:row + 2]
            positions.extend(range(a, b))
            slots.extend((2 * ordinal + self.sides[a:b]).tolist())
        for a in range(0, len(positions), chunk):
            cut = slice(a, a + chunk)
            idx = positions[cut]
            values = self.logp[idx], self.target[idx], self.q[idx], torch.tensor(slots[cut])
            yield values + (self.packed[idx],) if board else values


def inputs():
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(3, 52)).astype(np.float32)
    raw[:, 50:] = [[1, 0], [0, 1], [0, 0]]
    initial = {
        "means": rng.normal(size=(4, 52)).astype(np.float32),
        "scales": rng.uniform(.8, 2., size=(4, 52)).astype(np.float32),
        "weights": rng.normal(0., .2, (4, 10, 52)).astype(np.float32),
        "biases": rng.normal(0., .2, (4, 10)).astype(np.float32),
    }
    questions = [np.array([0, 2]), np.array([0, 1, 2])]
    return Cache(), raw, initial, questions, np.array([3, 5]), np.array([5, 20])


def compare_gradients(actual, expected):
    for actual_param, expected_param in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(actual_param.grad, expected_param.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("architecture,full", [("R10", False), ("R10", True),
                                                ("CNN-R10", False), ("CNN-R10", True)])
def test_joint_gradient_step_zero_and_updates(architecture, full):
    cache, raw, initial, questions, labels, k = inputs()
    rows = np.arange(len(raw))
    model = encoder.build(architecture, full, 2027, raw_a=True)
    head = B2(initial, trainable=True)
    got = encoder.game_features(model, cache, raw, rows, 2)
    expected = torch.from_numpy(raw)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    torch.testing.assert_close(head(encoder.question_mean(got, questions), k),
                               B2(initial)(encoder.question_mean(expected, questions), k), atol=0, rtol=0)

    with torch.no_grad():
        model.h.weight.normal_(0, .02, generator=torch.Generator().manual_seed(33))
        model.c.weight.normal_(0, .02, generator=torch.Generator().manual_seed(35))
    reference = copy.deepcopy(model)
    direct_head = copy.deepcopy(head)
    direct_delta = reference(cache.logp, cache.target, cache.q)
    side_values = []
    for a, b in zip(cache.bounds[:-1], cache.bounds[1:]):
        for flag in (0, 1):
            chosen = cache.sides[a:b] == flag
            side_values.append(direct_delta[a:b][chosen].mean(0))
    side = torch.stack(side_values).reshape(-1, 2, 25)
    features = expected + torch.cat((side[:, :, :15].reshape(-1, 30),
                                     side[:, :, 15:].reshape(-1, 20),
                                     torch.zeros((len(raw), 2))), dim=1)
    direct_ce = encoder.question_ce(features, questions, labels, k, direct_head)
    direct_loss = direct_ce + direct_head.weights.square().sum() / 160_000
    direct_loss.backward()
    ce = encoder.backward_questions(model, cache, raw, questions, labels, k, head, 2, joint.head_penalty(head))
    assert ce == pytest.approx(float(direct_ce.detach()), abs=2e-6)
    compare_gradients(model, reference)
    compare_gradients(head, direct_head)
    torch.testing.assert_close(head.weights.grad[1], head.weights[1] / 80_000)
    torch.testing.assert_close(head.biases.grad[1], torch.zeros_like(head.biases[1]))

    optimizer = torch.optim.AdamW([{"params": model.parameters()},
                                   {"params": head.parameters(), "weight_decay": 0}], lr=joint.LR)
    optimizer.step()
    assert not torch.equal(model.h.weight, reference.h.weight)
    assert not torch.equal(model.c.weight, reference.c.weight)
    assert model.hidden[0].weight.grad.count_nonzero()
    assert not torch.equal(head.weights.detach(), torch.from_numpy(initial["weights"]))
    assert not torch.equal(head.biases.detach(), torch.from_numpy(initial["biases"]))
    torch.testing.assert_close(head.means, torch.from_numpy(initial["means"]), atol=0, rtol=0)
    torch.testing.assert_close(head.scales, torch.from_numpy(initial["scales"]), atol=0, rtol=0)
    policy, target, raw_q = cache.logp, cache.target, cache.q
    shift_h, shift_c = model.delta(policy, target, raw_q)
    residual = model(policy, target, raw_q)
    h = raw_q[:, :15]
    torch.testing.assert_close(residual[:, :15], (h + shift_h).log_softmax(-1) - h.log_softmax(-1))
    torch.testing.assert_close(residual[:, 15:], shift_c, atol=0, rtol=0)
    p, lp, legal = model.policy_inputs(policy, target, raw_q)
    torch.testing.assert_close(lp[torch.arange(len(target)), :, target], raw_q, atol=0, rtol=0)
    torch.testing.assert_close(p[torch.arange(len(target)), :, target], raw_q.exp(), atol=0, rtol=0)
    assert not legal[:, 27].any() and (lp[:, :, 27] == 0).all() and (p[:, :, 27] == 0).all()
    changed = policy.clone()
    changed[:, :, 55] += 2
    if full:
        assert not torch.equal(model.policy_inputs(changed, target, raw_q)[0], p)
    else:
        torch.testing.assert_close(model(changed, target, raw_q), residual, atol=0, rtol=0)



def test_weighted_chunk_gradient_and_uniform_start():
    cache, raw, initial, questions, labels, k = inputs()
    rows = np.arange(len(raw))
    mother = encoder.build("CNN-R10", True, 2027, raw_a=True).requires_grad_(False)
    with torch.no_grad():
        mother.c.weight.fill_(.03)
        mother.h.weight.copy_(torch.arange(15)[:, None] / 100)
    torch.manual_seed(19)
    weight = joint.MoveWeights()
    head = B2(initial)
    counts = cache.counts(rows)
    base = torch.from_numpy(raw)
    args = (mother, weight, cache, rows, 2, counts)
    values = joint.weighted_moves(*args)
    uniform, concentration = joint.weighted_reduce(base, *values, counts)
    mother_features = encoder.game_features(mother, cache, raw, rows, 2)
    torch.testing.assert_close(uniform, mother_features, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(head(uniform[:2], k), head(mother_features[:2], k), atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(concentration, torch.ones_like(concentration))
    progress = torch.cat([x[-1] for x in joint.move_chunks(cache, rows, 2, counts)])
    torch.testing.assert_close(progress, cache.progress, atol=0, rtol=0)
    torch.testing.assert_close(uniform[:, -2:], base[:, -2:], atol=0, rtol=0)

    with torch.no_grad():
        weight.head[-1].weight.fill_(6)
    reference = copy.deepcopy(weight)
    p, _, _ = mother.policy_inputs(cache.logp, cache.target, cache.q)
    image = reference.board_inputs(p, cache.packed, cache.target, cache.progress)
    assert image.shape == (8, 29, 19, 19)
    torch.testing.assert_close(image[:, :25].flatten(2), p[..., :361])
    torch.testing.assert_close(image[:, 25:27].flatten(2), cache.stones.float())
    assert image[1, 26, 2, 9] == image[2, 25, 2, 9] == image[2, 26, 15, 17] == 1
    assert image[0, 27, 2, 9] == image[1, 27, 15, 17] == 1
    assert image[2, 27].count_nonzero() == 0
    torch.testing.assert_close(image[:, 28], cache.progress[:, None, None].expand(-1, 19, 19))
    logits = reference(p, cache.packed, cache.target, cache.progress)
    delta = mother(cache.logp, cache.target, cache.q)
    q = cache.q
    corrections, concentrations = [], []
    for a, b in zip(cache.bounds[:-1], cache.bounds[1:]):
        for side in range(2):
            ids = np.arange(a, b)[cache.sides[a:b].numpy() == side]
            w = logits[ids].softmax(0)
            corrections.append((w[:, None] * (q[ids] + delta[ids])).sum(0) - q[ids].mean(0))
            concentrations.append(len(w) * w.square().sum())
    direct_features = encoder.add_side_features(base, torch.stack(corrections))
    direct_ce = encoder.question_ce(direct_features, questions, labels, k, head)
    direct_penalty = .01 * (torch.stack(concentrations) - 1).mean()
    (direct_ce + direct_penalty).backward()

    got, concentration = joint.weighted_reduce(base, *joint.weighted_moves(*args), counts)
    torch.testing.assert_close(got, direct_features, atol=2e-6, rtol=1e-6)
    torch.testing.assert_close(concentration, torch.stack(concentrations))
    ce, penalty = joint.backward_weighted(mother, weight, cache, raw, questions, labels, k, head, 2)
    assert ce == pytest.approx(float(direct_ce.detach()), abs=2e-6)
    assert penalty == pytest.approx(float(direct_penalty.detach()), abs=2e-8)
    compare_gradients(weight, reference)
    assert all(parameter.grad is None for parameter in mother.parameters())
    assert not list(head.parameters())


def test_extract_all_rows_without_e2(tmp_path):
    cache, raw, *_ = inputs()
    arm = "A/R10-D"
    path = tmp_path / arm
    path.mkdir(parents=True)
    model = encoder.build("R10", True, 2027, raw_a=True)
    torch.save({"arm": arm, "identity": "test", "mother_sha256": None,
                "model": model.state_dict()}, path / "model.pt")
    joint.extract_one(arm, cache, raw, torch.device("cpu"), tmp_path, "test", len(raw))
    np.testing.assert_array_equal(np.load(path / "fit.npy"), raw)
    assert np.load(path / "e2.npy").shape == (0, 52)


def test_game_parallel_matches_single_device(tmp_path):
    cache, raw, initial, questions, labels, k = inputs()
    arm = "A/R10-D"
    batches = np.tile(np.arange(len(questions)), (2, 1))
    device = torch.device("cpu")
    payloads, outputs = [], []
    for name, replicas in (("single", ()), ("parallel", ("cpu", "cpu"))):
        out = tmp_path / name
        joint.train_one(arm, cache, raw, questions, labels, k, batches, initial,
                        device, out, name, replicas=replicas)
        joint.extract_one(arm, cache, raw, device, out, name, len(raw), replicas=replicas)
        path = out / arm
        payloads.append(joint.load_checkpoint(path / "model.pt", arm, name, None))
        outputs.append(np.load(path / "fit.npy"))
    for group in ("model", "head"):
        for key in payloads[0][group]:
            torch.testing.assert_close(payloads[0][group][key], payloads[1][group][key],
                                       atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(payloads[0]["diagnostics"]["ce"], payloads[1]["diagnostics"]["ce"], atol=2e-6)
    np.testing.assert_allclose(*outputs, atol=2e-6)


@pytest.mark.parametrize("arm", ["A/R10-C", "A/CNN-R10-D-weighted"])
def test_resume_and_extract(arm, tmp_path, monkeypatch):
    cache, raw, initial, questions, labels, k = inputs()
    monkeypatch.setattr(joint, "CHECKPOINT_EVERY", 1)
    monkeypatch.setattr(joint, "POSITION_CHUNK", 2)
    complete = tmp_path / "complete"
    resume = tmp_path / "resume"
    batches = np.array([[0, 1], [1, 1]])

    def train(arm, out):
        return joint.train_one(arm, cache, raw, questions, labels, k, batches, initial,
                               cache.device, out, "test-v1", Event())

    if arm.endswith("weighted"):
        train("A/CNN-R10-D", complete)
        train("A/CNN-R10-D", resume)
        mother = resume / "A/CNN-R10-D/model.pt"
        mother_hash = file_hash(mother)
    expected = train(arm, complete)

    def interrupted(value, path):
        atomic_save(value, path)
        if path.name == "checkpoint.pt":
            raise RuntimeError("interruption")

    atomic_save = joint.atomic_save
    with monkeypatch.context() as patch:
        patch.setattr(joint, "atomic_save", interrupted)
        with pytest.raises(RuntimeError, match="interruption"):
            train(arm, resume)

    resumed = train(arm, resume)
    assert expected["ce"] == resumed["ce"]
    assert expected["regularizer"] == resumed["regularizer"]
    before = torch.load(complete / arm / "model.pt", weights_only=True)
    after = torch.load(resume / arm / "model.pt", weights_only=True)
    for name in ("head", "weight" if arm.endswith("weighted") else "model"):
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
    record = joint.extract_one(arm, cache, raw, cache.device, resume, "test-v1", 2)
    assert joint.extract_one(arm, cache, raw, cache.device, resume, "test-v1", 2) == record
    np.testing.assert_array_equal(np.load(resume / arm / "fit.npy")[:, -2:], raw[:2, -2:])
    np.testing.assert_array_equal(np.load(resume / arm / "e2.npy")[:, -2:], raw[2:, -2:])
    np.testing.assert_array_equal(np.load(resume / arm / "fit_40.npy"),
                                  np.load(resume / arm / "fit.npy")[:, joint.FIT_PROJECTION])
    if arm.endswith("weighted"):
        assert file_hash(mother) == after["mother_sha256"] == mother_hash
        assert 0 < record["effective_moves"]["e2"]["own"]["p05"] <= 1
