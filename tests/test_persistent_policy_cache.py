from concurrent.futures import ThreadPoolExecutor
import io
import threading
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.policy_residual.persistent_cache import (
    complete_shard, encode_supplement, infer, input_pipeline, validate_inputs,
    validate_policy, validate_shard, zip_sample,
)


def example():
    rows = pd.DataFrame(dict(game_index=[17, 23], length=[3, 2], player_color=[0, 1]))
    bins = np.zeros((5, 22, 19, 19), np.uint8)
    bins[:, 0] = 1
    bins[:, 1, 0, 2] = 1  # occupied
    bins[:, 6, 0, 3] = 1  # ko
    target = np.array([0, 1, 361, 4, 5], dtype=np.int64)
    side = np.array([0, 1, 0, 1, 0], np.uint8)
    bits = np.packbits(bins.reshape(5, -1), axis=1)
    return rows, bits, np.zeros((5, 19), np.float32), target, side


def forward(x, g, rank, side):
    logits = torch.arange(362, device=x.device).float()[None].expand(len(x), -1) / 1000
    return logits + rank[:, None] / 10 + side[:, None] / 20


def test_exact_q_tail_mask_and_atomic_resume(tmp_path):
    rows, bits, glob, target, side = example()
    values, checks, events = infer(forward, torch.device('cpu'), rows, bits, glob, target, side, 4)
    metrics = validate_policy(values, checks)
    assert events is None
    assert np.isneginf(values['logp'][:, :, [2, 3]]).all()
    assert values['logp'].shape == (5, 10, 362)
    assert values['q'].dtype == np.float32
    legal = torch.arange(362)
    logits = legal.float() / 1000
    logits[[2, 3]] = -torch.inf
    expected = logits.log_softmax(-1)[target].numpy()
    np.testing.assert_allclose(values['q'], np.repeat(expected[:, None], 10, axis=1), atol=1e-6)
    # The FP32 chosen value survives independently of rounded FP16 logP.
    recovered = values['logp'][np.arange(5), 0, target].astype(np.float32)
    assert np.any(values['q'][:, 0] != recovered)
    path = tmp_path / 'shard'
    assert validate_shard(path, 'teacher-v1', rows) is None
    receipt = complete_shard(path, 'teacher-v1', rows, values, checks, events)
    assert receipt['positions'] == 5 and receipt['metrics'] == metrics
    assert not path.with_name('shard.partial').exists()
    assert validate_shard(path, 'teacher-v1', rows)['positions'] == 5
    with pytest.raises(ValueError, match='identity'):
        validate_shard(path, 'teacher-v2', rows)
    q = np.load(path / 'q.npy')
    q[0, 0] += 1
    np.save(path / 'q.npy', q)
    with pytest.raises(ValueError, match='corruption'):
        validate_shard(path, 'teacher-v1', rows)


def test_input_game_order_and_mismatched_played_move():
    rows, bits, glob, target, side = example()
    raw = [(target[:3], np.arange(3) % 2), (target[3:], np.arange(2) % 2)]
    validate_inputs(rows, bits, glob, target, side, raw)
    with pytest.raises(ValueError, match='order mismatch'):
        validate_inputs(rows, bits, glob, target[::-1], side, raw)
    bad_target = target.copy()
    bad_target[0] = 2
    with pytest.raises(ValueError, match='excluded'):
        values, checks, _ = infer(forward, torch.device('cpu'), rows, bits, glob, bad_target, side, 4)
        validate_policy(values, checks)


def test_replay_keeps_working_before_gpu_consumes_cached_inputs():
    release_cached = threading.Event()
    replay_finished = threading.Event()

    def cached():
        assert release_cached.wait(5)
        yield 'cached'

    def replayed():
        for i in range(12):
            yield i
        replay_finished.set()

    stats = {}
    with input_pipeline(cached(), replayed(), 12, stats) as ready:
        try:
            # Every replay shard can finish while the GPU has consumed nothing.
            assert replay_finished.wait(5)
            assert next(ready) == 0
        finally:
            release_cached.set()
        remaining = list(ready)
    assert sorted(x for x in remaining if isinstance(x, int)) == list(range(1, 12))
    assert remaining.count('cached') == 1
    assert stats['replayed']['shards'] == 12
    assert stats['cached']['shards'] == 1


def test_cached_inputs_do_not_wait_for_replay_and_early_exit_closes_producers():
    release_replay = threading.Event()

    def cached():
        yield from range(100)

    def replayed():
        assert release_replay.wait(5)
        yield 'replay'

    stats = {}
    with input_pipeline(cached(), replayed(), 1, stats) as ready:
        try:
            assert next(ready) == 0
            assert not release_replay.is_set()
        finally:
            release_replay.set()
    assert stats['cached']['shards'] <= 5  # The cached producer has a four-shard bound.


def test_input_producer_failure_is_propagated():
    def empty():
        yield from ()

    def broken():
        raise RuntimeError('replay failed')
        yield

    with pytest.raises(RuntimeError, match='replay failed'):
        with input_pipeline(empty(), broken(), 0, {}) as ready:
            list(ready)


def test_encoder_packs_before_ipc_and_gpu_unpack_preserves_planes(monkeypatch):
    from go_aicup.features import humansl_torch as ht
    from go_aicup.features.humansl_rankvec import unpack
    rng = np.random.default_rng(4)
    bins = rng.integers(0, 2, (3, 22, 19, 19), dtype=np.uint8)
    glob = np.zeros((3, 19), np.float32)
    target = np.arange(3)
    monkeypatch.setattr(ht, 'encode_game', lambda raw: (bins, glob, target))
    monkeypatch.setattr(ht, '_KOMI', None)
    bits, actual_glob, actual_target = encode_supplement((1, (target, target % 2)))
    assert bits.shape == (3, 993) and bits.dtype == np.uint8
    np.testing.assert_array_equal(unpack(torch.from_numpy(bits)).numpy(), bins)
    np.testing.assert_array_equal(actual_glob, glob)
    np.testing.assert_array_equal(actual_target, target)
    assert ht._KOMI == .5


def test_nonfinite_inference_withholds_publication(tmp_path):
    rows, bits, glob, target, side = example()

    def bad_forward(x, g, rank, side):
        return torch.full((len(x), 362), float('nan'))

    values, checks, events = infer(bad_forward, torch.device('cpu'), rows, bits, glob, target, side, 4)
    with pytest.raises(ValueError, match='nonfinite'):
        complete_shard(tmp_path / 'bad', 'identity', rows, values, checks, events)
    assert not (tmp_path / 'bad').exists()


def test_zip_sample_shape_dtype_and_bounds():
    source = np.arange(24, dtype=np.int16).reshape(6, 4)
    raw, zipped = io.BytesIO(), io.BytesIO()
    np.save(raw, source)
    with zipfile.ZipFile(zipped, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('x.npy', raw.getvalue())
    zipped.seek(0)
    with zipfile.ZipFile(zipped) as archive:
        shape, dtype, value = zip_sample(archive, 'x.npy', 2, 3)
        assert shape == (6, 4) and dtype == np.int16
        np.testing.assert_array_equal(value, source[2:5])
        with pytest.raises(ValueError, match='outside'):
            zip_sample(archive, 'x.npy', 5, 2)


def test_background_writer_waits_for_copy_and_does_not_block_caller(tmp_path):
    rows, bits, glob, target, side = example()
    values, checks, _ = infer(forward, torch.device('cpu'), rows, bits, glob, target, side, 4)
    original = {name: values[name].copy() for name in ('logp', 'q')}
    for name in original:
        values[name].fill(np.nan)
    entered, copied = threading.Event(), threading.Event()

    class CopyEvent:
        def synchronize(self):
            entered.set()
            assert copied.wait(5)

        def elapsed_time(self, other):
            return 1.5

    path = tmp_path / 'async_shard'
    with ThreadPoolExecutor(max_workers=1) as writer:
        pending = writer.submit(complete_shard, path, 'identity', rows, values, checks,
                                (CopyEvent(), CopyEvent(), CopyEvent()))
        try:
            assert entered.wait(5)
            assert not pending.done() and not path.exists()
            for name in original:
                values[name][:] = original[name]
        finally:
            copied.set()
        receipt = pending.result(timeout=5)
    assert receipt['metrics']['output_copy_ms'] == 1.5
    assert validate_shard(path, 'identity', rows)['positions'] == 5
