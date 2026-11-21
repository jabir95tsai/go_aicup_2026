from concurrent.futures import Future
from threading import Event, Lock
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from go_aicup.tasks.strength.cvec2_policy import all_report, all_run, joint


def test_train_on_fit_then_extract_on_full(monkeypatch, tmp_path):
    fit, full = object(), object()
    full_ready, abort, lock = Future(), Event(), Lock()
    data = SimpleNamespace(groups={"fit": [0]}, features={"A": np.zeros((2, 52))},
                           indices=[], labels=[], k=[], batches=[], heads={"A": object()})
    trained, extracted = set(), set()

    def train(arm, cache, raw, *args, on_first_update):
        with lock:
            assert cache is fit and raw.shape == (1, 52)
            assert not full_ready.done()
            if arm == all_report.WEIGHTED:
                assert all_report.MOTHER in trained
            on_first_update()
            trained.add(arm)
            if len(trained) == len(all_report.LEARNED_ARMS):
                full_ready.set_result(full)
        return {}

    def extract(arm, cache, raw, device, out, identity, n_fit, abort):
        with lock:
            assert arm in trained and cache is full and raw.shape == (2, 52) and n_fit == 1
            extracted.add(arm)
        return {}

    monkeypatch.setattr(joint, "train_one", train)
    monkeypatch.setattr(joint, "extract_one", extract)
    report = all_run.run_tasks(tmp_path, data, fit, full_ready, "id", abort, monotonic())
    assert trained == extracted == set(all_report.LEARNED_ARMS)
    assert report["first_update_seconds_from_gpu_start"].keys() == trained
    assert not abort.is_set()


def test_e2_failure_aborts_the_scheduler(tmp_path):
    full_ready, abort = Future(), Event()
    full_ready.set_exception(ValueError("E2 failed"))
    data = SimpleNamespace(groups={"fit": [0]})
    with pytest.raises(ValueError, match="E2 failed"):
        all_run.run_tasks(tmp_path, data, object(), full_ready, "id", abort, monotonic())
    assert abort.is_set()


@pytest.mark.parametrize("reserved", [0, 8])
def test_h15_work_is_allocated_once_and_helpers_use_peer_shards(reserved):
    from go_aicup.tasks.strength.cvec2_policy.full_cache import FitWork

    store = SimpleNamespace(bounds=np.arange(257) * 220, ranges=[(0, 86), (86, 173), (173, 256)])
    work = FitWork(store, reserved)
    claims = [work.take(0)]
    helper = work.take(0, helper=True)
    assert helper is not None
    if reserved:
        assert helper == (0, 78, 86)
    claims.append(helper)
    for owner in range(len(store.ranges)):
        while (item := work.take(owner)) is not None:
            claims.append(item)
    for owner in range(len(store.ranges)):
        while (item := work.take(owner, helper=True)) is not None:
            claims.append(item)
    visited = np.zeros(256, dtype=int)
    for owner, first, last in claims:
        a, b = store.ranges[owner]
        assert a <= first < last <= b
        visited[first:last] += 1
    np.testing.assert_array_equal(visited, 1)
