"""CPU-only regression checks for latest-only checkpoint publication."""

import json

import pytest
import torch

from go_aicup.tasks.strength.runtime import model_state_hash
from go_aicup.tasks.strength.train_rank_v1 import resolve_checkpoint, save_checkpoint


def checkpoint_writer(directory):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = {"optimizer_steps": 0, "epoch": 0, "next_batch": 0}

    def save():
        state["optimizer_steps"] += 1
        return save_checkpoint(directory, model, optimizer, scheduler, scaler,
                               {"precision": "fp32"},
                               {"dataset_id": "test", "cache_build_id": "test"},
                               state, torch.device("cpu"), 0, 1)

    return model, save


def test_latest_checkpoint_replaces_old_and_remains_loadable(tmp_path):
    model, save = checkpoint_writer(tmp_path)
    first = save()
    (tmp_path / "best.json").write_text(json.dumps({"file": first.name}))
    second = save()
    assert not first.exists()
    assert not first.with_suffix(".complete.json").exists()
    assert not (tmp_path / "best.json").exists()
    assert list(second.parent.glob("*.pt")) == [second]
    assert list(second.parent.glob("*.complete.json")) == [second.with_suffix(".complete.json")]
    assert resolve_checkpoint(tmp_path / "last.json") == second
    assert resolve_checkpoint(second) == second
    payload = torch.load(second, map_location="cpu", weights_only=False)
    assert payload["state"]["optimizer_steps"] == 2
    for name, value in model.state_dict().items():
        assert torch.equal(value, payload["model"][name])


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    _, save = checkpoint_writer(tmp_path)
    first = save()

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        save()
    assert resolve_checkpoint(tmp_path / "last.json") == first
    assert first.with_suffix(".complete.json").exists()


def test_evaluation_fingerprint_tracks_weights_and_scalar_bn_buffers():
    model = torch.nn.BatchNorm1d(2)
    original = model_state_hash(model)
    assert model_state_hash(model) == original
    model.num_batches_tracked.add_(1)
    assert model_state_hash(model) != original
    model.num_batches_tracked.sub_(1)
    with torch.no_grad():
        model.weight.add_(1)
    assert model_state_hash(model) != original
