"""CPU-only regression checks for latest-only checkpoint publication."""

import json

import pytest
import torch

from go_aicup.tasks.strength.shared.runtime import model_state_hash
from go_aicup.tasks.strength.v1.train import resolve_checkpoint, save_checkpoint


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


def test_loss_retention_keeps_two_global_best_and_latest(tmp_path):
    from go_aicup.tasks.strength.shared.checkpoint import save_checkpoint as shared_save
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)

    def save(step, loss):
        return shared_save(tmp_path, model, optimizer, scheduler, scaler,
                           {'precision': 'fp32'}, {'dataset_id': 'test', 'cache_build_id': 'test'},
                           dict(optimizer_steps=step, epoch=0, next_batch=step, checkpoint_loss=loss),
                           torch.device('cpu'), 0, 1, code_sha256={}, loss=loss)

    first = save(1000, 3.0)
    equal = save(2000, 3.0)
    worse = save(3000, 4.0)
    assert first.exists() and equal.exists() and worse.exists()
    better = save(4000, 3.5)
    assert first.exists() and equal.exists() and not worse.exists()
    assert not worse.with_suffix('.complete.json').exists()
    assert resolve_checkpoint(tmp_path / 'last.json') == better
    assert json.loads((tmp_path / 'last.json').read_text())['loss'] == 3.5
    loaded = torch.load(better, weights_only=False)
    assert loaded['state']['checkpoint_loss'] == 3.5
    with pytest.raises(ValueError, match='finite'):
        save(5000, float('nan'))
    assert resolve_checkpoint(tmp_path / 'last.json') == better
    lowest = save(5000, 2.0)
    assert not first.exists() and not better.exists()
    assert equal.exists() and lowest.exists()
    ranked = json.loads((tmp_path / 'best.json').read_text())['checkpoints']
    assert [record['loss'] for record in ranked] == [2.0, 3.0]
    assert [record['optimizer_steps'] for record in ranked] == [5000, 2000]
    latest = save(6000, 9.0)
    assert equal.exists() and lowest.exists() and latest.exists()
    assert len(list(latest.parent.glob('*.pt'))) == 3
    from go_aicup.tasks.strength.reproduce_paper.train import archive_best_checkpoints
    archive_best_checkpoints(tmp_path)
    archived = json.loads((tmp_path / 'best_checkpoints.json').read_text())['checkpoints']
    assert [record['loss'] for record in archived] == [2.0, 3.0]
    for record in archived:
        payload = torch.load(resolve_checkpoint(tmp_path / record['file']), weights_only=False)
        assert payload['state']['checkpoint_loss'] == record['loss']
    archive_best_checkpoints(tmp_path)


def test_loss_checkpoint_failed_publication_keeps_previous(tmp_path, monkeypatch):
    from go_aicup.tasks.strength.shared import checkpoint
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    args = (tmp_path, model, optimizer, scheduler, scaler, {'precision': 'fp32'},
            {'dataset_id': 'test', 'cache_build_id': 'test'},
            dict(optimizer_steps=1000, epoch=0, next_batch=1000), torch.device('cpu'), 0, 1)
    first = checkpoint.save_checkpoint(*args, code_sha256={}, loss=3.0)
    original = checkpoint.write_json

    def fail_last(path, record):
        if path.name == 'last.json':
            raise OSError('simulated publication failure')
        original(path, record)

    monkeypatch.setattr(checkpoint, 'write_json', fail_last)
    with pytest.raises(OSError, match='publication failure'):
        checkpoint.save_checkpoint(*args, code_sha256={}, loss=2.0)
    assert resolve_checkpoint(tmp_path / 'last.json') == first
    assert first.with_suffix('.complete.json').exists()


def test_archive_includes_latest_when_best_index_publication_failed(tmp_path, monkeypatch):
    from go_aicup.tasks.strength.shared import checkpoint
    from go_aicup.tasks.strength.reproduce_paper.train import archive_best_checkpoints
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    def save(step, loss):
        return checkpoint.save_checkpoint(tmp_path, model, optimizer, scheduler, scaler,
            {'precision': 'fp32'}, {'dataset_id': 'test', 'cache_build_id': 'test'},
            dict(optimizer_steps=step, epoch=0, next_batch=step, checkpoint_loss=loss),
            torch.device('cpu'), 0, 1, code_sha256={}, loss=loss)
    save(1000, 3.0)
    save(2000, 2.0)
    original = checkpoint.write_json
    def fail_best(path, record):
        if path.name == 'best.json':
            raise OSError('simulated best index publication failure')
        original(path, record)
    monkeypatch.setattr(checkpoint, 'write_json', fail_best)
    with pytest.raises(OSError, match='best index'):
        save(3000, 1.0)
    assert resolve_checkpoint(tmp_path / 'last.json').exists()
    archive_best_checkpoints(tmp_path)
    records = json.loads((tmp_path / 'best_checkpoints.json').read_text())['checkpoints']
    assert [record['loss'] for record in records] == [1.0, 2.0]
    assert [record['optimizer_steps'] for record in records] == [3000, 2000]
