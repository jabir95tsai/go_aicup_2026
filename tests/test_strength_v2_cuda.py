"""CUDA/NCCL/SyncBN acceptance on synthetic data; skipped without 2 GPUs.

Run on an allocated compute node (S8), never on a login node:
    PYTHONPATH=src python -m pytest -q tests/test_strength_v2_cuda.py
"""
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from go_aicup.tasks.strength.shared.runtime import load_dataset_manifest
from go_aicup.tasks.strength.v2.infer import infer_games
from go_aicup.tasks.strength.v2.model import StrengthResNet
from go_aicup.tasks.strength.v2.train import CATEGORY_FILES, LOSS_DTYPE, train

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 CUDA devices")


def cuda_config(base, output, precision):
    config = json.loads(json.dumps(base))
    config.update(device="cuda", precision=precision)
    config["paths"]["output_dir"] = str(output)
    config["training"]["sync_batch_norm"] = True
    return config


def _worker(rank, world, config, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    train(config)


def run(config, world):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_worker, args=(world, config, port), nprocs=world, join=True)
    output = Path(config["paths"]["output_dir"])
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    return output, payload


def test_syncbn_two_gpus_match_one_gpu_with_partial_tail(v2_train_data, tmp_path):
    single = cuda_config(v2_train_data, tmp_path / "one", "fp32")
    manifest = load_dataset_manifest(single)
    samples = manifest["sets"]["train"]["num_samples"]
    assert samples % single["training"]["global_batch_size"] != 0  # A partial tail batch is trained.
    one, first = run(single, 1)
    two, second = run(cuda_config(v2_train_data, tmp_path / "two", "fp32"), 2)
    assert first["samples_seen"] == second["samples_seen"] == samples
    assert second["environment"]["world_size"] == 2
    a, b = (np.load(path / "losses.npy", allow_pickle=False) for path in (one, two))
    np.testing.assert_array_equal(a["step"], b["step"])
    for name in LOSS_DTYPE.names[1:]:
        np.testing.assert_allclose(a[name], b[name], rtol=2e-3, atol=2e-4, err_msg=name)
    for name in CATEGORY_FILES.values():
        x, y = (np.load(path / name, allow_pickle=False) for path in (one, two))
        np.testing.assert_allclose(x["confidence_sum"], y["confidence_sum"], rtol=2e-3, atol=2e-3)


def test_bf16_two_gpus_categories_reproduce_from_model(v2_train_data, tmp_path):
    config = cuda_config(v2_train_data, tmp_path / "bf16", "bf16")
    output, payload = run(config, 2)
    losses = np.load(output / "losses.npy", allow_pickle=False)
    for name in LOSS_DTYPE.names[1:]:
        assert np.isfinite(losses[name]).all()
    model = StrengthResNet(**config["model"]).cuda()
    model.load_state_dict(payload["model"])
    for prefix, name in CATEGORY_FILES.items():
        saved = np.load(output / name, allow_pickle=False)
        again = infer_games(model, config, prefix, torch.device("cuda", 0))
        np.testing.assert_array_equal(saved["num_moves"], again["num_moves"])
        np.testing.assert_allclose(saved["confidence_sum"], again["confidence_sum"], rtol=1e-2, atol=1e-2)
        assert (saved["predicted_rank_id"] == again["predicted_rank_id"]).mean() >= 0.99


def _resume_worker(rank, world, config, port, max_steps, resume):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      LOCAL_WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    train(config, max_steps=max_steps, resume=resume)


def test_bf16_two_gpu_resume_matches_uninterrupted(v2_train_data, tmp_path):
    full = cuda_config(v2_train_data, tmp_path / "full", "bf16")
    _, uninterrupted = run(full, 2)
    partial = cuda_config(v2_train_data, tmp_path / "partial", "bf16")
    output = Path(partial["paths"]["output_dir"])
    for max_steps, resume in ((5, None), (None, output / "last.json")):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        torch.multiprocessing.spawn(_resume_worker, args=(2, partial, port, max_steps, resume),
                                    nprocs=2, join=True)
    resumed = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    exact = uninterrupted["model_state_sha256"] == resumed["model_state_sha256"]
    print(f"bf16 2-GPU resume bit-identical: {exact}")
    for key, value in uninterrupted["model"].items():
        if value.is_floating_point():
            torch.testing.assert_close(resumed["model"][key], value, rtol=1e-3, atol=1e-4)
    a, b = (np.load(Path(c["paths"]["output_dir"]) / "losses.npy", allow_pickle=False) for c in (full, partial))
    for name in LOSS_DTYPE.names:
        np.testing.assert_allclose(a[name], b[name], rtol=1e-3, atol=1e-4, err_msg=name)
