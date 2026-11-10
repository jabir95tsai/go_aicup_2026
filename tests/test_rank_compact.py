"""Synthetic CPU checks for compact training and saved-score evaluation."""
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.data.board_cache import convert_training
from go_aicup.tasks.strength.v1.evaluate import (
    decode_scores, evaluate_saved, fit_cutpoints, load_saved_run, score_report)
from go_aicup.tasks.strength.v1.infer import LOSS_DTYPE, infer_split
from go_aicup.tasks.strength.v1.losses import ranking_loss_from_means
from go_aicup.tasks.strength.v1.prepare import prepare
from go_aicup.tasks.strength.shared.runtime import load_dataset_manifest
from go_aicup.tasks.strength.v1.config import read_config
from go_aicup.tasks.strength.v1.train import train


@pytest.fixture(scope="module")
def compact_data(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("compact_rank")
    raw, cache = root / "raw", root / "cache"
    raw.mkdir()
    ranks = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
    for rank_id, rank in enumerate(ranks):
        with (raw / f"train_{rank}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("player_id", "game_id", "rank", "color", "sgf_content"))
            writer.writeheader()
            for game in range(40):
                x = chr(ord("b") + rank_id)
                writer.writerow(dict(player_id=f"p{rank}", game_id=f"g{game}", rank=rank, color="B",
                                     sgf_content=f"(;SZ[19];B[{x}b];W[ss];B[{x}c];W[sr])"))
    convert_training(raw, cache, workers=1, games_per_shard=40, ranks=ranks, expected_games_per_rank=40)
    path = Path(__file__).resolve().parents[1] / "configs/strength/rank_cnn_v1_seed2027.json"
    config = read_config(path)
    config.update(device="cpu", precision="fp32")
    config["paths"].update(cache_dir=str(cache), dataset_dir=str(root / "processed"), output_dir=str(root / "run"))
    config["model"].update(channels=2, blocks=1, hidden_size=4)
    config["split"].update(expected_games_per_rank=40, train_games_per_rank=20)
    config["questions"] = {"validation": 40}
    config["loader"].update(num_workers=0, persistent_workers=False, pin_memory=False)
    config["training"].update(global_batch_size=100, evaluate_every_steps=2, sync_batch_norm=False)
    config["evaluation"]["global_batch_size"] = 74  # Ordered batches can miss ranks.
    prepare(config)
    return config


@pytest.fixture(autouse=True)
def small_validation_sample(monkeypatch):
    # Synthetic splits have only 40 boards per rank.
    monkeypatch.setattr("go_aicup.tasks.strength.v1.train.VALIDATION_SAMPLES_PER_RANK", 20)


def test_validation_sampling_is_balanced_fixed_and_without_replacement(tmp_path, monkeypatch):
    import go_aicup.tasks.strength.v1.train as training
    monkeypatch.setattr(training, "VALIDATION_SAMPLES_PER_RANK", 10000)
    labels = np.repeat(np.arange(10, dtype=np.uint8), 12000)
    pd.DataFrame({"rank_id": labels}).to_parquet(tmp_path / "strength_v1_games.parquet")
    games = np.random.default_rng(42).permutation(len(labels))
    np.save(tmp_path / "valid_game.npy", games)
    config = {"seed": 2027, "paths": {"dataset_dir": str(tmp_path)}}
    indices = training.validation_sample_indices(config)
    assert len(indices) == len(np.unique(indices)) == 100000
    np.testing.assert_array_equal(np.bincount(labels[games[indices]], minlength=10), [10000] * 10)
    np.testing.assert_array_equal(indices, training.validation_sample_indices(config))
    config["seed"] = 2028
    assert not np.array_equal(indices, training.validation_sample_indices(config))


def clone_config(config):
    return json.loads(json.dumps(config))


def test_cutpoints_match_exhaustive_ordered_partitions(tmp_path):
    scores = np.array([0, 0, 1, 2, 2, 3], dtype=np.float32)
    labels = np.array([0, 2, 4, 6, 8, 9], dtype=np.int8)
    result = fit_cutpoints(scores, labels, scratch_dir=tmp_path)
    best = 0
    # Every assignment of four distinct scores to nondecreasing rank IDs.
    for assignments in itertools.combinations_with_replacement(range(10), 4):
        prediction = np.asarray(assignments)[scores.astype(int)]
        error = abs(prediction - labels)
        best = max(best, np.where(error == 0, 1, np.where(error == 1, np.exp(-1), 0)).mean())
    actual = score_report(scores, labels, result["boundaries"])["score"]
    assert actual == pytest.approx(best)
    assert result["training_game_score"] == pytest.approx(best)
    assert not list(tmp_path.iterdir())
    tied = fit_cutpoints(np.zeros(10, np.float32), np.arange(10, dtype=np.int8))
    assert len(set(decode_scores(np.zeros(10), tied["boundaries"]))) == 1
    with pytest.raises(ValueError):
        fit_cutpoints([np.nan], np.array([0]))


def test_full_loss_frozen_bn_and_index_aligned_scores(compact_data, tmp_path):
    from go_aicup.tasks.strength.v1.model import StrengthResNet
    config = clone_config(compact_data)
    manifest = load_dataset_manifest(config)
    model = StrengthResNet(**config["model"]).train()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    path = tmp_path / "scores.npy"
    loss = infer_split(model, config, manifest, "valid", torch.device("cpu"), output=path)
    assert model.training
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    scores = np.load(path)
    directory = Path(config["paths"]["dataset_dir"])
    meta = pd.read_parquet(directory / "strength_v1_games.parquet")
    indices = np.load(directory / "valid_game.npy")
    labels = meta.rank_id.to_numpy()[indices]
    means = torch.tensor([scores[labels == rank].mean(dtype=np.float64) for rank in range(10)])
    assert loss == pytest.approx(ranking_loss_from_means(means).item(), abs=1e-6)
    from go_aicup.tasks.strength.v1.train import validation_sample_indices
    selected = validation_sample_indices(config)
    sampled_loss = infer_split(model, config, manifest, "valid", torch.device("cpu"),
                               sample_indices=selected)
    selected_labels = labels[selected]
    sampled_means = torch.tensor([scores[selected][selected_labels == rank].mean(dtype=np.float64)
                                  for rank in range(10)])
    assert sampled_loss == pytest.approx(ranking_loss_from_means(sampled_means).item(), abs=1e-6)
    # Verify individual values rather than just aggregate coverage.
    from go_aicup.tasks.strength.shared.dataset import PackedBoardDataset
    dataset = PackedBoardDataset(directory / "valid_game.npy", directory / "valid_ply.npy", Path(config["paths"]["cache_dir"]))
    model.eval()
    with torch.no_grad():
        for index in (0, 73, len(scores) - 1):
            expected = model(torch.from_numpy(dataset[index]["planes"])[None].float()).item()
            assert scores[index] == pytest.approx(expected, abs=1e-6)
    dataset.close()
    assert set(manifest["sets"]) == {"train", "valid"}
    assert not list(directory.glob("*calibration*"))
    assert "cv_dir" not in config["paths"]


def test_resume_matches_uninterrupted_and_only_four_final_files(compact_data, tmp_path):
    full = clone_config(compact_data)
    full["paths"]["output_dir"] = str(tmp_path / "full")
    resumed = clone_config(compact_data)
    resumed["paths"]["output_dir"] = str(tmp_path / "resumed")
    train(full)
    checkpoint = train(resumed, max_steps=1)
    assert checkpoint.exists()
    train(resumed, resume=Path(resumed["paths"]["output_dir"]) / "last.json")
    expected_files = {"model.pt", "losses.npy", "train_scores.npy", "valid_scores.npy"}
    for config in (full, resumed):
        run = Path(config["paths"]["output_dir"])
        assert {path.name for path in run.iterdir()} == expected_files
        losses = np.load(run / "losses.npy", allow_pickle=False)
        assert losses.dtype == LOSS_DTYPE
        np.testing.assert_array_equal(losses["step"], [2, 4])
        payload, _, _ = load_saved_run(run)
        assert "optimizer" not in payload and payload["seed"] == config["seed"]
        result = evaluate_saved(run)
        assert set(result["questions"]["by_k"]) >= {"5", "10", "15", "20"}
        assert payload["cutpoints"]["num_training_games"] == 200
        assert payload["cutpoints"]["training_game_score"] == pytest.approx(result["train"]["games"]["score"])
        assert "equal weight per game" in payload["cutpoints"]["scope"]
    first, second = Path(full["paths"]["output_dir"]), Path(resumed["paths"]["output_dir"])
    for name in ("losses.npy", "train_scores.npy", "valid_scores.npy"):
        np.testing.assert_array_equal(np.load(first / name), np.load(second / name))
    model1 = torch.load(first / "model.pt", weights_only=False)["model"]
    model2 = torch.load(second / "model.pt", weights_only=False)["model"]
    for name in model1:
        assert torch.equal(model1[name], model2[name])
    import subprocess
    import sys
    plots = tmp_path / "plots"
    subprocess.run([sys.executable, "-m", "scripts.strength.plot_rank_v1", "--run-dir", str(first),
                    "--output-dir", str(plots)], check=True, capture_output=True, text=True)
    assert {path.name for path in plots.iterdir()} == {
        "loss_vs_step.png", "confusion_matrix.png", "score_distributions_cutpoints.png"}
    # A changed score is rejected before evaluation can use wrong data.
    values = np.load(second / "valid_scores.npy")
    values[0] += 1
    np.save(second / "valid_scores.npy", values)
    with pytest.raises(ValueError, match="integrity"):
        load_saved_run(second)


def test_interrupted_evaluation_can_resume(compact_data, tmp_path, monkeypatch):
    import go_aicup.tasks.strength.v1.train as training
    config = clone_config(compact_data)
    config["paths"]["output_dir"] = str(tmp_path / "interrupted")
    real_infer = training.infer_split
    calls = [0]
    def interrupt(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise InterruptedError("synthetic interruption")
        return real_infer(*args, **kwargs)
    monkeypatch.setattr(training, "infer_split", interrupt)
    path = train(config)
    payload = torch.load(path, weights_only=False)
    assert payload["state"]["pending_evaluation"]
    assert payload["state"]["optimizer_steps"] == 2
    monkeypatch.setattr(training, "infer_split", real_infer)
    run = Path(config["paths"]["output_dir"])
    train(config, resume=run / "last.json")
    np.testing.assert_array_equal(np.load(run / "losses.npy")["step"], [2, 4])


def _distributed_inference_worker(rank, config, manifest, model_state, rendezvous, output):
    import torch.distributed as dist
    from go_aicup.tasks.strength.v1.model import StrengthResNet
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = StrengthResNet(**config["model"])
        model.load_state_dict(model_state)
        value = infer_split(model, config, manifest, "valid", torch.device("cpu"), rank, 2, output=Path(output))
        if rank == 0:
            Path(output + ".loss").write_text(str(value))
    finally:
        dist.destroy_process_group()


def test_two_process_inference_matches_serial_with_empty_tail_rank(compact_data, tmp_path):
    from go_aicup.tasks.strength.v1.model import StrengthResNet
    config = clone_config(compact_data)
    manifest = load_dataset_manifest(config)
    source = Path(config["paths"]["dataset_dir"])
    config["paths"]["dataset_dir"] = str(tmp_path)
    # 401 samples with batch 100: final rank zero has one, rank one has none.
    for kind in ("game", "ply"):
        array = np.load(source / f"valid_{kind}.npy")
        np.save(tmp_path / f"valid_{kind}.npy", np.r_[array, array[:1]])
    first_game = int(np.load(tmp_path / "valid_game.npy")[0])
    rank_id = int(pd.read_parquet(source / "strength_v1_games.parquet").iloc[first_game].rank_id)
    manifest["sets"]["valid"]["num_samples"] += 1
    manifest["sets"]["valid"]["per_rank_samples"][rank_id] += 1
    config["evaluation"]["global_batch_size"] = 100
    model = StrengthResNet(**config["model"])
    serial = tmp_path / "serial.npy"
    expected = infer_split(model, config, manifest, "valid", torch.device("cpu"), output=serial)
    output = str(tmp_path / "parallel.npy")
    torch.multiprocessing.spawn(_distributed_inference_worker,
        args=(config, manifest, model.state_dict(), (tmp_path / "gloo_init").as_uri(), output),
        nprocs=2, join=True)
    assert float(Path(output + ".loss").read_text()) == pytest.approx(expected, abs=1e-6)
    np.testing.assert_allclose(np.load(output), np.load(serial), atol=1e-6, rtol=1e-5)


def test_final_calibration_interruption_keeps_recoverable_outputs(compact_data, tmp_path, monkeypatch):
    import go_aicup.tasks.strength.v1.evaluate as evaluation
    config = clone_config(compact_data)
    config["paths"]["output_dir"] = str(tmp_path / "final_interrupted")
    real_fit = evaluation.fit_cutpoints
    def interrupt(*args, **kwargs):
        raise InterruptedError("interrupted during final cutpoints")
    monkeypatch.setattr(evaluation, "fit_cutpoints", interrupt)
    checkpoint = train(config)
    assert checkpoint.exists()
    run = Path(config["paths"]["output_dir"])
    assert (run / "train_scores.npy").exists()
    assert not (run / "model.pt").exists()
    assert torch.load(checkpoint, weights_only=False)["state"]["epoch"] == 1
    monkeypatch.setattr(evaluation, "fit_cutpoints", real_fit)
    train(config, resume=run / "last.json")
    assert {p.name for p in run.iterdir()} == {"model.pt", "losses.npy", "train_scores.npy", "valid_scores.npy"}


def test_resume_legacy_checkpoint_with_changed_code_preserves_updates(compact_data, tmp_path):
    from go_aicup.tasks.strength.shared.runtime import file_hash
    full = clone_config(compact_data)
    full["paths"]["output_dir"] = str(tmp_path / "reference")
    resumed = clone_config(compact_data)
    resumed["paths"]["output_dir"] = str(tmp_path / "legacy")
    train(full)
    checkpoint = train(resumed, max_steps=2)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["state"].pop("last_batch_loss")
    saved["state"]["pending_evaluation"] = True
    saved["state"]["loss_rows"] = [(2, 99.0, 88.0)]
    saved["code_sha256"] = {"old_code.py": "different"}
    torch.save(saved, checkpoint)
    run = Path(resumed["paths"]["output_dir"])
    record = json.loads((run / "last.json").read_text())
    record["sha256"] = file_hash(checkpoint)
    for marker in (run / "last.json", checkpoint.with_suffix(".complete.json")):
        marker.write_text(json.dumps(record))
    train(resumed, resume=run / "last.json")
    archive = json.loads((tmp_path / "legacy.legacy_full_losses.json").read_text())
    assert archive["loss_rows"] == [[2, 99.0, 88.0]]
    np.testing.assert_array_equal(np.load(run / "losses.npy")["step"], [4])
    expected = torch.load(Path(full["paths"]["output_dir"]) / "model.pt", weights_only=False)["model"]
    actual = torch.load(run / "model.pt", weights_only=False)["model"]
    for name in expected:
        assert torch.equal(actual[name], expected[name])
