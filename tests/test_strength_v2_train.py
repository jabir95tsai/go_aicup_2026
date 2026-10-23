"""V2 trainer: end-to-end artifacts, exact resume and two-process Gloo smoke."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from go_aicup.tasks.strength.shared.runtime import load_dataset_manifest, model_state_hash
from go_aicup.tasks.strength.v2.aggregate import CATEGORY_DTYPE, monitor_metrics
from go_aicup.tasks.strength.v2.infer import infer_games
from go_aicup.tasks.strength.v2.model import StrengthResNet
from go_aicup.tasks.strength.v2.train import ARTIFACTS, LOSS_DTYPE, train

RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")


def clone(config, output):
    config = json.loads(json.dumps(config))
    config["paths"]["output_dir"] = str(output)
    return config


def test_end_to_end_artifacts_and_monitor_rows(v2_train_data, tmp_path):
    config = clone(v2_train_data, tmp_path / "run")
    manifest = load_dataset_manifest(config)
    total = -(-manifest["sets"]["train"]["num_samples"] // 512)
    assert total >= 6
    path = train(config)
    run = Path(config["paths"]["output_dir"])
    assert path == run / "model.pt"
    assert sorted(p.name for p in run.iterdir()) == sorted(["model.pt", *ARTIFACTS])
    losses = np.load(run / "losses.npy", allow_pickle=False)
    assert losses.dtype == LOSS_DTYPE
    expected_steps = [step for step in range(4, total, 4)] + [total]
    np.testing.assert_array_equal(losses["step"], expected_steps)
    for name in LOSS_DTYPE.names[1:]:
        assert np.isfinite(losses[name]).all()
    payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    assert payload["optimizer_steps"] == total and payload["epochs"] == 1
    assert payload["samples_seen"] == manifest["sets"]["train"]["num_samples"]
    assert payload["class_names"] == list(RANKS)
    assert payload["loss_columns"] == list(LOSS_DTYPE.names)
    assert payload["player_overlap"]["valid_unseen_overlap_fraction"] == 0.0
    assert [row[0] for row in payload["timings"]["monitor_seconds"]] == expected_steps
    model = StrengthResNet(**config["model"])
    model.load_state_dict(payload["model"])
    assert model_state_hash(model) == payload["model_state_sha256"]
    # Saved categories and the final monitor row reproduce from model.pt.
    for prefix, name in (("valid_unseen", "validation_unseen_category.npy"),
                         ("valid_seen", "validation_seen_category.npy")):
        saved = np.load(run / name, allow_pickle=False)
        assert saved.dtype == CATEGORY_DTYPE
        assert len(saved) == manifest["sets"][prefix]["num_games"]
        recomputed = infer_games(model, config, prefix, torch.device("cpu"))
        np.testing.assert_array_equal(saved["predicted_rank_id"], recomputed["predicted_rank_id"])
        np.testing.assert_allclose(saved["confidence_sum"], recomputed["confidence_sum"], rtol=1e-9)
    for prefix, loss, score in (("monitor_unseen", "unseen_loss", "unseen_game_score"),
                                ("monitor_seen", "seen_loss", "seen_game_score")):
        metrics = monitor_metrics(infer_games(model, config, prefix, torch.device("cpu")))
        assert losses[loss][-1] == pytest.approx(metrics["loss"], rel=1e-9)
        assert losses[score][-1] == pytest.approx(metrics["game_score"], rel=1e-9)
    with pytest.raises(ValueError, match="nonempty"):
        train(config)


def test_epoch_snapshots_survive_cleanup(v2_train_data, tmp_path):
    config = clone(v2_train_data, tmp_path / "run")
    config["training"]["epochs"] = 2
    config["training"]["epoch_snapshots"] = True
    train(config)
    run = Path(config["paths"]["output_dir"])
    snapshots = sorted(p.name for p in (run / "snapshots").iterdir())
    assert snapshots == ["epoch_0001.pt", "epoch_0002.pt"]
    payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    last = torch.load(run / "snapshots" / "epoch_0002.pt", map_location="cpu")
    assert last.keys() == payload["model"].keys()
    assert all(torch.equal(last[name], payload["model"][name]) for name in last)
    first = torch.load(run / "snapshots" / "epoch_0001.pt", map_location="cpu")
    assert any(not torch.equal(first[name], last[name]) for name in last)


def test_resume_matches_uninterrupted(v2_train_data, tmp_path):
    full = clone(v2_train_data, tmp_path / "full")
    train(full)
    partial = clone(v2_train_data, tmp_path / "partial")
    stopped = train(partial, max_steps=5)  # After the step-4 monitor.
    assert stopped.parent.name == "checkpoints"
    run = Path(partial["paths"]["output_dir"])
    np.testing.assert_array_equal(np.load(run / "losses.npy")["step"], [4])
    train(partial, resume=run / "last.json")
    first, second = (torch.load(Path(c["paths"]["output_dir"]) / "model.pt", map_location="cpu",
                                weights_only=False) for c in (full, partial))
    assert first["model_state_sha256"] == second["model_state_sha256"]
    for name in ARTIFACTS:
        a, b = (np.load(Path(c["paths"]["output_dir"]) / name, allow_pickle=False) for c in (full, partial))
        assert a.tobytes() == b.tobytes(), name


def _ddp_train_worker(rank, config, port):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    train(config)


def test_two_process_gloo_training_completes(v2_train_data, tmp_path):
    import socket
    config = clone(v2_train_data, tmp_path / "ddp")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    torch.multiprocessing.spawn(_ddp_train_worker, args=(config, port), nprocs=2, join=True)
    run = Path(config["paths"]["output_dir"])
    assert sorted(p.name for p in run.iterdir()) == sorted(["model.pt", *ARTIFACTS])
    payload = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    manifest = load_dataset_manifest(config)
    assert payload["samples_seen"] == manifest["sets"]["train"]["num_samples"]
    assert payload["environment"]["world_size"] == 2
    model = StrengthResNet(**config["model"])
    model.load_state_dict(payload["model"])
    saved = np.load(run / "validation_unseen_category.npy", allow_pickle=False)
    recomputed = infer_games(model, config, "valid_unseen", torch.device("cpu"))
    np.testing.assert_array_equal(saved["predicted_rank_id"], recomputed["predicted_rank_id"])
    np.testing.assert_allclose(saved["confidence_sum"], recomputed["confidence_sum"], rtol=1e-6, atol=1e-6)


def _independent_equal_k(category, questions):
    """Plain-loop re-implementation of the official question rule."""
    rows = {int(game): row for row, game in enumerate(category["game_index"])}
    scores = {k: [] for k in (5, 10, 15, 20)}
    for question in questions.itertuples():
        p = np.mean([category["confidence_sum"][rows[g]] / category["num_moves"][rows[g]]
                     for g in question.games], axis=0)
        gain = [p[r] + ((p[r - 1] if r > 0 else 0) + (p[r + 1] if r < 9 else 0)) / np.e for r in range(10)]
        guess = int(np.argmax(gain))
        error = abs(guess - question.rank_id)
        scores[question.k].append(1.0 if error == 0 else (1 / np.e if error == 1 else 0.0))
    return float(np.mean([np.mean(values) for values in scores.values()]))


def test_evaluation_reports_and_figures(v2_train_data, tmp_path, capsys):
    from go_aicup.tasks.strength.v2.evaluate import OFFICIAL, evaluate
    config = clone(v2_train_data, tmp_path / "eval")
    train(config)
    run = Path(config["paths"]["output_dir"])
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    result = evaluate(run)
    assert "OFFICIAL" in capsys.readouterr().out
    dataset = Path(config["paths"]["dataset_dir"])
    official = {}
    for name in ("unseen", "seen"):
        category = np.load(run / f"validation_{name}_category.npy", allow_pickle=False)
        questions = pd.read_parquet(dataset / f"strength_v2_valid_{name}_questions.parquet")
        official[name] = _independent_equal_k(category, questions)
        assert result[name]["questions"][OFFICIAL]["equal_k"] == pytest.approx(official[name], abs=1e-12)
        game = result[name]["game"]
        assert game["games"] == len(category) == game["confusion"].sum()
        expected = np.where(category["predicted_rank_id"] == category["rank_id"], 1.0,
                            np.where(np.abs(category["predicted_rank_id"].astype(int) - category["rank_id"]) == 1,
                                     1 / np.e, 0.0)).mean()
        assert game["score"] == pytest.approx(expected)
        assert game["off_by_one"] + game["accuracy"] + game["off_by_two_plus"] == pytest.approx(1.0)
    assert result["verdict"]["gap"] == pytest.approx(official["seen"] - official["unseen"])
    assert result["verdict"]["passes_baseline"] == (official["unseen"] >= 0.4631)
    assert sorted(p.name for p in (run / "plot").iterdir()) == sorted(
        f"{name}.png" for name in ("training_curves", "game_confusion_unseen", "game_confusion_seen",
                                   "game_confidence_unseen", "game_confidence_seen", "game_by_rank",
                                   "question_by_k", "question_confusion_score_rank",
                                   "question_confusion_argmax", "question_aggregation"))
    assert before == {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}
    again = evaluate(run, plots=False)
    assert again["verdict"] == result["verdict"]
