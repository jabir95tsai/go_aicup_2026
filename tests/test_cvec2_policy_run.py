import numpy as np
import pandas as pd
from scipy.special import log_softmax
import torch

from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength import humansl
from go_aicup.tasks.strength.cvec2_policy import played
from go_aicup.tasks.strength.cvec2_policy.played import H_COLUMNS, means_and_offsets
from go_aicup.tasks.strength.cvec2_policy.run import stage1_features
from go_aicup.tasks.strength.cvec2_policy.train import FrozenB2
from scripts.strength.rank_more_players import PriorMatched


def test_stage1_transform_is_mean_of_move_log_softmax():
    rng = np.random.default_rng(17)
    h, c = rng.normal(size=(12, 15)), rng.normal(size=(12, 10))
    slot = np.repeat(np.arange(4), 3)
    h9 = H_COLUMNS[:9]
    h15_mean, _ = means_and_offsets(h, slot, 2)
    _, hnorm = means_and_offsets(h[:, h9], slot, 2)
    c_mean, cnorm = means_and_offsets(c, slot, 2)
    base = np.column_stack((h15_mean, c_mean, [[0, 1], [1, 0]]))
    arms = stage1_features(base, H_COLUMNS, hnorm, cnorm)
    np.testing.assert_array_equal(arms["A"], base)
    np.testing.assert_array_equal(arms["B"][:, :18], base[:, H_COLUMNS])
    np.testing.assert_array_equal(arms["B"][:, 18:], arms["C"][:, 18:])
    np.testing.assert_allclose(arms["C"][:, :18], log_softmax(h[:, h9], axis=1).reshape(4, 3, 9)
                               .mean(1).reshape(2, 18))
    np.testing.assert_allclose(arms["C"][:, 18:38], log_softmax(c, axis=1).reshape(4, 3, 10)
                               .mean(1).reshape(2, 20))


def test_draws_and_frozen_b2_match_original_contract(monkeypatch):
    monkeypatch.setattr(humansl, "SYNTHETIC", 300)
    rng = np.random.default_rng(37)
    games = pd.DataFrame({"player_id": np.repeat([str(p) for p in range(20)], 23),
                          "rank_id": np.repeat(np.arange(10), 46)})
    x = rng.normal(size=(len(games), 40))
    prior = {k: np.arange(1., 11.) / 55 for k in K_BUCKETS}
    model = PriorMatched(prior).fit(games, pd.DataFrame(), x)
    draws_rng = np.random.default_rng(humansl.SEED)
    draws = list(model.synthetic_rows(games, games.rank_id.to_numpy(), humansl.SEED))
    grouped = list(games.groupby("player_id").indices.values())
    for k, rows, labels in draws:
        original_ranks = draws_rng.choice(10, humansl.SYNTHETIC, p=prior[k])
        original_rows = []
        for rank in original_ranks:
            pool = [g for g in grouped if games.rank_id.iloc[g[0]] == rank]
            original_rows.append(draws_rng.choice(pool[draws_rng.integers(len(pool))], k, replace=False))
        np.testing.assert_array_equal(rows, original_rows)
        np.testing.assert_array_equal(labels, original_ranks)
    questions = np.stack([x[rows[0]].mean(0) for _, rows, _ in draws])
    logits = FrozenB2(model)(torch.as_tensor(questions, dtype=torch.float32), np.asarray(K_BUCKETS)).numpy()
    expected = np.vstack([model.per_k[k][1].decision_function(model.per_k[k][0].transform(questions[i:i+1]))
                          for i, k in enumerate(K_BUCKETS)])
    np.testing.assert_allclose(logits, expected, atol=2e-5, rtol=1e-5)


def test_h_backfill_dispatch_keeps_games_and_canonical_rows_aligned(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    games = pd.DataFrame({"game_index": [70, 24, 12, 98, 51],
                          "player_color": [0, 1, 1, 0, 1], "handicap": [0, 1, 0, 0, 0]},
                         index=[4, 11, 30, 37, 42])
    canon = np.arange(5 * 18).reshape(5, 18)
    shard = tmp_path / "cached" / "shard_0"
    shard.mkdir(parents=True)
    (shard / "meta.json").touch()
    games.iloc[[1]].to_parquet(shard / "games.parquet", index=False)
    monkeypatch.setattr(played, "H_CACHE", [shard.parent])
    monkeypatch.setattr(played, "h_means", lambda *_: (canon[[1]], np.array([[24, -24]])))
    monkeypatch.setattr(played, "ProcessPoolExecutor", lambda workers, **_: ThreadPoolExecutor(workers))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")

    def backfill(device, table, raw, out):
        rows = [0, 3] if device == "3" else [2, 4]
        ids = table.game_index.to_numpy()
        np.testing.assert_array_equal(ids, games.iloc[rows].game_index)
        np.testing.assert_array_equal(raw, canon[rows])
        path = out.with_suffix(".npz")
        np.savez(path, games=ids, raw=raw, offsets=np.column_stack((ids, -ids)))
        return [path]

    monkeypatch.setattr(played, "_h_worker", backfill)
    offsets, error, reused = played.h_offsets(games, canon, tmp_path)
    np.testing.assert_array_equal(offsets[:, 0], games.game_index)
    np.testing.assert_array_equal(offsets[:, 1], -games.game_index)
    assert error == 0 and reused == 1


def test_h_backfill_resume_reuses_saved_parts_and_resplits_the_rest(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    games = pd.DataFrame({"game_index": [70, 24, 12, 98, 51, 33],
                          "player_color": 0, "handicap": 0})
    canon = np.arange(6 * 18, dtype=float).reshape(6, 18)
    monkeypatch.setattr(played, "H_CACHE", [])
    monkeypatch.setattr(played, "ProcessPoolExecutor", lambda workers, **_: ThreadPoolExecutor(workers))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    old = tmp_path / "h_backfill_06"
    old.mkdir()
    np.savez(old / "part_000000.npz", games=[70, 98], raw=canon[[0, 3]], offsets=[[70, -70], [98, -98]])
    np.savez(old / "part_002048.partial.npz", games=[24], raw=canon[[1]], offsets=[[0, 0]])
    seen = []

    def part(table, raw, path):
        seen.append(table.game_index.tolist())
        if 33 in table.game_index.tolist():
            raise ValueError("boom")
        np.savez(path, games=table.game_index, raw=raw, offsets=np.column_stack((table.game_index, -table.game_index)))

    monkeypatch.setattr(played, "_h_part", part)
    try:
        played.h_offsets(games, canon, tmp_path)
        raise AssertionError("expected failure")
    except RuntimeError as error:
        assert "boom" in str(error)
    assert sorted(seen) == [[12, 33], [24, 51]]
    assert len(list(tmp_path.glob("h_backfill_*_01/part_000000.error.json"))) == 1
    monkeypatch.setattr(played, "_h_part", lambda table, raw, path: np.savez(
        path, games=table.game_index, raw=raw, offsets=np.column_stack((table.game_index, -table.game_index))))
    (next(tmp_path.glob("h_backfill_*_01/part_000000.error.json"))).unlink()
    offsets, error, reused = played.h_offsets(games, canon, tmp_path)
    np.testing.assert_array_equal(offsets[:, 0], games.game_index)
    assert error == 0 and reused == 0


def test_export_test_offsets_align_shards_with_release_rows(monkeypatch, tmp_path):
    import json
    from go_aicup.features import humansl as hfeat
    from go_aicup.tasks.strength.cvec2_policy import export

    rng = np.random.default_rng(5)
    index = np.array([3, 8, 1, 6])
    hd1 = np.array([0, 1, 0, 0])
    moves = {g: rng.normal(size=(6, 15)) for g in index}
    cmoves = {g: rng.normal(size=(6, 10)) for g in index}
    slots = np.tile([0, 1], 3)
    x = np.zeros((4, 52))
    x[:, 50] = hd1
    for row, g in enumerate(index):
        h15, _ = means_and_offsets(moves[g], slots, 1)
        x[row, :30] = h15[0]
        x[row, 30:50] = means_and_offsets(cmoves[g], slots, 1)[0][0]

    def h_shard(path, ids, komi):
        path.mkdir(parents=True)
        pd.DataFrame({"game_index": ids, "player_color": 0}).to_parquet(path / "games.parquet")
        (path / "meta.json").write_text(json.dumps({"profiles": hfeat.PROFILES, "rules": hfeat.RULES,
                                                    "model": hfeat.MODEL.name, "komi": komi}))
        np.save(path / "offsets.npy", np.arange(len(ids) + 1) * 6)
        np.save(path / "colors.npy", np.tile(np.tile([0, 1], 3), len(ids)))
        np.save(path / "done.npy", np.ones((len(ids), 15), bool))
        np.save(path / "logp.npy", np.vstack([moves[g] for g in ids]))

    def c_shard(path, ids, komi):
        path.mkdir(parents=True)
        pd.DataFrame({"game_index": ids}).to_parquet(path / "games.parquet")
        (path / "meta.json").write_text(json.dumps({"per_move": True, "finetuned": None, "komi": komi,
                                                    "vectors": "x/cvec_v2/learn/vectors.pt"}))
        np.save(path / "done.npy", np.ones(len(ids), bool))
        np.save(path / "move_logp.npy", np.vstack([cmoves[g] for g in ids]))
        np.save(path / "move_slot.npy", np.concatenate([slots + 2 * i for i in range(len(ids))]))

    h_shard(tmp_path / "rel/rank_public/humansl/shard_0", [8, 1, 3, 6], 6.5)
    h_shard(tmp_path / "p3/h_test_public/shard_0", [8], .5)
    c_shard(tmp_path / "c/c_test_public_hd0/shard_0", [6, 3, 1], 6.5)
    c_shard(tmp_path / "c/c_test_public_hd1/shard_0", [8], .5)
    monkeypatch.setattr(export.prod, "RELEASE", tmp_path / "rel")
    monkeypatch.setattr(export.p3, "OUT", tmp_path / "p3")
    monkeypatch.setattr(export, "C_CACHE", tmp_path / "c")
    h_off, c_off, audit = export.test_offsets("public", x, index)
    assert audit["h_max_error"] < 1e-6 and audit["c_max_error"] < 1e-6
    arms = stage1_features(x, H_COLUMNS, h_off, c_off)
    for row, g in enumerate(index):
        np.testing.assert_allclose(arms["C"][row, :18], log_softmax(moves[g][:, H_COLUMNS[:9]], axis=1)
                                   .reshape(3, 2, 9).mean(0).reshape(18), atol=1e-6)
        np.testing.assert_allclose(arms["C"][row, 18:38], log_softmax(cmoves[g], axis=1)
                                   .reshape(3, 2, 10).mean(0).reshape(20), atol=1e-5)
