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
