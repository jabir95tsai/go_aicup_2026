"""Unit tests for the W1 foundation: feature policy, CV invariants, metrics,
question melting. Uses a small synthetic metadata table, no real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from go_aicup import metrics
from go_aicup.data import cv, preprocess
from go_aicup.data import sgf as sgflib
from go_aicup.data.store import GameStore
from go_aicup.features import feature_policy as fp


def synthetic_meta(n_players: int = 1_700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_players):
        rank = p % 10
        n = int(rng.integers(20, 300)) if p < 1_200 else 260
        colors = np.where(np.arange(n) % 2 == 0, "B", "W")
        for g in range(n):
            rows.append((f"p{p:05d}", rank, colors[g]))
    meta = pd.DataFrame(rows, columns=["player_id", "rank_id", "color"])
    return meta


@pytest.fixture(scope="module")
def meta():
    return synthetic_meta()


# ---------------------------------------------------------------- policy

def test_banned_columns_caught():
    with pytest.raises(fp.FeaturePolicyError):
        fp.check_feature_columns(["n_moves", "player_id"])
    with pytest.raises(fp.FeaturePolicyError):
        fp.check_feature_columns(["game_id_num"])
    with pytest.raises(fp.FeaturePolicyError):
        fp.check_feature_columns(["rank_id"])
    fp.check_feature_columns(["n_moves", "first20_corner_frac", "k"])


def test_sgf_property_gate():
    assert fp.check_sgf_properties({"SZ"}, {"SZ"}) == ["SZ"]
    assert fp.check_sgf_properties({"SZ", "KM"}, {"SZ", "KM"}) == ["SZ"]
    with pytest.raises(fp.FeaturePolicyError):
        fp.check_sgf_properties({"SZ"}, {"SZ", "HA"})


def test_augment_policy():
    fp.check_augment("rank", d4=True)
    fp.check_augment("pid", d4=False)
    with pytest.raises(fp.FeaturePolicyError):
        fp.check_augment("pid", d4=True)


# ---------------------------------------------------------------- cv

def test_rank_folds_disjoint_and_stratified(meta):
    players = cv.make_rank_folds(meta)
    assert set(players["fold"]) == {-1, 0, 1, 2, 3, 4}
    assert players["player_id"].is_unique
    per_cell = players.pivot_table(index="rank_id", columns="fold",
                                   values="player_id", aggfunc="count")
    assert per_cell.notna().all().all() and (per_cell > 0).all().all()
    frac = (players["fold"] == -1).mean()
    assert 0.08 < frac < 0.12


def test_rank_questions_shape(meta):
    players = cv.make_rank_folds(meta)
    ids = players.loc[players["fold"] == 0, "player_id"]
    q = cv.make_rank_questions(meta, ids, 2_000, seed=1)
    assert len(q) == 2_000
    assert q["k"].value_counts().to_dict() == {5: 500, 10: 500, 15: 500, 20: 500}
    assert set(q["player_id"]) <= set(ids)
    for row in q.itertuples():
        games = meta.loc[row.games]
        assert (games["player_id"] == row.player_id).all()
        assert (games["color"] == row.color).all()
        assert len(set(row.games)) == row.k


def test_pid_splits(meta):
    gallery, questions = cv.make_pid_splits(meta, n_splits=4)
    # 100 players with 260 games (130 black) + some random ones qualify
    for split, group in questions.groupby("split"):
        assert len(group) == cv.POOL_SIZE
        assert group["k"].value_counts().to_dict() == {5: 100, 10: 100, 15: 100, 20: 100}
        g = gallery[gallery["split"] == split]
        assert g.groupby("player_id").size().eq(cv.GALLERY).all()
        assert (meta.loc[g["game"], "color"] == "B").all()
    cv.assert_pid_disjoint(gallery, questions)
    held = set(gallery.loc[gallery["split"] == 0, "player_id"])
    assert not held & set(cv.pid_train_players(meta, gallery, 0))


# ---------------------------------------------------------------- metrics

def test_rank_report_buckets():
    k = np.array([5, 5, 10, 10, 15, 15, 20, 20])
    true = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    pred = np.array([0, 2, 2, 3, 4, 5, 6, 9])
    rep = metrics.rank_report(pred, true, k)
    assert list(rep.index) == ["5", "10", "15", "20", "all"]
    assert rep.loc["5", "acc"] == 0.5
    assert rep.loc["all", "mae"] == pytest.approx(3 / 8)
    # exact x6, off-by-one x1, off-by-two x1
    assert rep.loc["all", "score"] == pytest.approx((6 + np.exp(-1)) / 8)


def test_pid_report():
    ranked = np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5], [0, 0, 0, 0, 0]])
    true = np.array([1, 5, 3])
    rep = metrics.pid_report(ranked, true, np.array([5, 10, 20]))
    assert rep.loc["all", "top1"] == pytest.approx(1 / 3)
    assert rep.loc["all", "top5"] == pytest.approx(2 / 3)
    assert rep.loc["all", "map5"] == pytest.approx((1 + 0.2) / 3)
    assert rep.loc["all", "score"] == pytest.approx((1 + np.exp(-4)) / 3)


def test_decoders():
    proba = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    assert metrics.median_rank(proba).tolist() == [2, 0]
    assert metrics.expected_rank(proba).tolist() == [2, 0]


def test_score_rank_decode():
    # argmax is rank 0 (0.40), but rank 1 collects 0.35 + (0.40 + 0.25) / e = 0.589
    proba = np.array([[0.40, 0.35, 0.25, 0, 0, 0, 0, 0, 0, 0]])
    assert metrics.score_rank(proba).tolist() == [1]
    # a peaked distribution keeps its argmax
    assert metrics.score_rank(np.eye(10)[[3]]).tolist() == [3]


# ---------------------------------------------------------------- preprocess

def test_melt_questions_and_parse():
    game = "(;SZ[19];B[pd];W[dp];B[qp])"
    frame = pd.DataFrame({
        "question_id": ["q1", "q2"], "num_games": [2, 1],
        **{f"sgf_{i}": [game if i <= 2 else None, game if i == 1 else None]
           for i in range(1, 21)},
        **{f"color_{i}": ["B", "W"] for i in range(1, 21)},
    })
    meta, contents = preprocess.melt_questions(frame)
    assert len(meta) == 3 and len(contents) == 3
    moves, colors, meta = preprocess._to_store(meta, contents, workers=1)
    assert meta["length"].tolist() == [3, 3, 3]
    assert moves[:3].tolist() == [sgflib.point_to_index(p) for p in ("pd", "dp", "qp")]
    assert meta["alternating"].all()


@pytest.mark.parametrize("color", ["B", "W", "X", None])
def test_store_player_perspective_requires_color(tmp_path, color):
    frame = pd.DataFrame({"color": [color]}) if color is not None else pd.DataFrame(index=[0])
    moves, colors, meta = preprocess._to_store(frame, ["(;SZ[19];B[pd];W[dp];B[qp])"], workers=1)
    preprocess._write_store("sample", moves, colors, meta, tmp_path)
    if color not in ("B", "W"):
        with pytest.raises(ValueError, match="B/W player color"):
            GameStore("sample", tmp_path)
    else:
        store = GameStore("sample", tmp_path)
        np.testing.assert_array_equal(store.perspective(0)[1], [0, 1, 0] if color == "B" else [1, 0, 1])
