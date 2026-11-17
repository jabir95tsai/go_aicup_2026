import pandas as pd
import pytest

from go_aicup.data.cv import make_cached_policy_games, make_rank_questions


def test_cache_first_split_and_questions():
    rows = []
    for rank in range(10):
        for player in range(4):
            for color, count in [(0, 25), (1, 20)]:
                for ply in range(count):
                    rows.append(dict(game_index=len(rows), rank_id=rank,
                                     player_id=rank * 4 + player, player_color=color,
                                     handicap=0, length=100))
    meta = pd.DataFrame(rows)
    # Prefer white because the reusable pack contains more white games.
    cached = meta[meta.player_color.eq(1)].groupby('player_id').head(12).copy()
    dev = cached[cached.player_id.mod(4).eq(3)]
    cached = cached[~cached.player_id.mod(4).eq(3)]
    games = make_cached_policy_games(meta, cached, dev, train_per_rank=1, monitor_per_rank=1, dev_per_rank=1)
    pd.testing.assert_frame_equal(games, make_cached_policy_games(
        meta, cached, dev, train_per_rank=1, monitor_per_rank=1, dev_per_rank=1))
    assert games.player_color.eq(1).all()
    assert games.reuse_pack.sum() == 30 * 12
    assert games.groupby('player_id').size().eq(20).all()
    assert games.groupby('player_id').role.nunique().eq(1).all()
    assert games.groupby('role').player_id.nunique().to_dict() == dict(dev=10, monitor=10, train=10)
    for role in ['train', 'monitor', 'dev']:
        part = games[games.role.eq(role)].set_index('game_index', drop=False)
        qs = make_rank_questions(part, part.player_id.unique(), 80)
        assert qs.groupby('k').size().to_dict() == {5: 20, 10: 20, 15: 20, 20: 20}
        for q in qs.itertuples():
            selected = part.loc[q.games]
            assert selected.player_id.nunique() == selected.player_color.nunique() == 1
            assert len(selected) == len(set(q.games)) == q.k
    bad = cached.copy()
    bad.loc[bad.index[0], 'handicap'] = 1
    with pytest.raises(ValueError, match='mismatch'):
        make_cached_policy_games(meta, bad, dev, train_per_rank=1, monitor_per_rank=1, dev_per_rank=1)
    with pytest.raises(ValueError, match='insufficient'):
        make_cached_policy_games(meta, cached, dev, train_per_rank=3, monitor_per_rank=1)

    with pytest.raises(ValueError, match='overlap'):
        make_cached_policy_games(meta, cached, cached, train_per_rank=1, monitor_per_rank=1)
    assert set(games.loc[games.role.eq('dev'), 'player_id']) == set(dev.player_id)
