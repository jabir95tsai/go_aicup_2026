import numpy as np
import pandas as pd
import pytest
from go_aicup.data.cv import make_large_policy_tables


def source():
    rows=[]
    for rank in range(10):
        for player in range(8):
            for color in (0,1):
                for j in range(25+player):
                    rows.append(dict(game_index=len(rows),rank_id=rank,player_id=f'{rank}:{player}',
                                     player_color=color,handicap=player%2,length=100+j))
    return pd.DataFrame(rows)


def options():
    return dict(train_games_per_rank=80,monitor_players_per_rank=1,dev_players_per_rank=1,
                train_questions=160,monitor_questions=40,dev_questions=40)


def test_split_is_deterministic_disjoint_balanced_and_covers_all_games():
    m=source();excluded={f'{r}:0' for r in range(10)}
    games,questions=make_large_policy_tables(m,excluded,**options())
    g2,q2=make_large_policy_tables(m.sample(frac=1,random_state=4),excluded,**options())
    pd.testing.assert_frame_equal(games,g2)
    assert not games.groupby('player_id').role.nunique().gt(1).any()
    assert not set(games.loc[games.role.ne('train'),'player_id']) & excluded
    for role,q in questions.items():
        pd.testing.assert_frame_equal(q,q2[role])
        selected=games[games.role.eq(role)].set_index('game_index')
        assert set(np.concatenate(q.games.to_numpy())) == set(selected.index)
        assert q.groupby(['rank_id','k']).size().nunique()==1
        for item in q.itertuples():
            members=selected.loc[item.games]
            assert len(set(item.games))==item.k
            assert members.player_id.nunique()==members.player_color.nunique()==members.rank_id.nunique()==1
            assert members.rank_id.iloc[0]==item.rank_id
    assert games[games.role.eq('train')].groupby('rank_id').size().eq(80).all()


def test_ineligible_sources_fail():
    with pytest.raises(ValueError,match='holdout'):
        make_large_policy_tables(source(),source().player_id,**options())
    m=source();m.loc[len(m)]=m.iloc[0]
    with pytest.raises(ValueError,match='duplicate'):
        make_large_policy_tables(m,[],**options())
