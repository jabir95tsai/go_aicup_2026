from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import numpy as np
import pandas as pd
import torch
import pytest
from go_aicup.tasks.strength.policy_large.cache import PolicyCache,Shard,split_games


def sample():
    lengths=np.array([2,3,2,4,3,2,4,2])
    games=pd.DataFrame(dict(length=lengths))
    bounds=np.r_[0,np.cumsum(lengths)]
    shards=[]
    counts=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        counts.append(np.bincount(np.arange(a,b)%2,minlength=2))
    for first,last in split_games(lengths,4):
        ids=torch.arange(int(bounds[first]),int(bounds[last]))
        lp=torch.zeros(len(ids),2,362,dtype=torch.float16);lp[:,0,0]=ids
        shards.append(Shard(first,last,lp,ids[:,None].float().expand(-1,2).clone(),
                            ids.to(torch.int16),(ids%2).to(torch.uint8)))
    return PolicyCache(games,shards,np.zeros((len(games),4),np.float32),np.array(counts))


def test_encoder_uses_release_board_root(tmp_path, monkeypatch):
    from go_aicup.tasks.strength.policy_large import cache
    from go_aicup.features import humansl_torch
    pd.DataFrame([dict(game_index=77, length=2, shard_id=0, local_offset=1)]).to_parquet(tmp_path/'games.parquet')
    np.save(tmp_path/'shard_00000_moves.npy', np.array([99, 12, 34], np.int16))
    np.save(tmp_path/'shard_00000_colors.npy', np.array([0, 0, 1], np.int8))
    monkeypatch.setattr(cache, '_LAST_HD', None)
    monkeypatch.setattr(humansl_torch, '_init_worker', lambda config, komi: None)
    def encode(item):
        np.testing.assert_array_equal(item[0], [12, 34])
        np.testing.assert_array_equal(item[1], [0, 1])
        return np.zeros((2, 1, 19, 19), np.uint8), np.zeros((2, 1)), np.array([12, 34])
    monkeypatch.setattr(humansl_torch, 'encode_game', encode)
    cache.init_encoder({}, tmp_path)
    bits, glob, target, side = cache.encode((77, 1, 1, 2))
    np.testing.assert_array_equal(target, [12, 34])
    np.testing.assert_array_equal(side, [1, 0])


def test_cross_shard_batches_duplicates_tail_and_counts():
    c=sample();v=c.for_device('cpu');rows=np.array([7,0,5,7,3])
    chunks=list(v.chunks(rows,5))
    assert all(len(x[0])<=5 for x in chunks)
    lp,t,q,slot=[torch.cat([x[i] for x in chunks]) for i in range(4)]
    expected=np.concatenate([np.arange(c.bounds[r],c.bounds[r+1]) for r in rows])
    owners=np.repeat(np.arange(len(rows)),np.diff(c.bounds)[rows])
    np.testing.assert_array_equal(lp[:,0,0],expected)
    np.testing.assert_array_equal(t,expected)
    np.testing.assert_array_equal(q[:,0],expected)
    np.testing.assert_array_equal(slot,2*owners+expected%2)
    np.testing.assert_array_equal(v.counts(rows),c.arrays['counts'][rows].flatten())
    assert list(v.chunks([],4))==[]
    assert v.cache is c.for_device('cpu').cache


@pytest.mark.parametrize('rows', [[-1],[8],[1.5],[[1]]])
def test_invalid_rows(rows):
    with pytest.raises((ValueError,IndexError)):
        list(sample().for_device('cpu').chunks(rows,5))


def test_shards_balance_positions():
    lengths=np.array([1]*8+[20]*4)
    ranges=split_games(lengths,4)
    loads=[lengths[a:b].sum() for a,b in ranges]
    assert max(loads)-min(loads)<=max(lengths)


def test_teacher_forward_serializes_calls_and_initialization():
    from go_aicup.tasks.strength.policy_large.cache import _INIT_LOCK, teacher_forward
    entered = threading.Event()
    release = threading.Event()
    def forward(value):
        assert _INIT_LOCK.locked()
        entered.set()
        assert release.wait(5)
        return value
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(teacher_forward, forward, 1)
        try:
            assert entered.wait(5)
            assert not _INIT_LOCK.acquire(blocking=False)
            second = pool.submit(teacher_forward, lambda value: value, 2)
            with pytest.raises(TimeoutError):
                second.result(timeout=.05)
        finally:
            release.set()
        assert first.result(timeout=5) == 1
        assert second.result(timeout=5) == 2
    def failing_forward():
        raise ValueError('compile failed')
    with pytest.raises(ValueError, match='compile failed'):
        teacher_forward(failing_forward)
    assert not _INIT_LOCK.locked()
    assert teacher_forward(lambda: 3) == 3
