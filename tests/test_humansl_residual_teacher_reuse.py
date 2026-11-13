"""SE3 provenance, partial-game import, eight-shard resume and writer exclusion."""
import json
from types import SimpleNamespace
from pathlib import Path
import threading
import numpy as np
import pandas as pd
import pytest
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.tasks.strength.humansl_residual_b2 import teacher
from go_aicup.tasks.strength.humansl_residual_b2.teacher_reuse import SE3Values


def source(tmp_path):
    cache=tmp_path/'cache'; cache.mkdir()
    (cache/'games.parquet').write_bytes(b'canonical')
    write_json(cache/'manifest.json',dict(games=dict(sha256=file_hash(cache/'games.parquet'))))
    d=tmp_path/'prepared';d.mkdir()
    pd.DataFrame([dict(game_index=9,rank='D',game_id='g',player_color=1,length=34)]).to_parquet(d/'games.parquet')
    for name, array in [('train_game',[9]),('train_offsets',[0,2]),('train_ply',[31,33])]:
        np.save(d/f'{name}.npy',np.asarray(array))
    identity=dict(cache_manifest_sha256=file_hash(cache/'manifest.json'))
    write_json(d/'manifest.json',dict(identity=identity,build_id=object_hash(identity),cache_dir=str(cache),
        files={p.name:file_hash(p) for p in d.iterdir()}))
    v=tmp_path/'value.npy';np.save(v,np.array([-.25,.75],np.float32))
    t=tmp_path/'training.json';write_json(t,dict(prepared_manifest_sha256=file_hash(d/'manifest.json'),value_sha256=file_hash(v)))
    old=dict(engine_sha256='e',model_sha256='m',engine_config_sha256='c',source_manifest_sha256=file_hash(d/'manifest.json'),
        source_build_id=object_hash(identity),split='train',num_shards=8,limit_games=None,rules='japanese',komi=6.5,
        visits=1,winrate_perspective='BLACK',value_perspective='designated_player',ignorePreRootHistory=False,
        target='per_effective_pre_action_position')
    r=tmp_path/'recovery.json';write_json(r,dict(original_manifests={str(i):dict(shard=i,identity=old,positions=2 if i==0 else 0) for i in range(8)}))
    options=dict(dataset_dir=str(d),value_path=str(v),recovery_path=str(r),training_identity_path=str(t))
    current=dict(engine_sha256='e',model_sha256='m',config_sha256='c',source_files={str(cache/'games.parquet'):file_hash(cache/'games.parquet')})
    return options,current


def test_source_alignment_and_rejections(tmp_path):
    options,current=source(tmp_path)
    s=SE3Values(options,current);g=SimpleNamespace(game_index=9,rank='D',game_id='g',player_color=1,length=34,hd=0)
    p,v=s.positions(g,np.arange(34)%2)
    assert p.tolist()==[31,33] and v.tolist()==[-.25,.75]
    g.hd=1;assert len(s.positions(g,np.arange(34)%2)[0])==0
    g.hd=0;g.game_id='wrong'
    with pytest.raises(ValueError,match='alignment'):s.positions(g,np.arange(34)%2)
    with pytest.raises(ValueError,match='conditions'):SE3Values(options,dict(current,model_sha256='other'))
    np.save(options['value_path'],np.array([0.,0.],np.float32))
    with pytest.raises(ValueError,match='identity'):SE3Values(options,current)


def test_locks_exclude_duplicate_writer_and_release_after_failure(tmp_path):
    with teacher.teacher_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with teacher.teacher_lock(tmp_path):pass
    with pytest.raises(RuntimeError):
        with teacher.teacher_lock(tmp_path):raise RuntimeError()
    with teacher.teacher_lock(tmp_path):pass
    entered=threading.Event()
    def second():
        with teacher.teacher_lock(tmp_path,wait=True):entered.set()
    with teacher.teacher_lock(tmp_path,wait=True):
        worker=threading.Thread(target=second);worker.start()
        assert not entered.wait(.05)
    worker.join(2);assert entered.is_set()


def test_eight_shards_import_then_resume_without_overwrite(tmp_path,monkeypatch):
    table=pd.DataFrame([dict(game_index=i,rank='D',game_id=f'g{i}',store_index=i,hd=0,length=34,
        player_color=i%2,role='D',oof_fold=1) for i in range(8)])
    cfg=dict(data=dict(teacher_dir=str(tmp_path/'labels'),store_dir='unused'),teacher=dict(num_shards=8,visits=1,se3_reuse={}))
    # A nonempty config enables the importer.
    cfg['teacher']['se3_reuse']={'enabled':True}
    for key in ('engine','model_path','analysis_config'):
        p=tmp_path/key;p.write_text(key);cfg['teacher'][key]=str(p)
    monkeypatch.setattr(teacher,'MODEL',tmp_path/'model_path')
    monkeypatch.setattr(teacher,'load_prepared',lambda *a,**k:dict(identity=dict(sources={}),dataset_id='p'))
    monkeypatch.setattr(teacher,'games',lambda *a:table.copy())
    monkeypatch.setattr(teacher,'MoveStore',lambda _:SimpleNamespace(game=lambda _:(np.arange(34),np.arange(34)%2)))
    class Reuse:
        receipt={'source':'verified'}
        def __init__(self,*a):pass
        def positions(self,g,colors):
            plies=np.flatnonzero((np.arange(34)>=30)&(colors==g.player_color))
            return plies,np.full(len(plies),.25,np.float32)
    import go_aicup.tasks.strength.humansl_residual_b2.teacher_reuse as module
    monkeypatch.setattr(module,'SE3Values',Reuse)
    queries=[]
    class Engine:
        fail=True
        def __init__(self,*a):pass
        def close(self):pass
        def analyze(self,q):
            queries.append(q)
            if self.fail:
                Engine.fail=False
                raise RuntimeError('interrupted')
            for ply in q['analyzeTurns']:yield ply,dict(rootInfo=dict(winrate=.75,scoreLead=0.,visits=1))
    monkeypatch.setattr(teacher,'Engine',Engine)
    with pytest.raises(RuntimeError):teacher.label(cfg,shard=0)
    for i in range(8):
        metrics=teacher.label(cfg,shard=i)
        assert metrics['imported_positions']==metrics['new_positions']==2
        folder=tmp_path/f'labels/shard_{i:05d}'
        before=file_hash(folder/'values.npy')
        assert teacher.label(cfg,shard=i)['new_positions']==0
        assert file_hash(folder/'values.npy')==before
        assert np.load(folder/'done.npy').tolist()==[True]
    assert all(len(q['analyzeTurns'])==2 for q in queries)
    current={'teacher_code_sha256':'new','settings':1}
    old=dict(current,teacher_code_sha256=teacher.LEGACY_CODE)
    assert teacher.compatible_identity(old,current)
    assert not teacher.compatible_identity(dict(old,settings=2),current)
    with teacher.stop_signals() as stopped:
        import os,signal
        os.kill(os.getpid(),signal.SIGUSR1)
        assert stopped[0]
