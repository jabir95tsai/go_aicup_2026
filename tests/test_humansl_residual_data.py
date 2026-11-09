"""Unequal-rank exhaustion, dual-side semantics, replay and immutable data identity."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from go_aicup.data.cv import make_humansl_residual_roles
from go_aicup.tasks.strength.humansl_residual_b2.sampler import GroupBatchSampler
from go_aicup.tasks.strength.humansl_residual_b2.dataset import effective_plies, encode_selected, SelectedDataset
from go_aicup.tasks.strength.humansl_residual_b2.data import _align, MoveStore, load_prepared
from go_aicup.tasks.strength.humansl_residual_b2.teacher import TeacherLabels
from go_aicup.artifacts import file_hash, object_hash, write_json


def dataset_for_sampler():
    games = pd.DataFrame([dict(game_index=100 * rank + j, rank_id=rank)
                          for rank in range(10) for j in range(5 + rank)])
    return SimpleNamespace(seed=2026, games=games)


def test_independent_rank_streams_no_drop_or_duplicate_and_prefetch_resume():
    dataset = dataset_for_sampler()
    options = dict(groups_per_update=4, microbatch_groups=1, updates=40)
    sampler = GroupBatchSampler(dataset, **options)
    batches = list(sampler)
    # Iteration/prefetch cannot change durable checkpoint state.
    assert sampler.state_dict()['completed_step'] == 0
    for step in range(40):
        tasks = sum(batches[step * 4:(step + 1) * 4], [])
        for level in range(10):
            selected = [row for row, _ in tasks if dataset.games.iloc[row].rank_id == level]
            assert len(selected) == len(set(selected)) == 4
    for level in range(10):
        observed = {row for batch in batches for row, _ in batch if dataset.games.iloc[row].rank_id == level}
        assert observed == set(np.flatnonzero(dataset.games.rank_id.eq(level)))
    state = sampler.state_dict(13)
    for level,stream in enumerate(state['ranks']):
        assert stream['round'] * len(sampler.rows[level]) + stream['cursor'] - len(stream['pending']) == 13 * 4
    resumed = GroupBatchSampler(dataset, **options, start_step=13, state=state)
    assert list(resumed) == batches[13 * 4:]
    assert json.loads(json.dumps(state)) == state
    with pytest.raises(ValueError, match='identity'):
        GroupBatchSampler(dataset, **options, start_step=12, state=state)
    # Complete groups are globally partitioned without changing sequence.
    ranks = [list(GroupBatchSampler(dataset, **options, world_size=2, rank=r)) for r in range(2)]
    for step in range(40):
        assert ranks[0][step * 2:(step + 1) * 2] + ranks[1][step * 2:(step + 1) * 2] == batches[step * 4:(step + 1) * 4]


def test_role_identity_unchanged_and_release_mapping_stratified():
    records = [dict(player_id=f'{rank}-{p}',rank_id=rank,player_color=0)
               for rank in range(10) for p in range(50) for _ in range(3)]
    games = pd.DataFrame(records)
    folds = pd.DataFrame([dict(player_id=f'{rank}-{p}',fold=p % 6 - 1)
                          for rank in range(10) for p in range(50)])
    original = make_humansl_residual_roles(games,folds)
    assert original.player_id.is_unique
    assert set(original.role) == {'D','M','U','R'}
    assert (original.oof_fold == original.fold).all()
    assert original.loc[original.fold.eq(-1),'role'].eq('R').all()
    assert original.loc[original.fold.eq(0),'role'].eq('U').all()
    release = make_humansl_residual_roles(games,folds,release=True)
    assert set(release.role) == {'P','R'}
    for rank in range(10):
        group=release[release.role.eq('P') & release.rank_id.eq(rank)]
        assert set(group.oof_fold)=={1,2,3,4}
        assert group.oof_fold.value_counts().max()-group.oof_fold.value_counts().min()<=1
    with pytest.raises(ValueError,match='missing'):
        make_humansl_residual_roles(games,folds.iloc[1:])


def test_effective_boundary_actual_colors_and_store_alignment():
    colors=np.arange(65)%2
    own,opp=effective_plies(colors,1)
    assert own.min()==31 and opp.min()==30
    assert len(own)==17 and len(opp)==18 and not set(own)&set(opp)
    meta=pd.DataFrame([dict(game_index=9,rank='D',game_id='same',player_id='a',color='B',length=65),
                       dict(game_index=8,rank='C',game_id='same',player_id='b',color='W',length=70)])
    store=SimpleNamespace(meta=meta.iloc[::-1].copy())
    aligned=_align(meta,store)
    assert aligned.store_index.tolist()==[1,0]
    store.meta.loc[store.meta.game_index.eq(9),'color']='W'
    with pytest.raises(ValueError,match='color'):
        _align(meta,store)


def test_selected_encoder_only_calls_features_on_selected_pre_action_turns(monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2 import dataset as module
    histories=[];rules_seen=[]
    class Board:
        BLACK=1;WHITE=2;PASS_LOC=361
    class State:
        RULES_JAPANESE={'whiteKomi':6.5}
        def __init__(self,size,rules):
            rules_seen.append(rules);self.board=SimpleNamespace(pla=1,loc=lambda x,y:y*19+x);self.turn=0
        def get_input_features(self,encoder):
            histories.append(self.turn)
            b=np.zeros((1,22,19,19),np.uint8);b[0,0,self.turn%19,0]=1
            return b,np.asarray([[self.turn]],np.float32)
        def play(self,pla,loc):
            self.turn+=1;self.board.pla=3-pla
    class Features:
        def __init__(self,config,size):pass
        def loc_to_tensor_pos(self,loc,board):return loc
    monkeypatch.setattr(module,'_katago',lambda:(SimpleNamespace(Board=Board),SimpleNamespace(Features=Features),SimpleNamespace(GameState=State),None))
    monkeypatch.setattr(module,'profile_metadata',lambda _:SimpleNamespace(get_metadata_row=lambda **kw:np.asarray([kw['nextPlayer']],np.float32)))
    moves=np.arange(65);colors=np.arange(65)%2
    output=encode_selected(moves,colors,[34,30,33],{},1,transform=1)
    assert histories==[30,33,34]
    assert output['global_inputs'].ravel().tolist()==[34,30,33]
    assert rules_seen[0]['whiteKomi']==.5
    assert output['spatial'].dtype==np.uint8
    assert output['metadata'].ravel().tolist()==[1,1,2]
    assert output['actions'].tolist()!=[34,30,33]
    with pytest.raises(ValueError,match='pass'):
        encode_selected(np.full(65,361),colors,[30],{},0)
    colors[10]=1
    with pytest.raises(ValueError,match='replay'):
        encode_selected(moves,colors,[30],{},0)


def test_dataset_fixed_positions_both_sides_and_worker_safe_state(monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2 import dataset as module
    game=pd.DataFrame([dict(game_index=7,store_index=0,rank_id=3,player_color=1,hd=0,length=100)])
    config=dict(seed=2026,training=dict(positions_per_side=15),data=dict(store_dir='unused'))
    monkeypatch.setattr(module,'MoveStore',lambda _:SimpleNamespace(game=lambda _: (np.arange(100),np.arange(100)%2)))
    monkeypatch.setattr(module,'encode_selected',lambda moves,colors,plies,*args,**kwargs:dict(spatial=np.zeros((len(plies),22,19,19),np.uint8)))
    ds=SelectedDataset(config,game,{},teacher_required=False,augment=False)
    a,b,c=ds[(0,0)],ds[(0,0)],ds[(0,1)]
    assert np.array_equal(a['ply'],b['ply']) and not np.array_equal(a['ply'],c['ply'])
    assert a['sides'].tolist()==[0]*15+[1]*15
    assert len(set(a['ply']))==30
    assert (a['labels']==3).all() and 'values' not in a
    assert ds.__getstate__()['_store'] is None
    batches=list(ds.iter_positions(0,batch_size=11))
    assert max(len(x['ply']) for x in batches)<=11
    assert np.array_equal(np.concatenate([x['ply'] for x in batches]),np.arange(30,100))


def test_teacher_exact_game_komi_ply_identity_and_finite_corruption_rejected(tmp_path,monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2.teacher import value_digest
    from go_aicup.tasks.strength.humansl_residual_b2 import teacher as module
    monkeypatch.setattr(module,'load_prepared',lambda cfg:dict(identity=dict(sources={})))
    cfg=dict(data=dict(teacher_dir=str(tmp_path)),teacher=dict(num_shards=1,visits=1))
    identity=dict(perspective='actual_side_to_move',visits=1,num_shards=1,rules='japanese',komi={'0':6.5,'1':.5},min_ply=30,ignorePreRootHistory=False)
    for key in ('engine','model_path','analysis_config'):
        path=tmp_path/key;path.write_text(key);cfg['teacher'][key]=str(path)
    identity.update(source_files={},teacher_code_sha256=file_hash(module.__file__),dataset_code_sha256=file_hash(module.Path(module.__file__).with_name('dataset.py')),engine_sha256=file_hash(tmp_path/'engine'),model_sha256=file_hash(tmp_path/'model_path'),
                    config_sha256=file_hash(tmp_path/'analysis_config'))
    write_json(tmp_path/'identity.json',identity)
    root=tmp_path/'shard_00000';root.mkdir()
    game=pd.Series(dict(game_index=7,rank='D',game_id='same',hd=1,length=34,player_color=1))
    index=pd.DataFrame([{**game.to_dict(),'row':0,'offset':0}]);index.to_parquet(root/'index.parquet')
    targets=np.asarray([.1,-.2,.3,-.4],np.float32)
    np.save(root/'values.npy',targets);np.save(root/'done.npy',np.asarray([True]));np.save(root/'checksums.npy',np.asarray([value_digest(targets)],np.uint64))
    write_json(root/'manifest.json',dict(teacher_id=object_hash(identity),index_sha256=file_hash(root/'index.parquet')))
    teacher=TeacherLabels(cfg)
    wrong_identity=dict(identity,dataset_code_sha256='wrong')
    write_json(tmp_path/'identity.json',wrong_identity)
    with pytest.raises(ValueError,match='code identity'):
        TeacherLabels(cfg)
    write_json(tmp_path/'identity.json',dict(identity,source_files={'wrong':'hash'}))
    with pytest.raises(ValueError,match='sources'):
        TeacherLabels(cfg)
    write_json(tmp_path/'identity.json',identity)
    assert np.array_equal(teacher.values(game,np.asarray([33,30])),targets[[3,0]])
    wrong=game.copy();wrong['hd']=0
    with pytest.raises(ValueError,match='identity'):
        teacher.values(wrong,np.asarray([30]))
    with pytest.raises(ValueError,match='missing'):
        teacher.values(game,np.asarray([29]))
    corrupt=np.load(root/'values.npy',mmap_mode='r+');corrupt[0]=.9;corrupt.flush()
    with pytest.raises(ValueError,match='checksum'):
        teacher.values(game,np.asarray([30]))


def test_prepare_full_legacy_roles_hd_join_and_manifest_reuse(tmp_path):
    from go_aicup.tasks.strength.humansl_residual_b2.data import prepare, games, questions
    records=[];folds=[];offset=0
    for rank in range(10):
        for player in range(50):
            pid=f'{rank}-{player}';folds.append(dict(player_id=pid,rank_id=rank,fold=player%6-1))
            for number in range(25):
                records.append(dict(game_index=len(records),rank=str(rank),rank_id=rank,game_id=f'g{player}_{number}',
                                    player_id=pid,color='B',player_color=0,length=60,shard_id=0,local_offset=offset))
                offset+=60
    table=pd.DataFrame(records)
    store=tmp_path/'store';store.mkdir();table.to_parquet(store/'games.parquet')
    np.save(store/'shard_00000_moves.npy',np.zeros(offset,np.int16))
    np.save(store/'shard_00000_colors.npy',np.tile(np.arange(60)%2,len(table)).astype(np.int8))
    hd=table[['rank','game_id']].copy();hd['handicap']=0;hd.loc[0,'handicap']=1
    hd.to_csv(tmp_path/'hd.csv',index=False)
    pd.DataFrame(folds).to_parquet(tmp_path/'folds.parquet')
    legacy=table.copy();legacy['role']=np.where(legacy.player_id.map(dict((x['player_id'],x['fold']) for x in folds)).eq(0),'U','S-fit')
    legacy.to_parquet(tmp_path/'legacy.parquet')
    config=dict(seed=2026,split_seed=2027,training=dict(groups_per_update=1),b2=dict(questions_per_k=200),
                monitoring=dict(questions_per_cell=1),data=dict(store_dir=str(store),metadata_path=str(store/'games.parquet'),
                handicap_path=str(tmp_path/'hd.csv'),folds_path=str(tmp_path/'folds.parquet'),
                legacy_split_paths=[str(tmp_path/'legacy.parquet')],prepared_dir=str(tmp_path/'prepared')))
    record=prepare(config)
    assert prepare(config)==record
    saved=games(config);fit=questions(config,'fit');monitor=questions(config,'monitor');unseen=questions(config,'U')
    assert len(saved)==len(table) and saved.loc[saved.game_index.eq(0),'hd'].item()==1
    role=saved.set_index('player_id').role.to_dict()
    assert set(fit.player_id.map(role))=={'D'} and set(monitor.player_id.map(role))=={'M'} and set(unseen.player_id.map(role))=={'U'}
    assert not set(fit.player_id)&set(monitor.player_id) and not set(fit.player_id)&set(unseen.player_id)
    lookup=saved.set_index('game_index')
    for question in monitor.itertuples():
        chosen=lookup.loc[question.games]
        assert chosen.player_id.nunique()==chosen.player_color.nunique()==1
        assert len(set(question.games))==question.k
    bad=dict(config,seed=9)
    with pytest.raises(ValueError,match='config changed'):
        load_prepared(bad)
    hd.loc[0,'handicap']=0;hd.to_csv(tmp_path/'hd.csv',index=False)
    with pytest.raises(ValueError,match='source changed'):
        load_prepared(config)


def test_teacher_label_partial_retry_query_perspective_and_new_work_metrics(tmp_path,monkeypatch):
    from go_aicup.tasks.strength.humansl_residual_b2 import teacher as module
    table=pd.DataFrame([dict(game_index=i,rank='D',game_id=f'g{i}',store_index=i,hd=i%2,length=34,
                            player_color=0,role='D' if i<2 else 'M',oof_fold=1) for i in range(3)])
    config=dict(data=dict(teacher_dir=str(tmp_path/'labels'),store_dir='unused'),teacher=dict(num_shards=1,visits=1))
    for key in ('engine','model_path','analysis_config'):
        path=tmp_path/key;path.write_text(key);config['teacher'][key]=str(path)
    monkeypatch.setattr(module,'MODEL',tmp_path/'model_path')
    monkeypatch.setattr(module,'load_prepared',lambda cfg,**kwargs:dict(identity=dict(sources={}),dataset_id='prepared'))
    monkeypatch.setattr(module,'games',lambda cfg,phase:table.copy())
    monkeypatch.setattr(module,'MoveStore',lambda _:SimpleNamespace(game=lambda _: (np.arange(34),np.arange(34)%2)))
    queries=[]; closed=[]
    class FakeEngine:
        fail=True
        def __init__(self,*args):pass
        def close(self):closed.append(True)
        def analyze(self,query):
            queries.append(query)
            for index,ply in enumerate(query['analyzeTurns']):
                yield ply,dict(rootInfo=dict(winrate=.75,scoreLead=3.,visits=1))
                if self.fail and query['id']=='D/g1' and index==0:
                    FakeEngine.fail=False
                    raise RuntimeError('interrupted engine')
    monkeypatch.setattr(module,'Engine',FakeEngine)
    with pytest.raises(RuntimeError,match='interrupted'):
        module.label(config)
    root=tmp_path/'labels'/'shard_00000'
    first=json.loads((root/'metrics.json').read_text())
    assert first['status']=='failed' and first['new_games']==1 and first['new_positions']==5
    assert np.load(root/'done.npy').tolist()==[True,False,False]
    second=module.label(config)
    assert second['status']=='complete' and second['new_games']==1 and second['new_positions']==4
    assert second['gpu_count']==1 and second['gpu_hours']==second['seconds']/3600
    assert second['positions_per_second']==4/second['seconds']
    assert second['setup_seconds']<=second['seconds']
    assert len(queries)==3 and len(closed)==2
    assert [q['komi'] for q in queries]==[6.5,.5,.5]
    assert all(q['rules']=='japanese' and q['maxVisits']==1 and q['analyzeTurns']==[30,31,32,33] for q in queries)
    assert all(len(q['moves'])==33 and q['overrideSettings']['ignorePreRootHistory'] is False for q in queries)
    teacher=TeacherLabels(config)
    assert np.array_equal(teacher.values(table.iloc[1],np.arange(30,34)),np.asarray([.5,-.5,.5,-.5],np.float32))
    final=module.label(config)
    assert final['new_games']==final['new_positions']==final['positions_per_second']==0
    assert len(queries)==3
