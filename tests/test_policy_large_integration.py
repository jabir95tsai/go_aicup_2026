"""CPU orchestration contract: once per teacher, ten networks, twenty B3, delayed dev."""
from contextlib import nullcontext
import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
import torch
from go_aicup.artifacts import write_json
from go_aicup.tasks.strength.policy_large import run,fit,b3,models
from go_aicup.tasks.strength.policy_residual.train import atomic_save
from test_policy_large_training import synthetic


def test_complete_orchestration_and_completed_reopen(tmp_path):
    torch.set_num_threads(1)
    hc,q=synthetic(9);cc,_=synthetic(10)
    games=hc.games
    policies={f:SimpleNamespace(arrays=c.arrays,for_device=lambda device,c=c:c) for f,c in [('H',hc),('C',cc)]}
    config=json.load(open('configs/strength/policy_large_200k.json'))
    record={'config':config,'data':{'sources':{}},'code':{}}
    calls={'H':0,'C':0,'upstream':0,'b3':0,'dev':0}
    c_waiting = threading.Event()
    h_deep_started = threading.Event()
    c_teacher_done = threading.Event()
    a0_done = {f:threading.Event() for f in ('H','C')}
    def teacher(family,games,config,out,cancel):
        calls[family]+=1
        if family == 'C':
            c_waiting.set()
            assert h_deep_started.wait(10), 'H waited for the unfinished C teacher'
            c_teacher_done.set()
        else:
            assert c_waiting.wait(10)
        return policies[family],{'teacher_passes_per_game':1}
    def train(family,arm,cache,questions,initial,config,identity,out):
        if arm != 'A0':
            assert a0_done[family].is_set()
        if family == 'H' and arm in ('A0','R10'):
            assert not c_teacher_done.is_set()
        if family == 'H' and arm == 'R10':
            h_deep_started.set()
        out.mkdir(parents=True)
        model=models.build(family,arm,config['seed'])
        head=models.module_for(family).BranchHead(model.width,initial)
        atomic_save({'family':family,'arm':arm,'seed':config['seed'],'identity':identity,'step':0,
                     'model':model.state_dict(),'head':head.state_dict(),'initial_head':initial},out/'model.pt')
        calls['upstream']+=1
        if arm == 'A0':
            a0_done[family].set()
        return {'selected_step':0,'completed_updates':5000}
    def frozen(family,arm,policy,out,device,chunk,identity):
        raw=policy.arrays['c'].numpy()
        if arm in ('R10','CNN-R10'):
            return raw.copy()
        p=9 if family=='H' else 10
        return np.concatenate((raw.reshape(-1,2,p),np.zeros((len(raw),2,32),np.float32)),2).reshape(len(raw),-1)
    def meta_fit(arm,k,values,raw,questions,config,identity,out,device):
        assert calls['upstream']==10
        assert set(values)==set(raw)=={'train','monitor'}
        mask=questions['train'].k.eq(k).to_numpy()
        head=b3.Head(values['train'].shape[1])
        head.load_state_dict(b3.initial_state(arm,values['train'][mask],raw['train'][mask],
                                            questions['train'].loc[mask,'rank_id'].to_numpy()))
        (out/arm/f'k{k}').mkdir(parents=True,exist_ok=True)
        calls['b3']+=1
        return head,{'selected_step':0,'completed_updates':5000}
    original_predict=run.predict
    def dev_predict(*args):
        assert calls['upstream']==10 and calls['b3']==20
        calls['dev']+=1
        return original_predict(*args)
    def threaded_families(games,questions,config,identity,out):
        return run.parallel([(f,run.run_family,(f,games,questions,config,identity,out)) for f in ('H','C')],2)
    with patch.object(run,'run_families',side_effect=threaded_families),\
         patch.object(run,'preflight',return_value=(games,q,record,'test-identity')),\
         patch.object(run.cache,'infer_family',side_effect=teacher),\
         patch.object(run.fit,'train_arm',side_effect=train),\
         patch.object(run,'frozen_features',side_effect=frozen),\
         patch.object(run.b3,'fit',side_effect=meta_fit),\
         patch.object(run,'predict',side_effect=dev_predict),\
         patch.object(torch.cuda,'device_count',return_value=8),\
         patch.object(torch.cuda,'device',side_effect=lambda device:nullcontext()),\
         patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'):
        result=run.run(config,tmp_path)
        reopened=run.run(config,tmp_path)
    assert result==reopened
    assert calls=={'H':1,'C':1,'upstream':10,'b3':20,'dev':10}
    assert len(result['networks'])==10 and len(result['b3'])==20
    assert len(list(tmp_path.rglob('dev_predictions.parquet')))==15
    assert not list(tmp_path.rglob('*features*'))
    assert result['features_persisted'] is False
    assert json.loads((tmp_path/'complete.json').read_text())['status']=='complete'


def _init_precision_probe(cancel, barrier):
    global _PRECISION_BARRIER
    _PRECISION_BARRIER = barrier
    run.init_family(cancel)


def _precision_probe(enabled):
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    if enabled:
        run.ABORT.set()
    _PRECISION_BARRIER.wait(timeout=30)
    return (os.getpid(),torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
            run.ABORT is fit.ABORT is run.cache.ABORT is run.stopping.ABORT,run.ABORT.is_set())


def test_spawn_isolates_precision_and_shares_cancellation():
    context = mp.get_context('spawn')
    cancel,barrier = context.Event(),context.Barrier(2)
    with ProcessPoolExecutor(2,mp_context=context,initializer=_init_precision_probe,
                             initargs=(cancel,barrier)) as pool:
        futures = [pool.submit(_precision_probe,enabled) for enabled in (True,False)]
        results = [future.result(timeout=40) for future in futures]
    assert results[0][0] != results[1][0]
    assert results[0][1:] == (True,True,True,True)
    assert results[1][1:] == (False,False,True,True)
    assert cancel.is_set()


def test_nested_parallel_preserves_cancellation():
    def fail():
        raise ValueError('arm failed')
    run.ABORT.clear()
    try:
        with pytest.raises(ValueError,match='arm failed'):
            run.parallel([('failed',fail,())],1,reset_abort=False)
        assert run.ABORT.is_set()
        with pytest.raises(RuntimeError,match='another experiment worker failed'):
            run.parallel([],1,reset_abort=False)
        assert run.ABORT.is_set()
    finally:
        run.ABORT.clear()


def test_teacher_failure_cancels_other_family_before_training(tmp_path):
    c_waiting = threading.Event()
    def teacher(family,games,config,out,cancel):
        if family == 'H':
            assert c_waiting.wait(5)
            raise ValueError('teacher failed')
        c_waiting.set()
        assert cancel.wait(5), 'teacher did not receive the other family failure'
        raise RuntimeError('teacher cancelled')
    with patch.object(run.cache,'infer_family',side_effect=teacher),\
         patch.object(run.fit,'fit_initial') as initial:
        with pytest.raises((ValueError,RuntimeError)):
            run.parallel([(f,run.run_family,(f,None,None,{},'id',tmp_path)) for f in ('H','C')],2)
    initial.assert_not_called()
    assert not run.ABORT.is_set()


def _family_process_probe(family,games,questions,config,identity,out):
    (out/f'{family}_ready').touch()
    deadline = time.monotonic()+20
    other = 'C' if family == 'H' else 'H'
    while not (out/f'{other}_ready').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('other family did not start')
        time.sleep(.01)
    if config.get('fail'):
        if family == 'C':
            raise ValueError('family process failed')
        assert run.ABORT.wait(20)
        (out/'H_cancelled').touch()
        raise RuntimeError('other family cancelled')
    return {'pid':os.getpid(),'compact':np.arange(6,dtype=np.float32).reshape(2,3),
            'cancelled':run.ABORT.is_set()}


def test_family_process_coordinator_returns_numpy_features(tmp_path):
    with patch.object(run,'run_family',_family_process_probe):
        result=run.run_families(None,None,{},'id',tmp_path)
    assert set(result)=={'H','C'}
    assert result['H']['pid']!=result['C']['pid']
    for family in result.values():
        np.testing.assert_array_equal(family['compact'],np.arange(6,dtype=np.float32).reshape(2,3))
        assert not family['cancelled']


def test_family_process_failure_signals_running_peer(tmp_path):
    with patch.object(run,'run_family',_family_process_probe):
        with pytest.raises(ValueError,match='family process failed'):
            run.run_families(None,None,{'fail':True},'id',tmp_path)
    assert (tmp_path/'H_cancelled').exists()
