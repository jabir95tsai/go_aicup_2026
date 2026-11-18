"""CPU orchestration contract: once per teacher, ten networks, twenty B3, delayed dev."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
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
    def teacher(family,games,config,out,cancel):
        calls[family]+=1
        return policies[family],{'teacher_passes_per_game':1}
    def train(family,arm,cache,questions,initial,config,identity,out):
        out.mkdir(parents=True)
        model=models.build(family,arm,config['seed'])
        head=models.module_for(family).BranchHead(model.width,initial)
        atomic_save({'family':family,'arm':arm,'seed':config['seed'],'identity':identity,'step':0,
                     'model':model.state_dict(),'head':head.state_dict(),'initial_head':initial},out/'model.pt')
        calls['upstream']+=1
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
    with patch.object(run,'preflight',return_value=(games,q,record,'test-identity')),\
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
