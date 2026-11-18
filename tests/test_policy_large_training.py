import json
import numpy as np
import pandas as pd
import torch
from go_aicup.tasks.strength.policy_large import fit,b3,models
from go_aicup.tasks.strength.policy_residual.data import Cache


def synthetic(p):
    rng=np.random.default_rng(4)
    games=[];questions={role:[] for role in ('train','monitor','dev')}
    for role in questions:
        for rank in range(10):
            ids=[]
            for j in range(20):
                ids.append(len(games))
                games.append(dict(game_index=len(games),rank_id=rank,player_id=f'{role}{rank}',player_color=0,
                                  handicap=0,length=2,role=role,color='B'))
            for k in (5,10,15,20):
                questions[role].append(dict(question_id=len(questions[role]),player_id=f'{role}{rank}',rank_id=rank,k=k,color='B',games=np.array(ids[:k])))
    games=pd.DataFrame(games)
    questions={r:pd.DataFrame(v) for r,v in questions.items()}
    n=len(games)*2
    target=torch.zeros(n,dtype=torch.int16)
    lp=torch.from_numpy(rng.normal(size=(n,p,362)).astype('f')).log_softmax(-1)
    q=lp[:,:,0].clone()
    arrays={'logp':lp.half(),'q':q,'target':target,'side':torch.arange(n).remainder(2).to(torch.uint8),
            'c':q.reshape(len(games),2*p),'counts':torch.ones(len(games),2)}
    return Cache(games,np.arange(len(games)+1)*2,arrays,torch.device('cpu')),questions


def config():
    return dict(seed=2027,position_chunk=128,questions_per_k=1,epochs=2,patience=5,
                evaluate_every=5,lr=.0003,weight_decay=.0001,train_probe_questions=40)


def test_real_cpu_upstream_gradients_and_artifacts(tmp_path):
    torch.set_num_threads(1)
    cache,q=synthetic(9)
    initial=fit.fit_initial('H',cache.arrays['c'],cache.games,q['train'])
    result=fit.train_arm('H','CNN-R10',cache,q,initial,config(),'id',tmp_path/'H')
    assert result['completed_updates']==20 and result['steps_per_epoch']==10
    assert result['training_games_seen']==200
    model,head=fit.load_arm('H','CNN-R10',tmp_path/'H/model.pt','cpu')
    assert model.conv(torch.zeros(1,27,19,19)).shape[-1]==7
    saved=torch.load(tmp_path/'H/last.pt',weights_only=False)
    assert saved['stopping']['best']>0
    assert not list(tmp_path.rglob('*features*'))
    assert json.loads((tmp_path/'H/complete.json').read_text())['identity']=='id'


def test_real_cpu_b3_training_and_no_dev_access(tmp_path):
    torch.set_num_threads(1)
    cache,q=synthetic(9)
    rng=np.random.default_rng(5)
    values={r:rng.normal(size=(len(q[r]),38)).astype('f') for r in ('train','monitor')}
    head,result=b3.fit('A0',5,values,values,q,config(),'id',tmp_path,torch.device('cpu'))
    assert result['completed_updates']==20
    assert result['train_questions']==10
    assert not list(tmp_path.rglob('*features*'))
    assert 'dev' not in result['roles']
