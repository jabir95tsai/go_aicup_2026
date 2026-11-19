"""Eight-GPU, once-per-family policy experiment through all twenty frozen B3 heads."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import torch
from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_residual.data import question_rows
from go_aicup.tasks.strength.policy_residual.train import predict
from go_aicup.tasks.strength.policy_residual.branches import paired_difference, gate
from . import data, cache, fit, b3, stopping
from .models import ARMS, module_for
from .stopping import ABORT


def parallel(tasks, workers, reset_abort=True):
    if reset_abort:
        ABORT.clear()
    elif ABORT.is_set():
        raise RuntimeError('another experiment worker failed')
    try:
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(fn, *args): key for key, fn, args in tasks}
            result = {}
            try:
                for future in as_completed(futures):
                    result[futures[future]] = future.result()
            except BaseException:
                ABORT.set()
                for future in futures:
                    future.cancel()
                raise
        return result
    finally:
        if reset_abort:
            ABORT.clear()


def code_sources():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'src/go_aicup'/p for p in [
        'data/cv.py','metrics.py','artifacts.py','features/humansl.py','features/humansl_torch.py',
        'features/humansl_rankvec.py','tasks/strength/humansl_finetune.py',
        'tasks/strength/policy_h/five_arms.py','tasks/strength/policy_h/extract.py',
        'tasks/strength/policy_residual/five_arms.py','tasks/strength/policy_residual/branches.py',
        'tasks/strength/policy_residual/model.py','tasks/strength/policy_residual/data.py',
        'tasks/strength/policy_residual/train.py','tasks/strength/policy_residual/epoch.py',
        'tasks/strength/policy_hc_b3/train.py']]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def preflight(config):
    if config['gpu_count'] != 8 or config['family_devices'] != {'H':[0,1,2,3],'C':[4,5,6,7]}:
        raise ValueError('the approved run requires H on GPUs0..3 and C on GPUs4..7')
    for key in ('epochs','questions_per_k','evaluate_every','patience','position_chunk','encoder_workers','inference_positions','games_per_batch'):
        if not isinstance(config[key],int) or config[key] < 1:
            raise ValueError(f'invalid positive setting {key}')
    if config['evaluate_every'] != 500 or config['patience'] != 5 or config['epochs'] != 12:
        raise ValueError('monitor, patience and maximum epoch budget differ from approved plan')
    if not math.isfinite(config['lr']) or config['lr'] <= 0 or not 0 < config['cache_fraction'] <= .8:
        raise ValueError('invalid LR or cache fraction')
    games,questions,report = data.prepare(config)
    record = {'config':config,'data':report,'code':code_sources()}
    identity = object_hash(record)
    budgets = {}
    bounds = np.r_[0,np.cumsum(games.length)]
    for family,p in [('H',9),('C',10)]:
        budgets[family] = [int(bounds[b]-bounds[a])*(p*362*2+p*4+3)
                          for a,b in cache.split_games(games.length,4)]
    print(json.dumps({'stage':'preflight','identity':identity,'roles':report['roles'],
                      'cache_gib_by_device':{k:[round(x/2**30,3) for x in v] for k,v in budgets.items()}},default=int),flush=True)
    return games,questions,record,identity


def frozen_features(family,arm,policy,out,device,chunk,identity):
    with torch.cuda.device(device):
        model,head = fit.load_arm(family,arm,out/family/arm/'model.pt',device)
        if torch.load(out/family/arm/'model.pt',map_location='cpu',weights_only=False)['identity'] != identity:
            raise ValueError('selected checkpoint identity changed')
        view = policy.for_device(device)
        values = module_for(family).game_features(model,view,np.arange(len(policy.games)),chunk).cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('nonfinite selected per-game feature')
        return values


def init_family(cancel):
    global ABORT
    ABORT = stopping.ABORT = fit.ABORT = cache.ABORT = cancel
    torch.set_num_threads(1)


def run_family(family,games,questions,config,identity,out):
    # Separate processes keep teacher TF32/FX state isolated from the other family's training.
    try:
        if ABORT.is_set():
            raise RuntimeError('another experiment worker failed')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        policy,teacher = cache.infer_family(family,games,config,out,ABORT)
        if ABORT.is_set():
            raise RuntimeError('another experiment worker failed')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print({'stage':'family_training_start','family':family},flush=True)
        initial = fit.fit_initial(family,policy.arrays['c'],games,questions['train'])
        devices = config['family_devices'][family]
        networks = {'A0':fit.train_arm(family,'A0',policy.for_device(torch.device('cuda',devices[0])),
                    questions,initial,config,identity,out/family/'A0')}
        tasks = [(arm,fit.train_arm,(family,arm,policy.for_device(torch.device('cuda',device)),
                  questions,initial,config,identity,out/family/arm))
                 for arm,device in zip(ARMS[1:],devices)]
        networks.update(parallel(tasks,4,reset_abort=False))
        compact = {'A0':policy.arrays['c'].numpy().copy()}
        compact.update(parallel([(arm,frozen_features,(family,arm,policy,out,
                       torch.device('cuda',device),config['position_chunk'],identity))
                       for arm,device in zip(ARMS[1:],devices)],4,reset_abort=False))
        print({'stage':'family_complete','family':family},flush=True)
        return {'teacher':teacher,'networks':networks,'compact':compact}
    except BaseException:
        ABORT.set()
        raise


def run_families(games,questions,config,identity,out):
    context = mp.get_context('spawn')
    cancel = context.Event()
    with ProcessPoolExecutor(2,mp_context=context,initializer=init_family,initargs=(cancel,)) as pool:
        futures, result = {}, {}
        try:
            for family in ('H','C'):
                futures[pool.submit(run_family,family,games,questions,config,identity,out)] = family
            for future in as_completed(futures):
                result[futures[future]] = future.result()
        except BaseException:
            cancel.set()
            for future in futures:
                future.cancel()
            raise
    return result


def prediction_frame(q,pred,scores,probs):
    frame = q[['question_id','player_id','rank_id','k']].copy()
    frame['pred_rank'],frame['score'] = pred,scores
    for i in range(10):
        frame[f'p{i}'] = probs[:,i]
    return frame


def comparisons(frames,seed):
    pairs = [(a,'A0') for a in ARMS if a != 'A0']
    pairs += [('CNN-R10','R10'),('CNN-Z32','Z32'),('Z32','R10'),('CNN-Z32','CNN-R10')]
    results = {}
    for a,b in pairs:
        left,right = frames[a],frames[b]
        columns = ['question_id','player_id','rank_id','k']
        if not left[columns].equals(right[columns]):
            raise ValueError('dev predictions are not paired')
        delta = left.score.to_numpy()-right.score.to_numpy()
        diff = paired_difference(left.score.to_numpy(),right.score.to_numpy(),left.k.to_numpy(),
                                 left.player_id.to_numpy(),np.random.default_rng(seed+99))
        diff['by_k'] = {str(k):float(delta[left.k.eq(k)].mean()) for k in (5,10,15,20)}
        diff['equal_k'] = float(np.mean(list(diff['by_k'].values())))
        diff['gate_pass'] = gate(diff)
        results[f'{a}-{b}'] = diff
    return results


def run(config,out,prepare_only=False):
    started = time.monotonic()
    games,questions,record,identity = preflight(config)
    if prepare_only:
        return {'identity':identity,'data':record['data']}
    out = Path(out)
    if (out/'complete.json').exists():
        marker = json.loads((out/'complete.json').read_text())
        if marker['identity'] != identity:
            raise ValueError('completed run identity differs')
        return json.loads((out/'summary.json').read_text())
    if (out/'identity.json').exists():
        raise RuntimeError('volatile policy was lost; refuse automatic teacher re-inference in an incomplete run')
    if torch.cuda.device_count() != 8:
        raise ValueError('exactly eight visible CUDA GPUs are required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'identity.json',dict(record,identity=identity))
    data_dir = out/'data';data_dir.mkdir(exist_ok=True)
    games.to_parquet(data_dir/'games.parquet',index=False)
    for role,q in questions.items():
        q.to_parquet(data_dir/f'{role}_questions.parquet',index=False)
    for relative in record['code']:
        dest = out/'source'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    families = run_families(games,questions,config,identity,out)
    teacher_reports = {f:r['teacher'] for f,r in families.items()}
    results = {(f,a):v for f,r in families.items() for a,v in r['networks'].items()}
    compact = {(f,a):v for f,r in families.items() for a,v in r['compact'].items()}
    del families
    train_values,raw_values={},{}
    for role in ('train','monitor'):
        raw = b3.assemble('A0',None,None,compact['H','A0'],compact['C','A0'])
        raw_values[role]=b3.question_values(games,questions[role],raw)
    for arm in ARMS:
        values=b3.assemble(arm,compact['H',arm],compact['C',arm],compact['H','A0'],compact['C','A0'])
        train_values[arm]={role:b3.question_values(games,questions[role],values) for role in ('train','monitor')}
    def train_b3(arm,k,device):
        with torch.cuda.device(device):
            return b3.fit(arm,k,train_values[arm],raw_values,questions,config,identity,out/'B3',device)
    b3_results=parallel([((arm,k),train_b3,(arm,k,torch.device('cuda',(a*4+j)%8)))
                        for a,arm in enumerate(ARMS) for j,k in enumerate((5,10,15,20))],8)
    # All thirty selected checkpoints are fixed before any dev scores are read.
    q=questions['dev'];rows=question_rows(games,q)
    dev_reports,frames={},{'H':{},'C':{},'B3':{}}
    for family in ('H','C'):
        dev_reports[family]={}
        for arm in ARMS:
            model,head=fit.load_arm(family,arm,out/family/arm/'model.pt','cpu')
            report,pred,scores,probs=predict(head,torch.from_numpy(compact[family,arm]),q,rows)
            frame=prediction_frame(q,pred,scores,probs)
            frame.to_parquet(out/family/arm/'dev_predictions.parquet',index=False)
            frames[family][arm]=frame
            dev_reports[family][arm]=report
    dev_reports['B3']={}
    for arm in ARMS:
        values=b3.assemble(arm,compact['H',arm],compact['C',arm],compact['H','A0'],compact['C','A0'])
        x=b3.question_values(games,q,values)
        frame=q[['question_id','player_id','rank_id','k']].copy()
        by_k,ce={},{}
        for k in (5,10,15,20):
            mask=q.k.eq(k).to_numpy()
            head,result=b3_results[arm,k]
            head=head.cpu()
            report,pred,scores,probs=b3.evaluate(head,torch.as_tensor(x[mask]),q.loc[mask,'rank_id'].to_numpy(),b3.REGULARIZATION_COUNT)
            frame.loc[mask,'pred_rank']=pred;frame.loc[mask,'score']=scores
            frame.loc[mask,[f'p{i}' for i in range(10)]]=probs
            by_k[str(k)],ce[str(k)]=report['score'],report['ce']
        frame['pred_rank']=frame.pred_rank.astype(int)
        frame.to_parquet(out/'B3'/arm/'dev_predictions.parquet',index=False)
        frames['B3'][arm]=frame
        dev_reports['B3'][arm]={'by_k':by_k,'equal_k':float(np.mean(list(by_k.values()))),'ce_by_k':ce}
    comparison_reports={group:comparisons(fs,config['seed']) for group,fs in frames.items()}
    # The source identities describe the complete run, including the frozen C-vector learning population.
    for relative,digest in record['data']['sources'].items():
        if file_hash(ROOT/relative)!=digest:
            raise ValueError(f'source changed while running: {relative}')
    summary={'identity':identity,'data':record['data'],'teachers':teacher_reports,
             'networks':{f'{f}/{a}':r for (f,a),r in results.items()},
             'b3':{f'{a}/k{k}':r for (a,k),(head,r) in b3_results.items()},
             'dev':dev_reports,'comparisons':comparison_reports,'seconds':time.monotonic()-started,
             'limitations':['single seed','new split is not certified historically unseen',
                           'B3 uses in-sample upstream TRAIN features','both stages share monitor'],
             'features_persisted':False,'gpu_count':8}
    write_json(out/'summary.json',summary)
    write_json(out/'complete.json',{'identity':identity,'status':'complete'})
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/strength/policy_large_200k.json')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    out=args.out or ROOT/config['out']/os.environ.get('SLURM_JOB_ID','manual')
    run(config,out,args.prepare_only)


if __name__=='__main__':
    main()
