"""Universal ground policies: parallel expert cloning, PPO and per-model evaluation.

One policy per tactical family (navigation, combat, defense, support) is shared by
every 2027 ground model. The robot descriptor in the observation conditions it on
role, drive and capabilities. Rollouts come from decoupled worker processes; the
PPO update runs on CPU, CUDA or Apple MPS.
"""
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from .bridge import configure,provenance,check_provenance
from .learning import dump,weight_hash
from .ground_encoding import SCHEMA,FEATURES,ACTION_SPACE,OBS_SPACE,FAMILIES
from .robots import GROUND_MODELS_2027

FORMAT='arena-ground-policy-v1'
POLICY_KWARGS={'net_arch':dict(pi=[256,256],vf=[256,256])}
ALL_GROUND=tuple(m for ms in GROUND_MODELS_2027.values() for m in ms)


def resolve_device(device='auto'):
    """auto prefers CUDA; Apple MPS is opt-in because small MLP updates run faster on CPU."""
    import torch
    if device=='auto':return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but torch.cuda.is_available() is false')
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS requested but unavailable')
    if device not in ('cpu','cuda','mps') and not device.startswith('cuda:'):raise ValueError('device must be auto, cpu, mps, cuda or cuda:N')
    return device


def seed_learner(seed,device,threads):
    import random
    import torch
    if device.startswith('cuda'):os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(max(1,threads))
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    # MPS lacks deterministic kernels for several ops; CPU/CUDA updates stay deterministic.
    if device=='cpu' or device.startswith('cuda'):torch.use_deterministic_algorithms(True,warn_only=device!='cpu')


class _SpaceEnv:
    """Space holder for SB3 construction; rollouts come from RolloutWorkers."""
    def __init__(self):
        import gymnasium as gym
        class Stub(gym.Env):
            observation_space=OBS_SPACE;action_space=ACTION_SPACE
            def reset(self,*,seed=None,options=None):return np.zeros(OBS_SPACE.shape,dtype=np.float32),{}
            def step(self,action):return np.zeros(OBS_SPACE.shape,dtype=np.float32),0.,True,False,{}
        self.cls=Stub


def _model(workers,n_steps,seed,device,learning_rate,policy_kwargs=POLICY_KWARGS):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    stub=_SpaceEnv().cls
    env=DummyVecEnv([stub]*workers)
    return PPO('MlpPolicy',env,n_steps=n_steps,batch_size=min(1024,n_steps*workers),n_epochs=6,learning_rate=learning_rate,gamma=.99,gae_lambda=.95,
               clip_range=.15,ent_coef=.003,vf_coef=.5,max_grad_norm=.5,target_kl=.04,seed=seed,device=device,policy_kwargs=policy_kwargs,verbose=0)


def _cpu_state(model):return {k:v.detach().to('cpu').clone() for k,v in model.policy.state_dict().items()}


def _class(info):return f"{info.get('role')}/{info.get('drive')}"


def summarize(episodes):
    by=defaultdict(lambda:[0,0]);variant=defaultdict(lambda:[0,0]);model=defaultdict(lambda:[0,0])
    for e in episodes:
        for table,key in ((by,_class(e)),(variant,e.get('variant')),(model,str(e.get('model')))):
            table[key][0]+=1;table[key][1]+=int(bool(e.get('is_success')))
    rate=lambda table:{k:{'episodes':v[0],'success_rate':v[1]/max(1,v[0])} for k,v in sorted(table.items(),key=lambda kv:str(kv[0]))}
    return {'episodes':len(episodes),'success_rate':sum(bool(e.get('is_success')) for e in episodes)/max(1,len(episodes)),
            'mean_return':float(np.mean([e['return'] for e in episodes])) if episodes else None,'by_class':rate(by),'by_variant':rate(variant),'by_model':rate(model)}


def collect_demonstrations(pool,model,target_episodes,steps_per_call=256,gamma=.99,max_calls=400):
    """Expert rollouts from all workers; only successful complete episodes are kept."""
    pending=[[] for _ in pool.connections];kept_obs=[];kept_actions=[];accepted=[];rejected=defaultdict(int);calls=0
    while len(accepted)<target_episodes and calls<max_calls:
        results=pool.collect(None if calls else _cpu_state(model),POLICY_KWARGS,steps_per_call,gamma,expert=True);calls+=1
        for w,r in enumerate(results):
            finished=iter(r['episodes'])
            for t in range(len(r['rewards'])):
                if r['starts'][t] and pending[w]:pending[w]=[]
                pending[w].append((r['obs'][t],r['actions'][t]))
                ends=t+1<len(r['rewards']) and r['starts'][t+1] or t+1==len(r['rewards']) and r['last_start']
                if ends:
                    info=next(finished)
                    if info.get('is_success'):
                        kept_obs+= [o for o,_ in pending[w]];kept_actions+=[a for _,a in pending[w]];accepted.append(info)
                    else:rejected[f"{info.get('variant')}:{info.get('reason')}"]+=1
                    pending[w]=[]
        print(json.dumps({'type':'demonstrations','accepted':len(accepted),'samples':len(kept_obs),'calls':calls}),flush=True)
    return np.asarray(kept_obs,dtype=np.float32),np.asarray(kept_actions,dtype=np.int64),accepted,dict(rejected)


def behavior_clone(model,x,y,epochs,seed,batch=512,lr=1e-3):
    import torch
    rng=np.random.default_rng(seed);optimizer=torch.optim.Adam(model.policy.parameters(),lr=lr);losses=[]
    xt=torch.as_tensor(x,device=model.device);yt=torch.as_tensor(y,device=model.device)
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        order=torch.as_tensor(rng.permutation(len(x)),device=model.device);total=0.
        for start in range(0,len(x),batch):
            index=order[start:start+batch]
            distribution=model.policy.get_distribution(xt[index])
            loss=-distribution.log_prob(yt[index]).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step();total+=float(loss.detach())*len(index)
        losses.append(total/len(x))
    model.policy.set_training_mode(False);model.policy.optimizer.state.clear()
    with torch.no_grad():predicted=model.policy.get_distribution(xt).mode().cpu().numpy()
    # Single-head (Discrete) and multi-head (MultiDiscrete) actions share one report.
    predicted=np.asarray(predicted).reshape(len(y),-1);target=np.asarray(y).reshape(len(y),-1)
    return {'epochs':epochs,'initial_loss':losses[0],'final_loss':losses[-1],'per_head_accuracy':(predicted==target).mean(axis=0).round(4).tolist(),
            'full_action_accuracy':float(np.mean(np.all(predicted==target,axis=1)))}


def train_ground(*,family,steps=262144,workers=8,seed=501,output,arena_root=None,models=None,device='auto',n_steps=256,
                 expert_episodes=240,bc_epochs=40,learning_rate=1e-4,resume=None,initialize_from=None,threads=None,checkpoint_every=8,demonstrations_from=None):
    """Train a universal ground policy; returns its metadata."""
    import torch
    if family not in FAMILIES:raise ValueError('Unknown ground family')
    if not 1<=workers<=64 or steps<1 or n_steps<16:raise ValueError('Invalid training size')
    root=configure(arena_root);device=resolve_device(device);output=Path(output)
    if (output/'metadata.json').exists() and not resume:raise ValueError('Training output already exists; use a new directory or --resume')
    output.mkdir(parents=True,exist_ok=True)
    seed_learner(seed,device,threads or max(1,(os.cpu_count() or 4)-workers))
    models=tuple(int(m) for m in (models or ALL_GROUND))
    source=provenance(root);started=time.monotonic()
    training_source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    from .parallel import RolloutWorkers
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure as sb3_logger
    pool=RolloutWorkers(family,workers,seed,models,root)
    try:
        parent=None
        if resume or initialize_from:
            parent_path=Path(resume or initialize_from);parent,parent_model=load_ground_policy(parent_path,root,device)
            model=_model(workers,n_steps,seed,device,learning_rate)
            model.policy.load_state_dict(parent_model.policy.state_dict())
            if resume:model.num_timesteps=int(parent['actual_steps'])
        else:model=_model(workers,n_steps,seed,device,learning_rate)
        model.set_logger(sb3_logger(str(output/'sb3'),['csv']))
        before=weight_hash(model);demonstrations=None
        if demonstrations_from and not (resume or initialize_from):
            # Reuse a previous run's successful expert dataset (same schema), then clone again.
            data=np.load(Path(demonstrations_from)/'demonstrations.npz');x,y=data['observations'],data['actions']
            if x.shape[1]!=len(FEATURES) or y.shape[1]!=len(ACTION_SPACE.nvec):raise ValueError('Demonstration dataset schema mismatch')
            np.savez_compressed(output/'demonstrations.npz',observations=x,actions=y)
            demonstrations={'method':'behavior cloning of successful chassis-aware expert episodes (reused dataset), then PPO','samples':int(len(x)),
                            'source':str(demonstrations_from),'dataset_sha256':hashlib.sha256((output/'demonstrations.npz').read_bytes()).hexdigest(),
                            **behavior_clone(model,x,y,bc_epochs,seed)}
            print(json.dumps({'type':'behavior_cloning',**demonstrations}),flush=True)
            model.save(output/'demonstration-policy.zip')
        elif expert_episodes and not (resume or initialize_from):
            x,y,accepted,rejected=collect_demonstrations(pool,model,expert_episodes)
            if not len(accepted):raise RuntimeError('No successful expert demonstrations')
            np.savez_compressed(output/'demonstrations.npz',observations=x,actions=y)
            demonstrations={'method':'behavior cloning of successful chassis-aware expert episodes, then PPO','episodes':len(accepted),'samples':int(len(x)),
                            'by_class':summarize(accepted)['by_class'],'rejected':rejected,
                            'dataset_sha256':hashlib.sha256((output/'demonstrations.npz').read_bytes()).hexdigest(),
                            **behavior_clone(model,x,y,bc_epochs,seed)}
            print(json.dumps({'type':'behavior_cloning',**{k:v for k,v in demonstrations.items() if k not in ('by_class','rejected')}}),flush=True)
            model.save(output/'demonstration-policy.zip')
        before_ppo=weight_hash(model)
        total=model.num_timesteps+steps;per_iteration=workers*n_steps;iterations=math.ceil(steps/per_iteration)
        window=[];history=[];buffer=model.rollout_buffer;initial=model.num_timesteps
        for iteration in range(iterations):
            tick=time.monotonic()
            results=pool.stream(_cpu_state(model),POLICY_KWARGS,n_steps,model.gamma,workers,iteration)
            collect_s=time.monotonic()-tick
            buffer.reset()
            for t in range(n_steps):
                buffer.add(np.stack([r['obs'][t] for r in results]),np.stack([r['actions'][t] for r in results]),
                           np.array([r['rewards'][t] for r in results]),np.array([r['starts'][t] for r in results]),
                           torch.as_tensor([r['values'][t] for r in results]),torch.as_tensor([r['log_probs'][t] for r in results]))
            buffer.compute_returns_and_advantage(last_values=torch.as_tensor([r['last_value'] for r in results]),dones=np.array([r['last_start'] for r in results]))
            model.num_timesteps+=per_iteration
            model._current_progress_remaining=max(0.,1.-(model.num_timesteps-initial)/max(1,steps))
            tick=time.monotonic();model.train();update_s=time.monotonic()-tick
            diagnostics={k.split('/')[-1]:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/')}
            model.logger.dump(model.num_timesteps)
            episodes=[e for r in results for e in r['episodes']];window=(window+episodes)[-400:]
            row={'timesteps':model.num_timesteps,'iteration':iteration+1,'wall_seconds':time.monotonic()-started,'collect_seconds':collect_s,'update_seconds':update_s,
                 'samples_per_second':per_iteration/max(1e-6,collect_s+update_s),'window':summarize(window),'new_episodes':len(episodes),'ppo':diagnostics,
                 'worker_seconds':[round(r['seconds'],2) for r in results],'policy_lag':[iteration-r['version'] for r in results],'workers_used':sorted({r['worker'] for r in results})}
            history.append(row)
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(json.dumps({'type':'ppo','family':family,'timesteps':row['timesteps'],'sps':round(row['samples_per_second']),'success':round(row['window']['success_rate'],3),
                              'episodes':row['window']['episodes'],'by_class':{k:round(v['success_rate'],2) for k,v in row['window']['by_class'].items()}}),flush=True)
            if (iteration+1)%checkpoint_every==0:model.save(output/'policy-latest.zip')
        model.save(output/'policy.zip');after=weight_hash(model)
        if after==before_ppo:raise RuntimeError('PPO did not update ground policy weights')
        metadata={'format':FORMAT,'schema':SCHEMA,'family':family,'features':list(FEATURES),'action_nvec':ACTION_SPACE.nvec.tolist(),'action_hz':10,
                  'season':'2027','models':list(models),'seed':seed,'workers':workers,'n_steps':n_steps,'device':device,'policy_kwargs':{'net_arch':POLICY_KWARGS['net_arch']},
                  'learning_rate':learning_rate,'actual_steps':model.num_timesteps,'additional_steps':model.num_timesteps-initial,'ppo_updates':model._n_updates,
                  'weights_before':before,'weights_before_ppo':before_ppo,'weights_after':after,'ppo_weights_changed':True,
                  'wall_seconds':time.monotonic()-started,'final_window':history[-1]['window'] if history else None,
                  'arena':source,'aim_mode':'SDK ballistic assistance with learned residuals','controlled_robots':'one learner per simulator; others declared scripted',
                  'policy_sha256':hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest(),
                  'training_source_sha256':training_source,
                  'versions':{'torch':torch.__version__,'numpy':np.__version__}}
        if demonstrations:metadata['demonstrations']=demonstrations
        if parent:metadata['parent_policy_sha256']=parent['policy_sha256'];metadata['parent_mode']='resume' if resume else 'initialize'
        dump(output/'metadata.json',metadata)
        print(json.dumps({'type':'trained_ground','family':family,'steps':model.num_timesteps,'seconds':round(metadata['wall_seconds'])}),flush=True)
        return metadata
    finally:pool.close()


def load_ground_policy(checkpoint,root,device='cpu'):
    from stable_baselines3 import PPO
    path=Path(checkpoint);meta=json.loads((path/'metadata.json').read_text())
    if meta.get('format')!=FORMAT or meta.get('schema')!=SCHEMA or meta.get('features')!=list(FEATURES):raise ValueError('Ground policy schema mismatch')
    if hashlib.sha256((path/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:raise ValueError('Ground checkpoint digest mismatch')
    check_provenance(meta['arena'],provenance(root))
    return meta,PPO.load(path/'policy.zip',device=resolve_device(device))


def evaluate_ground(*,family,checkpoint=None,baselines=('expert','random'),models=None,episodes_per_model=12,workers=8,seed=7_100_000,output=None,arena_root=None,replays=None):
    """Same held-out seeds for the policy and baselines, reported per robot model."""
    root=configure(arena_root);models=tuple(int(m) for m in (models or ALL_GROUND))
    meta,model=load_ground_policy(checkpoint,root) if checkpoint else (None,None)
    family=family or meta['family'];report={'format':'arena-ground-evaluation-v1','family':family,'checkpoint':str(checkpoint) if checkpoint else None,
        'policy_sha256':meta['policy_sha256'] if meta else None,'episodes_per_model':episodes_per_model,'seed':seed,'models':{}}
    from .parallel import RolloutWorkers
    started=time.monotonic()
    for k,m in enumerate(models):
        seeds=[seed+k*10_000+i for i in range(episodes_per_model)]
        with RolloutWorkers(family,min(workers,episodes_per_model),seed+k,(m,),root,training_rules=False) as pool:
            entry={}
            for mode in (('policy',) if model else ())+tuple(baselines):
                rows=pool.evaluate(seeds,mode,_cpu_state(model) if mode=='policy' else None,POLICY_KWARGS if mode=='policy' else None,replays if mode=='policy' else None)
                entry[mode]={**summarize(rows),'reasons':dict(sorted(Counter(str(r.get('reason')) for r in rows).items()))}
                entry[mode].pop('by_model',None)
        report['models'][str(m)]=entry
        print(json.dumps({'type':'evaluated','family':family,'model':m,**{mode:round(v['success_rate'],3) for mode,v in entry.items()}}),flush=True)
    report['wall_seconds']=time.monotonic()-started
    totals={mode:np.mean([v[mode]['success_rate'] for v in report['models'].values()]) for mode in next(iter(report['models'].values()))}
    report['mean_success_rate']={k:float(v) for k,v in totals.items()}
    if output:dump(output,report)
    return report
