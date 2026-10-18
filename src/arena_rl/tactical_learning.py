"""Train and measure the goal-conditioned execution policies used by an LLM."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from .bridge import configure,provenance,check_provenance
from .learning import seed_everything,weight_hash,Metrics,dump
from .tactical_envs import TacticalVecEnv,TacticalBatch,FAMILIES
from .tactical_encoding import SCHEMA,ACTION_SPACE,FEATURES,encode


def train_tactics(*,family,steps=32768,envs=4,seed=107,output,arena_root=None,resume=None):
    root=configure(arena_root);seed_everything(seed);output=Path(output)
    if family not in FAMILIES or steps<1 or not 1<=envs<=128:raise ValueError('Invalid tactical training configuration')
    if (output/'metadata.json').exists() and not resume:raise ValueError('Training output already exists')
    output.mkdir(parents=True,exist_ok=True);source=provenance(root)
    training_source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    environment=TacticalVecEnv(envs,family=family,seed=seed,arena_root=root)
    try:
        if resume:
            model,previous=load_tactical_policy(resume,root)
            if previous['family']!=family:raise ValueError('Resume family mismatch')
            model.set_env(environment)
        else:
            model=PPO('MlpPolicy',environment,seed=seed,device='cpu',n_steps=128,batch_size=128,n_epochs=6,
                      learning_rate=1e-3,gamma=.99,gae_lambda=.95,ent_coef=.01,
                      policy_kwargs={'net_arch':dict(pi=[96,96],vf=[96,96])})
        before=weight_hash(model);initial=model.num_timesteps;started=time.monotonic()
        model.learn(total_timesteps=steps,callback=Metrics(output/'metrics.jsonl'),reset_num_timesteps=False)
        model.save(output/'policy.zip');after=weight_hash(model)
        if after==before:raise RuntimeError('PPO did not update tactical weights')
        metadata={'schema':SCHEMA,'format':'arena-tactical-policy-v1','family':family,'model_id':3,'season':'2027',
                  'features':FEATURES,'action_nvec':ACTION_SPACE.nvec.tolist(),'action_hz':10,'seed':seed,
                  'actual_steps':model.num_timesteps,'additional_steps':model.num_timesteps-initial,
                  'ppo_updates':model._n_updates,'weights_before':before,'weights_after':after,'weights_changed':True,
                  'arena':source,'aim_mode':'SDK ballistic assistance with learned residuals',
                  'navigation_mode':'public SDK route hint; learned local mode, speed, gait and fire decisions',
                  'policy_sha256':hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest(),
                  'wall_seconds':time.monotonic()-started,'version':{'torch':torch.__version__,'numpy':np.__version__},
                  'episode_seed_range':[seed,environment.batch.next_seed-1],
                  'training_source_sha256':training_source}
        if resume:metadata['parent_policy_sha256']=previous['policy_sha256']
        dump(output/'metadata.json',metadata)
        print(json.dumps({'type':'trained_tactics','family':family,'steps':model.num_timesteps,
                          'weights_changed':True,'seconds':metadata['wall_seconds']}),flush=True)
        return metadata
    finally:environment.close()


def load_tactical_policy(checkpoint,root):
    path=Path(checkpoint);meta=json.loads((path/'metadata.json').read_text())
    if meta.get('schema')!=SCHEMA or meta.get('features')!=FEATURES:raise ValueError('Tactical policy schema mismatch')
    if hashlib.sha256((path/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:raise ValueError('Tactical checkpoint digest mismatch')
    check_provenance(meta['arena'],provenance(root))
    return PPO.load(path/'policy.zip',device='cpu'),meta


def actions_for_baseline(batch,ids,kind,rng):
    if kind=='random':return np.stack([rng.integers(ACTION_SPACE.nvec) for _ in ids])
    if kind not in ('idle','heuristic'):raise ValueError('Unknown tactical baseline')
    actions=[]
    for i in ids:
        ex=batch.executors[i];plan=ex.plan
        actions.append([int(kind=='heuristic' and plan['distance']>ex.command.radius_m),2,1,1,
                        int(kind=='heuristic' and plan['visible']),1,0])
    return np.asarray(actions)


def evaluate_tactics(*,checkpoint=None,family=None,episodes=32,envs=4,seed=3000000,arena_root=None,
                     baseline=None,output=None,replays=None):
    root=configure(arena_root);seed_everything(seed)
    if (checkpoint is None)==(baseline is None):raise ValueError('Choose a checkpoint or a baseline')
    if episodes<1 or not 1<=envs<=128:raise ValueError('Invalid evaluation size')
    model,meta=load_tactical_policy(checkpoint,root) if checkpoint else (None,None)
    family=family or (meta['family'] if meta else 'navigation')
    batch=TacticalBatch(min(envs,episodes),family=family,seed=seed,arena_root=root,record=bool(replays))
    rng=np.random.default_rng(seed);returns=np.zeros(batch.num_envs);rows=[];next_seed=seed+batch.num_envs
    try:
        observations=batch.reset();active=list(range(batch.num_envs))
        while active:
            actions=model.predict(observations[active],deterministic=True)[0] if model else actions_for_baseline(batch,active,baseline,rng)
            obs,rewards,terms,truncs,infos=batch.step(actions,env_ids=active)
            reset_ids=[];reset_seeds=[];remaining=[]
            for k,i in enumerate(active):
                observations[i]=obs[k];returns[i]+=rewards[k]
                if terms[k] or truncs[k]:
                    row={**infos[k],'return':float(returns[i]),'terminated':bool(terms[k]),'truncated':bool(truncs[k])}
                    if replays:
                        path=Path(replays)/f'{family}-{row["seed"]}.json';batch.engine.save_replay([path],env_ids=[i]);row['replay']=str(path)
                    rows.append(row)
                    if next_seed<seed+episodes:
                        reset_ids.append(i);reset_seeds.append(next_seed);next_seed+=1;returns[i]=0;remaining.append(i)
                else:remaining.append(i)
            if reset_ids:
                new=batch.reset(reset_ids,seeds=reset_seeds)
                for i,o in zip(reset_ids,new):observations[i]=o
            active=remaining
        result={'schema':SCHEMA,'evaluation_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'family':family,'episodes':episodes,'seed_range':[seed,seed+episodes-1],
                'checkpoint':str(checkpoint) if checkpoint else None,'baseline':baseline,
                'success_rate':sum(r['is_success'] for r in rows)/episodes,
                'mean_target_damage':float(np.mean([r['target_damage'] for r in rows])),
                'mean_return':float(np.mean([r['return'] for r in rows])),
                'skills':{skill:{'episodes':sum(r['skill']==skill for r in rows),
                    'successes':sum(r['skill']==skill and r['is_success'] for r in rows)} for skill in sorted(set(r['skill'] for r in rows))},
                'reasons':{reason:sum(r['reason']==reason for r in rows) for reason in sorted(set(r['reason'] for r in rows),key=str)},
                'results':sorted(rows,key=lambda r:r['seed'])}
        if output:dump(output,result)
        print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True);return result
    finally:batch.close()
