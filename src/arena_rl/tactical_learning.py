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


def train_tactics(*,family,steps=32768,envs=4,seed=107,output,arena_root=None,resume=None,expert_episodes=None,model_id=3,initialize_from=None):
    root=configure(arena_root);seed_everything(seed);output=Path(output)
    if family not in FAMILIES or steps<1 or not 1<=envs<=128:raise ValueError('Invalid tactical training configuration')
    if (output/'metadata.json').exists() and not resume:raise ValueError('Training output already exists')
    output.mkdir(parents=True,exist_ok=True);source=provenance(root)
    training_source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    features=FEATURES;schema=SCHEMA
    if family in ('support','flight'):
        from .extended_encoding import FEATURES as features,SCHEMAS
        schema=SCHEMAS[family]
    if family=='mobility' or model_id==305 and family=='navigation':
        from .mobility_encoding import FEATURES as features,SCHEMA as schema
        if family=='navigation':schema='serial305-navigation-v1'
    environment=TacticalVecEnv(envs,family=family,seed=seed,arena_root=root,model_id=model_id)
    try:
        initialization=None
        if initialize_from:
            if resume:raise ValueError('Choose initialization or resume')
            model,initialization=load_tactical_policy(initialize_from,root)
            if initialization.get('model_id',3)!=model_id or initialization['features']!=features:raise ValueError('Initialization model/schema mismatch')
            model.set_env(environment);model.num_timesteps=0;model._n_updates=0;model.policy.optimizer.state.clear()
        elif resume:
            model,previous=load_tactical_policy(resume,root)
            if previous['family']!=family or previous.get('model_id',3)!=model_id:raise ValueError('Resume family mismatch')
            model.set_env(environment)
        else:
            model=PPO('MlpPolicy',environment,seed=seed,device='cpu',n_steps=128,batch_size=128,n_epochs=6,
                      learning_rate=1e-3,gamma=.99,gae_lambda=.95,ent_coef=.01,
                      policy_kwargs={'net_arch':dict(pi=[96,96],vf=[96,96])})
        before=weight_hash(model);demonstrations=None
        count=(64 if (family in ('flight','mobility') or model_id==305) and not resume and not initialize_from else 0) if expert_episodes is None else expert_episodes
        if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=2048:raise ValueError('Invalid demonstration count')
        if count:
            if resume:raise ValueError('Demonstration initialization requires a new policy')
            from .demonstrations import initialize_from_demonstrations
            demonstrations=initialize_from_demonstrations(model,family=family,episodes=count,seed=seed+100000,arena_root=root,output=output,model_id=model_id)
            print(json.dumps({'type':'demonstrations',**demonstrations}),flush=True)
        if family=='flight' or demonstrations or initialize_from:
            model.learning_rate=2e-5;model.lr_schedule=lambda _:2e-5;model.ent_coef=.001
        before_ppo=weight_hash(model);initial=model.num_timesteps;started=time.monotonic()
        model.learn(total_timesteps=steps,callback=Metrics(output/'metrics.jsonl'),reset_num_timesteps=False)
        model.save(output/'policy.zip');after=weight_hash(model)
        if after==before_ppo:raise RuntimeError('PPO did not update tactical weights')
        metadata={'schema':schema,'format':'arena-tactical-policy-v1','family':family,'model_id':model_id,'season':environment.batch.season,
                  'features':features,'action_nvec':ACTION_SPACE.nvec.tolist(),'action_hz':10,'seed':seed,
                  'actual_steps':model.num_timesteps,'additional_steps':model.num_timesteps-initial,
                  'ppo_updates':model._n_updates,'weights_before':before,'weights_before_ppo':before_ppo,'ppo_weights_changed':after!=before_ppo,'weights_after':after,'weights_changed':True,
                  'arena':source,'aim_mode':'SDK ballistic assistance with learned residuals',
                  'navigation_mode':'public SDK route hint; learned local mode, speed, gait and fire decisions',
                  'policy_sha256':hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest(),
                  'wall_seconds':time.monotonic()-started,'version':{'torch':torch.__version__,'numpy':np.__version__},
                  'episode_seed_range':[seed,environment.batch.next_seed-1],
                  'training_source_sha256':training_source}
        if initialization:metadata['initialization_policy_sha256']=initialization['policy_sha256']
        if demonstrations:metadata['demonstrations']=demonstrations
        if resume:metadata['parent_policy_sha256']=previous['policy_sha256']
        dump(output/'metadata.json',metadata)
        print(json.dumps({'type':'trained_tactics','family':family,'steps':model.num_timesteps,
                          'weights_changed':True,'seconds':metadata['wall_seconds']}),flush=True)
        return metadata
    finally:environment.close()


def load_tactical_policy(checkpoint,root):
    path=Path(checkpoint);meta=json.loads((path/'metadata.json').read_text())
    expected=SCHEMA;features=FEATURES
    if meta.get('family') in ('support','flight'):
        from .extended_encoding import SCHEMAS,FEATURES as features
        expected=SCHEMAS[meta['family']]
    if meta.get('family')=='mobility' or meta.get('model_id')==305 and meta.get('family')=='navigation':
        from .mobility_encoding import FEATURES as features,SCHEMA as expected
        if meta['family']=='navigation':expected='serial305-navigation-v1'
    if meta.get('schema')!=expected or meta.get('features')!=features:raise ValueError('Tactical policy schema mismatch')
    if hashlib.sha256((path/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:raise ValueError('Tactical checkpoint digest mismatch')
    check_provenance(meta['arena'],provenance(root))
    return PPO.load(path/'policy.zip',device='cpu'),meta


def actions_for_baseline(batch,ids,kind,rng):
    if kind=='random':return np.stack([rng.integers(ACTION_SPACE.nvec) for _ in ids])
    if kind not in ('idle','heuristic'):raise ValueError('Unknown tactical baseline')
    actions=[]
    for i in ids:
        ex=batch.executors[i];plan=ex.plan
        jump=0;speed=2;leg=1;interaction=int(kind=='heuristic' and plan['visible'])
        if batch.model_id==305 and ex.command.skill in ('navigate','retreat'):
            speed=3 if plan['distance']>1.2 else 1
            from .executor import own
            r=own(batch.states[i]['observations']['red'])
            if any(t['kind']=='bumps' and t['x']-.5<r['x']<t['x']+t['w']+.5 and t['y']-.5<r['y']<t['y']+t['h']+.5 for t in ex.arena['terrain']):leg=2;speed=3
        if batch.model_id==305 and ex.route_state.get('escape_until',0)>ex.time:leg=2;speed=1
        if ex.command.skill=='resupply':interaction=int(kind=='heuristic')
        if ex.command.skill=='fly_slope':
            from .executor import own
            r=own(batch.states[i]['observations']['red'])
            from rmuc_arena import sdk
            lane=sdk.fly_slope_lane(ex.arena,ex.command.lane_team);axis=np.array(lane['crest'])-lane['lip'];axis=axis/np.linalg.norm(axis)
            progress=np.dot([r['x']-lane['crest'][0],r['y']-lane['crest'][1]],axis)
            reverse=ex.command.flight_direction=='reverse'
            speed=(2 if reverse else 3) if ex.skill_state.get('launched') else 0;interaction=1
            if not reverse and progress>(1.05 if batch.model_id==305 else .6):speed=0
            if not reverse and batch.model_id==305:jump=int(-.2<progress<.35)
            jump=int(.1<progress<1.05) if reverse else jump
            if reverse and batch.model_id==305:
                speed=1 if ex.skill_state.get('launched') else 0;leg=0
                if any(v.get('type')=='jump' and v.get('robot')=='red-3' for v in batch.states[i]['info'].get('events',[])):ex.skill_state['demo_jumped']=True
                jump=int(.1<progress<1.2 and not ex.skill_state.get('demo_jumped'))
            if ex.skill_state.get('launched') and plan['distance']<1.:speed=0
        if ex.command.skill=='recover' or ex.plan['phase']=='recover':
            speed=0;interaction=0;jump=0
        if ex.command.skill=='retreat' and batch.model_id!=305:speed=1 if plan['distance']<1.2 else 2
        threshold=.025 if ex.command.skill=='fly_slope' else ex.command.radius_m
        actions.append([int(kind=='heuristic' and ex.plan['phase']!='recover' and plan['distance']>(.3 if ex.command.skill=='fly_slope' and ex.skill_state.get('launched') else threshold)),speed,1,1,interaction,leg,jump])
    return np.asarray(actions)


def evaluate_tactics(*,checkpoint=None,family=None,episodes=32,envs=4,seed=3000000,arena_root=None,
                     baseline=None,output=None,replays=None,model_id=None):
    root=configure(arena_root);seed_everything(seed)
    if (checkpoint is None)==(baseline is None):raise ValueError('Choose a checkpoint or a baseline')
    if episodes<1 or not 1<=envs<=128:raise ValueError('Invalid evaluation size')
    model,meta=load_tactical_policy(checkpoint,root) if checkpoint else (None,None)
    family=family or (meta['family'] if meta else 'navigation')
    batch=TacticalBatch(min(envs,episodes),family=family,seed=seed,arena_root=root,record=bool(replays),model_id=meta.get('model_id',3) if meta else model_id or (305 if family=='mobility' else 3))
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
        schema=meta['schema'] if meta else SCHEMA
        if not meta and family=='mobility':
            from .mobility_encoding import SCHEMA as schema
        if not meta and family in ('support','flight'):
            from .extended_encoding import SCHEMAS
            schema=SCHEMAS[family]
        result={'schema':schema,'evaluation_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'family':family,'episodes':episodes,'seed_range':[seed,seed+episodes-1],
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
