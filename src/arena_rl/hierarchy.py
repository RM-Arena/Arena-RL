"""One-Hz attack/defend PPO manager executing frozen ten-Hz robot skills."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from .bridge import configure,provenance,check_provenance
from .encoding import encode
from .envs import SkillBatch
from .learning import load_policy,seed_everything,weight_hash,Metrics,dump
from .tasks import sample,begin,robot

HIGH_SCHEMA='w3-manager-v1'
HIGH_OBS=spaces.Box(-10.,10.,shape=(14,),dtype=np.float32)
OPTIONS=('outpost','defend')


class HierarchicalVecEnv(VecEnv):
    """Curriculum of local outpost assaults and base-approach interceptions.

    The manager observes only red telemetry/vision. Reward can use the simulator's
    blue telemetry to measure an intrusion; that telemetry is never a policy input.
    No episode resets occur between options. Low policies cannot update in this env.
    """
    def __init__(self,skills,num_envs=4,seed=17,arena_root=None,record=False):
        self.root=configure(arena_root);self.skills=Path(skills).resolve()
        self.models={name:load_policy(self.skills/name,self.root)[0] for name in OPTIONS}
        self.skill_hashes={name:hashlib.sha256((self.skills/name/'policy.zip').read_bytes()).hexdigest() for name in OPTIONS}
        self.batch=SkillBatch(num_envs,task='outpost',seed=seed,arena_root=self.root,record=record)
        self.next_seed=seed;self.context=[None]*num_envs;self.pending=None
        self.steps=np.zeros(num_envs,dtype=int);self.returns=np.zeros(num_envs);self.low_lengths=np.zeros(num_envs,dtype=int)
        self.start_hp=np.zeros(num_envs);self.last_post=np.zeros(num_envs);self.last_attack=np.zeros(num_envs)
        self.damage=np.zeros(num_envs);self.options=np.zeros(num_envs,dtype=int);self.correct=np.zeros(num_envs,dtype=int)
        super().__init__(num_envs,HIGH_OBS,spaces.Discrete(2))

    def _observe(self,i):
        state=self.batch.states[i];obs=state['observations']['red'];r=robot(state)
        enemies=[e for e in obs['enemies'] if e['alive']]
        enemy=min(enemies,key=lambda e:np.linalg.norm([e['x']-r['x'],e['y']-r['y']]),default=None)
        post=obs['outposts']['blue']
        return np.clip(np.array([r['x']/28,r['y']/15,r['hp']/max(1,r['max_hp']),r['ammo']/200,
             r.get('heat',0)/max(1,r.get('max_heat',40)),float(enemy is not None),
             (enemy['x']-r['x'])/10 if enemy else 0,(enemy['y']-r['y'])/10 if enemy else 0,
             np.linalg.norm([post['x']-r['x'],post['y']-r['y']])/28,
             np.linalg.norm([4.1-r['x'],5.1-r['y']])/28,post['hp']/max(1,post['max_hp']),
             obs['bases']['red']['hp']/max(1,obs['bases']['red']['max_hp']),
             self.steps[i]/12,float(self.options[i])],dtype=np.float32),-10,10)

    def _reset_ids(self,ids,seeds=None):
        if seeds is None:
            seeds=list(range(self.next_seed,self.next_seed+len(ids)));self.next_seed+=len(ids)
        tasks=[]
        for seed in seeds:
            rng=np.random.default_rng(seed);context=int(rng.integers(0,2))
            t=sample(OPTIONS[context],int(seed));tasks.append((context,t))
        states=self.batch.engine.reset(seeds,env_ids=ids,scenarios=[t.scenario for _,t in tasks])
        for i,(context,t),state in zip(ids,tasks,states):
            self.context[i]=context;self.batch.tasks[i]=t;self.batch.states[i]=state;self.batch.invaders[i]=context==1;begin(t,state)
            self.steps[i]=0;self.returns[i]=0;self.low_lengths[i]=0;self.damage[i]=0;self.correct[i]=0
            self.options[i]=0
            self.start_hp[i]=robot(state)['hp']
            self.last_post[i]=state['observations']['red']['outposts']['blue']['hp']
            self.last_attack[i]=state['observations']['red']['teams']['red']['attack_damage']
            self.reset_infos[i]={'scenario':OPTIONS[context],'seed':int(seed)}
        return np.stack([self._observe(i) for i in ids])

    def reset(self):
        seeds=self._seeds if any(s is not None for s in self._seeds) else None
        obs=self._reset_ids(list(range(self.num_envs)),seeds)
        self._reset_seeds();self._reset_options();return obs

    def step_async(self,actions):
        actions=np.asarray(actions)
        if actions.shape!=(self.num_envs,) or not np.isin(actions,[0,1]).all():raise ValueError('One attack/defend option per robot required')
        self.pending=actions.astype(int)

    def step_wait(self):
        if self.pending is None:raise RuntimeError('step_async required')
        decisions=self.pending;self.pending=None
        for i,choice in enumerate(decisions):
            task=self.batch.tasks[i];task.name=OPTIONS[choice]
            task.goal=np.array([17.024,3.6] if choice==0 else [4.1,5.1])
            begin(task,self.batch.states[i]);task.steps=0;task.max_steps=600
            self.options[i]=choice;self.correct[i]+=int(choice==self.context[i])
        rewards=np.zeros(self.num_envs,dtype=np.float32);dones=np.zeros(self.num_envs,dtype=bool)
        successes=np.zeros(self.num_envs,dtype=bool);truncs=np.zeros(self.num_envs,dtype=bool)
        low_steps=np.zeros(self.num_envs,dtype=int)
        for _ in range(10):
            actions=np.zeros((self.num_envs,6),dtype=np.int64)
            for option,name in enumerate(OPTIONS):
                ids=np.flatnonzero(decisions==option)
                if len(ids):
                    observations=np.stack([encode(self.batch.tasks[i],self.batch.states[i]) for i in ids])
                    actions[ids]=self.models[name].predict(observations,deterministic=True)[0]
            states_before=self.batch.states.copy()
            active=np.flatnonzero(~dones).tolist()
            self.batch.step(actions[active],env_ids=active)
            for i in active:
                state=self.batch.states[i]
                low_steps[i]+=1;obs=state['observations']['red'];r=robot(state)
                post=obs['outposts']['blue']['hp'];attack=obs['teams']['red']['attack_damage']
                post_damage=max(0.,self.last_post[i]-post)
                enemy_damage=max(0.,attack-self.last_attack[i]-post_damage)
                increment=post_damage if self.context[i]==0 else enemy_damage
                self.damage[i]+=increment
                # Rewards depend on physical damage and survival, never a label for the chosen option.
                rewards[i]+=.12*increment-.01
                rewards[i]-=.03*max(0.,1.-r['upright'])
                rewards[i]-=.08*max(0.,robot(states_before[i])['hp']-r['hp'])
                self.last_post[i]=post;self.last_attack[i]=attack
                intruder=robot(state,'blue')
                breached=self.context[i]==1 and intruder['x']<4.8 and intruder['alive']
                success=self.damage[i]>=40
                failure=not r['alive'] or r['upright']<.25 or breached
                if success or failure or state['terminated'] or state['truncated']:
                    dones[i]=True;successes[i]=success;truncs[i]=state['truncated'] and not(success or failure)
                    rewards[i]+=8. if success else -4. if failure else 0.
            if dones.all():break
        self.steps+=1;self.low_lengths+=low_steps;self.returns+=rewards
        for i in range(self.num_envs):
            if not dones[i] and self.steps[i]>=12:dones[i]=True;truncs[i]=True
        obs=np.stack([self._observe(i) for i in range(self.num_envs)])
        infos=[]
        for i in range(self.num_envs):
            info={'task':'manager','scenario':OPTIONS[self.context[i]],'option':OPTIONS[decisions[i]],
                  'is_success':bool(successes[i]),'target_damage':float(self.damage[i]),
                  'decision_accuracy':float(self.correct[i]/self.steps[i]),'low_steps':int(low_steps[i]),
                  'seed':self.batch.tasks[i].seed,'TimeLimit.truncated':bool(truncs[i])}
            if dones[i]:
                info['terminal_observation']=obs[i].copy();info['episode']={'r':float(self.returns[i]),'l':int(self.steps[i])}
            infos.append(info)
        ids=np.flatnonzero(dones).tolist()
        # Evaluation can export terminal replay files before actors reset.
        if self.on_episode_end:
            for i in ids:self.on_episode_end(i,infos[i])
        if ids:
            reset=self._reset_ids(ids)
            for i,new in zip(ids,reset):obs[i]=new
        return obs,rewards,dones,infos

    on_episode_end=None
    def close(self):self.batch.close()
    def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name,None) for _ in self._get_indices(indices)]
    def set_attr(self,attr_name,value,indices=None):raise AttributeError('Configuration is immutable')
    def env_method(self,method_name,*args,indices=None,**kwargs):raise AttributeError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]


def train_high(*,skills,steps=4096,envs=4,seed=17,output,arena_root=None):
    root=configure(arena_root);seed_everything(seed);output=Path(output)
    if steps<1 or not 1<=envs<=128:raise ValueError('Positive steps and 1..128 actors required')
    if (output/'metadata.json').exists():raise ValueError('Output run already exists')
    output.mkdir(parents=True,exist_ok=True)
    environment=HierarchicalVecEnv(skills,envs,seed,root)
    try:
        frozen={name:weight_hash(m) for name,m in environment.models.items()}
        model=PPO('MlpPolicy',environment,seed=seed,device='cpu',n_steps=64,batch_size=64,n_epochs=6,
                  learning_rate=1e-3,ent_coef=.01,gamma=.98,policy_kwargs={'net_arch':dict(pi=[32,32],vf=[32,32])})
        before=weight_hash(model);started=time.monotonic()
        model.learn(total_timesteps=steps,callback=Metrics(output/'metrics.jsonl'))
        model.save(output/'policy.zip')
        after=weight_hash(model)
        if before==after:raise RuntimeError('Manager network did not update')
        if any(weight_hash(environment.models[n])!=h for n,h in frozen.items()):raise RuntimeError('Frozen skill changed')
        meta={'format':'arena-rl-manager-v1','schema':HIGH_SCHEMA,'algorithm':'stable-baselines3.PPO',
              'seed':seed,'actual_steps':model.num_timesteps,'ppo_updates':model._n_updates,
              'manager_hz':1,'skill_hz':10,'max_low_steps_per_option':10,
              'skills':str(Path(skills).resolve()),'skill_sha256':environment.skill_hashes,
              'arena':provenance(root),'weights_before':before,'weights_after':after,'weights_changed':True,
              'skills_frozen':True,'wall_seconds':time.monotonic()-started,
              'policy_sha256':hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest()}
        dump(output/'metadata.json',meta);print(json.dumps({'type':'trained_high','output':str(output),'steps':model.num_timesteps,'seconds':meta['wall_seconds']}),flush=True)
        return meta
    finally:environment.close()


def evaluate_high(*,checkpoint,episodes=16,envs=4,seed=2000000,output=None,replays=None,arena_root=None):
    root=configure(arena_root);seed_everything(seed);checkpoint=Path(checkpoint)
    meta=json.loads((checkpoint/'metadata.json').read_text());check_provenance(meta['arena'],provenance(root))
    if meta['schema']!=HIGH_SCHEMA:raise ValueError('Manager schema mismatch')
    if hashlib.sha256((checkpoint/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:raise ValueError('Manager digest mismatch')
    # Checkpoint skills are stored beside the manager, so a run directory can be moved.
    skills=checkpoint.parent if all((checkpoint.parent/n/'policy.zip').exists() for n in OPTIONS) else Path(meta['skills'])
    environment=HierarchicalVecEnv(skills,min(envs,episodes),seed,root,record=bool(replays))
    try:
        if environment.skill_hashes!=meta['skill_sha256']:raise ValueError('Low-level policy digest mismatch')
        model=PPO.load(checkpoint/'policy.zip',device='cpu');obs=environment.reset();rows=[]
        def ended(i,info):
            if replays and seed<=info['seed']<seed+episodes:
                path=Path(replays)/f'manager-{info["seed"]}.json'
                environment.batch.engine.save_replay([path],env_ids=[i]);info['replay']=str(path)
        environment.on_episode_end=ended
        while len(rows)<episodes:
            obs,_,dones,infos=environment.step(model.predict(obs,deterministic=True)[0])
            rows += [{k:v for k,v in info.items() if k!='terminal_observation'} for done,info in zip(dones,infos)
                     if done and seed<=info['seed']<seed+episodes]
        result={'schema':HIGH_SCHEMA,'episodes':episodes,'seed_range':[seed,seed+episodes-1],
                'success_rate':sum(r['is_success'] for r in rows)/episodes,
                'decision_accuracy':float(np.mean([r['decision_accuracy'] for r in rows])),
                'scenarios':{name:{'episodes':sum(r['scenario']==name for r in rows),
                    'successes':sum(r['scenario']==name and r['is_success'] for r in rows)} for name in OPTIONS},
                'results':sorted(rows,key=lambda r:r['seed'])}
        if output:dump(output,result)
        print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True);return result
    finally:environment.close()
