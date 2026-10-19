"""Learned dive timing for 2027 aerial robots ramming outpost armor.

One flying robot of a hangar is learned per simulator; the others stay docked. The
policy picks a manoeuvre every 0.1 s (hold the standoff, radial dive, dive at the
nearest plate, back off, orbit, dock); Arena's flight physics, battery, guard wear
and strike validity (>=1.5 m/s normal approach, 0.2 s plate interval) decide the
outcome. Episodes cover the rotating outpost (before 180 s) and the stopped rotor.
At runtime the same policy runs independently for every drone of the hangar.
"""
import functools
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .bridge import configure,provenance,check_provenance
from .executor import own

SCHEMA='arena-aerial-v1'
MODELS=(2761,2762,2763)
MANEUVERS=('hold','dive_radial','dive_plate','back_off','orbit_plus','orbit_minus','dock')
STATES=('docked','flying','down','disabled')
OBS_SPACE=spaces.Box(-10.,10.,shape=(48,),dtype=np.float32)
ACTION_SPACE=spaces.Discrete(len(MANEUVERS))
STANDOFF=1.1


def target_outpost(observation):
    team=observation['team']
    return next((o for o in observation['outposts'].values() if o['hp']>0 and o.get('owner')!=team),None)


def _plates(observation):
    from rmuc_arena import sdk
    return sdk.hostile_plates(observation,'outpost')


def encode(observation,slot,index,previous,dive_s):
    robot=own(observation,slot);drone=next(d for d in robot.get('drones',[]) if d.get('index')==index)
    outpost=target_outpost(observation);values=[float(drone['state']==s) for s in STATES]
    p=np.array([drone['x'],drone['y'],drone['z']])
    if outpost is None:return np.zeros(OBS_SPACE.shape,dtype=np.float32)
    c=np.array([outpost['x'],outpost['y']]);radial=p[:2]-c;r=float(np.linalg.norm(radial));u=radial/max(1e-6,r)
    plates=sorted(_plates(observation),key=lambda q:float(np.linalg.norm(np.asarray(q['center'])-p)))[:3]
    height=plates[0]['center'][2] if plates else 1.2
    values+=[r/3,(p[2]-height)/1.,u[0],u[1],drone['vx']/6,drone['vy']/6,drone['vz']/3,
             float(np.dot([drone['vx'],drone['vy']],-u))/6,drone['battery_wh']/4,drone['integrity']/6,
             float(observation['time']<drone.get('recover_until',0.)),min(2.,observation['time']-drone.get('last_strike',-1e3))/2,
             float(not outpost.get('rotation_stopped',False)),outpost['hp']/1000]
    spin=0.8*math.pi*min(1.,observation['time']/5.) if not outpost.get('rotation_stopped',False) else 0.
    for k in range(3):
        if k<len(plates):
            q=plates[k];delta=(np.asarray(q['center'])-p)/3;n=np.asarray(q['normal']);qr=np.asarray(q['center'][:2])-c
            angle=math.atan2(qr[1],qr[0])-math.atan2(u[1],u[0]);angle=(angle+math.pi)%math.tau-math.pi
            values+=[*delta.tolist(),n[0],n[1],math.sin(angle),math.cos(angle)]
        else:values+=[0.]*7
    values+=[spin/2.6,min(2.,dive_s)/2]+[float(previous==m) for m in range(len(MANEUVERS))]
    out=np.clip(np.nan_to_num(np.asarray(values,dtype=np.float32)),-10,10)
    if out.shape!=OBS_SPACE.shape:raise RuntimeError(f'Aerial schema mismatch {out.shape}')
    return out


def command(observation,slot,index,action,bearing):
    """Map a manoeuvre to the hangar's drone command for one aerial robot."""
    robot=own(observation,slot);drone=next(d for d in robot.get('drones',[]) if d.get('index')==index)
    outpost=target_outpost(observation)
    from .aerial import detour,dock_or_detour
    if outpost is None or MANEUVERS[action]=='dock':return dock_or_detour(drone,robot),bearing
    plates=_plates(observation);p=np.array([drone['x'],drone['y'],drone['z']]);c=np.array([outpost['x'],outpost['y']])
    height=min(plates,key=lambda q:float(np.linalg.norm(np.asarray(q['center'])-p)))['center'][2] if plates else 1.2
    if bearing is None:
        radial=p[:2]-c;bearing=math.atan2(radial[1],radial[0]) if np.linalg.norm(radial)>.3 else float(index)*2.1
    name=MANEUVERS[action]
    if name=='orbit_plus':bearing+=.35
    elif name=='orbit_minus':bearing-=.35
    u=np.array([math.cos(bearing),math.sin(bearing)])
    if name=='dive_radial':return {'move':[-u[0],-u[1],float(np.clip((height-p[2])*2,-1,1))]},bearing
    if name=='dive_plate' and plates:
        q=min(plates,key=lambda q:float(np.linalg.norm(np.asarray(q['center'])-p)));d=np.asarray(q['center'])-p;d=d/max(1e-6,np.linalg.norm(d))
        return {'move':[float(d[0]),float(d[1]),float(np.clip(d[2]*2,-1,1))]},bearing
    radius=1.7 if name=='back_off' else STANDOFF;stand=c+u*radius
    return {'goto':detour(p,[float(stand[0]),float(stand[1]),float(height)])},bearing


class AerialEnv(gym.Env):
    metadata={'render_modes':[]}
    observation_space=OBS_SPACE
    action_space=ACTION_SPACE

    def __init__(self,seed=0,models=None,arena_root=None,max_steps=350):
        super().__init__();configure(arena_root)
        self.next_seed=int(seed);self.models=tuple(models or MODELS);self.max_steps=max_steps
        self.engine=None;self.model=None;self.state=None;self.index=None;self.bearing=None;self.previous=0;self.dive_s=0.;self.steps=0;self.expert_dive=-1.

    def _obs(self):return self.state['observations']['red']

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        from rmuc_arena.env import ArenaEnv
        episode=int(seed) if seed is not None else self.next_seed
        if seed is None:self.next_seed+=1
        rng=np.random.default_rng(episode);model=int(self.models[int(rng.integers(0,len(self.models)))])
        if self.engine is None or model!=self.model:
            # One simulator per hangar model, kept for the life of the environment.
            engines=self.__dict__.setdefault('engines',{})
            if self.engine is not None:engines[self.model]=self.engine
            self.engine=engines.pop(model,None) or ArenaEnv(season='2027',duration=420,lineups={'red':[{'slot':6,'model':model}],'blue':[{'slot':3,'model':2731}]},compact=True)
            self.model=model
        self.state=self.engine.reset(seed=episode,scenario={'robots':{'blue-3':{'x':25.,'y':5.4,'heading':math.pi,'ammo':0}}})
        drones=own(self._obs(),'6').get('drones',[]);self.index=int(drones[int(rng.integers(0,len(drones)))]['index'])
        self.variant='static' if rng.random()<.4 else 'rotating'
        if self.variant=='static':
            # Wait for the rotor to stop at 180 s with a coarse repeat; nothing else moves.
            self.engine.action_repeat=20
            try:
                while self._obs()['time']<181:self.state=self.engine.step({})
            finally:self.engine.action_repeat=2
        else:
            for _ in range(int(rng.integers(0,40))):self.state=self.engine.step({})
        self.bearing=float(rng.uniform(-math.pi,math.pi));self.previous=0;self.dive_s=0.;self.steps=0;self.expert_dive=-1.;self.damage=0.;self.strikes=0
        self.start_strikes=self._drone()['strikes'];self.start_hp=target_outpost(self._obs())['hp']
        return encode(self._obs(),'6',self.index,self.previous,self.dive_s),{'model':model,'variant':self.variant,'drone':self.index}

    def _drone(self):return next(d for d in own(self._obs(),'6')['drones'] if d.get('index')==self.index)

    def expert_action(self):
        """The explicit guidance law: line up at the standoff, then commit to a radial dive."""
        o=self._obs();drone=self._drone();outpost=target_outpost(o)
        if outpost is None:return 6
        now=o['time']
        if self.expert_dive>=now:return 1
        if drone.get('last_strike',-1e3)>now-.6:return 3
        c=np.array([outpost['x'],outpost['y']]);plates=_plates(o);height=plates[0]['center'][2] if plates else 1.2
        if outpost.get('rotation_stopped') and plates:
            # A stopped rotor: line up on the nearest plate's own bearing (its outward normal).
            nearest=min(plates,key=lambda q:abs((math.atan2(q['center'][1]-c[1],q['center'][0]-c[0])-self.bearing+math.pi)%math.tau-math.pi))
            target=math.atan2(nearest['center'][1]-c[1],nearest['center'][0]-c[0])
            error=(target-self.bearing+math.pi)%math.tau-math.pi
            if abs(error)>.03:return 4 if error>0 and abs(error)>=.35 else 5 if abs(error)>=.35 else self._nudge(error)
        u=np.array([math.cos(self.bearing),math.sin(self.bearing)]);stand=c+u*STANDOFF
        if np.linalg.norm(np.array([drone['x'],drone['y']])-stand)<.1 and abs(drone['z']-height)<.08:
            self.expert_dive=now+1.2;return 1
        return 0

    def _nudge(self,error):
        # Small bearing corrections are applied by the host before holding the standoff.
        self.bearing+=error;return 0

    def step(self,action):
        action=int(np.asarray(action).reshape(-1)[0]);o=self._obs()
        move,self.bearing=command(o,'6',self.index,action,self.bearing)
        self.dive_s=self.dive_s+.1 if MANEUVERS[action].startswith('dive') else 0.;self.previous=action
        hp_before=(target_outpost(o) or {'hp':0})['hp'];strikes_before=self._drone()['strikes']
        self.state=self.engine.step({'red':{'6':{'drones':{str(self.index):move}}}});self.steps+=1
        o=self._obs();drone=self._drone();outpost=target_outpost(o)
        hit=drone['strikes']-strikes_before;damage=max(0.,hp_before-(outpost['hp'] if outpost else 0.))
        self.strikes+=hit;self.damage+=damage
        reward=4.*hit-.004*float(drone['state']=='flying')-1.5*float(drone['state']=='down')
        destroyed=outpost is None
        term=bool(destroyed or drone['state'] in ('down','disabled') or drone['battery_wh']<.45 or self.state['terminated'])
        trunc=bool(not term and (self.steps>=self.max_steps or self.state['truncated']))
        info={'task':'aerial-'+self.variant,'model':self.model,'variant':self.variant,'drone':self.index,'strikes':self.strikes,'damage':self.damage,
              'first_hit':None,'is_success':self.strikes>0,'steps':self.steps,'state':drone['state']}
        return encode(o,'6',self.index,self.previous,self.dive_s),float(reward),term,trunc,info

    def close(self):
        from .ground_envs import dispose
        for engine in list(self.__dict__.get('engines',{}).values())+[self.engine]:
            if engine is not None:dispose(engine)
        self.engine=None


def _make(seed,models,arena_root):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(AerialEnv(seed=seed,models=models,arena_root=arena_root),info_keywords=('strikes','damage','variant','model'))


def train_aerial(*,steps=204800,envs=8,seed=811,output,arena_root=None,device='auto',expert_episodes=96,bc_epochs=40,learning_rate=1e-4):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from .ground_learning import resolve_device,seed_learner,behavior_clone
    from .learning import dump,weight_hash
    root=configure(arena_root);device=resolve_device(device);output=Path(output)
    if (output/'metadata.json').exists():raise ValueError('Aerial training output already exists')
    saved=output/'demonstrations.npz'
    output.mkdir(parents=True,exist_ok=True);seed_learner(seed,device,4);started=time.monotonic()
    vec=SubprocVecEnv([functools.partial(_make,seed+i*100_003,MODELS,str(root)) for i in range(envs)],start_method='spawn')
    try:
        model=PPO('MlpPolicy',vec,n_steps=256,batch_size=512,n_epochs=6,learning_rate=learning_rate,gamma=.98,gae_lambda=.95,clip_range=.15,
                  ent_coef=.005,seed=seed,device=device,policy_kwargs={'net_arch':dict(pi=[128,128],vf=[128,128])},verbose=0)
        before=weight_hash(model)
        # Successful guidance-law episodes initialize the network before PPO.
        obs=vec.reset();buffers=[[] for _ in range(envs)];x=[];y=[];accepted=0;episodes=0
        if saved.is_file():
            # Resume after an interrupted run: the successful expert dataset is already on disk.
            data=np.load(saved);x=list(data['observations']);y=list(data['actions'][:,0]);accepted=-1
        while 0<=accepted<expert_episodes and episodes<expert_episodes*4:
            actions=np.array(vec.env_method('expert_action'))
            for i in range(envs):buffers[i].append((obs[i].copy(),int(actions[i])))
            obs,_,dones,infos=vec.step(actions)
            for i,done in enumerate(dones):
                if done:
                    episodes+=1
                    if infos[i].get('strikes',0)>0:accepted+=1;x+=[o for o,_ in buffers[i]];y+=[a for _,a in buffers[i]]
                    buffers[i]=[]
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.int64)[:,None]
        np.savez_compressed(output/'demonstrations.npz',observations=x,actions=y)
        bc=behavior_clone(model,x,y[:,0],bc_epochs,seed)
        demonstrations={'episodes':accepted if accepted>=0 else None,'reused_dataset':accepted<0,'attempted':episodes,'samples':int(len(x)),**bc,'dataset_sha256':hashlib.sha256((output/'demonstrations.npz').read_bytes()).hexdigest()}
        print(json.dumps({'type':'aerial_behavior_cloning',**demonstrations}),flush=True)
        model.save(output/'demonstration-policy.zip');before_ppo=weight_hash(model)
        from .learning import Metrics
        model.learn(total_timesteps=steps,callback=Metrics(output/'metrics.jsonl'),reset_num_timesteps=False)
        model.save(output/'policy.zip');after=weight_hash(model)
        if after==before_ppo:raise RuntimeError('PPO did not update aerial weights')
        metadata={'format':'arena-aerial-policy-v1','schema':SCHEMA,'maneuvers':list(MANEUVERS),'observation_dim':OBS_SPACE.shape[0],'models':list(MODELS),
                  'seed':seed,'envs':envs,'device':device,'actual_steps':model.num_timesteps,'ppo_updates':model._n_updates,'learning_rate':learning_rate,
                  'weights_before':before,'weights_before_ppo':before_ppo,'weights_after':after,'demonstrations':demonstrations,
                  'arena':provenance(root),'wall_seconds':time.monotonic()-started,'policy_sha256':hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest(),
                  'versions':{'torch':torch.__version__,'numpy':np.__version__}}
        dump(output/'metadata.json',metadata);return metadata
    finally:vec.close()


def load_aerial_policy(checkpoint,root,device='cpu'):
    from stable_baselines3 import PPO
    path=Path(checkpoint);meta=json.loads((path/'metadata.json').read_text())
    if meta.get('schema')!=SCHEMA or meta.get('observation_dim')!=OBS_SPACE.shape[0]:raise ValueError('Aerial policy schema mismatch')
    if hashlib.sha256((path/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:raise ValueError('Aerial checkpoint digest mismatch')
    check_provenance(meta['arena'],provenance(root))
    return meta,PPO.load(path/'policy.zip',device=device)


def evaluate_aerial(*,checkpoint=None,episodes=24,seed=7_300_000,output=None,arena_root=None,modes=('policy','expert','random')):
    from .learning import dump
    root=configure(arena_root);meta,model=load_aerial_policy(checkpoint,root) if checkpoint else (None,None)
    env=AerialEnv(seed=seed,arena_root=root);report={'format':'arena-aerial-evaluation-v1','checkpoint':str(checkpoint) if checkpoint else None,'episodes':episodes,'modes':{}}
    rng=np.random.default_rng(seed)
    try:
        for mode in modes:
            if mode=='policy' and model is None:continue
            rows=[]
            for k in range(episodes):
                obs,info=env.reset(seed=seed+k);done=False;first=None
                while not done:
                    action=model.predict(obs,deterministic=True)[0] if mode=='policy' else env.expert_action() if mode=='expert' else int(rng.integers(0,len(MANEUVERS)))
                    obs,_,term,trunc,info=env.step(action);done=term or trunc
                    if first is None and info['strikes']>0:first=info['steps']/10
                rows.append({**info,'first_hit_s':first,'seed':seed+k})
            hits=[r['strikes'] for r in rows];first=[r['first_hit_s'] for r in rows if r['first_hit_s'] is not None]
            report['modes'][mode]={'mean_strikes':float(np.mean(hits)),'mean_damage':float(np.mean([r['damage'] for r in rows])),'episodes_with_hit':sum(h>0 for h in hits),
                'median_first_hit_s':float(np.median(first)) if first else None,
                'by_variant':{v:float(np.mean([r['strikes'] for r in rows if r['variant']==v])) for v in sorted({r['variant'] for r in rows})},'results':rows}
            print(json.dumps({'type':'aerial_evaluated','mode':mode,**{k:v for k,v in report['modes'][mode].items() if k!='results'}}),flush=True)
    finally:env.close()
    if output:dump(output,report)
    return report
