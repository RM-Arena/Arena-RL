"""Goal-conditioned training and live single-robot sessions on VectorArena."""
from dataclasses import dataclass
import math
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from .bridge import configure
from .commands import Command
from .executor import TacticalExecutor,own,xy
from .tactical_encoding import OBS_SPACE,ACTION_SPACE,encode,decode

FAMILIES=('navigation','combat','defense','support','flight')
LINEUPS={'red':[{'slot':3,'model':3}],
         'blue':[{'slot':1,'model':2711},{'slot':3,'model':3},{'slot':4,'model':4}]}


@dataclass
class Episode:
    family:str
    seed:int
    scenario:dict
    command:dict
    behavior:dict
    max_steps:int=300


def sample_episode(family,seed):
    if family in ('support','flight'):
        from .extended_curricula import sample
        return sample(family,seed)
    if family not in FAMILIES:raise ValueError('Unknown tactical policy family')
    rng=np.random.default_rng(seed)
    blue={'blue-1':{'x':25.,'y':9.5,'heading':math.pi,'ammo':0},
          'blue-3':{'x':25.,'y':5.2,'heading':math.pi,'ammo':0},
          'blue-4':{'x':23.5,'y':9.5,'heading':math.pi,'ammo':0}}
    behavior={};red=np.array([4.7,5.15])+rng.uniform(-.3,.3,2)
    command={'id':f'episode-{seed}','skill':'navigate','point':[7.8,5.15],
             'timeout_s':28.,'target_lost_timeout_s':3.,'low_hp_fraction':.1}
    if family=='navigation':
        # Four open routes plus a route across the low fortress and the real road ramp.
        routes=[([4.3,5.1],[8.2,5.1]),([4.3,9.8],[8.2,9.8]),
                ([19.7,5.1],[23.,5.1]),([19.7,9.8],[23.,9.8]),
                ([4.8,7.5],[8.4,7.5]),([7.42,11.25],[5.4,12.3]),
                ([4.6,13.7],[8.2,13.7]),([19.8,1.3],[23.4,1.3]),
                ([4.5,5.15],[19.4,4.9]),([19.4,4.9],[4.5,5.15])]
        start,goal=routes[int(rng.integers(0,len(routes)))];red=np.array(start)+rng.uniform(-.12,.12,2)
        if rng.random()<.4:start,goal=goal,start;red=np.array(start)+rng.uniform(-.12,.12,2)
        command['point']=(np.array(goal)+rng.uniform(-.12,.12,2)).tolist()
        if np.linalg.norm(red-command['point'])>8:command['timeout_s']=75.
    elif family=='combat':
        if rng.random()<.35:
            red=np.array([19.5,4.8])+rng.uniform(-.5,.5,2)
            command.update(skill='attack_structure',target_id='blue_outpost',damage_goal=60.,preferred_range_m=float(rng.uniform(2.1,3.2)))
            command.pop('point')
        else:
            identity=str(rng.choice(['blue-1','blue-3','blue-4']))
            red=np.array([4.5,5.1])+rng.uniform(-.2,.2,2)
            blue[identity]={'x':float(rng.uniform(7.1,9.0)),'y':float(rng.uniform(4.9,5.6)),'heading':math.pi,'ammo':0}
            # A closer distractor must never replace an explicitly selected target.
            distractor='blue-3' if identity!='blue-3' else 'blue-4'
            blue[distractor]={'x':6.3,'y':5.65,'heading':math.pi,'ammo':0}
            command.update(skill=str(rng.choice(['engage_target','pursue_target'])),target_id=identity,
                           preferred_range_m=float(rng.uniform(2.0,3.5)),damage_goal=60.,leash_radius_m=12.)
            command.pop('point')
            behavior[identity]={'mode':'weave','speed':.12,'origin':[blue[identity]['x'],blue[identity]['y']]}
    else:
        red=np.array([4.5,5.15])+rng.uniform(-.15,.15,2)
        center=[4.1,5.15]
        blue['blue-1']={'x':float(rng.uniform(7.5,9.0)),'y':5.15,'heading':math.pi,'ammo':0}
        blue['blue-3']={'x':6.7,'y':5.75,'heading':math.pi,'ammo':0}
        command.update(skill='defend_region',center=center,priority_ids=['blue-1'],region_radius_m=2.5,
                       leash_radius_m=4.,preferred_range_m=2.5,damage_goal=60.)
        command.pop('point')
        behavior['blue-1']={'mode':'invade','center':center,'speed':.16}
    target=np.array(command.get('point',command.get('center',[7.5,5.1])))
    if command.get('target_id')=='blue_outpost':target=np.array([17.024,3.6])
    elif command.get('target_id') in blue:target=np.array([blue[command['target_id']]['x'],blue[command['target_id']]['y']])
    heading=math.atan2(target[1]-red[1],target[0]-red[0])
    scenario={'robots':{'red-3':{'x':float(red[0]),'y':float(red[1]),'heading':heading,'aim':0. if family=='defense' else heading,'ammo':200},**blue}}
    return Episode(family,int(seed),scenario,command,behavior,900 if command['timeout_s']>30 else 300)


def scripted_actions(episode,state,team='blue'):
    observation=state['observations'][team];actions={}
    for r in observation['robots']:
        spec=episode.behavior.get(r['id'])
        if not spec:continue
        if spec['mode']=='weave':
            origin=spec['origin'];phase=state['info']['time']*.8
            target=np.array(origin)+np.array([math.sin(phase)*1.2,math.cos(phase)*.35])
        else:target=np.array(spec['center'])
        delta=target-xy(r);length=np.linalg.norm(delta)
        move=delta/max(.01,length)*spec['speed'] if length>.25 else np.zeros(2)
        actions[str(r['slot'])]={'move':move.tolist(),'heading':math.atan2(move[1],move[0]) if np.linalg.norm(move) else r['heading'],'fire':False}
    return actions


class TacticalBatch:
    def __init__(self,num_envs=4,*,family='navigation',seed=7,arena_root=None,record=False,opponent=None,lineups=None,season=None):
        configure(arena_root)
        from rmuc_arena.vector_env import VectorArena
        self.num_envs,self.family,self.next_seed=num_envs,family,seed
        self.executors=[TacticalExecutor() for _ in range(num_envs)]
        self.episodes=[None]*num_envs;self.states=[None]*num_envs;self.previous=[None]*num_envs
        self.steps=np.zeros(num_envs,dtype=int);self.finished=np.ones(num_envs,dtype=bool)
        self.season=season or ('2026' if family=='flight' else '2027')
        if lineups is None and family in ('support','flight'):
            from .extended_curricula import SUPPORT_LINEUPS,FLIGHT_LINEUPS
            lineups=FLIGHT_LINEUPS if family=='flight' else SUPPORT_LINEUPS
        lineups=LINEUPS if lineups is None else lineups
        if not any(d.get('slot')==3 and d.get('model')==3 for d in lineups.get('red',[])):
            raise ValueError('Tactical policies require red slot 3 using W3 model 3')
        self.engine=VectorArena(num_envs,season=self.season,duration=420,lineups=lineups,compact=True,record=record,opponent=opponent)
        self.opponent=opponent;self.runtime_control=False

    def reset(self,env_ids=None,seeds=None,episodes=None):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if seeds is None:
            seeds=list(range(self.next_seed,self.next_seed+len(ids)));self.next_seed+=len(ids)
        if len(seeds)!=len(ids):raise ValueError('One seed per selected actor required')
        episodes=[sample_episode(self.family,int(s)) for s in seeds] if episodes is None else list(episodes)
        if len(episodes)!=len(ids):raise ValueError('One episode per selected actor required')
        states=self.engine.reset(seeds,env_ids=ids,scenarios=[e.scenario for e in episodes])
        arenas=[s['observations']['red']['arena'] for s in states]
        from .extended_curricula import warmup
        states,prepared=warmup(self.engine,ids,states,episodes) if any(getattr(e,'warmup',None) for e in episodes) else (states,[None]*len(ids))
        observations=[]
        for k,(i,e,state) in enumerate(zip(ids,episodes,states)):
            self.states[i]=state;self.episodes[i]=e;self.executors[i]=prepared[k] or TacticalExecutor();self.steps[i]=0;self.finished[i]=False
            self.executors[i].arena=arenas[k]
            self.executors[i].submit(e.command,state['observations']['red'])
            self.previous[i]=self._metrics(i);observations.append(encode_observation(self.executors[i],state['observations']['red'],self.family))
        return np.stack(observations)

    def command(self,env_id,value):
        if self.states[env_id] is None:raise ValueError('Reset the actor first')
        response=self.executors[env_id].submit(value,self.states[env_id]['observations']['red'])
        if not response['duplicate']:
            self.finished[env_id]=False;self.steps[env_id]=0;self.previous[env_id]=self._metrics(env_id)
        return response

    def _metrics(self,i):
        obs=self.states[i]['observations']['red'];r=own(obs);ex=self.executors[i];plan=ex.plan
        return {'position':xy(r),'hp':r['hp'],'heat':r.get('heat',0),'distance':plan['distance'],
                'observed_damage':ex.damage,'hint':plan['hint'].copy(),
                'fire_ready':bool(plan['target'] and plan['visible']),
                'ammo':r['ammo'],'waypoints':ex.extra.get('waypoint_index',0),'coverage':ex.extra.get('coverage',0),
                'range_error':abs(float(np.linalg.norm(xy(plan['target'])-xy(r)))-ex.command.preferred_range_m) if plan['target'] and ex.command else 0.}

    def step(self,actions,env_ids=None,training=True):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if len(actions)!=len(ids):raise ValueError('One learned action per selected actor required')
        if training and any(self.finished[i] for i in ids):raise RuntimeError('Reset terminated training episodes before step')
        commands=[]
        for i,action in zip(ids,actions):
            family=self.executors[i].execution_family if self.runtime_control else self.family
            red=decode_action(self.executors[i],self.states[i]['observations']['red'],action,family)
            red.update({slot:a for slot,a in scripted_actions(self.episodes[i],self.states[i],team='red').items() if slot!='3'})
            commands.append({'red':red,
                             'blue':{} if self.opponent else scripted_actions(self.episodes[i],self.states[i])})
        states=self.engine.step(commands,env_ids=ids)
        observations=[];rewards=[];terms=[];truncs=[];infos=[]
        for i,state in zip(ids,states):
            self.states[i]=state;ex=self.executors[i];ex.update(state['observations']['red']);self.steps[i]+=1
            m=self._metrics(i);old=self.previous[i];r=own(state['observations']['red'])
            movement=float(np.dot(m['position']-old['position'],old['hint']))
            damage=m['observed_damage']-old['observed_damage']
            locomotion=2.*movement if ex.command and ex.command.skill=='navigate' else .8*(old['range_error']-m['range_error'])
            reward=locomotion+damage*.2-.012-.08*max(0.,old['hp']-m['hp'])
            if self.family=='support':
                reward+=.12*max(0,m['hp']-old['hp'])+.06*max(0,m['ammo']-old['ammo'])+3.*max(0,m['waypoints']-old['waypoints'])
                if ex.command.skill=='escort':reward+=.08*float(ex.extra.get('coverage',0)>.7)
            if self.family=='flight':
                reward+=.15*float(ex.plan['phase']=='flight_air')+.1*max(0,r['upright']-.8)
            reward-=.015*max(0.,m['heat']-r.get('max_heat',40)*.85)
            success=ex.status=='succeeded';term=ex.status in ('succeeded','failed','cancelled') or state['terminated']
            trunc=bool(not term and (self.steps[i]>=self.episodes[i].max_steps or state['truncated']))
            if success:reward+=8.
            elif term:reward-=4.
            if trunc and ex.status=='running':ex._end('failed','training_time_limit')
            self.finished[i]=bool(term or trunc);self.previous[i]=m
            info={**ex.feedback(),'task':self.family,'is_success':success,'seed':self.episodes[i].seed,
                  'target_damage':ex.damage,'target_kind':ex.plan['target'].get('role',ex.plan['target'].get('kind','unknown')) if ex.plan['target'] else None,
                  'steps':int(self.steps[i]),'engine_hash':state['info']['engine_hash'],
                  'rules_hash':state['info']['rules_hash']}
            observations.append(encode_observation(ex,state['observations']['red'],self.family));rewards.append(reward);terms.append(term);truncs.append(trunc);infos.append(info)
        return np.stack(observations),np.array(rewards,dtype=np.float32),np.array(terms,dtype=bool),np.array(truncs,dtype=bool),infos

    def close(self):self.engine.close()


class TacticalVecEnv(VecEnv):
    def __init__(self,num_envs=4,**options):
        self.batch=TacticalBatch(num_envs,**options);self.pending=None
        self.returns=np.zeros(num_envs);self.lengths=np.zeros(num_envs,dtype=int)
        space=OBS_SPACE
        if self.batch.family in ('support','flight'):
            from .extended_encoding import OBS_SPACE as space
        super().__init__(num_envs,space,ACTION_SPACE)
    def reset(self):
        seeds=self._seeds if any(s is not None for s in self._seeds) else None
        result=self.batch.reset(seeds=seeds);self._reset_seeds();self._reset_options()
        self.returns[:]=0;self.lengths[:]=0
        self.reset_infos=[{'seed':e.seed,'family':e.family} for e in self.batch.episodes]
        return result
    def step_async(self,actions):self.pending=np.array(actions).copy()
    def step_wait(self):
        if self.pending is None:raise RuntimeError('step_async required')
        obs,rewards,terms,truncs,infos=self.batch.step(self.pending);self.pending=None
        dones=terms|truncs;self.returns+=rewards;self.lengths+=1
        ids=np.flatnonzero(dones).tolist()
        for i in ids:
            infos[i]['terminal_observation']=obs[i].copy();infos[i]['TimeLimit.truncated']=bool(truncs[i])
            infos[i]['episode']={'r':float(self.returns[i]),'l':int(self.lengths[i])}
        if ids:
            reset=self.batch.reset(ids)
            for i,new in zip(ids,reset):
                obs[i]=new;self.returns[i]=0;self.lengths[i]=0
                self.reset_infos[i]={'seed':self.batch.episodes[i].seed,'family':self.batch.episodes[i].family}
        return obs,rewards,dones,infos
    def close(self):self.batch.close()
    def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name,None) for _ in self._get_indices(indices)]
    def set_attr(self,attr_name,value,indices=None):raise AttributeError('Configuration is immutable')
    def env_method(self,method_name,*args,indices=None,**kwargs):raise AttributeError(method_name)
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]


def encode_observation(ex,obs,family):
    if family in ('support','flight'):
        from .extended_encoding import encode as extended_encode
        return extended_encode(ex,obs)
    return encode(ex,obs,policy_family=family if family=='navigation' else None)


def decode_action(ex,obs,action,family):
    if family in ('support','flight'):
        from .extended_encoding import decode as extended_decode
        return extended_decode(ex,obs,action)
    return decode(ex,obs,action,policy_family=family)
