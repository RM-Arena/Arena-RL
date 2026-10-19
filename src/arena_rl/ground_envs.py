"""Single-learner Gymnasium environment for universal ground policies.

Each environment owns one in-process ArenaEnv (one Rust simulator), so SB3's
SubprocVecEnv can spread the Python executor, route planning and scripted
opponents over every CPU core. The learner commands only its own slot; every
other robot runs a declared scripted behavior from the curriculum.
"""
from collections import OrderedDict
import json
import math
import numpy as np
import gymnasium as gym
from .bridge import configure
from .executor import TacticalExecutor,own,xy,visible_enemies
from .ground_curricula import sample,Stage,LANDMARKS
from .ground_encoding import OBS_SPACE,ACTION_SPACE,FAMILIES,encode,decode,scripted_action
from .robots import profile,can_fire

SHOT_COST={'17mm':.03,'42mm':.3}
ENGINE_CACHE=6


def dispose(engine):
    """Close a discarded simulator without blocking: Match.close waits for the Rust
    process to notice EOF, which can take tens of seconds on a loaded machine."""
    match=getattr(engine,'match',None)
    if match is not None and not match.closed and match.process.poll() is None:match.process.kill()
    engine.close()


def _aim_fire(robot,target,out):
    from rmuc_arena import sdk
    out.update(aim=sdk.aim_at(robot,target),pitch=sdk.aim_pitch(robot,target),fire=bool(robot['ammo']>0 and robot.get('heat',0)+10<=max(robot.get('max_heat',0),10)))


def _drive(robot,goal,arena,speed,out,stop=.25):
    from rmuc_arena import sdk
    from .navigation import clearance_hint
    distance=float(np.linalg.norm(np.asarray(goal)-xy(robot)))
    if distance<=stop:return out
    hint,heading=clearance_hint(robot,sdk.navigate(robot,goal,arena),arena)
    move=np.asarray(hint,dtype=float);length=float(np.linalg.norm(move))
    if length>.001:
        # Direction from the planner, magnitude from the behavior: weak pushes stall on bevels.
        out['move']=(move/length*speed*float(np.clip(distance/.5,.6,1.))).tolist()
        out['heading']=heading if heading is not None else math.atan2(move[1],move[0])
    return out


def on_bumps(arena,robot,margin=.05):
    return any(t['kind']=='bumps' and t['x']-margin<robot['x']<t['x']+t['w']+margin and t['y']-margin<robot['y']<t['y']+t['h']+margin for t in arena['terrain'])


def behave(spec,robot,observation,arena,now):
    """Declared scripted behaviors; each robot sees only its own team's observation."""
    out={'move':[0.,0.],'heading':robot['heading'],'aim':robot['aim'],'pitch':robot['pitch'],'fire':False,'brake':True}
    if robot.get('chassis_type')=='wheel_legged':out['leg_height']=.245
    if not spec or not robot['alive']:return out
    visible={e['id']:e for e in visible_enemies(observation,str(robot['slot']))}
    mode=spec['mode'];target=visible.get(spec.get('target_id'))
    if mode=='weave':
        phase=now*.8;origin=np.asarray(spec['origin'])
        _drive(robot,origin+np.array([math.sin(phase)*1.2,math.cos(phase)*.35]),arena,spec['speed'],out,.15)
    elif mode=='flee':
        _drive(robot,spec['refuge'],arena,spec['speed'],out,.4)
    elif mode in ('invade','hold','move'):
        _drive(robot,spec['center'],arena,spec['speed'],out,.9 if mode=='invade' else .2)
    elif mode=='structure':
        target={**observation['outposts' if spec['kind']=='outpost' else 'bases'][spec['team']],'kind':spec['kind']}
        _aim_fire(robot,target,out);return out
    if target is not None and (mode=='shoot' or spec.get('armed')):_aim_fire(robot,target,out)
    return out


class GroundEnv(gym.Env):
    """One learned robot per simulator. Episode stages continue the same match."""
    metadata={'render_modes':[]}
    observation_space=OBS_SPACE
    action_space=ACTION_SPACE

    def __init__(self,family='navigation',seed=0,models=None,arena_root=None,record=False,training_rules=True):
        super().__init__()
        if family not in FAMILIES:raise ValueError('Unknown ground policy family')
        configure(arena_root)
        self.family,self.next_seed,self.models,self.record=family,int(seed),None if models is None else tuple(int(m) for m in models),record
        self.engine=None;self.lineups=None;self.episode=None;self.stage=0;self.executor=None;self.state=None
        self.steps=0;self.previous=None;self.finished=True;self.behavior={};self.stats={}
        # Training-only shaping (ending flat chassis on bump tiles) is off for evaluation.
        self.training_rules=training_rules

    # ---- simulator lifecycle -------------------------------------------------------
    def _engine(self,lineups):
        """Simulators are cached per lineup: model and opponent mixes repeat across episodes."""
        if self.engine is not None and lineups==self.lineups:return self.engine
        from rmuc_arena.env import ArenaEnv
        cache=self.__dict__.setdefault('engines',OrderedDict());key=json.dumps(lineups,sort_keys=True)
        if self.engine is not None:cache[json.dumps(self.lineups,sort_keys=True)]=self.engine
        self.engine=cache.pop(key,None) or ArenaEnv(season='2027',duration=420,lineups=lineups,compact=True,record=self.record)
        while len(cache)>ENGINE_CACHE:dispose(cache.popitem(last=False)[1])
        self.lineups=lineups
        return self.engine

    def _observation(self,team='red'):return self.state['observations'][team]

    def _script_arena(self):
        # Declared scripted robots use the same bump-free public routes as flat learners.
        if getattr(self,'_scripts',None) is None:
            from .navigation import navigation_arena
            self._scripts=navigation_arena(self.executor.arena,3,avoid_bumps=True)
        return self._scripts

    def _robot(self):return own(self._observation(),self.episode.slot)

    def _step_engine(self,red,blue):
        self.state=self.engine.step({'red':red,'blue':blue})
        return self.state

    def _scripted(self,team,skip=None):
        observation=self._observation(team);arena=self._script_arena();actions={}
        for r in observation['robots']:
            if r['id']==skip:continue
            spec=self.behavior.get(r['id'])
            if spec is not None:actions[str(r['slot'])]=behave(spec,r,observation,arena,observation['time'])
        return actions

    def _warm_done(self,task,observation):
        kind=task['kind']
        if kind=='launcher':return own(observation,self.episode.slot).get('launcher')=='installed'
        if kind=='outpost_down':return observation['outposts']['blue']['hp']<=0
        if kind=='capture':return observation['positions']['blue_outpost']['controller']=='red'
        if kind=='armor_open':return bool(observation['bases']['blue'].get('armor_open'))
        if kind=='own_outpost_lost':return observation['positions']['red_outpost']['controller']!='red'
        if kind=='clock':return observation['time']>=task['until']
        raise ValueError('Unknown warmup task')

    def _warmup(self,steps_limit=4400):
        """Earn launcher fitting, captured positions and opened base armor by ordinary play."""
        tasks=list(self.episode.warmup)
        if not tasks:return True
        observation=self._observation();ex=self.executor;installer=None
        cards={z['id']:[z['x'],z['y']] for z in ex.arena['zones']}
        if any(t['kind']=='launcher' for t in tasks):
            installer=TacticalExecutor(self.episode.slot);installer.profile=ex.profile;installer.arena=ex.arena
            installer.submit({'id':'warmup-launcher','skill':'install_launcher','timeout_s':400.},observation)
        holds={}
        for t in tasks:
            if t['kind']=='capture':holds[t['robot']]=cards['blue_central']
            elif t['kind']=='armor_open':holds[t['holder']]=cards['blue_fortress']
            elif t['kind']=='own_outpost_lost':holds[t['robot']]=cards['red_central']
        for _ in range(steps_limit):
            observation=self._observation()
            if all(self._warm_done(t,observation) for t in tasks):return True
            if self.state['terminated'] or self.state['truncated']:return False
            # A chain that has not completed by these times will not leave a usable match.
            if observation['time']>260 and any(t['kind'] in ('armor_open','capture','outpost_down') and not self._warm_done(t,observation) for t in tasks):return False
            if observation['time']>330:return False
            red={};blue={}
            if installer is not None and not self._warm_done({'kind':'launcher'},observation):
                installer.update(observation)
                if installer.status!='running':return False
                red.update(decode(installer,observation,scripted_action(installer,observation),'navigation'))
            shooters=[s for t in tasks if t['kind']=='outpost_down' and not self._warm_done(t,observation) for s in t['shooters']]
            settled=not shooters and (installer is None or installer.status!='running' or installer.plan['phase']=='dock_wait')
            for team,actions in (('red',red),('blue',blue)):
                view=self._observation(team)
                for r in view['robots']:
                    if r['id'] in shooters:actions[str(r['slot'])]=behave({'mode':'structure','kind':'outpost','team':'blue'},r,view,self._script_arena(),view['time'])
                    elif r['id'] in holds:
                        actions[str(r['slot'])]=behave({'mode':'hold','center':holds[r['id']],'speed':.5},r,view,self._script_arena(),view['time'])
                        card=next((z['id'] for z in ex.arena['zones'] if math.dist((z['x'],z['y']),holds[r['id']])<.05),None)
                        on_card=card is not None and r['id'] in view['zones'][card]['occupants']
                        settled&=bool(r['alive']) and (on_card or math.dist((r['x'],r['y']),holds[r['id']])<.45 and math.hypot(r['vx'],r['vy'])<.3)
            # Pure waiting (clock, launcher unlock, card timers) runs with a coarse action repeat.
            self.engine.action_repeat=20 if settled else 2
            try:self._step_engine(red,blue)
            finally:self.engine.action_repeat=2
        return False

    def _healthy(self):
        r=self._robot()
        return r['alive'] and r['powered'] and r['upright']>.3 and not (self.state['terminated'] or self.state['truncated']) and self._observation()['time']<405

    def _metrics(self):
        observation=self._observation();r=own(observation,self.episode.slot);ex=self.executor;plan=ex.plan or {}
        target=plan.get('target')
        return {'position':xy(r),'hp':r['hp'],'heat':r.get('heat',0),'ammo':r['ammo'],'upright':r['upright'],
                'distance':plan.get('distance',0.),'damage':ex.damage,'hint':np.asarray(plan.get('hint',np.zeros(2)),dtype=float).copy(),
                'range_error':abs(float(np.linalg.norm(xy(target)-xy(r)))-ex.command.preferred_range_m) if target and ex.command else 0.,
                'waypoints':ex.extra.get('waypoint_index',0),'contact':bool(ex.extra.get('control_contact',False)),'coverage':ex.extra.get('coverage',0.)}

    def _begin(self,stage):
        observation=self._observation();ex=self.executor
        self.behavior=dict(stage.behavior);self.steps=0
        ex.submit(dict(stage.command,id=f"{stage.command['id']}-s{self.stage}"),observation)
        self.previous=self._metrics();self.finished=False

    def _run_unrecorded(self,stage):
        # Expert transit decides at 5 Hz: unrecorded steps only need to arrive, not to train.
        self.engine.action_repeat=4
        try:self._transit(stage)
        finally:self.engine.action_repeat=2

    def _transit(self,stage):
        for _ in range(stage.max_steps//2+1):
            observation=self._observation();ex=self.executor
            if ex.status!='running' or not self._healthy():break
            red=decode(ex,observation,scripted_action(ex,observation),ex.execution_family if ex.execution_family in FAMILIES else 'navigation')
            red.update(self._scripted('red',skip=self.episode.learner))
            self._step_engine(red,self._scripted('blue'));ex.update(self._observation())

    def _advance(self):
        """Start the next recorded stage, driving any unrecorded transit with the expert."""
        while self.stage<len(self.episode.stages):
            stage=self.episode.stages[self.stage]
            if not self._healthy():return False
            self._begin(stage)
            if stage.recorded:return True
            self._run_unrecorded(stage)
            self.stage+=1
        return False

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is None and self.episode is not None and self.stage+1<len(self.episode.stages):
            self.stage+=1
            if self._advance():return self._encode(),self._info()
        for attempt in range(8):
            episode_seed=int(seed) if seed is not None and attempt==0 else self.next_seed
            if seed is None or attempt:self.next_seed+=1
            self.episode=sample(self.family,episode_seed,self.models)
            members={f"{team}-{d['slot']}" for team,ds in self.episode.lineups.items() for d in ds}
            scenario={'robots':{k:v for k,v in self.episode.scenario['robots'].items() if k in members}}
            self.state=self._engine(self.episode.lineups).reset(seed=episode_seed,scenario=scenario)
            observation=self._observation();ex=TacticalExecutor(self.episode.slot)
            ex.profile=profile(own(observation,self.episode.slot),observation);ex.arena=observation['arena'];self.executor=ex
            self.behavior={};self.stage=0
            if self._warmup() and self._advance():return self._encode(),self._info()
        raise RuntimeError('Could not prepare a ground training episode')

    def _encode(self):return encode(self.executor,self._observation(),self.family)

    def _info(self):
        p=self.executor.profile
        return {'seed':self.episode.seed,'family':self.family,'variant':self.episode.variant,'model':p.model_id,'role':p.role,'drive':p.drive,'stage':self.stage}

    def expert_action(self):
        return scripted_action(self.executor,self._observation())

    def step(self,action):
        if self.finished:raise RuntimeError('Reset the finished ground environment before step')
        ex=self.executor;observation=self._observation();p=ex.profile
        red=decode(ex,observation,np.asarray(action,dtype=np.int64),self.family)
        if set(red)!={self.episode.slot}:raise ValueError('Learned controller must own only its slot')
        red.update(self._scripted('red',skip=self.episode.learner))
        self._step_engine(red,self._scripted('blue'));self.steps+=1
        observation=self._observation();ex.update(observation)
        m=self._metrics();old=self.previous;r=own(observation,self.episode.slot)
        movement=float(np.dot(m['position']-old['position'],old['hint']/max(1e-6,np.linalg.norm(old['hint'])))) if np.linalg.norm(old['hint'])>1e-6 else 0.
        per_shot=200. if p.caliber=='42mm' else 20.
        navigating=ex.command.skill in ('navigate','retreat') or self.family=='navigation'
        reward=(2.*movement if navigating else .8*(old['range_error']-m['range_error'])+.6*movement*float(old['range_error']>.5 or not ex.plan['visible']))
        reward+=4.*(m['damage']-old['damage'])/per_shot-.012-16.*max(0.,old['hp']-m['hp'])/max(1.,r['max_hp'])
        reward-=SHOT_COST[p.caliber]*max(0,old['ammo']-m['ammo'])*float(ex.command.skill!='resupply')
        if self.family=='defense':reward+=.08*float(m['contact'])
        if self.family=='support':
            reward+=.12*max(0.,m['hp']-old['hp'])/max(1.,r['max_hp'])*200+.06*max(0,m['ammo']-old['ammo'])*(300./p.ammo_scale)+3.*max(0,m['waypoints']-old['waypoints'])
            if ex.command.skill=='escort':reward+=.08*float(m['coverage']>.7)
        if r.get('max_heat',0)>0:reward-=.015*max(0.,m['heat']-r['max_heat']*.85)*(10./(100. if p.caliber=='42mm' else 10.))
        reward-=.4*max(0.,old['upright']-m['upright'])
        if self.training_rules and not p.legged and ex.status=='running' and on_bumps(ex.arena,r):
            # Flat wheels on the hump tiles are both pointless and the slowest contact physics.
            ex._end('failed','bump_tile')
        success=ex.status=='succeeded';term=ex.status in ('succeeded','failed','cancelled') or bool(self.state['terminated'])
        stage=self.episode.stages[self.stage]
        trunc=bool(not term and (self.steps>=stage.max_steps or self.state['truncated']))
        if success:reward+=8.
        elif term:reward-=4.
        if trunc and ex.status=='running':ex._end('failed','training_time_limit')
        self.finished=bool(term or trunc);self.previous=m
        info={**self._info(),**ex.feedback(),'is_success':success,'steps':self.steps,'target_damage':ex.damage}
        return self._encode(),float(reward),bool(term),bool(trunc),info

    def save_replay(self,path):return self.engine.save_replay(path)

    def close(self):
        for engine in list(self.__dict__.get('engines',{}).values())+[self.engine]:
            if engine is not None:dispose(engine)
        self.engine=None;self.__dict__.get('engines',{}).clear()
