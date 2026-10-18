"""Persistent tactical state driven exclusively by the controlled team's observations."""
from collections import OrderedDict
import copy
import math
import numpy as np
from .commands import Command

TERMINAL={'succeeded','failed','cancelled'}


def own(observation,slot='3'):
    return next(r for r in observation['robots'] if str(r['slot'])==str(slot))


def xy(value):return np.array([value['x'],value['y']],dtype=float)


def visible_enemies(observation,slot='3'):
    if 'vision' not in observation:return observation['enemies']
    identities=set(observation['vision'].get(str(slot),{}).get('enemies',[]))
    return [e for e in observation['enemies'] if e['id'] in identities]


class TacticalExecutor:
    """Commands do not reset physics, policies, ammo or observation memory.

    submit() validates atomically. update() can be called at most once per new
    observation tick; repeated calls are idempotent. Plan is policy input, not an
    actuator command. Only decode() translates learned actions to SDK commands.
    """
    def __init__(self,slot='3'):
        self.slot=str(slot);self.command=None;self.status='idle';self.reason=None
        self.memory={};self.identities=OrderedDict();self.revision=0;self.last_basis_tick=-1
        self.tick=-1;self.time=0.;self.started=0.;self.completed_at=None;self.plan=None;self.target_id=None
        self.damage=0.;self.last_hp={};self.events=[];self.arena=None
        self.last_position=None;self.last_motion=0.;self.initial_distance=None;self.navigation_arena=None

    def _end(self,status,reason):
        if self.status=='running':
            self.status,self.reason=status,reason;self.completed_at=self.time

    def cancel(self,reason='cancelled_by_planner'):
        if self.command and self.status=='running':
            self._end('cancelled',reason);self.events.append(self.feedback())
            if reason!='replaced_by_new_command':self.last_basis_tick=max(self.last_basis_tick,self.tick)
        return self.feedback()

    def submit(self,value,observation):
        command=Command.parse(value.json() if isinstance(value,Command) else value)
        if observation['tick']<self.tick:raise ValueError('Cannot submit against an older observation')
        previous=self.identities.get(command.id)
        if previous is not None:
            if previous!=command.json():raise ValueError('Command id was reused with different content')
            return {'accepted':True,'duplicate':True,'revision':self.revision,'feedback':self.feedback()}
        arena=observation.get('arena',self.arena)
        if arena is None:raise ValueError('Initial observation must contain arena geometry')
        robot=own(observation,self.slot);team=observation['team'];now=observation['time']
        if command.basis_tick is not None:
            age=(observation['tick']-command.basis_tick)*observation['dt']
            if age<0 or age>command.max_staleness_s or command.basis_tick<self.last_basis_tick:
                raise ValueError('Stale, out-of-order or future command basis_tick')
        for point in (command.point,command.center):
            if point and not (robot['radius']<=point[0]<=arena['width']-robot['radius'] and robot['radius']<=point[1]<=arena['height']-robot['radius']):
                raise ValueError('Command point is outside the robot field bounds')
        if command.target_id and command.target_id.startswith(team+('-' if '-' in command.target_id else '_')):
            raise ValueError('Cannot attack a friendly target')
        roster=observation['roster']
        if command.target_id and '-' in command.target_id:
            target_team,slot=command.target_id.split('-')
            legal=roster[target_team] if isinstance(roster,dict) else roster
            if int(slot) not in legal:raise ValueError('Target id is absent from the public roster')
        for target in command.priority_ids:
            target_team,slot=target.split('-')
            legal=roster.get(target_team,[]) if isinstance(roster,dict) else roster
            if target_team==team or int(slot) not in legal:raise ValueError('Invalid defense priority target')
        self.cancel('replaced_by_new_command')
        self.last_basis_tick=max(self.last_basis_tick,command.basis_tick if command.basis_tick is not None else observation['tick'])
        self.identities[command.id]=command.json()
        if len(self.identities)>1024:self.identities.popitem(last=False)
        self.revision+=1;self.command=command;self.status='running';self.reason=None
        self.started=now;self.completed_at=None;self.last_motion=now;self.last_position=xy(robot)
        self.anchor=np.array(command.center) if command.center else xy(robot)
        self.target_id=command.target_id;self.damage=0.;self.last_hp={};self.initial_distance=None;self.hold_point=None
        self.tick=-1;self.plan=None
        self.update(observation)
        return {'accepted':True,'duplicate':False,'revision':self.revision,'feedback':self.feedback()}

    def update(self,observation):
        from rmuc_arena import sdk
        if observation['tick']<self.tick:raise ValueError('Observation moved backwards; create a new executor on episode reset')
        if observation['tick']==self.tick:return self.plan
        self.tick=observation['tick'];self.time=observation['time']
        self.arena=observation.get('arena',self.arena)
        if self.navigation_arena is None and self.arena is not None:
            from .navigation import navigation_arena
            self.navigation_arena=navigation_arena(self.arena)
        if self.arena is None:raise ValueError('Arena geometry is required on first observation')
        robot=own(observation,self.slot);position=xy(robot);now=self.time
        visible={e['id']:e for e in visible_enemies(observation,self.slot)}
        for identity,enemy in visible.items():
            self.memory[identity]={'value':copy.deepcopy(enemy),'seen_at':now}
        for identity in list(self.memory):
            if now-self.memory[identity]['seen_at']>10:del self.memory[identity]
        if self.command and self.status!='running':return self.plan
        if not self.command:
            self.plan={'phase':'idle','goal':position,'hint':np.zeros(2),'target':None,'visible':False,'distance':0.,'age':0.}
            return self.plan
        c=self.command
        if not robot['alive'] or not robot['powered'] or robot['upright']<.3:self._end('failed','robot_unavailable')
        elif robot['hp']/max(1,robot['max_hp'])<c.low_hp_fraction:self._end('failed','low_hp')
        elif c.skill!='navigate' and robot['ammo']==0:self._end('failed','out_of_ammo')
        elapsed=now-self.started
        if elapsed>=c.timeout_s:
            defended=c.skill=='defend_region' and c.damage_goal is None
            contested=defended and any(e['alive'] and np.linalg.norm(xy(e)-self.anchor)<=c.region_radius_m for e in visible.values())
            self._end('succeeded' if defended and not contested else 'failed',
                      'region_contested' if contested else 'defense_interval_complete' if defended else 'timeout')
        target=None;is_visible=False;age=0.;phase='navigate';goal=np.array(c.point or c.center or position)
        if c.skill=='defend_region':
            from .navigation import safe_hold_point
            if self.hold_point is None:self.hold_point=safe_hold_point(robot,c.center,c.region_radius_m,self.navigation_arena)
            hold=self.hold_point
            if hold is None:self._end('failed','region_blocked')
            else:goal=np.array(hold)
            eligible=[e for e in visible.values() if e['alive'] and e.get('targetable',True)
                      and np.linalg.norm(xy(e)-self.anchor)<=c.region_radius_m+c.leash_radius_m]
            def rank(e):
                priority=c.priority_ids.index(e['id']) if e['id'] in c.priority_ids else len(c.priority_ids)
                return (priority,float(np.linalg.norm(xy(e)-self.anchor)),e['id'])
            target=min(eligible,key=rank,default=None);self.target_id=target['id'] if target else None
            is_visible=target is not None
        elif c.skill=='attack_structure':
            team,kind=c.target_id.split('_');target={**observation['outposts' if kind=='outpost' else 'bases'][team],'kind':kind}
            is_visible=True
            if target.get('owner',team)==observation['team']:self._end('failed','target_friendly')
            if kind=='base':
                protection=observation.get('positions',{}).get(team+'_outpost',{})
                if protection.get('control')=='full' and protection.get('controller')==team:self._end('failed','structure_protected')
        elif c.skill in ('engage_target','pursue_target'):
            record=self.memory.get(c.target_id)
            is_visible=c.target_id in visible
            if record:
                age=now-record['seen_at'];target=copy.deepcopy(record['value'])
                if not is_visible:
                    horizon=min(age,c.target_lost_timeout_s)
                    target['x']+=target.get('vx',0)*horizon;target['y']+=target.get('vy',0)*horizon
            else:age=elapsed
            if not is_visible and age>=c.target_lost_timeout_s:self._end('failed','target_lost')
        if target:
            identity=self.target_id
            if is_visible:
                hp=target.get('hp',0.)
                lost=max(0.,self.last_hp.get(identity,hp)-hp)
                self.damage+=lost;self.last_hp[identity]=hp
                if lost>0:self.last_motion=now
                destroyed=(not target['alive']) if 'alive' in target else (hp<=0)
                if destroyed:
                    if c.skill!='defend_region':self._end('succeeded','target_destroyed')
                if c.damage_goal is not None and self.damage>=c.damage_goal:self._end('succeeded','damage_goal_reached')
            target_xy=xy(target);separation=float(np.linalg.norm(target_xy-position))
            direction=(position-target_xy)/max(.01,separation)
            if c.skill=='pursue_target':
                intercept=np.array([target.get('vx',0),target.get('vy',0)])*min(1.,separation/4.)
                target_xy=target_xy+intercept
            goal=target_xy+direction*c.preferred_range_m
            if not is_visible:goal=target_xy;phase='search'
            elif abs(separation-c.preferred_range_m)<=.5:goal=position;phase='engage'
            else:phase='approach'
            if c.skill=='defend_region' and np.linalg.norm(goal-self.anchor)>c.region_radius_m:
                delta=goal-self.anchor;goal=self.anchor+delta/np.linalg.norm(delta)*c.region_radius_m
        elif c.skill!='navigate':phase='hold' if c.skill=='defend_region' else 'search'
        goal=np.clip(goal,[robot['radius'],robot['radius']],[self.arena['width']-robot['radius'],self.arena['height']-robot['radius']])
        distance=float(np.linalg.norm(goal-position))
        if c.skill=='navigate' and distance<=c.radius_m and math.hypot(robot['vx'],robot['vy'])<.8:self._end('succeeded','arrived')
        if c.skill in ('engage_target','pursue_target') and np.linalg.norm(position-self.anchor)>c.leash_radius_m:self._end('failed','leash_exceeded')
        if self.last_position is None or np.linalg.norm(position-self.last_position)>.25:
            self.last_position=position;self.last_motion=now
        if phase in ('navigate','approach','search') and distance>c.radius_m and now-self.last_motion>6:
            self._end('failed','blocked')
        planned=sdk.navigate(robot,goal,self.navigation_arena) if distance>c.radius_m else (0.,0.)
        hint=np.array(planned)
        held_heading=getattr(planned,'heading',None)
        self.plan={'phase':phase,'goal':goal,'hint':hint,'target':target,'visible':is_visible,'distance':distance,'age':age,'held_heading':held_heading}
        if self.initial_distance is None:self.initial_distance=distance
        if self.status in TERMINAL:
            self.events.append(self.feedback());self.events=self.events[-128:]
        return self.plan

    @property
    def execution_family(self):
        if self.command is None:return 'navigation'
        family=self.command.policy_family
        if family!='navigation' and self.plan and self.plan['distance']>1.2 and self.plan['phase'] in ('approach','search','hold'):
            return 'navigation'
        return family

    def feedback(self):
        plan=self.plan or {}
        distance=float(plan.get('distance',0.))
        return {'command_id':self.command.id if self.command else None,'revision':self.revision,
                'status':self.status,'reason':self.reason,'tick':self.tick,'time':self.time,
                'elapsed_s':max(0.,(self.completed_at if self.completed_at is not None else self.time)-self.started),'skill':self.command.skill if self.command else None,
                'phase':plan.get('phase','idle'),'execution_family':self.execution_family,'target_id':self.target_id,
                'target_visible':bool(plan.get('visible',False)),'last_seen_age_s':float(plan.get('age',0.)),
                'distance_m':distance,'observed_target_hp_loss':self.damage,
                'progress_m':(self.initial_distance-distance) if self.initial_distance is not None else 0.}
