"""Universal ground policy contract: one schema for every 2027 ground chassis.

The network sees body-frame route/target geometry plus a public robot descriptor
(role, drive, caliber, size, mobility capabilities). The same seven discrete
action heads are interpreted by chassis:

* move mode / speed: stop, along the route hint, back, strafe left/right.
* aim / pitch residuals around SDK ballistic assistance; fire.
* posture: balancing chassis -> low/nominal/high ride height;
  holonomic chassis -> normal / spinning-top (yaw_rate) / supercap boost.
* evade: balancing chassis -> jump (navigation only);
  holonomic chassis -> alternating sidestep across the line of fire.

Only the controlled robot is commanded. Physics, referee and rules stay in Arena.
"""
import math
import numpy as np
from gymnasium import spaces
from .executor import own,xy,visible_enemies
from .robots import DESCRIPTOR,can_fire

SCHEMA='arena-ground-v1'
SKILLS=('navigate','attack_structure','engage_target','pursue_target','defend_region','resupply','escort','patrol','retreat','install_launcher')
PHASES=('navigate','approach','engage','search','hold','supply','follow','patrol','retreat','install')
PHASE_MAP={'supply_hold':'supply','escort_follow':'follow','escort_hold':'follow','patrol_move':'patrol','patrol_dwell':'patrol',
           'dock_wait':'install','confirm':'install'}
FAMILIES=('navigation','combat','defense','support')
ACTION_SPACE=spaces.MultiDiscrete([5,4,3,3,2,3,2])
IDLE_ACTION=np.array([0,0,1,1,0,1,0])
SPEEDS=(.2,.45,.75,1.)
PROBES=(('f',0.),('fl',.25),('l',.5),('bl',.75),('b',1.),('br',-.75),('r',-.5),('fr',-.25))
FEATURES=(['skill_'+s for s in SKILLS]+['phase_'+p for p in PHASES]+['robot_'+d for d in DESCRIPTOR]+
    ['x','y','z','vx_body','vy_body','vz','yaw_rate','hp','ammo','has_ammo','heat','fire_ready','upright','grounded','leg','weak','launcher_ready','supercap',
     'hint_x','hint_y','hint_fwd','hint_left','hint_norm','held_sin','held_cos','goal_fwd','goal_left','distance','arrival_radius','terrain_crossing','stalled',
     'preferred_range','range_error','target_visible','target_age','target_fwd','target_left','target_vfwd','target_vleft','target_hp',
     'target_heavy','target_infantry','target_sentry','target_structure','aim_error','pitch_error',
     'region_fwd','region_left','region_radius','leash','card_task','card_contact',
     'threats','nearest_fwd','nearest_left','nearest_distance','recent_damage',
     'supply_contact','ammo_goal','ally_fwd','ally_left','coverage','patrol_progress','install_progress']+
    ['terrain_'+name for name,_ in PROBES])
OBS_SPACE=spaces.Box(-10.,10.,shape=(len(FEATURES),),dtype=np.float32)


def wrap(angle):return (angle+math.pi)%math.tau-math.pi


def body(vector,heading):
    c,s=math.cos(heading),math.sin(heading)
    return np.array([c*vector[0]+s*vector[1],-s*vector[0]+c*vector[1]])


def target_class(target):
    if not target:return [0.,0.,0.,0.]
    if target.get('kind') in ('base','outpost'):return [0.,0.,0.,1.]
    role=target.get('role')
    return [float(role in ('heavy','hero')),float(role=='infantry'),float(role=='sentry'),0.]


def recent_damage(ex,robot,tick):
    """HP lost over roughly the last second; idempotent per observation tick."""
    history=ex.__dict__.setdefault('hp_history',{})
    history[tick]=robot['hp']
    for old in [t for t in history if t<tick-20]:del history[old]
    return max(0.,history[min(history)]-robot['hp'])/max(1.,robot['max_hp'])


def nearest_threat(ex,observation,robot):
    enemies=[e for e in visible_enemies(observation,ex.slot) if e['alive'] and e.get('targetable',True) and e.get('role')!='aerial']
    if not enemies:return None,0
    position=xy(robot);close=[e for e in enemies if np.linalg.norm(xy(e)-position)<8.]
    return min(enemies,key=lambda e:float(np.linalg.norm(xy(e)-position))),len(close)


def encode(ex,observation,family=None):
    from rmuc_arena import sdk
    robot=own(observation,ex.slot);c=ex.command;plan=ex.plan;profile=ex.profile
    if profile is None:raise ValueError('Universal ground encoding requires a RobotProfile')
    if c is None or plan is None:return np.zeros(OBS_SPACE.shape,dtype=np.float32)
    navigation=family=='navigation' or c.skill in ('navigate','retreat','install_launcher')
    heading=robot['heading'];position=xy(robot);extra=ex.extra or {}
    skill='navigate' if family=='navigation' and c.skill not in ('retreat','install_launcher') else c.skill
    phase=PHASE_MAP.get(plan['phase'],plan['phase'])
    if family=='navigation' and phase not in ('retreat','install'):phase='navigate'
    values=[float(skill==s) for s in SKILLS]+[float(phase==p) for p in PHASES]+profile.descriptor().tolist()
    velocity=body([robot['vx'],robot['vy']],heading);legs=robot.get('legs')
    leg=(float(np.mean(legs))-(profile.leg_min_m or .12))/max(.05,(profile.leg_max_m or .34)-(profile.leg_min_m or .12)) if legs else 0.
    ready=can_fire(robot,profile)
    values+=[0. if navigation else robot['x']/28,0. if navigation else robot['y']/15,robot['z'],velocity[0]/4,velocity[1]/4,robot['vz']/4,
             robot.get('angular_velocity',[0,0,0])[2]/6,robot['hp']/max(1,robot['max_hp']),min(2.,robot['ammo']/profile.ammo_scale),float(robot['ammo']>0),
             robot.get('heat',0)/max(1.,robot.get('max_heat',0) or 100.),float(ready),robot['upright'],float(robot['grounded']),leg,float(robot.get('weak',False)),
             float(profile.role!='heavy' or robot.get('launcher') in ('fitted','installed')),robot.get('supercap_energy_j',0)/1800]
    hint=np.asarray(plan['hint'],dtype=float);norm=float(np.linalg.norm(hint));local=body(hint,heading)
    held=plan.get('held_heading');goal=np.asarray(plan['goal'],dtype=float)-position
    distance=float(plan['distance'])
    if navigation:goal=hint/max(.001,norm)*min(2.,distance)
    goal=body(goal,heading)
    stalled=min(3.,max(0.,ex.time-ex.last_motion)/10.)
    values+=[float(hint[0]),float(hint[1]),float(local[0]),float(local[1]),min(2.,norm),
             math.sin(wrap(held-heading)) if held is not None else 0.,math.cos(wrap(held-heading)) if held is not None else 1.,
             goal[0]/5,goal[1]/5,min(2.,distance)/10 if navigation else min(30.,distance)/10,plan.get('arrival_radius',c.radius_m)/3,
             float(bool(ex.route_state.get('terrain_crossing'))),stalled]
    target=None if navigation else plan['target'];visible=bool(target) and bool(plan['visible'])
    if target:
        delta=body(xy(target)-position,heading);tv=body([target.get('vx',0.),target.get('vy',0.)],heading)
        separation=float(np.linalg.norm(xy(target)-position))
        desired=sdk.aim_at(robot,target) if visible else math.atan2(target['y']-robot['y'],target['x']-robot['x'])
        aim_error=wrap(desired-robot['aim']);pitch_error=(sdk.aim_pitch(robot,target)-robot['pitch']) if visible else 0.
        hp=target.get('hp',0)/max(1.,target.get('max_hp',400. if target.get('kind') not in ('base','outpost') else 1000.))
    else:
        delta=tv=np.zeros(2);separation=0.;aim_error=pitch_error=0.;hp=0.
    values+=[(2.5 if navigation else c.preferred_range_m)/8,(separation-c.preferred_range_m)/4 if target else 0.,float(visible),
             0. if navigation else min(2.,plan['age']/max(.1,c.target_lost_timeout_s)),delta[0]/10,delta[1]/10,tv[0]/4,tv[1]/4,hp]
    values+=target_class(target)+[math.sin(aim_error),pitch_error]
    region=np.asarray(c.center) if c.center is not None else getattr(ex,'anchor',position)
    region=body(np.asarray(region)-position,heading) if not navigation else np.zeros(2)
    card=extra.get('control_card') is not None and not navigation
    values+=[region[0]/10,region[1]/10,(3. if navigation else c.region_radius_m)/10,(8. if navigation else c.leash_radius_m)/35,float(card),float(bool(extra.get('control_contact')) and card)]
    threat,count=nearest_threat(ex,observation,robot)
    if threat:
        tdelta=body(xy(threat)-position,heading);tdistance=float(np.linalg.norm(xy(threat)-position))
    else:tdelta=np.zeros(2);tdistance=10.
    values+=[count/4,tdelta[0]/10,tdelta[1]/10,min(10.,tdistance)/10,recent_damage(ex,robot,observation['tick'])*10]
    ally=next((a for a in observation['robots'] if c.target_id and a['id']==c.target_id),None) if c.skill=='escort' else None
    adelta=body(xy(ally)-position,heading) if ally else np.zeros(2)
    values+=[float(bool(extra.get('supply_contact'))),c.ammo_target/max(1.,profile.ammo_scale) if c.skill=='resupply' else 0.,adelta[0]/5,adelta[1]/5,
             float(extra.get('coverage',0.)),extra.get('waypoint_index',0)/max(1,extra.get('waypoints_total',1)),float(extra.get('launcher_progress',0.))]
    ground=sdk.terrain_height(ex.arena,*position)
    for _,turn in PROBES:
        angle=heading+turn*math.pi
        values.append(sdk.terrain_height(ex.arena,position[0]+.5*math.cos(angle),position[1]+.5*math.sin(angle))-ground)
    result=np.clip(np.nan_to_num(np.asarray(values,dtype=np.float32)),-10,10)
    if result.shape!=OBS_SPACE.shape:raise RuntimeError('Ground schema dimension mismatch')
    return result


def _turret(ex,observation,robot,out,target,a=None):
    from rmuc_arena import sdk
    out['aim']=sdk.aim_at(robot,target)+((int(a[2])-1)*.025 if a is not None else 0.)
    out['pitch']=sdk.aim_pitch(robot,target)+((int(a[3])-1)*.02 if a is not None else 0.)


def _buy(ex,observation,robot,out):
    from rmuc_arena import sdk
    c=ex.command;caliber=ex.profile.caliber
    rule=sdk.rules(observation)['economy']['ammo'][caliber];batch=int(rule['local_batch']);cost=rule['local_cost']/batch
    team=observation['teams'][observation['team']]
    remaining=max(0,c.ammo_target-robot['ammo']);limit=min(team['resources'],c.budget_limit-ex.extra.get('spent',0))
    cap=rule['team_purchase_cap']-team.get('purchased_'+caliber[:2],0)
    amount=min(math.ceil(remaining/batch)*batch,int(max(0,limit)/cost)//batch*batch,int(cap)//batch*batch)
    if amount>0:out['buy_ammo']={'amount':amount,'remote':False}


def decode(ex,observation,action,family):
    a=np.asarray(action)
    if a.shape!=(7,) or not np.issubdtype(a.dtype,np.integer) or not ACTION_SPACE.contains(a):raise ValueError('Invalid ground MultiDiscrete action')
    robot=own(observation,ex.slot);profile=ex.profile;plan=ex.plan;c=ex.command
    out={'move':[0.,0.],'heading':robot['heading'],'aim':robot['aim'],'pitch':robot['pitch'],'fire':False,'brake':True}
    if profile.legged:out.update(leg_height=profile.leg_height(1),jump=0.)
    if ex.status!='running' or plan is None or c is None:return {ex.slot:out}
    if plan['phase']=='recover':
        # Serial self-righting firmware supplies hip torques; the host only settles.
        out.update(power=100.,leg_height=profile.leg_height(int(a[5])) if profile.legged else None)
        if out.get('leg_height') is None:out.pop('leg_height')
        return {ex.slot:out}
    navigation=family=='navigation'
    hint=np.asarray(plan['hint'],dtype=float);length=float(np.linalg.norm(hint))
    target=plan['target'];visible=bool(target) and bool(plan['visible']) and target.get('alive',target.get('hp',0)>0)
    card=bool(ex.extra.get('control_card'))
    if length<.001 and visible and not navigation and not card and c.skill in ('engage_target','pursue_target','defend_region'):
        # Reposition/orbit inside the preferred range of a robot target.
        hint=xy(target)-xy(robot);length=float(np.linalg.norm(hint))
        hint=hint*(abs(float(np.linalg.norm(hint))-c.preferred_range_m)/max(.01,length));length=float(np.linalg.norm(hint))
    move=np.zeros(2)
    if length>.001:
        unit=hint/length;left=np.array([-unit[1],unit[0]])
        move=(np.zeros(2),unit,-unit,left,-left)[int(a[0])]*SPEEDS[int(a[1])]
        if length<.05:move*=length/.05
    threat=target if visible else None
    if profile.holonomic:
        if int(a[6]) and threat is not None and not navigation:
            line=xy(threat)-xy(robot);line=line/max(.01,np.linalg.norm(line));side=np.array([-line[1],line[0]])
            sign=1. if int(observation['time']/.4)%2==0 else -1.
            move=move+side*sign*.6
            if np.linalg.norm(move)>1.:move=move/np.linalg.norm(move)
        out['move']=move.tolist()
        if visible:out['heading']=math.atan2(target['y']-robot['y'],target['x']-robot['x'])
        elif plan.get('held_heading') is not None:out['heading']=plan['held_heading']
        elif np.linalg.norm(move)>.01:out['heading']=math.atan2(move[1],move[0])
        if int(a[5])==1:
            out.pop('heading');out['yaw_rate']=6. if int(ex.slot)%2 else -6.
        elif int(a[5])==2:out['power']=200.
    else:
        out['leg_height']=profile.leg_height(int(a[5]))
        if navigation and ex.route_state.get('terrain_crossing'):out['leg_height']=profile.leg_height(1)
        if np.linalg.norm(move)>.001:
            course=plan.get('held_heading')
            if course is None:
                course=math.atan2(move[1],move[0]);reverse=course+math.pi
                if abs(wrap(reverse-robot['heading']))<abs(wrap(course-robot['heading'])):course=reverse
            out['heading']=course
        elif length>.001:
            bearing=math.atan2(hint[1],hint[0])
            out['heading']=plan.get('held_heading') if plan.get('held_heading') is not None else min((bearing,bearing+math.pi),key=lambda h:abs(wrap(h-robot['heading'])))
        error=abs(wrap(out['heading']-robot['heading']))
        out['move']=move.tolist() if error<=.35 or ex.route_state.get('terrain_crossing') else [0.,0.]
        if ex.route_state.get('terrain_crossing'):out['power']=200. if int(a[4]) else 100.
        out['jump']=float(a[6]) if navigation and profile.jump else 0.
    ready=can_fire(robot,profile) and robot['ammo']>0
    if visible and not navigation:
        _turret(ex,observation,robot,out,target,a)
        out['fire']=bool(a[4]) and ready and c.skill not in ('resupply','install_launcher')
    elif not navigation and c.skill not in ('navigate','retreat','resupply','install_launcher'):
        out['aim']=(robot['aim']+.5)%math.tau
    else:
        # Opportunistic turret while travelling: explicit SDK auto-aim on the nearest visible robot.
        threat,_=nearest_threat(ex,observation,robot)
        if threat is not None and np.linalg.norm(xy(threat)-xy(robot))<6. and c.skill not in ('resupply','install_launcher'):
            _turret(ex,observation,robot,out,threat);out['fire']=ready
        else:out['aim']=out.get('heading',robot['heading']);out['pitch']=.04
    if profile.role=='sentry':
        posture='attack' if out['fire'] else 'defense' if plan['phase'] in ('hold','engage') else 'move'
        if robot.get('posture')!=posture:out['posture']=posture
    if c.skill=='resupply' and not navigation:
        out['fire']=False
        if int(a[4]) and ex.extra.get('supply_contact'):_buy(ex,observation,robot,out)
        if ex.extra.get('supply_contact') and profile.legged:out['leg_height']=min(profile.leg_height(1),out['leg_height'])
    if c.skill=='install_launcher':
        from .team_skills import install_action
        install_action(ex,observation,out)
    return {ex.slot:out}


def scripted_action(ex,observation,kind='heuristic'):
    """Chassis-aware expert used for demonstrations and as the measured baseline."""
    if kind=='idle' or ex.status!='running' or ex.plan is None:return IDLE_ACTION.copy()
    robot=own(observation,ex.slot);plan=ex.plan;profile=ex.profile;c=ex.command
    distance=plan['distance'];threshold=plan.get('arrival_radius',c.radius_m)
    target=plan['target'];visible=bool(target) and plan['visible']
    family=ex.execution_family
    mode=int(distance>threshold*.8 and plan['phase']!='recover')
    speed=3 if distance>2.5 else 2 if distance>1. else 1
    # Brake early enough for the chassis momentum (heavier robots coast further).
    v=math.hypot(robot['vx'],robot['vy']);stopping=v*v/(5. if profile.role!='heavy' else 3.)
    if distance<stopping+.3 and v>.6:mode=0
    elif distance<stopping+1.:speed=min(speed,1)
    if visible and family!='navigation' and c.skill in ('engage_target','pursue_target','defend_region') and not ex.extra.get('control_card'):
        separation=float(np.linalg.norm(xy(target)-xy(robot)))
        error=separation-c.preferred_range_m
        mode=1 if error>.4 else 2 if error<-.6 else 0;speed=2 if abs(error)>1. else 1
        if mode and np.linalg.norm(plan['hint'])<.001:mode=1 if error>0 else 2
    posture=1;evade=0
    if profile.holonomic:
        exposed=visible and family!='navigation' and target.get('kind') not in ('base','outpost')
        # The heavy's 50 W chassis needs the supercap for slopes, bevels and card approaches.
        posture=1 if exposed and c.skill!='resupply' else 2 if robot['role']=='heavy' and mode and distance>.6 else 0
        evade=int(exposed and mode==0)
    else:
        if ex.route_state.get('terrain_crossing'):speed=3
        if any(t['kind']=='bumps' and t['x']-.5<robot['x']<t['x']+t['w']+.5 and t['y']-.5<robot['y']<t['y']+t['h']+.5 for t in ex.arena['terrain']):posture=2;speed=3
    fire=int(visible and family!='navigation')
    if fire:
        from rmuc_arena import sdk
        # Spend shells only once the turret is on the ballistic solution and in useful range.
        error=abs(wrap(sdk.aim_at(robot,target)-robot['aim']));separation=float(np.linalg.norm(xy(target)-xy(robot)))
        fire=int(error<(.03 if profile.caliber=='42mm' else .08) and separation<(8. if profile.caliber=='42mm' else 7.))
    if c.skill=='resupply' and family!='navigation':fire=1
    # On a balancing chassis the fire head doubles as the supercap boost over road humps.
    if profile.legged and ex.route_state.get('terrain_crossing'):fire=1
    return np.array([mode,speed,1,1,fire,posture,evade])
