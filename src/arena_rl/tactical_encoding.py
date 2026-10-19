"""Goal-conditioned local RL action/observation contract, independent of the LLM."""
import math
import numpy as np
from gymnasium import spaces
from .commands import BASE_SKILLS as SKILLS
from .executor import own,xy

SCHEMA='w3-tactics-v3'
ACTION_SPACE=spaces.MultiDiscrete([5,4,3,3,2,3,2])
FEATURES=(['skill_'+s for s in SKILLS]+[
    'self_x','self_y','self_z','vx','vy','vz','heading_sin','heading_cos','hp','ammo','heat','upright','grounded',
    'hint_x','hint_y','goal_dx','goal_dy','goal_distance','preferred_range','radius','region_dx','region_dy',
    'region_radius','leash','phase_navigate','phase_approach','phase_engage','phase_search','phase_hold',
    'target_visible','last_seen_age','target_dx','target_dy','target_vx','target_vy','target_hp',
    'target_heavy','target_infantry','target_sentry','target_structure','aim_sin','aim_cos','pitch',
    'terrain_e','terrain_w','terrain_n','terrain_s'])
OBS_SPACE=spaces.Box(-10.,10.,shape=(len(FEATURES),),dtype=np.float32)


def encode(executor,observation,policy_family=None):
    from rmuc_arena import sdk
    robot=own(observation,executor.slot);position=xy(robot);c=executor.command;plan=executor.plan
    if c is None or plan is None:return np.zeros(OBS_SPACE.shape,dtype=np.float32)
    goal=plan['goal'];delta=goal-position;hint=plan['hint'];target=plan['target']
    navigation=c.skill=='navigate' or policy_family=='navigation'
    if navigation:
        target=None
        # The local controller sees a bounded route waypoint, not a distant global
        # destination whose scale changes when commands span the whole map.
        delta=hint/max(.001,np.linalg.norm(hint))*min(2.,plan['distance'])
    target_delta=xy(target)-position if target else np.zeros(2)
    region=np.array(c.center) if c.center else executor.anchor
    if navigation:region=position
    values=[float(('navigate' if navigation else c.skill)==s) for s in SKILLS]
    values += [0. if navigation else robot['x']/28,0. if navigation else robot['y']/15,robot['z'],robot['vx']/4,robot['vy']/4,robot['vz']/4,
               math.sin(robot['heading']),math.cos(robot['heading']),robot['hp']/max(1,robot['max_hp']),
               1. if navigation else robot['ammo']/200,0. if navigation else robot.get('heat',0)/max(1,robot.get('max_heat',40)),robot['upright'],float(robot['grounded']),
               float(hint[0]),float(hint[1]),delta[0]/5,delta[1]/5,(min(2.,plan['distance']) if navigation else plan['distance'])/10,
               (2.5 if navigation else c.preferred_range_m)/8,c.radius_m/3,(region[0]-position[0])/10,(region[1]-position[1])/10,
               (3. if navigation else c.region_radius_m)/10,(8. if navigation else c.leash_radius_m)/35]
    values += [float(('navigate' if navigation else plan['phase'])==p) for p in ('navigate','approach','engage','search','hold')]
    values += [0. if navigation else float(plan['visible']),0. if navigation else plan['age']/max(.1,c.target_lost_timeout_s),target_delta[0]/10,target_delta[1]/10,
               target.get('vx',0)/4 if target else 0,target.get('vy',0)/4 if target else 0,
               target.get('hp',0)/max(1,target.get('max_hp',400)) if target else 0]
    values += [float(bool(target) and (target.get('role')==role if role!='structure' else target.get('kind') in ('base','outpost')))
               for role in ('heavy','infantry','sentry','structure')]
    values += [0. if navigation else math.sin(robot['aim']-robot['heading']),1. if navigation else math.cos(robot['aim']-robot['heading']),0. if navigation else robot['pitch']]
    ground=sdk.terrain_height(executor.arena,*position)
    values += [sdk.terrain_height(executor.arena,position[0]+dx,position[1]+dy)-ground for dx,dy in ((.5,0),(-.5,0),(0,.5),(0,-.5))]
    result=np.clip(np.asarray(values,dtype=np.float32),-10,10)
    if result.shape!=OBS_SPACE.shape:raise RuntimeError('Tactical schema dimension mismatch')
    return result


def decode(executor,observation,action,policy_family=None):
    from rmuc_arena import sdk
    a=np.asarray(action)
    if a.shape!=(7,) or not np.issubdtype(a.dtype,np.integer) or not ACTION_SPACE.contains(a):raise ValueError('Invalid tactical MultiDiscrete action')
    robot=own(observation,executor.slot)
    result={'move':[0.,0.],'heading':robot['heading'],'aim':robot['aim'],'pitch':robot['pitch'],
            'fire':False,'brake':True,'leg_height':[.18,.245,.31][int(a[5])],'jump':0.}
    if executor.command and (executor.command.skill=='navigate' or policy_family=='navigation'):
        result['leg_height']=max(.245,result['leg_height'])
    if executor.status!='running':return {executor.slot:result}
    plan=executor.plan;hint=np.array(plan['hint']);length=np.linalg.norm(hint)
    if length<.001 and plan['target'] and plan['visible'] and policy_family!='navigation':
        # Orbit/reposition remains possible inside the preferred firing range.
        hint=xy(plan['target'])-xy(robot);length=np.linalg.norm(hint)
    if length>.001:
        unit=hint/length;left=np.array([-unit[1],unit[0]])
        modes=(np.zeros(2),unit,-unit,left,-left)
        move=modes[int(a[0])]*[.15,.4,.7,1.][int(a[1])]
        # Respect the public planner's very small command when turning before bumps.
        if length<.05:move*=length/.05
        result['move']=move.tolist()
        if np.linalg.norm(move):result['heading']=plan.get('held_heading') if plan.get('held_heading') is not None else math.atan2(move[1],move[0])
    target=plan['target']
    if target and plan['visible'] and target.get('alive',target.get('hp',0)>0):
        result['aim']=sdk.aim_at(robot,target)+(int(a[2])-1)*.025
        result['pitch']=sdk.aim_pitch(robot,target)+(int(a[3])-1)*.02
        result['fire']=bool(a[4]) and policy_family!='navigation'
    elif executor.command.skill!='navigate':
        result['aim']=(robot['aim']+.5)%math.tau
    result['jump']=float(a[6]) if executor.command.skill=='navigate' or policy_family=='navigation' else 0.
    return {executor.slot:result}
