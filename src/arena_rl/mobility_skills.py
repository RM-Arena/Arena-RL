"""Low-health withdrawal and supervised mechanical recovery, preserving current goals."""
import math
import numpy as np
from .executor import own,xy


def can_recover(robot):return robot.get('model_slot')==305


def recovery_plan(ex,obs):
    robot=own(obs,ex.slot);state=ex.recovery;now=obs['time']
    state.setdefault('started',now)
    ex.skill_state['last_time']=now
    ex.skill_state['dwell']=0.;ex.skill_state['landing_hold']=0.
    angular=np.linalg.norm(robot.get('angular_velocity',[0,0,0]))
    stable=robot['upright']>.93 and robot['grounded'] and math.hypot(robot['vx'],robot['vy'])<.6 and angular<1.5 and not robot.get('self_righting',False)
    previous=state.get('last',now);dt=max(0.,now-previous);state['last']=now
    state['hold']=state.get('hold',0.)+dt if stable else 0.
    state['min_upright']=min(state.get('min_upright',1.),robot['upright'])
    state['events_seen']=state.get('events_seen',False) or robot.get('self_righting',False)
    if state['hold']>=.5:
        ex.recovery_count+=1;ex.recovery={}
        ex.extra.update(stable_hold_s=state['hold'],recoveries_completed=ex.recovery_count)
        if ex.command.skill=='recover':ex._end('succeeded','recovered')
        return False
    if now-state['started']>=min(20.,ex.command.timeout_s):
        ex._end('failed','recovery_timeout');return False
    ex.plan={'phase':'recover','goal':xy(robot),'hint':np.zeros(2),'target':None,'visible':False,'distance':0.,'age':0.,'held_heading':robot['heading']}
    ex.extra={'recovery_elapsed_s':now-state['started'],'stable_hold_s':state['hold'],'min_upright':state['min_upright'],
              'mechanical_self_righting':robot.get('self_righting',False),'recoveries_completed':ex.recovery_count}
    return True


def retreat_plan(ex,obs):
    robot=own(obs,ex.slot);c=ex.command
    if c.point:goal=np.array(c.point)
    else:
        zone=next(z for z in ex.arena['zones'] if z['kind']=='supply' and z.get('owner')==obs['team'])
        goal=np.array([zone['x'],zone['y']])
    from .navigation import plan_motion
    hint,heading=plan_motion(ex,robot,goal,obs['time'])
    distance=float(np.linalg.norm(goal-xy(robot)))
    if distance<c.radius_m and math.hypot(robot['vx'],robot['vy'])<.8:ex._end('succeeded','retreated')
    if obs['time']-ex.started>=c.timeout_s:ex._end('failed','timeout')
    ex.plan={'phase':'retreat','goal':goal,'hint':hint,'target':None,'visible':False,'distance':distance,'age':0.,'held_heading':heading}
    ex.extra={'recoveries_completed':ex.recovery_count,'escape_attempts':ex.route_state.get('attempts',0)}
    if ex.initial_distance is None:ex.initial_distance=distance
    return ex.plan
