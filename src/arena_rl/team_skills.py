"""Heavy-robot launcher logistics on public 2027 geometry.

The referee unlocks the modular 42 mm launcher (level-3 assembly or the time rule).
The heavy then parks still on its own workbench card for the published install time,
and its first shot confirms the fitting. Navigation to the dock uses the learned
navigation policy; the stationary docking and confirmation shot are explicit host
logic because they are referee procedures, not motor skills.
"""
import math
import numpy as np
from .executor import own,xy

DOCK_SPEED=.04


def validate_install(ex,command,observation):
    robot=own(observation,ex.slot)
    if robot['role']!='heavy':raise ValueError('install_launcher requires the heavy robot')
    if robot.get('launcher') in ('installed',):raise ValueError('Launcher is already installed')
    if not any(z['kind']=='workbench' and z.get('owner')==observation['team'] for z in (observation.get('arena') or ex.arena)['zones']):
        raise ValueError('No own workbench on this map')


def dock_point(ex,team):
    zone=next(z for z in ex.arena['zones'] if z['kind']=='workbench' and z.get('owner')==team)
    bench=next((f for f in ex.arena.get('field_geometry',{}).get('fixtures',[]) if f['kind']=='workbench' and f.get('team')==team),None)
    return np.asarray(bench['dock_point'] if bench and 'dock_point' in bench else [zone['x'],zone['y']],dtype=float),zone


def install_plan(ex,obs):
    r=own(obs,ex.slot);c=ex.command;now=obs['time'];p=xy(r)
    goal,zone=dock_point(ex,obs['team']);distance=float(np.linalg.norm(goal-p))
    state=r.get('launcher','locked');st=ex.skill_state
    if state=='installed':ex._end('succeeded','launcher_installed')
    elif now-ex.started>=c.timeout_s:ex._end('failed','timeout' if state!='locked' else 'launcher_locked')
    docked=distance<=max(.2,zone.get('radius',.45)-.1)
    phase='navigate'
    if docked:
        phase='install' if state in ('unlocked','installing') else 'confirm' if state=='fitted' else 'dock_wait'
    st['progress']=r.get('launcher_progress',0.)
    if ex.last_position is None or np.linalg.norm(p-ex.last_position)>.25:ex.last_position=p;ex.last_motion=now
    if phase=='navigate' and now-ex.last_motion>12:ex._end('failed','blocked')
    planned=np.zeros(2);held=None
    if phase=='navigate':
        from .navigation import plan_motion
        planned,held=plan_motion(ex,r,goal,now)
    ex.plan={'phase':phase,'goal':goal,'hint':np.asarray(planned,dtype=float),'target':None,'visible':False,
             'distance':distance,'age':0.,'held_heading':held,'arrival_radius':.2}
    ex.extra={'launcher':state,'launcher_progress':st['progress'],'docked':bool(docked),'ammo':r['ammo']}
    if ex.initial_distance is None:ex.initial_distance=distance
    if ex.status in ('succeeded','failed','cancelled'):ex.events.append(ex.feedback());ex.events=ex.events[-128:]
    return ex.plan


def install_action(ex,obs,out):
    """Hold still on the dock, request fitting, buy one shell if needed and fire the confirmation shot."""
    from rmuc_arena import sdk
    r=own(obs,ex.slot);phase=ex.plan['phase']
    if phase=='navigate':
        if ex.plan['distance']<.6:out['move']=[v*min(1.,ex.plan['distance']/.6) for v in out['move']]
        return out
    out.update(move=[0.,0.],brake=True,fire=False)
    if phase=='install':out['install_launcher']=True
    if phase=='confirm':
        if r['ammo']<1:
            rule=sdk.rules(obs)['economy']['ammo']['42mm']
            if obs['teams'][obs['team']]['resources']>=rule['local_cost'] and obs['teams'][obs['team']].get('purchased_42',0)<rule['team_purchase_cap']:
                out['buy_ammo']={'amount':int(rule['local_batch']),'remote':False}
        else:
            # Shoot away from the own base into the open field; the shot only confirms the fitting.
            out['aim']=0. if obs['team']=='red' else math.pi;out['pitch']=.05;out['fire']=True
    return out
