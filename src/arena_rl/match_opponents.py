"""Declared deterministic opponents for full-duration single-robot evaluations.

These controllers receive only their own team's legal observation. They are not
part of the LLM/RL agent and never write actions for the learned red slot.
"""
import math
import numpy as np
from .executor import xy
from .navigation import clearance_hint

PROFILES=('siege','raid','hunt')
LINEUPS={'red':[{'slot':3,'model':3}],
         'blue':[{'slot':1,'model':2711},{'slot':3,'model':3}]}


def actions(observation,arena,profile='siege'):
    from rmuc_arena import sdk
    if profile not in PROFILES:raise ValueError('Unknown match opponent')
    team=observation['team'];enemy='red' if team=='blue' else 'blue';commands={}
    supply=next(z for z in arena['zones'] if z['kind']=='supply' and z.get('owner')==team)
    for robot in observation['robots']:
        slot=str(robot['slot']);position=xy(robot)
        action={'move':[0.,0.],'heading':robot['heading'],'aim':robot['aim'],'pitch':robot['pitch'],'fire':False,
                'brake':True}
        if robot.get('chassis_type')=='wheel_legged':action['leg_height']=.245
        commands[slot]=action
        if not robot['alive'] or not robot['powered']:continue
        coins=observation['teams'][team]['resources'];caliber='42mm' if robot['role'] in ('hero','heavy') else '17mm'
        rule=sdk.rules(observation)['economy']['ammo'][caliber]
        ledger=observation['teams'][team].get('purchased_'+caliber[:2],0)
        desired_ammo=20 if caliber=='42mm' else 150
        contact=robot['id'] in observation['zones'][supply['id']]['occupants']
        restock=robot['ammo']==0 and ledger<rule['team_purchase_cap']
        if restock or (robot['hp']<robot['max_hp']*.4 and coins>0):
            goal=np.array([supply['x'],supply['y']])
            if contact:
                batch=int(rule['local_batch']);unit=rule['local_cost']/batch
                amount=min(math.ceil(max(0,desired_ammo-robot['ammo'])/batch)*batch,
                           int(coins/unit)//batch*batch,int(rule['team_purchase_cap']-ledger)//batch*batch)
                if amount>0:action['buy_ammo']={'amount':amount,'remote':False}
                goal=position
            target=None
        else:
            visible=set(observation.get('vision',{}).get(slot,{}).get('enemies',[]))
            threats=[r for r in observation['enemies'] if r['id'] in visible and r['alive'] and r.get('targetable',True)]
            nearest=min(threats,key=lambda e:np.linalg.norm(xy(e)-position),default=None)
            # Siege commits to structures; hunt seeks visible robots; raid takes local fights.
            intercept=nearest is not None and (profile=='hunt' or np.linalg.norm(xy(nearest)-position)<(2.8 if profile=='siege' else 4.5))
            if intercept:target=nearest
            else:
                post=observation['outposts'][enemy]
                protection=observation.get('positions',{}).get(enemy+'_outpost',{})
                protected=protection.get('control')=='full' and protection.get('controller')==enemy
                target={**(post if protected or (post['hp']>0 and profile=='siege') else observation['bases'][enemy]),'kind':'outpost' if protected or (post['hp']>0 and profile=='siege') else 'base'}
            delta=position-xy(target);separation=np.linalg.norm(delta)
            goal=xy(target)+delta/max(.01,separation)*(3. if caliber=='42mm' else 2.5)
            if abs(separation-(3. if caliber=='42mm' else 2.5))<.45:goal=position
            action.update(aim=sdk.aim_at(robot,target),pitch=sdk.aim_pitch(robot,target),fire=bool(robot['ammo'] and separation<4.))
        if np.linalg.norm(goal-position)>.3:
            hint=sdk.navigate(robot,goal,arena);hint,heading=clearance_hint(robot,hint,arena)
            move=np.array(hint);length=np.linalg.norm(move)
            speed=.65 if np.linalg.norm(goal-position)>1.2 else .22
            if length>.001:
                action['move']=(move*speed/max(1.,length)).tolist()
                action['heading']=heading if heading is not None else math.atan2(move[1],move[0])
    return commands
