"""Aerial-system orders for the 2027 hangar: ramming strikes on hostile armor plates.

The hangar is fixed; its 2-4 aerial robots launch, fly and ram middle armor plates
with Arena's own flight physics (thrust, tilt, battery, guard integrity). This
controller is explicit guidance built on the SDK strike plan, not a learned
policy: the hangar's decision space is which plates and when, which belongs to
the team commander.
"""
import math
import numpy as np
from .executor import own

ORDERS=('idle','strike_outpost','strike_base','recall')
MIN_BATTERY_WH=.45
# Public 2027 rune gantry: posts 1.12 m either side of (14, 7.5) along a 45-degree axis, rune
# disk between them. Both hangars sit beside it, so straight flights can hit the frame.
GANTRY=np.array([14.,7.5]);GANTRY_AXIS=np.array([math.cos(math.pi/4),math.sin(math.pi/4)])


def detour(position,goal):
    """A waypoint around the gantry end when the straight flight would cross its frame, else goal."""
    s=np.asarray(position[:2],dtype=float);g=np.asarray(goal[:2],dtype=float);normal=np.array([-GANTRY_AXIS[1],GANTRY_AXIS[0]])
    ds=float(np.dot(s-GANTRY,normal));dg=float(np.dot(g-GANTRY,normal))
    if ds*dg>=0 or abs(ds-dg)<1e-6:return list(goal)
    cross=s+(g-s)*(ds/(ds-dg));along=float(np.dot(cross-GANTRY,GANTRY_AXIS))
    if abs(along)>1.7:return list(goal)
    end=GANTRY+GANTRY_AXIS*(2.0 if along>=0 else -2.0)+normal*(.35 if ds>0 else -.35)
    return [float(end[0]),float(end[1]),float(goal[2]) if len(goal)>2 else 1.8]


def dock_or_detour(drone,robot):
    dock=drone.get('dock') or [robot['x'],robot['y'],robot['z']]
    waypoint=detour([drone['x'],drone['y']],[dock[0],dock[1],max(1.8,dock[2])])
    return {'dock':True} if waypoint[:2]==[float(dock[0]),float(dock[1])] or math.dist(waypoint[:2],dock[:2])<1e-6 else {'goto':waypoint}


class AerialController:
    """policy: an optional learned dive-timing network (aerial_learning) used per drone on outposts."""
    def __init__(self,slot='6',policy=None):
        self.slot=str(slot);self.policy=policy;self.learned={};self.order='idle';self.order_id=None;self.status='idle';self.reason=None
        self.started=0.;self.strikes_at_start=0;self.assignment={};self.retreat={};self.dives={};self.time=0.

    def submit(self,order,observation,order_id='aerial'):
        if order not in ORDERS:raise ValueError(f'Aerial order must be one of {ORDERS}')
        robot=own(observation,self.slot)
        if robot['role']!='aerial':raise ValueError('Aerial orders require the hangar slot')
        self.order=order;self.order_id=order_id;self.status='running' if order in ('strike_outpost','strike_base') else 'idle';self.reason=None
        self.started=observation['time'];self.strikes_at_start=sum(d.get('strikes',0) for d in robot.get('drones',[]))
        self.assignment={};self.retreat={};self.dives={};self.learned={}
        return self.feedback(observation)

    def _plates(self,observation):
        from rmuc_arena import sdk
        kind='outpost' if self.order=='strike_outpost' else 'base'
        return sdk.hostile_plates(observation,kind)

    def action(self,observation):
        from rmuc_arena import sdk
        robot=own(observation,self.slot);self.time=observation['time'];commands={}
        drones=sdk.drones(robot)
        if self.order in ('strike_outpost','strike_base'):
            plates=self._plates(observation)
            if not plates and self.status=='running':
                struck=sum(d.get('strikes',0) for d in drones)>self.strikes_at_start
                self.status,self.reason=('succeeded','target_destroyed') if struck else ('failed','no_hostile_plates')
            if self.status=='running' and not any(d['battery_wh']>MIN_BATTERY_WH and d['integrity']>0 for d in drones):
                self.status,self.reason='failed','drones_exhausted'
        for k,drone in enumerate(drones):
            index=str(drone.get('index',k+1));usable=drone['battery_wh']>MIN_BATTERY_WH and drone['integrity']>0 and drone['state'] not in ('disabled',)
            if self.order in ('idle','recall') or not usable or self.status!='running':
                if drone['state'] in ('flying','down') and drone['battery_wh']>0:commands[index]=dock_or_detour(drone,robot)
                continue
            if self.policy is not None and self.order=='strike_outpost':
                from .aerial_learning import encode,command,MANEUVERS
                memory=self.learned.setdefault(index,{'bearing':None,'previous':0,'dive_s':0.})
                action=int(self.policy.predict(encode(observation,self.slot,int(index),memory['previous'],memory['dive_s']),deterministic=True)[0])
                commands[index],memory['bearing']=command(observation,self.slot,int(index),action,memory['bearing'])
                memory['dive_s']=memory['dive_s']+.1 if MANEUVERS[action].startswith('dive') else 0.;memory['previous']=action
                continue
            plate=self.assignment.get(index)
            if plate is None or plate>=len(plates):
                # Spread the drones over the hostile plates.
                plate=k%len(plates);self.assignment[index]=plate
            target=plates[plate]
            if drone.get('last_strike',-1e3)>observation['time']-.6 or self.retreat.get(index,-1)>observation['time']:
                # Back off along the plate normal after a hit, then line up again.
                if index not in self.retreat or self.retreat[index]<observation['time']:self.retreat[index]=observation['time']+.8
                n=target['normal'];c=target['center'];flat=math.hypot(n[0],n[1]) or 1.
                commands[index]={'goto':detour([drone['x'],drone['y']],[c[0]+n[0]/flat*1.4,c[1]+n[1]/flat*1.4,min(2.1,c[2]+.2)])};continue
            rotating=self.order=='strike_outpost' and not any(o.get('rotation_stopped',True) for o in observation['outposts'].values() if o['hp']>0 and o.get('owner')!=observation['team'])
            if rotating:
                # A spinning rotor never holds still for a line-up: dive radially at plate height
                # so the next passing plate meets the drone head-on.
                outpost=next(o for o in observation['outposts'].values() if o['hp']>0 and o.get('owner')!=observation['team'])
                angle=float(int(index)*2.1);u=np.array([math.cos(angle),math.sin(angle)])
                stand=np.array([outpost['x'],outpost['y']])+u*1.1;z=target['center'][2]
                p=np.array([drone['x'],drone['y']]);dive=self.dives.get(index,-1.)
                if dive<observation['time'] and np.linalg.norm(p-stand)<.1 and abs(drone['z']-z)<.08:dive=self.dives[index]=observation['time']+1.2
                if dive>=observation['time']:commands[index]={'move':[-u[0],-u[1],float(np.clip((z-drone['z'])*2,-1,1))]}
                else:commands[index]={'goto':detour([drone['x'],drone['y']],[float(stand[0]),float(stand[1]),float(z)])}
                continue
            phase,value=sdk.strike_plan(drone,target,standoff=1.)
            commands[index]={'goto':detour([drone['x'],drone['y']],list(value))} if phase=='approach' else {'move':list(value)}
        return {self.slot:{'drones':commands}}

    def feedback(self,observation=None):
        drones=own(observation,self.slot).get('drones',[]) if observation else []
        strikes=sum(d.get('strikes',0) for d in drones)
        return {'robot':'aerial','order':self.order,'order_id':self.order_id,'status':self.status,'reason':self.reason,
                'strikes_this_order':strikes-self.strikes_at_start,'drones':[{'index':d.get('index'),'state':d['state'],'battery_wh':round(d['battery_wh'],2),
                'integrity':d['integrity'],'strikes':d.get('strikes',0)} for d in drones]}
