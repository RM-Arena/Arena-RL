"""Seeded tasks for support and real 2026 fly-slope training."""
import math
import numpy as np
from .tactical_envs import Episode

SUPPORT_LINEUPS={'red':[{'slot':3,'model':3},{'slot':4,'model':4}],
                'blue':[{'slot':1,'model':2711},{'slot':3,'model':3},{'slot':4,'model':4}]}
FLIGHT_LINEUPS={t:[{'slot':3,'model':3}] for t in ('red','blue')}


def sample(family,seed):
    rng=np.random.default_rng(seed)
    parked={'blue-1':{'x':25.,'y':9.5,'heading':math.pi,'ammo':0},'blue-3':{'x':25.,'y':5.2,'heading':math.pi,'ammo':0},
            'blue-4':{'x':23.5,'y':9.5,'heading':math.pi,'ammo':0},'red-4':{'x':2.8,'y':5.1,'heading':0.,'ammo':0}}
    behavior={};warmup=None
    if family=='flight':
        reverse=bool(rng.integers(0,2));lane_team='red' if rng.integers(0,2)==0 else 'blue'
        direction=1. if lane_team=='red' else -1.
        x=(10. if lane_team=='red' else 18.)+rng.uniform(-.15,.15);y=(14.4 if lane_team=='red' else .6)+rng.uniform(-.06,.06)
        if reverse:x=10. if lane_team=='red' else 18.;y=14.4 if lane_team=='red' else .6
        scenario={'robots':{'red-3':{'x':float(x),'y':float(y),'heading':0. if direction>0 else math.pi,'aim':0.,'ammo':0}}}
        command={'id':f'flight-{seed}','skill':'fly_slope','lane_team':lane_team,'flight_direction':'reverse' if reverse else 'forward','timeout_s':18.,'low_hp_fraction':0.}
        warmup='forward_flight' if reverse else None
        e=Episode(family,int(seed),scenario,command,behavior,180)
    else:
        kind=('resupply','escort','patrol')[int(rng.integers(0,3))]
        if kind=='resupply':
            red={'x':float(4.2+rng.uniform(-.15,.15)),'y':float(10.6+rng.uniform(-.12,.12)),'heading':2.5,'aim':0.,'ammo':0}
            parked['blue-3']={'x':7.3,'y':10.6,'heading':math.pi,'ammo':100}
            command={'id':f'supply-{seed}','skill':'resupply','ammo_target':100,'hp_target_fraction':.9,'budget_limit':150.,'timeout_s':60.}
            warmup='damage'
        elif kind=='escort':
            red={'x':3.4,'y':5.35,'heading':0.,'aim':0.,'ammo':100}
            parked['red-4']={'x':4.8,'y':5.35,'heading':0.,'ammo':0}
            goal=[float(7.5+rng.uniform(-.2,.2)),5.35]
            behavior['red-4']={'mode':'move','center':goal,'speed':.14}
            command={'id':f'escort-{seed}','skill':'escort','target_id':'red-4','point':goal,'follow_distance_m':1.2,
                     'required_coverage':.6,'leash_radius_m':8.,'timeout_s':30.}
        else:
            red={'x':4.6,'y':5.15,'heading':0.,'aim':0.,'ammo':0}
            dx=float(rng.uniform(-.15,.15));waypoints=[[6.8+dx,5.15],[6.8+dx,5.7],[4.8,5.5]]
            command={'id':f'patrol-{seed}','skill':'patrol','waypoints':waypoints,'patrol_loops':1,'dwell_s':.3,'timeout_s':40.}
        e=Episode(family,int(seed),{'robots':{'red-3':red,**parked}},command,behavior,600)
    e.warmup=warmup
    return e


def warmup(engine,ids,states,episodes):
    """Ordinary actions establish wounds or reverse-flight permission; no referee edits."""
    from rmuc_arena import sdk
    from .executor import TacticalExecutor,own
    from .extended_skills import update
    states=list(states);executors=[None]*len(ids)
    # Prepare reverse episodes by genuinely completing a forward crossing first.
    for k,e in enumerate(episodes):
        if getattr(e,'warmup',None)=='forward_flight':
            ex=TacticalExecutor();ex.submit({'id':'prelude-forward','skill':'fly_slope','lane_team':e.command['lane_team'],'timeout_s':20.,'low_hp_fraction':0.},states[k]['observations']['red']);executors[k]=ex
    for step in range(200):
        active=[];commands=[];positions=[]
        for k,e in enumerate(episodes):
            if getattr(e,'warmup',None)=='damage':
                r=own(states[k]['observations']['red'])
                if r['hp']<=.7*r['max_hp']:continue
                blue=next(r for r in states[k]['observations']['blue']['robots'] if r['id']=='blue-3')
                target=next((enemy for enemy in states[k]['observations']['blue']['enemies'] if enemy['id']=='red-3'),None)
                if target is None:
                    commands.append({'blue':{'3':{'move':[0,0],'aim':blue['aim']+.5,'fire':False}}})
                    active.append(ids[k]);positions.append(k);continue
                commands.append({'blue':{'3':{'move':[0,0],'aim':sdk.aim_at(blue,target),'pitch':sdk.aim_pitch(blue,target),'fire':True}}})
            elif getattr(e,'warmup',None)=='forward_flight':
                ex=executors[k]
                if ex.status=='succeeded':continue
                r=own(states[k]['observations']['red']);lane=sdk.fly_slope_lane(ex.arena,e.command['lane_team'])
                lip=np.array(lane['lip']);crest=np.array(lane['crest']);axis=(crest-lip)/np.linalg.norm(crest-lip);landing=np.array(lane['landing'])
                progress=np.dot(np.array([r['x'],r['y']])-crest,axis)
                speed=1. if progress<(1.05 if r.get('model_slot')==305 else .6) else float(np.clip(np.dot(landing-[r['x'],r['y']],axis)*.8,-.4,.4))
                if ex.plan['phase'] in ('flight_align','flight_land'):speed=0.
                move=axis*speed;move[1]+=float(np.clip((lip[1]-r['y'])*1.5,-.2,.2))
                commands.append({'red':{'3':{'move':move.tolist(),
                     'heading':math.atan2(axis[1],axis[0]),'power':200.,'jump':float(r.get('model_slot')==305 and -.2<progress<.35),'leg_height':.3 if r.get('model_slot')==305 else .245,'brake':True}}})
            else:continue
            active.append(ids[k]);positions.append(k)
        if not active:break
        new=engine.step(commands,env_ids=active)
        for k,state in zip(positions,new):
            states[k]=state
            if executors[k]:executors[k].update(state['observations']['red'])
    for k,e in enumerate(episodes):
        if getattr(e,'warmup',None)=='damage' and own(states[k]['observations']['red'])['hp']>.7*own(states[k]['observations']['red'])['max_hp']:
            raise RuntimeError('Damage preparation did not establish a wounded robot')
        if executors[k] and executors[k].status!='succeeded':raise RuntimeError(f'Forward crossing prelude failed: {e.seed} {executors[k].feedback()} {own(states[k]["observations"]["red"])}')
    return states,executors
