"""Wounded withdrawal, kerb escape, and physical upset preparation for model 305."""
import math
import numpy as np
from .tactical_envs import Episode
from .executor import own


def sample(seed):
    rng=np.random.default_rng(seed);kind=int(rng.integers(0,5));warmup=None
    if kind==4:kind=int(rng.choice([4,5,6,7]))
    blue={'blue-1':{'x':25.,'y':9.5,'heading':math.pi,'ammo':0},
          'blue-3':{'x':25.,'y':5.2,'heading':math.pi,'ammo':0},'blue-4':{'x':23.5,'y':9.5,'heading':math.pi,'ammo':0}}
    if kind<2 or kind==6:
        # A genuine fall is induced by ordinary high-leg/jump/lean commands against
        # the platform edge. No quaternion, HP or referee state is assigned.
        mirror=kind==1
        point=(14.5,2.25,-math.pi/2) if mirror else (9.5,7.5,0.)
        red={'x':point[0],'y':point[1],'heading':point[2],'ammo':0}
        command={'id':f'recover-{seed}','skill':'recover','timeout_s':14.,'low_hp_fraction':0.}
        if kind==6:command={'id':f'recover-retreat-{seed}','skill':'retreat','point':[1.4,13.],'timeout_s':110.,'low_hp_fraction':0.}
        warmup='tip'
    else:
        if kind==2:
            red={'x':4.2,'y':10.6,'heading':2.5,'ammo':0}
            blue['blue-3']={'x':7.3,'y':10.6,'heading':math.pi,'ammo':100};warmup='weak'
            goal=[1.4,13.]
        elif kind==3:
            red={'x':14.52,'y':2.23,'heading':-math.pi/2,'ammo':0};goal=[1.4,13.]
        elif kind==4:
            red={'x':11.1,'y':6.51,'heading':-.47,'ammo':0};goal=[8.3,5.2]
        elif kind==5 or kind==7:
            red={'x':float(3.89+rng.uniform(-.02,.02)),'y':float(13.43+rng.uniform(-.05,.05)),'heading':float(rng.uniform(-.04,.04)),'ammo':0};goal=[1.4,13.]
            if kind==7:goal=[float(rng.uniform(4.55,5.1)),float(rng.uniform(13.2,13.8))]
        command={'id':f'retreat-{seed}','skill':'retreat','point':goal,'timeout_s':110.,'low_hp_fraction':0.}
    episode=Episode('mobility',int(seed),{'robots':{'red-3':red,**blue}},command,{},1200)
    episode.warmup=warmup;episode.mobility_variant=kind
    return episode


def warmup(engine,ids,states,episodes):
    from .extended_curricula import warmup as wound
    states=list(states);dead=set();prepared=set()
    from rmuc_arena import sdk
    for step in range(1300):
        active=[];commands=[];positions=[]
        for k,e in enumerate(episodes):
            if getattr(e,'warmup',None)!='weak' or k in prepared:continue
            r=own(states[k]['observations']['red'])
            if not r['alive']:dead.add(k)
            if k in dead and r['alive']:
                prepared.add(k);continue
            action={'move':[0,0],'fire':False}
            if k not in dead:
                blue=next(a for a in states[k]['observations']['blue']['robots'] if a['id']=='blue-3')
                target=next((a for a in states[k]['observations']['blue']['enemies'] if a['id']=='red-3'),None)
                if target:action.update(aim=sdk.aim_at(blue,target),pitch=sdk.aim_pitch(blue,target),fire=True)
                else:action['aim']=blue['aim']+.5
            active.append(ids[k]);positions.append(k);commands.append({'blue':{'3':action}})
        if not active:break
        for k,state in zip(positions,engine.step(commands,env_ids=active)):states[k]=state
    if any(getattr(e,'warmup',None)=='weak' and k not in prepared for k,e in enumerate(episodes)):
        raise RuntimeError('Weak-respawn preparation did not complete through ordinary damage and respawn')
    states=list(states);done=set()
    for step in range(180):
        active=[];commands=[];positions=[]
        for k,e in enumerate(episodes):
            if getattr(e,'warmup',None)!='tip' or k in done:continue
            r=own(states[k]['observations']['red'])
            if r['upright']<.4:done.add(k);continue
            heading=e.scenario['robots']['red-3']['heading']
            action={'move':[math.cos(heading),math.sin(heading)],'heading':heading,'power':200.,'leg_height':.34,
                    'jump':float(step%10==0),'lean':.35}
            active.append(ids[k]);positions.append(k);commands.append({'red':{'3':action}})
        if not active:break
        for k,state in zip(positions,engine.step(commands,env_ids=active)):states[k]=state
    for k,e in enumerate(episodes):
        if getattr(e,'warmup',None)=='tip' and own(states[k]['observations']['red'])['upright']>=.4:
            raise RuntimeError('Physical recovery preparation did not produce an upset robot')
    return states,[None]*len(ids)
