"""Real RFID occupation and armed approach courses for one serial-leg infantry."""
import math
import numpy as np


def sample(family,seed):
    from .tactical_envs import Episode,sample_episode
    rng=np.random.default_rng(seed)
    if rng.random()<.25:return sample_episode(family,seed,305)
    robots={'blue-1':{'x':25.,'y':9.5,'ammo':0,'heading':math.pi},'blue-3':{'x':25.,'y':5.2,'ammo':0,'heading':math.pi}}
    command={'id':f'objective-{seed}','timeout_s':24.,'low_hp_fraction':.1}
    behavior={}
    if family=='defense':
        x=10.8 if rng.integers(2)==0 else 17.2
        center=[x,7.5];start=[x+float(rng.uniform(-.12,.12)),7.5+float(rng.choice([-1,1])*rng.uniform(.6,1.5))]
        armed=bool(rng.integers(2));enemy=[x+(1.2 if x<14 else -1.2),5.5 if start[1]>7.5 else 9.5]
        robots['blue-3'].update(x=enemy[0],y=enemy[1],ammo=100 if armed else 0)
        if armed:behavior['blue-3']={'mode':'shoot','target_id':'red-3'}
        command.update(skill='defend_region',center=center,region_radius_m=2.5)
        variant='armed_control_hold' if armed else 'control_entry'
    else:
        start=[19.2+float(rng.uniform(-.2,.2)),5.2+float(rng.uniform(-.2,.2))]
        robots['blue-3'].update(x=21.,y=5.,ammo=100)
        behavior['blue-3']={'mode':'shoot','target_id':'red-3'}
        command.update(skill='attack_structure',target_id='blue_outpost',preferred_range_m=2.5,damage_goal=120.)
        center=[17.024,3.6];variant='outpost_under_fire'
    heading=math.atan2(center[1]-start[1],center[0]-start[0])
    robots['red-3']={'x':start[0],'y':start[1],'heading':heading,'aim':heading,'ammo':200}
    episode=Episode(family,int(seed),{'robots':robots},command,behavior,240);episode.variant=variant
    return episode
