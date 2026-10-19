"""305 stopping, road-edge and perturbed fly-slope courses; reference seeds stay fixed."""
import math
import numpy as np


def sample(family,seed):
    from .tactical_envs import sample_episode
    episode=sample_episode(family,seed,305)
    rng=np.random.default_rng(seed+41000000)
    episode.variant='reference'
    if family=='navigation' and rng.random()<.65:
        kind=int(rng.integers(0,4));robot=episode.scenario['robots']['red-3']
        if kind==0:
            start=np.asarray(([4.7,5.15],[7.5,9.8],[20.5,9.8],[23.5,5.1])[int(rng.integers(0,4))])
            angle=float(rng.uniform(-math.pi,math.pi));goal=start+np.array([math.cos(angle),math.sin(angle)])*rng.uniform(.42,1.6)
        else:
            start,goal=(([19.8,1.3],[23.4,1.3]),([4.6,13.7],[8.2,13.7]),([5.4,12.3],[7.42,11.25]))[kind-1]
            if rng.random()<.5:start,goal=goal,start
            start=np.asarray(start)+rng.uniform(-.12,.12,2);goal=np.asarray(goal)+rng.uniform(-.12,.12,2)
        robot.update(x=float(start[0]),y=float(start[1]),heading=float(math.atan2(goal[1]-start[1],goal[0]-start[0])+rng.uniform(-.12,.12)))
        episode.command['point']=goal.tolist();episode.command['timeout_s']=28.;episode.max_steps=300
        episode.variant=('short_stop','blue_road_edge','red_road_edge','road_turn')[kind]
    if family=='flight' and episode.command['flight_direction']=='forward':
        robot=episode.scenario['robots']['red-3'];red=episode.command['lane_team']=='red'
        robot.update(x=float((10. if red else 18.)+rng.uniform(-.22,.22)),
                     y=float((14.4 if red else .6)+rng.uniform(-.1,.1)),
                     heading=float((0. if red else math.pi)+rng.uniform(-.08,.08)))
        episode.variant='perturbed_forward_'+episode.command['lane_team']
    elif family=='flight':episode.variant='earned_reverse_'+episode.command['lane_team']
    return episode
