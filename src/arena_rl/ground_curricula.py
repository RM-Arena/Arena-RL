"""Seeded 2027 courses for universal ground policies across every legal ground model.

Each episode deploys one learner (heavy slot 1, infantry slot 3 or sentry slot 7)
plus declared scripted robots. Initial poses and ammunition use Arena's training
scenario fields only. Structure damage, opened base armor and fitted launchers are
earned through ordinary actions during warmup; no referee state is edited.

An episode can contain several stages. Later stages continue the same simulator
match with a new order, exactly as the LLM changes orders in a live match.
Unrecorded transit stages are driven by the chassis-aware expert and never enter
PPO batches.
"""
from dataclasses import dataclass,field
import functools
import math
import numpy as np
from .robots import GROUND_MODELS_2027

ROLE_SLOT={'heavy':'1','infantry':'3','sentry':'7'}
# Public 2027 landmarks, red perspective (map is point-symmetric about (14, 7.5)).
LANDMARKS={'red_base_front':(3.9,7.5),'red_spawn_n':(3.2,8.85),'red_spawn_s':(3.2,6.15),'red_fortress':(6.571,7.5),
           'red_highland':(8.77,7.5),'red_central':(10.8,7.5),'red_outpost_front':(9.9,10.4),'red_supply':(1.4,13.0),
           'red_road':(9.0,13.9),'red_trapezoid':(5.5,2.2),'open_south':(8.2,5.0),'open_north':(8.2,10.0),
           'rune':(14.0,7.5),'blue_central':(17.2,7.5),'blue_outpost_west':(15.0,4.6),'blue_outpost_east':(19.3,4.9),
           'blue_highland':(19.23,7.5),'blue_fortress':(21.429,7.5),'blue_base_front':(23.6,7.5),'open_far_south':(19.8,5.0),
           'open_far_north':(19.8,10.0),'blue_road':(19.0,1.1)}
BLUE_OUTPOST=(17.024,3.6);BLUE_BASE=(25.589,7.5)
PARK={'blue-1':(25.,9.6),'blue-3':(25.,5.4),'blue-4':(23.6,9.6),'blue-7':(26.6,10.4),'red-1':(2.9,9.6),'red-3':(2.6,6.0),'red-4':(2.6,9.0),'red-7':(1.6,5.6)}


@dataclass
class Stage:
    command:dict
    behavior:dict
    max_steps:int=300
    recorded:bool=True


@dataclass
class GroundEpisode:
    family:str
    seed:int
    model:int
    slot:str
    lineups:dict
    scenario:dict
    stages:list
    warmup:list=field(default_factory=list)
    variant:str=''

    @property
    def learner(self):return f'red-{self.slot}'


@functools.lru_cache(maxsize=1)
def _catalog():
    from rmuc_arena import sdk
    from rmuc_arena.models import robot_models
    return sdk.ARENAS_BY_SEASON['2027'],robot_models()


def role_of(model):return _catalog()[1][str(model)]['role']


def radius_of(model):
    size=_catalog()[1][str(model)]['chassis_size_m'];return math.hypot(size[0],size[1])/2


@functools.lru_cache(maxsize=64)
def _route_arena(model):
    from .navigation import navigation_arena
    design=_catalog()[1][str(model)];legged=design['chassis_type']=='wheel_legged'
    return navigation_arena(_catalog()[0],305 if legged else int(model),avoid_bumps=not legged)


def clear(point,model,team='red'):
    from .navigation import clear_point
    return clear_point(point,{'radius':radius_of(model),'team':team},_route_arena(model))


def near(rng,center,model,low=0.,high=.25,team='red',tries=40):
    """A clear point within an annulus around center (falls back to center)."""
    for _ in range(tries):
        angle=rng.uniform(-math.pi,math.pi);length=rng.uniform(low,high)
        p=(float(center[0]+length*math.cos(angle)),float(center[1]+length*math.sin(angle)))
        if clear(p,model,team):return p
    return (float(center[0]),float(center[1]))


def route_length(a,b,model):
    """Public grid route length (falls back to a detour-weighted straight line)."""
    from .navigation_grid import route_distance
    length=route_distance(a,b,_route_arena(model),'red')
    return length if math.isfinite(length) else 1.6*math.dist(a,b)


def mirror(point):return (28.-point[0],15.-point[1])


def choose_model(rng,models=None,role=None):
    pool=[m for r,ms in GROUND_MODELS_2027.items() if role in (None,r) for m in ms] if models is None else list(models)
    return int(pool[int(rng.integers(0,len(pool)))])


def pose(point,heading=0.,ammo=0,aim=None):
    return {'x':float(point[0]),'y':float(point[1]),'heading':float(heading),'aim':float(heading if aim is None else aim),'ammo':int(ammo)}


def facing(a,b):return math.atan2(b[1]-a[1],b[0]-a[0])


def ammo_for(model,rounds_17=150,rounds_42=30):return rounds_42 if role_of(model)=='heavy' else rounds_17


def damage_goal(model,goal_17=80.,goal_42=400.):return goal_42 if role_of(model)=='heavy' else goal_17


def preferred_range(rng,model):
    return float(rng.uniform(3.,5.5)) if role_of(model)=='heavy' else float(rng.uniform(2.,3.6))


def prune(enemies,robots,behavior,command,keep=()):
    """Only deploy opponents an episode uses: parked robots still cost physics and observations."""
    used={k for k in behavior if k.startswith('blue')}|set(keep)|{command.get('target_id')}|set(command.get('priority_ids',()))
    chosen=[(s,m) for s,m in enemies if f'blue-{s}' in used] or [(1,2711)]
    for k in [k for k in robots if k.startswith('blue') and k not in {f'blue-{s}' for s,_ in chosen}]:robots.pop(k)
    return chosen


def lineup(model,slot,allies=(),enemies=((1,2711),)):
    red=[{'slot':int(slot),'model':int(model)}]+[{'slot':int(s),'model':int(m)} for s,m in allies if str(s)!=str(slot)]
    blue=[{'slot':int(s),'model':int(m)} for s,m in enemies]
    return {'red':red,'blue':blue}


def _launcher_prefix(model,robots=None,behavior=None):
    """A heavy can only fire after the referee unlocks and the workbench fits its launcher.

    With a blue robot holding our central card, blue controls both outpost positions and
    the public unlock formula (300 - M/50 - 2C/3 s) releases our launcher near 185 s.
    """
    if role_of(model)!='heavy':return []
    if robots is not None:
        robots['blue-4']=pose((10.8,7.5),math.pi,0);behavior['blue-4']={'mode':'hold','center':[10.8,7.5],'speed':.4}
        return [{'kind':'launcher'},{'kind':'own_outpost_lost','robot':'blue-4'}]
    return [{'kind':'launcher'}]


def _transit(point,seed,timeout=40.):
    return Stage({'id':f'transit-{seed}','skill':'navigate','point':[float(point[0]),float(point[1])],'radius_m':.5,'timeout_s':timeout,'low_hp_fraction':0.},{},int(timeout*10),False)


def sample(family,seed,models=None):
    rng=np.random.default_rng(seed)
    if models is None and family in ('combat','defense'):
        # A heavy episode costs several times the wall time (launcher warmup and transit).
        role=str(rng.choice(['infantry','heavy','sentry'],p=[.74,.12,.14]));model=choose_model(rng,None,role)
    else:model=choose_model(rng,models)
    role=role_of(model);slot=ROLE_SLOT[role]
    if family=='navigation':return _navigation(rng,seed,model,slot)
    if family=='combat':return _combat(rng,seed,model,slot)
    if family=='defense':return _defense(rng,seed,model,slot)
    if family=='support':return _support(rng,seed,model,slot)
    raise ValueError('Unknown ground policy family')


def _navigation(rng,seed,model,slot):
    _,_=_catalog();names=list(LANDMARKS)
    from rmuc_arena import sdk
    nodes=[tuple(p) for p in sdk.ARENAS_BY_SEASON['2027']['navigation']['nodes']]
    points=[LANDMARKS[n] for n in names]+nodes
    def pick():
        for _ in range(60):
            p=points[int(rng.integers(0,len(points)))];q=near(rng,p,model,0.,.35)
            if clear(q,model):return q
        return LANDMARKS['red_base_front']
    stages=[];start=pick();previous=start
    variant=str(rng.choice(['route','route','route','retreat']))
    for k in range(3):
        for _ in range(30):
            goal=pick();length=route_length(previous,goal,model)
            if 2.5<=math.dist(goal,previous) and length<=(20. if k==0 else 13.):break
        timeout=float(np.clip(10.+3.*min(length,30.),15.,80.))
        command={'id':f'nav-{seed}-{k}','skill':'navigate','point':[goal[0],goal[1]],'radius_m':float(rng.choice([.35,.45,.6])),'timeout_s':timeout,'low_hp_fraction':0.}
        if variant=='retreat' and k==2:
            command={'id':f'retreat-{seed}','skill':'retreat','radius_m':.6,'timeout_s':70.}
        stages.append(Stage(command,{},int(timeout*10)+10));previous=goal
    scenario={'robots':{f'red-{slot}':pose(start,rng.uniform(-math.pi,math.pi),ammo_for(model,40,4)),
                        **{k:pose(v,math.pi) for k,v in PARK.items() if k.startswith('blue-1')}}}
    return GroundEpisode('navigation',int(seed),model,slot,lineup(model,slot),scenario,stages,variant=variant)


def _combat(rng,seed,model,slot):
    variant=str(rng.choice(['outpost','outpost','engage','pursue','base']))
    learner=f'red-{slot}';ammo=ammo_for(model);heavy=role_of(model)=='heavy'
    enemies=[(1,2711),(3,int(rng.choice([2731,2732,3,4]))),(4,2732),(7,2771)];allies=[];keep=[]
    robots={k:pose(v,math.pi) for k,v in PARK.items() if k.startswith('blue')}
    behavior={};stages=[];warmup=_launcher_prefix(model,robots,behavior)
    if variant=='outpost':
        range_=preferred_range(rng,model)
        start=near(rng,BLUE_OUTPOST,model,range_+.5,range_+3.)
        if rng.random()<.45:
            defender=near(rng,(19.6,5.2),2731,0.,.6,'blue');robots['blue-3']=pose(defender,facing(defender,start),100);behavior['blue-3']={'mode':'shoot','target_id':learner}
        command={'skill':'attack_structure','target_id':'blue_outpost','preferred_range_m':range_,'damage_goal':damage_goal(model,100.,400.)}
        focus=BLUE_OUTPOST
    elif variant in ('engage','pursue'):
        identity=str(rng.choice(['blue-1','blue-3','blue-7'])) if variant=='engage' else 'blue-1'
        anchor=LANDMARKS[str(rng.choice(['open_south','open_north','open_far_south','open_far_north','red_highland']))]
        target=near(rng,anchor,2711,0.,.6,'blue');range_=preferred_range(rng,model)
        start=near(rng,target,model,range_+.8,range_+3.)
        armed=rng.random()<.5
        robots[identity]=pose(target,facing(target,start),60 if armed else 0)
        if variant=='pursue':
            refuge=LANDMARKS['blue_base_front'] if rng.random()<.6 else LANDMARKS[str(rng.choice(['blue_fortress','blue_central','open_far_north']))]
            behavior[identity]={'mode':'flee','refuge':list(refuge),'speed':float(rng.uniform(.25,.5)),'armed':bool(armed),'target_id':learner}
        else:
            behavior[identity]={'mode':'weave','origin':list(target),'speed':float(rng.uniform(.12,.3)),'armed':bool(armed),'target_id':learner}
        distractor=next(d for d in ('blue-3','blue-7','blue-1') if d!=identity);keep=[distractor]
        robots[distractor]=pose(near(rng,start,2731,1.2,2.,'blue'),0.)
        command={'skill':variant+'_target','target_id':identity,'preferred_range_m':range_,'leash_radius_m':14.,
                 'damage_goal':damage_goal(model,60.,200.),'target_lost_timeout_s':3.}
        focus=target
    else:
        # Base push (2027 chain): a helper captures the enemy outpost position on its central
        # card and keeps it; after 180 s a second helper holds the enemy fortress card for 10 s,
        # which opens the base armor. Both are earned by ordinary RFID occupation.
        # [capturer, fortress holder]; the holder is a balancing chassis that climbs the bevel.
        helpers={'1':[(4,2732),(3,2731)],'3':[(7,2771),(4,2731)],'7':[(4,2732),(3,2731)]}[slot]
        allies=helpers;capturer=f'red-{helpers[0][0]}';holder=f'red-{helpers[1][0]}'
        warmup=warmup+[{'kind':'capture','robot':capturer},{'kind':'armor_open','holder':holder}]
        # Helpers start on their cards; the referee still demands real contact time (5 s capture,
        # then 10 s on the fortress once it activates at 180 s).
        robots[capturer]=pose((17.2,7.5),-math.pi/2,0);robots[holder]=pose((21.429,7.5),0.,0)
        behavior[capturer]={'mode':'hold','center':[17.2,7.5],'speed':.4}
        range_=preferred_range(rng,model)+.6
        start=near(rng,BLUE_BASE,model,range_+1.,range_+3.)
        command={'skill':'attack_structure','target_id':'blue_base','preferred_range_m':range_,'damage_goal':damage_goal(model,100.,400.)}
        focus=BLUE_BASE
        if rng.random()<.4:
            robots['blue-3']=pose(near(rng,(23.,9.8),2731,0.,.4,'blue'),math.pi,100);behavior['blue-3']={'mode':'shoot','target_id':learner}
    timeout=float(rng.uniform(18.,26.)) if not heavy else 30.
    command.update(id=f'combat-{seed}',timeout_s=timeout,low_hp_fraction=.08)
    if heavy:
        # Fitted at the workbench, then driven to the course by the expert (unrecorded).
        stages.append(_transit(start,seed,60.))
        robots[learner]=pose(LANDMARKS['red_base_front'],0.,ammo)
    else:robots[learner]=pose(start,facing(start,focus),ammo)
    stages.append(Stage(command,behavior,int(timeout*10)+10))
    # Further orders in the same match amortize warmups (structures have HP to spare).
    if heavy or variant in ('outpost','base'):
        for k in range(2 if heavy or variant=='base' else 1):stages.append(Stage(dict(command,id=f'combat-{seed}-{k}'),behavior,int(timeout*10)+10))
    enemies=prune(enemies,robots,behavior,command,keep)
    return GroundEpisode('combat',int(seed),model,slot,lineup(model,slot,allies,enemies),{'robots':robots},stages,warmup,variant)


def _defense(rng,seed,model,slot):
    variant=str(rng.choice(['base_defense','base_defense','central_own','central_enemy','fortress_enemy','fortress_own','outpost_guard']))
    learner=f'red-{slot}';ammo=ammo_for(model);heavy=role_of(model)=='heavy'
    enemies=[(1,2711),(3,int(rng.choice([2731,2732,4,303]))),(4,int(rng.choice([2731,2732])))]
    robots={k:pose(v,math.pi) for k,v in PARK.items() if k.startswith('blue') and k!='blue-7'}
    behavior={};allies=[];warmup=[]
    if heavy and variant!='central_own':warmup=_launcher_prefix(model,robots,behavior)
    elif heavy:warmup=_launcher_prefix(model)
    if variant in ('base_defense','outpost_guard'):
        center=(LANDMARKS['red_base_front'] if rng.random()<.5 else (5.15,7.5)) if variant=='base_defense' else (10.2,10.2)
        start=near(rng,center,model,.3,1.6)
        invaders=['blue-3'] if rng.random()<.6 else ['blue-3','blue-1']
        for k,identity in enumerate(invaders):
            entry=near(rng,(center[0]+rng.uniform(4.5,7.),center[1]+rng.uniform(-2.,2.)),2711,0.,.5,'blue')
            armed=rng.random()<.5
            robots[identity]=pose(entry,math.pi,80 if armed else 0)
            behavior[identity]={'mode':'invade','center':list(center),'speed':float(rng.uniform(.12,.25)),'armed':bool(armed),'target_id':learner}
        command={'skill':'defend_region','center':list(center),'region_radius_m':2.5,'leash_radius_m':4.,
                 'priority_ids':invaders,'preferred_range_m':preferred_range(rng,model),'damage_goal':damage_goal(model,60.,200.)}
        timeout=24.
    else:
        card={'central_own':'red_central','central_enemy':'blue_central','fortress_enemy':'blue_fortress','fortress_own':'red_fortress'}[variant]
        zone=next(z for z in _catalog()[0]['zones'] if z['id']==card);center=(zone['x'],zone['y'])
        start=near(rng,center,model,2.,5.)
        if variant=='fortress_enemy':
            # The enemy fortress counts after 180 s and only while we control the enemy outpost position.
            helper=(4,2731) if slot!='4' else (3,2731);allies=[helper];capturer=f'red-{helper[0]}'
            robots[capturer]=pose((17.2,7.5),-math.pi/2,0)
            behavior[capturer]={'mode':'hold','center':[17.2,7.5],'speed':.4}
            warmup=warmup+[{'kind':'capture','robot':capturer},{'kind':'clock','until':float(rng.uniform(181.,200.))}]
        elif variant=='fortress_own':
            # The own fortress counts only after the enemy takes our outpost position.
            robots['blue-4']=pose((10.8,7.5),math.pi/2,0)
            behavior['blue-4']={'mode':'hold','center':[10.8,7.5],'speed':.4}
            warmup=warmup+[{'kind':'own_outpost_lost','robot':'blue-4'}]
        if rng.random()<.45:
            contester=near(rng,center,2731,1.5,2.6,'blue');armed=rng.random()<.6
            robots['blue-3']=pose(contester,facing(contester,start),80 if armed else 0)
            behavior['blue-3']={'mode':'hold' if variant.startswith('central') else 'invade','center':list(center),'speed':.15,'armed':bool(armed),'target_id':learner}
        command={'skill':'defend_region','center':[float(center[0]),float(center[1])],'region_radius_m':1.5,'leash_radius_m':3.,
                 'preferred_range_m':preferred_range(rng,model)}
        timeout=float(rng.uniform(16.,22.)) if variant!='fortress_enemy' else 24.
        # Flat chassis reach highland cards by the end slopes, not the band steps.
        timeout=float(min(60.,max(timeout,timeout-8.+2.6*route_length(start,center,model))))
    command.update(id=f'defense-{seed}',timeout_s=timeout,low_hp_fraction=.08)
    stages=[]
    if heavy:stages.append(_transit(start,seed,60.));robots[learner]=pose(LANDMARKS['red_base_front'],0.,ammo)
    else:robots[learner]=pose(start,facing(start,center),ammo)
    stages.append(Stage(command,behavior,int(timeout*10)+10))
    if heavy or variant in ('central_own','central_enemy','fortress_own'):stages.append(Stage(dict(command,id=f'defense-{seed}-b'),behavior,int(timeout*10)+10))
    enemies=prune(enemies,robots,behavior,command)
    return GroundEpisode('defense',int(seed),model,slot,lineup(model,slot,allies,enemies),{'robots':robots},stages,warmup,variant)


def _support(rng,seed,model,slot):
    variant=str(rng.choice(['resupply','resupply','escort','patrol']))
    learner=f'red-{slot}';heavy=role_of(model)=='heavy'
    robots={k:pose(v,math.pi) for k,v in PARK.items() if k in ('blue-1','blue-3')}
    behavior={};allies=[]
    if variant=='resupply':
        start=near(rng,(4.2,11.),model,0.,2.2)
        if not clear(start,model):start=near(rng,(3.2,10.2),model,0.,.6)
        command={'skill':'resupply','ammo_target':10 if heavy else int(rng.choice([50,100,150])),'hp_target_fraction':.5,'budget_limit':300.,'timeout_s':40.}
        robots[learner]=pose(start,facing(start,(1.4,13.)),0)
    elif variant=='escort':
        ally='red-4' if slot!='4' else 'red-3';allies=[(int(ally[-1]),2731)]
        origin=near(rng,(4.6,5.2 if rng.random()<.5 else 9.8),2731,0.,.4)
        goal=(float(origin[0]+rng.uniform(2.6,3.6)),float(origin[1]))
        start=near(rng,(origin[0]-1.3,origin[1]),model,0.,.3)
        robots[ally]=pose(origin,0.,0);robots[learner]=pose(start,0.,ammo_for(model))
        behavior[ally]={'mode':'move','center':list(goal),'speed':.14}
        command={'skill':'escort','target_id':ally,'point':list(goal),'follow_distance_m':1.4 if heavy else 1.2,'required_coverage':.55,'leash_radius_m':8.,'timeout_s':35.}
    else:
        base=LANDMARKS[str(rng.choice(['red_base_front','red_fortress','open_south','open_north']))]
        points=[near(rng,base,model,.8,1.6) for _ in range(3)]
        start=near(rng,base,model,0.,.5)
        robots[learner]=pose(start,0.,ammo_for(model))
        command={'skill':'patrol','waypoints':[list(p) for p in points],'patrol_loops':1,'dwell_s':.3,'radius_m':.45,'timeout_s':45.}
    command.update(id=f'support-{seed}')
    timeout=command['timeout_s']
    enemies=prune([(1,2711),(3,2731)],robots,behavior,command)
    return GroundEpisode('support',int(seed),model,slot,lineup(model,slot,allies,enemies),{'robots':robots},[Stage(command,behavior,int(timeout*10)+10)],[],variant)
