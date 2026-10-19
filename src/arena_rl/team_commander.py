"""Team-level harness between a strategic LLM and the per-robot RL executors.

The commander chooses one high-level objective per robot (attack the outpost,
defend the base, pursue the enemy heavy, occupy the enemy fortress, push the base,
...). The harness owns everything the model should not improvise:

* a public, team-relative scene summary with rule status (fortress activation,
  base protection/armor, launcher state) and recent events;
* per-robot capability and feasibility filtering, with reasons, so tool enums only
  contain objectives the robot can start now;
* compilation of objectives into validated tactical commands (named places, never
  free coordinates; ranges and timeouts per caliber);
* a deterministic doctrine used as the fallback when inference fails or times out,
  as an LLM-free baseline, and as the declared opponent commander.
"""
import math
from .executor import xy

GROUND_OBJECTIVES={
 'continue':'Keep the current order (or stay idle).',
 'hold_position':'Stop and guard the current spot (or a named place) for 30 s, returning fire.',
 'goto':'Drive to a named place.',
 'attack_outpost':'Destroy the enemy outpost from a safe firing position.',
 'push_base':'Shoot the enemy base. Only offered while it is unprotected and its armor is open.',
 'defend_base':'Guard the area in front of our base and shoot invaders (nearest first).',
 'capture_enemy_central':'Stand on the enemy central-highland card to capture the enemy outpost position (5 s contact) and keep it.',
 'hold_own_central':'Stand on our central-highland card to keep our outpost position.',
 'occupy_enemy_fortress':'Stand on the enemy fortress card; 10 s of contact opens the enemy base armor.',
 'hold_own_fortress':'Stand on our fortress card (defense buff and reserve ammunition while our outpost position is lost).',
 'pursue_enemy':'Chase a visible/recently seen enemy robot (e.g. the enemy heavy) within a leash and shoot it.',
 'engage_enemy':'Fight a visible enemy robot locally at preferred range.',
 'resupply':'Drive to our supply zone, buy ammunition for this caliber and heal.',
 'retreat':'Withdraw to our supply zone (allowed at low HP).',
 'escort':'Follow and protect an allied ground robot.',
 'patrol':'Patrol a named route, returning fire.',
 'install_launcher':'Heavy only: fit the unlocked 42 mm launcher at our workbench and fire its confirmation shot.',
}
AERIAL_OBJECTIVES={
 'continue':'Keep the current aerial order.',
 'drone_strike_outpost':'Launch the aerial robots to ram the enemy outpost armor (200 damage per valid hit).',
 'drone_strike_base':'Ram the exposed enemy base armor.',
 'drone_recall':'Return all aerial robots to the hangar.',
}
# Red-side public coordinates; blue uses the point reflection (28-x, 15-y).
PLACES={'own_base_front':(3.9,7.5),'own_fortress_front':(5.15,7.5),'own_highland':(8.77,7.5),'own_supply':(1.4,13.0),
        'own_outpost_front':(9.9,10.4),'north_lane':(8.2,10.0),'south_lane':(8.2,5.0),'center_rune':(14.0,7.5),
        'enemy_outpost_west':(15.0,4.6),'enemy_outpost_east':(19.3,4.9),'enemy_highland':(19.23,7.5),
        'enemy_fortress_front':(22.6,9.2),'enemy_base_front':(23.6,7.5),'far_north_lane':(19.8,10.0),'far_south_lane':(19.8,5.0)}
PATROLS={'base_perimeter':[(3.9,9.3),(5.2,7.5),(3.9,5.7)],'fortress_ring':[(5.4,9.0),(7.8,7.5),(5.4,6.0)],
         'outpost_line':[(9.9,10.4),(8.2,10.0),(8.2,8.6)]}
WEAPON_OBJECTIVES=('attack_outpost','push_base','pursue_enemy','engage_enemy')


def enemy_of(team):return 'blue' if team=='red' else 'red'


def place(team,name):
    x,y=PLACES[name];return [x,y] if team=='red' else [28.-x,15.-y]


def zone_center(arena,zone_id):
    z=next(z for z in arena['zones'] if z['id']==zone_id);return [float(z['x']),float(z['y'])]


class InfeasibleOrder(ValueError):pass


def rule_status(observation):
    team=observation['team'];enemy=enemy_of(team);positions=observation.get('positions',{})
    own_outpost=positions.get(f'{team}_outpost',{});enemy_outpost=positions.get(f'{enemy}_outpost',{})
    enemy_base=observation['bases'][enemy]
    return {'time':observation['time'],
            'enemy_base_protected':enemy_outpost.get('control')=='full' and enemy_outpost.get('controller')==enemy,
            'enemy_base_armor_open':bool(enemy_base.get('armor_open')),
            'we_control_enemy_outpost_position':enemy_outpost.get('controller')==team,
            'enemy_fortress_available':observation['time']>=180 and enemy_outpost.get('controller')==team,
            'own_fortress_active':own_outpost.get('controller')!=team,
            'own_base_protected':own_outpost.get('control')=='full' and own_outpost.get('controller')==team,
            'own_base_armor_open':bool(observation['bases'][team].get('armor_open')),
            'enemy_outpost_alive':observation['outposts'][enemy]['hp']>0 and observation['outposts'][enemy].get('owner',enemy)==enemy,
            'launcher_unlock_s':observation['teams'][team].get('launcher_unlock_s')}


def team_visible(observation):
    seen=set()
    for view in observation.get('vision',{}).values():seen|=set(view.get('enemies',[]))
    return [e for e in observation['enemies'] if e['id'] in seen]


def feasibility(controller,observation,robot_id,objective,target=None):
    """None when the objective can start now, else a short reason the planner can act on."""
    status=rule_status(observation);team=observation['team']
    robot=next((r for r in observation['robots'] if r['id']==robot_id),None)
    if robot is None:return 'unknown_robot'
    if objective=='continue':return None
    if robot['role']=='aerial':
        if objective not in AERIAL_OBJECTIVES:return 'not_an_aerial_objective'
        if objective=='drone_recall':return None
        drones=robot.get('drones') or []
        if not any(d['battery_wh']>.45 and d['integrity']>0 for d in drones):return 'no_usable_aerial_robots'
        if objective=='drone_strike_outpost' and not status['enemy_outpost_alive']:return 'enemy_outpost_destroyed'
        if objective=='drone_strike_base' and status['enemy_base_protected']:return 'enemy_base_protected_by_full_outpost_control'
        return None
    if objective not in GROUND_OBJECTIVES:return 'not_a_ground_objective'
    if not robot['alive'] or not robot['powered']:return 'robot_down_wait_for_respawn'
    if objective in WEAPON_OBJECTIVES:
        if robot['role']=='heavy' and robot.get('launcher') not in ('fitted','installed'):return f"launcher_{robot.get('launcher','locked')}"
        if robot['ammo']<=0:return 'no_ammo_resupply_first'
    if objective=='attack_outpost' and not status['enemy_outpost_alive']:return 'enemy_outpost_destroyed'
    if objective=='push_base':
        if status['enemy_base_protected']:return 'enemy_base_protected_capture_enemy_central_or_destroy_outpost'
        if not status['enemy_base_armor_open']:return 'enemy_base_armor_closed_occupy_enemy_fortress_first'
    if objective=='occupy_enemy_fortress':
        if status['enemy_base_armor_open']:return 'enemy_base_armor_already_open'
        if observation['time']<180:return 'enemy_fortress_inactive_before_180s'
        if not status['we_control_enemy_outpost_position']:return 'requires_enemy_outpost_position_capture_enemy_central_first'
    if objective=='hold_own_fortress' and not status['own_fortress_active']:return 'own_fortress_inactive_while_we_control_our_outpost_position'
    if objective in ('pursue_enemy','engage_enemy'):
        if target is None:return 'target_required'
        if target not in {e['id'] for e in team_visible(observation)} and target not in controller_memory(controller,observation):return 'target_not_seen_recently'
    if objective=='install_launcher':
        if robot['role']!='heavy':return 'heavy_only'
        if robot.get('launcher') not in ('unlocked','installing','fitted'):return f"launcher_{robot.get('launcher','locked')}"
    if objective=='resupply':
        from rmuc_arena import sdk
        caliber='42mm' if robot['role'] in ('heavy','hero') else '17mm';rule=sdk.rules(observation)['economy']['ammo'][caliber]
        ledger=observation['teams'][team].get('purchased_'+caliber[:2],0)
        if ledger>=rule['team_purchase_cap'] and robot['hp']>=robot['max_hp']*.9:return 'team_ammo_cap_reached'
        # The referee pays the 800 starting coins at 1.1 s; only later shortages block resupply.
        if observation['time']>2. and observation['teams'][team]['resources']<rule['local_cost'] and robot['hp']>=robot['max_hp']*.9:return 'insufficient_coins'
    if objective=='escort':
        if target is None:return 'target_required'
        ally=next((r for r in observation['robots'] if r['id']==target),None)
        if ally is None or ally['id']==robot_id or ally['role']=='aerial' or not ally['alive']:return 'escort_needs_living_ground_ally'
    return None


def controller_memory(controller,observation):
    seen=set()
    for ex in controller.executors.values():
        seen|={k for k,v in ex.memory.items() if observation['time']-v['seen_at']<=5.}
    return seen


def compile_order(controller,observation,robot_id,order):
    """Validated tactical command (ground) or aerial order name; raises InfeasibleOrder."""
    objective=order['objective'];target=order.get('target');where=order.get('place')
    reason=feasibility(controller,observation,robot_id,objective,target)
    if reason:raise InfeasibleOrder(reason)
    team=observation['team'];enemy=enemy_of(team);arena=controller.arena
    robot=next(r for r in observation['robots'] if r['id']==robot_id)
    if robot['role']=='aerial':return {'drone_strike_outpost':'strike_outpost','drone_strike_base':'strike_base','drone_recall':'recall'}[objective]
    heavy=robot['role']=='heavy';rng=4.5 if heavy else 3.
    identity=f"{robot_id}-{objective}-{observation['tick']}"
    base={'id':identity[:64],'low_hp_fraction':.12}
    if objective=='goto':
        if where not in PLACES:raise InfeasibleOrder('place_required')
        point=place(team,where);return {**base,'skill':'navigate','point':point,'radius_m':.5,'timeout_s':float(min(90.,15.+3.*math.dist(point,xy(robot))))}
    if objective=='hold_position':
        center=place(team,where) if where in PLACES else [float(robot['x']),float(robot['y'])]
        return {**base,'skill':'defend_region','center':center,'region_radius_m':1.5,'leash_radius_m':3.,'preferred_range_m':rng,'timeout_s':30.}
    if objective=='attack_outpost':return {**base,'skill':'attack_structure','target_id':f'{enemy}_outpost','preferred_range_m':rng,'timeout_s':60.}
    if objective=='push_base':return {**base,'skill':'attack_structure','target_id':f'{enemy}_base','preferred_range_m':rng+.6,'timeout_s':60.}
    if objective=='defend_base':
        front=place(team,'own_base_front')
        threats=sorted(team_visible(observation),key=lambda e:math.dist(xy(e),front))
        return {**base,'skill':'defend_region','center':front,'region_radius_m':3.5,'leash_radius_m':5.,'preferred_range_m':rng,
                'priority_ids':[e['id'] for e in threats if math.dist(xy(e),front)<9.][:4],'timeout_s':30.,'low_hp_fraction':.05}
    cards={'capture_enemy_central':f'{enemy}_central','hold_own_central':f'{team}_central','occupy_enemy_fortress':f'{enemy}_fortress','hold_own_fortress':f'{team}_fortress'}
    if objective in cards:
        return {**base,'skill':'defend_region','center':zone_center(arena,cards[objective]),'region_radius_m':1.5,'leash_radius_m':3.,'preferred_range_m':rng,
                'timeout_s':40. if objective!='occupy_enemy_fortress' else 30.}
    if objective in ('pursue_enemy','engage_enemy'):
        return {**base,'skill':'pursue_target' if objective=='pursue_enemy' else 'engage_target','target_id':target,'preferred_range_m':rng,
                'leash_radius_m':14. if objective=='pursue_enemy' else 8.,'target_lost_timeout_s':4.,'timeout_s':35.}
    if objective=='resupply':
        return {**base,'skill':'resupply','ammo_target':12 if heavy else 150,'hp_target_fraction':.6,'budget_limit':400.,'timeout_s':45.}
    if objective=='retreat':return {'id':base['id'],'skill':'retreat','radius_m':.7,'timeout_s':45.}
    if objective=='escort':return {**base,'skill':'escort','target_id':target,'follow_distance_m':1.4,'required_coverage':.6,'escort_duration_s':20.,'leash_radius_m':10.,'timeout_s':30.}
    if objective=='patrol':
        route=PATROLS.get(where or 'base_perimeter',PATROLS['base_perimeter'])
        points=[[x,y] if team=='red' else [28.-x,15.-y] for x,y in route]
        return {**base,'skill':'patrol','waypoints':points,'patrol_loops':2,'dwell_s':.5,'radius_m':.5,'engage_on_contact':True,'timeout_s':60.}
    if objective=='install_launcher':return {'id':base['id'],'skill':'install_launcher','timeout_s':90.}
    raise InfeasibleOrder('unsupported_objective')


def objective_of(command):
    """Inverse summary of a running command for scene/feedback text."""
    if command is None:return None
    if isinstance(command,str):return command
    return {'skill':command.get('skill'),**{k:command[k] for k in ('target_id','center','point') if k in command}}


def area_name(team,point):
    best=min(PLACES,key=lambda n:math.dist(place(team,n),point));return best if math.dist(place(team,best),point)<4. else 'field'


def team_scene(controller,observation,events=(),history=()):
    """Compact, public, team-relative state for the commander."""
    team=observation['team'];enemy=enemy_of(team);status=rule_status(observation);feedback=controller.feedback(observation)
    robots={}
    for r in observation['robots']:
        identity=r['id'];entry={'role':r['role'],'model':r.get('model_slot'),'alive':r['alive']}
        if r['role']=='aerial':
            entry['drones']=[{k:d[k] for k in ('state','battery_wh','integrity','strikes') if k in d} for d in r.get('drones',[])]
        else:
            ex=controller.executors.get(identity)
            entry.update(chassis=ex.profile.drive if ex else None,hp=round(r['hp']),max_hp=r['max_hp'],ammo=r['ammo'],heat=round(r.get('heat',0)),
                         area=area_name(team,xy(r)),xy=[round(r['x'],1),round(r['y'],1)])
            if r['role']=='heavy':entry['launcher']=r.get('launcher')
            if r['role']=='sentry':entry['posture']=r.get('posture')
        fb=feedback.get(identity,{})
        entry['order']={k:(round(fb[k],1) if isinstance(fb[k],float) else fb[k]) for k in ('skill','status','reason','phase','order','distance_m','strikes_this_order','source') if fb.get(k) is not None}
        objectives=AERIAL_OBJECTIVES if r['role']=='aerial' else GROUND_OBJECTIVES
        allowed=[];blocked={}
        for name in objectives:
            why=feasibility(controller,observation,identity,name)
            if name in ('pursue_enemy','engage_enemy','escort'):
                # Targeted objectives are offered when at least one legal target exists.
                options=_targets(controller,observation,name,identity)
                why=None if any(feasibility(controller,observation,identity,name,t) is None for t in options) else (why if why!='target_required' else 'no_target_seen')
            if why is None:allowed.append(name)
            else:blocked[name]=why
        entry['allowed']=allowed;entry['blocked']=blocked
        robots[identity]=entry
    enemies=[{'id':e['id'],'role':e.get('role'),'hp':round(e.get('hp',0)),'area':area_name(team,xy(e)),'xy':[round(e['x'],1),round(e['y'],1)]} for e in team_visible(observation) if e['alive']]
    memory={}
    for ex in controller.executors.values():
        for k,v in ex.memory.items():
            age=observation['time']-v['seen_at']
            if 0<age<=10 and k not in {e['id'] for e in enemies}:memory[k]={'id':k,'role':v['value'].get('role'),'age_s':round(age,1),'xy':[round(v['value']['x'],1),round(v['value']['y'],1)]}
    teams=observation['teams']
    return {'team':team,'tick':observation['tick'],'time_s':round(observation['time'],1),'remaining_s':round(420-observation['time'],1),
            'score':{t:{k:round(teams[t].get(k,0),1) for k in ('control','attack_damage','kills')} for t in (team,enemy)},
            'structures':{f'{t}_{k}':{'hp':round(observation[k+'s'][t]['hp']),**({'owner':observation[k+'s'][t].get('owner',t)} if k=='outpost' else {'armor_open':bool(observation[k+'s'][t].get('armor_open'))})}
                          for t in (team,enemy) for k in ('outpost','base')},
            'positions':{k:{'control':v.get('control'),'controller':v.get('controller')} for k,v in observation.get('positions',{}).items()},
            'rules':status,'economy':{k:teams[team].get(k) for k in ('resources','purchased_17','purchased_42')},
            'robots':robots,'visible_enemies':enemies,'recent_enemies':list(memory.values()),'events':list(events)[-12:],'recent_decisions':list(history)[-6:]}


def _targets(controller,observation,objective,robot_id):
    if objective=='escort':return [r['id'] for r in observation['robots'] if r['id']!=robot_id and r['role'] not in ('aerial',) and r['alive']]
    return sorted({e['id'] for e in team_visible(observation)}|controller_memory(controller,observation))


class Doctrine:
    """Deterministic team commander: LLM fallback, LLM-free baseline and declared opponent.

    Roles: slot 4 anchors the enemy central card (control degree ranks first at timeout),
    slot 3 takes the outpost then the fortress/base chain, the sentry keeps our central card
    and hunts nearby heavies, the heavy holds a card until its launcher is fitted.
    """
    def __init__(self):
        self.last={}

    def decide(self,controller,observation):
        team=observation['team'];enemy=enemy_of(team);status=rule_status(observation);orders={}
        feedback=controller.feedback(observation);front=place(team,'own_base_front')
        visible=[e for e in team_visible(observation) if e['alive']]
        invaders=[e for e in visible if math.dist(xy(e),front)<6.5 and e.get('role')!='aerial']
        enemy_heavy=next((e for e in visible if e.get('role') in ('heavy','hero')),None)
        own_central=zone_center(controller.arena,f'{team}_central')
        def holder(center):
            return next((k for k,ex in controller.executors.items() if ex.status=='running' and ex.command and ex.command.skill=='defend_region'
                         and ex.command.center and math.dist(ex.command.center,center)<.1),None)
        defenders=0
        ground=[r for r in observation['robots'] if r['role']!='aerial']
        for r in sorted(ground,key=lambda r:math.dist(xy(r),front)):
            identity=r['id']
            if not r['alive']:continue
            fb=feedback.get(identity,{});running=fb.get('status')=='running';current=fb.get('skill')
            def want(objective,target=None,where=None):
                if feasibility(controller,observation,identity,objective,target) is None:return {'objective':objective,'target':target,'place':where}
                return None
            def first(*options):return next((o for o in options if o),None)
            hp=r['hp']/max(1,r['max_hp']);armed=r['ammo']>0 and (r['role']!='heavy' or r.get('launcher') in ('fitted','installed'))
            near_heavy=lambda radius:enemy_heavy is not None and math.dist(xy(enemy_heavy),xy(r))<radius
            central_free=holder(own_central) in (None,identity)
            if hp<.3 and current not in ('resupply','retreat'):choice=first(want('resupply'),want('retreat'))
            elif invaders and defenders<2 and armed and math.dist(xy(r),front)<12.:choice=want('defend_base');defenders+=1
            elif r['role']=='heavy':
                if r.get('launcher') in ('unlocked','installing','fitted'):choice=want('install_launcher')
                elif r.get('launcher')=='installed':
                    choice=first(want('resupply') if r['ammo']<=0 else None,want('pursue_enemy',enemy_heavy['id']) if near_heavy(7) else None,
                                 want('push_base'),want('attack_outpost'),want('occupy_enemy_fortress'),want('capture_enemy_central'))
                else:choice=first(want('hold_own_central') if central_free else None,want('capture_enemy_central'))
            elif not armed:choice=first(want('resupply'),want('hold_own_central') if central_free else None,want('capture_enemy_central'))
            elif r['role']=='sentry':
                choice=first(want('pursue_enemy',enemy_heavy['id']) if near_heavy(8) else None,
                             want('hold_own_fortress') if status['own_fortress_active'] else None,
                             want('hold_own_central') if central_free else None,want('attack_outpost'),want('push_base'),want('capture_enemy_central'))
            elif r['slot']==4:
                choice=first(want('pursue_enemy',enemy_heavy['id']) if near_heavy(4) else None,want('capture_enemy_central'),want('attack_outpost'))
            else:
                choice=first(want('pursue_enemy',enemy_heavy['id']) if near_heavy(5) else None,want('push_base'),want('occupy_enemy_fortress'),
                             want('attack_outpost'),want('capture_enemy_central'),want('hold_own_central') if central_free else None)
            if choice is None:continue
            key=(choice['objective'],choice.get('target'))
            if running and self.last.get(identity)==key:continue
            orders[identity]=choice;self.last[identity]=key
        aerial=controller.aerial_id()
        if aerial:
            fb=feedback.get(aerial,{});choice=None
            if fb.get('status')!='running':
                if status['enemy_outpost_alive'] and feasibility(controller,observation,aerial,'drone_strike_outpost') is None:choice='drone_strike_outpost'
                elif status['enemy_base_armor_open'] and feasibility(controller,observation,aerial,'drone_strike_base') is None:choice='drone_strike_base'
            if choice and self.last.get(aerial)!=(choice,None):orders[aerial]={'objective':choice,'target':None,'place':None};self.last[aerial]=(choice,None)
        return orders


def apply_orders(controller,observation,orders,source):
    """Compile and submit orders atomically per robot; returns per-robot results."""
    results={}
    for robot_id,order in orders.items():
        if order.get('objective','continue')=='continue':results[robot_id]={'action':'continued'};continue
        try:
            command=compile_order(controller,observation,robot_id,order)
            current=controller.orders.get(robot_id)
            if isinstance(command,dict) and isinstance(current,dict) and controller.executors[robot_id].status=='running':
                same=lambda c:{k:v for k,v in c.items() if k not in ('id','priority_ids')}
                if same(command)==same(current):results[robot_id]={'action':'same_goal_preserved'};continue
            feedback=controller.submit(robot_id,command,observation,source)
            results[robot_id]={'action':'commanded','objective':order['objective'],'command':command if isinstance(command,str) else {k:v for k,v in command.items() if k!='id'}}
        except (InfeasibleOrder,ValueError) as error:
            results[robot_id]={'action':'rejected','objective':order.get('objective'),'reason':str(error)[:160]}
    return results
