import copy
import math
import numpy as np
import pytest
from arena_rl.commands import Command,capabilities
from arena_rl.bridge import configure
from arena_rl.executor import TacticalExecutor
from arena_rl.tactical_encoding import decode,encode,OBS_SPACE
from arena_rl.tactical_envs import LINEUPS

configure()


@pytest.fixture
def observation():
    from rmuc_arena.env import ArenaEnv
    with ArenaEnv(season='2027',lineups=LINEUPS) as env:
        value=env.reset(11,scenario={'robots':{'red-3':{'x':4.5,'y':5.15,'heading':0.,'aim':0.,'ammo':200},
            'blue-1':{'x':8.,'y':5.15,'heading':math.pi,'ammo':0},
            'blue-3':{'x':6.,'y':5.65,'heading':math.pi,'ammo':0}}})['observations']['red']
    assert {e['id'] for e in value['enemies']} >= {'blue-1','blue-3'}
    return value


def tick(obs,seconds=.1):
    value=copy.deepcopy(obs);value['tick']+=int(seconds/.05);value['time']+=seconds
    return value


def test_target_lock_does_not_switch_to_nearer_enemy(observation):
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1'},observation)
    assert executor.plan['target']['id']=='blue-1'
    action=decode(executor,observation,np.array([0,0,1,1,1,1,0]))
    assert set(action)=={'3'} and action['3']['fire']
    assert executor.target_id=='blue-1' and OBS_SPACE.contains(encode(executor,observation))
    newer=tick(observation)
    for enemy in newer['enemies']:
        if enemy['id']=='blue-3':enemy['hp']=0
    executor.update(newer);assert executor.damage==0


def test_occlusion_uses_last_seen_and_never_fires(observation):
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'pursue_target','target_id':'blue-1','target_lost_timeout_s':1.},observation)
    hidden=tick(observation,.5);hidden['enemies']=[e for e in hidden['enemies'] if e['id']!='blue-1']
    executor.update(hidden);assert executor.status=='running' and executor.plan['age']==.5
    assert not decode(executor,hidden,np.array([1,1,1,1,1,1,0]))['3']['fire']
    lost=tick(hidden,1.);executor.update(lost)
    assert executor.status=='failed' and executor.reason=='target_lost'


def test_invalid_or_stale_replacement_does_not_cancel_valid_command(observation):
    executor=TacticalExecutor();first={'id':'move','skill':'navigate','point':[7.8,5.15]}
    executor.submit(first,observation);revision=executor.revision
    with pytest.raises(ValueError):executor.submit({'id':'new','skill':'navigate','point':[999,2]},observation)
    later=tick(observation,4.);executor.update(later)
    with pytest.raises(ValueError,match='Stale'):executor.submit({'id':'new','skill':'navigate','point':[8,5.15],'basis_tick':0},later)
    assert executor.command.id=='move' and executor.revision==revision


def test_idempotent_commands_and_explicit_replacement(observation):
    executor=TacticalExecutor();first={'id':'move','skill':'navigate','point':[7.8,5.15]}
    executor.submit(first,observation);started=executor.started
    response=executor.submit(first,tick(observation));assert response['duplicate'] and executor.started==started
    with pytest.raises(ValueError):executor.submit({**first,'point':[8,5.15]},observation)
    executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1'},observation)
    assert executor.revision==2 and executor.events[-1]['reason']=='replaced_by_new_command'
    assert executor.events[-1]['command_id']=='move'


def test_defense_priorities_and_leash(observation):
    executor=TacticalExecutor();executor.submit({'id':'guard','skill':'defend_region','center':[4.5,5.15],
                                                'region_radius_m':1.,'priority_ids':['blue-1']},observation)
    assert executor.target_id=='blue-1'
    assert np.linalg.norm(executor.plan['goal']-[4.5,5.15])<=1.000001
    assert executor.plan['target']['id']!='blue-3'


def test_friendly_and_protected_structure_rejected(observation):
    executor=TacticalExecutor()
    with pytest.raises(ValueError):executor.submit({'id':'friend','skill':'engage_target','target_id':'red-3'},observation)
    executor.submit({'id':'base','skill':'attack_structure','target_id':'blue_base'},observation)
    assert executor.reason=='structure_protected' and executor.status=='failed'


def test_failure_cannot_be_overwritten_by_target_success(observation):
    bad=copy.deepcopy(observation);bad['robots'][0]['hp']=1
    for enemy in bad['enemies']:
        if enemy['id']=='blue-1':enemy['hp']=0;enemy['alive']=False
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1'},bad)
    assert executor.reason=='low_hp' and executor.status=='failed'


def test_damage_training_goal_is_not_success_at_defense_timeout(observation):
    executor=TacticalExecutor();executor.submit({'id':'guard','skill':'defend_region','center':[4.5,5.15],
                                                'damage_goal':60.,'timeout_s':1.},observation)
    executor.update(tick(observation,1.))
    assert executor.status=='failed' and executor.reason=='timeout'


def test_repeated_observation_does_not_duplicate_damage(observation):
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1'},observation)
    value=tick(observation)
    for e in value['enemies']:
        if e['id']=='blue-1':e['hp']-=20
    executor.update(value);executor.update(value)
    assert executor.damage==20


def test_teammate_camera_is_not_used_as_the_controlled_robot_camera(observation):
    other=copy.deepcopy(observation)
    other['vision']['3']['enemies']=[]
    other['vision']['4']={'enemies':['blue-1','blue-3']}
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1'},other)
    assert not executor.plan['visible'] and not executor.memory
    action=decode(executor,other,np.array([0,0,1,1,1,1,0]))
    assert not action['3']['fire']


def test_pursuit_stops_at_its_command_leash(observation):
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'pursue_target','target_id':'blue-1','leash_radius_m':1.},observation)
    moved=tick(observation);moved['robots'][0]['x']+=1.5
    executor.update(moved)
    assert executor.status=='failed' and executor.reason=='leash_exceeded'
    assert decode(executor,moved,np.array([1,3,1,1,1,1,1]))['3']['move']==[0.,0.]


def test_completed_feedback_keeps_execution_duration(observation):
    executor=TacticalExecutor();executor.submit({'id':'move','skill':'navigate','point':[7.8,5.15]},observation)
    arrived=tick(observation,1.);arrived['robots'][0]['x']=7.8;arrived['robots'][0]['vx']=0;arrived['robots'][0]['vy']=0
    executor.update(arrived);assert executor.status=='succeeded'
    elapsed=executor.feedback()['elapsed_s'];executor.update(tick(arrived,2.))
    assert executor.feedback()['elapsed_s']==elapsed


def test_defense_center_can_be_the_solid_base_center(observation):
    from arena_rl.navigation import safe_hold_point
    executor=TacticalExecutor();base=observation['bases']['red']
    obs=copy.deepcopy(observation);obs['enemies']=[];obs['vision']['3']['enemies']=[]
    executor.submit({'id':'guard','skill':'defend_region','center':[base['x'],base['y']],'region_radius_m':3.},obs)
    assert executor.status=='running'
    point=executor.plan['goal'];assert np.linalg.norm(point-[base['x'],base['y']])<=3
    assert np.linalg.norm(point-[base['x'],base['y']])>base['radius']+.3


def test_out_of_order_llm_reply_cannot_replace_a_newer_decision(observation):
    executor=TacticalExecutor()
    newer=tick(observation,.5)
    executor.submit({'id':'newer','skill':'navigate','point':[7.8,5.15],'basis_tick':newer['tick']},newer)
    with pytest.raises(ValueError,match='out-of-order'):
        executor.submit({'id':'older','skill':'navigate','point':[7.5,5.15],'basis_tick':observation['tick']},tick(newer,.1))
    assert executor.command.id=='newer'
    retry=executor.submit({'id':'newer','skill':'navigate','point':[7.8,5.15],'basis_tick':newer['tick']},tick(newer,4.))
    assert retry['duplicate'] and executor.revision==1


def test_successive_planner_updates_use_their_observation_order(observation):
    executor=TacticalExecutor()
    for k in range(3):
        seen=tick(observation,.1*k)
        executor.submit({'id':f'order-{k}','skill':'navigate','point':[7.8,5.15],'basis_tick':seen['tick']},tick(seen,.5))
    assert executor.revision==3 and executor.command.id=='order-2'


def test_far_attack_uses_navigation_policy_before_local_combat(observation):
    executor=TacticalExecutor();executor.submit({'id':'post','skill':'attack_structure','target_id':'blue_outpost'},observation)
    assert executor.execution_family=='navigation'
    features=encode(executor,observation,policy_family=executor.execution_family)
    assert features[0]==1 and features[1]==0  # Navigation goal encoding during approach.
    action=decode(executor,observation,np.array([1,3,1,1,1,1,0]),policy_family=executor.execution_family)
    assert not action['3']['fire']


def test_last_ammunition_can_still_complete_the_damage_goal(observation):
    executor=TacticalExecutor();executor.submit({'id':'hero','skill':'engage_target','target_id':'blue-1','damage_goal':60.},observation)
    value=tick(observation);value['robots'][0]['ammo']=0
    for e in value['enemies']:
        if e['id']=='blue-1':e['hp']-=60
    executor.update(value)
    assert executor.status=='succeeded' and executor.reason=='damage_goal_reached'
