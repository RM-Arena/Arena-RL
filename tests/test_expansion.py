"""Physical readiness, real-card holds and nonholonomic route regressions."""
import copy
import math
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.executor import TacticalExecutor,own

ROOT=configure()


@pytest.fixture
def observation():
    from rmuc_arena.env import ArenaEnv
    with ArenaEnv(season='2027',lineups={'red':[{'slot':3,'model':305}],'blue':[{'slot':3,'model':3}]}) as env:
        return env.reset(3310,scenario={'robots':{'red-3':{'x':10.8,'y':8.4,'heading':-math.pi/2,'ammo':0}}})['observations']['red']


def test_control_hold_aims_inside_card_and_does_not_require_ammo(observation):
    from arena_rl.navigation import safe_hold_point
    ex=TacticalExecutor();ex.submit({'id':'occupy','skill':'defend_region','center':[10.8,7.5],'timeout_s':10.},observation)
    assert ex.status=='running' and ex.plan['arrival_radius']==.15
    assert safe_hold_point(own(observation),[10.8,7.5],3.,ex.navigation_arena)==(10.8,7.5)
    np.testing.assert_allclose(ex.plan['goal'],[10.8,7.5])
    assert ex.extra['control_contact'] is False
    late=copy.deepcopy(observation);late['tick']+=200;late['time']+=10.
    ex.update(late)
    assert ex.status=='failed' and ex.reason=='control_card_not_reached'


def test_card_center_alone_never_creates_referee_contact(observation):
    ex=TacticalExecutor();ex.submit({'id':'hold','skill':'defend_region','center':[17.2,7.5]},observation)
    moved=copy.deepcopy(observation);moved['tick']+=2;moved['time']+=.1;own(moved).update(x=17.2,y=7.5)
    ex.update(moved)
    assert not ex.extra['control_contact'] and not ex.skill_state['contact_seen']


def test_road_heading_survives_second_clearance_pass(observation):
    from arena_rl.tactical_encoding import decode
    ex=TacticalExecutor();ex.submit({'id':'road','skill':'navigate','point':[8.2,13.7]},observation)
    ex.route_state['terrain_crossing']=True
    ex.plan.update(hint=np.array([1.,0.]),held_heading=0.,distance=2.)
    own(observation)['heading']=.6
    low=decode(ex,observation,[1,3,1,1,0,1,0])['3'];high=decode(ex,observation,[1,3,1,1,1,1,0])['3']
    assert low['power']==100. and high['power']==200.
    assert high['heading']==0. and high['move']==[1.,0.] and high['leg_height']==.245


def test_long_flight_reapproach_uses_dedicated_learned_flight_controller(observation):
    ex=TacticalExecutor();ex.model_id=305
    from arena_rl.commands import Command
    ex.command=Command('retry','fly_slope');ex.status='running';ex.plan={'phase':'flight_approach','distance':3.}
    assert ex.execution_family=='flight'


def test_objective_training_uses_real_ammunition_and_no_pose_shortcuts():
    from arena_rl.tactical_envs import sample_episode
    for family in ('combat','defense'):
        for seed in range(50):
            episode=sample_episode(family,seed,305,'objectives')
            assert episode.scenario==sample_episode(family,seed,305,'objectives').scenario
            for robot in episode.scenario['robots'].values():
                assert set(robot)<= {'x','y','heading','aim','ammo'}
            assert len([k for k in episode.behavior if k.startswith('red-')])==0
