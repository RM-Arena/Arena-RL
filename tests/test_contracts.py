import numpy as np
import pytest
from arena_rl.tasks import sample
from arena_rl.encoding import ACTION_SPACE,OBS_SPACE


def test_curricula_seeded_and_single_robot_ownership():
    for name in ('move','ramp','outpost','defend','mixed'):
        a,b=sample(name,17),sample(name,17)
        assert a.name==b.name and np.array_equal(a.goal,b.goal) and a.scenario==b.scenario
        assert set(a.scenario['robots'])=={'red-3','blue-3'}
        assert a.scenario['robots']['red-3']['ammo']==200
        assert a.max_steps>0


def test_schema_dimensions():
    assert ACTION_SPACE.nvec.tolist()==[9,3,3,2,3,2]
    assert OBS_SPACE.shape==(33,)


def test_goal_and_spawn_are_distinct_for_navigation():
    for task in ('move','ramp'):
        for seed in range(100):
            t=sample(task,seed);spawn=t.scenario['robots']['red-3']
            assert np.linalg.norm(t.goal-[spawn['x'],spawn['y']])>.35


def test_unknown_task_rejected():
    with pytest.raises(ValueError):sample('unknown',17)


def test_tactical_command_roundtrip_and_strict_validation():
    from arena_rl.commands import Command,capabilities
    command=Command.parse({'id':'pursue','skill':'pursue_target','target_id':'blue-1'})
    assert Command.parse(command.json())==command
    assert 'pursue_target' in capabilities()['skills']
    for invalid in ({'id':'x','skill':'pursue_target'},
                    {'id':'x','skill':'navigate','point':[True,5]},
                    {'id':'x','skill':'navigate','point':[float('nan'),5]},
                    {'id':'x','skill':'navigate','point':[5,5],'timeout_s':True},
                    {'id':'x','skill':'navigate','point':[5,5],'debug':True}):
        with pytest.raises(ValueError):Command.parse(invalid)


@pytest.mark.parametrize('value',[
    {'id':'supply','skill':'resupply','ammo_target':100,'budget_limit':150},
    {'id':'protect','skill':'escort','target_id':'red-4','point':[7.5,5.3]},
    {'id':'route','skill':'patrol','waypoints':[[4,5],[6,5]],'patrol_loops':2},
    {'id':'jump','skill':'fly_slope','lane_team':'blue','flight_direction':'reverse'},
])
def test_extended_command_roundtrip(value):
    from arena_rl.commands import Command
    c=Command.parse(value)
    assert Command.parse(c.json())==c
    assert c.policy_family in ('support','flight')


@pytest.mark.parametrize('value',[
    {'id':'x','skill':'navigate','point':None},
    {'id':'x','skill':'defend_region','center':None},
    {'id':'x','skill':'patrol','waypoints':[[1,2]]},
    {'id':'x','skill':'patrol','waypoints':[[1,2],[2,3]],'patrol_loops':True},
    {'id':'x','skill':'escort','target_id':'red_base'},
    {'id':'x','skill':'resupply','ammo_target':1.5},
    {'id':'x','skill':'resupply','engage_on_contact':1},
    {'id':'x','skill':'fly_slope','lane_team':'green'},
    {'id':'x','skill':'fly_slope','flight_direction':'sideways'},
])
def test_extended_command_validation(value):
    from arena_rl.commands import Command
    with pytest.raises(ValueError):Command.parse(value)


def test_old_and_extended_schemas_remain_separate():
    from arena_rl.tactical_encoding import FEATURES as old,OBS_SPACE as old_space
    from arena_rl.extended_encoding import FEATURES,OBS_SPACE
    from arena_rl.commands import capabilities
    assert len(old)==52 and old_space.shape==(52,)
    assert FEATURES[:52]==old and OBS_SPACE.shape==(74,)
    assert len(capabilities()['command_json_schema']['oneOf'])==12
    assert capabilities()['skill_seasons']['fly_slope']==['2026']
