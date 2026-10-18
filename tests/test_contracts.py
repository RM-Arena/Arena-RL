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
