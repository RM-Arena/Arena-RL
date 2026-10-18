import copy
import os
from pathlib import Path
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from arena_rl.bridge import configure
from arena_rl.encoding import ACTION_SPACE,OBS_SPACE,decode,encode
from arena_rl.envs import SingleRobotEnv,SkillBatch,SkillVecEnv
from arena_rl.tasks import sample,robot

ROOT=configure()


def test_gymnasium_single_robot_contract():
    env=SingleRobotEnv(task='move',arena_root=ROOT)
    try:check_env(env,skip_render_check=True)
    finally:env.close()


def test_parallel_robot_control_isolation_and_seeded_reset():
    env=SkillBatch(2,task='move',arena_root=ROOT)
    try:
        before=env.reset(seeds=[51,51]);assert np.array_equal(before[0],before[1])
        pids=[s['info']['simulator_pid'] for s in env.states];assert len(set(pids))==2
        commands=decode(env.tasks[0],env.states[0],np.array([1,1,1,0,1,0]))
        assert set(commands['red'])=={'3'} and commands['blue']=={}
        env.step([[1,1,1,0,1,0],[0,1,1,0,1,0]])
        assert robot(env.states[0])['x']>robot(env.states[1])['x']
        again=env.reset([0],seeds=[51]);assert np.array_equal(before[0],again[0])
        assert env.states[0]['info']['simulator_pid']==pids[0]
    finally:env.close()


def test_policy_inputs_do_not_include_hidden_blue_state():
    env=SkillBatch(1,task='move',arena_root=ROOT)
    try:
        env.reset(seeds=[51]);state=env.states[0];original=encode(env.tasks[0],state)
        altered=copy.deepcopy(state);altered['observations']['blue']['robots'][0]['x']=7
        assert np.array_equal(original,encode(env.tasks[0],altered))
        assert OBS_SPACE.contains(original)
    finally:env.close()


def test_invalid_action_is_rejected_before_advancing():
    env=SingleRobotEnv(task='move',arena_root=ROOT)
    try:
        env.reset(seed=1)
        with pytest.raises(ValueError):env.step([99,1,1,0,1,0])
        assert env.batch.states[0]['info']['tick']==0
    finally:env.close()


def test_sb3_autoreset_preserves_terminal_observation_and_timeout():
    env=SkillVecEnv(1,task='move',arena_root=ROOT)
    try:
        env.reset();old=env.batch.tasks[0].seed;env.batch.tasks[0].max_steps=1
        obs,rewards,dones,infos=env.step(np.array([[0,1,1,0,1,0]]))
        assert dones[0] and infos[0]['TimeLimit.truncated']
        assert infos[0]['terminal_observation'].shape==OBS_SPACE.shape
        assert env.batch.tasks[0].seed!=old
    finally:env.close()


@pytest.mark.parametrize('task',['move','ramp','outpost','defend'])
def test_scripted_feasibility_and_real_damage(task):
    from arena_rl.learning import baseline_actions
    env=SkillBatch(1,task=task,seed=31,arena_root=ROOT,record=True)
    try:
        env.reset();rng=np.random.default_rng(1)
        for _ in range(env.tasks[0].max_steps):
            _,_,terms,truncs,infos=env.step(baseline_actions(env,'heuristic',rng))
            if terms[0] or truncs[0]:break
        assert infos[0]['is_success'],infos
        if task=='ramp':assert infos[0]['ramp_crossed']
        if task=='outpost':assert infos[0]['outpost_damage']>=40
        if task=='defend':assert infos[0]['attack_damage']>=40
    finally:env.close()


def test_replay_contains_scenario_and_verifies(tmp_path):
    from rmuc_arena.replays import verify_replay
    env=SkillBatch(1,task='outpost',arena_root=ROOT,record=True)
    try:
        env.reset(seeds=[73]);env.step([[0,1,1,1,1,0]])
        path=tmp_path/'episode.json';env.engine.save_replay([path])
        result=verify_replay(path)
        assert result['verified_ticks']==2
    finally:env.close()
