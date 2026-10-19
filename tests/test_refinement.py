"""Regressions for arrival control and physical fly-slope recovery."""
import copy
import math
from pathlib import Path
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.executor import TacticalExecutor,own
from arena_rl.tactical_envs import TacticalBatch,LINEUPS,sample_episode,encode_observation
from arena_rl.tactical_runtime import TacticalSession

ROOT=configure();POLICIES=Path(__file__).resolve().parents[1]/'checkpoints/serial305'


@pytest.fixture
def flight_observation():
    from rmuc_arena.env import ArenaEnv
    lineups={'red':[{'slot':3,'model':305}],'blue':[{'slot':3,'model':3}]}
    with ArenaEnv(season='2026',lineups=lineups) as env:
        return env.reset(2210,scenario={'robots':{'red-3':{'x':18.,'y':.6,'heading':math.pi,'ammo':0}}})['observations']['red']


def later(obs,dt=.1):
    result=copy.deepcopy(obs);result['tick']+=round(dt/.05);result['time']+=dt;return result


def test_airtime_is_observed_during_recovery_without_duplicate_credit(flight_observation):
    from rmuc_arena import sdk
    ex=TacticalExecutor();ex.submit({'id':'cross','skill':'fly_slope','lane_team':'blue','timeout_s':18.},flight_observation)
    ex.skill_state['launched']=True
    lane=sdk.fly_slope_lane(ex.arena,'blue');obs=later(flight_observation);r=own(obs)
    r.update(x=lane['crest'][0]-.2,y=lane['crest'][1],upright=.27,grounded=False)
    ex.update(obs)
    assert ex.status=='running' and ex.plan['phase']=='flight_air'
    assert ex.skill_state['air_s']==pytest.approx(.1)
    ex.update(obs);assert ex.skill_state['air_s']==pytest.approx(.1)
    landed=later(obs);own(landed)['grounded']=True;ex.update(landed)
    assert ex.plan['phase']=='recover'
    resumed=later(landed);own(resumed)['grounded']=False;ex.update(resumed)
    assert ex.skill_state['air_s']==pytest.approx(.2)
    assert ex.command.id=='cross'
    ex.update(later(obs,17.9));assert ex.status=='failed' and ex.reason=='timeout'


def test_lateral_alignment_can_steer_a_two_wheel_chassis(flight_observation):
    from arena_rl.extended_encoding import decode
    ex=TacticalExecutor();ex.submit({'id':'align','skill':'fly_slope','lane_team':'blue'},flight_observation)
    ex.plan.update(phase='flight_approach',hint=np.array([0.,1.]),distance=.2)
    action=decode(ex,flight_observation,np.array([1,0,1,1,1,1,0]))['3']
    assert abs(math.sin(action['heading']))>.99 and np.linalg.norm(action['move'])==0
    aligned=copy.deepcopy(flight_observation);own(aligned)['heading']=action['heading']
    assert np.linalg.norm(decode(ex,aligned,np.array([1,0,1,1,1,1,0]))['3']['move'])>0


def test_launch_waits_for_stable_entry_support(flight_observation):
    from rmuc_arena import sdk
    obs=copy.deepcopy(flight_observation);lane=sdk.fly_slope_lane(obs['arena'],'blue')
    r=own(obs);r.update(x=lane['lip'][0]+2.55,y=lane['lip'][1],heading=math.pi,upright=1.,grounded=True,vx=0.,vy=0.,angular_velocity=[0.,0.,0.])
    ex=TacticalExecutor();ex.submit({'id':'settle','skill':'fly_slope','lane_team':'blue'},obs)
    assert not ex.skill_state.get('launched')
    for _ in range(5):obs=later(obs);ex.update(obs)
    assert ex.skill_state['launched'] and ex.command.id=='settle'


@pytest.mark.parametrize("seed",[3000008,3000013,3000027])
def test_original_navigation_failure_finishes_with_original_budget(seed):
    lineups=copy.deepcopy(LINEUPS);lineups['red'][0]['model']=305
    session=TacticalSession(POLICIES,lineups=lineups)
    try:
        batch=session.batch;episode=sample_episode('navigation',seed,305)
        assert episode.command['timeout_s']==(28. if seed==3000013 else 75.)
        batch.reset(seeds=[3000013],episodes=[episode]);pid=batch.states[0]['info']['simulator_pid']
        for _ in range(round(episode.command['timeout_s']*10)):
            if batch.finished[0]:break
            session.advance()
        assert batch.executors[0].status=='succeeded'
        assert batch.executors[0].command.id==episode.command['id']
        assert batch.states[0]['info']['simulator_pid']==pid
    finally:session.close()


def test_original_flight_failures_cross_the_gap_and_verify_native_replay(tmp_path):
    from arena_rl.tactical_learning import load_tactical_policy
    policy,_=load_tactical_policy(POLICIES/'flight',ROOT)
    batch=TacticalBatch(4,family='flight',model_id=305,record=True)
    try:
        batch.reset(seeds=[6000011,6000016,6000020,6000021]);pids=[s['info']['simulator_pid'] for s in batch.states]
        for _ in range(180):
            ids=np.flatnonzero(~batch.finished).tolist()
            if not ids:break
            features=np.stack([encode_observation(batch.executors[i],batch.states[i]['observations']['red'],'flight') for i in ids])
            batch.step(policy.predict(features,deterministic=True)[0],env_ids=ids)
        from rmuc_arena.replays import verify_replay
        for i,ex in enumerate(batch.executors):
            assert ex.status=='succeeded' and ex.extra['airborne_s']+1e-8>=.1 and ex.extra['landing_hold_s']+1e-8>=.3
            assert batch.states[i]['info']['simulator_pid']==pids[i]
            path=tmp_path/f'flight-{i}.json';batch.engine.save_replay([path],env_ids=[i])
            assert verify_replay(path)['verified_ticks']==batch.states[i]['info']['tick']
    finally:batch.close()


def test_forward_entry_must_center_before_launch(flight_observation):
    from rmuc_arena import sdk
    obs=copy.deepcopy(flight_observation);lane=sdk.fly_slope_lane(obs['arena'],'blue')
    own(obs).update(x=lane['lip'][0]+2.55,y=lane['lip'][1]-.06,heading=math.pi,upright=1.,grounded=True,vx=0.,vy=0.,angular_velocity=[0.,0.,0.])
    ex=TacticalExecutor();ex.submit({'id':'off-center','skill':'fly_slope','lane_team':'blue'},obs)
    for _ in range(5):obs=later(obs);ex.update(obs)
    assert ex.plan['phase']=='flight_approach' and not ex.skill_state.get('launched')


def test_flight_control_is_invariant_to_unrelated_gimbal_and_economy(flight_observation):
    from arena_rl.flight_encoding import encode,OBS_SPACE
    ex=TacticalExecutor();ex.submit({'id':'physical','skill':'fly_slope','lane_team':'blue'},flight_observation)
    before=encode(ex,flight_observation);changed=copy.deepcopy(flight_observation)
    own(changed).update(aim=2.1,pitch=.4,ammo=123,heat=12.,hp=75.)
    changed['teams']['red']['resources']=5000.
    assert OBS_SPACE.contains(before) and len(before)==84
    np.testing.assert_array_equal(before,encode(ex,changed))


def test_entry_waits_for_angular_velocity_to_settle(flight_observation):
    from rmuc_arena import sdk
    obs=copy.deepcopy(flight_observation);lane=sdk.fly_slope_lane(obs['arena'],'blue')
    own(obs).update(x=lane['lip'][0]+2.55,y=lane['lip'][1],heading=math.pi,upright=1.,grounded=True,vx=0.,vy=0.,angular_velocity=[0.,0.,.3])
    ex=TacticalExecutor();ex.submit({'id':'rotating','skill':'fly_slope','lane_team':'blue'},obs)
    for _ in range(10):obs=later(obs);ex.update(obs)
    assert not ex.skill_state.get('launched')
    own(obs)['angular_velocity']=[0.,0.,0.]
    for _ in range(5):obs=later(obs);ex.update(obs)
    assert ex.skill_state['launched']
