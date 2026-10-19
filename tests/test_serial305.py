import json
from pathlib import Path
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.tactical_envs import TacticalBatch
from arena_rl.tactical_learning import load_tactical_policy
from arena_rl.executor import own

ROOT=configure();POLICIES=Path(__file__).resolve().parents[1]/'checkpoints/serial305'


def run_policy(batch,policy,limit):
    from arena_rl.tactical_envs import encode_observation
    for _ in range(limit):
        ids=np.flatnonzero(~batch.finished).tolist()
        if not ids:break
        features=np.stack([encode_observation(batch.executors[i],batch.states[i]['observations']['red'],batch.family) for i in ids])
        batch.step(policy.predict(features,deterministic=True)[0],env_ids=ids)


def test_serial_recovery_is_a_physical_fall_and_preserves_pose_history(tmp_path):
    policy,metadata=load_tactical_policy(POLICIES/'mobility',ROOT)
    assert metadata['model_id']==305 and metadata['ppo_weights_changed']
    batch=TacticalBatch(1,family='mobility',model_id=305,record=True)
    try:
        batch.reset(seeds=[101]);r=own(batch.states[0]['observations']['red'])
        assert r['upright']<.4 and r['model_slot']==305
        tick=batch.states[0]['info']['tick'];pid=batch.states[0]['info']['simulator_pid']
        run_policy(batch,policy,140)
        ex=batch.executors[0];assert ex.status=='succeeded' and ex.recovery_count>=1
        assert own(batch.states[0]['observations']['red'])['upright']>.93
        assert batch.states[0]['info']['tick']>tick and batch.states[0]['info']['simulator_pid']==pid
        path=tmp_path/'physical-recovery.json';batch.engine.save_replay([path])
        from rmuc_arena.replays import verify_replay
        assert verify_replay(path)['verified_ticks']==batch.states[0]['info']['tick']
        data=json.loads(path.read_text())
        assert any(r['upright']<-.5 for f in data['frames'] for r in f['robots'] if r['id']=='red-3')
        assert not set(data['metadata']['training_scenario']['robots']['red-3']) & {'quaternion','hp','upright'}
    finally:batch.close()


def test_low_hp_respawn_can_retreat_and_exact_kerb_case_finishes():
    policy,_=load_tactical_policy(POLICIES/'mobility',ROOT)
    batch=TacticalBatch(2,family='mobility',model_id=305)
    try:
        batch.reset(seeds=[102,104]);r=own(batch.states[0]['observations']['red'])
        assert r['alive'] and r['hp']/r['max_hp']<=.11 and batch.executors[0].status=='running'
        run_policy(batch,policy,1100)
        assert all(e.status=='succeeded' for e in batch.executors)
    finally:batch.close()


def test_recovery_suspends_and_resumes_original_goal_without_reset():
    policy,_=load_tactical_policy(POLICIES/'mobility',ROOT)
    batch=TacticalBatch(1,family='mobility',model_id=305)
    try:
        batch.reset(seeds=[101]);before=batch.states[0]['info'].copy()
        batch.command(0,{'id':'withdraw-after-fall','skill':'retreat','timeout_s':110.,'low_hp_fraction':0.})
        assert batch.executors[0].plan['phase']=='recover'
        # The recovery subcontroller must see the same intent as explicit recover.
        from arena_rl.mobility_encoding import encode,FEATURES
        features=encode(batch.executors[0],batch.states[0]['observations']['red'])
        assert features[FEATURES.index('retreat_intent')]==0.
        assert batch.states[0]['info']['tick']==before['tick']
        run_policy(batch,policy,1100)
        ex=batch.executors[0]
        assert ex.command.id=='withdraw-after-fall' and ex.status=='succeeded' and ex.recovery_count>=1
        assert batch.states[0]['info']['simulator_pid']==before['simulator_pid']
    finally:batch.close()


def test_mixed_model_roster_is_not_silently_rewritten():
    with pytest.raises(ValueError):
        TacticalBatch(1,model_id=305,lineups={'red':[{'slot':3,'model':3}],'blue':[{'slot':3,'model':3}]})


def test_serial_live_idle_and_cancel_are_valid_observation_states():
    from arena_rl.tactical_runtime import TacticalSession
    session=TacticalSession(POLICIES,arena_root=ROOT)
    try:
        session.reset();session.advance(2)
        assert session.observe()['robot']['model_slot']==305
        assert session.observe()['feedback']['status']=='idle'
        session.command({'id':'recover-live','skill':'recover','timeout_s':12.})
        session.cancel();session.advance(2)
        assert session.observe()['feedback']['status']=='cancelled'
    finally:session.close()


def test_contact_escape_policy_is_used_by_live_navigation():
    from arena_rl.tactical_runtime import TacticalSession
    session=TacticalSession(POLICIES,envs=2,arena_root=ROOT)
    try:
        scenario={'robots':{'red-3':{'x':3.89,'y':13.43,'heading':0.,'ammo':100}}}
        session.reset([7300000,7300001],[scenario,scenario])
        session.command({'id':'contact-nav','skill':'navigate','point':[4.6,13.6],'timeout_s':40.},0)
        session.command({'id':'contact-retreat','skill':'retreat','timeout_s':110.},1)
        families=[set(),set()]
        for _ in range(1100):
            for i in range(2):families[i].add(session.batch.executors[i].execution_family)
            if all(e.status!='running' for e in session.batch.executors):break
            session.advance()
        assert all(e.status=='succeeded' for e in session.batch.executors)
        assert 'mobility' in families[0] and session.batch.executors[0].command.id=='contact-nav'
    finally:session.close()


def test_escape_detection_has_no_gap_outside_the_arrival_radius():
    from arena_rl.navigation import plan_motion
    batch=TacticalBatch(1,model_id=305)
    try:
        batch.reset(seeds=[7300100])
        obs=batch.states[0]['observations']['red'];r=own(obs);ex=batch.executors[0]
        goal=[r['x']+.43,r['y']]
        ex.submit({'id':'near-arrival','skill':'navigate','point':goal,'radius_m':.4},obs)
        ex.route_state={}
        plan_motion(ex,r,goal,obs['time'])
        hint,heading=plan_motion(ex,r,goal,obs['time']+2.)
        assert ex.route_state['attempts']==1 and np.linalg.norm(hint)>.1 and heading is not None
    finally:batch.close()
