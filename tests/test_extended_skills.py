import copy
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.executor import TacticalExecutor,own
from arena_rl.extended_encoding import decode,encode,OBS_SPACE
from arena_rl.tactical_envs import TacticalBatch

configure()


@pytest.fixture
def observation():
    from rmuc_arena.env import ArenaEnv
    from arena_rl.extended_curricula import SUPPORT_LINEUPS
    with ArenaEnv(season='2027',lineups=SUPPORT_LINEUPS) as env:
        return env.reset(301,scenario={'robots':{
            'red-3':{'x':4.5,'y':5.15,'heading':0.,'ammo':0},
            'red-4':{'x':5.8,'y':5.15,'heading':0.,'ammo':0}}})['observations']['red']


def later(obs,seconds=.1):
    obs=copy.deepcopy(obs);obs['tick']+=round(seconds/.05);obs['time']+=seconds
    return obs


def test_escort_ownership_and_atomic_rejection(observation):
    ex=TacticalExecutor();ex.submit({'id':'move','skill':'navigate','point':[7,5]},observation)
    for target in ('red-3','blue-3'):
        with pytest.raises(ValueError):ex.submit({'id':'escort','skill':'escort','target_id':target},observation)
        assert ex.command.id=='move' and ex.status=='running'
    ex.submit({'id':'escort','skill':'escort','target_id':'red-4'},observation)
    assert ex.plan['target'] is None and ex.extra['protected_ally']=='red-4'
    assert set(decode(ex,observation,np.array([1,1,1,1,0,1,0])))=={'3'}
    assert OBS_SPACE.contains(encode(ex,observation))
    dead=later(observation);next(r for r in dead['robots'] if r['id']=='red-4')['alive']=False
    ex.update(dead);assert ex.reason=='escort_target_destroyed'


def test_patrol_requires_every_waypoint_and_full_dwell(observation):
    ex=TacticalExecutor();ex.submit({'id':'patrol','skill':'patrol','waypoints':[[4.5,5.15],[6.5,5.15]],
        'patrol_loops':2,'dwell_s':.3},observation)
    state=observation
    for visit,point in enumerate([[4.5,5.15],[6.5,5.15]]*2):
        for k in range(3):
            state=later(state);r=own(state);r.update(x=point[0],y=point[1],vx=0.,vy=0.)
            ex.update(state);ex.update(state)
        assert ex.extra['waypoint_index']==visit+1
        assert ex.status==('succeeded' if visit==3 else 'running')


def test_resupply_preserves_real_contact_budget_and_failure_feedback(observation):
    ex=TacticalExecutor();ex.submit({'id':'supply','skill':'resupply','budget_limit':15},observation)
    assert 'buy_ammo' not in decode(ex,observation,np.array([0,0,1,1,1,1,0]))['3']
    contact=later(observation,2.)
    zone=next(z for z in ex.arena['zones'] if z['kind']=='supply' and z.get('owner')=='red')
    own(contact).update(x=zone['x'],y=zone['y'])
    contact['zones'][zone['id']]['occupants']=['red-3'];contact['teams']['red']['resources']=800
    ex.update(contact)
    action=decode(ex,contact,np.array([0,0,1,1,1,1,0]))['3']
    assert action['buy_ammo']=={'amount':10,'remote':False}
    paid=later(contact);paid['teams']['red'].update(resources=900,purchased_17=10)
    own(paid)['ammo']=10;ex.update(paid)
    assert ex.extra['spent']==10 and ex.reason=='insufficient_supply_budget'
    assert not decode(ex,paid,np.array([1,3,1,1,1,1,1]))['3']['fire']


def test_2027_rejects_fly_slope_without_cancelling(observation):
    ex=TacticalExecutor();ex.submit({'id':'move','skill':'navigate','point':[7,5]},observation)
    with pytest.raises(ValueError,match='no selected'):ex.submit({'id':'fly','skill':'fly_slope'},observation)
    assert ex.command.id=='move'


def test_reverse_permission_and_real_crossing():
    from arena_rl.tactical_learning import actions_for_baseline
    from arena_rl.extended_curricula import sample
    batch=TacticalBatch(2,family='flight',seed=100,record=False)
    try:
        # Both mirror lanes earn their reverse access through actual forward actions.
        batch.reset(seeds=[100,103])
        assert all(ex.flight_authorized for ex in batch.executors)
        for _ in range(150):
            ids=np.flatnonzero(~batch.finished).tolist()
            if not ids:break
            batch.step(actions_for_baseline(batch,ids,'heuristic',np.random.default_rng(1)),env_ids=ids)
        for ex in batch.executors:
            assert ex.status=='succeeded' and ex.extra['airborne_s']+1e-8>=.1 and ex.extra['landing_hold_s']+1e-8>=.3
        batch.reset([0],seeds=[102])
        ex=batch.executors[0];obs=batch.states[0]['observations']['red']
        assert not ex.flight_authorized
        with pytest.raises(ValueError,match='completed forward'):
            ex.submit({'id':'shortcut','skill':'fly_slope','flight_direction':'reverse'},obs)
    finally:batch.close()


def test_original_failed_navigation_seeds():
    from arena_rl.tactical_learning import load_tactical_policy
    from pathlib import Path
    from arena_rl.bridge import configure
    policy,_=load_tactical_policy(Path(__file__).resolve().parents[1]/'checkpoints/tactics/navigation',configure())
    batch=TacticalBatch(2,family='navigation')
    try:
        obs=batch.reset(seeds=[3000008,3000027])
        for _ in range(750):
            ids=np.flatnonzero(~batch.finished).tolist()
            if not ids:break
            new,*_=batch.step(policy.predict(obs[ids],deterministic=True)[0],env_ids=ids)
            for i,o in zip(ids,new):obs[i]=o
        assert all(ex.status=='succeeded' for ex in batch.executors)
    finally:batch.close()
