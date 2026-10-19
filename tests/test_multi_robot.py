import json
import math
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
import pytest
from arena_rl.bridge import configure

ROOT=configure()


def robot_from_design(model):
    from rmuc_arena.models import robot_models
    d=robot_models()[str(model)];size=d['chassis_size_m']
    return {'model_slot':model,'role':d['role'],'chassis_type':d['chassis_type'],'radius':math.hypot(size[0],size[1])/2,'mass_kg':d.get('mass_kg',25)},{'robot_models':robot_models()}


def test_profiles_cover_every_2027_ground_model():
    from arena_rl.robots import GROUND_MODELS_2027,profile,DESCRIPTOR
    seen=set()
    for role,models in GROUND_MODELS_2027.items():
        for model in models:
            robot,obs=robot_from_design(model);p=profile(robot,obs)
            assert p.role==role and p.descriptor().shape==(len(DESCRIPTOR),)
            assert p.legged==(p.drive=='wheel_legged') and p.holonomic==(p.drive in ('mecanum','omni','swerve'))
            assert p.caliber==('42mm' if role=='heavy' else '17mm')
            assert p.self_righting==(model in (305,2772))
            seen.add(p.drive)
    assert seen=={'wheel_legged','mecanum','omni','swerve'}


def test_install_launcher_is_heavy_only_protocol_extension():
    from arena_rl.commands import Command,capabilities
    c=Command.parse({'id':'fit','skill':'install_launcher'})
    assert c.policy_family=='navigation' and 'install_launcher' in capabilities()['skills']
    with pytest.raises(ValueError):Command.parse({'id':'fit','skill':'install_launcher','point':[1,1]})


def test_curricula_are_valid_for_every_family():
    from arena_rl.ground_curricula import sample,clear,ROLE_SLOT,role_of
    for family in ('navigation','combat','defense','support'):
        for seed in range(30):
            e=sample(family,1000+seed)
            assert e.slot==ROLE_SLOT[role_of(e.model)] and e.stages and any(s.recorded for s in e.stages)
            assert e.learner in e.scenario['robots'] and any(d['slot']==int(e.slot) and d['model']==e.model for d in e.lineups['red'])
            if not e.warmup:assert clear((e.scenario['robots'][e.learner]['x'],e.scenario['robots'][e.learner]['y']),e.model)
            for stage in e.stages:
                from arena_rl.commands import Command
                Command.parse({k:v for k,v in stage.command.items()})


@pytest.mark.parametrize('family,model',[('navigation',2732),('navigation',305),('combat',2731),('defense',303),('support',2771)])
def test_ground_env_steps_with_expert(family,model):
    from arena_rl.ground_envs import GroundEnv
    env=GroundEnv(family,seed=11,models=[model])
    try:
        obs,info=env.reset()
        assert obs.shape==env.observation_space.shape and np.isfinite(obs).all() and info['model']==model
        for _ in range(40):
            obs,reward,term,trunc,info=env.step(env.expert_action())
            assert np.isfinite(reward) and env.observation_space.contains(obs)
            if term or trunc:obs,_=env.reset()
    finally:env.close()


def test_decoder_uses_chassis_semantics():
    from arena_rl.ground_envs import GroundEnv
    from arena_rl.ground_encoding import decode
    flat=GroundEnv('combat',seed=3,models=[2732]);legged=GroundEnv('navigation',seed=3,models=[2731])
    try:
        flat.reset();legged.reset()
        out=decode(flat.executor,flat._observation(),np.array([1,2,1,1,0,1,0]),'combat')[flat.episode.slot]
        assert 'yaw_rate' in out and 'heading' not in out and 'leg_height' not in out
        boost=decode(flat.executor,flat._observation(),np.array([1,2,1,1,0,2,0]),'combat')[flat.episode.slot]
        assert boost['power']==200.
        out=decode(legged.executor,legged._observation(),np.array([1,2,1,1,0,2,1]),'navigation')[legged.episode.slot]
        assert 'leg_height' in out and 'yaw_rate' not in out and out['jump']==1.
        with pytest.raises(ValueError):decode(legged.executor,legged._observation(),np.array([9,0,0,0,0,0,0]),'navigation')
    finally:flat.close();legged.close()


def test_rollout_workers_collect_and_evaluate():
    from arena_rl.parallel import RolloutWorkers
    from arena_rl.ground_learning import _model,_cpu_state,POLICY_KWARGS
    model=_model(2,32,1,'cpu',1e-4)
    with RolloutWorkers('navigation',2,5,(2732,305),ROOT) as pool:
        results=pool.collect(_cpu_state(model),POLICY_KWARGS,32,.99)
        assert len(results)==2 and all(r['obs'].shape[0]==32 and r['actions'].shape==(32,7) for r in results)
        expert=pool.collect(None,POLICY_KWARGS,16,.99,expert=True)
        assert all(np.isfinite(r['values']).all() for r in expert)


def test_tiny_training_round_trip_into_policy_bank(tmp_path):
    from arena_rl.ground_learning import train_ground,load_ground_policy
    from arena_rl.team_runtime import GroundPolicyBank
    meta=train_ground(family='navigation',steps=128,workers=2,n_steps=64,expert_episodes=1,bc_epochs=1,seed=9,models=(2732,),output=tmp_path/'navigation',arena_root=ROOT)
    assert meta['ppo_weights_changed'] and meta['schema']=='arena-ground-v1' and meta['demonstrations']['episodes']>=1
    loaded,_=load_ground_policy(tmp_path/'navigation',ROOT)
    assert loaded['policy_sha256']==meta['policy_sha256']
    bank=GroundPolicyBank(tmp_path,ROOT);assert set(bank.models)=={'navigation'}


def _team(lineup=None):
    from rmuc_arena.env import ArenaEnv
    from arena_rl.team_runtime import TeamController
    from arena_rl.team_matches import LINEUP_2027
    env=ArenaEnv(season='2027',duration=420,lineups={'red':lineup or LINEUP_2027,'blue':LINEUP_2027},compact=True)
    state=env.reset(seed=4)
    return env,state,TeamController('red',None,'expert').attach(state['observations']['red'])


def test_feasibility_encodes_2027_rule_chains():
    from arena_rl.team_commander import feasibility,compile_order,InfeasibleOrder
    env,state,red=_team()
    try:
        o=state['observations']['red']
        assert feasibility(red,o,'red-3','push_base')=='no_ammo_resupply_first'
        assert feasibility(red,o,'red-7','push_base').startswith('enemy_base_protected')
        assert feasibility(red,o,'red-7','occupy_enemy_fortress')=='enemy_fortress_inactive_before_180s'
        assert feasibility(red,o,'red-7','hold_own_fortress').startswith('own_fortress_inactive')
        assert feasibility(red,o,'red-1','attack_outpost')=='launcher_locked'
        assert feasibility(red,o,'red-1','install_launcher')=='launcher_locked'
        assert feasibility(red,o,'red-6','drone_strike_outpost') is None and feasibility(red,o,'red-6','push_base')=='not_an_aerial_objective'
        command=compile_order(red,o,'red-7',{'objective':'capture_enemy_central','target':None,'place':None})
        assert command['skill']=='defend_region' and command['center']==[17.2,7.5]
        assert compile_order(red,o,'red-3',{'objective':'resupply','target':None,'place':None})['skill']=='resupply'
        assert compile_order(red,o,'red-4',{'objective':'goto','target':None,'place':'own_fortress_front'})['point']==[5.15,7.5]
        with pytest.raises(InfeasibleOrder):compile_order(red,o,'red-3',{'objective':'attack_outpost','target':None,'place':None})
    finally:env.close()


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','fake-test-credential')
    holder={'requests':[],'arguments':None}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));holder['requests'].append(body)
            response={'id':'r1','model':'deepseek-flash','usage':{'total_tokens':99},'choices':[{'finish_reason':'tool_calls','message':{'tool_calls':[
                {'id':'c1','type':'function','function':{'name':'issue_orders','arguments':json.dumps(holder['arguments'])}}]}}]}
            self.send_response(200);self.end_headers();self.wfile.write(json.dumps(response).encode())
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    holder['url']=f'http://127.0.0.1:{server.server_port}'
    try:yield holder
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_team_llm_call_is_strict_and_applies_through_harness(provider):
    from arena_rl.team_commander import team_scene,apply_orders
    from arena_rl.team_planner import TeamLLMPlanner
    from arena_rl.llm_planner import PlannerError
    env,state,red=_team()
    try:
        o=state['observations']['red'];scene=team_scene(red,o)
        orders={k:{'objective':'continue','intent':'wait','target':None,'place':None} for k,v in scene['robots'].items()}
        orders['red-3'].update(objective='resupply',intent='buy ammo');orders['red-7'].update(objective='capture_enemy_central',intent='take card')
        orders['red-6'].update(objective='drone_strike_outpost',intent='ram')
        provider['arguments']={'plan':'reload, take their card, strike outpost','orders':orders}
        decision=TeamLLMPlanner(base_url=provider['url']).decide(scene)
        body=provider['requests'][0]
        assert 'fake-test-credential' not in json.dumps(body) and body['tool_choice']=='required' and len(body['tools'])==1
        enum=body['tools'][0]['function']['parameters']['properties']['orders']['properties']['red-1']['properties']['objective']['enum']
        assert 'attack_outpost' not in enum and 'hold_own_central' in enum
        results=apply_orders(red,o,decision.orders,'llm')
        assert results['red-3']['action']=='commanded' and results['red-7']['command']['skill']=='defend_region'
        assert results['red-6']['action']=='commanded' and red.aerial.status=='running' and results['red-1']['action']=='continued'
        orders['red-1'].update(objective='attack_outpost');provider['arguments']={'plan':'x','orders':orders}
        with pytest.raises(PlannerError,match='invalid_tool_call'):TeamLLMPlanner(base_url=provider['url']).decide(scene)
    finally:env.close()


def test_short_doctrine_team_match_produces_verified_replay(tmp_path):
    from arena_rl.team_matches import run_team_matches
    report=run_team_matches(policies=None,output=tmp_path/'match',matches=1,commander='doctrine',low_level='expert',speed=0,duration=20,arena_root=ROOT)
    row=report['matches'][0]
    assert row['reason']=='timeout' and row['replay_verification'] and row['actions'].get('commanded',0)>0
    assert (tmp_path/'match'/row['replay']).is_file()


def test_drone_routes_detour_around_the_rune_gantry():
    from arena_rl.aerial import detour
    start=[13.01,8.49];goal=[18.12,3.54,1.6]
    waypoint=detour(start,goal)
    assert waypoint!=goal and math.dist(waypoint[:2],(14.,7.5))>1.8
    assert detour(waypoint,goal)==goal and detour([12.,9.],[11.,12.,1.6])==[11.,12.,1.6]


def test_aerial_env_and_guidance_controller():
    from arena_rl.aerial_learning import AerialEnv,OBS_SPACE
    env=AerialEnv(seed=21,models=(2762,),max_steps=40)
    try:
        obs,info=env.reset(seed=21)
        assert obs.shape==OBS_SPACE.shape and info['model']==2762
        done=False
        while not done:
            obs,reward,term,trunc,info=env.step(env.expert_action());done=term or trunc
            assert OBS_SPACE.contains(obs) and np.isfinite(reward)
        assert info['steps']<=40 and info['state'] in ('flying','docked','down','disabled')
    finally:env.close()
