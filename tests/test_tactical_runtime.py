import copy
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.tactical_runtime import TacticalSession
from arena_rl.tactical_envs import TacticalVecEnv

ROOT=configure()
POLICIES=Path(os.environ.get('TACTICAL_POLICIES',Path(__file__).resolve().parents[1]/'checkpoints/tactics'))


def test_new_goal_preserves_simulator_tick_and_ammo(tmp_path):
    session=TacticalSession(POLICIES,arena_root=ROOT,record=True)
    try:
        session.reset();session.command({'id':'move','skill':'navigate','point':[4.5,5.15]})
        session.advance(5);before=session.observe();pid=session.batch.states[0]['info']['simulator_pid']
        session.command({'id':'guard','skill':'defend_region','center':[4.5,5.15]})
        after=session.observe()
        assert before['tick']==after['tick'] and before['robot']['ammo']==after['robot']['ammo']
        assert before['robot']['x']==after['robot']['x']
        assert session.batch.states[0]['info']['simulator_pid']==pid
        session.advance(2);assert session.observe()['tick']==before['tick']+4
        path=tmp_path/'continuous.json';session.save_replay(path)
        from rmuc_arena.replays import verify_replay
        assert verify_replay(path)['verified_ticks']==14
        assert len(json.loads(path.with_suffix('.commands.json').read_text())['commands'])==3
        scene=session.observe();assert 'observations' not in scene and 'state' not in scene
    finally:session.close()


def test_vecenv_keeps_terminal_state_and_supports_subset_reset():
    env=TacticalVecEnv(2,family='navigation',arena_root=ROOT)
    try:
        env.reset();pid=env.batch.states[1]['info']['simulator_pid'];env.batch.episodes[0].max_steps=1
        obs,r,dones,infos=env.step(np.array([[0,0,1,1,0,1,0]]*2))
        assert dones[0] and infos[0]['TimeLimit.truncated']
        assert infos[0]['terminal_observation'].shape==obs[0].shape
        assert env.batch.states[1]['info']['simulator_pid']==pid and env.batch.states[1]['info']['tick']==2
    finally:env.close()


@pytest.mark.skipif(os.name=='nt',reason='POSIX select on subprocess pipes; raw service reads are portable')
def test_jsonl_errors_and_realtime_execution_during_planner_delay():
    process=subprocess.Popen([sys.executable,'-m','arena_rl','serve-tactics','--policies',str(POLICIES),'--realtime'],
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    def send(value):process.stdin.write(value+'\n');process.stdin.flush()
    def receive(predicate,timeout=20):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            readable,_,_=select.select([process.stdout],[],[],max(0,end-time.monotonic()))
            if not readable:break
            row=json.loads(process.stdout.readline())
            if predicate(row):return row
        raise AssertionError('Missing protocol response')
    try:
        ready=receive(lambda r:r['type']=='ready');initial=ready['states'][0]['tick']
        send('{bad-json');assert not receive(lambda r:r.get('type')=='error')['ok']
        send(json.dumps({'id':'move','op':'command','command':{'id':'move','skill':'navigate','point':[4.5,5.15]}}))
        assert receive(lambda r:r.get('id')=='move')['ok']
        # No planner input is supplied while the real-time controller advances.
        state=receive(lambda r:r.get('type')=='state')
        assert state['states'][0]['tick']>=initial+10
        assert state['states'][0]['robot']['x']!=ready['states'][0]['robot']['x']
        send(json.dumps({'id':'replace','op':'command','command':{'id':'guard','skill':'defend_region','center':[4.5,5.15]}}))
        replacement=receive(lambda r:r.get('id')=='replace')
        assert replacement['ok'] and replacement['result']['feedback']['revision']==2
        send('{"op":"close"}');process.wait(timeout=20);assert process.returncode==0
    finally:
        if process.poll() is None:process.terminate();process.wait(timeout=20)
        process.stdin.close();process.stdout.close();process.stderr.close()


def test_runtime_scene_does_not_expose_opponent_own_state():
    session=TacticalSession(POLICIES,arena_root=ROOT)
    try:
        session.reset();before=session.observe()
        hidden=session.batch.states[0]['observations']['blue']['robots'][0]
        hidden['x']=7.7;hidden['hp']=17
        assert session.observe()==before
    finally:session.close()


def test_capabilities_filter_map_and_loaded_models():
    for season in ('2026','2027'):
        session=TacticalSession(POLICIES,arena_root=ROOT,season=season)
        try:
            session.reset();cap=session.capabilities()
            assert len(cap['skills'])==11 and 'navigate' in cap['available_skills']
            assert ('fly_slope' in cap['available_skills'])==(season=='2026')
            assert ('resupply' in cap['available_skills'])==(season=='2027')
            assert 'resources' in session.observe()['economy']
        finally:session.close()


def test_runtime_support_commands_keep_ammo_and_are_idempotent():
    session=TacticalSession(POLICIES,arena_root=ROOT)
    try:
        session.reset();scene=session.observe()
        command={'id':'supply','skill':'resupply','ammo_target':100}
        session.command(command);assert session.command(command)['duplicate']
        assert session.observe()['tick']==scene['tick'] and session.observe()['robot']['ammo']==scene['robot']['ammo']
        before=session.observe()['feedback']
        with pytest.raises(ValueError):session.command({'id':'wrong-map','skill':'fly_slope'})
        assert session.observe()['feedback']==before
    finally:session.close()


def test_one_attack_command_composes_navigation_and_combat():
    session=TacticalSession(POLICIES,arena_root=ROOT)
    try:
        session.reset([4000000],[{'robots':{'red-3':{'x':4.5,'y':5.15,'heading':0.,'aim':0.,'ammo':200}}}])
        session.command({'id':'attack','skill':'attack_structure','target_id':'blue_outpost','damage_goal':60.,'timeout_s':90.})
        families=set()
        for _ in range(900):
            families.add(session.batch.executors[0].execution_family)
            if session.batch.executors[0].status!='running':break
            session.advance()
        assert session.batch.executors[0].status=='succeeded'
        assert families=={'navigation','combat'}
    finally:session.close()
