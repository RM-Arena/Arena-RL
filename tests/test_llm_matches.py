import time
import pytest
from arena_rl.bridge import configure
from arena_rl.llm_matches import run_llm_matches,decision_due,planner_skills
from arena_rl.llm_planner import Decision,PlannerError
from pathlib import Path

ROOT=configure();POLICIES=Path(__file__).resolve().parents[1]/'checkpoints/tactics'


class FakePlanner:
    def decide(self,scene,skills,context,history):
        time.sleep(.02)
        if scene['feedback']['status']=='running':name='continue_current';arguments={'intent':'continue test goal'}
        elif scene['robot']['ammo']==0:
            name='resupply';arguments={'ammo_target':100,'hp_target_fraction':.9,'budget_limit':150,'timeout_s':60,'intent':'test supply'}
        else:
            name='defend_region';base=scene['bases']['red'];arguments={'center':[base['x'],base['y']],'region_radius_m':3.,
                 'priority_ids':None,'timeout_s':5.,'low_hp_fraction':None,'intent':'test guard'}
        return Decision(name,arguments,'test-double','test-double','fake',{},.02,scene['tick'],'fake-call')


def test_full_referee_match_outlives_goal_and_training_time_limits(tmp_path):
    report=run_llm_matches(policies=POLICIES,matches=1,seed=999000,output=tmp_path/'full',speed=20,
        max_calls=150,max_age=5,model='test-double',planner=FakePlanner(),verify=False)
    match=report['matches'][0]
    assert report['all_matches_completed'] and report['healthy_control']
    assert match['time_s']==420. and match['final_tick']==8400 and match['terminated'] and not match['truncated']
    assert match['command_continuity_passed'] and match['planner_calls']>5
    assert match['initial_scenario'] is None and match['controlled_robot']=='red-3'
    assert match['event_counts'].get('invalid_action',0)==0
    assert (tmp_path/'full'/match['replay']).is_file()


class BrokenPlanner:
    def decide(self,*args):raise PlannerError('provider_http_error',False,401)


def test_auth_failure_preserves_partial_evidence_without_faking_a_full_match(tmp_path):
    report=run_llm_matches(policies=POLICIES,matches=1,output=tmp_path/'partial',model='test-double',
        speed=20,planner=BrokenPlanner(),verify=False)
    assert not report['all_matches_completed'] and not report['healthy_control']
    assert report['fatal_error']=='provider_http_error'
    assert report['matches'][0]['time_s']<420
    assert report['matches'][0]['actions'].get('commanded',0)==0


def test_decision_schedule_handles_completion_and_respawn():
    scene={'time':8.,'feedback':{'status':'failed'},'robot':{'alive':True,'powered':True}}
    assert decision_due(scene,6.,('running',True),5.)
    assert decision_due(scene,6.,('failed',False),5.)
    scene['robot']['alive']=False
    assert not decision_due(scene,6.,('failed',True),5.)
