"""Asynchronous LLM supervision over full 420-second referee matches."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit
import time
import numpy as np
from .bridge import configure,provenance
from .learning import dump
from .llm_planner import DeepSeekPlanner,PlannerError,apply_decision,MODEL,BASE_URL
from .match_opponents import LINEUPS,PROFILES,actions as opponent_actions
from .tactical_runtime import TacticalSession


def public_context(session):
    arena=session.batch.executors[0].arena
    return {'season':session.batch.season,'robot_id':'red-3','match_duration_s':420,
            'field':{'width':arena['width'],'height':arena['height']},
            'supply_zones':[{k:z[k] for k in ('id','owner','x','y') if k in z} for z in arena['zones'] if z['kind']=='supply'],
            'terrain':[{k:t[k] for k in ('id','kind','x','y','w','h') if k in t} for t in arena['terrain']],
            'roster':session.batch.states[0]['observations']['red']['roster'],
            'base_protection':'full enemy outpost control protects its base',
            'control_hz':10,'ballistic_assistance':True}


def planner_skills(session,scene):
    skills=list(session.capabilities()['available_skills'])
    if not any(a['alive'] and a['role']!='aerial' for a in scene['allies']):skills.remove('escort') if 'escort' in skills else None
    return skills


def decision_due(scene,last_time,last_feedback,interval):
    feedback=scene['feedback'];alive=scene['robot']['alive'] and scene['robot']['powered']
    if not alive:return scene['time']-last_time>=max(10.,interval)
    previous_status,previous_alive=last_feedback
    changed=(feedback['status']!=previous_status and feedback['status']!='running') or not previous_alive
    return last_time<0 or changed and scene['time']-last_time>=1. or scene['time']-last_time>=interval


def same_goal_identity(ex):return (ex.command.id if ex.command else None,ex.status)


def run_llm_matches(*,policies,matches=3,seed=9000000,output,model=MODEL,base_url=BASE_URL,api_key_env='DEEPSEEK_API_KEY',
                    plan_interval=5.,timeout=8.,max_age=5.,max_calls=120,speed=1.,arena_root=None,
                    opponent_profiles=None,verify=True,planner=None):
    if isinstance(matches,bool) or not isinstance(matches,int) or not 1<=matches<=8:raise ValueError('matches must be 1..8')
    for name,value,lo,hi in [('plan_interval',plan_interval,1,30),('timeout',timeout,.1,60),('max_age',max_age,.1,30),('speed',speed,.1,20)]:
        if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+name)
    if isinstance(max_calls,bool) or not isinstance(max_calls,int) or not 1<=max_calls<=500:raise ValueError('Invalid inference call cap')
    profiles=list(opponent_profiles or PROFILES)
    if not profiles or any(p not in PROFILES for p in profiles):raise ValueError('Unknown opponent profile')
    out=Path(output)
    if (out/'report.json').exists() or (out/'decisions.jsonl').exists():raise ValueError('Match output already exists; use a new directory')
    root=configure(arena_root);client=planner or DeepSeekPlanner(model=model,base_url=base_url,api_key_env=api_key_env,timeout=timeout)
    out.mkdir(parents=True,exist_ok=True)
    session=TacticalSession(policies,envs=matches,seed=seed,record=True,lineups=LINEUPS,arena_root=root)
    session.batch.opponent_controller=lambda i,obs,arena:opponent_actions(obs,arena,profiles[i%len(profiles)])
    pool=ThreadPoolExecutor(max_workers=matches,thread_name_prefix='tactical-llm')
    pending={};last_request=[-100.]*matches;last_feedback=[('idle',True)]*matches
    histories=[[] for _ in range(matches)];counts=[Counter() for _ in range(matches)];events=[Counter() for _ in range(matches)]
    latencies=[[] for _ in range(matches)];calls=[0]*matches;tokens=[Counter() for _ in range(matches)]
    errors=[[] for _ in range(matches)];transitions=[[] for _ in range(matches)];previous=[None]*matches
    initial_pids=[];continuity=[True]*matches;api_failed=False;fatal_error=None;missed=0
    start=time.monotonic();last_progress=start;deadline=start;period=.1/speed
    trace=(out/'decisions.jsonl').open('w');telemetry=(out/'telemetry.jsonl').open('w')
    def log(value):trace.write(json.dumps(value,allow_nan=False,separators=(',',':'))+'\n');trace.flush()
    try:
        scenes=session.reset();initial_pids=[s['info']['simulator_pid'] for s in session.batch.states]
        context=public_context(session)
        log({'type':'configuration','model':model,'endpoint_origin':urlsplit(base_url).hostname,
             'seed':seed,'matches':matches,'profiles':[profiles[i%len(profiles)] for i in range(matches)],
             'duration_s':420,'speed':speed,'plan_interval_s':plan_interval,'max_age_s':max_age,'max_calls':max_calls,
             'controlled_robot':'red-3','lineups':LINEUPS,'scenario':None,'public_context':context,
             'policy_sha256':{k:m['policy_sha256'] for k,m in session.bank.metadata.items()}})
        while not all(scene['episode_finished'] for scene in scenes):
            for i,(future,basis,sent_tick,identity) in list(pending.items()):
                if not future.done():continue
                del pending[i]
                if scenes[i]['episode_finished']:
                    log({'type':'discarded_after_match','env_id':i,'identity':identity});continue
                try:
                    decision=future.result();latencies[i].append(decision.latency_s)
                    tokens[i].update({k:decision.usage.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
                    log({'type':'decision','env_id':i,'identity':identity,'received_tick':scenes[i]['tick'],'advanced_ticks_while_inference':scenes[i]['tick']-sent_tick,**asdict(decision)})
                    before=session.observe(i)
                    result=apply_decision(session,decision,i,identity,max_age)
                    after=session.observe(i)
                    continuity[i]&=before['tick']==after['tick'] and before['robot']['ammo']==after['robot']['ammo'] and before['robot']['x']==after['robot']['x'] and before['robot']['hp']==after['robot']['hp']
                    counts[i][result['action']]+=1
                    if result['action']=='commanded':counts[i]['skill:'+decision.tool]+=1
                    histories[i].append({'tick':scenes[i]['tick'],'tool':decision.tool,'intent':decision.arguments['intent'],
                                         'action':result['action'],'feedback':after['feedback']})
                    histories[i]=histories[i][-8:]
                    log({'type':'applied','env_id':i,'identity':identity,'tick':after['tick'],**result})
                except (PlannerError,ValueError) as error:
                    code=error.code if isinstance(error,PlannerError) else str(error)
                    entry={'type':'planner_error','env_id':i,'identity':identity,'tick':scenes[i]['tick'],'code':code,'details':error.details if isinstance(error,PlannerError) else {}}
                    errors[i].append(entry);log(entry)
                    histories[i].append({'tick':scenes[i]['tick'],'error':code,'details':entry['details'],'feedback':scenes[i]['feedback']});histories[i]=histories[i][-8:]
                    if isinstance(error,PlannerError) and not error.retryable:
                        api_failed=True;fatal_error=code
            if api_failed:break
            for i,scene in enumerate(scenes):
                if scene['episode_finished'] or i in pending or calls[i]>=max_calls:continue
                if decision_due(scene,last_request[i],last_feedback[i],plan_interval):
                    snapshot=json.loads(json.dumps(session.observe(i),allow_nan=False))
                    identity=f'llm-{seed+i}-{calls[i]+1}'
                    future=pool.submit(client.decide,snapshot,planner_skills(session,snapshot),context,list(histories[i]))
                    pending[i]=(future,snapshot,scene['tick'],identity);calls[i]+=1;last_request[i]=scene['time']
                    last_feedback[i]=(scene['feedback']['status'],scene['robot']['alive'] and scene['robot']['powered'])
                    log({'type':'request','env_id':i,'identity':identity,'scene':snapshot})
            scenes=session.advance()
            for i,scene in enumerate(scenes):
                event_list=session.batch.states[i]['info'].get('events',[])
                # Finished actors no longer step; do not recount their last events.
                if previous[i] is None or scene['tick']!=previous[i][0]:events[i].update(e['type'] for e in event_list)
                key=same_goal_identity(session.batch.executors[i])
                if previous[i] is None or key!=previous[i][1]:
                    transitions[i].append({'tick':scene['tick'],**scene['feedback']})
                    log({'type':'execution_transition','env_id':i,'tick':scene['tick'],'feedback':scene['feedback']})
                fresh=previous[i] is None or previous[i][0]!=scene['tick']
                previous[i]=(scene['tick'],key)
                continuity[i]&=session.batch.states[i]['info']['simulator_pid']==initial_pids[i]
                if fresh and scene['tick']%20==0:
                    telemetry.write(json.dumps({'env_id':i,**scene},allow_nan=False,separators=(',',':'))+'\n')
            if time.monotonic()-last_progress>=15:
                print(json.dumps({'type':'llm_match_progress','wall_s':round(time.monotonic()-start,1),
                      'games':[{'seed':seed+i,'time':s['time'],'status':s['feedback']['status'],'skill':s['feedback']['skill'],
                                'alive':s['robot']['alive'],'ammo':s['robot']['ammo'],'calls':calls[i],'errors':len(errors[i])} for i,s in enumerate(scenes)]}),flush=True)
                last_progress=time.monotonic()
            deadline+=period;remaining=deadline-time.monotonic()
            if remaining>0:time.sleep(min(remaining,period))
            elif remaining< -period:missed+=1;deadline=time.monotonic()
    except BaseException as error:
        fatal_error=type(error).__name__;raise
    finally:
        for future,*_ in pending.values():future.cancel()
        # Requests have bounded timeouts; wait for any in-flight sockets before cleanup.
        pool.shutdown(wait=True,cancel_futures=True)
        rows=[]
        for i,state in enumerate(session.batch.states):
            if state is None:continue
            path=out/f'match-{seed+i}.json';session.save_replay(path,i)
            verification=None
            if verify:
                from rmuc_arena.replays import verify_replay
                verification=verify_replay(path)
            scene=session.observe(i);info=state['info'];obs=state['observations']['red']
            rows.append({'seed':seed+i,'opponent_profile':profiles[i%len(profiles)],'time_s':scene['time'],'final_tick':scene['tick'],
                'finished':scene['episode_finished'],'terminated':bool(state['terminated']),'truncated':bool(state['truncated']),
                'winner':info['winner'],'reason':info['reason'],'controlled_robot':'red-3','initial_scenario':None,
                'command_continuity_passed':bool(continuity[i]),'planner_calls':calls[i],'actions':dict(counts[i]),
                'errors':errors[i],'usage':dict(tokens[i]),'latency_s':{'median':float(np.median(latencies[i])) if latencies[i] else None,
                     'p95':float(np.percentile(latencies[i],95)) if latencies[i] else None,'max':max(latencies[i],default=None)},
                'team_results':{team:{k:v for k,v in value.items() if isinstance(v,(str,int,float,bool))} for team,value in ((t,state['observations'][t]['teams'][t]) for t in ('red','blue'))},
                'event_counts':dict(events[i]),'execution_transitions':transitions[i],
                'final_bases':obs['bases'],'final_outposts':obs['outposts'],'final_robot':scene['robot'],
                'budget_limited':calls[i]>=max_calls,'replay':path.name,'replay_verification':verification,
                'replay_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        completed=bool(len(rows)==matches and all(r['finished'] and r['terminated'] and not r['truncated'] for r in rows))
        healthy=bool(completed and all(r['command_continuity_passed'] and not r['budget_limited'] and r['actions'].get('commanded',0)>0 for r in rows) and not api_failed and fatal_error is None)
        report={'format':'arena-llm-matches-v1','requested_model':model,'wall_seconds':time.monotonic()-start,
                'duration_s':420,'speed':speed,'simulated_action_hz':10,'missed_wall_deadlines':missed,
                'matches':rows,'all_matches_completed':completed,'healthy_control':healthy,'fatal_error':fatal_error,
                'arena':provenance(root),'policy_sha256':{k:m['policy_sha256'] for k,m in session.bank.metadata.items()},
                'execution_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'results_scope':'one learned red W3 versus two declared scripted blue opponents; normal spawn, economy, ammo and 420-second referee',
                'decisions_sha256':hashlib.sha256((out/'decisions.jsonl').read_bytes()).hexdigest()}
        dump(out/'report.json',report);trace.close();telemetry.close();session.close()
    print(json.dumps({'type':'llm_matches_complete','completed':completed,'healthy_control':healthy,
          'games':[{k:r[k] for k in ('seed','time_s','winner','reason','planner_calls','latency_s')} for r in rows]}),flush=True)
    return report
