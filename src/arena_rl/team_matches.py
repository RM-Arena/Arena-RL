"""Full 2027 team matches: strategic commander + RL execution for every red robot.

Red: an LLM (or the doctrine) chooses objectives; universal ground policies and
the aerial controller execute them at 10 Hz. Blue: a declared scripted team (the
same doctrine driving the chassis-aware expert). Each match runs in its own
process with an in-process Arena simulator; LLM calls are asynchronous, so the
robots keep executing their current orders during inference.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time
import numpy as np
from .bridge import configure,provenance
from .learning import dump

LINEUP_2027=[{'slot':1,'model':2711},{'slot':3,'model':2731},{'slot':4,'model':2732},{'slot':6,'model':2761},{'slot':7,'model':2771}]
PUBLIC_EVENTS=('destroy','respawn','outpost_destroyed','outpost_rebuilt','base_armor_opened','control_changed','launcher_unlocked',
               'launcher_installed','drone_strike','rune_activated','yellow_card','red_card','base_destroyed')


def public_events(events,team,time_s):
    rows=[]
    for e in events:
        kind=e.get('type')
        if kind not in PUBLIC_EVENTS:continue
        if kind in ('launcher_unlocked','launcher_installed') and e.get('team')!=team:continue
        rows.append({'t':round(time_s,1),'type':kind,**{k:e[k] for k in ('robot','team','target','killer','position','controller','previous_control','control') if k in e and isinstance(e[k],(str,int,float))}})
    return rows


class TeamSession:
    """One 2027 match: red orders from any commander, blue from the declared doctrine."""
    def __init__(self,*,policies=None,low_level='policy',duration=420,red_lineup=None,blue_lineup=None,record=True,arena_root=None,device='cpu'):
        from .bridge import configure as configure_arena
        root=configure_arena(arena_root)
        from rmuc_arena.env import ArenaEnv
        from .team_runtime import GroundPolicyBank,TeamController
        from .team_commander import Doctrine
        self.env=ArenaEnv(season='2027',duration=duration,lineups={'red':red_lineup or LINEUP_2027,'blue':blue_lineup or LINEUP_2027},compact=True,record=record)
        self.bank=GroundPolicyBank(policies,root,device) if low_level=='policy' else None
        self.red=TeamController('red',self.bank,low_level);self.blue=TeamController('blue',None,'expert');self.blue_doctrine=Doctrine()
        self.state=None;self.events=[];self.last_blue=-1e9

    def reset(self,seed):
        self.state=self.env.reset(seed=int(seed));self.events=[];self.last_blue=-1e9;self.blue_doctrine=type(self.blue_doctrine)()
        self.red.attach(self.state['observations']['red']);self.blue.attach(self.state['observations']['blue'])
        return self.red_obs

    @property
    def red_obs(self):return self.state['observations']['red']

    @property
    def finished(self):return bool(self.state['terminated'] or self.state['truncated'])

    def scene(self,history=()):
        from .team_commander import team_scene
        return team_scene(self.red,self.red_obs,self.events,history)

    def apply(self,orders,source):
        from .team_commander import apply_orders
        return apply_orders(self.red,self.red_obs,orders,source)

    def step(self):
        from .team_commander import apply_orders
        if self.finished:raise ValueError('Match finished; reset first')
        blue_obs=self.state['observations']['blue']
        if blue_obs['time']-self.last_blue>=1.:
            apply_orders(self.blue,blue_obs,self.blue_doctrine.decide(self.blue,blue_obs),'blue-doctrine');self.last_blue=blue_obs['time']
        self.state=self.env.step({'red':self.red.actions(self.red_obs),'blue':self.blue.actions(blue_obs)})
        self.events=(self.events+public_events(self.state['info'].get('events',[]),'red',self.red_obs['time']))[-40:]
        return self.state

    def result(self):
        final=self.red_obs;teams=final['teams']
        return {'winner':self.state['info']['winner'],'reason':self.state['info']['reason'],'time_s':final['time'],
                'score':{t:{k:teams[t].get(k) for k in ('control','attack_damage','kills','score')} for t in ('red','blue')},
                'structures':{f'{t}_{k}':final[k+'s'][t]['hp'] for t in ('red','blue') for k in ('outpost','base')},
                'blue_base_armor_open':bool(final['bases']['blue'].get('armor_open'))}

    def close(self):self.env.close()


def _play(index,options):
    from .team_commander import Doctrine
    from .team_planner import DoctrinePlanner,TeamLLMPlanner
    from .llm_planner import PlannerError
    seed=options['seed']+index;out=Path(options['output'])
    session=TeamSession(policies=options['policies'],low_level=options['low_level'],duration=options['duration'],
                        red_lineup=options['red_lineup'],blue_lineup=options['blue_lineup'],arena_root=options.get('arena_root'))
    red=session.red;red_doctrine=Doctrine()
    planner=TeamLLMPlanner(model=options['model'],base_url=options['base_url'],api_key_env=options['api_key_env'],timeout=options['timeout']) if options['commander']=='llm' else DoctrinePlanner()
    trace=(out/f'decisions-{seed}.jsonl').open('w')
    def log(row):trace.write(json.dumps(row,allow_nan=False,separators=(',',':'),default=str)+'\n');trace.flush()
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='team-llm')
    counts=Counter();latencies=[];tokens=Counter();errors=[];history=[];pending=None;calls=0
    last_request=-1e9;last_red_doctrine=-1e9;idle_since={};started=time.monotonic();period=.1/options['speed'] if options['speed']>0 else 0.
    deadline=time.monotonic();missed=0
    try:
        session.reset(seed)
        log({'type':'configuration','seed':seed,'commander':options['commander'],'model':getattr(planner,'model',None),'low_level':options['low_level'],
             'red_lineup':options['red_lineup'],'blue_lineup':options['blue_lineup'],'policies':session.bank.digests() if session.bank else None,'plan_interval_s':options['plan_interval']})
        previous_status={}
        while not session.finished:
            red_obs=session.red_obs;now=red_obs['time']
            feedback=red.feedback(red_obs)
            changed=any(previous_status.get(k)!=v.get('status') and v.get('status')!='running' for k,v in feedback.items())
            previous_status={k:v.get('status') for k,v in feedback.items()}
            if options['commander']=='doctrine':
                if now-last_red_doctrine>=1.:
                    orders=red_doctrine.decide(red,red_obs);results=session.apply(orders,'doctrine');last_red_doctrine=now
                    counts.update(r['action'] for r in results.values())
                    if orders:log({'type':'doctrine_orders','tick':red_obs['tick'],'orders':orders,'results':results})
            else:
                if pending is not None and pending[0].done():
                    future,basis,identity=pending;pending=None
                    try:
                        decision=future.result();latencies.append(decision.latency_s);tokens.update({k:decision.usage.get(k,0) for k in ('prompt_tokens','completion_tokens','total_tokens')})
                        age=now-basis['time_s']
                        if age>options['max_age']:raise PlannerError('stale_decision',details={'age_s':round(age,2)})
                        results=session.apply(decision.orders,'llm')
                        counts.update(r['action'] for r in results.values());counts['decisions']+=1
                        history.append({'t':basis['time_s'],'plan':decision.plan,'orders':{k:v['objective'] for k,v in decision.orders.items() if v['objective']!='continue'},
                                        'rejected':{k:v['reason'] for k,v in results.items() if v['action']=='rejected'}})
                        history=history[-6:]
                        log({'type':'decision','identity':identity,'basis_tick':basis['tick'],'applied_tick':red_obs['tick'],'latency_s':decision.latency_s,
                             'plan':decision.plan,'orders':decision.orders,'results':results,'usage':decision.usage,'response_model':decision.response_model})
                    except (PlannerError,ValueError) as error:
                        code=getattr(error,'code',str(error));errors.append({'t':now,'code':code,'details':getattr(error,'details',{})})
                        log({'type':'planner_error','identity':identity,'tick':red_obs['tick'],'code':code,'details':getattr(error,'details',{})})
                        results=session.apply(red_doctrine.decide(red,red_obs),'doctrine-fallback');counts['fallback_cycles']+=1
                        counts.update('fallback_'+r['action'] for r in results.values())
                        if isinstance(error,PlannerError) and not error.retryable:raise
                due=pending is None and calls<options['max_calls'] and (now-last_request>=options['plan_interval'] or changed and now-last_request>=1.5)
                if due:
                    scene=session.scene(history);calls+=1;identity=f'team-{seed}-{calls}'
                    log({'type':'request','identity':identity,'scene':scene})
                    if isinstance(planner,DoctrinePlanner):
                        from concurrent.futures import Future
                        future=Future();future.set_result(planner.decide_with(red,red_obs,scene))
                    else:future=pool.submit(planner.decide,scene)
                    pending=(future,scene,identity);last_request=now
                # Safety net: a robot idle for 8 s without a commander order gets a doctrine order.
                for k,v in feedback.items():
                    if v.get('status')=='running':idle_since.pop(k,None)
                    else:idle_since.setdefault(k,now)
                stale=[k for k,t in idle_since.items() if now-t>=8.]
                if stale:
                    proposal={k:v for k,v in red_doctrine.decide(red,red_obs).items() if k in stale}
                    if proposal:
                        results=session.apply(proposal,'doctrine-safety');counts.update('safety_'+r['action'] for r in results.values())
                        log({'type':'safety_orders','tick':red_obs['tick'],'orders':proposal,'results':results})
                        for k in proposal:idle_since[k]=now
            session.step()
            if period:
                deadline+=period;remaining=deadline-time.monotonic()
                if remaining>0:time.sleep(remaining)
                elif remaining<-period:missed+=1;deadline=time.monotonic()
        replay=out/f'match-{seed}.json';session.env.save_replay(replay)
        verification=None
        if options['verify']:
            from rmuc_arena.replays import verify_replay
            verification=verify_replay(replay)
        return {'seed':seed,**session.result(),'commander':options['commander'],'low_level':options['low_level'],
                'planner_calls':calls,'actions':dict(counts),'errors':errors[-20:],'error_count':len(errors),'usage':dict(tokens),
                'latency_s':{'median':float(np.median(latencies)) if latencies else None,'p95':float(np.percentile(latencies,95)) if latencies else None},
                'events':dict(Counter(e['type'] for e in session.events)),'missed_realtime_deadlines':missed,'wall_seconds':time.monotonic()-started,
                'replay':replay.name,'replay_sha256':hashlib.sha256(replay.read_bytes()).hexdigest(),'replay_verification':verification,'decisions_sha256':None}
    finally:
        pool.shutdown(wait=True,cancel_futures=True);trace.close();session.close()


def run_team_matches(*,policies,output,matches=2,seed=9100000,commander='llm',model=None,base_url=None,api_key_env='DEEPSEEK_API_KEY',
                     timeout=12.,plan_interval=6.,max_age=8.,max_calls=100,speed=1.,low_level='policy',duration=420,verify=True,
                     red_lineup=None,blue_lineup=None,arena_root=None):
    from .team_planner import MODEL,BASE_URL
    if commander not in ('llm','doctrine'):raise ValueError('commander must be llm or doctrine')
    if not 1<=matches<=8 or not 1<=duration<=420:raise ValueError('Invalid match count or duration')
    for name,value,lo,hi in (('plan_interval',plan_interval,1,60),('timeout',timeout,.1,120),('max_age',max_age,.5,60),('speed',speed,0,20)):
        if not math.isfinite(value) or not lo<=value<=hi:raise ValueError('Invalid '+name)
    out=Path(output)
    if (out/'report.json').exists():raise ValueError('Match output already exists; use a new directory')
    out.mkdir(parents=True,exist_ok=True);root=configure(arena_root)
    options={'policies':str(policies) if policies else None,'output':str(out),'seed':seed,'commander':commander,'model':model or MODEL,'base_url':base_url or BASE_URL,
             'api_key_env':api_key_env,'timeout':timeout,'plan_interval':plan_interval,'max_age':max_age,'max_calls':max_calls,'speed':speed,
             'low_level':low_level,'duration':duration,'verify':verify,'red_lineup':red_lineup or LINEUP_2027,'blue_lineup':blue_lineup or LINEUP_2027,'arena_root':str(root)}
    started=time.monotonic()
    context=multiprocessing.get_context('spawn')
    with context.Pool(matches) as pool:rows=pool.starmap(_play,[(i,options) for i in range(matches)])
    for row in rows:row['decisions_sha256']=hashlib.sha256((out/f"decisions-{row['seed']}.jsonl").read_bytes()).hexdigest()
    report={'format':'arena-team-matches-v1','commander':commander,'model':options['model'] if commander=='llm' else None,'low_level':low_level,
            'matches':rows,'wins':Counter(r['winner'] for r in rows),'wall_seconds':time.monotonic()-started,'arena':provenance(root),
            'scope':'red: commander + universal RL execution (aerial: explicit guidance); blue: declared doctrine with scripted expert controllers; normal 2027 referee'}
    report['wins']=dict(report['wins']);dump(out/'report.json',report)
    print(json.dumps({'type':'team_matches_complete','games':[{k:r[k] for k in ('seed','winner','reason','score','structures','planner_calls','error_count')} for r in rows]},default=str),flush=True)
    return report


def serve_team(*,policies=None,low_level='policy',seed=9200000,realtime=False,speed=1.,duration=420,max_age=8.,record=False,arena_root=None,stdin=None,stdout=None):
    """Provider-neutral JSONL service: an external LLM client commands the red team.

    Requests (one JSON object per line): {"op":"scene"}, {"op":"orders","plan":...,"orders":{...},"basis_tick":N},
    {"op":"advance","steps":k} (stepped mode), {"op":"reset","seed":s}, {"op":"save_replay","path":...}, {"op":"close"}.
    "scene" returns the harness scene and the strict issue_orders tool schema for the client's model.
    """
    import sys
    import threading
    from .team_planner import order_tool,validate_orders,SYSTEM_PROMPT
    stdin=stdin or sys.stdin;stdout=stdout or sys.stdout
    session=TeamSession(policies=policies,low_level=low_level,duration=duration,record=record,arena_root=arena_root)
    lock=threading.Lock();stop=threading.Event();history=[]
    def write(row):stdout.write(json.dumps(row,allow_nan=False,separators=(',',':'),default=str)+'\n');stdout.flush()
    def loop():
        period=.1/max(speed,1e-6);deadline=time.monotonic()
        while not stop.is_set():
            with lock:
                if not session.finished:session.step()
            deadline+=period;time.sleep(max(0.,deadline-time.monotonic()))
    try:
        session.reset(seed)
        write({'type':'ready','protocol':'arena-team-v1','mode':'realtime' if realtime else 'stepped','tick':session.red_obs['tick'],
               'robots':sorted(session.red.executors)+([session.red.aerial_id()] if session.red.aerial else []),'system_prompt':SYSTEM_PROMPT})
        runner=threading.Thread(target=loop,daemon=True) if realtime else None
        if runner:runner.start()
        for line in stdin:
            if not line.strip():continue
            try:
                request=json.loads(line,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Non-finite JSON')))
                if not isinstance(request,dict):raise ValueError('Request must be a JSON object')
                op=request.get('op')
                with lock:
                    if op=='close':write({'type':'closed'});break
                    if op=='scene':
                        scene=session.scene(history);write({'type':'scene','scene':scene,'tools':order_tool(scene),'finished':session.finished})
                    elif op=='orders':
                        scene=session.scene(history);basis=request.get('basis_tick',scene['tick'])
                        if isinstance(basis,bool) or not isinstance(basis,int) or not 0<=scene['tick']-basis<=max_age*20:raise ValueError('Stale or future basis_tick')
                        plan,orders=validate_orders(scene,{'plan':request.get('plan',''),'orders':request.get('orders')})
                        results=session.apply(orders,'client')
                        history.append({'t':scene['time_s'],'plan':plan,'orders':{k:v['objective'] for k,v in orders.items() if v['objective']!='continue'},
                                        'rejected':{k:v['reason'] for k,v in results.items() if v['action']=='rejected'}});history[:]=history[-6:]
                        write({'type':'applied','tick':session.red_obs['tick'],'results':results})
                    elif op=='advance':
                        if realtime:raise ValueError('advance is only for stepped mode')
                        steps=request.get('steps',1)
                        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=4200:raise ValueError('steps must be 1..4200')
                        for _ in range(steps):
                            if session.finished:break
                            session.step()
                        write({'type':'advanced','tick':session.red_obs['tick'],'time_s':session.red_obs['time'],'finished':session.finished,
                               **({'result':session.result()} if session.finished else {})})
                    elif op=='reset':session.reset(request.get('seed',seed));history.clear();write({'type':'reset','tick':session.red_obs['tick']})
                    elif op=='save_replay':write({'type':'replay',**session.env.save_replay(request['path'])})
                    else:raise ValueError('op must be scene, orders, advance, reset, save_replay or close')
            except (ValueError,TypeError,KeyError) as error:
                write({'type':'error','error':str(error)[:300]})
    finally:
        stop.set();session.close()
