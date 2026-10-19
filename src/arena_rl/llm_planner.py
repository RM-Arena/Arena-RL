"""Tool-only tactical inference; transport never receives credentials in prompts/logs."""
import copy
from dataclasses import dataclass
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from .commands import Command,command_schema

MODEL='deepseek-flash'
BASE_URL='https://api.deepseek.com'
SYSTEM_PROMPT='''You are the tactical planner of ONE red W3 infantry robot (red-3) in Arena.
Call exactly one supplied function. The frozen RL controller executes at 10 Hz while you think.
Optimize the real match outcome: preserve our base, exploit enemy exposure, attack enemy outpost
and then unprotected base when feasible. Use current battlefield state, not a fixed action cycle.
Only control red-3. Teammates/opponents have their own controllers. Never issue actuator commands.
Use continue_current while a useful running skill makes progress; replacing it restarts its timer.
Replan on danger, goal completion/failure or changed objectives. Do not repeatedly restart a blocked goal:
choose another waypoint, defend a reachable region or resupply. Only attack visible or recently seen
robot IDs; blue-1 is heavy/hero when deployed. Don't invent enemies or use hidden state.
Normal 2027 infantry starts with zero ammo: resupply locally before combat and after ammunition runs out.
Ammo purchase/healing uses actual contact, coins and team caps. Budget is shared with teammates.
Ammunition team caps do not reset on death. When capped and empty, patrol/defend without firing.
A dead/unpowered robot must wait via continue_current for the normal referee respawn; do not spam goals.
Prioritize emergency defense near our base and low-HP supply when it is reachable. Target base attacks
are protected while the enemy outpost position remains fully controlled; attack the outpost first.
Navigation can span the field using SDK routes; combat intercepts and patrol/escort are local courses.
Keep combat range around 2–3.5 m. Patrol needs at least two distinct reachable points. Escort requires
another deployed mobile ally. Fly-slope is only on 2026; reverse requires a completed same-lane forward
crossing this episode. The current tools already filter unavailable seasons/rosters/policies.
Changing commands never restores health, ammunition or position. Do not end/reset the match.
The intent field is one short action label, not a reasoning transcript. Use null for optional defaults.
Your recent decisions and execution feedback identify failures and changes; don't claim success yourself.'''

FIELDS={
 'navigate':['point','radius_m','timeout_s'],
 'attack_structure':['target_id','preferred_range_m','timeout_s','low_hp_fraction'],
 'engage_target':['target_id','preferred_range_m','leash_radius_m','timeout_s','target_lost_timeout_s'],
 'pursue_target':['target_id','preferred_range_m','leash_radius_m','timeout_s','target_lost_timeout_s'],
 'defend_region':['center','region_radius_m','priority_ids','timeout_s','low_hp_fraction'],
 'resupply':['ammo_target','hp_target_fraction','budget_limit','timeout_s'],
 'escort':['target_id','point','follow_distance_m','required_coverage','escort_duration_s','leash_radius_m','engage_on_contact','timeout_s'],
 'patrol':['waypoints','patrol_loops','dwell_s','engage_on_contact','timeout_s'],
 'fly_slope':['lane_team','flight_direction','landing_hold_s','timeout_s'],
}
DESCRIPTIONS={
 'navigate':'Move to a field point; arrive slowly. Full-map route guidance is available.',
 'attack_structure':'Attack specified enemy outpost/base until destroyed or timeout. Protected bases cannot be attacked.',
 'engage_target':'Engage a specified visible enemy robot ID at local range.',
 'pursue_target':'Pursue a specified visible/recently seen robot within a starting-position leash.',
 'defend_region':'Defend a region for the requested interval. Optional enemy ID priorities; center may be own base.',
 'resupply':'Reach own supply card, buy local ammunition within shared budget, wait for passive HP healing.',
 'escort':'Follow a deployed ally; optional ally destination or escort duration. Never controls that ally.',
 'patrol':'Visit all ordered waypoints, dwell, complete all loops; can optionally shoot visible threats.',
 'fly_slope':'Cross the real 2026 gap and settle. Reverse needs an earned forward crossing this episode.',
}


def tools_for(skills):
    variants={v['properties']['skill']['const']:v for v in command_schema()['oneOf']}
    tools=[]
    for skill in skills:
        if skill not in FIELDS:raise ValueError('Unsupported planner tool')
        variant=variants[skill];properties={}
        for name in FIELDS[skill]:
            schema=copy.deepcopy(variant['properties'][name])
            if name not in variant['required']:
                schema={'anyOf':[schema,{'type':'null'}]}
            properties[name]=schema
        properties['intent']={'type':'string','maxLength':160}
        tools.append({'type':'function','function':{'name':skill,'description':DESCRIPTIONS[skill],
            'parameters':{'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}}})
    tools.extend({'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':{'intent':{'type':'string','maxLength':160}},'required':['intent'],'additionalProperties':False}}}
        for name,description in [('continue_current','Keep executing current RL skill; if idle/dead, wait for a future decision.'),
                                 ('cancel_current','Stop the current robot skill and wait at rest.')])
    return tools


class PlannerError(RuntimeError):
    def __init__(self,code,retryable=False,status=None,details=None):
        self.code,self.retryable,self.status=code,retryable,status
        self.details=details or {}
        super().__init__(code+(f' (HTTP {status})' if status else ''))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise PlannerError('endpoint_redirect_refused')


def strict_json(value):
    def pairs(items):
        result={}
        for key,item in items:
            if key in result:raise ValueError('Duplicate JSON property')
            result[key]=item
        return result
    return json.loads(value,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Non-finite JSON')))


@dataclass
class Decision:
    tool:str
    arguments:dict
    requested_model:str
    response_model:str
    response_id:str
    usage:dict
    latency_s:float
    basis_tick:int
    call_id:str

    def command(self,identity,max_age=5.):
        if self.tool in ('continue_current','cancel_current'):return None
        args={k:v for k,v in self.arguments.items() if k!='intent' and v is not None}
        return Command.parse(dict(args,id=identity,skill=self.tool,basis_tick=self.basis_tick,max_staleness_s=max_age)).json()


class DeepSeekPlanner:
    def __init__(self,*,model=MODEL,base_url=BASE_URL,api_key_env='DEEPSEEK_API_KEY',timeout=8.,max_tokens=768):
        parsed=urllib.parse.urlsplit(base_url)
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost','::1')):
            raise ValueError('LLM endpoint requires HTTPS (local test servers may use HTTP)')
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Invalid LLM base URL')
        if not math.isfinite(timeout) or not .1<=timeout<=60 or not 64<=max_tokens<=4096:raise ValueError('Invalid inference limits')
        self.key=os.environ.get(api_key_env)
        if not self.key and api_key_env=='DEEPSEEK_API_KEY':self.key=os.environ.get('DeepSeek_API_Key')
        if not self.key:raise PlannerError('missing_api_key_environment_variable')
        self.model=model;self.endpoint=base_url.rstrip('/')+'/chat/completions';self.timeout=timeout;self.max_tokens=max_tokens
        self.opener=urllib.request.build_opener(NoRedirect())

    def decide(self,scene,skills,context,history=()):
        tools=tools_for(skills)
        body={'model':self.model,'messages':[{'role':'system','content':SYSTEM_PROMPT+'\nPublic context:\n'+json.dumps(context,separators=(',',':'))},
             {'role':'user','content':json.dumps({'scene':scene,'recent_decisions':list(history)[-8:]},separators=(',',':'),allow_nan=False)}],
              'tools':tools,'tool_choice':'required','parallel_tool_calls':False,'thinking':{'type':'disabled'},
              'max_tokens':self.max_tokens,'temperature':.15,'stream':False}
        request=urllib.request.Request(self.endpoint,json.dumps(body,allow_nan=False).encode(),method='POST',
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'Arena-RL-planner/0.4'})
        start=time.monotonic()
        try:
            with self.opener.open(request,timeout=self.timeout) as response:
                data=response.read(4_000_001)
            if len(data)>4_000_000:raise PlannerError('response_too_large')
            result=strict_json(data)
        except urllib.error.HTTPError as error:
            # Provider bodies may echo credentials. Never retain them or raw headers.
            raise PlannerError('provider_http_error',error.code in (408,409,429) or error.code>=500,error.code) from None
        except (urllib.error.URLError,TimeoutError,OSError):raise PlannerError('provider_network_timeout',True) from None
        except (ValueError,UnicodeDecodeError):raise PlannerError('invalid_provider_json',True) from None
        try:
            choice=result['choices'][0];calls=choice['message'].get('tool_calls',[])
            if choice.get('finish_reason')!='tool_calls' or len(calls)!=1:raise ValueError(f'Expected one completed tool; finish={choice.get("finish_reason")}, count={len(calls)}')
            call=calls[0];name=call['function']['name'];arguments=strict_json(call['function']['arguments'])
            definition=next(t['function'] for t in tools if t['function']['name']==name)
            expected=set(definition['parameters']['properties'])
            if not isinstance(arguments,dict) or set(arguments)!=expected:raise ValueError('Tool argument shape mismatch')
            if not isinstance(arguments['intent'],str) or len(arguments['intent'])>160:raise ValueError('Invalid intent label')
            decision=Decision(name,arguments,self.model,result.get('model',''),result['id'],result.get('usage',{}),
                time.monotonic()-start,scene['tick'],call['id'])
            decision.command('validated')  # Validate numeric limits/targets before returning.
            return decision
        except (KeyError,IndexError,TypeError,ValueError,StopIteration) as error:
            raise PlannerError('invalid_tool_call',True,details={'validation_error':str(error)[:240]}) from None


def apply_decision(session,decision,env_id,identity,max_age=5.):
    scene=session.observe(env_id)
    age=(scene['tick']-decision.basis_tick)*.05
    if age<0 or age>max_age:raise ValueError('Stale planner decision')
    if decision.tool=='continue_current':return {'action':'continued','feedback':scene['feedback']}
    if decision.tool=='cancel_current':return {'action':'cancelled','feedback':session.cancel(env_id)}
    command=decision.command(identity,max_age)
    current=session.batch.executors[env_id]
    if current.status=='running':
        def semantic(c):return {k:v for k,v in c.items() if k not in ('id','basis_tick','max_staleness_s')}
        if semantic(command)==semantic(current.command.json()):return {'action':'same_goal_preserved','feedback':scene['feedback']}
    response=session.command(command,env_id)
    return {'action':'commanded','command':command,**response}
