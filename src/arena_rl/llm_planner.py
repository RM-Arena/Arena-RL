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
BASE_URL='https://api.deepseek.com/beta'
SYSTEM_PROMPT='''You are the tactical planner of ONE red infantry robot (red-3) in Arena.
Call exactly one supplied function. The frozen RL controller executes at 10 Hz while you think.
Optimize the real match outcome: preserve our base, capture and hold control positions, exploit enemy exposure,
attack enemy outpost and then unprotected base when feasible. At the 420 s timeout the referee ranks control degree
BEFORE damage, then surviving ground robot HP sum. We control fewer robots: a passive zero-score game loses.
Use defend_region at a named central card to capture its outpost position and keep holding it. These objectives
already route to and stand inside the card; use an interval long enough for travel and 5 s capture. With ammo, fire
at visible enemies while holding. Once the enemy outpost position is temporarily ours, stay on its card to maintain it.
Do not endlessly visit navigation nodes while armed. After supply choose a scoring/defense/combat objective; use
navigate only to resolve a specific obstruction. Check scoreboard and position controller/progress, not just damage. Use current battlefield state, not a fixed action cycle.
Only control red-3. Teammates/opponents have their own controllers. Never issue actuator commands.
Use continue_current while a useful running skill makes progress; replacing it restarts its timer.
Replan on danger, goal completion/failure or changed objectives. Do not repeatedly restart a blocked goal:
choose another waypoint, defend a reachable region or resupply. Only attack visible or recently seen
robot IDs; blue-1 is heavy/hero when deployed. Don't invent enemies or use hidden state.
The controlled model is stated in context. Model 305 has full-turn serial hips with mechanical self-righting.
If upset, recover and wait for stable stance before moving. A recovery subphase can temporarily suspend the current goal.
When low on HP, use retreat to a reachable safe point or resupply; ordinary navigate stops at its low-HP threshold.
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
Prefer integrated objectives: attack_structure, defend_region, pursue_target and resupply already include approach/navigation.
Do not split every objective into repeated short navigate/cancel calls. When waypoint catalogs are supplied,
choose only the supplied IDs; the host resolves them to safe public coordinates. Keep useful goals running
while traversing a ramp or detouring, even when straight-line distance temporarily increases.
Your recent decisions and execution feedback identify failures and changes; don't claim success yourself.'''

FIELDS={
 'navigate':['point','radius_m','timeout_s','low_hp_fraction'],
 'retreat':['point','radius_m','timeout_s'],
 'recover':['timeout_s'],
 'attack_structure':['target_id','preferred_range_m','timeout_s','low_hp_fraction'],
 'engage_target':['target_id','preferred_range_m','leash_radius_m','timeout_s','target_lost_timeout_s'],
 'pursue_target':['target_id','preferred_range_m','leash_radius_m','timeout_s','target_lost_timeout_s'],
 'defend_region':['center','region_radius_m','priority_ids','timeout_s','low_hp_fraction'],
 'resupply':['ammo_target','hp_target_fraction','budget_limit','timeout_s'],
 'escort':['target_id','point','follow_distance_m','required_coverage','escort_duration_s','leash_radius_m','engage_on_contact','timeout_s'],
 'patrol':['waypoints','patrol_loops','dwell_s','engage_on_contact','timeout_s'],
 'fly_slope':['lane_team','flight_direction','landing_hold_s','timeout_s'],
 'install_launcher':['timeout_s'],
}
DESCRIPTIONS={
 'retreat':'Withdraw to a safe field point or own supply if point is null; allowed at low HP, using learned mobility.',
 'recover':'Settle and self-right the serial-leg 305 using its physical hip firmware and learned stabilization; no teleport.',
 'navigate':'Move to a field point; arrive slowly. Full-map route guidance is available.',
 'attack_structure':'Attack specified enemy outpost/base until destroyed or timeout. Protected bases cannot be attacked.',
 'engage_target':'Engage a specified visible enemy robot ID at local range.',
 'pursue_target':'Pursue a specified visible/recently seen robot within a starting-position leash.',
 'defend_region':'Approach and hold a region for the requested interval, firing at visible enemies. Named central cards capture the corresponding outpost position through actual RFID contact. Can hold without ammo. Center may also be own base.',
 'resupply':'Reach own supply card, buy local ammunition within shared budget, wait for passive HP healing.',
 'escort':'Follow a deployed ally; optional ally destination or escort duration. Never controls that ally.',
 'patrol':'Visit all ordered waypoints, dwell, complete all loops; can optionally shoot visible threats.',
 'fly_slope':'Cross the real 2026 gap and settle. Reverse needs an earned forward crossing this episode.',
 'install_launcher':'Heavy only: dock at the own workbench, fit the unlocked modular 42 mm launcher and fire its confirmation shot.',
}


def tools_for(skills,field=None,waypoints=None,regions=None):
    variants={v['properties']['skill']['const']:v for v in command_schema()['oneOf']}
    tools=[]
    field=field or {'width':100.,'height':100.,'margin':0.}
    def coordinate():
        return {'type':'object','properties':{'x':{'type':'number','minimum':field.get('margin',0.),'maximum':field['width']-field.get('margin',0.)},'y':{'type':'number','minimum':field.get('margin',0.),'maximum':field['height']-field.get('margin',0.)}},'required':['x','y'],'additionalProperties':False}
    def supported(value):
        if isinstance(value,dict):return {k:supported(v) for k,v in value.items() if k not in ('minItems','maxItems','minLength','maxLength','uniqueItems')}
        if isinstance(value,list):return [supported(v) for v in value]
        return value
    for skill in skills:
        if skill not in FIELDS:raise ValueError('Unsupported planner tool')
        variant=variants[skill];properties={}
        for name in FIELDS[skill]:
            schema=copy.deepcopy(variant['properties'][name])
            if name in ('point','center'):
                catalog=regions if name=='center' else waypoints
                schema={'type':'string','enum':list(catalog)} if catalog else coordinate()
            if name=='waypoints':schema={'type':'array','items':{'type':'string','enum':list(waypoints)} if waypoints else coordinate()}
            if name not in variant['required']:
                schema={'anyOf':[schema,{'type':'null'}]}
            properties[name]=schema
        properties['intent']={'type':'string','pattern':'^.{0,160}$'}
        tools.append({'type':'function','function':{'name':skill,'strict':True,'description':DESCRIPTIONS[skill],
            'parameters':supported({'type':'object','properties':properties,'required':list(properties),'additionalProperties':False})}})
    tools.extend({'type':'function','function':{'name':name,'strict':True,'description':description,
        'parameters':{'type':'object','properties':{'intent':{'type':'string','pattern':'^.{0,160}$'}},'required':['intent'],'additionalProperties':False}}}
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
    coordinates:dict|None=None

    def command(self,identity,max_age=5.):
        if self.tool in ('continue_current','cancel_current'):return None
        args={k:v for k,v in self.arguments.items() if k!='intent' and v is not None}
        for name in ('point','center'):
            if isinstance(args.get(name),str):
                if not self.coordinates or args[name] not in self.coordinates:raise ValueError('Unknown waypoint ID')
                args[name]=self.coordinates[args[name]]
            if name in args and isinstance(args[name],dict):
                if set(args[name])!={'x','y'}:raise ValueError('Coordinate needs x and y')
                args[name]=[args[name]['x'],args[name]['y']]
        if 'waypoints' in args:
            args['waypoints']=[self.coordinates[p] if isinstance(p,str) and self.coordinates and p in self.coordinates else [p['x'],p['y']] if isinstance(p,dict) and set(p)=={'x','y'} else p for p in args['waypoints']]
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
        field={**context.get('field',{'width':100.,'height':100.}),'margin':scene.get('robot',{}).get('radius',.35) if 'field' in context else 0.}
        waypoints=context.get('safe_waypoints');regions=context.get('defense_regions')
        tools=tools_for(skills,field,waypoints,regions)
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
                time.monotonic()-start,scene['tick'],call['id'],{**(waypoints or {}),**(regions or {})} or None)
            command=decision.command('validated')
            for key in ('point','center'):
                value=arguments.get(key)
                if value is not None:
                    catalog=regions if key=='center' else waypoints
                    if catalog:
                        if not isinstance(value,str) or value not in catalog:raise ValueError('Unknown or unsafe waypoint ID')
                    elif not isinstance(value,dict) or set(value)!={'x','y'}:raise ValueError('Wire coordinate must have exactly x and y')
            if command:
                for point in [command.get('point'),command.get('center'),*command.get('waypoints',[])]:
                    if point and not (field['margin']<=point[0]<=field['width']-field['margin'] and field['margin']<=point[1]<=field['height']-field['margin']):raise ValueError('Coordinate outside current robot field bounds')
            return decision
        except (KeyError,IndexError,TypeError,ValueError,StopIteration) as error:
            raise PlannerError('invalid_tool_call',True,details={'validation_error':str(error)[:240]}) from None


def apply_decision(session,decision,env_id,identity,max_age=5.):
    scene=session.observe(env_id)
    age=(scene['tick']-decision.basis_tick)*.05
    if age<0 or age>max_age or decision.basis_tick<getattr(session.batch.executors[env_id],'last_basis_tick',-1):raise ValueError('Stale planner decision')
    if decision.tool=='continue_current':return {'action':'continued','feedback':scene['feedback']}
    if decision.tool=='cancel_current':return {'action':'cancelled','feedback':session.cancel(env_id)}
    command=decision.command(identity,max_age)
    current=session.batch.executors[env_id]
    if current.status=='running':
        def semantic(c):return {k:v for k,v in c.items() if k not in ('id','basis_tick','max_staleness_s')}
        if semantic(command)==semantic(current.command.json()):return {'action':'same_goal_preserved','feedback':scene['feedback']}
    response=session.command(command,env_id)
    return {'action':'commanded','command':command,**response}
