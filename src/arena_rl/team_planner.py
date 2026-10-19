"""Strategic LLM commander over the whole team, with a strict single-tool contract.

The model sees the harness scene (team_commander.team_scene) and calls exactly one
function, issue_orders, with one entry per controlled robot. Per-robot enums only
contain objectives, targets and places the harness found feasible at the basis
tick, so a well-formed call is executable; the harness still re-validates at
application time. Transport: OpenAI-compatible chat completions (DeepSeek's strict
function-call beta endpoint by default). Credentials stay in the environment.
"""
from dataclasses import dataclass
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from .llm_planner import NoRedirect,PlannerError,strict_json
from .team_commander import AERIAL_OBJECTIVES,GROUND_OBJECTIVES,PATROLS,PLACES

MODEL='deepseek-flash'
BASE_URL='https://api.deepseek.com/beta'
SYSTEM_PROMPT='''You are the strategic commander of one RoboMaster 2027 (RMUC) team in the Arena simulator.
Each call, give every robot ONE high-level objective via the issue_orders function; trained RL policies execute
movement, aiming and firing at 10 Hz while you think. Use "continue" to keep a useful running order: replacing
an order restarts it. Re-plan when orders finish or fail, robots die/respawn, structures change, or threats appear.

How the 2027 match is decided at 420 s: control degree first, then attack damage, then surviving ground HP.
Control: holding BOTH outpost positions gives +1/s, holding BOTH base positions +3/s. An outpost position = its
outpost + central-highland card; a base position = base + fortress card. Standing 5 s on a card captures it
temporarily; if the temporary holder leaves for 5 s, control returns to the other team.

Key chains (the harness lists rule status and only offers feasible objectives, with reasons for blocked ones):
- Enemy base is invulnerable while the enemy FULLY controls its outpost position: destroy their outpost
  (attack_outpost / drone_strike_outpost) and/or capture_enemy_central and keep someone on that card.
- After 180 s, while we control the enemy outpost position, occupy_enemy_fortress for 10 s opens the enemy base armor.
- push_base only once the enemy base is unprotected and its armor is open. 42 mm heavy shots and aerial rams deal 200.
- Our fortress card only works after we lose our outpost position (defense buff + reserve ammo).
- The heavy (slot 1) cannot fire until its launcher unlocks (~300 s, earlier if we trail) and install_launcher finishes.
  Before that it can hold cards (hold_own_central / capture_enemy_central) and absorb fire.
- Infantry start with 0 ammo: resupply first, and again when empty. Sentry starts armed. Ammo caps are team-wide.
- pursue_enemy chases a specific visible/recently-seen robot (e.g. the enemy heavy, role "heavy"); engage_enemy fights locally.
- defend_base when enemies are near our base; retreat or resupply at low HP. Dead robots wait for respawn ("continue").
- The aerial hangar rams armor with drones; batteries do not recharge, so strike when it matters.
Avoid sending every robot to the same place; combine control-card holding with damage. Never invent ids.
"plan" is a one-sentence team intent; "intent" per robot is a short label, not a reasoning transcript.'''


def order_tool(scene):
    robots={}
    for identity,info in scene['robots'].items():
        allowed=list(dict.fromkeys(['continue']+[o for o in info['allowed'] if o!='continue']))
        targets=sorted({e['id'] for e in scene['visible_enemies']}|{e['id'] for e in scene['recent_enemies']}|
                       ({r for r in scene['robots'] if r!=identity and scene['robots'][r]['role']!='aerial'} if 'escort' in allowed else set()))
        aerial=info['role']=='aerial'
        # Every robot's order has the same four fields so models emit one uniform shape.
        properties={'objective':{'type':'string','enum':allowed},'intent':{'type':'string','pattern':'^.{0,120}$'}}
        properties['target']={'anyOf':[{'type':'string','enum':targets},{'type':'null'}]} if targets and not aerial else {'type':'null'}
        properties['place']={'anyOf':[{'type':'string','enum':sorted(PLACES)+sorted(PATROLS)},{'type':'null'}]} if not aerial else {'type':'null'}
        robots[identity]={'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    parameters={'type':'object','properties':{'plan':{'type':'string','pattern':'^.{0,240}$'},
        'orders':{'type':'object','properties':robots,'required':list(robots),'additionalProperties':False}},
        'required':['plan','orders'],'additionalProperties':False}
    description='Assign one objective per robot. Objectives: '+'; '.join(f'{k}: {v}' for k,v in {**GROUND_OBJECTIVES,**AERIAL_OBJECTIVES}.items())
    return [{'type':'function','function':{'name':'issue_orders','strict':True,'description':description[:4000],'parameters':parameters}}]


@dataclass
class TeamDecision:
    plan:str
    orders:dict
    basis_tick:int
    requested_model:str
    response_model:str
    response_id:str
    usage:dict
    latency_s:float


class TeamLLMPlanner:
    def __init__(self,*,model=MODEL,base_url=BASE_URL,api_key_env='DEEPSEEK_API_KEY',timeout=12.,max_tokens=900,temperature=.2):
        parsed=urllib.parse.urlsplit(base_url)
        if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost','::1')):
            raise ValueError('LLM endpoint requires HTTPS (local test servers may use HTTP)')
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:raise ValueError('Invalid LLM base URL')
        if not math.isfinite(timeout) or not .1<=timeout<=120 or not 64<=max_tokens<=8192:raise ValueError('Invalid inference limits')
        self.key=os.environ.get(api_key_env)
        if not self.key and api_key_env=='DEEPSEEK_API_KEY':self.key=os.environ.get('DeepSeek_API_Key')
        if not self.key:raise PlannerError('missing_api_key_environment_variable')
        self.model=model;self.endpoint=base_url.rstrip('/')+'/chat/completions';self.timeout=timeout;self.max_tokens=max_tokens;self.temperature=temperature
        self.opener=urllib.request.build_opener(NoRedirect())

    def decide(self,scene):
        tools=order_tool(scene)
        body={'model':self.model,'messages':[{'role':'system','content':SYSTEM_PROMPT},
              {'role':'user','content':json.dumps(scene,separators=(',',':'),allow_nan=False)}],
              'tools':tools,'tool_choice':'required','parallel_tool_calls':False,'thinking':{'type':'disabled'},
              'max_tokens':self.max_tokens,'temperature':self.temperature,'stream':False}
        request=urllib.request.Request(self.endpoint,json.dumps(body,allow_nan=False).encode(),method='POST',
            headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'Arena-RL-team-commander/0.6'})
        start=time.monotonic()
        try:
            with self.opener.open(request,timeout=self.timeout) as response:data=response.read(4_000_001)
            if len(data)>4_000_000:raise PlannerError('response_too_large')
            result=strict_json(data)
        except urllib.error.HTTPError as error:
            raise PlannerError('provider_http_error',error.code in (408,409,429) or error.code>=500,error.code) from None
        except (urllib.error.URLError,TimeoutError,OSError):raise PlannerError('provider_network_timeout',True) from None
        except (ValueError,UnicodeDecodeError):raise PlannerError('invalid_provider_json',True) from None
        try:
            choice=result['choices'][0];calls=choice['message'].get('tool_calls',[])
            if len(calls)!=1 or calls[0]['function']['name']!='issue_orders':raise ValueError('Expected exactly one issue_orders call')
            arguments=strict_json(calls[0]['function']['arguments'])
            return TeamDecision(*validate_orders(scene,arguments,tools),scene['tick'],self.model,result.get('model',''),result.get('id',''),result.get('usage',{}),time.monotonic()-start)
        except (KeyError,IndexError,TypeError,ValueError) as error:
            raise PlannerError('invalid_tool_call',True,details={'validation_error':str(error)[:240]}) from None


def validate_orders(scene,arguments,tools=None):
    """Check a tool call against the scene's own enums; returns (plan, orders)."""
    tools=tools or order_tool(scene);schema=tools[0]['function']['parameters']
    if not isinstance(arguments,dict) or set(arguments)!={'plan','orders'}:raise ValueError('issue_orders needs plan and orders')
    if not isinstance(arguments['plan'],str) or len(arguments['plan'])>240:raise ValueError('Invalid plan text')
    orders=arguments['orders'];robots=schema['properties']['orders']['properties']
    if not isinstance(orders,dict) or set(orders)!=set(robots):raise ValueError('Orders must name every controlled robot exactly once')
    clean={}
    for identity,order in orders.items():
        spec=robots[identity]['properties']
        if isinstance(order,dict):order={'intent':'',**order}
        if not isinstance(order,dict) or set(order)!=set(spec):raise ValueError(f'Order shape mismatch for {identity}: {sorted(order) if isinstance(order,dict) else type(order).__name__}')
        if order['objective'] not in spec['objective']['enum']:raise ValueError(f'{identity}: objective not offered')
        for key in ('target','place'):
            if key in spec and order[key] is not None:
                options=next((s['enum'] for s in spec[key].get('anyOf',[]) if 'enum' in s),[])
                if order[key] not in options:raise ValueError(f'{identity}: {key} not offered')
        if not isinstance(order['intent'],str) or len(order['intent'])>120:raise ValueError('Invalid intent label')
        clean[identity]={'objective':order['objective'],'target':order.get('target'),'place':order.get('place'),'intent':order['intent']}
    return arguments['plan'],clean


class DoctrinePlanner:
    """Offline stand-in with the planner interface; used for tests and LLM-free runs."""
    def __init__(self):
        from .team_commander import Doctrine
        self.doctrine=Doctrine();self.model='doctrine'

    def decide_with(self,controller,observation,scene):
        orders=self.doctrine.decide(controller,observation)
        full={r:{'objective':'continue','target':None,'place':None,'intent':''} for r in scene['robots']}
        full.update({k:{**v,'intent':'doctrine'} for k,v in orders.items()})
        return TeamDecision('doctrine',full,scene['tick'],'doctrine','doctrine','',{},0.)
