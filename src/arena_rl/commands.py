"""Strict, provider-independent tactical commands for one robot."""
from dataclasses import asdict, dataclass
import math
import re

PROTOCOL = "arena-tactics-v1"
BASE_SKILLS = ("navigate", "attack_structure", "engage_target", "pursue_target", "defend_region")
NEW_SKILLS = ("resupply", "escort", "patrol", "fly_slope")
MOBILITY_SKILLS = ('retreat','recover')
SKILLS = BASE_SKILLS + NEW_SKILLS + MOBILITY_SKILLS


def number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum},{maximum}]")
    return float(value)


@dataclass(frozen=True)
class Command:
    id: str
    skill: str
    target_id: str | None = None
    point: tuple | None = None
    center: tuple | None = None
    radius_m: float = .4
    preferred_range_m: float = 2.5
    region_radius_m: float = 3.
    leash_radius_m: float = 8.
    timeout_s: float = 30.
    target_lost_timeout_s: float = 2.
    low_hp_fraction: float = .15
    damage_goal: float | None = None
    priority_ids: tuple = ()
    basis_tick: int | None = None
    max_staleness_s: float = 3.
    ammo_target: int = 100
    hp_target_fraction: float = .9
    budget_limit: float = 500.
    waypoints: tuple = ()
    patrol_loops: int = 1
    dwell_s: float = .4
    follow_distance_m: float = 1.2
    escort_duration_s: float = 10.
    required_coverage: float = .75
    engage_on_contact: bool = False
    lane_team: str = 'red'
    flight_direction: str = 'forward'
    landing_hold_s: float = .3

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value)-set(cls.__dataclass_fields__):
            raise ValueError("Command must contain only documented fields")
        value=dict(value)
        if not isinstance(value.get('id'),str) or not re.fullmatch(r'[\w.-]{1,64}',value['id']):
            raise ValueError('Command id must be 1..64 letters, digits, underscores, dots or hyphens')
        skill=value.get('skill')
        if skill not in SKILLS:raise ValueError(f'Unsupported skill; choose {SKILLS}')
        for key in ('point','center'):
            if key in value and value[key] is not None:
                p=value[key]
                if not isinstance(p,(list,tuple)) or len(p)!=2:raise ValueError(f'{key} needs two coordinates')
                value[key]=tuple(number(v,key,0,100) for v in p)
        defaults=cls(value['id'],skill)
        limits={'radius_m':(.15,3),'preferred_range_m':(.8,8),'region_radius_m':(.5,10),
                'leash_radius_m':(.5,35),'timeout_s':(.1,420),'target_lost_timeout_s':(0,10),
                'low_hp_fraction':(0,1),'max_staleness_s':(0,30),'hp_target_fraction':(.1,1),
                'budget_limit':(0,5000),'dwell_s':(0,30),'follow_distance_m':(.6,5),
                'escort_duration_s':(.5,420),'required_coverage':(0,1),'landing_hold_s':(.1,3)}
        for key,(lo,hi) in limits.items():value[key]=number(value.get(key,getattr(defaults,key)),key,lo,hi)
        if 'damage_goal' in value and value['damage_goal'] is not None:value['damage_goal']=number(value['damage_goal'],'damage_goal',1,10000)
        if skill=='navigate' and value.get('point') is None:raise ValueError('navigate requires point')
        if skill=='defend_region' and value.get('center') is None:raise ValueError('defend_region requires center')
        if skill in ('attack_structure','engage_target','pursue_target','escort') and not isinstance(value.get('target_id'),str):
            raise ValueError('This skill requires target_id')
        if skill=='attack_structure' and not re.fullmatch(r'(red|blue)_(outpost|base)',value['target_id']):
            raise ValueError('Structure target must be red/blue_outpost/base')
        if skill in ('engage_target','pursue_target','escort') and not re.fullmatch(r'(red|blue)-[1-9][0-9]*',value['target_id']):
            raise ValueError('Robot target must be a team-slot id')
        priorities=value.get('priority_ids',())
        if not isinstance(priorities,(list,tuple)) or len(priorities)>32 or any(not isinstance(p,str) or not re.fullmatch(r'(red|blue)-[1-9][0-9]*',p) for p in priorities):
            raise ValueError('priority_ids must contain up to 32 robot ids')
        if len(set(priorities))!=len(priorities):raise ValueError('Duplicate priority target')
        value['priority_ids']=tuple(priorities)
        tick=value.get('basis_tick')
        if tick is not None and (isinstance(tick,bool) or not isinstance(tick,int) or tick<0):raise ValueError('basis_tick must be a nonnegative integer')
        for name,lo,hi in (('ammo_target',0,1000),('patrol_loops',1,100)):
            n=value.get(name,getattr(defaults,name))
            if isinstance(n,bool) or not isinstance(n,int) or not lo<=n<=hi:raise ValueError(f'Invalid {name}')
            value[name]=n
        if not isinstance(value.get('engage_on_contact',False),bool):raise ValueError('engage_on_contact must be boolean')
        if value.get('lane_team','red') not in ('red','blue'):raise ValueError('Invalid lane_team')
        if value.get('flight_direction','forward') not in ('forward','reverse'):raise ValueError('Invalid flight_direction')
        points=value.get('waypoints',())
        if not isinstance(points,(list,tuple)) or len(points)>32:raise ValueError('waypoints must be an array of at most 32 points')
        value['waypoints']=tuple(tuple(number(x,'waypoint',0,100) for x in p) if isinstance(p,(list,tuple)) and len(p)==2 else (_ for _ in ()).throw(ValueError('Each waypoint needs two coordinates')) for p in points)
        if skill=='patrol' and len(value['waypoints'])<2:raise ValueError('patrol requires at least two waypoints')
        allowed={'navigate' :{'point'},'defend_region':{'center','priority_ids'},
                 'attack_structure':{'target_id'},'engage_target':{'target_id'},'pursue_target':{'target_id'},
                 'resupply':{'point'},'escort':{'target_id','point','priority_ids'},'patrol':{'waypoints','priority_ids'},'fly_slope':set(),'retreat':{'point'},'recover':set()}[skill]
        for field in ('point','center','target_id','priority_ids','waypoints'):
            if value.get(field) and field not in allowed:raise ValueError(f'{field} is not valid for {skill}')
        return cls(**value)

    @property
    def policy_family(self):
        return 'mobility' if self.skill in MOBILITY_SKILLS else 'flight' if self.skill=='fly_slope' else 'support' if self.skill in ('resupply','escort','patrol') else 'navigation' if self.skill=='navigate' else 'defense' if self.skill=='defend_region' else 'combat'

    def json(self):
        value={k:v for k,v in asdict(self).items() if v is not None}
        if self.skill not in ('defend_region','escort','patrol'):value.pop('priority_ids',None)
        if self.skill!='patrol':value.pop('waypoints',None)
        for key in ('point','center','priority_ids'):
            if key in value:value[key]=list(value[key])
        if 'waypoints' in value:value['waypoints']=[list(p) for p in value['waypoints']]
        return value


def capabilities():
    return {'protocol':PROTOCOL,'robot_model':3,'skills':list(SKILLS),'command_json_schema':command_schema(),
            'policy_families':['navigation','combat','defense','support','flight','mobility'],'action_hz':10,
            'target_selection':'explicit robot id; defense accepts ordered priority_ids',
            'aim_mode':'SDK ballistic assistance with learned residuals',
            'trained_profiles':{'combat_range_m':[2.,3.5],'defense_radius_m':2.5,'season':'2027',
                                'enemy_models':[2711,3,4],'calibrated_success_probability':False},
            'command_fields':list(Command.__dataclass_fields__),
            'feedback_statuses':['idle','running','succeeded','failed','cancelled'],
            'skill_seasons':{s:['2026'] if s=='fly_slope' else ['2026','2027'] if s=='navigate' else ['2027'] for s in SKILLS},
            'unimplemented':[]}


def command_schema():
    """JSON Schema usable as an LLM function/tool argument contract."""
    common={'id':{'type':'string','pattern':r'^[\w.-]{1,64}$'},
            'radius_m':{'type':'number','minimum':.15,'maximum':3},
            'preferred_range_m':{'type':'number','minimum':.8,'maximum':8},
            'region_radius_m':{'type':'number','minimum':.5,'maximum':10},
            'leash_radius_m':{'type':'number','minimum':.5,'maximum':35},
            'timeout_s':{'type':'number','minimum':.1,'maximum':420},
            'target_lost_timeout_s':{'type':'number','minimum':0,'maximum':10},
            'low_hp_fraction':{'type':'number','minimum':0,'maximum':1},
            'damage_goal':{'type':['number','null'],'minimum':1,'maximum':10000},
            'basis_tick':{'type':['integer','null'],'minimum':0},
            'max_staleness_s':{'type':'number','minimum':0,'maximum':30}}
    point={'type':'array','items':{'type':'number','minimum':0,'maximum':100},'minItems':2,'maxItems':2}
    common.update({
        'ammo_target':{'type':'integer','minimum':0,'maximum':1000},
        'hp_target_fraction':{'type':'number','minimum':.1,'maximum':1},
        'budget_limit':{'type':'number','minimum':0,'maximum':5000},
        'patrol_loops':{'type':'integer','minimum':1,'maximum':100},
        'dwell_s':{'type':'number','minimum':0,'maximum':30},
        'follow_distance_m':{'type':'number','minimum':.6,'maximum':5},
        'escort_duration_s':{'type':'number','minimum':.5,'maximum':420},
        'required_coverage':{'type':'number','minimum':0,'maximum':1},
        'engage_on_contact':{'type':'boolean'},'lane_team':{'enum':['red','blue']},
        'flight_direction':{'enum':['forward','reverse']},'landing_hold_s':{'type':'number','minimum':.1,'maximum':3}})
    variants=[]
    for skill in SKILLS:
        properties={**common,'skill':{'const':skill}};required=['id','skill']
        if skill=='navigate':properties['point']=point;required.append('point')
        elif skill=='defend_region':
            properties['center']=point;required.append('center')
            properties['priority_ids']={'type':'array','items':{'type':'string','pattern':'^(red|blue)-[1-9][0-9]*$'},'maxItems':32,'uniqueItems':True}
        elif skill in ('resupply','retreat'):properties['point']=point
        elif skill=='patrol':
            properties['waypoints']={'type':'array','items':point,'minItems':2,'maxItems':32};required.append('waypoints')
            properties['priority_ids']={'type':'array','items':{'type':'string'},'maxItems':32,'uniqueItems':True}
        elif skill in ('fly_slope','recover'):pass
        else:
            if skill=='escort':
                properties['point']=point
                properties['priority_ids']={'type':'array','items':{'type':'string'},'maxItems':32,'uniqueItems':True}
            properties['target_id']={'type':'string','pattern':'^(red|blue)_(outpost|base)$' if skill=='attack_structure' else '^(red|blue)-[1-9][0-9]*$'}
            required.append('target_id')
        variants.append({'type':'object','properties':properties,'required':required,'additionalProperties':False})
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','oneOf':variants}
