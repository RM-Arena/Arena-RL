"""Strict, provider-independent tactical commands for one robot."""
from dataclasses import asdict, dataclass
import math
import re

PROTOCOL = "arena-tactics-v1"
SKILLS = ("navigate", "attack_structure", "engage_target", "pursue_target", "defend_region")


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
                'low_hp_fraction':(0,1),'max_staleness_s':(0,30)}
        for key,(lo,hi) in limits.items():value[key]=number(value.get(key,getattr(defaults,key)),key,lo,hi)
        if 'damage_goal' in value and value['damage_goal'] is not None:value['damage_goal']=number(value['damage_goal'],'damage_goal',1,10000)
        if skill=='navigate' and 'point' not in value:raise ValueError('navigate requires point')
        if skill=='defend_region' and 'center' not in value:raise ValueError('defend_region requires center')
        if skill in ('attack_structure','engage_target','pursue_target') and not isinstance(value.get('target_id'),str):
            raise ValueError('This skill requires target_id')
        if skill=='attack_structure' and not re.fullmatch(r'(red|blue)_(outpost|base)',value['target_id']):
            raise ValueError('Structure target must be red/blue_outpost/base')
        if skill in ('engage_target','pursue_target') and not re.fullmatch(r'(red|blue)-[1-9][0-9]*',value['target_id']):
            raise ValueError('Robot target must be a team-slot id')
        priorities=value.get('priority_ids',())
        if not isinstance(priorities,(list,tuple)) or len(priorities)>32 or any(not isinstance(p,str) or not re.fullmatch(r'(red|blue)-[1-9][0-9]*',p) for p in priorities):
            raise ValueError('priority_ids must contain up to 32 robot ids')
        if len(set(priorities))!=len(priorities):raise ValueError('Duplicate priority target')
        value['priority_ids']=tuple(priorities)
        tick=value.get('basis_tick')
        if tick is not None and (isinstance(tick,bool) or not isinstance(tick,int) or tick<0):raise ValueError('basis_tick must be a nonnegative integer')
        allowed={'navigate':{'point'},'defend_region':{'center','priority_ids'},
                 'attack_structure':{'target_id'},'engage_target':{'target_id'},'pursue_target':{'target_id'}}[skill]
        for field in ('point','center','target_id','priority_ids'):
            if value.get(field) and field not in allowed:raise ValueError(f'{field} is not valid for {skill}')
        return cls(**value)

    @property
    def policy_family(self):
        return 'navigation' if self.skill=='navigate' else 'defense' if self.skill=='defend_region' else 'combat'

    def json(self):
        value={k:v for k,v in asdict(self).items() if v is not None}
        if self.skill!='defend_region':value.pop('priority_ids',None)
        for key in ('point','center','priority_ids'):
            if key in value:value[key]=list(value[key])
        return value


def capabilities():
    return {'protocol':PROTOCOL,'robot_model':3,'skills':list(SKILLS),'command_json_schema':command_schema(),
            'policy_families':['navigation','combat','defense'],'action_hz':10,
            'target_selection':'explicit robot id; defense accepts ordered priority_ids',
            'aim_mode':'SDK ballistic assistance with learned residuals',
            'trained_profiles':{'combat_range_m':[2.,3.5],'defense_radius_m':2.5,'season':'2027',
                                'enemy_models':[2711,3,4],'calibrated_success_probability':False},
            'command_fields':list(Command.__dataclass_fields__),
            'feedback_statuses':['idle','running','succeeded','failed','cancelled'],
            'unimplemented':['resupply','escort','patrol','airborne fly-slope skill']}


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
    variants=[]
    for skill in SKILLS:
        properties={**common,'skill':{'const':skill}};required=['id','skill']
        if skill=='navigate':properties['point']=point;required.append('point')
        elif skill=='defend_region':
            properties['center']=point;required.append('center')
            properties['priority_ids']={'type':'array','items':{'type':'string','pattern':'^(red|blue)-[1-9][0-9]*$'},'maxItems':32,'uniqueItems':True}
        else:
            properties['target_id']={'type':'string','pattern':'^(red|blue)_(outpost|base)$' if skill=='attack_structure' else '^(red|blue)-[1-9][0-9]*$'}
            required.append('target_id')
        variants.append({'type':'object','properties':properties,'required':required,'additionalProperties':False})
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','oneOf':variants}
