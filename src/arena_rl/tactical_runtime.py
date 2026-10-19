"""LLM-neutral command execution, JSONL transport, and continuous mission evaluation."""
import json
import hashlib
import math
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time
import numpy as np
from .bridge import configure
from .commands import capabilities,Command,SKILLS
from .executor import TacticalExecutor,own,xy,visible_enemies
from .tactical_encoding import encode
from .tactical_envs import TacticalBatch,Episode,FAMILIES,encode_observation
from .tactical_learning import load_tactical_policy
from .learning import dump,seed_everything


class PolicyBank:
    def __init__(self,path,root):
        self.models={};self.metadata={}
        for name in FAMILIES:
            if name in ('support','flight','mobility') and not (Path(path)/name/'metadata.json').is_file():continue
            model,meta=load_tactical_policy(Path(path)/name,root)
            if meta['family']!=name:raise ValueError('Policy family does not match its directory')
            self.models[name]=model;self.metadata[name]=meta
        ids={m.get('model_id',3) for m in self.metadata.values()}
        if len(ids)!=1:raise ValueError('All loaded policies must control the same model')
        self.model_id=ids.pop()
        if self.model_id==305 and 'mobility' not in self.models:raise ValueError('Serial 305 policy bank requires its trained mobility policy')
        self.transit_model=None
        if (Path(path)/'navigation/transit/metadata.json').is_file():
            self.transit_model,meta=load_tactical_policy(Path(path)/'navigation/transit',root)
            if self.model_id!=305 or meta['family']!='navigation' or meta['model_id']!=self.model_id:raise ValueError('Invalid navigation transit model')
            self.metadata['navigation_transit']=meta

    def predict(self,batch,ids):
        result=np.tile([0,0,1,1,0,1,0],(len(ids),1))
        for family,model in self.models.items():
            positions=[k for k,i in enumerate(ids) if batch.executors[i].status=='running' and batch.executors[i].execution_family==family]
            if positions:
                transit=[k for k in positions if family=='navigation' and self.transit_model is not None and batch.executors[ids[k]].plan['distance']>4. and not batch.executors[ids[k]].route_state.get('terrain_crossing')]
                local=[k for k in positions if k not in transit]
                for selected,policy in [(local,model),(transit,self.transit_model)]:
                    if selected:
                        obs=np.stack([encode_observation(batch.executors[ids[k]],batch.states[ids[k]]['observations']['red'],family,legacy_flight=family=='flight' and self.metadata[family]['schema']=='w3-flight-v1') for k in selected])
                        result[selected]=policy.predict(obs,deterministic=True)[0]
        return result


class TacticalSession:
    def __init__(self,policies,envs=1,seed=207,arena_root=None,record=False,opponent=None,lineups=None,season='2027'):
        root=configure(arena_root);seed_everything(seed)
        self.bank=PolicyBank(policies,root)
        if lineups is None:
            import copy
            from .extended_curricula import SUPPORT_LINEUPS,FLIGHT_LINEUPS
            lineups=copy.deepcopy(SUPPORT_LINEUPS if season=='2027' else FLIGHT_LINEUPS)
            next(d for d in lineups['red'] if d['slot']==3)['model']=self.bank.model_id
        self.batch=TacticalBatch(envs,seed=seed,arena_root=root,record=record,opponent=opponent,lineups=lineups,season=season,model_id=self.bank.model_id)
        self.batch.runtime_control=True
        self.envs=envs;self.seed=seed;self.logs=[[] for _ in range(envs)]

    def reset(self,seeds=None,scenarios=None,behaviors=None):
        seeds=list(range(self.seed,self.seed+self.envs)) if seeds is None else seeds
        if len(seeds)!=self.envs:raise ValueError('One seed per actor required')
        scenarios=[{'robots':{}} for _ in seeds] if scenarios is None else scenarios
        behaviors=[{} for _ in seeds] if behaviors is None else behaviors
        if len(scenarios)!=self.envs or len(behaviors)!=self.envs:raise ValueError('One scenario per actor required')
        episodes=[Episode('runtime',int(s),scenario,{'id':'initial','skill':'navigate','point':[4.5,5.15]},behavior,4200)
                  for s,scenario,behavior in zip(seeds,scenarios,behaviors)]
        self.batch.reset(seeds=seeds,episodes=episodes)
        for i in range(self.envs):
            self.batch.executors[i]=TacticalExecutor();self.batch.executors[i].update(self.batch.states[i]['observations']['red'])
            self.logs[i]=[{'op':'reset','seed':seeds[i],'scenario':scenarios[i]}]
        return [self.observe(i) for i in range(self.envs)]

    def _index(self,i):
        if isinstance(i,bool) or not isinstance(i,int) or not 0<=i<self.envs:raise ValueError('Invalid env_id')
        if self.batch.states[i] is None:raise ValueError('Reset before commands or observations')

    def capabilities(self):
        value=capabilities();season=self.batch.season
        value.update(season=season,robot_model=self.bank.model_id,available_families=list(self.bank.models),
            available_skills=[s for s in SKILLS if Command('capability',s).policy_family in self.bank.models and season in value['skill_seasons'][s] and (s!='recover' or self.bank.model_id==305)])
        return value

    def command(self,value,env_id=0):
        self._index(env_id)
        if self.batch.states[env_id]['terminated'] or self.batch.states[env_id]['truncated']:raise ValueError('Episode has ended; reset before commanding')
        candidate=Command.parse(value)
        if candidate.policy_family not in self.bank.models:raise ValueError('No trained policy loaded for this skill')
        if self.batch.season=='2026' and candidate.skill not in ('navigate','fly_slope'):raise ValueError('This skill is currently trained for 2027')
        response=self.batch.command(env_id,value)
        if not response['duplicate']:
            self.logs[env_id].append({'op':'command','tick':self.batch.states[env_id]['info']['tick'],'command':self.batch.executors[env_id].command.json()})
        return response

    def cancel(self,env_id=0):
        self._index(env_id);result=self.batch.executors[env_id].cancel()
        self.logs[env_id].append({'op':'cancel','tick':self.batch.states[env_id]['info']['tick']})
        return result

    def advance(self,steps=1):
        if isinstance(steps,bool) or not isinstance(steps,int) or not 1<=steps<=4200:raise ValueError('steps must be an integer in [1,4200]')
        for _ in range(steps):
            ids=[i for i,s in enumerate(self.batch.states) if s is not None and not(s['terminated'] or s['truncated'])]
            if not ids:break
            self.batch.step(self.bank.predict(self.batch,ids),env_ids=ids,training=False)
        return [self.observe(i) for i,s in enumerate(self.batch.states) if s is not None]

    def observe(self,env_id=0):
        self._index(env_id);state=self.batch.states[env_id];obs=state['observations']['red'];ex=self.batch.executors[env_id]
        keys=('id','slot','role','model_slot','x','y','z','vx','vy','vz','hp','max_hp','ammo','heat','max_heat','alive','powered','heading','aim','upright','radius','quaternion','angular_velocity','grounded','self_righting','tipped','weak','invulnerable')
        compact=lambda r:{k:r[k] for k in keys if k in r}
        return {'env_id':env_id,'tick':obs['tick'],'time':obs['time'],'episode_finished':bool(state['terminated'] or state['truncated']),
                'robot':compact(own(obs)),'allies':[compact(r) for r in obs['robots'] if str(r['slot'])!='3'],'visible_enemies':[compact(e) for e in visible_enemies(obs)],
                'last_seen_enemies':[{**compact(record['value']),'age_s':obs['time']-record['seen_at']}
                                     for identity,record in ex.memory.items() if identity not in {e['id'] for e in visible_enemies(obs)}],
                'economy':{k:v for k,v in obs['teams']['red'].items() if k in ('resources','purchased_17','purchased_42')},
                'supply_contacts':[identity for identity,zone in obs['zones'].items() if 'red-3' in zone.get('occupants',[])],
                'bases':obs['bases'],'outposts':obs['outposts'],'positions':obs.get('positions',{}),
                'scoreboard':{team:{k:v for k,v in value.items() if k in ('control','attack_damage','score','kills')} for team,value in obs['teams'].items()},
                'feedback':ex.feedback()}

    def save_replay(self,path,env_id=0):
        self._index(env_id);path=Path(path)
        result=self.batch.engine.save_replay([path],env_ids=[env_id])[0]
        dump(path.with_suffix('.commands.json'),{'protocol':'arena-tactics-v1','commands':self.logs[env_id],
             'policies':{k:m['policy_sha256'] for k,m in self.bank.metadata.items()}})
        return result

    def close(self):self.batch.close()


def serve_tactics(*,policies,envs=1,seed=207,realtime=False,speed=1.,record=False,opponent=None,lineups=None,season='2027',arena_root=None):
    if not math.isfinite(speed) or not .1<=speed<=20:raise ValueError('speed must be in [.1,20]')
    if lineups is not None:
        if isinstance(lineups,str):
            lineups=json.loads(lineups if lineups.lstrip().startswith('{') else Path(lineups).read_text())
    session=TacticalSession(policies,envs,seed,arena_root,record,opponent,lineups,season);incoming=queue.Queue(maxsize=128)
    write=lambda row: print(json.dumps(row,allow_nan=False,separators=(',',':')),flush=True)
    def reader():
        # Raw file-descriptor reads avoid a daemon holding Python's buffered stdin
        # lock when a close command shuts down with the parent pipe still open.
        pending=bytearray();discard=False
        def parse(raw):
            try:
                request=json.loads(raw,parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Non-finite JSON')))
                if not isinstance(request,dict):raise ValueError('Request must be an object')
                return request
            except (ValueError,UnicodeDecodeError,RecursionError) as e:return {'_parse_error':str(e)}
        try:
            while True:
                chunk=os.read(sys.stdin.fileno(),65536)
                if not chunk:break
                pending.extend(chunk)
                while b'\n' in pending:
                    line,_,rest=pending.partition(b'\n');pending=bytearray(rest)
                    if discard or len(line)>65536:
                        incoming.put({'_parse_error':'Request exceeds 64 KiB'});discard=False
                    else:incoming.put(parse(line))
                if len(pending)>65536:pending.clear();discard=True
            if pending and not discard:incoming.put(parse(pending))
        finally:incoming.put(None)

    old_signal=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        states=session.reset();write({'type':'ready',**session.capabilities(),'realtime':realtime,'states':states,'season':season,'available_families':list(session.bank.models),
            'arena':session.batch.executors[0].arena,'opponent_mode':'saved-policy' if opponent else 'stationary','roster':session.batch.states[0]['observations']['red']['roster']})
        threading.Thread(target=reader,daemon=True).start()
        period=.1/speed;deadline=time.monotonic()+period;last_report=time.monotonic()
        while True:
            timeout=max(0.,deadline-time.monotonic()) if realtime else None
            try:request=incoming.get(timeout=timeout)
            except queue.Empty:request='tick'
            if request is None:break
            if request!='tick':
                identity=request.get('id')
                try:
                    if identity is not None and (isinstance(identity,bool) or not isinstance(identity,(str,int))):
                        identity=None;raise ValueError('id must be a string or integer')
                    if '_parse_error' in request:raise ValueError(request['_parse_error'])
                    op=request.get('op');i=request.get('env_id',0)
                    if op=='command':result=session.command(request['command'],i)
                    elif op=='cancel':result=session.cancel(i)
                    elif op=='observe':result=session.observe(i)
                    elif op=='step':
                        if realtime:raise ValueError('Realtime mode advances independently; use observe')
                        result=session.advance(request.get('steps',1))
                    elif op=='reset':result=session.reset(request.get('seeds'),request.get('scenarios'))
                    elif op=='save_replay':result=session.save_replay(request['path'],i)
                    elif op=='capabilities':result=session.capabilities()
                    elif op=='close':write({'id':identity,'ok':True,'type':'close'});break
                    else:raise ValueError('Unknown operation')
                    write({'id':identity,'ok':True,'type':op,'result':result})
                except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                    write({'id':identity,'ok':False,'type':'error','error':str(exc)})
            if realtime and time.monotonic()>=deadline:
                states=session.advance();deadline=max(deadline+period,time.monotonic())
                if time.monotonic()-last_report>=1.:
                    write({'type':'state','states':states});last_report=time.monotonic()
    finally:
        signal.signal(signal.SIGTERM,old_signal);session.close()


def run_mission(*,policies,seed=4000000,output,arena_root=None):
    """A deterministic command issuer tests composition; it is not an LLM benchmark."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    session=TacticalSession(policies,seed=seed,arena_root=arena_root,record=True)
    scenario={'robots':{'red-3':{'x':4.5,'y':5.15,'heading':0.,'aim':0.,'ammo':300},
                           'blue-1':{'x':8.5,'y':5.15,'heading':math.pi,'ammo':0},
                           'blue-3':{'x':6.7,'y':5.8,'heading':math.pi,'ammo':0},
                           'blue-4':{'x':24.,'y':5.8,'heading':math.pi,'ammo':0}}}
    stages=[{'id':'approach','skill':'navigate','point':[19.4,4.9],'timeout_s':75.},
            {'id':'outpost','skill':'attack_structure','target_id':'blue_outpost','preferred_range_m':2.5,'damage_goal':60.,'timeout_s':30.},
            {'id':'return-home','skill':'navigate','point':[4.5,5.15],'timeout_s':75.},
            {'id':'guard','skill':'defend_region','center':[4.5,5.15],'region_radius_m':3.,'priority_ids':['blue-1'],
             'preferred_range_m':2.5,'damage_goal':60.,'timeout_s':45.},
            {'id':'pursue-heavy','skill':'pursue_target','target_id':'blue-1','preferred_range_m':2.5,
             'damage_goal':60.,'target_lost_timeout_s':3.,'leash_radius_m':10.,'timeout_s':20.}]
    results=[];continuity=[]
    try:
        session.reset([seed],[scenario]);simulator=session.batch.states[0]['info']['simulator_pid']
        for n,command in enumerate(stages):
            if n==2:session.batch.episodes[0].behavior={'blue-1':{'mode':'invade','center':[6.7,5.15],'speed':.14}}
            if n==4:session.batch.episodes[0].behavior={'blue-1':{'mode':'invade','center':[9.5,5.15],'speed':.2}}
            before=session.observe();session.command(command)
            after=session.observe()
            continuity.append({'id':command['id'],'before_tick':before['tick'],'after_tick':after['tick'],
                               'before_ammo':before['robot']['ammo'],'after_ammo':after['robot']['ammo'],
                               'same_simulator':session.batch.states[0]['info']['simulator_pid']==simulator})
            for _ in range(int(command['timeout_s']*10)+1):
                if session.batch.executors[0].status!='running':break
                session.advance()
            result=session.observe()['feedback'];results.append(result)
            print(json.dumps({'type':'mission_stage',**result}),flush=True)
            if result['status']!='succeeded':break
        session.save_replay(output/'mission.json')
        report={'seed':seed,'execution_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'stages':results,'continuity':continuity,'success':len(results)==len(stages) and all(r['status']=='succeeded' for r in results),
                'final_tick':session.observe()['tick'],'policy_sha256':{k:m['policy_sha256'] for k,m in session.bank.metadata.items()},
                'issuer':'scripted commands; no LLM API invoked','replay':str(output/'mission.json')}
        dump(output/'report.json',report);return report
    finally:session.close()
