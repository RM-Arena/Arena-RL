"""Whole-team execution: one tactical executor per ground robot plus the aerial hangar.

A TeamController owns the low-level side of a team. Orders arrive as tactical
commands (from the team commander); every ground robot's executor tracks its own
order, and the universal ground policies turn plans into 10 Hz actions. The same
controller can instead drive robots with the chassis-aware expert (used for the
declared scripted opponent team and as an ablation).
"""
import copy
import json
from pathlib import Path
import numpy as np
from .aerial import AerialController
from .executor import TacticalExecutor,own,xy,visible_enemies
from .ground_encoding import FAMILIES,IDLE_ACTION,encode,decode,scripted_action
from .robots import ROLES,profile,can_fire


class GroundPolicyBank:
    """Universal ground policies by family, with optional per-model fine-tuned overrides."""
    def __init__(self,path,root,device='cpu'):
        from .ground_learning import load_ground_policy
        self.path=Path(path);self.models={};self.metadata={};self.overrides=[]
        for family in FAMILIES:
            if (self.path/family/'metadata.json').is_file():
                meta,model=load_ground_policy(self.path/family,root,device)
                if meta['family']!=family:raise ValueError('Ground policy family does not match its directory')
                self.models[family]=model;self.metadata[family]=meta
        for candidate in sorted(self.path.glob('*-*/metadata.json')):
            meta,model=load_ground_policy(candidate.parent,root,device)
            self.overrides.append((meta['family'],set(meta['models']),model));self.metadata[candidate.parent.name]=meta
        if not self.models:raise ValueError(f'No universal ground policies under {self.path}')
        self.aerial=None
        if (self.path/'aerial'/'metadata.json').is_file():
            from .aerial_learning import load_aerial_policy
            meta,self.aerial=load_aerial_policy(self.path/'aerial',root,device);self.metadata['aerial']=meta

    def model_for(self,family,model_id):
        for name,models,policy in self.overrides:
            if name==family and model_id in models:return policy
        return self.models.get(family)

    def digests(self):return {k:m['policy_sha256'] for k,m in self.metadata.items()}


class TeamController:
    def __init__(self,team,bank=None,low_level='policy'):
        if low_level not in ('policy','expert'):raise ValueError('low_level must be policy or expert')
        if low_level=='policy' and bank is None:raise ValueError('Policy execution requires a GroundPolicyBank')
        self.team=team;self.bank=bank;self.low_level=low_level
        self.aerial_policy=getattr(bank,'aerial',None) if low_level=='policy' else None
        self.executors={};self.aerial=None;self.arena=None;self.orders={};self.sources={}

    def attach(self,observation):
        if observation['team']!=self.team:raise ValueError('Observation belongs to another team')
        self.arena=observation['arena'];self.executors={};self.aerial=None;self.orders={};self.sources={}
        for r in observation['robots']:
            if r['role']=='aerial':self.aerial=AerialController(str(r['slot']),self.aerial_policy)
            elif r['role'] in ROLES:
                ex=TacticalExecutor(str(r['slot']));ex.profile=profile(r,observation);ex.arena=self.arena;ex.update(observation)
                self.executors[r['id']]=ex
        return self

    def aerial_id(self):return f'{self.team}-{self.aerial.slot}' if self.aerial else None

    def submit(self,robot_id,order,observation,source='commander'):
        """order: a tactical command dict for ground robots or an aerial order name."""
        if robot_id==self.aerial_id():
            result=self.aerial.submit(order,observation,order_id=f'{source}-{observation["tick"]}')
        elif robot_id in self.executors:
            result=self.executors[robot_id].submit(order,observation)
        else:raise ValueError(f'{robot_id} is not a controlled robot')
        self.orders[robot_id]=copy.deepcopy(order);self.sources[robot_id]=source
        return result

    def cancel(self,robot_id,observation):
        self.orders.pop(robot_id,None)
        if robot_id==self.aerial_id():return self.aerial.submit('idle',observation)
        return self.executors[robot_id].cancel()

    def _share_vision(self,observation):
        # Allies' cameras feed every executor's short-term target memory (team communication).
        seen={}
        for slot,view in observation.get('vision',{}).items():
            for identity in view.get('enemies',[]):seen[identity]=True
        enemies={e['id']:e for e in observation['enemies'] if e['id'] in seen}
        for ex in self.executors.values():
            for identity,enemy in enemies.items():
                ex.memory[identity]={'value':copy.deepcopy(enemy),'seen_at':observation['time']}

    def _reflex(self,ex,observation,out):
        """Idle or finished robots still return fire on the nearest visible robot within 6 m."""
        from rmuc_arena import sdk
        robot=own(observation,ex.slot)
        if not robot['alive'] or robot['ammo']<=0 or not can_fire(robot,ex.profile):return out
        threats=[e for e in visible_enemies(observation,ex.slot) if e['alive'] and e.get('targetable',True) and e.get('role')!='aerial']
        target=min(threats,key=lambda e:float(np.linalg.norm(xy(e)-xy(robot))),default=None)
        if target is not None and np.linalg.norm(xy(target)-xy(robot))<6.:
            out.update(aim=sdk.aim_at(robot,target),pitch=sdk.aim_pitch(robot,target),fire=True)
        return out

    def actions(self,observation):
        self._share_vision(observation)
        for ex in self.executors.values():ex.update(observation)
        out={};batches={}
        for identity,ex in self.executors.items():
            family=ex.execution_family if ex.command is not None else 'navigation'
            if ex.status!='running' or family not in FAMILIES:
                action=IDLE_ACTION
                out.update(decode(ex,observation,action,'navigation'))
                if ex.status!='running':self._reflex(ex,observation,out[ex.slot])
                continue
            if self.low_level=='expert' or self.bank.model_for(family,ex.profile.model_id) is None:
                out.update(decode(ex,observation,scripted_action(ex,observation),family));continue
            batches.setdefault((family,ex.profile.model_id),[]).append(ex)
        for (family,model_id),group in batches.items():
            policy=self.bank.model_for(family,model_id)
            obs=np.stack([encode(ex,observation,family) for ex in group])
            actions=policy.predict(obs,deterministic=True)[0]
            for ex,action in zip(group,actions):out.update(decode(ex,observation,np.asarray(action,dtype=np.int64),family))
        if self.aerial is not None:out.update(self.aerial.action(observation))
        return out

    def feedback(self,observation):
        rows={identity:{**ex.feedback(),'source':self.sources.get(identity)} for identity,ex in self.executors.items()}
        if self.aerial is not None:rows[self.aerial_id()]={**self.aerial.feedback(observation),'source':self.sources.get(self.aerial_id())}
        return rows
