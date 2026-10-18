"""PPO training, independent seeded evaluation, and reproducible checkpoints."""
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .bridge import configure, provenance, check_provenance
from .encoding import SCHEMA, ACTION_SPACE, DIRECTIONS
from .envs import SkillVecEnv, SkillBatch
from .tasks import robot


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def seed_everything(seed,threads=1):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def weight_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.policy.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class Metrics(BaseCallback):
    def __init__(self,path):
        super().__init__();self.path=Path(path);self.history=deque(maxlen=100);self.started=time.monotonic()
    def _on_step(self):
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if done:self.history.append(info)
        return True
    def _on_rollout_end(self):
        by_task=defaultdict(list)
        for info in self.history:by_task[info['task']].append(info)
        row={'timesteps':self.num_timesteps,'wall_seconds':time.monotonic()-self.started,
             'tasks':{task:{'episodes':len(rows),'success_rate':sum(r['is_success'] for r in rows)/len(rows)} for task,rows in by_task.items()}}
        with self.path.open('a') as stream:stream.write(json.dumps(row)+'\n')
        if self.n_calls%(self.model.n_steps*8)==0:print(json.dumps(row),flush=True)


def train(*,task='move',steps=32768,envs=4,seed=7,output='runs/move',arena_root=None,resume=None,allow_engine_change=False):
    root=configure(arena_root);output=Path(output)
    if steps<1 or envs<1 or envs>128:raise ValueError('Positive steps and 1..128 environments required')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'metadata.json').exists() and not resume:raise ValueError('Output already contains a run; use a new directory or --resume')
    seed_everything(seed);source=provenance(root)
    environment=SkillVecEnv(envs,task=task,seed=seed,arena_root=root)
    try:
        if resume:
            meta=json.loads((Path(resume)/'metadata.json').read_text())
            if meta['schema']!=SCHEMA or meta['task']!=task:raise ValueError('Checkpoint task/schema mismatch')
            check_provenance(meta['arena'],source,allow_engine_change)
            model=PPO.load(Path(resume)/'policy.zip',env=environment,device='cpu')
            # Continuation starts fresh episodes; the optimizer and network are restored.
        else:
            model=PPO('MlpPolicy',environment,seed=seed,device='cpu',n_steps=128,batch_size=128,
                      n_epochs=6,learning_rate=1e-3,gamma=.98,gae_lambda=.95,ent_coef=.01,
                      policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64])},verbose=0)
        before=weight_hash(model);started=time.monotonic();initial_steps=model.num_timesteps
        metadata={'format':'arena-rl-checkpoint-v1','schema':SCHEMA,'task':task,'model_id':3,'slot':3,
                  'season':'2027','seed':seed,'requested_steps':steps,'num_envs':envs,'action_hz':10,
                  'aim_mode':'sdk-ballistics-with-learned-residuals','arena':source,
                  'algorithm':'stable-baselines3.PPO','weights_before':before,'resumed_from':str(resume) if resume else None,
                  'versions':{'torch':torch.__version__,'numpy':np.__version__}}
        if resume:
            metadata["parent_checkpoint"]={"policy_sha256":meta["policy_sha256"],"actual_steps":meta["actual_steps"],
                                           "arena":meta["arena"],"engine_change_allowed":allow_engine_change}
        dump(output/'config.json',metadata)
        model.learn(total_timesteps=steps,callback=Metrics(output/'metrics.jsonl'),reset_num_timesteps=False)
        model.save(output/'policy.zip');after=weight_hash(model)
        metadata.update(actual_steps=model.num_timesteps,additional_steps=model.num_timesteps-initial_steps,
                        wall_seconds=time.monotonic()-started,weights_after=after,weights_changed=before!=after,
                        ppo_updates=model._n_updates,episode_seed_range=[seed,environment.batch.next_seed-1],policy_sha256=hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest())
        if before==after:raise RuntimeError('Training did not update network weights')
        dump(output/'metadata.json',metadata)
        print(json.dumps({'type':'trained','output':str(output),'steps':model.num_timesteps,
                          'updates':model._n_updates,'weights_changed':True,'seconds':metadata['wall_seconds']}),flush=True)
        return metadata
    finally:environment.close()


def load_policy(checkpoint,root,allow_engine_change=False):
    checkpoint=Path(checkpoint);meta=json.loads((checkpoint/'metadata.json').read_text())
    if meta['schema']!=SCHEMA:raise ValueError('Unsupported policy schema')
    if hashlib.sha256((checkpoint/'policy.zip').read_bytes()).hexdigest()!=meta['policy_sha256']:
        raise ValueError('Checkpoint digest mismatch')
    check_provenance(meta['arena'],provenance(root),allow_engine_change)
    return PPO.load(checkpoint/'policy.zip',device='cpu'),meta


def baseline_actions(batch,kind,rng):
    if kind=='random':return np.stack([rng.integers(ACTION_SPACE.nvec) for _ in batch.tasks])
    if kind=='idle':return np.tile([0,1,1,0,1,0],(batch.num_envs,1))
    if kind!='heuristic':raise ValueError('Unknown baseline')
    actions=[]
    for t,s in zip(batch.tasks,batch.states):
        r=robot(s);delta=t.goal-[r['x'],r['y']];length=np.linalg.norm(delta)
        k=int(np.argmax(DIRECTIONS@(delta/max(.01,length)))) if length>.24 else 0
        if t.name=='outpost':k=0
        actions.append([k,1,1,int(t.name in ('outpost','defend')),2 if t.name=='ramp' else 1,
                        int(t.name=='ramp' and 12.35<r['y']<12.85)])
    return np.array(actions)


def evaluate(*,checkpoint=None,task=None,episodes=16,envs=4,seed=1000000,arena_root=None,baseline=None,
             output=None,replays=None,allow_engine_change=False):
    root=configure(arena_root);seed_everything(seed);rng=np.random.default_rng(seed)
    policy,meta=load_policy(checkpoint,root,allow_engine_change) if checkpoint else (None,None)
    task=task or (meta['task'] if meta else 'move')
    if (policy is None)==(baseline is None):raise ValueError('Choose one checkpoint or baseline')
    if episodes<1:raise ValueError('Positive episode count required')
    batch=SkillBatch(min(envs,episodes),task=task,seed=seed,arena_root=root,record=bool(replays))
    rows=[];active={i:seed+i for i in range(batch.num_envs)};next_seed=seed+batch.num_envs
    returns=np.zeros(batch.num_envs);started=time.monotonic()
    try:
        obs=batch.reset()
        while active:
            actions=policy.predict(obs,deterministic=True)[0] if policy else baseline_actions(batch,baseline,rng)
            obs,rewards,terms,truncs,infos=batch.step(actions);returns+=rewards
            reset_ids=[];reset_seeds=[]
            for i in list(active):
                if terms[i] or truncs[i]:
                    row={**infos[i],'return':float(returns[i]),'terminated':bool(terms[i]),'truncated':bool(truncs[i])}
                    rows.append(row)
                    if replays:
                        path=Path(replays)/f'{task}-{active[i]}.json'
                        batch.engine.save_replay([path],env_ids=[i]);row['replay']=str(path)
                    del active[i]
                    if next_seed<seed+episodes:
                        active[i]=next_seed;reset_ids.append(i);reset_seeds.append(next_seed);next_seed+=1;returns[i]=0
            # Reset idle actors too, because the vector API steps every actor in this evaluation batch.
            idle=[i for i in range(batch.num_envs) if i not in active and (terms[i] or truncs[i])]
            ids=reset_ids+idle;seeds=reset_seeds+[seed+episodes+100000+i for i in idle]
            if active and ids:
                reset=batch.reset(ids,seeds=seeds)
                for i,o in zip(ids,reset):obs[i]=o
        result={'task':task,'checkpoint':str(checkpoint) if checkpoint else None,'baseline':baseline,
                'episodes':episodes,'seed_range':[seed,seed+episodes-1],
                'success_rate':sum(r['is_success'] for r in rows)/episodes,
                'mean_return':float(np.mean([r['return'] for r in rows])),
                'mean_outpost_damage':float(np.mean([r['outpost_damage'] for r in rows])),
                'mean_attack_damage':float(np.mean([r['attack_damage'] for r in rows])),
                'wall_seconds':time.monotonic()-started,'results':sorted(rows,key=lambda r:r['seed'])}
        if output:dump(output,result)
        print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
        return result
    finally:batch.close()
