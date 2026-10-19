"""Decoupled rollout workers that keep every local core busy.

SB3's SubprocVecEnv waits for all environments after every 10 Hz step, so one slow
route search or warmup stalls the whole batch. Here each worker process owns one
GroundEnv and a CPU copy of the current policy, collects a complete rollout
segment on its own, and synchronizes with the learner once per PPO iteration.
The learner (CPU, CUDA or MPS) only runs the PPO update.
"""
import multiprocessing
import os
import time
import traceback
import numpy as np

THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _worker(connection,options):
    for key in THREAD_ENV:os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    from .bridge import configure
    from .ground_envs import GroundEnv
    configure(options.get('arena_root'))
    env=GroundEnv(options['family'],seed=options['seed'],models=options.get('models'),arena_root=options.get('arena_root'),training_rules=options.get('training_rules',True))
    policy=None;observation=None;episode_start=True;returns=0.;length=0
    try:
        while True:
            command,payload=connection.recv()
            if command=='close':break
            try:
                if command=='collect':
                    if policy is None:
                        from stable_baselines3.common.policies import ActorCriticPolicy
                        policy=ActorCriticPolicy(env.observation_space,env.action_space,lambda _:0.,**payload['policy_kwargs'])
                        policy.set_training_mode(False)
                    if payload.get('state_dict') is not None:policy.load_state_dict(payload['state_dict'])
                    if observation is None:observation,_=env.reset()
                    steps=payload['steps'];gamma=payload['gamma'];expert=payload.get('expert',False)
                    obs=np.zeros((steps,)+env.observation_space.shape,dtype=np.float32);actions=np.zeros((steps,7),dtype=np.int64)
                    rewards=np.zeros(steps,dtype=np.float32);starts=np.zeros(steps,dtype=np.float32)
                    values=np.zeros(steps,dtype=np.float32);log_probs=np.zeros(steps,dtype=np.float32);episodes=[];started=time.monotonic()
                    parent=os.getppid()
                    for t in range(steps):
                        if t%64==0 and os.getppid()!=parent:raise SystemExit(0)
                        obs[t]=observation;starts[t]=float(episode_start)
                        with torch.no_grad():
                            tensor=torch.as_tensor(observation[None])
                            if expert:
                                action=np.asarray(env.expert_action(),dtype=np.int64)
                                value=policy.predict_values(tensor);log_prob=policy.evaluate_actions(tensor,torch.as_tensor(action[None]))[1]
                            else:
                                action_t,value,log_prob=policy(tensor);action=action_t.numpy()[0]
                        actions[t]=action;values[t]=float(value.reshape(-1)[0]);log_probs[t]=float(log_prob.reshape(-1)[0])
                        observation,reward,terminated,truncated,info=env.step(action)
                        returns+=reward;length+=1
                        if truncated and not terminated:
                            # Bootstrap the time-limit cut as SB3 does for TimeLimit.truncated.
                            with torch.no_grad():reward+=gamma*float(policy.predict_values(torch.as_tensor(observation[None])).reshape(-1)[0])
                        rewards[t]=reward;episode_start=bool(terminated or truncated)
                        if episode_start:
                            episodes.append({**{k:v for k,v in info.items() if isinstance(v,(str,int,float,bool)) or v is None},'return':float(returns),'length':length})
                            returns=0.;length=0
                            observation,_=env.reset()
                    with torch.no_grad():last=float(policy.predict_values(torch.as_tensor(observation[None])).reshape(-1)[0])
                    connection.send({'ok':True,'obs':obs,'actions':actions,'rewards':rewards,'starts':starts,'values':values,'log_probs':log_probs,
                                     'last_value':last,'last_start':float(episode_start),'episodes':episodes,'seconds':time.monotonic()-started,
                                     'version':payload.get('version')})
                elif command=='evaluate':
                    # Deterministic or baseline episodes for held-out evaluation.
                    from stable_baselines3.common.policies import ActorCriticPolicy
                    if payload.get('state_dict') is not None:
                        policy=policy or ActorCriticPolicy(env.observation_space,env.action_space,lambda _:0.,**payload['policy_kwargs'])
                        policy.load_state_dict(payload['state_dict']);policy.set_training_mode(False)
                    rows=[];rng=np.random.default_rng(payload['seeds'][0] if payload['seeds'] else 0)
                    for seed in payload['seeds']:
                        observation,_=env.reset(seed=int(seed));done=False;total=0.;info={}
                        while not done:
                            if payload['mode']=='policy':
                                with torch.no_grad():action=policy.predict(observation,deterministic=True)[0]
                            elif payload['mode']=='expert':action=env.expert_action()
                            elif payload['mode']=='random':action=rng.integers(env.action_space.nvec)
                            else:action=np.array([0,0,1,1,0,1,0])
                            observation,reward,terminated,truncated,info=env.step(np.asarray(action,dtype=np.int64));total+=reward;done=terminated or truncated
                        # Later stages of the same seed continue the match; evaluate only the first recorded order.
                        rows.append({**{k:v for k,v in info.items() if isinstance(v,(str,int,float,bool)) or v is None},'return':float(total),'eval_seed':int(seed)})
                        if payload.get('replays'):
                            path=f"{payload['replays']}/{options['family']}-{seed}.json";env.save_replay(path);rows[-1]['replay']=path
                    observation=None
                    connection.send({'ok':True,'rows':rows})
                else:raise ValueError('Unknown worker command')
            except Exception as error:
                connection.send({'ok':False,'error':f'{type(error).__name__}: {error}','trace':traceback.format_exc()[-2000:]})
    finally:
        env.close();connection.close()


class RolloutWorkers:
    """Persistent worker pool; results always come back in worker order."""
    def __init__(self,family,num_workers,seed,models=None,arena_root=None,record=False,training_rules=True):
        context=multiprocessing.get_context('spawn');self.connections=[];self.processes=[];self.closed=False
        try:
            for i in range(num_workers):
                parent,child=context.Pipe()
                options={'family':family,'seed':int(seed)+i*1_000_003,'models':models,'arena_root':str(arena_root) if arena_root else None,'record':record,'training_rules':training_rules}
                process=context.Process(target=_worker,args=(child,options),daemon=True);process.start();child.close()
                self.connections.append(parent);self.processes.append(process)
        except BaseException:
            self.close();raise

    def _gather(self,payloads):
        for connection,payload in zip(self.connections,payloads):connection.send(payload)
        results=[connection.recv() for connection in self.connections]
        failed=[r for r in results if not r['ok']]
        if failed:raise RuntimeError('Rollout worker failed: '+failed[0]['error']+'\n'+failed[0].get('trace',''))
        return results

    def stream(self,state_dict,policy_kwargs,steps,gamma,count,version):
        """Asynchronous collection: return the first `count` finished segments.

        Every worker that finishes is re-dispatched at once with the newest weights,
        so a worker stuck in a slow warmup never idles the others. Segments carry the
        policy version that generated them; PPO's stored behavior log-probabilities and
        clipping absorb the small lag.
        """
        from multiprocessing.connection import wait
        if not hasattr(self,'busy'):self.busy=set()
        payload=('collect',{'state_dict':state_dict,'policy_kwargs':policy_kwargs,'steps':steps,'gamma':gamma,'version':version})
        for i,connection in enumerate(self.connections):
            if i not in self.busy:connection.send(payload);self.busy.add(i)
        results=[]
        while len(results)<count:
            for connection in wait([self.connections[i] for i in self.busy]):
                # Stop at the quota; other finished workers are read by the next call.
                if len(results)>=count:break
                i=self.connections.index(connection);result=connection.recv();self.busy.discard(i)
                if not result['ok']:raise RuntimeError('Rollout worker failed: '+result['error']+'\n'+result.get('trace',''))
                results.append({**result,'worker':i})
                if len(results)<count:connection.send(payload);self.busy.add(i)
        return results

    def drain(self):
        """Wait for in-flight segments (before synchronous calls or shutdown)."""
        for i in list(getattr(self,'busy',())):
            self.connections[i].recv();self.busy.discard(i)

    def collect(self,state_dict,policy_kwargs,steps,gamma,expert=False):
        self.drain()
        return self._gather([('collect',{'state_dict':state_dict,'policy_kwargs':policy_kwargs,'steps':steps,'gamma':gamma,'expert':expert})]*len(self.connections))

    def evaluate(self,seeds,mode,state_dict=None,policy_kwargs=None,replays=None):
        self.drain()
        shards=[list(seeds[i::len(self.connections)]) for i in range(len(self.connections))]
        results=self._gather([('evaluate',{'seeds':s,'mode':mode,'state_dict':state_dict,'policy_kwargs':policy_kwargs,'replays':replays}) for s in shards])
        return [row for r in results for row in r['rows']]

    def close(self):
        if self.closed:return
        self.closed=True
        for connection in self.connections:
            try:connection.send(('close',None))
            except (OSError,EOFError,BrokenPipeError):pass
        for process in self.processes:
            process.join(10)
            if process.is_alive():process.terminate();process.join(2)
        for connection in self.connections:connection.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
