"""Single robot Gymnasium and SB3 adapters sharing Arena's persistent actors."""
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
from .bridge import configure
from .encoding import ACTION_SPACE, OBS_SPACE, encode, decode
from .tasks import sample, begin, reward, MODEL


class SkillBatch:
    def __init__(self,num_envs=1,task="mixed",seed=0,arena_root=None,record=False,difficulty=1.):
        configure(arena_root)
        from rmuc_arena.vector_env import VectorArena
        if task not in ("move","ramp","outpost","defend","mixed"):
            raise ValueError("Unknown skill task")
        self.num_envs,self.task,self.next_seed,self.difficulty=num_envs,task,seed,difficulty
        self.tasks=[None]*num_envs;self.states=[None]*num_envs;self.invaders=[False]*num_envs
        self.engine=VectorArena(num_envs,season="2027",duration=60,compact=True,record=record,
                               lineups={t:[{"slot":3,"model":MODEL}] for t in ("red","blue")})

    def reset(self,env_ids=None,seeds=None):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if seeds is None:
            seeds=list(range(self.next_seed,self.next_seed+len(ids)));self.next_seed+=len(ids)
        if len(seeds)!=len(ids):raise ValueError("One seed per environment required")
        tasks=[sample(self.task,int(s),self.difficulty) for s in seeds]
        states=self.engine.reset(seeds,env_ids=ids,scenarios=[t.scenario for t in tasks])
        observations=[]
        for i,t,s in zip(ids,tasks,states):
            self.tasks[i],self.states[i]=t,s;self.invaders[i]=t.name=="defend"
            begin(t,s);observations.append(encode(t,s))
        return np.asarray(observations,dtype=np.float32)

    def step(self,actions,env_ids=None):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if len(actions)!=len(ids):raise ValueError("One action per selected environment required")
        commands=[decode(self.tasks[i],self.states[i],a) for i,a in zip(ids,actions)]
        # The invader belongs to the scenario and keeps running when a manager changes its option.
        for i,c in zip(ids,commands):
            if self.invaders[i]:
                from rmuc_arena import sdk
                other=self.states[i]["observations"]["blue"];r=other["robots"][0]
                target={**other["bases"]["red"],"kind":"base"}
                c["blue"]={"3":{"move":[-.22,0],"heading":3.141592653589793,
                                  "aim":sdk.aim_at(r,target),"pitch":sdk.aim_pitch(r,target),"fire":True}}
        states=self.engine.step(commands,env_ids=ids)
        observations=[];rewards=[];terminated=[];truncated=[];infos=[]
        for i,s in zip(ids,states):
            self.states[i]=s;t=self.tasks[i]
            r,term,trunc,info=reward(t,s)
            info["engine_hash"]=s["info"]["engine_hash"];info["rules_hash"]=s["info"]["rules_hash"]
            observations.append(encode(t,s));rewards.append(r);terminated.append(term);truncated.append(trunc);infos.append(info)
        return np.asarray(observations),np.asarray(rewards,dtype=np.float32),np.asarray(terminated),np.asarray(truncated),infos

    def close(self):self.engine.close()


class SingleRobotEnv(gym.Env):
    """Standard single-agent API; no learner can command the blue actor."""
    metadata={"render_modes":[]}
    observation_space=OBS_SPACE
    action_space=ACTION_SPACE

    def __init__(self,**kwargs):
        super().__init__();self.batch=SkillBatch(1,**kwargs);self.finished=True

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        obs=self.batch.reset(seeds=None if seed is None else [seed])[0];self.finished=False
        return obs,{"task":self.batch.tasks[0].name,"seed":self.batch.tasks[0].seed}

    def step(self,action):
        if self.finished:raise RuntimeError("Reset the finished environment before step")
        obs,r,t,tr,info=self.batch.step([action]);self.finished=bool(t[0] or tr[0])
        return obs[0],float(r[0]),bool(t[0]),bool(tr[0]),info[0]

    def close(self):self.batch.close()


class SkillVecEnv(VecEnv):
    def __init__(self,num_envs=4,**kwargs):
        self.batch=SkillBatch(num_envs,**kwargs);self.pending=None
        self.returns=np.zeros(num_envs);self.lengths=np.zeros(num_envs,dtype=int)
        super().__init__(num_envs,OBS_SPACE,ACTION_SPACE)

    def reset(self):
        seeds=self._seeds if any(s is not None for s in self._seeds) else None
        obs=self.batch.reset(seeds=seeds);self._reset_seeds();self._reset_options()
        self.returns[:]=0;self.lengths[:]=0
        self.reset_infos=[{"task":t.name,"seed":t.seed} for t in self.batch.tasks]
        return obs

    def step_async(self,actions):self.pending=np.asarray(actions).copy()

    def step_wait(self):
        if self.pending is None:raise RuntimeError("step_async required")
        obs,rewards,terminated,truncated,infos=self.batch.step(self.pending);self.pending=None
        dones=terminated|truncated;self.returns+=rewards;self.lengths+=1
        ids=np.flatnonzero(dones).tolist()
        for i in ids:
            infos[i]["terminal_observation"]=obs[i].copy()
            infos[i]["TimeLimit.truncated"]=bool(truncated[i])
            infos[i]["episode"]={"r":float(self.returns[i]),"l":int(self.lengths[i])}
        if ids:
            reset=self.batch.reset(ids)
            for i,new in zip(ids,reset):
                obs[i]=new;self.returns[i]=0;self.lengths[i]=0
                self.reset_infos[i]={"task":self.batch.tasks[i].name,"seed":self.batch.tasks[i].seed}
        return obs,rewards,dones,infos

    def close(self):self.batch.close()
    def get_attr(self,attr_name,indices=None):return [getattr(self,attr_name,None) for _ in self._get_indices(indices)]
    def set_attr(self,attr_name,value,indices=None):
        if attr_name=="render_mode" and value is None:return
        raise AttributeError("Environment configuration is immutable")
    def env_method(self,method_name,*method_args,indices=None,**method_kwargs):
        raise AttributeError("Call the explicit single-robot environment interface")
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
