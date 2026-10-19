"""Optional successful-rollout initialization before on-policy PPO training."""
import hashlib
import numpy as np
import torch
from .tactical_envs import TacticalBatch


def initialize_from_demonstrations(model, *, family, episodes, seed, arena_root, output,model_id=3):
    from .tactical_learning import actions_for_baseline
    batch=TacticalBatch(min(4,episodes),family=family,seed=seed,arena_root=arena_root,model_id=model_id)
    rng=np.random.default_rng(seed);observations=[];actions=[];accepted=[];rejected=[]
    try:
        obs=batch.reset();active=list(range(batch.num_envs));next_seed=seed+batch.num_envs
        trajectories={i:[] for i in active}
        while active:
            act=actions_for_baseline(batch,active,'heuristic',rng)
            for k,i in enumerate(active):trajectories[i].append((obs[i].copy(),act[k].copy()))
            new,_,terms,truncs,infos=batch.step(act,env_ids=active)
            remaining=[];reset_ids=[];reset_seeds=[]
            for k,i in enumerate(active):
                obs[i]=new[k]
                if terms[k] or truncs[k]:
                    if infos[k]['is_success']:
                        accepted.append(infos[k]['seed'])
                        for observation,action in trajectories[i]:observations.append(observation);actions.append(action)
                    else:rejected.append({'seed':infos[k]['seed'],'reason':infos[k]['reason']})
                    if next_seed<seed+episodes:
                        reset_ids.append(i);reset_seeds.append(next_seed);next_seed+=1;remaining.append(i);trajectories[i]=[]
                else:remaining.append(i)
            if reset_ids:
                for i,value in zip(reset_ids,batch.reset(reset_ids,seeds=reset_seeds)):obs[i]=value
            active=remaining
    finally:batch.close()
    if not accepted:raise RuntimeError('No successful demonstrations; refusing empty initialization')
    x=np.asarray(observations,dtype=np.float32);y=np.asarray(actions,dtype=np.int64)
    path=output/'demonstrations.npz';np.savez_compressed(path,observations=x,actions=y,seeds=accepted)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    losses=[]
    for epoch in range(160):
        order=rng.permutation(len(x))
        for start in range(0,len(x),128):
            selection=order[start:start+128]
            tensor=torch.as_tensor(x[selection],device=model.device)
            target=torch.as_tensor(y[selection],device=model.device)
            distribution=model.policy.get_distribution(tensor)
            loss=-distribution.log_prob(target).mean()
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step()
        losses.append(float(loss.detach()))
    model.policy.optimizer.state.clear()
    with torch.no_grad():
        predicted=model.predict(x,deterministic=True)[0]
    return {'method':'behavior cloning of successful scripted rollouts, followed by PPO',
            'episodes_requested':episodes,'accepted_seeds':accepted,'rejected':rejected,'samples':len(x),
            'epochs':160,'initial_loss':losses[0],'final_loss':losses[-1],
            'action_accuracy':float(np.mean(predicted==y)),
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
