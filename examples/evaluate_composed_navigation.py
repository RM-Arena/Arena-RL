"""Evaluate the deployed policy bank, including mobility subphase dispatch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from arena_rl.bridge import configure
from arena_rl.learning import dump
from arena_rl.tactical_envs import LINEUPS,sample_episode
from arena_rl.tactical_runtime import TacticalSession


def main():
    p=argparse.ArgumentParser();p.add_argument('--policies',default='checkpoints/serial305');p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--seed',type=int,default=3000000);p.add_argument('--output',required=True)
    p.add_argument('--curriculum',choices=['reference','refinement'],default='reference');args=p.parse_args()
    root=configure();lineups=copy.deepcopy(LINEUPS)
    lineups['red'][0]['model']=json.loads((Path(args.policies)/'navigation/metadata.json').read_text())['model_id']
    session=TacticalSession(args.policies,envs=min(4,args.episodes),lineups=lineups,arena_root=root)
    batch=session.batch;rows=[];used=[set() for _ in range(session.envs)]
    import arena_rl
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(arena_rl.__file__).parent.glob('*.py')}
    next_seed=args.seed+session.envs
    try:
        seeds=list(range(args.seed,next_seed));batch.reset(seeds=seeds,episodes=[sample_episode('navigation',s,batch.model_id,args.curriculum) for s in seeds]);active=list(range(session.envs))
        while active:
            for i in active:used[i].add(batch.executors[i].execution_family)
            _,_,terms,truncs,infos=batch.step(session.bank.predict(batch,active),env_ids=active,training=False)
            remaining=[];reset=[];new_seeds=[]
            for k,i in enumerate(active):
                if terms[k] or truncs[k]:
                    rows.append({**infos[k],'families_used':sorted(used[i])})
                    if next_seed<args.seed+args.episodes:
                        reset.append(i);new_seeds.append(next_seed);next_seed+=1;remaining.append(i);used[i]=set()
                else:remaining.append(i)
            if reset:batch.reset(reset,seeds=new_seeds,episodes=[sample_episode('navigation',s,batch.model_id,args.curriculum) for s in new_seeds])
            active=remaining
        report={'format':'serial305-composed-navigation-v1','model_id':batch.model_id,'episodes':len(rows),
                'curriculum':args.curriculum,'execution_source_sha256':source,
                'successes':sum(r['is_success'] for r in rows),'seed_range':[args.seed,args.seed+args.episodes-1],
                'policy_sha256':{k:m['policy_sha256'] for k,m in session.bank.metadata.items()},
                'issuer':'scripted navigation goals; deployed bank handles recovery/escape subphases',
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':sorted(rows,key=lambda r:r['seed'])}
        dump(args.output,report);print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
    finally:session.close()


if __name__=='__main__':main()
