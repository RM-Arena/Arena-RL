"""Verify one physical episode per 305 mobility template through native replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from arena_rl.bridge import configure
from arena_rl.learning import dump
from arena_rl.tactical_envs import TacticalBatch,encode_observation,sample_episode
from arena_rl.tactical_learning import load_tactical_policy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='checkpoints/serial305/mobility')
    parser.add_argument('--output',required=True);args=parser.parse_args();root=configure();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    policy,metadata=load_tactical_policy(args.checkpoint,root);selected={}
    for seed in range(7000200,7001200):
        episode=sample_episode('mobility',seed,305);selected.setdefault(episode.mobility_variant,episode)
        if len(selected)==8:break
    episodes=[selected[k] for k in sorted(selected)]
    batch=TacticalBatch(len(episodes),family='mobility',model_id=305,record=True);rows=[]
    try:
        batch.reset(episodes=episodes);pids=[s['info']['simulator_pid'] for s in batch.states]
        for _ in range(1200):
            ids=np.flatnonzero(~batch.finished).tolist()
            if not ids:break
            features=np.stack([encode_observation(batch.executors[i],batch.states[i]['observations']['red'],'mobility') for i in ids])
            batch.step(policy.predict(features,deterministic=True)[0],env_ids=ids)
        from rmuc_arena.replays import verify_replay
        for i,episode in enumerate(episodes):
            raw=out/f'variant-{episode.mobility_variant}.json';batch.engine.save_replay([raw],env_ids=[i])
            verification=verify_replay(raw);digest=hashlib.sha256(raw.read_bytes()).hexdigest()
            with raw.open('rb') as source,gzip.GzipFile(str(raw)+'.gz','wb',mtime=0) as target:
                import shutil
                shutil.copyfileobj(source,target)
            raw.unlink()
            row={**batch.executors[i].feedback(),'seed':episode.seed,'variant':episode.mobility_variant,
                 'same_simulator_pid':pids[i]==batch.states[i]['info']['simulator_pid'],'replay':raw.name+'.gz',
                 'uncompressed_replay_sha256':digest,'verification':verification}
            if row['status']!='succeeded' or not row['same_simulator_pid']:raise RuntimeError('Physical regression failed: '+json.dumps(row))
            rows.append(row)
        report={'format':'serial305-physical-regressions-v1','policy_sha256':metadata['policy_sha256'],
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows}
        dump(out/'report.json',report);print(json.dumps({'verified':len(rows),'ticks':[r['verification']['verified_ticks'] for r in rows]}),flush=True)
    finally:batch.close()


if __name__=='__main__':main()
