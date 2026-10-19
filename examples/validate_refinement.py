"""Verify original failures through the deployed bank; optionally run a full 2026 match."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from arena_rl.bridge import configure
from arena_rl.executor import own
from arena_rl.learning import dump
from arena_rl.tactical_envs import LINEUPS,sample_episode
from arena_rl.tactical_runtime import TacticalSession


def save(session,out,name,env_id):
    from rmuc_arena.replays import verify_replay
    path=out/(name+'.json');session.save_replay(path,env_id);verification=verify_replay(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open('rb') as source,gzip.GzipFile(str(path)+'.gz','wb',mtime=0) as target:shutil.copyfileobj(source,target)
    path.unlink()
    return {'replay':path.name+'.gz','uncompressed_replay_sha256':digest,'verification':verification}


def main():
    p=argparse.ArgumentParser();p.add_argument('--policies',default='checkpoints/serial305');p.add_argument('--output',required=True)
    p.add_argument('--full-flight',action='store_true');args=p.parse_args();root=configure();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    lineups=copy.deepcopy(LINEUPS);lineups['red'][0]['model']=305
    session=TacticalSession(args.policies,envs=1,lineups=lineups,record=True)
    try:
        episode=sample_episode('navigation',3000013,305);batch=session.batch
        batch.reset(seeds=[episode.seed],episodes=[episode]);pid=batch.states[0]['info']['simulator_pid']
        for _ in range(280):
            if batch.executors[0].status!='running':break
            session.advance()
        ex=batch.executors[0];assert ex.status=='succeeded' and ex.command.id==episode.command['id']
        assert batch.states[0]['info']['simulator_pid']==pid
        rows.append({'kind':'navigation','seed':episode.seed,'budget_s':28.,**ex.feedback(),**save(session,out,'navigation-3000013',0)})
    finally:session.close()
    flight_lineups={t:[{'slot':3,'model':305 if t=='red' else 3}] for t in ('red','blue')}
    session=TacticalSession(args.policies,envs=4,lineups=flight_lineups,season='2026',record=True)
    try:
        seeds=[6000011,6000016,6000020,6000021];batch=session.batch
        batch.reset(seeds=seeds,episodes=[sample_episode('flight',s,305) for s in seeds]);pids=[s['info']['simulator_pid'] for s in batch.states]
        for _ in range(180):
            if all(e.status!='running' for e in batch.executors):break
            session.advance()
        for i,seed in enumerate(seeds):
            ex=batch.executors[i];assert ex.status=='succeeded' and ex.extra['airborne_s']+1e-8>=.1 and ex.extra['landing_hold_s']+1e-8>=.3
            assert batch.states[i]['info']['simulator_pid']==pids[i]
            rows.append({'kind':'flight','seed':seed,'budget_s':18.,**ex.feedback(),**save(session,out,f'flight-{seed}',i)})
    finally:session.close()
    if args.full_flight:
        # Declared staging start, scripted objective issuer, normal 420-second referee.
        from arena_rl.match_opponents import LINEUPS as MATCH_LINEUPS,actions
        lineups=copy.deepcopy(MATCH_LINEUPS);lineups['red'][0]['model']=305;lineups['blue'][0]['model']=1
        session=TacticalSession(args.policies,envs=1,lineups=lineups,season='2026',record=True)
        try:
            scenario={'robots':{'red-3':{'x':10.,'y':14.4,'heading':0.,'ammo':0}}}
            session.reset([9100030],[scenario]);batch=session.batch;pid=batch.states[0]['info']['simulator_pid']
            batch.opponent_controller=lambda i,obs,arena:actions(obs,arena,'raid')
            completed=[];session.command({'id':'continuous-forward','skill':'fly_slope','lane_team':'red','timeout_s':18.,'low_hp_fraction':0.})
            for _ in range(4200):
                scene=session.observe();ex=batch.executors[0]
                if scene['episode_finished']:break
                if ex.status=='failed':raise RuntimeError('Continuous crossing failed: '+json.dumps(ex.feedback()))
                if ex.status=='succeeded' and len(completed)<2:
                    completed.append(ex.feedback())
                    if len(completed)==1:session.command({'id':'continuous-reverse','skill':'fly_slope','lane_team':'red','flight_direction':'reverse','timeout_s':18.,'low_hp_fraction':0.})
                session.advance()
            scene=session.observe();assert scene['episode_finished'] and len(completed)==2 and batch.states[0]['terminated']
            assert batch.states[0]['info']['simulator_pid']==pid
            rows.append({'kind':'full_2026','issuer':'scripted forward/reverse objectives, then stationary; blue raid opponents',
                'initial_scenario':scenario,'seed':9100030,'time_s':scene['time'],'completed_goals':completed,
                'winner':batch.states[0]['info']['winner'],**save(session,out,'full-2026',0)})
        finally:session.close()
    dump(out/'report.json',{'format':'serial305-refinement-regressions-v1','model_id':305,
        'policy_sha256':{k:m['policy_sha256'] for k,m in session.bank.metadata.items()},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows})
    print(json.dumps({'verified':len(rows),'ticks':[r['verification']['verified_ticks'] for r in rows]}),flush=True)


if __name__=='__main__':main()
