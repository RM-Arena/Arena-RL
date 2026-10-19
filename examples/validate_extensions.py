"""Held-out live-policy checks, including command continuity and native replays."""
import argparse
import hashlib
import json
from pathlib import Path
from arena_rl.bridge import configure
from arena_rl.extended_curricula import sample
from arena_rl.learning import dump
from arena_rl.tactical_runtime import TacticalSession


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policies',default='checkpoints/tactics')
    parser.add_argument('--output',default='runs/extended-runtime')
    parser.add_argument('--episodes',type=int,default=12)
    args=parser.parse_args();root=configure();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    from rmuc_arena.replays import verify_replay
    rows=[];replayed=set()
    for family,seed in [('support',5000000),('flight',6000000)]:
        session=TacticalSession(args.policies,record=True,season='2026' if family=='flight' else '2027',arena_root=root)
        try:
            for n in range(args.episodes):
                episode=sample(family,seed+n)
                session.batch.reset([0],seeds=[seed+n],episodes=[episode])
                session.logs[0]=[{'op':'reset','seed':seed+n,'scenario':episode.scenario}]
                before=session.observe();pid=session.batch.states[0]['info']['simulator_pid']
                session.command({**episode.command,'id':'runtime-'+episode.command['id']})
                after=session.observe();families=set()
                for _ in range(episode.max_steps+1):
                    ex=session.batch.executors[0];families.add(ex.execution_family)
                    if ex.status!='running':break
                    session.advance()
                feedback=session.observe()['feedback']
                row={'seed':seed+n,**feedback,'families_used':sorted(families),
                     'command_preserved_tick':before['tick']==after['tick'],
                     'command_preserved_ammo':before['robot']['ammo']==after['robot']['ammo'],
                     'same_process':pid==session.batch.states[0]['info']['simulator_pid']}
                key=(episode.command['skill'],episode.command.get('flight_direction'),episode.command.get('lane_team'))
                if key not in replayed:
                    path=out/f'{family}-{seed+n}.json';session.save_replay(path)
                    verification=verify_replay(path)
                    row.update(replay=path.name,replay_verified_ticks=verification['verified_ticks'],replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    replayed.add(key)
                rows.append(row);print(json.dumps(row),flush=True)
        finally:session.close()
    report={'episodes':len(rows),'successes':sum(r['status']=='succeeded' for r in rows),
            'continuity_passed':all(r['command_preserved_tick'] and r['command_preserved_ammo'] and r['same_process'] for r in rows),
            'issuer':'scripted external commands; no LLM inference','results':rows}
    dump(out/'report.json',report)
    if report['successes']!=len(rows) or not report['continuity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
