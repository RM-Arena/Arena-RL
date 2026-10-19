"""Downsample a native Arena replay into a small trace for figures.

Usage: uv run python examples/extract_match_traces.py runs/team/llm-rl/match-9300001.json out.json.gz [--every 10]
Keeps per-robot position/HP/alive, drone positions and structure HP every N frames (10 = 0.5 s).
"""
import argparse
import gzip
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('replay');parser.add_argument('output');parser.add_argument('--every',type=int,default=10)
    args=parser.parse_args()
    replay=json.loads(Path(args.replay).read_bytes())
    meta=replay['metadata'];frames=replay['frames'];rows=[]
    for index in range(0,len(frames),args.every):
        frame=frames[index]
        robots=[{'id':r['id'],'role':r.get('role'),'x':round(r['x'],2),'y':round(r['y'],2),'hp':round(r.get('hp',0)),'alive':bool(r.get('alive',True)),
                 **({'drones':[[round(d['x'],2),round(d['y'],2),round(d['z'],2),d['state']] for d in r.get('drones',[])]} if r.get('drones') else {})}
                for r in frame.get('robots',[])]
        rows.append({'t':round(frame.get('time',index*meta.get('dt',.05)),2),'robots':robots,
                     'outposts':{k:round(v['hp']) for k,v in frame.get('outposts',{}).items()},'bases':{k:round(v['hp']) for k,v in frame.get('bases',{}).items()},
                     'control':{k:round(v.get('control') or 0,1) for k,v in frame.get('teams',{}).items()}})
    out={'format':'arena-match-trace-v1','source':Path(args.replay).name,'seed':meta.get('seed'),'lineups':meta.get('lineups'),'every_frames':args.every,'frames':rows}
    data=json.dumps(out,separators=(',',':')).encode();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(data,mtime=0) if path.suffix=='.gz' else data)
    print(args.output,len(rows),'samples')


if __name__=='__main__':main()
