"""Command-line entry points for reproducible training and evaluation."""
import argparse
import json
from pathlib import Path
from .bridge import configure,provenance


def main(argv=None):
    p=argparse.ArgumentParser(prog='arena-rl');p.add_argument('--arena-root')
    sub=p.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train');train.add_argument('--task',choices=['move','ramp','outpost','defend','mixed'],default='move')
    train.add_argument('--steps',type=int,default=32768);train.add_argument('--envs',type=int,default=4)
    train.add_argument('--seed',type=int,default=7);train.add_argument('--output',required=True);train.add_argument('--resume')
    train.add_argument('--allow-engine-change',action='store_true')
    ev=sub.add_parser('evaluate');ev.add_argument('--checkpoint');ev.add_argument('--task',choices=['move','ramp','outpost','defend','mixed'])
    ev.add_argument('--baseline',choices=['random','idle','heuristic']);ev.add_argument('--episodes',type=int,default=16)
    ev.add_argument('--envs',type=int,default=4);ev.add_argument('--seed',type=int,default=1000000)
    ev.add_argument('--output');ev.add_argument('--replays');ev.add_argument('--allow-engine-change',action='store_true')
    sub.add_parser('doctor')
    high=sub.add_parser('train-high');high.add_argument('--skills',required=True);high.add_argument('--steps',type=int,default=4096)
    high.add_argument('--envs',type=int,default=4);high.add_argument('--seed',type=int,default=17);high.add_argument('--output',required=True)
    he=sub.add_parser('evaluate-high');he.add_argument('--checkpoint',required=True);he.add_argument('--episodes',type=int,default=16)
    he.add_argument('--envs',type=int,default=4);he.add_argument('--seed',type=int,default=2000000);he.add_argument('--output');he.add_argument('--replays')
    suite=sub.add_parser('suite');suite.add_argument('--config',default='configs/w3.json');suite.add_argument('--output',required=True)
    args=vars(p.parse_args(argv));command=args.pop('command')
    try:
        if command=='doctor':
            root=configure(args['arena_root']);print(json.dumps(provenance(root),indent=2));return 0
        if command=='train':
            from .learning import train
            train(**args)
        elif command=='evaluate':
            from .learning import evaluate
            evaluate(**args)
        elif command=='train-high':
            from .hierarchy import train_high
            train_high(**args)
        elif command=='evaluate-high':
            from .hierarchy import evaluate_high
            evaluate_high(**args)
        elif command=='suite':
            from .suite import run_suite
            run_suite(**args)
        return 0
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        p.exit(1,f'{type(exc).__name__}: {exc}\n')
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
