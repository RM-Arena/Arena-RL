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
    tactical=sub.add_parser('train-tactics');tactical.add_argument('--expert-episodes',type=int);tactical.add_argument('--family',choices=['navigation','combat','defense','support','flight'],required=True)
    tactical.add_argument('--steps',type=int,default=32768);tactical.add_argument('--envs',type=int,default=4)
    tactical.add_argument('--seed',type=int,default=107);tactical.add_argument('--output',required=True);tactical.add_argument('--resume')
    tev=sub.add_parser('evaluate-tactics');tev.add_argument('--checkpoint');tev.add_argument('--family',choices=['navigation','combat','defense','support','flight'])
    tev.add_argument('--baseline',choices=['random','idle','heuristic']);tev.add_argument('--episodes',type=int,default=32)
    tev.add_argument('--envs',type=int,default=4);tev.add_argument('--seed',type=int,default=3000000)
    tev.add_argument('--output');tev.add_argument('--replays')
    live=sub.add_parser('serve-tactics');live.add_argument('--policies',required=True);live.add_argument('--envs',type=int,default=1)
    live.add_argument('--seed',type=int,default=207);live.add_argument('--realtime',action='store_true');live.add_argument('--speed',type=float,default=1.)
    live.add_argument('--record',action='store_true');live.add_argument('--opponent');live.add_argument('--lineups');live.add_argument('--season',choices=['2026','2027'],default='2027')
    llm=sub.add_parser('llm-matches');llm.add_argument('--policies',required=True);llm.add_argument('--output',required=True)
    llm.add_argument('--matches',type=int,default=3);llm.add_argument('--seed',type=int,default=9000000)
    llm.add_argument('--model',default='deepseek-flash');llm.add_argument('--base-url',default='https://api.deepseek.com')
    llm.add_argument('--api-key-env',default='DEEPSEEK_API_KEY');llm.add_argument('--timeout',type=float,default=8.)
    llm.add_argument('--plan-interval',type=float,default=5.);llm.add_argument('--max-age',type=float,default=5.)
    llm.add_argument('--max-calls',type=int,default=120);llm.add_argument('--speed',type=float,default=1.)
    llm.add_argument('--opponent-profiles',nargs='+',choices=['siege','raid','hunt'])
    llm.add_argument('--no-verify',action='store_false',dest='verify',default=True)
    demo=sub.add_parser('mission');demo.add_argument('--policies',required=True);demo.add_argument('--seed',type=int,default=4000000)
    demo.add_argument('--output',required=True)
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
        elif command=='train-tactics':
            from .tactical_learning import train_tactics
            train_tactics(**args)
        elif command=='evaluate-tactics':
            from .tactical_learning import evaluate_tactics
            evaluate_tactics(**args)
        elif command=='serve-tactics':
            from .tactical_runtime import serve_tactics
            serve_tactics(**args)
        elif command=='llm-matches':
            from .llm_matches import run_llm_matches
            return 0 if run_llm_matches(**args)['healthy_control'] else 1
        elif command=='mission':
            from .tactical_runtime import run_mission
            return 0 if run_mission(**args)['success'] else 1
        elif command=='suite':
            from .suite import run_suite
            run_suite(**args)
        return 0
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        p.exit(1,f'{type(exc).__name__}: {exc}\n')
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
