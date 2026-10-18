"""Evaluate current tactical checkpoints and preserve every result; no LLM is invoked."""
import argparse
from pathlib import Path
from arena_rl.tactical_learning import evaluate_tactics
from arena_rl.tactical_runtime import run_mission


def main():
 p=argparse.ArgumentParser();p.add_argument('--policies',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 output=Path(args.output)
 for family in ('navigation','combat','defense'):
  evaluate_tactics(checkpoint=Path(args.policies)/family,episodes=32,output=output/f'{family}.json',replays=output/family)
 for seed in range(4000000,4000003):
  run_mission(policies=args.policies,seed=seed,output=output/f'mission-{seed}')
if __name__=='__main__':main()
