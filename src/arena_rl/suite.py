"""Run the staged single-robot training and evaluation recipe."""
import json
from pathlib import Path
from .learning import train,evaluate,dump
from .hierarchy import train_high,evaluate_high


def run_suite(*,config,output,arena_root=None):
    recipe=json.loads(Path(config).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if recipe['robot_model']!=3 or recipe['season']!='2027':
        raise ValueError('This recipe currently supports only W3 model 3 and season 2027')
    report={'recipe':recipe,'skills':{}}
    for task in ('move','ramp','outpost','defend'):
        path=output/task
        baseline=evaluate(task=task,baseline='random',episodes=recipe['eval_episodes'],envs=recipe['envs'],
                          seed=recipe['eval_seed'],arena_root=arena_root,output=output/f'{task}-random.json')
        if not (path/'metadata.json').exists():
            train(task=task,steps=recipe['skill_steps'],envs=recipe['envs'],seed=recipe['train_seed'],
                  output=path,arena_root=arena_root)
        result=evaluate(checkpoint=path,episodes=recipe['eval_episodes'],envs=recipe['envs'],seed=recipe['eval_seed'],
                        arena_root=arena_root,output=path/'evaluation.json',replays=path/'replays')
        report['skills'][task]={'random_success_rate':baseline['success_rate'],'trained_success_rate':result['success_rate']}
        dump(output/'report.json',report)
    # Low-level models must have at least one held-out success before a manager is trained.
    failed=[name for name,row in report['skills'].items() if row['trained_success_rate']==0]
    if failed:raise RuntimeError(f'No evaluation success for skills {failed}; improve or continue them before training the manager')
    if not (output/'manager'/'metadata.json').exists():
        train_high(skills=output,steps=recipe['manager_steps'],envs=recipe['envs'],seed=recipe['train_seed']+10,
                   output=output/'manager',arena_root=arena_root)
    high=evaluate_high(checkpoint=output/'manager',episodes=recipe['eval_episodes'],envs=recipe['envs'],
                       seed=recipe['eval_seed']+1000000,output=output/'manager'/'evaluation.json',
                       replays=output/'manager'/'replays',arena_root=arena_root)
    report['manager']={k:v for k,v in high.items() if k!='results'};dump(output/'report.json',report)
    return report
