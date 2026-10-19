"""Markdown tables from evaluate-ground / evaluate-aerial reports.

Usage: uv run python examples/summarize_ground.py checkpoints/ground
"""
import json
import sys
from pathlib import Path

ROLE={3:'步兵 W3',4:'步兵 I4',7:'步兵 I7',301:'步兵 I-T',302:'步兵 I-S',303:'步兵 I-O',304:'步兵 I-W',305:'步兵 I-L',2731:'步兵 I27-W',2732:'步兵 I27-T',
      2711:'重型 HV-A',2712:'重型 HV-D',2713:'重型 HV-O',2771:'哨兵 S27-O',2772:'哨兵 S27-L'}


def main(root):
    root=Path(root);families=[f for f in ('navigation','combat','defense','support') if (root/f/'evaluation.json').is_file()]
    reports={f:json.loads((root/f/'evaluation.json').read_text()) for f in families}
    # Per-model overrides (e.g. combat-heavy) replace the universal rows for their models.
    for f in families:
        for extra in sorted(root.glob(f'{f}-*/evaluation.json')):
            override=json.loads(extra.read_text())
            for model,row in override['models'].items():reports[f]['models'][model]={**row,'override':extra.parent.name}
        rows=reports[f]['models'].values();modes=next(iter(rows)).keys()
        reports[f]['mean_success_rate']={m:sum(r[m]['success_rate'] for r in rows if m in r)/len(rows) for m in modes if m!='override'}
    print('| 族 | 策略 | 专家 | 随机 | 每型号回合 |\n| --- | --- | --- | --- | --- |')
    for f,r in reports.items():
        m=r['mean_success_rate'];print(f"| {f} | {m.get('policy',0):.2f} | {m.get('expert',0):.2f} | {m.get('random',0):.2f} | {r['episodes_per_model']} |")
    print('\n| 型号 | 底盘 | '+' | '.join(f'{f} 策略/专家' for f in families)+' |\n| --- | --- | '+' | '.join('---' for _ in families)+' |')
    models=sorted({m for r in reports.values() for m in r['models']},key=int)
    for model in models:
        cells=[]
        for f in families:
            row=reports[f]['models'].get(model,{})
            cells.append(f"{row.get('policy',{}).get('success_rate',0):.2f} / {row.get('expert',{}).get('success_rate',0):.2f}")
        first=next((reports[f]['models'][model] for f in families if model in reports[f]['models']),{})
        drive=next(iter(first.get('expert',first.get('policy',{})).get('by_class',{}).keys()),'').split('/')[-1]
        print(f"| {model} {ROLE.get(int(model),'')} | {drive} | "+' | '.join(cells)+' |')
    if any('override' in r for f in families for r in reports[f]['models'].values()):
        print('\n覆盖策略：'+'；'.join(sorted({f"{r['override']}（{m}）" for f in families for m,r in reports[f]['models'].items() if 'override' in r})))
    aerial=root/'aerial'/'evaluation.json'
    if aerial.is_file():
        r=json.loads(aerial.read_text())
        print('\n| 空中 | 平均有效撞击 | 平均伤害 | 有命中的回合 | 首次命中中位数 (s) |\n| --- | --- | --- | --- | --- |')
        for mode,v in r['modes'].items():
            print(f"| {mode} | {v['mean_strikes']:.2f} | {v['mean_damage']:.0f} | {v['episodes_with_hit']}/{r['episodes']} | {v['median_first_hit_s']} |")


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'checkpoints/ground')
