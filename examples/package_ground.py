"""Copy trained universal policies into a shippable bank with a digest manifest.

Usage: uv run python examples/package_ground.py runs/ground-v1 checkpoints/ground
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

FILES=('policy.zip','metadata.json','metrics.jsonl','evaluation.json','demonstrations.npz')


def main(source,target):
    source=Path(source);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    manifest={'format':'arena-ground-bank-v1','families':{}}
    for family in sorted(p.name for p in source.iterdir() if (p/'metadata.json').is_file() and (p/'policy.zip').is_file()):
        out=target/family;out.mkdir(exist_ok=True)
        for name in FILES:
            if (source/family/name).is_file():shutil.copy2(source/family/name,out/name)
        meta=json.loads((out/'metadata.json').read_text())
        entry={'policy_sha256':meta['policy_sha256'],'actual_steps':meta.get('actual_steps'),'schema':meta.get('schema'),
               'files':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in FILES if (out/n).is_file()}}
        if (out/'evaluation.json').is_file():
            report=json.loads((out/'evaluation.json').read_text())
            entry['evaluation']=report.get('mean_success_rate') or {k:v.get('mean_strikes') for k,v in report.get('modes',{}).items()}
        manifest['families'][family]=entry
    (target/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v.get('evaluation') for k,v in manifest['families'].items()},ensure_ascii=False))


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
