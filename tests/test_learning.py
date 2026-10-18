import json
import os
from pathlib import Path
import numpy as np
import pytest
from arena_rl.bridge import configure
from arena_rl.learning import train,load_policy
from arena_rl.hierarchy import HierarchicalVecEnv,train_high,evaluate_high

ROOT=configure()


def test_real_ppo_updates_weights_and_rejects_tampered_checkpoint(tmp_path):
    path=tmp_path/'move'
    meta=train(task='move',steps=512,envs=2,seed=9,output=path,arena_root=ROOT)
    assert meta['weights_changed'] and meta['ppo_updates']>0
    assert meta['weights_before']!=meta['weights_after'] and meta['actual_steps']==512
    model,loaded=load_policy(path,ROOT);assert loaded['policy_sha256']==meta['policy_sha256']
    with (path/'policy.zip').open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='digest'):load_policy(path,ROOT)


def test_hierarchy_freezes_skills_and_does_not_disable_invaders(tmp_path):
    for task in ('outpost','defend'):
        train(task=task,steps=256,envs=2,seed=9,output=tmp_path/task,arena_root=ROOT)
    env=HierarchicalVecEnv(tmp_path,1,seed=0,arena_root=ROOT)
    try:
        obs=env.reset();assert env.context[0]==1
        before=env.batch.states[0]['observations']['blue']['robots'][0]['x']
        env.step(np.array([0]))  # Attack while an invader is present.
        assert env.batch.states[0]['observations']['blue']['robots'][0]['x']<before
        assert env.batch.invaders[0]
    finally:env.close()
    meta=train_high(skills=tmp_path,steps=128,envs=2,seed=9,output=tmp_path/'manager',arena_root=ROOT)
    assert meta['weights_changed'] and meta['skills_frozen']
    result=evaluate_high(checkpoint=tmp_path/'manager',episodes=2,envs=2,seed=2000000,arena_root=ROOT)
    assert len(result['results'])==2
