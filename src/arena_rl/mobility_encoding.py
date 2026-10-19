"""Serial-305 mobility policy: new schema; old tactical features stay unchanged."""
import numpy as np
from gymnasium import spaces
from .executor import own
from .tactical_encoding import FEATURES as BASE_FEATURES,encode as base_encode,decode as base_decode,ACTION_SPACE

SCHEMA='serial305-mobility-v1'
FEATURES=BASE_FEATURES+['retreat_intent','recovery_phase','qx','qy','qz','qw','wx','wy','wz','self_righting','stable_hold','escape_attempts']
OBS_SPACE=spaces.Box(-10.,10.,shape=(len(FEATURES),),dtype=np.float32)


def encode(ex,obs):
    if ex.command is None:return np.zeros(OBS_SPACE.shape,dtype=np.float32)
    r=own(obs,ex.slot)
    core=base_encode(ex,obs,policy_family='navigation' if ex.plan['phase']!='recover' else None)
    if ex.plan['phase']=='recover':core[:5]=0.
    if ex.route_state.get('escape_until',0)>ex.time:core[BASE_FEATURES.index('radius')]=.4/3
    extra=[float(ex.command.skill!='recover' and ex.plan['phase']!='recover'),float(ex.plan['phase']=='recover')]+r.get('quaternion',[0.,0.,0.,1.])+[v/5 for v in r.get('angular_velocity',[0.,0.,0.])]
    extra+=[float(r.get('self_righting',False)),ex.recovery.get('hold',0.),ex.route_state.get('attempts',0)/10]
    return np.clip(np.concatenate([core,np.asarray(extra,dtype=np.float32)]),-10,10).astype(np.float32)


def decode(ex,obs,action):
    out=base_decode(ex,obs,action,policy_family='navigation');r=own(obs,ex.slot);a=np.asarray(action)
    command=out[ex.slot];command['fire']=False
    if ex.plan['phase']=='recover' and ex.status=='running':
        # The 305's existing full-turn hip firmware supplies bounded physical joint
        # torques. The learned action chooses settling/gait; no pose is overwritten.
        command['heading']=r['heading'];command['power']=100.
        command['leg_height']=[.12,.245,.31][int(a[5])]
        command['jump']=0.
        if r['upright']<.8 or r.get('self_righting',False):command['move']=[0.,0.]
        command['lean']=(int(a[2])-1)*.12
    return out
