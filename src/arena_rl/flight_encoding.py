"""Phase-aware 305 fly-slope observations; legacy W3 flight stays at 74 features."""
import numpy as np
from gymnasium import spaces
from .extended_encoding import FEATURES as BASE_FEATURES,encode as base_encode
from .tactical_encoding import FEATURES as TACTICAL_FEATURES,encode as route_encode
from .executor import own,xy

SCHEMA='serial305-flight-v3'
PHASES=('flight_approach','flight_align','flight_run','flight_air','flight_land','recover')
FEATURES=BASE_FEATURES+list(PHASES)+['flight_launched','flight_in_gap','flight_along_speed','landing_distance_error']
OBS_SPACE=spaces.Box(-10.,10.,shape=(len(FEATURES),),dtype=np.float32)


def encode(ex,obs):
    if ex.command is None:return np.zeros(OBS_SPACE.shape,dtype=np.float32)
    from rmuc_arena import sdk
    r=own(obs,ex.slot);lane=sdk.fly_slope_lane(ex.arena,ex.command.lane_team)
    crest=np.asarray(lane['crest']);axis=crest-np.asarray(lane['lip']);axis/=np.linalg.norm(axis)
    end=np.asarray(lane['landing'])-axis*.75;p=xy(r)
    in_gap=np.dot(p-crest,axis)>0 and np.dot(p-end,axis)<0
    extra=[float(ex.plan['phase']==phase) for phase in PHASES]
    direction=1. if ex.command.flight_direction=='forward' else -1.
    extra += [float(ex.skill_state.get('launched',False)),float(in_gap),float(np.dot([r['vx'],r['vy']],axis)*direction)/4.,(ex.plan['distance']-.65)/5.]
    legacy=base_encode(ex,obs);core=route_encode(ex,obs,policy_family='navigation')
    # Ground control should not change when a previous skill rotates the gimbal,
    # buys ammo, changes team coins or approaches the mirror lane. Keep the
    # versioned slots, but use a lane-relative physical frame and mask unrelated
    # support/combat inputs for this dedicated controller.
    course=axis*direction;side=np.array([-course[1],course[0]])
    for x_name,y_name in [('vx','vy'),('hint_x','hint_y'),('goal_dx','goal_dy')]:
        ix,iy=TACTICAL_FEATURES.index(x_name),TACTICAL_FEATURES.index(y_name)
        value=core[[ix,iy]].copy();core[ix]=np.dot(value,course);core[iy]=np.dot(value,side)
    angle=r['heading']-np.arctan2(course[1],course[0])
    core[TACTICAL_FEATURES.index('heading_sin')]=np.sin(angle);core[TACTICAL_FEATURES.index('heading_cos')]=np.cos(angle)
    core[TACTICAL_FEATURES.index('hp')]=1.
    if course[0]<0:
        a,b,c,d=[TACTICAL_FEATURES.index(n) for n in ['terrain_e','terrain_w','terrain_n','terrain_s']]
        core[[a,b,c,d]]=core[[b,a,d,c]]
    result=np.zeros(len(FEATURES),dtype=np.float32);result[:len(core)]=core
    for name in ['new_fly_slope','flight_forward','flight_progress','flight_lateral','flight_air','flight_landed']:
        index=BASE_FEATURES.index(name);result[index]=legacy[index]
    result[len(BASE_FEATURES):]=extra
    return np.clip(result,-10.,10.).astype(np.float32)
