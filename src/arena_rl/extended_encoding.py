"""Versioned support/flight policy interfaces, retaining the original combat codec."""
import math
import numpy as np
from gymnasium import spaces
from .commands import NEW_SKILLS
from .executor import own,xy
from .tactical_encoding import encode as base_encode,decode as base_decode,FEATURES as BASE_FEATURES,ACTION_SPACE

EXTRA=['new_'+s for s in NEW_SKILLS]+['ammo_goal','hp_goal','coins','spent','supply_contact','ally_dx','ally_dy','ally_vx','ally_vy',
    'follow_distance','coverage','patrol_progress','dwell','flight_forward','flight_progress','flight_lateral','flight_air','flight_landed']
FEATURES=BASE_FEATURES+EXTRA
OBS_SPACE=spaces.Box(-10.,10.,shape=(len(FEATURES),),dtype=np.float32)
SCHEMAS={'support':'w3-support-v1','flight':'w3-flight-v1'}


def encode(ex,obs):
    from rmuc_arena import sdk
    r=own(obs,ex.slot);c=ex.command;d=ex.extra
    # New intents have their own one-hot; the first five intent bits remain stable.
    core=base_encode(ex,obs)
    ally=next((a for a in obs['robots'] if c and a['id']==c.target_id),None)
    rel=xy(ally)-xy(r) if ally else np.zeros(2)
    progress=lateral=0.
    if c and c.skill=='fly_slope':
        lane=sdk.fly_slope_lane(ex.arena,c.lane_team);lip=np.array(lane['lip']);crest=np.array(lane['crest'])
        axis=(crest-lip)/np.linalg.norm(crest-lip);delta=xy(r)-lip
        progress=float(np.dot(delta,axis))/5.;lateral=float(axis[0]*delta[1]-axis[1]*delta[0])
    values=[float(c is not None and c.skill==s) for s in NEW_SKILLS]
    values += [c.ammo_target/1000 if c else 0,c.hp_target_fraction if c else 0,obs['teams'][obs['team']].get('resources',0)/1000,
               d.get('spent',0)/500,float(d.get('supply_contact',False)),rel[0]/5,rel[1]/5,
               ally['vx']/4 if ally else 0,ally['vy']/4 if ally else 0,c.follow_distance_m/5 if c else 0,
               d.get('coverage',0),d.get('waypoint_index',0)/max(1,d.get('waypoints_total',1)),d.get('dwell_s',0)/5,
               float(c is not None and c.flight_direction=='forward'),progress,lateral,d.get('airborne_s',0),d.get('landing_hold_s',0)]
    result=np.clip(np.concatenate([core,np.asarray(values,dtype=np.float32)]),-10,10)
    if result.shape!=OBS_SPACE.shape:raise RuntimeError('Extended schema mismatch')
    return result.astype(np.float32)


def decode(ex,obs,action):
    from rmuc_arena import sdk
    actions=base_decode(ex,obs,action);out=actions[ex.slot]
    if ex.status!='running':return actions
    c=ex.command;r=own(obs,ex.slot);a=np.asarray(action)
    if c.skill=='resupply':
        out['fire']=False
        if a[4] and ex.extra.get('supply_contact'):
            rule=sdk.rules(obs)['economy']['ammo']['17mm'];batch=int(rule['local_batch']);cost=rule['local_cost']/batch
            remaining=max(0,c.ammo_target-r['ammo']);funds=obs['teams'][obs['team']]['resources']
            limit=min(funds,c.budget_limit-ex.extra.get('spent',0))
            cap=rule['team_purchase_cap']-obs['teams'][obs['team']].get('purchased_17',0)
            amount=min(math.ceil(remaining/batch)*batch,int(max(0,limit)/cost)//batch*batch,int(cap)//batch*batch)
            if amount>0:out['buy_ammo']={'amount':amount,'remote':False}
        # Nominal ride height keeps the physical RFID receiver in range of the card.
        if ex.extra.get('supply_contact'):out['leg_height']=min(.245,out['leg_height'])
    elif c.skill=='fly_slope':
        out['fire']=False
        if r.get('model_slot')==305:out['leg_height']=([.18,.245,.3] if c.flight_direction=='reverse' else [.22,.3,.34])[int(a[5])]
        if a[1]==2:out['move']=[v*.6/.7 for v in out['move']]
        lane=sdk.fly_slope_lane(ex.arena,c.lane_team);axis=np.array(lane['crest'])-lane['lip']
        if c.flight_direction=='reverse':axis=-axis
        out['heading']=math.atan2(axis[1],axis[0])
        out['power']=(150. if r.get('model_slot')==305 and c.flight_direction=='reverse' else 200.) if a[4] else 100.
        out['lean']=(int(a[2])-1)*.12
        out['jump']=float(a[6]) if ex.skill_state.get('launched') else 0.
        if ex.plan['phase']=='flight_land':out['jump']=0.
        if ex.plan['phase']=='flight_align':out.update(move=[0.,0.],leg_height=.245,jump=0.,power=100.)
    return actions
