"""Bounded physical re-approach of a ramp or landing lip; never changes pose."""
import math
import numpy as np


def guidance(ex,r,lane,goal,phase):
    st=ex.skill_state;now=ex.time;p=np.array([r['x'],r['y']])
    if ex.model_id!=305 or not st.get('launched'):return None
    crest=np.asarray(lane['crest']);axis=crest-np.asarray(lane['lip']);axis/=np.linalg.norm(axis)
    forward=ex.command.flight_direction=='forward';course=axis if forward else -axis
    end=np.asarray(lane['landing'])-axis*.75
    in_gap=np.dot(p-crest,axis)>0 and np.dot(p-end,axis)<0
    previous=np.asarray(st.setdefault('guidance_position',p.tolist()))
    st.setdefault('guidance_moved_at',now)
    if np.linalg.norm(p-previous)>.12:st.update(guidance_position=p.tolist(),guidance_moved_at=now)
    retry=st.get('flight_retry')
    if retry is None and r['grounded'] and r['upright']>.85 and now>st.get('retry_cooldown',0):
        stalled=now-st['guidance_moved_at']>.75
        near_lip=np.dot(p-crest,axis)>-.05 and np.dot(p-end,axis)<.35
        if stalled and near_lip and r['z']<lane['landing_height']+.12 and forward:
            retry={'kind':'gap_stage','goal':(((crest+end)/2)+axis*.1).tolist()};st['flight_retry']=retry
        elif stalled and not in_gap and phase=='flight_run' and abs((r['heading']-math.atan2(course[1],course[0])+math.pi)%math.tau-math.pi)>.3:
            axle=np.array([math.cos(r['heading']),math.sin(r['heading'])])
            sign=1 if np.dot(np.asarray(lane['lip'])-p,axle)>0 else -1
            retreat=p+axle*sign*.45
            retry={'kind':'backoff','goal':np.clip(retreat,[.35,.35],[ex.arena['width']-.35,ex.arena['height']-.35]).tolist(),'heading':r['heading']};st['flight_retry']=retry
    if retry is None:return None
    kind=retry['kind']
    if kind in ('backoff','gap_stage'):
        target=np.asarray(retry['goal']);distance=np.linalg.norm(target-p)
        if distance>(.25 if kind=='backoff' else .18):
            return 'flight_approach',target,target-p,retry.get('heading')
        if kind=='backoff':
            st.pop('flight_retry');st.pop('launched',None);st['align_hold']=0.;st['air_s']=0.;st['retry_cooldown']=now+1.
            entry=np.asarray(lane['lip'])-axis*2.55 if forward else np.asarray(lane['landing'])+axis*.45
            return 'flight_approach',entry,entry-p,None
        retry['kind']='gap_align';retry['hold']=0.;kind='gap_align'
    if kind=='gap_align':
        heading=math.atan2(course[1],course[0]);error=abs((r['heading']-heading+math.pi)%math.tau-math.pi)
        if r['grounded'] and r['upright']>.95 and error<.12 and math.hypot(r['vx'],r['vy'])<.3:
            retry['hold']=retry.get('hold',0.)+max(0.,now-retry.get('observed_at',now))
        else:retry['hold']=0.
        retry['observed_at']=now
        if retry['hold']<.3:return 'flight_align',p,np.zeros(2),heading
        retry.update(kind='gap_jump',until=now+.6);kind='gap_jump'
    if kind=='gap_jump':
        if now>retry['until']:
            st.pop('flight_retry');st['retry_cooldown']=now+1.;st['guidance_moved_at']=now;return None
        return 'flight_run',goal,course,None
    return None
