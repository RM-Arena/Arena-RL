"""Public-map straight axle transfers over the 2027 road bumps."""
import math
import numpy as np


def road_transfer(ex,robot,goal):
    if ex.model_id!=305:return None
    p=np.array([robot['x'],robot['y']]);state=ex.route_state
    transfer=state.get('road_transfer')
    if transfer is None:
        for t in ex.arena['terrain']:
            if t['kind']!='bumps' or t.get('axis')!='x':continue
            x0,x1=t['x'],t['x']+t['w'];y0,y1=t['y'],t['y']+t['h']
            inside=x0<=p[0]<=x1 and y0<=p[1]<=y1
            crosses=min(p[0],goal[0])<x1 and max(p[0],goal[0])>x0
            same_lane=y0+.32<p[1]<y1-.32 and y0+.32<goal[1]<y1-.32
            if not inside and not (same_lane and crosses and x0-.8<p[0]<x1+.8):continue
            sign=1 if goal[0]>p[0] else -1
            if not same_lane:
                from .navigation_grid import route_distance
                exits=[np.array([x0-.3,np.clip(p[1],y0+.4,y1-.4)]),np.array([x1+.3,np.clip(p[1],y0+.4,y1-.4)])]
                end=min(exits,key=lambda q:np.linalg.norm(q-p)+route_distance(q,goal,ex.navigation_arena,robot['team']))
                sign=1 if end[0]>p[0] else -1
            y=float(np.clip(p[1],y0+.4,y1-.4))
            entry=x0-.35 if sign>0 else x1+.35
            exit_=x1+.4 if sign>0 else x0-.4
            transfer={'tile':t['id'],'sign':sign,'y':y,'entry':entry,'exit':exit_,'crossing':bool(inside)}
            state['road_transfer']=transfer;state.pop('escape_until',None);break
    if transfer is None:return None
    sign=transfer['sign'];y=transfer['y']
    if transfer['crossing'] and sign*(p[0]-transfer['exit'])>=0:
        state.pop('road_transfer');state.pop('escape_until',None);state['moved_at']=ex.time
        return None
    if not transfer['crossing']:
        entry=np.array([transfer['entry'],y])
        heading_error=min(abs((robot['heading']+math.pi)%math.tau-math.pi),abs((robot['heading'])%math.tau-math.pi))
        if np.linalg.norm(p-entry)<.22 and heading_error<.12:transfer['crossing']=True
        elif sign*(p[0]-entry[0])>0 and abs(p[1]-y)<.12 and heading_error<.12:transfer['crossing']=True
        else:return entry-p,None
    # Do not turn sideways toward an SDK waypoint while either wheel is on a
    # transverse hump. The network still chooses drive/stop, speed and gait.
    heading=min((0.,math.pi),key=lambda a:abs((a-robot['heading']+math.pi)%math.tau-math.pi))
    return np.array([float(sign),0.]),heading
