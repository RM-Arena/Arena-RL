"""Conservative public-map route hints for W3's local learned controller."""
import copy
import hashlib
import json


def navigation_arena(arena,model_id=3):
    """Avoid the assembly plinth unless a future skill explicitly handles it.

    Arena's SDK permits the bevel in its generic graph. W3 can get its wheel
    trapped against its edge while turning. Routing around its public footprint
    does not alter simulator geometry, collisions, terrain or robot performance.
    """
    result=copy.deepcopy(arena)
    for t in arena.get('terrain',[]):
        if t.get('id')=='assembly_plinth':
            result['obstacles'].append({'id':'tactical_avoid_'+t['id'],'x':t['x'],'y':t['y'],
                'w':t['w'],'h':t['h'],'height':1.,'bottom':0.,'rotation':0.,'kind':'barrier'})
    if model_id==305:
        import math
        nodes=result['navigation']['nodes']
        for t in arena['terrain']:
            if t['kind']!='ramp' or 'ramp_start' not in t:continue
            a,b=t['ramp_start'],t['ramp_end'];dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy)
            if length<.01:continue
            for x,y in (a,b,[(a[0]+b[0])/2,(a[1]+b[1])/2],[a[0]-dx/length*.45,a[1]-dy/length*.45],[b[0]+dx/length*.45,b[1]+dy/length*.45]):
                nodes.append([x,y])
    # SDK caches graphs by revision; the derived public geometry needs its own key.
    digest=hashlib.sha256(json.dumps([result['obstacles'],result['terrain'],result['navigation']['nodes']],sort_keys=True).encode()).hexdigest()[:12]
    result['layout_revision']=arena.get('layout_revision','map')+':w3-tactical-'+digest
    return result


def safe_hold_point(robot,center,radius,arena):
    """A defense center may be a base's center; stand beside its solid pedestal."""
    import math
    margin=robot.get('radius',.3)+.08
    def clear(point):
        x,y=point
        if not margin<=x<=arena['width']-margin or not margin<=y<=arena['height']-margin:return False
        boxes=list(arena['obstacles'])
        boxes += [w for w in arena.get('forbidden',[]) if w['kind'] in ('base','dart_station') or w.get('owner') not in (None,robot['team'])]
        for w in boxes:
            if w.get('bottom',0)>.8:continue
            angle=w.get('rotation',0);c,s=math.cos(angle),math.sin(angle)
            dx,dy=x-w['x']-w['w']/2,y-w['y']-w['h']/2
            if abs(dx*c+dy*s)<w['w']/2+margin and abs(-dx*s+dy*c)<w['h']/2+margin:return False
        return True
    candidates=[tuple(center)]
    for scale in (.35,.6,.85):
        for k in range(16):
            angle=k*math.tau/16;candidates.append((center[0]+radius*scale*math.cos(angle),center[1]+radius*scale*math.sin(angle)))
    feasible=[p for p in candidates if clear(p)]
    return min(feasible,key=lambda p:(p[0]-robot['x'])**2+(p[1]-robot['y'])**2) if feasible else None


def clearance_hint(robot,hint,arena):
    """Keep a two-wheel chassis square until both wheels clear a road-bump edge.

    A center just outside the footprint can still have a wheel on the first hump.
    Turning there pins the axle. Back/drive along its existing heading to a free
    end before resuming the route; physics and legal speed/power remain unchanged.
    """
    import math
    if robot.get('chassis_type')!='wheel_legged' or math.hypot(*hint)<.001:return tuple(hint),None
    heading=robot['heading'];forward=(math.cos(heading),math.sin(heading))
    error=abs((math.atan2(hint[1],hint[0])-heading+math.pi)%math.tau-math.pi)
    error=min(error,math.pi-error)
    if error<math.radians(25):return tuple(hint),None
    for t in arena['terrain']:
        if t['kind']!='bumps':continue
        x0,y0,x1,y1=t['x'],t['y'],t['x']+t['w'],t['y']+t['h']
        x,y=robot['x'],robot['y'];margin=robot.get('radius',.3)+.12
        if not x0-margin<x<x1+margin or not y0-margin<y<y1+margin:continue
        if abs(forward[0])>.7:
            direction=-1 if x-x0<x1-x else 1
            sign=1 if forward[0]*direction>0 else -1
        else:
            direction=-1 if y-y0<y1-y else 1
            sign=1 if forward[1]*direction>0 else -1
        return (forward[0]*sign,forward[1]*sign),heading
    return tuple(hint),None


def clear_point(point,robot,arena):
    """Conservative chassis-center clearance on the derived public route map."""
    import math
    x,y=point;margin=robot.get('radius',.3)+.1
    if not margin<x<arena['width']-margin or not margin<y<arena['height']-margin:return False
    walls=list(arena['obstacles'])+[w for w in arena.get('forbidden',[]) if w['kind'] in ('base','pad','highway','dart_station') or w.get('owner') not in (None,robot['team'])]
    for w in walls:
        if 'roof' in w.get('id',''):continue
        angle=w.get('rotation',0);c,s=math.cos(angle),math.sin(angle)
        dx,dy=x-w['x']-w['w']/2,y-w['y']-w['h']/2
        if abs(dx*c+dy*s)<w['w']/2+margin and abs(-dx*s+dy*c)<w['h']/2+margin:return False
    return True


def firing_point(robot,target,distance,arena):
    """Pick a clear firing standoff instead of steering directly onto deck kerbs."""
    import math
    candidates=[]
    for k in range(24):
        a=k*math.tau/24;p=(target['x']+distance*math.cos(a),target['y']+distance*math.sin(a))
        if clear_point(p,robot,arena):candidates.append(p)
    return min(candidates,key=lambda p:(p[0]-robot['x'])**2+(p[1]-robot['y'])**2) if candidates else None


def plan_motion(ex,robot,goal,now):
    import math
    import numpy as np
    from rmuc_arena import sdk
    p=np.array([robot['x'],robot['y']]);state=ex.route_state
    if 'position' not in state:state.update(position=p.copy(),moved_at=now,attempts=0)
    if np.linalg.norm(p-state['position'])>.16:state.update(position=p.copy(),moved_at=now)
    held=None
    if now-state['moved_at']>1.5 and np.linalg.norm(np.array(goal)-p)>ex.command.radius_m:
        state['attempts']+=1;state['moved_at']=now
        directions=[]
        for wall in ex.arena['obstacles']:
            if 'roof' in wall.get('id',''):continue
            angle=wall.get('rotation',0);c,s=math.cos(angle),math.sin(angle)
            delta=p-[wall['x']+wall['w']/2,wall['y']+wall['h']/2]
            local=np.array([delta[0]*c+delta[1]*s,-delta[0]*s+delta[1]*c])
            half=np.array([wall['w']/2,wall['h']/2])
            if np.all(abs(local)<half+robot.get('radius',.3)+.25):
                axis=int(np.argmax(abs(local)-half));out=np.zeros(2);out[axis]=1. if local[axis]>=0 else -1.
                directions.append(np.array([out[0]*c-out[1]*s,out[0]*s+out[1]*c]))
        held=robot['heading'];forward=np.array([math.cos(held),math.sin(held)])
        def clearance(point):
            score=10.
            for wall in ex.arena['obstacles']:
                if 'roof' in wall.get('id',''):continue
                angle=wall.get('rotation',0);c,s=math.cos(angle),math.sin(angle)
                delta=point-[wall['x']+wall['w']/2,wall['y']+wall['h']/2]
                score=min(score,max(abs(delta[0]*c+delta[1]*s)-wall['w']/2,abs(-delta[0]*s+delta[1]*c)-wall['h']/2))
            height=sdk.terrain_height(ex.arena,*point)-sdk.terrain_height(ex.arena,*p)
            return score-max(0.,abs(height)-.13)*5.
        options=[p+forward*.95,p-forward*.95]
        previous=getattr(ex,'last_drive',np.zeros(2))
        if np.linalg.norm(previous)>.1 and abs(np.dot(previous,forward))>.1:
            target=p-forward*np.sign(np.dot(previous,forward))*.95
        else:target=max(options,key=clearance)
        state.update(escape=np.clip(target,[.5,.5],[ex.arena['width']-.5,ex.arena['height']-.5]),escape_heading=held,escape_until=now+2.5)
    if state.get('escape_until',0)>now and np.linalg.norm(state['escape']-p)>.2:
        held=state['escape_heading'];forward=np.array([math.cos(held),math.sin(held)])
        sign=1 if np.dot(state['escape']-p,forward)>0 else -1
        return forward*sign,held
    route_goal=np.asarray(goal);route_arena=ex.navigation_arena
    deck=next((t for t in ex.arena['terrain'] if t.get('id')=='central_highland'),None)
    if ex.model_id==305 and deck is not None:
        inside=sdk.point_in_polygon(*p,deck['polygon']);destination=sdk.point_in_polygon(*route_goal,deck['polygon'])
        if inside or destination:
            if not hasattr(ex,'interior_arena'):ex.interior_arena=navigation_arena(ex.arena,3)
            route_arena=ex.interior_arena
        ramps=[t for t in ex.arena['terrain'] if t.get('id') in ('central_red_end_slope','central_blue_end_slope')]
        transfer=state.get('deck_transfer')
        if transfer is None and inside!=destination:
            from .navigation_grid import route_distance
            def transfer_cost(t):
                low=np.asarray(t['ramp_start']);high=np.asarray(t['ramp_end']);axis=(high-low)/np.linalg.norm(high-low)
                if inside:return route_distance(p,high+axis*.3,route_arena,robot['team'])+route_distance(low-axis*.65,route_goal,route_arena,robot['team'])
                return route_distance(p,low-axis*.5,route_arena,robot['team'])+route_distance(high+axis*.6,route_goal,route_arena,robot['team'])
            ramp=min(ramps,key=transfer_cost)
            transfer={'ramp':ramp['id'],'exit':inside,'near':False};state['deck_transfer']=transfer
        if transfer is not None:
            ramp=next(t for t in ramps if t['id']==transfer['ramp']);low=np.asarray(ramp['ramp_start']);high=np.asarray(ramp['ramp_end'])
            axis=(high-low)/np.linalg.norm(high-low);leaving=transfer['exit']
            near=high+axis*.3 if leaving else low-axis*.5
            far=low-axis*.65 if leaving else high+axis*.6
            if transfer['near'] and np.linalg.norm(p-far)<.4:
                state.pop('deck_transfer',None);state.pop('grid_active',None)
            else:
                if np.linalg.norm(p-near)<.45:transfer['near']=True
                route_goal=far if transfer['near'] else near
                if not hasattr(ex,'interior_arena'):ex.interior_arena=navigation_arena(ex.arena,3)
                route_arena=ex.interior_arena
    planned=sdk.navigate(robot,route_goal,route_arena)
    if ex.model_id==305 and (state.get('grid_active') or np.linalg.norm(planned)<.001) and np.linalg.norm(route_goal-p)>.4:
        from .navigation_grid import hint
        planned=hint(robot,route_goal,route_arena);state['grid_active']=True
    state['waypoint']=route_goal.tolist()
    corrected,heading=clearance_hint(robot,planned,ex.arena)
    return np.asarray(corrected),heading if heading is not None else getattr(planned,'heading',None)
