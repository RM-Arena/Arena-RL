"""Conservative public-map route hints for W3's local learned controller."""
import copy
import hashlib
import json


def navigation_arena(arena):
    """Avoid the assembly plinth unless a future skill explicitly handles it.

    Arena's SDK permits the bevel in its generic graph. W3 can get its wheel
    trapped against its edge while turning. Routing around its public footprint
    does not alter simulator geometry, collisions, terrain or robot performance.
    """
    result=copy.deepcopy(arena)
    for t in arena.get('terrain',[]):
        if t.get('id')=='assembly_plinth':
            result['obstacles'].append({'id':'tactical_assembly_avoidance','x':t['x'],'y':t['y'],
                'w':t['w'],'h':t['h'],'height':1.,'bottom':0.,'rotation':0.,'kind':'barrier'})
    # SDK caches graphs by revision; the derived public geometry needs its own key.
    digest=hashlib.sha256(json.dumps([result['obstacles'],result['terrain']],sort_keys=True).encode()).hexdigest()[:12]
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
