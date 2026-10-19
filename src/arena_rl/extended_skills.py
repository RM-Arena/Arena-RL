"""Stateful support objectives and real gap-crossing objectives on public geometry."""
import math
import numpy as np
from .executor import own,xy


def validate(ex,c,obs,arena):
    from rmuc_arena import sdk
    team=obs['team']
    if c.skill!='fly_slope' and arena.get('season')!='2027':raise ValueError('Support skills currently require season 2027')
    if c.skill=='escort':
        if not c.target_id.startswith(team+'-') or c.target_id==f'{team}-{ex.slot}':raise ValueError('Escort requires another allied robot')
        if not any(r['id']==c.target_id and r['role']!='aerial' for r in obs['robots']):raise ValueError('Escort target must be a deployed mobile ally')
    if c.skill=='resupply':
        zones=[z for z in arena['zones'] if z['kind']=='supply' and z.get('owner')==team]
        if not zones:raise ValueError('No own supply zone on this map')
        if c.point and not any(sdk.point_in_polygon(*c.point,z['polygon']) for z in zones):raise ValueError('Resupply point must be inside the own supply zone')
    if c.skill=='fly_slope':
        if not any(t.get('id')==c.lane_team+'_fly_slope' for t in arena['terrain']):raise ValueError('This map has no selected fly-slope lane (use season 2026)')
        if c.flight_direction=='reverse' and c.lane_team not in ex.flight_authorized:raise ValueError('Reverse flight requires a completed forward crossing in this episode')


def reset(ex):
    ex.skill_state={};ex.extra={}


def _support_target(ex,c,obs,visible,center):
    if not c.engage_on_contact:return None
    candidates=[e for e in visible.values() if e['alive'] and e.get('targetable',True) and np.linalg.norm(xy(e)-center)<=c.region_radius_m]
    def rank(e):return (c.priority_ids.index(e['id']) if e['id'] in c.priority_ids else len(c.priority_ids),np.linalg.norm(xy(e)-center),e['id'])
    return min(candidates,key=rank,default=None)


def update(ex,obs,visible):
    from rmuc_arena import sdk
    c=ex.command;r=own(obs,ex.slot);p=xy(r);st=ex.skill_state;now=ex.time;elapsed=now-ex.started
    previous=st.get('last_time',now);dt=max(0.,now-previous);st['last_time']=now
    target=None;phase='navigate';goal=p.copy();metrics={}
    if c.skill=='resupply':
        zone=next(z for z in ex.arena['zones'] if z['kind']=='supply' and z.get('owner')==obs['team'])
        goal=np.array(c.point or [zone['x'],zone['y']]);team=obs['teams'][obs['team']]
        resources=team['resources'];purchased=team.get('purchased_17',0)
        st.setdefault('initial_purchased',purchased)
        price=sdk.rules(obs)['economy']['ammo']['17mm'];unit_cost=price['local_cost']/price['local_batch']
        st['spent']=max(0,purchased-st['initial_purchased'])*unit_cost
        contact=f"{obs['team']}-{ex.slot}" in obs['zones'][zone['id']]['occupants']
        if contact:phase='supply_hold';goal=p.copy()
        ready=r['ammo']>=c.ammo_target and r['hp']/max(1,r['max_hp'])>=c.hp_target_fraction
        if ready:ex._end('succeeded','resupplied')
        # Income is provided by the ordinary referee; no invented training coins.
        if contact and r['ammo']<c.ammo_target and (resources<10 or st['spent']+10>c.budget_limit) and now>1.1:
            ex._end('failed','insufficient_supply_budget')
        if contact and r['ammo']<c.ammo_target and purchased>=price['team_purchase_cap']:ex._end('failed','team_ammo_cap_reached')
        metrics={'supply_contact':contact,'ammo':r['ammo'],'ammo_target':c.ammo_target,'hp_fraction':r['hp']/max(1,r['max_hp']),
                 'hp_target_fraction':c.hp_target_fraction,'spent':st['spent'],'resources':resources}
    elif c.skill=='escort':
        ally=next(a for a in obs['robots'] if a['id']==c.target_id)
        if not ally['alive']:ex._end('failed','escort_target_destroyed')
        ap=xy(ally);velocity=np.array([ally['vx'],ally['vy']]);speed=np.linalg.norm(velocity)
        direction=velocity/speed if speed>.1 else np.array([math.cos(ally['heading']),math.sin(ally['heading'])])
        goal=ap-direction*c.follow_distance_m
        separation=np.linalg.norm(p-ap);close=.45<=separation<=c.follow_distance_m+1.1
        st['covered']=st.get('covered',0.)+(dt if close else 0.)
        coverage=st['covered']/max(.1,elapsed)
        target=_support_target(ex,c,obs,visible,ap)
        phase='escort_follow' if np.linalg.norm(goal-p)>.4 else 'escort_hold'
        destination=c.point is not None and np.linalg.norm(ap-np.array(c.point))<=c.radius_m
        complete=destination if c.point else elapsed>=c.escort_duration_s
        if complete and close and coverage>=c.required_coverage:ex._end('succeeded','escort_completed')
        if separation>c.leash_radius_m:ex._end('failed','escort_separation_exceeded')
        metrics={'protected_ally':ally['id'],'ally_hp':ally['hp'],'separation_m':float(separation),'coverage':min(1.,coverage)}
    elif c.skill=='patrol':
        index=st.setdefault('index',0);goal=np.array(c.waypoints[index%len(c.waypoints)])
        arrived=np.linalg.norm(p-goal)<=c.radius_m and math.hypot(r['vx'],r['vy'])<1.2
        st['dwell']=st.get('dwell',0.)+dt if arrived else 0.
        phase='patrol_dwell' if arrived else 'patrol_move'
        if arrived and st['dwell']+1e-8>=c.dwell_s:
            st['index']+=1;st['dwell']=0.
            if st['index']>=len(c.waypoints)*c.patrol_loops:ex._end('succeeded','patrol_completed')
            else:goal=np.array(c.waypoints[st['index']%len(c.waypoints)])
        target=_support_target(ex,c,obs,visible,p)
        metrics={'waypoint_index':st['index'],'waypoints_total':len(c.waypoints)*c.patrol_loops,'dwell_s':st['dwell']}
    else:
        lane=sdk.fly_slope_lane(ex.arena,c.lane_team);lip=np.array(lane['lip']);crest=np.array(lane['crest']);landing=np.array(lane['landing'])
        axis=(crest-lip)/np.linalg.norm(crest-lip);direction=1. if c.flight_direction=='forward' else -1.
        entry=lip-axis*2.55 if direction>0 else landing+axis*.45
        goal=landing if direction>0 else lip-axis*.9
        axis*=direction
        desired_heading=math.atan2(axis[1],axis[0])
        heading_error=abs((r['heading']-desired_heading+math.pi)%math.tau-math.pi)
        entry_tolerance=.11 if direction>0 else .055
        if 'launched' not in st and np.linalg.norm(p-entry)<entry_tolerance and math.hypot(r['vx'],r['vy'])<.8 and heading_error<.12:
            st['launched']=True;st['heading']=r['heading']
        if not st.get('launched'):
            phase='flight_align' if np.linalg.norm(p-entry)<entry_tolerance else 'flight_approach';goal=entry
        else:
            phase='flight_run'
            gap_start=np.array(lane['crest']);lane_axis=(crest-lip)/np.linalg.norm(crest-lip)
            gap_end=landing-lane_axis*.75
            in_gap=np.dot(p-gap_start,lane_axis)>0 and np.dot(p-gap_end,lane_axis)<0
            if not r['grounded'] and in_gap:
                st['air_s']=st.get('air_s',0.)+dt;st['max_z']=max(st.get('max_z',r['z']),r['z']);phase='flight_air'
            beyond=(float(np.dot(p-crest,axis))> .65) if direction>0 else float(np.dot(p-lip,axis))> .2
            landed=beyond and r['grounded'] and r['upright']>.9 and np.linalg.norm(p-goal)<.65 and math.hypot(r['vx'],r['vy'])<1.2
            st['landing_hold']=st.get('landing_hold',0.)+dt if landed else 0.
            if landed:phase='flight_land'
            if st.get('air_s',0.)+1e-8>=.1 and st['landing_hold']+1e-8>=c.landing_hold_s:
                ex._end('succeeded','flight_landed')
                if direction>0:ex.flight_authorized.add(c.lane_team)
        metrics={'flight_direction':c.flight_direction,'airborne_s':st.get('air_s',0.),'landing_hold_s':st.get('landing_hold',0.),
                 'lane_team':c.lane_team,'launched':st.get('launched',False)}
    if target:
        identity=target['id'];hp=target['hp'];ex.damage+=max(0.,ex.last_hp.get(identity,hp)-hp);ex.last_hp[identity]=hp
        metrics['engagement_target_id']=identity
    goal=np.clip(goal,[r['radius'],r['radius']],[ex.arena['width']-r['radius'],ex.arena['height']-r['radius']])
    distance=float(np.linalg.norm(goal-p))
    if elapsed>=c.timeout_s:ex._end('failed','timeout')
    if ex.last_position is None or np.linalg.norm(p-ex.last_position)>.25:ex.last_position=p;ex.last_motion=now
    moving=phase in ('navigate','escort_follow','patrol_move','flight_approach','flight_run')
    if moving and distance>c.radius_m and now-ex.last_motion>(25 if ex.model_id==305 else 8):ex._end('failed','blocked')
    planned=(0.,0.)
    if c.skill=='fly_slope' and st.get('launched'):
        correction=goal-p;along=float(np.dot(correction,axis));lateral=correction-axis*along
        planned=axis+np.clip(lateral*1.5,-0.2,.2)
        if along<.9:planned=correction*1.5
        if phase=='flight_land':planned=np.zeros(2)
    elif c.skill=='fly_slope':
        planned=(goal-p)*2.
    elif distance>c.radius_m:
        if ex.model_id==305:
            from .navigation import plan_motion
            planned,held=plan_motion(ex,r,goal,now)
        else:planned=sdk.navigate(r,goal,ex.navigation_arena)
    ex.plan={'phase':phase,'goal':goal,'hint':np.array(planned),'target':target,'visible':bool(target),
             'distance':distance,'age':0.,'held_heading':held if 'held' in locals() else getattr(planned,'heading',None)}
    ex.extra=metrics
    if ex.initial_distance is None:ex.initial_distance=distance
    if ex.status in ('succeeded','failed','cancelled'):ex.events.append(ex.feedback());ex.events=ex.events[-128:]
    return ex.plan
