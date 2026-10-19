"""Public-map grid fallback for routes disconnected by conservative SDK margins."""
import heapq
import math
import numpy as np

CACHE={}
STEP=.25


class Grid:
    def __init__(self,arena,team):
        from rmuc_arena import sdk
        self.arena=arena;self.shape=(round(arena['width']/STEP)+1,round(arena['height']/STEP)+1)
        xs,ys=np.meshgrid(np.arange(self.shape[0])*STEP,np.arange(self.shape[1])*STEP,indexing='ij')
        margin=.31
        self.free=(xs>margin)&(xs<arena['width']-margin)&(ys>margin)&(ys<arena['height']-margin)
        walls=[w for w in arena['obstacles'] if 'roof' not in w.get('id','') or '_tunnel_roof' in w['id']]
        walls += [w for w in arena.get('forbidden',[]) if w['kind'] in ('base','pad','highway','dart_station') or w.get('owner') not in (None,team)]
        for wall in walls:
            angle=wall.get('rotation',0);c,s=math.cos(angle),math.sin(angle)
            dx=xs-wall['x']-wall['w']/2;dy=ys-wall['y']-wall['h']/2
            self.free &= (abs(dx*c+dy*s)>=wall['w']/2+margin)|(abs(-dx*s+dy*c)>=wall['h']/2+margin)
        self.heights=np.array([[sdk.terrain_height(arena,x*STEP,y*STEP) for y in range(self.shape[1])] for x in range(self.shape[0])])
        self.nodes=np.argwhere(self.free);self.costs={}

    def nearest(self,p):
        if not len(self.nodes):return None
        return tuple(self.nodes[np.argmin(np.sum((self.nodes*STEP-np.asarray(p))**2,axis=1))])

    def neighbors(self,node):
        x,y=node
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            a,b=x+dx,y+dy
            if not (0<=a<self.shape[0] and 0<=b<self.shape[1]) or not self.free[a,b]:continue
            if dx and dy and (not self.free[x+dx,y] or not self.free[x,y+dy]):continue
            if abs(self.heights[a,b]-self.heights[x,y])>.13:continue
            yield (a,b),math.hypot(dx,dy)*STEP

    def distances(self,goal):
        if goal in self.costs:return self.costs[goal]
        distance=np.full(self.shape,np.inf);distance[goal]=0.;heap=[(0.,goal)]
        while heap:
            cost,node=heapq.heappop(heap)
            if cost>distance[node]+1e-8:continue
            for neighbor,weight in self.neighbors(node):
                total=cost+weight
                if total+1e-8<distance[neighbor]:distance[neighbor]=total;heapq.heappush(heap,(total,neighbor))
        if len(self.costs)>32:self.costs.pop(next(iter(self.costs)))
        self.costs[goal]=distance;return distance

    def toward(self,position,goal):
        start,end=self.nearest(position),self.nearest(goal)
        if start is None or end is None:return np.zeros(2)
        distance=self.distances(end)
        if not np.isfinite(distance[start]):return np.zeros(2)
        node=start;path=[]
        for _ in range(4):
            if node==end:break
            options=[(distance[n]+weight,n) for n,weight in self.neighbors(node) if distance[n]<distance[node]-1e-6]
            if not options:break
            node=min(options)[1];path.append(node)
        # Follow a nearby grid waypoint. The engine still supplies all collision,
        # traction, slope and power behavior; this cannot move the robot directly.
        target=np.asarray(path[0] if path else end)*STEP
        if np.linalg.norm(np.asarray(goal)-position)<.8:target=np.asarray(goal)
        delta=target-position
        return delta/max(.001,np.linalg.norm(delta))


def hint(robot,goal,arena):
    key=(arena['layout_revision'],robot['team'])
    if key not in CACHE:CACHE[key]=Grid(arena,robot['team'])
    return CACHE[key].toward(np.array([robot['x'],robot['y']]),goal)


def route_distance(point,goal,arena,team):
    key=(arena['layout_revision'],team)
    if key not in CACHE:CACHE[key]=Grid(arena,team)
    graph=CACHE[key];start,end=graph.nearest(point),graph.nearest(goal)
    return float(graph.distances(end)[start]) if start is not None and end is not None else math.inf
