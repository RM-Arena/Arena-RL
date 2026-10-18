"""Versioned policy interface using only the controlled robot's legal observations."""
import math
import numpy as np
from gymnasium import spaces
from .tasks import TASKS, SLOT, robot

SCHEMA = "w3-skills-v1"
DIRECTIONS = np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]], dtype=float)
DIRECTIONS[5:] /= math.sqrt(2)
ACTION_SPACE = spaces.MultiDiscrete([9,3,3,2,3,2])
OBS_SPACE = spaces.Box(-10.,10.,shape=(33,),dtype=np.float32)


def encode(task, transition):
    obs=transition["observations"]["red"]; r=robot(transition)
    delta=task.goal-np.array([r["x"],r["y"]]); distance=np.linalg.norm(delta)
    heading=r["heading"]; angle=math.atan2(delta[1],delta[0])-heading
    enemies=[e for e in obs["enemies"] if e["alive"]]
    nearest=min(enemies,key=lambda e: math.hypot(e["x"]-r["x"],e["y"]-r["y"]),default=None)
    values=[float(task.name==t) for t in TASKS]
    values += [delta[0]/4,delta[1]/4,distance/6,math.sin(angle),math.cos(angle),
               r["vx"]/4,r["vy"]/4,r["vz"]/4,r["z"],r["upright"],
               math.sin(heading),math.cos(heading),r["hp"]/max(1,r["max_hp"]),
               r["ammo"]/200,r.get("heat",0)/max(1,r.get("max_heat",40)),
               float(r["grounded"]),float(r["powered"]),r.get("legs",[.245,.245])[0]/.34,
               math.sin(r["aim"]-heading),math.cos(r["aim"]-heading),r["pitch"],
               max(0,task.initial_hp-task.last_hp)/max(1,task.damage_goal),
               task.steps/task.max_steps,float(task.crossed)]
    values += [float(nearest is not None),
               (nearest["x"]-r["x"])/6 if nearest else 0,
               (nearest["y"]-r["y"])/6 if nearest else 0,
               (r["x"]-4.1)/14,(r["y"]-5.1)/15]
    return np.clip(np.array(values,dtype=np.float32),-10,10)


def decode(task, transition, action):
    from rmuc_arena import sdk
    action=np.asarray(action)
    if action.shape != (6,) or not np.issubdtype(action.dtype,np.integer) or not ACTION_SPACE.contains(action):
        raise ValueError("Action must be six valid MultiDiscrete integers")
    r=robot(transition); obs=transition["observations"]["red"]
    move=DIRECTIONS[int(action[0])]*.65
    heading=math.atan2(move[1],move[0]) if np.linalg.norm(move) else r["heading"]
    result={"move":move.tolist(),"heading":heading,"aim":r["aim"],"pitch":r["pitch"],"fire":False,
            "brake":True,"leg_height":[.18,.245,.31][int(action[4])],
            "jump":float(action[5]) if task.name=="ramp" else 0.}
    if task.name in ("outpost","defend"):
        if task.name=="outpost":
            target={**obs["outposts"]["blue"],"kind":"outpost"}
        else:
            enemies=[e for e in obs["enemies"] if e["alive"]]
            target=min(enemies,key=lambda e:math.hypot(e["x"]-r["x"],e["y"]-r["y"]),default=None)
        if target is None and task.name=="defend":
            result["aim"]=r["aim"]+.5  # Scan until an enemy enters the physical camera view.
        if target:
            # Explicit aim assistance: SDK ballistics plus learned yaw/pitch residuals.
            # Fire timing, movement, leg length and jump remain policy decisions.
            result["aim"]=sdk.aim_at(r,target)+(int(action[1])-1)*.025
            result["pitch"]=sdk.aim_pitch(r,target)+(int(action[2])-1)*.02
            result["fire"]=bool(action[3])
    return {"red":{SLOT:result},"blue":{}}
