"""Seeded curricula and task rewards; all motion and damage come from Arena."""
from dataclasses import dataclass
import math
import numpy as np

TASKS = ("move", "ramp", "outpost", "defend")
MODEL = 3
SLOT = "3"


@dataclass
class Task:
    name: str
    goal: np.ndarray
    scenario: dict
    max_steps: int
    seed: int
    radius: float = .35
    damage_goal: float = 40.0
    steps: int = 0
    initial_hp: float = 1000.0
    last_hp: float = 1000.0
    previous_distance: float = 0.0
    hold: int = 0
    crossed: bool = False
    initial_z: float = 0.0
    initial_attack: float = 0.0
    last_attack: float = 0.0


def sample(name, seed, difficulty=1.0):
    rng = np.random.default_rng(seed)
    if name == "mixed":
        name = TASKS[int(rng.integers(0, 4))]
    if name not in TASKS:
        raise ValueError(f"Unknown task: {name}")
    # The spare blue robot is scripted, never controlled by the learner.
    blue = {"x": 26.0, "y": 9.5, "heading": math.pi, "ammo": 0}
    if name == "move":
        start = np.array([4.7, 5.15]) + rng.uniform(-.35, .35, 2)
        angle = rng.uniform(-math.pi, math.pi)
        length = rng.uniform(.7, 1.2 + difficulty)
        goal = start + length * np.array([math.cos(angle), math.sin(angle)])
        goal = np.clip(goal, [3.4, 4.65], [7.5, 5.8])
        if np.linalg.norm(goal-start)<.6:
            goal=start+np.array([.9,0.])
        limit = 90
    elif name == "ramp":
        # 2027 red road slope: an actual 15 degree ramp on the shared map.
        bottom = np.array([6.6643784765, 11.6472375415])
        top = np.array([5.996, 11.9795])
        axis = (top-bottom)/np.linalg.norm(top-bottom)
        tangent = np.array([-axis[1],axis[0]])
        start = bottom-axis*.85+tangent*rng.uniform(-.12,.12)
        goal = top+axis*.65+tangent*rng.uniform(-.08,.08)
        limit = 110
    elif name == "outpost":
        start = np.array([19.0, 4.65]) + rng.uniform(-.25, .25, 2)
        goal = np.array([17.024, 3.6])
        limit = 120
    else:
        start = np.array([4.7, 5.15]) + rng.uniform(-.2, .2, 2)
        goal = np.array([4.1, 5.1])
        blue = {"x": 7.7, "y": 5.15, "heading": math.pi, "ammo": 100}
        limit = 150
    heading = math.atan2(goal[1]-start[1], goal[0]-start[0])
    scenario = {"robots": {"red-3": {"x": float(start[0]), "y": float(start[1]), "heading": heading,
                                    "aim": 0. if name=="defend" else heading, "ammo": 200}, "blue-3": blue}}
    return Task(name, goal, scenario, limit, int(seed))


def robot(transition, team="red"):
    return next(r for r in transition["observations"][team]["robots"] if str(r["slot"]) == SLOT)


def begin(task, transition):
    r = robot(transition)
    task.previous_distance = float(np.linalg.norm(task.goal - [r["x"], r["y"]]))
    task.initial_z = r["z"]
    task.initial_attack = task.last_attack = transition["observations"]["red"]["teams"]["red"]["attack_damage"]
    task.initial_hp = task.last_hp = transition["observations"]["red"]["outposts"]["blue"]["hp"]


def reward(task, transition):
    r = robot(transition)
    obs = transition["observations"]["red"]
    distance = float(np.linalg.norm(task.goal - [r["x"], r["y"]]))
    progress = task.previous_distance - distance
    task.previous_distance = distance
    task.steps += 1
    hp = obs["outposts"]["blue"]["hp"]
    damage = max(0., task.last_hp - hp)
    task.last_hp = hp
    attack = obs["teams"]["red"]["attack_damage"]
    inflicted = max(0., attack-task.last_attack)
    task.last_attack = attack
    terms = {"progress": 4.*progress if task.name != "outpost" else 0., "time": -.015,
             "damage": damage*.12 if task.name == "outpost" else 0.,
             "instability": -.03*max(0., 1.-r["upright"])}
    if task.name == "ramp":
        axis = np.array([-.895, .445])
        task.crossed |= float(np.dot(np.array([r["x"],r["y"]])-[5.996,11.9795],axis)) > .1 and r["z"] > task.initial_z + .12
        success = task.crossed and distance < task.radius and r["upright"] > .8
    elif task.name == "outpost":
        success = task.initial_hp - hp >= task.damage_goal
        terms["heat"] = -.025 * max(0., r.get("heat", 0.)-r.get("max_heat", 40.)*.9)
    elif task.name == "defend":
        task.hold = task.hold + 1 if distance < .9 else 0
        success = attack-task.initial_attack >= task.damage_goal
        terms["hold"] = .03 if distance < .9 else 0.
        terms["intercept_damage"] = .12*inflicted
    else:
        # Require a controlled arrival, rather than just racing through the circle.
        success = distance < task.radius and math.hypot(r["vx"], r["vy"]) < .8
    failure = not r["alive"] or r["upright"] < .25 or distance > 12.
    terminated = bool(success or failure or transition["terminated"])
    truncated = bool(not terminated and (task.steps >= task.max_steps or transition["truncated"]))
    terms["terminal"] = 8. if success else -4. if failure else 0.
    info = {"task": task.name, "is_success": bool(success), "distance": distance,
            "outpost_damage": max(0., task.initial_hp-hp), "attack_damage": max(0.,attack-task.initial_attack), "ramp_crossed": task.crossed,
            "reward_terms": terms, "seed": task.seed, "steps": task.steps}
    return float(sum(terms.values())), terminated, truncated, info
