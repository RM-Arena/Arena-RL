# LLM tactical commands and learned execution

The LLM chooses a single robot's objective. Five PPO policies execute nine parameterized commands at 10 Hz. The service is provider independent; it does not call an LLM API. Only W3 model 3 in red slot 3 is trained. The earlier two-choice PPO manager remains a baseline.

## Commands and supported maps

| Skill | Parameters | Completion / execution |
| --- | --- | --- |
| `navigate` | `point: [x,y]` | Arrive within `radius_m` at low speed; public routing plus learned local control |
| `attack_structure` | `target_id: blue_outpost` / `blue_base` | Attack the selected structure, respecting base protection |
| `engage_target` | Enemy robot ID | Approach and engage that identity at `preferred_range_m` |
| `pursue_target` | Enemy robot ID | Follow a short predicted intercept, preserving target identity and leash |
| `defend_region` | `center`, optional ordered `priority_ids` | Hold/intercept inside the region for the interval, or meet `damage_goal` |
| `resupply` | `ammo_target`, `hp_target_fraction`, `budget_limit`, optional `point` | Reach own supply zone, buy local ammunition and wait for real passive healing |
| `escort` | Allied `target_id`, optional destination `point` | Follow the moving ally; satisfy proximity coverage and destination/duration |
| `patrol` | Ordered `waypoints`, `patrol_loops`, `dwell_s` | Visit every waypoint in order, dwell at low speed, complete every loop |
| `fly_slope` | `lane_team`, `flight_direction`, `landing_hold_s` | Cross the actual 650 mm gap, record airborne time, land upright and settle |

The first eight commands are trained/evaluated on 2027. Dedicated fly-slope geometry exists on **2026**, so use `--season 2026` for this skill. The 2027 road ramps are ground navigation terrain, not a dedicated airborne crossing. Season selection is an episode property: a command cannot change maps. `navigate` is also accepted on 2026 for staging; its full-map held-out benchmark is on 2027. Aerial control, arbitrary terrain jumps, other controlled robot models and full-match wins are outside the recorded coverage.

`ready` and `capabilities` include `available_skills`, filtered by loaded weights and the current season. Protocol-wide `skills`/`command_json_schema` describe all nine commands. An unavailable command is rejected before replacing the current objective.

## JSON examples

```jsonl
{"id":"supply-1","skill":"resupply","ammo_target":100,"hp_target_fraction":0.9,"budget_limit":150,"timeout_s":60}
{"id":"protect-1","skill":"escort","target_id":"red-4","point":[7.5,5.35],"follow_distance_m":1.2,"required_coverage":0.6,"timeout_s":30}
{"id":"route-1","skill":"patrol","waypoints":[[6.8,5.15],[6.8,5.7],[4.8,5.5]],"patrol_loops":2,"dwell_s":0.4,"timeout_s":60}
{"id":"fly-1","skill":"fly_slope","lane_team":"red","flight_direction":"forward","timeout_s":18}
{"id":"return-1","skill":"fly_slope","lane_team":"red","flight_direction":"reverse","timeout_s":18}
```

These are command objects; wrap each in `{"op":"command","command": ...}` for the transport. Reverse flight is accepted only after that executor completed the same lane's forward crossing in the **same physical episode**. Reset clears this permission. The reverse curriculum first drives a genuine forward crossing; it never injects a referee pass. Flight is evaluated from lane staging positions, on both mirror lanes and in both directions, with no hostile fire. Arbitrary distant approaches and disturbed landings need additional courses.

Resupply sends ordinary `buy_ammo` actions only while on the actual own supply card. The simulator applies batch size, prices, team purchase cap, resource deductions and healing. No HP, ammunition or currency is patched during execution. `budget_limit` conservatively counts the own team's local 17 mm purchase ledger increase while this goal runs; simultaneous teammate purchases also consume this allowance. Insufficient funds/budget, team cap, timeout or robot loss produce explicit failure. A wounded robot can seek supply despite the usual low-HP stop. This skill currently covers local ammunition and passive supply healing, not remote purchases/paid remote healing. Normal 2027 starting ammunition remains zero: issue `resupply` before combat.

Escort uses legal allied telemetry and never takes control of the protected ally. With `point`, the ally must reach that destination; without it, use `escort_duration_s`. `required_coverage` is the fraction of elapsed goal time spent between 0.45 m and `follow_distance_m + 1.1 m` from the ally. Excessive separation (`leash_radius_m`) or ally destruction fails the command. The supplied escort course uses a slow scripted moving ally. It does not prove protection under hostile fire.

Patrol requires at least two points. Dwell resets when the robot leaves a waypoint or moves too fast. Zero ammunition does not prevent patrol/escort. Both accept `engage_on_contact: true` and ordered enemy `priority_ids` for optional firing while following their route/formation; only slot 3's own visible enemies are eligible. The current held-out support courses have no such combat interference, so this option needs separate combat evaluation.

## Command supervision

IDs make identical retries idempotent within a 1,024-command cache. Reusing an ID with changed content is rejected. A valid replacement preserves simulator process, tick, pose, health and ammunition. Invalid/stale/future/out-of-order commands leave the running goal intact. Optional `basis_tick` plus `max_staleness_s` limits delayed LLM replies. `cancel` stops execution; it is not a learned skill.

Robot attacks retain explicit target identity. Only slot 3's legal FPV enemies enter target memory; occluded targets are extrapolated from their last observed velocity for a bounded time, with no firing. Friendly attacks, absent roster IDs and protected structures are rejected/stopped. Attack completion uses target destruction or an optional positive `damage_goal`. HP loss is observed progress, not guaranteed shooter attribution when teammates also attack.

Feedback includes status (`idle/running/succeeded/failed/cancelled`), reason, phase, elapsed simulated time, distance, target identity/visibility and `execution_family`. `details` adds supply contact/ammo/HP/spending, escort separation/coverage, patrol waypoint progress, or flight airborne/landing time. Failure reasons include blocked, timeout, lost target, leash exceeded, unavailable robot, low HP, no ammo, budget exhaustion and team purchase cap. Ordinary routes report blocked after six seconds without significant motion/damage; support/flight movement uses eight seconds. Terminal feedback retains the completion duration.

The LLM scene exposes own robot and allied telemetry, own resources/purchase counters, own card contacts, visible enemies, age-marked last-seen enemies, public structures and control positions. Public map/roster arrive in `ready`. No opponent-own live observation is sent to the LLM or learned controller. Curriculum host controllers may use their own team's observations to move/shoot their scripted actors.

## Service and host integration

```bash
uv run arena-rl serve-tactics --policies checkpoints/tactics --realtime --record
uv run arena-rl serve-tactics --policies checkpoints/tactics --season 2026 --realtime --record
```

Default real-time control advances 100 ms of simulation per 100 ms wall time; `--speed` changes this ratio. External LLM inference can proceed while RL continues the current objective. Periodic state messages arrive approximately once a wall second. Without `--realtime`, manual `{"op":"step","steps":10}` advances the simulation.

Other operations are `observe`, `cancel`, `capabilities`, `reset`, `save_replay` and `close`. `env_id` selects one parallel actor. `reset` accepts per-actor seeds and optional declared initial training scenarios; only reset changes the episode. EOF/close/termination clean up simulator processes. Malformed or oversized (>64 KiB) messages return framed errors. `examples/tactical_client.py` illustrates a slow scripted planner, without paid inference.

`--lineups path.json` accepts the Arena roster format, retaining red slot 3 as W3 model 3. The default 2027 roster includes red slot 4 for escort. Teammates stay idle in the service unless the match host supplies controllers; the library never silently drives an escort target. `--opponent` selects an existing whole-team Arena submission through Arena's sandbox. Without it, live enemies remain stationary and `ready` declares this mode.

Within an existing match host:

```python
from arena_rl.executor import TacticalExecutor
from arena_rl.tactical_envs import encode_observation, decode_action

executor = TacticalExecutor(slot="3")
executor.submit(command_from_llm, legal_red_observation)
# Every 100 ms simulated time:
executor.update(legal_red_observation)
family = executor.execution_family
features = encode_observation(executor, legal_red_observation, family)
action, _ = policies[family].predict(features, deterministic=True)
robot_actions = decode_action(executor, legal_red_observation, action, family)
feedback_to_llm = executor.feedback()
```

Merge only this slot's returned action with teammate controllers. Long ground approaches can use the navigation policy before switching to the local support/combat/defense policy, retaining the command ID; firing is suppressed in navigation mode. The host owns observation delivery, inference scheduling and episode lifetime.

## Training contracts and reproducibility

```bash
uv run arena-rl train-tactics --family support --steps 32768 --envs 4 --output runs/support
uv run arena-rl train-tactics --family flight --steps 16384 --envs 4 --output runs/flight
uv run arena-rl evaluate-tactics --checkpoint runs/flight --episodes 40 --seed 6000000 --replays runs/flight/replays
uv run python examples/validate_extensions.py --policies checkpoints/tactics --output runs/runtime
```

Old navigation/combat/defense retain `w3-tactics-v3` (52 observations). New support and flight use separate `w3-support-v1` / `w3-flight-v1` contracts (74 observations). All use `MultiDiscrete([5,4,3,3,2,3,2])`. The new weights cannot be silently substituted for the old policy families. Checkpoint digests and simulator source/embedded rule-engine provenance are checked on loading.

Support uses ordinary PPO from random weights. Flight defaults to 64 attempted scripted demonstration episodes, retains only successful trajectories, fits the policy by behavior cloning, then runs PPO on fresh episodes. Its metadata records accepted/rejected seeds, dataset SHA256, imitation fit, PPO steps/updates and changed weight hashes. `--expert-episodes 0` disables initialization; a resumed policy is not initialized again. Demonstrations are training data, not a runtime fallback. The deployed flight actions come from the trained network plus the documented geometric heading/gap/landing contract. From-scratch flight PPO did not succeed in the initial 32,768-step trial.

Public SDK route hints and ballistic aiming are explicit assistance. W3 navigation now maintains nominal leg clearance and clears a bump along its existing axle before turning; physics/map geometry are unchanged. The network chooses local motion/speed/gait/fire/jump decisions, not motor current, joint torque or raw-image perception.

Native v4 replays preserve ordinary actions including injury/forward-crossing preparation. `.commands.json` sidecars record runtime command IDs and policy digests. The validation script checks command tick/ammo/process continuity and deterministically replays representative support/flight episodes through Arena's verifier. See [expansion results](extended-results.md) for exact seeds, baselines and limitations, and [earlier tactical results](tactical-results.md) for the original five-skill benchmarks.
