# LLM tactical commands and learned execution

The LLM chooses the current single-robot objective. Arena-RL executes it with three goal-conditioned PPO policies: navigation, combat, and defense. The existing two-choice PPO manager remains a baseline; it is not required for this interface. This version does not call an LLM API or choose a model/provider.

## Available commands

| Skill | Required target | Execution |
| --- | --- | --- |
| `navigate` | `point: [x,y]` | Public-map routing plus learned local motion and gait |
| `attack_structure` | `target_id: blue_outpost` or `blue_base` | Approach a shooting position and attack that structure |
| `engage_target` | Robot ID such as `blue-1` | Engage that robot at the requested range |
| `pursue_target` | Robot ID such as `blue-1` | Approach a short predicted intercept position, retain explicit target identity |
| `defend_region` | `center: [x,y]` | Return/hold within the region, select visible threats using ordered priorities |

Only W3 model 3 in red slot 3 is trained. The 2027 heavy uses slot 1, so a command intended to pursue that robot must use its actual ID `blue-1`, rather than an ambiguous role name. Other enemy infantry appear in training as both targets and closer distractors. The learned agent never commands their actions.

Example command:

```json
{
  "id": "pursue-hero-17",
  "skill": "pursue_target",
  "target_id": "blue-1",
  "preferred_range_m": 2.5,
  "leash_radius_m": 8,
  "timeout_s": 15,
  "target_lost_timeout_s": 2,
  "low_hp_fraction": 0.2,
  "basis_tick": 200,
  "max_staleness_s": 3
}
```

`id` makes identical retries idempotent (a bounded 1,024-command cache). Reusing an ID with different content is rejected. Valid new commands cancel the previous goal without resetting the simulator, robot pose, health or ammo. Invalid commands leave the current goal intact. Optional `basis_tick` rejects a decision based on stale/future information, which is useful for overlapping LLM inference calls.

`defend_region` additionally accepts `region_radius_m` and ordered `priority_ids`. Its intercept destination stays inside the defense region. Robot pursuit has a leash anchored to the command's starting position. Combat may use an optional positive `damage_goal`; otherwise the explicit target's destruction completes it. Without a damage goal, defense runs for its requested interval or until cancelled/failed. `cancel` is a protocol operation, not a learned skill.

The executor validates friendly targets, public rosters, field bounds, robot availability, ammo, low-health limits and protected bases. Target visibility comes from the controlled robot's legal observation. Short searches extrapolate only the last observed velocity; they never consult the other team's live state. Shooting stops during occlusion and the command fails as `target_lost` after the configured grace time. Firing and damage credit retain the specified target even when another enemy is closer.

## Feedback and supervision

Each state includes the current command ID/revision, `idle/running/succeeded/failed/cancelled`, phase, reason, elapsed simulated time, target identity/visibility, last-seen age, distance and observed target HP loss. Reasons include arrival, damage target reached, target destruction, target lost, blocked, timeout, leash exceeded, low HP, no ammo and a protected structure. HP loss is observed target progress, not an attribution claim when other friendly robots are present.

The LLM receives only the controlled team's legal observation: its own state, visible enemies, explicitly age-marked last-seen information, public structures and control positions. The ready response includes public map geometry and the roster. No live hidden enemy state is included. Opponent-own observations are used only by scripted curriculum controllers on the training host.

A blocked route is reported after six seconds without significant robot displacement or target damage. A timeout stops the command. A new tactical command can then select another point, target or defense region without resetting the episode. Automatic resupply and retreat policies are not implemented yet; low HP/no ammo is returned to the LLM for a new decision.

## JSONL service

```bash
uv run arena-rl serve-tactics --policies checkpoints/tactics --realtime --record
```

The default clock executes one low-level action every 100 ms of wall time, advancing 100 ms in the simulator. `--speed` changes this relation for testing. LLM inference occurs in a separate client; while it waits, the current goal keeps executing. State updates are emitted approximately once per wall second. The JSONL interface also works in manual stepping mode by omitting `--realtime`.

Requests:

```jsonl
{"id":"order-1","op":"command","command":{"id":"attack-1","skill":"attack_structure","target_id":"blue_outpost","timeout_s":40}}
{"id":"status","op":"observe"}
{"id":"change","op":"command","command":{"id":"guard-1","skill":"defend_region","center":[4.5,5.15],"region_radius_m":3,"priority_ids":["blue-1"],"timeout_s":30}}
{"id":"stop","op":"cancel"}
{"id":"replay","op":"save_replay","path":"runs/live.json"}
{"op":"close"}
```

Manual mode additionally accepts `{"op":"step","steps":10}`. `reset` accepts per-actor `seeds` and optional training `scenarios`; it is the only operation that resets an episode. `env_id` selects one actor for commands/observations/replays. EOF, close and termination clean up actor/simulator processes. Malformed or oversized (>64 KiB) requests return a framed error. Messages contain numeric/string request IDs and finite JSON values.

`capabilities` returns the supported skill set, limitations and a JSON Schema suitable for an LLM tool call. See `examples/tactical_client.py` for an asynchronous client skeleton with a simulated slow planner. That example issues scripted choices and does not invoke or benchmark an LLM.

## Training and evaluation

```bash
uv run arena-rl train-tactics --family navigation --steps 32768 --envs 4 --output runs/tactics/navigation
uv run arena-rl train-tactics --family combat --steps 65536 --envs 4 --output runs/tactics/combat
uv run arena-rl train-tactics --family defense --steps 32768 --envs 4 --output runs/tactics/defense
uv run arena-rl evaluate-tactics --checkpoint runs/tactics/combat --episodes 32 --replays runs/combat-replays
uv run arena-rl mission --policies runs/tactics --output runs/mission
```

Tactical checkpoints have their own versioned observation/action schema; the four earlier local-skill checkpoints are not silently substituted. Navigation uses a bounded local waypoint, while the executor retains the global destination. This separates the controller's input scale from command travel distance. Public SDK routing conservatively avoids the assembly plinth for W3; it does not change physical map geometry or collisions. The policy chooses motion mode, speed, ballistic aim residuals, firing, leg height and jump. Public ballistic aiming remains assisted. This is not raw motor/torque learning or learned path planning.

The mission issues navigation → outpost attack → return home → regional defense → specified heavy pursuit in one continuous episode. Its issuer is a deterministic script to test command composition, not an LLM. A failing stage returns a nonzero CLI exit code and preserves its replay/report. The report includes tick/ammo continuity, simulator identity, each stage outcome and policy digests. The native replay has a `.commands.json` sidecar recording the command timeline and weights used.

Coverage is limited to these five commands and the recorded curricula. Patrol, escort, resupply, aerial control, and a dedicated airborne fly-slope skill are unsupported. Full-map navigation and longer attack chains must be judged using the mission results and held-out route tests, separately from short-course success rates.

For a different legal match roster, `serve-tactics --lineups path/to/lineups.json` accepts the Arena deployment format. Red slot 3 must remain W3 model 3. Only that slot is controlled by the learner; other red robots remain idle unless the surrounding host supplies controllers. `--opponent` can select an existing 2027 whole-team opponent submission (using Arena's default Docker sandbox). New enemy models/roles beyond the training distribution need their own held-out evaluation. The pure executor/encoding modules can also be called from an existing match host that already has teammate/opponent controllers.

Inside an existing host, the interface can be called without the JSONL process:

```python
from arena_rl.executor import TacticalExecutor
from arena_rl.tactical_encoding import encode, decode

executor = TacticalExecutor(slot="3")
executor.submit(command_from_llm, legal_red_observation)
# Every 100 ms of simulated time:
executor.update(legal_red_observation)
family = executor.execution_family
matching_tactical_policy = policies[family]
features = encode(executor, legal_red_observation, policy_family=family)
action, _ = matching_tactical_policy.predict(features, deterministic=True)
robot_actions = decode(executor, legal_red_observation, action, policy_family=family)  # only slot 3
feedback_to_llm = executor.feedback()
```

The host loads the policy family named by `executor.execution_family`, merges this slot's action with any teammate controllers, and supplies fresh observations. It retains responsibility for scheduling inference, applying new commands and keeping the simulator alive. The pure executor filters enemies by slot 3's own FPV list even if a whole-team observation contains other cameras.

Live service reset preserves normal game initial resources. In the 2027 rules W3 starts with zero ammunition; attack commands then report `out_of_ammo`. For an unranked training demonstration, explicitly reset with `scenarios: [{"robots": {"red-3": {"ammo": 200}}}]`, or run the recorded mission, which declares its ammunition in the initial scenario. The current adapter does not learn an ammunition acquisition policy yet. Without `--opponent`, live enemies are stationary; the ready response states the opponent mode. With a custom roster, the scene also includes legal allied telemetry, while enemy visual data remains restricted to slot 3's camera. `capabilities.trained_profiles` lists the ranges and enemy models used in the combat curriculum; accepted parameter bounds are broader than demonstrated coverage.

Long approaches/searches automatically use the navigation policy with a projected local-waypoint observation. The runtime switches to the requested combat/defense policy near the target/region, retaining the same command ID. For example, one `attack_structure` command can navigate from the home half and then shoot. Navigation-mode firing is suppressed; family switching is visible in `feedback.execution_family`. The LLM does not need to issue each route waypoint.
