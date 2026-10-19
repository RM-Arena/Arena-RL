# Serial-leg 305: withdrawal, recovery and safe LLM coordinates

The current single-robot agent controls **red slot 3, model 305**, an existing legal serial-leg infantry design. The earlier W3 model 3 has no complete self-righting mechanism. Following the user's choice, the active policy bank was retrained for 305; the old `checkpoints/tactics` and `checkpoints/w3` banks remain available as historical model-3 references. Arena physics, rules and robot specifications were not changed for this migration.

The current correction update is **0.5.1**; see [refinement results](serial305-refinement-results.md) for navigation/flight regressions, expanded held-out tests and new full matches. The original 0.5.0 evidence remains [separately recorded](serial305-results.md).

## Run the current agent

```bash
uv run arena-rl llm-matches --policies checkpoints/serial305 \
  --matches 3 --seed 9000020 --max-calls 150 \
  --output runs/serial305-new
```

Use the existing `DEEPSEEK_API_KEY` or `DeepSeek_API_Key` environment variable. The selected API model remains `deepseek-flash`, with thinking disabled. One red robot is learned/LLM-controlled; the blue heavy and infantry are declared independent scripted opponents. Normal initial ammunition, income, damage, respawns and the 420-second referee are retained. API calls are asynchronous and RL runs at 10 Hz simulated time.

## Two additional objectives

`retreat` withdraws to an optional safe `point`; without it, it seeks the own supply zone. It remains executable below the ordinary navigation/combat low-HP threshold. Arrival requires low speed, and timeout/failure remains explicit. Reaching supply does not invent healing or ammunition: a subsequent `resupply` objective uses the normal card, purchases and passive healing.

`recover` settles the serial-leg chassis and waits for a stable upright stance. Recovery can also run as a temporary subphase of another objective, preserving that objective's ID, simulator tick, pose, HP and ammunition. It requires upright >0.93, grounded support, low linear/angular motion, no active mechanical righting, and a 0.5-second stable hold before continuing. A bounded timeout reports failure rather than claiming recovery. New commands after a failed recovery start a fresh supervision interval without changing physical state.

**Mechanical assistance is explicit.** The 305's existing full-turn hip firmware supplies physical self-righting torques. RL chooses local settling/motion/gait actions and resumes movement afterward; it does not learn raw hip torques or set a quaternion. A learned stop can retain a steering target, so the chassis can turn in place. Sharp turns finish before translation, and backward motion can retain the nearer axle heading. At road bumps, ride-height and momentum guards implement the public chassis clearance requirement.

Route planning uses public geometry: ramp entrances/exits are completed as a committed transfer, and portals are compared by legal route cost rather than straight-line distance. A public terrain/wall grid can connect routes that the generic SDK's conservative waypoint graph misses. Local escape backs out along the chassis axle before attempting another turn. None of these helpers changes a collider, forbidden zone or chassis power limit. All motor behavior remains in the original Rust engine.

## Coordinate generation

The integrated planner now selects **prevalidated public waypoint IDs** and defense-region IDs. Coordinates are resolved by the host. Catalog points are filtered using chassis clearance, obstacles and forbidden areas. Strategic tools such as `attack_structure`, `defend_region` and `resupply` already include navigation; the prompt discourages unnecessary short navigate/cancel cycles.

DeepSeek uses its [strict function-call endpoint](https://api-docs.deepseek.com/guides/tool_calls/) at `https://api.deepseek.com/beta`. Function arguments have closed object schemas and host validation. Custom hosts without a waypoint catalog can still use strictly typed `{x, y}` objects, with finite values and current-field bounds checked before command submission. The public tactical protocol continues to accept two-coordinate arrays; this preserves existing clients. Unknown catalog IDs, malformed coordinates and stale decisions leave the active goal intact. Credentials never enter model input or logs.

## Training

| Policy family | Controlled model | New PPO samples | Initialization | Observation contract |
| --- | --- | --- | --- | --- |
| combat | 305 | 32,768 | Successful demonstrations + PPO | 52 features |
| support | 305 | 32,768 | Successful demonstrations + PPO | 74 features |
| defense | 305 | 32,768 | Successful demonstrations + PPO | 52 features |
| mobility | 305 | 32,768 initial + 8,192 composed fine-tuning | 160 successful demonstrations, explicit parent initialization + PPO | `serial305-mobility-v1`, 64 features |
| navigation | 305 | 16,384 original + 16,384 local refinement | Original network retained for transit beyond 4 m; refined network for local approaches | `serial305-navigation-v1`, 64 features |
| flight | 305 | 8,192 new | 183/192 successful demonstrations, then PPO at 5e-6 | `serial305-flight-v3`, 84 lane-relative features, season 2026 |

All use four parallel actors. Metadata records model/season/schema, parent or demonstration provenance, PPO sample/update counts and weight hashes before and after PPO. Cross-family initialization requires matching controlled model and feature/action contract. The deployed bank rejects mixed model IDs and requires the trained mobility policy for 305. Explicit rosters are validated instead of silently rewritten.

The mobility curriculum includes:

- Real front/side/roof upset induced by ordinary jump/lean/leg commands against map edges, followed by stabilization. No initial quaternion or mid-episode pose edit is used.
- Withdrawal after actual projectile damage, destruction and normal automatic respawn at **10% HP**, with the referee's real weak/protection state.
- The recorded central-kerb contact around `(14.52, 2.23)` and the highland position around `(11.1, 6.51)`, with legal ramp exits and escape motions.
- Supply-wall contact around `(3.89, 13.43)`, with both short forward targets and withdrawal to supply.
- Physical fall followed by withdrawal under the same command ID, rather than stopping at the recovery goal.

There are eight template variants with bounded perturbations, not 64 unrelated map geometries. The composed fine-tuning set contains 160 successful episodes (20,317 transitions); the actual post-demonstration PPO weight update is checked separately. In recovery, intent features are identical whether recovery is explicit or temporarily suspends another goal. The local stall detector uses the command's actual arrival radius, so it also catches learned stops just outside the target region.

These are physical preparation sequences and are retained in native replay. The wounded curriculum's shooter stops after preparing the state; it does not establish learned bullet evasion under arbitrary sustained fire. Full matches separately exercise ongoing opposition.

```bash
uv run arena-rl train-tactics --family mobility --model-id 305 --steps 32768 --envs 4 --seed 1801 --output runs/mobility-new
uv run arena-rl train-tactics --family navigation --model-id 305 --initialize-from checkpoints/serial305/mobility \
  --steps 16384 --envs 4 --seed 1711 --output runs/navigation-new
uv run arena-rl evaluate-tactics --checkpoint checkpoints/serial305/mobility --episodes 64 --seed 7000200
uv run python examples/evaluate_composed_navigation.py --output runs/composed-navigation.json
uv run python examples/validate_recovery.py --output runs/physical-regressions
```

Training used the recorded source snapshots/hashes and saved demonstration datasets. Route/safety integration was subsequently repaired and independently evaluated; a fresh training run with the current helpers is not promised to reproduce identical weight bytes. Native action replays remain deterministically verifiable without inference.

## Evidence and limits

See the evaluation reports under `checkpoints/serial305` and [full-match evidence](serial305-results.md). Final held-out mobility testing uses seeds 7,000,200–7,000,263: 27 explicit recovery and 37 withdrawal objectives. Withdrawal includes the recovery-then-retreat composition. Physical regressions also preserve and deterministically verify representative native replays. Idle/random comparisons on the same seeds distinguish the existing firmware contribution from useful learned movement. A completed `recover` command can begin upright; `recoveries_completed` is an execution counter, not a count of real flips. Native `self_righting`/`self_righted` events and pose samples report physical events separately.

The original 32-case navigation and 32-case flight regressions now pass. Wider perturbed courses still contain failures; the [refinement report](serial305-refinement-results.md) gives exact counts separately from these regression sets. A serial chassis can still become mechanically wedged or exhaust its power budget. No global win-rate or universal recovery guarantee follows from a few matches. Flight migration uses its own 2026 curriculum and is unavailable as a dedicated map skill in the 2027 matches.

Arena-RL was private when this was recorded; it has been public since 2026-10-19. Arena-1's existing local simulator prerequisite remains unpushed, and this work does not commit, push or release Arena-1.


## Refined fly-slope controller

The 305 flight policy now has 84 inputs: versioned physical route features plus explicit approach/alignment/run/air/landing/recovery phase, launch state, gap occupancy, along-lane speed and landing-distance error. It uses a lane-relative frame and masks gimbal, ammunition and economy fields that should not change this motor skill. W3 and archived 305 `w3-flight-v1` policies retain their 74-input evaluation path. The unreleased intermediate phase-only prototypes are development runs, not shipped policies.

Entry alignment checks lateral error below 25 mm, orientation and stable support before launch. Ground self-righting supervision does not interrupt ordinary airborne tilt; physical air time remains observed while recovery is active. Landing still requires a real airborne gap crossing, an upright grounded chassis and the original low-speed/hold conditions. Forward jumps use a bounded ordinary stroke, with full available stroke on the gap floor when the network requests another jump. The actor never sets body pose or HP. Successful demonstration collection includes physical gap-lip retry actions instead of discarding all such attempts.

The refined navigation network is used within 4 m of its goal. An explicitly recorded frozen transit network preserves the previous long-distance behavior; recovery and local escape keep their existing supervision. Both networks share the model-305 64-input contract, and the full-match reports record both policy digests.
