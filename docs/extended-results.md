# Support, flight and navigation regression results

Measured locally on 2026-10-18 with the Arena-1 checkout and Rust engine pinned by each checkpoint's provenance. Arena-RL was private when this was recorded; it has been public since 2026-10-19. Arena-1's parallel-environment/scenario prerequisite is still local and has not been pushed or released; a fresh public Arena-1 clone is therefore not yet sufficient to reproduce this run. GitHub automatic CI checks the pure contracts; simulator integration needs the matching local prerequisite or a future explicitly selected Arena ref.

## Held-out results

| Course / command | Learned policy | Random actions, same seeds |
| --- | --- | --- |
| Local ammunition + healing (`resupply`) | 18 / 18 | 5 / 18 |
| Follow a moving ally to destination (`escort`) | 8 / 8 | 0 / 8 |
| Ordered route with dwell (`patrol`) | 10 / 10 | 0 / 10 |
| Dedicated forward/reverse gap crossing (`fly_slope`) | 40 / 40 | 0 / 40 |
| Navigation, including original failed routes | 32 / 32 | See original tactical baseline |
| Specified-target combat regression | 32 / 32 | See original tactical baseline |
| Regional defense regression | 32 / 32 | See original tactical baseline |

Support uses seeds 5,000,000–5,000,035. Flight uses 6,000,000–6,000,039: red forward 6, red reverse 9, blue forward 14, blue reverse 11. Original tactical regression uses 3,000,000–3,000,031. See machine-readable evaluation/random reports in `checkpoints/tactics/{support,flight,navigation,combat,defense}`. These are finite local curriculum counts, not calibrated success probabilities in a full match.

The separate live `PolicyBank`/`TacticalSession` validation completed **80 / 80** goals: 40 support seeds beginning at 5,000,000 and 40 flight seeds beginning at 6,000,000. This exercises actual family dispatch, including navigation-to-support composition. Every replacement retained simulation tick and ammunition; the same simulator process persisted through each command. `checkpoints/tactics/validation/runtime.json` records all outcomes and families used. The issuer was a script, with no LLM inference.

Seven representative native v4 replays cover all three support commands and all four lane/direction flight combinations. Arena deterministically verified 88, 202 and 102 ticks for escort, supply and patrol; flight verification covered 118 ticks for each reverse lane and 52/58 ticks for forward lanes. Preparation actions remain in these replays. Native files and command sidecars are in each new family's `replays/` directory.

## What was trained

| Family | Seed | PPO samples | Initialization |
| --- | --- | --- | --- |
| `support` | 507 | 32,768 | Random network weights; 4 parallel actors |
| `flight` | 809 | 16,384 | Successful scripted demonstrations, then PPO; 4 actors |

Support is one goal-conditioned network for resupply, escort and patrol. Flight attempted 64 demonstration episodes with seeds 100,809–100,872 and retained 63 successful episodes / 2,017 transitions. Behavior cloning ran 160 epochs; metadata records its losses, action fit and dataset SHA256. PPO then updated the weights on fresh simulator episodes. `flight/demonstrations.npz`, `metadata.json`, `metrics.jsonl` and `policy.zip` retain this evidence. Deployment uses network predictions, with no demonstration-script fallback. The initial 32,768-step from-scratch flight PPO trial had no successes; initialization was needed for this small training budget.

Navigation/combat/defense weights were retained. The navigation repair is in the public-route/action adapter, evaluated with the same navigation checkpoint and seeds. The original 52-feature schema remains unchanged; support/flight have separate 74-feature schemas. Simulator physics, collisions and rules were not altered for these skills.

## The two navigation repairs

Seed **3,000,008** tried to turn while a wheel was still on a hump, although its center had left the footprint. The route adapter now clears the physical wheel footprint along the existing axle heading before turning. Seed **3,000,027** compressed the legs while attempting a bump; navigation now maintains at least the normal 0.245 m ride height. With the original weights, the two exact scenarios complete in **27.6 s** and **27.3 s**. Their native replays verified **552** and **546** ticks. `validation/navigation-fixed.json` contains verification hashes and detailed outcomes.

The two larger native files are stored as `navigation/replays/navigation-3000008.json.gz` and `navigation-3000027.json.gz`. Decompress to import in the Arena replay page or verify them, for example:

```bash
gzip -dk checkpoints/tactics/navigation/replays/navigation-3000008.json.gz
cd ../RMUC-Arena
uv run rmuc-arena replay verify ../Arena-RL/checkpoints/tactics/navigation/replays/navigation-3000008.json
```

All other new representative replays are ordinary uncompressed v4 JSON. `checkpoints/tactics/sha256.json` covers published checkpoint/report/replay files.

## Physical task boundaries

Supply preparation wounds the robot with ordinary enemy projectiles, then the shooter stops. Healing, ammunition acquisition, purchase limits and resource deductions are the simulator's real rules. Escort uses a slow scripted ally and patrol uses three nearby route points. Optional contact shooting is available but hostile interference, fast allies, long patrols and arbitrary distant supply approaches have not been demonstrated by these counts.

Flight uses the actual **2026** 650 mm gap, both mirror lanes, and both directions. Reverse episodes first complete a real forward crossing in the same episode, earning access through ordinary rules. Success requires at least 0.1 s genuinely airborne over the gap, a grounded upright landing, speed below 1.2 m/s and a stable landing hold. The adapter aligns the lane heading and uses public gap geometry; the network chooses motion, speed, gait, power, lean and jump. Forward entrance tolerance is 0.11 m and reverse alignment tolerance is 0.055 m. Courses start near the staging positions, with small forward spawn perturbations; no enemy fire is present. There is no corresponding dedicated fly-slope object on the 2027 map.

## Validation commands

```bash
uv run pytest -q  # 63 passed locally
uv run arena-rl evaluate-tactics --checkpoint checkpoints/tactics/support --episodes 36 --seed 5000000
uv run arena-rl evaluate-tactics --checkpoint checkpoints/tactics/flight --episodes 40 --seed 6000000
uv run python examples/validate_extensions.py --episodes 40 --output runs/extended-runtime
```

The full local test suite checks old checkpoint compatibility, command validation and atomic rejection, ownership, budget/contact handling, ordered patrol dwell, actual forward/reverse permission, the two original failed navigation seeds, service map filtering, real-time operation, native replay verification and learning updates. The current automated CI contract subset can run without the unpublished Arena-1 prerequisite. None of these checks benchmarks a real high-level LLM's tactical judgment.
