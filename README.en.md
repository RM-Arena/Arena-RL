**English** | [中文](README.md)

# Arena-RL

A reinforcement learning training library built on [Arena-1](https://github.com/RM-Arena/Arena-1). It reuses Arena's Rust physics, rules and replays, as well as the `ArenaEnv` / `VectorArena` parallel environments, and does not maintain a separate simulator.

**Full-team version (0.6)**: a top-level LLM chooses a high-level objective for every robot on the 2027 team (attack the outpost, defend base, pursue the enemy heavy/hero, occupy the fortress, seize the central highland card, push base, resupply, ...). The Harness handles rule state, feasibility filtering and command compilation. Low-level execution uses a single PPO policy that is universal across the 15 ground models of 2027 and conditioned on robot parameters; the aerial hangar uses a learned dive-timing policy. Checkpoints are in `checkpoints/ground`; see [Full-team LLM + universal RL](docs/multi-robot.md) for documentation, training commands, Mac/CUDA resource utilization and evaluation.

```bash
uv run arena-rl team-match --policies checkpoints/ground --commander llm --matches 2 --output runs/team-llm
uv run arena-rl serve-team --policies checkpoints/ground --realtime   # an external LLM client commands the whole team over JSONL
sh examples/train_ground_all.sh runs/ground-v1 8 auto                  # retrain all ground policy families locally
```

<p align="center"><img src="docs/images/en/architecture.svg" width="780" alt="System architecture: LLM commander → Harness → TeamController → RL policy → Arena simulator"></p>

| One full match: LLM command + RL execution (red wins) | LLM command timeline for each red robot |
| --- | --- |
| ![Match trajectories](docs/images/en/match-trajectories.png) | ![LLM command timeline](docs/images/en/llm-timeline.png) |
| **Universal policy vs expert controller vs random (independent-seed evaluation)** | **Training curves for the 15 models** |
| ![Evaluation summary](docs/images/en/eval-summary.png) | ![Training curves](docs/images/en/training-curves.png) |
| **Per-model evaluation heatmap** | **Training throughput optimization on a local Mac** |
| ![Per-model evaluation](docs/images/en/eval-heatmap.png) | ![Training throughput](docs/images/en/throughput.png) |

All figures are generated from the checkpoints, evaluation reports and match records in the repository (`uv run --with matplotlib python examples/make_figures.py`); see the [full-team documentation](docs/multi-robot.md) for interpretation and limitations.

LLM matches in the single-robot version use the **2027 serial-leg infantry 305, red slot 3**. Its checkpoints are in `checkpoints/serial305` and support low-HP withdrawal and recovery after physical self-righting. See [305 training and interface notes](docs/serial305.md). The older W3 checkpoints are retained for reproducing earlier experiments.

The first version uses the **2027-season W3 two-wheel wheel-legged infantry, model 3, slot 3**. Each process trains only one red robot; blue robots run from scenario scripts. Training uses real PPO to update network weights.

## Installation

Requires Python 3.10+, Rust/Cargo and two sibling source directories:

```text
workspace/
  RMUC-Arena/    # Arena-1 source; must include the parallel-environment and training-scenario interfaces
  Arena-RL/
```

```bash
cd ../RMUC-Arena
cargo build --release --manifest-path simulation/Cargo.toml
cd ../Arena-RL
uv sync --locked --extra dev
uv run arena-rl --arena-root ../RMUC-Arena doctor
```

You can also specify the simulator with `ARENA_ROOT=/absolute/path/to/RMUC-Arena`. `doctor` checks the required interfaces and shows source and binary fingerprints. Arena's full frontend and desktop application are not training dependencies.

**Current development dependency**: requires `rmuc_arena/env.py`, `vector_env.py` and `simulation/src/training_scenario.rs` in Arena. If the downloaded Arena version lacks these files, use a source version that contains these changes. Training metadata records the actual Git commit, whether the working tree has changes, SHA256 hashes of key source files, the binary SHA256 and per-episode rule/engine hashes. If the interface cannot be satisfied, it fails with an error and does not fall back to another physics implementation.

## Training low-level skills

```bash
uv run arena-rl train --task move --steps 32768 --envs 4 --output runs/w3/move
uv run arena-rl train --task ramp --steps 32768 --envs 4 --output runs/w3/ramp
uv run arena-rl train --task outpost --steps 32768 --envs 4 --output runs/w3/outpost
uv run arena-rl train --task defend --steps 32768 --envs 4 --output runs/w3/defend
```

The low-level decision rate is **10 Hz (simulated time)**; computation can run faster than real time. Actions consist of movement direction, aim offset, pitch offset, fire, leg length and jump. The learner can control only its own slot.

| Skill | Initial curriculum | Success condition |
| --- | --- | --- |
| `move` | Random spawn and goal points in an open area | Enter the 0.35 m goal radius at a speed below 0.8 m/s |
| `ramp` | Red 15° road ramp on the real 2027 map, with lateral spawn perturbation | Cross the crest from the low end, rise more than 0.12 m, reach the platform beyond the ramp and stay upright |
| `outpost` | Start near the enemy outpost with 200 training rounds | Deal at least 40 cumulative damage to the real outpost armor |
| `defend` | Near the base entrance, with one scripted infantry approaching continuously | Deal at least 40 cumulative damage to the intruder |

**Aim assistance is explicit**: shooting skills use the ballistic aiming provided by the Arena SDK, while PPO learns movement, aim correction, fire timing and wheel-legged actions. The first version does not learn full perception and aiming from raw images or raw gimbal commands, nor does it train motor current or joint torque. The ramp task does not require going airborne, so its success rate must not be read as a success rate for jumping stunts.

`outpost` / `defend` are introductory curricula with short distances and low damage thresholds, intended to validate the training pipeline. They do not mean the agent can already destroy an entire outpost, eliminate a full invading team or win a full match. Reward and curriculum definitions are centralized in `src/arena_rl/tasks.py`, where distance, damage targets and disturbance strength can be raised step by step.

## High-level decisions

```bash
uv run arena-rl train-high --skills runs/w3 --steps 4096 --envs 4 --output runs/w3/manager
uv run arena-rl evaluate-high --checkpoint runs/w3/manager --episodes 32 --output runs/w3/manager/evaluation.json
```

Every **1 second of simulated time**, the manager policy chooses `outpost` (attack) or `defend` (defend base). It invokes the frozen low-level model within the same episode for at most 10 low-level actions; task success, robot disablement or the intruder crossing the boundary can end it early. The low-level models' weights and digests are checked before and after training.

The first-version high-level curriculum contains two local scenarios, "assault near the outpost" and "interception at the base entrance"; the scenario type is not fed directly into the model input. The high level decides from its own robot state and legal visual observations; rewards come from actual damage, survival and intruder boundary crossings, with no direct reward for a correct option label. Enemy behavior is independent of the chosen skill. The scenarios are simple; long-distance full-map scheduling and suddenly appearing new threats belong to later curricula.

## Evaluation, replay and resumed training

```bash
uv run arena-rl evaluate --checkpoint runs/w3/move --episodes 32 --seed 1000000 --output runs/w3/move/evaluation.json --replays runs/w3/move/replays
uv run arena-rl evaluate --task move --baseline random --episodes 32 --seed 1000000
uv run arena-rl train --task move --steps 32768 --envs 4 --resume runs/w3/move --output runs/w3/move-continued
uv run arena-rl suite --config configs/w3.json --output runs/full
```

Evaluation uses independent seeds: training starts from 7 by default, low-level evaluation from 1,000,000 and high-level evaluation from 2,000,000. `suite` compares random and trained policies on the same evaluation seeds; if a low-level skill has no evaluation successes at all, high-level training is stopped.

Each training directory contains `policy.zip` (weights and optimizer), `metadata.json` (versions and training evidence) and `metrics.jsonl` (learning progress). Resumed training restores the network and optimizer and starts from a new episode; it does not promise a bit-exact restoration of the sampler state at the moment of interruption. By default, checkpoints that do not match the simulator source and the embedded engine/rule hashes are rejected (builds of the same source on different platforms are compatible); only an explicit `--allow-engine-change` allows the simulator to change when resuming low-level training or evaluating. Only load PyTorch checkpoints that you generated yourself or that come from a trusted source.

Replays use Arena's native v4 format, preserving the training spawn point, ammunition and complete actions, and can be imported directly into Arena's replay page:

```bash
cd ../RMUC-Arena
uv run rmuc-arena replay verify ../Arena-RL/runs/w3/move/replays/move-1000000.json
```

The Gymnasium interface is available from `arena_rl.envs.SingleRobotEnv`; the SB3 adapter `SkillVecEnv` uses Arena's persistent parallel processes without wrapping them in an extra layer of SB3 subprocesses. The environment distinguishes success/failure termination from time truncation, preserves the terminal observation, and then resets automatically for SB3.

## Tests

```bash
uv run pytest -q
```

Covers the Gymnasium contract, action ownership, parallel process isolation, fixed-seed reproducibility, real ramps and damage, replay verification, network weight updates, digest checks and high-level training with frozen skills.

Implementation references: [Gymnasium Env interface](https://gymnasium.farama.org/api/env/), [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). The project follows the non-commercial learning license provided with the repository; Arena and the training dependencies each retain their own licenses.

## Working reference models

See the [first training run record](docs/first-training.md) and the [reference checkpoints](checkpoints/w3). Each of the four low-level skills completed all 32 independent scenarios; the high level achieved 31/32. The record also publishes random baselines: the short-distance, aim-assisted shooting curricula are easy to begin with, so they do not yet support a claim of stronger real-combat shooting ability.

## Top-level LLM commands and RL execution

A new [tactical interface](docs/llm-tactics.md) supports navigation, attacking a specified structure, engaging/pursuing a specified robot, regional defense, resupply, escort, patrol, the dedicated fly slope, and withdrawal and recovery for the 305. The top-level LLM dispatches parameterized JSON commands, and the low-level PPO executes continuously at 10 Hz; switching commands preserves the same simulation process, position and ammunition. Explicit target locking, short-term target memory, pursuit leash, command staleness checks and execution feedback are supported.

```bash
uv run arena-rl serve-tactics --policies checkpoints/tactics --realtime --record
uv run arena-rl mission --policies checkpoints/tactics --output runs/continuous-mission
```

In the generic service, LLM inference is provided by an external client; the repository also provides `llm-matches`, which calls the paid DeepSeek API when explicitly run. The older two-choice PPO high level remains available as a baseline.


## Historical W3 evaluation: resupply, escort, patrol and fly slope

`checkpoints/tactics` now contains five policy groups. Resupply, escort and patrol share the 2027 `support` policy; the dedicated fly slope uses the 2026 `flight` policy, which covers the forward crossing of the real 650 mm gap, the return jump and landing stabilization. The 2027 map has no such dedicated fly slope, so the service selects the corresponding map with `--season 2026`.

```bash
uv run arena-rl train-tactics --family support --steps 32768 --envs 4 --output runs/support
uv run arena-rl train-tactics --family flight --steps 16384 --envs 4 --output runs/flight
uv run arena-rl serve-tactics --policies checkpoints/tactics --season 2026 --realtime --record
```

Resupply uses real purchases, deductions and healing in the supply zone; escort controls only its own robot, while the target teammate is controlled by the match host; patrol completes all waypoints and dwell times in order. Fly slope is first initialized from successful demonstrations and then updated with PPO; deployment does not substitute the demonstration script for the policy. See the [tactical interface](docs/llm-tactics.md) and [extended evaluation](docs/extended-results.md) for specific commands, scope and training records. The two previously failing navigation scenarios have also been fixed, and the full navigation regression is 32/32.

## Connecting a top-level LLM

A real DeepSeek V4.1 Flash top-level scheduler is now provided: function calls select skills, and the frozen PPO policies execute only red slot 3. LLM requests run asynchronously, and the current skill continues while waiting; a timeout or invalid output keeps the current execution and logs an error. It uses `DEEPSEEK_API_KEY` from the existing shell (`DeepSeek_API_Key` is also accepted):

```bash
uv run arena-rl llm-matches --policies checkpoints/serial305 --matches 3 --max-calls 150 --output runs/serial305-full
```

By default it runs three normal 420-second 2027 matches in parallel against explicitly labeled scripted hero and infantry opponents, with only one 305 on red. This validates real LLM scheduling, skill switching and the full-match pipeline; it is not a standard full-roster team match. Decision logs, inference latency, failure reasons, match results and native replays are all saved. See the [LLM integration notes](docs/llm-agent.md) and [early W3 match records](docs/llm-match-results.md) for parameters, permission boundaries and how to interpret the records.


For the current 305 weights, withdrawal/recovery curriculum, safe-coordinate interface and full match records, see the [training notes](docs/serial305.md) and [evaluation results](docs/serial305-results.md). The old W3's 32/32 navigation result does not apply to the 305; the current model's limitations are reported separately.

In the 0.5.1 refinement release, the original navigation and fly-slope regressions are both 32/32, and forward, return and 420-second 2026 match validation was completed within the same process. See the [refinement evaluation](docs/serial305-refinement-results.md) for the limitations of the expanded perturbation tests and new real-LLM matches.

![Independent-evaluation success rate of each serial-leg 305 policy family](docs/images/en/serial305-evaluation.png)

## Documentation

Every document has English and Chinese versions; switch between them at the top of the page.

| Document | Contents |
| --- | --- |
| [Full-team LLM + universal RL](docs/multi-robot.md) | 0.6: full-team LLM command, Harness, a universal policy for 15 ground models, aerial dive policy, Mac/CUDA training, evaluation and full-team matches |
| [305 training and interface](docs/serial305.md) | Withdrawal, self-righting recovery and safe-coordinate interface for the serial-leg 305 |
| [305 refinement evaluation 0.5.1](docs/serial305-refinement-results.md) / [305 evaluation 0.5.0](docs/serial305-results.md) | 305 regressions, expanded tests and real-LLM matches |
| [LLM tactical interface](docs/llm-tactics.md) | Single-robot parameterized tactical commands with RL execution |
| [LLM integration notes](docs/llm-agent.md) / [W3 match records](docs/llm-match-results.md) | DeepSeek tactical planner, full-match runner and early results |
| [Tactical execution validation](docs/tactical-results.md) / [Extended evaluation](docs/extended-results.md) | W3 tactical policies; resupply/escort/patrol/fly-slope evaluation |
| [First training run record](docs/first-training.md) | First training of the W3 low-level skills and high-level manager policy |
| [Boundary and extension points](docs/architecture.md) | Boundary with Arena, parallel environments and how to extend |
