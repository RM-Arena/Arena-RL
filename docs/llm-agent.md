# DeepSeek tactical planner and full-match runner

Arena-RL now has a real API-backed tactical planner. DeepSeek chooses a parameterized skill; the existing frozen PPO policies execute only **red slot 3** (current bank: serial-leg model 305; legacy W3 bank: model 3). Inference runs on worker threads while the simulator and low-level controller continue. No scripted tactical policy replaces an API failure.

The requested provider for this integration is DeepSeek V4.1 Flash. Its documented API model ID is [`deepseek-flash`](https://api-docs.deepseek.com/quick_start/agent_integrations/). The client uses `https://api.deepseek.com/beta/chat/completions`, function calls, and `thinking: {"type":"disabled"}` for low-latency decisions. It records both requested and returned model IDs. It does not use the rejected OpenAI credential or silently substitute another model.

## Run

Use an existing shell credential. `DEEPSEEK_API_KEY` is preferred; the existing `DeepSeek_API_Key` variable is also supported. For another variable name, pass `--api-key-env NAME`. The program never reads a shell startup script, stores a credential, prints authorization headers, or gives credentials to the model.

```bash
uv run arena-rl llm-matches \
  --policies checkpoints/serial305 \
  --matches 3 --seed 9000000 --max-calls 150 \
  --output runs/deepseek-full
```

Each actor is a normal **420-second** 2027 referee match, ending earlier only if the game rules end it. The learner starts at its normal spawn with normal HP, zero starting ammunition and normal team income. Resupply, damage, deaths, respawns, captures and purchases use the existing simulator. A new skill cannot reset any of them. Goal completion/failure does not end the game; the LLM can issue another objective in the same process.

The default run contains one red model-305 infantry versus two **declared scripted** blue opponents (a model-2711 heavy in slot 1 and W3 in slot 3). This is an asymmetric single-robot evaluation, not a standard full-roster tournament. The three profiles are `siege`, `raid`, and `hunt`; they prioritize structures, opportunistic combat, or visible robot pursuit respectively. Each controller uses only its own team's legal observation, normal ammunition acquisition and SDK routing/aiming. They are independent evaluation opponents, not part of the learned red agent. There is no idle red teammate presented as an LLM-controlled robot.

`--matches 1..8` runs independent actors concurrently. Each has at most one pending inference request; command responses cannot be mixed between games. `--plan-interval` is a simulated-time interval (default 5 seconds), with earlier requests on completion/failure/respawn. `--timeout` bounds API I/O, `--max-age` rejects late results, `--max-calls` bounds requests per match, and `--speed` changes the wall/simulation clock ratio. Default speed 1 targets real time. Accelerated runs must be labeled separately because inference delay then spans more simulated time. `--no-verify` skips deterministic replay verification, not game rules or command validation.

## Model interface

The API receives only the legal red scene: own robot/resources, allied telemetry when present, own FPV enemies, age-marked last sightings, public bases/outposts/control state, and public terrain/roster context. It does not receive blue-own robot state, opponent resources, script profile, replay state, or training rewards. Historical input is bounded to eight recent decisions with execution feedback; no private reasoning transcript is requested or saved.

Function tools expose skill arguments, not raw motion/fire commands or simulator operations. Host code assigns command IDs and basis ticks. The tool surface includes available skills plus `continue_current` and `cancel_current`; roster/map filtering removes escort without a deployed ally and removes the dedicated fly-slope tool on 2027. In the current single-red-305 2027 matches, nine skills are available, including retreat and recover. The same planner client can be given a 2026 `TacticalSession` and its flight-capable tools by another host; this match runner does not claim a 2026 flight benchmark.

`continue_current` preserves progress/timers. Semantically identical running goals are also preserved instead of restarted. New valid commands cancel the previous goal atomically. Numeric bounds, target identities, friend/foe, public rosters, protected structures and observation age are checked by the existing command/executor layer. An invalid result is logged and rejected; an API timeout leaves the current RL skill running. If no valid skill is active, the robot waits for another valid decision. Authentication/permanent provider errors stop the evaluation and retain a partial report/replay; they cannot be reported as complete matches.

The existing 10 Hz action codec still uses public route hints and ballistic assistance. The API integration keeps the selected 305 motor policies frozen during matches; their retraining is documented separately. Local combat courses do not establish full-match strength. The LLM can request navigation, supply, attack, pursuit, defense and patrol, but these requests can fail physically; feedback and native events are the evidence.

## Artifacts and interpretation

Each output directory contains:

- `decisions.jsonl`: legal request snapshots, returned function calls, response IDs, usage, latency, command acceptance/rejection and execution transitions. `advanced_ticks_while_inference` proves that simulator time continued during requests.
- `telemetry.jsonl`: compact per-second legal red scenes.
- `match-SEED.json` and `.commands.json`: complete native v4 replay and command timeline with policy digests.
- `report.json`: referee outcome, elapsed match time, continuity checks, counts/latencies/errors, end-state team results, events, replay verification and source/checkpoint provenance.

Usage counters describe successfully parsed responses; they are not a billing ledger for failed/discarded requests. Replay files can be large for full matches and remain under ignored `runs/`; selected summaries/traces may be kept under `evaluations/`. Import native JSON into Arena's replay page. The replay reproduces the recorded physical actions without contacting the LLM API.

A successful runner exit means full referee completion with real accepted LLM commands and continuity, rather than a promised win. Read the recorded outcome, invalid-action events, failed skills, deaths and idle periods separately. Deterministic scripted/mock tests verify plumbing only and are labeled as test doubles, not model evaluations.

Arena-RL remains private. The simulator prerequisite in Arena-1 remains local and is not pushed or released by this work.


The current planner can select only prevalidated public waypoint IDs when the runner supplies a catalog. The host resolves IDs to coordinates; manual tactical JSON remains compatible with two-element arrays. Custom hosts without catalogs use bounded scalar x/y objects. See [serial-305 migration, curricula and mechanical assistance](serial305.md). Historical W3 results in the earlier report are retained as a separate model/agent version.
