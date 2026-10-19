# DeepSeek V4.1 Flash full-match validation

On 2026-10-18, the real `deepseek-flash` API supervised three independent 2027 matches. Each match had one learned red W3 (slot 3) against a scripted blue heavy (model 2711, slot 1) and W3 (slot 3). Profiles were siege, raid and hunt. This asymmetric 1-vs-2 setup is a declared evaluation scenario, not a standard full-roster tournament or a test against another LLM.

All three matches used normal initial robot states, zero starting ammunition, normal economy and the full **420-second** referee duration. The internal empty scenario specification did not override any robot property. No initial ammunition, HP, coins, positions or respawn state was injected. LLM commands never reset the simulator. Each game ended through the referee's normal timeout scoring.

## Results

| Seed | Opponent profile | Winner | Red / blue attack damage | Red / blue final base HP | Planner calls | Inference median / p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 9,000,000 | siege | red | 3,000 / 295 | 10,000 / 8,000 | 92 | 1.47 / 2.04 s |
| 9,000,001 | raid | blue | 100 / 2,280 | 9,020 / 10,000 | 72 | 1.44 / 1.93 s |
| 9,000,002 | hunt | blue | 100 / 4,160 | 7,080 / 10,000 | 79 | 1.53 / 2.19 s |

**1 win, 2 losses.** All three native v4 replays deterministically verified **8,400 ticks** each and the same recorded winner/digest. Actual 10 Hz learned execution owned only red slot 3; the opponent scripts received only blue-own legal observations. The evaluation used default wall/simulation speed 1. The games took about 422 seconds including startup/scheduling; the complete job took 456 seconds including saving and deterministic replay verification. Ten wall deadlines were missed across the batch; no simulation ticks were skipped or invented.

There were **243 actual API requests**, **229 valid decisions**, and **14 malformed coordinate tool calls** rejected by validation. No provider authentication/network fallback was substituted. Valid decisions had overall median latency **1.47 s**, p95 **2.09 s**. Every valid inference completed after simulated time had advanced; the logs contain a total 7,190 overlapping physical ticks during these requests (summed across actors/requests, not unique match ticks).

The host accepted 105 new goals, 123 explicit continuations and preserved one semantically identical running goal. New accepted skills were 66 resupply, 31 navigation and 8 structure-attack commands. Some were immediately rejected by robot availability/low-HP checks and ended as failed goals; command acceptance is not skill completion. Frequent retries in the losing games are a weakness, not evidence of useful strategy diversity. There were zero native `invalid_action` events in the three evaluated matches.

The model did not select `defend_region`, `engage_target`, `pursue_target` or `patrol` in this batch. The interface exposes them and tests their contracts, but this batch does **not** establish real-LLM tactical performance for those choices. Escort was unavailable because red deployed no ally; fly-slope was unavailable because the selected 2027 map lacks that dedicated terrain. The earlier local RL course results are separate evidence.

## What failed

In the siege game, the robot supplied, crossed the field, attacked the enemy outpost, then damaged its base. It later exhausted ammunition and spent significant time on blocked navigation/low-HP stops. A win against this profile does not establish robust long-horizon play.

In the raid game, the robot suffered repeated destruction/respawn cycles. Many retreat navigation attempts immediately stopped at the existing low-HP threshold, and supply routes also became blocked. The local navigation/supply courses did not cover enough contested retreats and recovery from these positions.

In the hunt game, the robot flipped around **63 s** and was below the upright threshold in 358 of 420 per-second snapshots. The current executor correctly stopped unavailable-robot actions. Its skill library has no self-righting/rescue option, and normal respawn did not repair this physical situation. The LLM repeatedly requested supply despite that condition; it needs better unavailable-state supervision and recovery-capable RL courses. The simulator was not patched to rescue it for the evaluation.

The 14 malformed coordinate calls were stopped before replacing a valid current goal. Their request context, validation errors and subsequent decisions are retained. Provider-side schema enforcement or a scalar-coordinate tool representation could reduce these failures; this version uses the stable Chat Completions endpoint plus host validation.

These results verify the real API → tactical tool → frozen RL → referee/replay pipeline. They show that **full-match reliability remains limited**, particularly for defensive choices, contested retreat and physical recovery. The runner's `healthy_control` field denotes completed integration/continuity with valid API commands, not robot health, tactical quality or victory.

## Reproduce and inspect

```bash
uv run arena-rl llm-matches --policies checkpoints/tactics \
  --matches 3 --seed 9000000 --max-calls 150 \
  --output runs/deepseek-full-new
```

Use a new output directory. API sampling and wall-clock arrival make a fresh run nondeterministic even with the same simulator seeds. Native replay verification is deterministic and does not call the API. The checkpoints remain frozen; no RL training or weight update occurred in this evaluation.

The private repository contains the [report, complete decision/telemetry logs, command sidecars and SHA256 manifest](../evaluations/deepseek-v4.1-flash-2026-10-18). Full native replays (about 190 MiB each) are retained locally under `runs/deepseek-v4.1-flash-full/`, excluded from Git to avoid adding hundreds of MiB to every clone. Their hashes and verified digests are in the committed report/manifest. The source hash snapshot records the evaluated controller; only package-version metadata changed afterward for 0.4.0.

All **78 local tests** passed across the pure protocol/provider tests and simulator integration/regressions. The new full-duration test uses an explicitly labeled fake planner, and the authentication-failure test checks that partial runs cannot be reported as full matches. Those tests are not included in the real API performance counts above. Automatic CI runs pure contracts/local mock-provider checks; full simulator integration still needs Arena-1's matching unpublished local prerequisite.

Arena-RL stays private. Arena-1 was not committed, pushed or released for this work.
