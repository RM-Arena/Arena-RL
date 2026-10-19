**English** | [中文](zh/serial305-results.md)

# Serial-leg 305 evaluation — 2026-10-18

This records version 0.5.0. See [0.5.1 refinement results](serial305-refinement-results.md) for the current policy bank.

Arena-RL 0.5.0 controls only **red-3, model 305**. It retains the existing Rust simulator and robot specifications. The current six-policy bank is frozen during matches; DeepSeek V4.1 Flash (`deepseek-flash`, thinking disabled) selects and switches objectives asynchronously.

## Held-out skill tests

| Execution | Seeds | Result | Scope |
| --- | --- | --- | --- |
| Learned mobility | 7,000,200–7,000,263 | **64/64** | 27 explicit recovery + 37 retreat goals; includes real damage/10% respawn, contact escape, and physical recovery followed by retreat |
| Idle mobility baseline | Same seeds | 27/64 | 27 recovery, **0/37 retreat** |
| Random mobility baseline | Same seeds | 31/64 | 27 recovery, **4/37 retreat** |
| Deployed navigation bank | 3,000,000–3,000,031 | **31/32** | Navigation plus recovery/escape dispatch; seed 3,000,013 times out at the original 28-second budget, 0.264 m from goal without the required low speed |
| Combat | 3,000,000–3,000,031 | 32/32 | Local assisted shooting, pursuit and low-damage structure objectives |
| Defense | 3,000,000–3,000,031 | 32/32 | Local scripted invasion course |
| Support | 5,000,000–5,000,035 | 36/36 | Resupply, escort and patrol courses |
| Dedicated flight | 6,000,000–6,000,031 | **29/32** | 2026 lane staging/return; three forward failures remain |

The mobility test contains eight bounded template variants with counts **18, 9, 17, 9, 1, 1, 3, 6**. This is seed-held-out testing within these templates, not arbitrary-map generalization. Existing mechanical firmware rights the serial chassis in idle/random recovery baselines. Learned control is useful for withdrawal and continuation; these results do not establish learned raw joint torque or universal recovery. Shooting uses explicit SDK ballistic assistance and a low damage threshold.

Eight [physical native replay regressions](../checkpoints/serial305/physical-regressions/report.json) are deterministically verified. Genuine upset samples reach negative upright values (approximately −0.99), and injury scenarios use ordinary projectiles and normal respawn. The exact supply-wall contact, same-ID recovery/retreat composition, and the 0.4–0.5 m stall-detection gap have automated regressions. No quaternion, HP or position is patched while executing a skill.

## Final real API matches

| Seed | Opponent profile | Winner | Simulated duration / verified ticks | API requests | Latency median / p95 (s) | Errors | Mechanical starts / completions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9000020 | siege | red | 420 s / 8400 | 92 | 1.570 / 2.192 | {} | 0 / 0 |
| 9000021 | raid | red | 420 s / 8400 | 95 | 1.549 / 1.962 | {'provider_network_timeout': 2} | 0 / 0 |
| 9000022 | hunt | blue | 420 s / 8400 | 86 | 1.603 / 2.057 | {'provider_network_timeout': 2} | 1 / 1 |

These three matches give **2 red wins and 1 blue win**, with 273 real API requests. All finish through the 420-second referee and all native replays verify 8,400 ticks. Command/process continuity passes. Error counts: **{'provider_network_timeout': 4}**. Malformed, unknown-waypoint and out-of-field coordinate errors are zero. Transient provider errors preserve the current RL objective. `healthy_control` is an integration check, not a tactical win/quality metric.

The requested speed is 1×, with 10 Hz simulated control. Wall duration was 675.3 s and 1138 wall deadlines were missed during concurrent local validation. This demonstrates asynchronous operation, not a hard real-time latency guarantee. The single learned red infantry faces two declared scripted blue robots (heavy 2711 and W3 3), using normal spawn, starting ammunition, income, damage, purchases and respawns. It is not a standard full-roster match or a win-rate benchmark.

Physical event totals and one-second pose samples are reported separately in [physical-summary.json](../evaluations/serial305-2026-10-18/physical-summary.json). `recoveries_completed` counts supervision intervals, which may start upright; it must not be interpreted as the number of real flips. Dead upset samples are separate from living robots able to use the hips.

## Evidence and earlier runs

[Final report](../evaluations/serial305-2026-10-18/final/report.json), [decisions](../evaluations/serial305-2026-10-18/final/decisions.jsonl.gz), [telemetry](../evaluations/serial305-2026-10-18/final/telemetry.jsonl.gz) and command sidecars retain source/weight hashes, requested/returned model IDs, usage, latency, goals, failures, native replay SHA256 and deterministic verification results. Large uncompressed full replays remain locally under `runs/serial305-full-matches-final/`; they are not included in Git. Decompress physical `.json.gz` samples before Arena import.

Two preliminary three-match series are retained under `evaluations/serial305-2026-10-18/preliminary-*`. Each gave one red win and two blue wins, zero coordinate errors and only transient API timeouts. They used earlier mobility weights and source snapshots; they are not counted as tests of the final bank. The earlier W3 report is historical and is not a controlled ablation of 305, since prompts, routing and weights also changed. A failed two-tick startup probe is excluded from full-match totals.

Arena-RL was private when this was recorded; it has been public since 2026-10-19. Arena-1's local prerequisites are unchanged by this migration and remain uncommitted/unpushed. GitHub's normal contract CI can run independently; simulator CI needs an explicitly available Arena source containing those local environment interfaces.
