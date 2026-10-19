**English** | [中文](zh/tactical-results.md)

# Tactical execution validation

The reference policies use schema `w3-tactics-v3` on Arena's existing Rust simulator. Navigation trained for 32,768 initial PPO transitions and 65,536 continuation transitions (98,304 total), combat for 65,536 and defense for 32,768. All use four persistent simulation actors; the weights and optimizer updates are recorded in metadata.

Independent seeds 3,000,000–3,000,031 were evaluated with deterministic learned actions. The same seeds were used for random-action baselines. Target selection, public-map route hints and ballistic aiming assistance apply to both baselines and learned policies.

| Policy / command | Completed | Random baseline |
| --- | ---: | ---: |
| Navigation (including road bumps and full-field routes) | 30/32 | 0/32 |
| Attack specified structure | 10/10 | 10/10 |
| Engage specified robot | 12/12 | 11/12 |
| Pursue specified robot | 10/10 | 10/10 |
| Regional defense | 32/32 | 31/32 |

Combat success means 60 HP loss on the specified target, not destruction or victory in a full match. The combat courses remain easy with ballistic assistance. These numbers validate command coverage and integration, not superior combat strength. The navigation failures at seeds 3,000,008 and 3,000,027 are retained in the evaluation report and local replays; both report `blocked` during reverse full-field travel.

## Continuous missions

Seeds 4,000,000, 4,000,001 and 4,000,002 use the same declared fixture with different simulator randomness. One W3 robot drives across the field, attacks the enemy outpost, returns to its base approach, defends the region with priority on `blue-1`, then pursues that same heavy robot. All five goals complete in all three runs. The command issuer is a deterministic test script, not an LLM.

The blue heavy starts near the home-side approach as a pre-existing infiltrator; it begins moving when the return-home command is issued, and moves away during the pursuit command. A closer infantry distractor is also present. No robot is teleported or reset between commands. Every stage records before/after tick and ammo and confirms the same simulator process. Native replays record every actual action.

Reference checkpoints, evaluation rows, source digests, random baselines, three mission reports and selected short native replays are versioned under `checkpoints/tactics/`. Full native mission/evaluation replays remain locally under `runs/tactics-v0.2/validation/` to keep the Git repository small. They can be regenerated with `examples/validate_tactics.py`. The Arena checkout must match the recorded engine/source fingerprints.

Remaining limits: only W3/red-slot-3, 2027, the recorded enemy models, assisted aiming, a limited route curriculum, and damage-based combat courses. Ammunition acquisition, dedicated airborne fly-slope control, escort and patrol remain unsupported. No real LLM API, full-match strategic benchmark, or calibrated skill success probability is included. The JSONL/runtime integration does verify that low-level control continues during a planner delay and that commands can switch without episode reset.

A separate single-command test issued only `attack_structure(blue_outpost)` from `[4.5, 5.15]`. It automatically used both navigation and combat policies, completed after 22.7 simulated seconds, and recorded 80 HP loss on the specified outpost. The corresponding full native replay is local at `runs/tactics-v0.2/validation/direct-attack.json`.
