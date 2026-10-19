# Serial-leg 305 reference policies

The current 0.5.1 six-family bank controls only red slot 3, model 305. Policies were trained with successful demonstration initialization followed by real PPO updates. The active mobility policy is the composed fine-tune; its original initialization policy remains in `lineage/mobility-initial`. Navigation was initialized from that original policy, then separately trained. Its 0.5.0 network is retained under `navigation/transit` for goals beyond 4 m; the new local network received another 16,384 PPO steps. Flight was retrained with 84 lane-relative inputs, explicit phase and gap observations, plus 8,192 PPO steps. Archived 74-input 305 flight weights remain in `lineage/flight-v050`.

See [training and command contracts](../../docs/serial305.md) and [current evaluation results](../../docs/serial305-refinement-results.md). `metadata.json` records simulator provenance, schema, model, seeds, PPO updates, demonstration hashes and policy digests. `manifest.json` summarizes the bank and hashes the shipped artifacts.

`mobility/evaluation.json`, `idle.json` and `random.json` use identical held-out seeds. `navigation/evaluation-composed.json` evaluates the deployed bank including recovery and local escape; `evaluation-single-policy.json` records the earlier isolated navigation-policy test. These are different execution conditions. Flight is separately evaluated on the 2026 map and is unavailable as a dedicated skill in 2027 matches.

`physical-regressions` contains eight deterministically verified native replay samples and a report. Decompress `.json.gz` before importing into Arena or using `rmuc-arena replay verify`. They preserve ordinary physical preparation, including actual damage, weak respawn, jump/lean-induced upset and serial-hip self-righting; no HP or pose is patched during a skill.

Mechanical righting belongs to the simulator's existing 305 firmware. The policy bank does not learn raw joint torque, perception from images, or an unrestricted full-map driving policy. See the explicit held-out failures and scripted-opponent match scope before interpreting results.
