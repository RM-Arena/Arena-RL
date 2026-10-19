**English** | [中文](zh/first-training.md)

# W3 first training run

All policies below were actually optimized with Stable-Baselines3 PPO on the shared Arena Rust simulator. Each skill used four persistent parallel actors. Move/ramp/defend used 32,768 initial transitions; outpost used 16,384. Move/ramp/outpost then continued for 512 transitions after a host JSONL reset-parser change; the metadata records this continuation and the allowed transport change. Rules and the embedded physics engine were identical throughout.

Low-level evaluation uses seeds 1,000,000–1,000,031, outside the training seeds. Random and learned policies run the same scenario seeds. Results are descriptive for these 32 simple curriculum episodes, not a statistical claim about full matches.

| Skill | Random success | Learned success | Random mean return | Learned mean return |
| --- | ---: | ---: | ---: | ---: |
| Move to point | 11/32 | 32/32 | 2.03 | 11.62 |
| Cross 15° ramp | 0/32 | 32/32 | -3.28 | 15.48 |
| Attack outpost (40 damage) | 31/32 | 32/32 | 11.94 | 13.53 |
| Intercept intruder (40 damage) | 32/32 | 32/32 | 12.71 | 14.13 |

The combat baselines are already strong because ballistic aiming is assisted and the 40-damage target is easy. These results establish a working training loop; they do not demonstrate superior combat skill. Moving/overcoming the ramp show a clearer learned improvement. Harder aim, ammo, distance, damage, moving-target and survival curricula should be evaluated next.

The manager trained for 4,096 one-second option transitions with frozen outpost/defend networks. Seeds 2,000,000–2,000,031 contain 20 local assaults and 12 local interceptions: 20/20 assault successes, 11/12 interception successes, 31/32 total. Chosen option accuracy was 100%; the remaining failure came from execution rather than option selection. It does not yet test long-distance navigation across the whole map or coordinated team tactics.

`checkpoints/w3/` includes the five networks, metadata, complete evaluation rows, learning logs, random baselines, one native replay per policy, and SHA256 checksums. Local `runs/w3/` retains all 160 generated evaluation replays. Initial conditions and actions are recorded in each replay. Full optimizer state is in each SB3 checkpoint.

Example:

```bash
uv run arena-rl evaluate --checkpoint checkpoints/w3/move --episodes 32
uv run arena-rl evaluate-high --checkpoint checkpoints/w3/manager --episodes 32
```

These reference checkpoints were produced from a development Arena checkout, recorded as dirty in the metadata. The exact source hashes are provided; the matching Arena source changes must be available before a fresh upstream-only checkout can use them. Different operating-system binaries with the same embedded engine/rules and source hashes are accepted. A stale simulator binary is rejected even if the rules file is unchanged.
