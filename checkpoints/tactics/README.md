# Tactical reference policies

Five policy directories cover nine single-W3 objectives. Read [command contracts](../../docs/llm-tactics.md) and [current results](../../docs/extended-results.md) before interpreting the success counts. These are assisted, local curricula, not a full-match tactical agent.

- `navigation`, `combat`, `defense`: original 52-feature policies, with current regression reports.
- `support`: 74-feature 2027 resupply/escort/patrol PPO policy.
- `flight`: 74-feature 2026 forward/reverse gap-crossing policy, initialized from demonstrations before PPO.
- `validation`: live dispatch/continuity results and verified navigation repairs.

Each policy includes weights, optimizer state, provenance, hashes and training/evaluation evidence. Representative native v4 replays include command sidecars where applicable. The two large navigation regression replays are gzip-compressed; decompress before import/verification. Original mission results remain historical records. The current simulator prerequisite is local to Arena-1 and has not been pushed; use the matching checkout to load these weights. This repository remains private.
