**English** | [中文](README.zh.md)

# Full-team 2027 matches, 2026-10-19

Evidence for the [team LLM + universal RL](../../docs/multi-robot.md) results. Every match is a full 420 s 2027 match with the default lineup on both sides, normal economy, respawn and referee. Blue is always the declared doctrine commander with scripted expert controllers.

| Directory | Red commander | Red low level | Seeds | Red wins |
| --- | --- | --- | --- | --- |
| `doctrine-expert/` | doctrine | expert controllers (mirror baseline) | 9300000–9300002 | 2 / 3 |
| `doctrine-rl/` | doctrine | universal RL policies | 9300000–9300002 | 2 / 3 |
| `llm-rl/` | DeepSeek (`deepseek-flash`, function calling) | universal RL policies | 9300000–9300001 | 1 / 2 |

- `*/report.json`: per-match winner, score, structures, harness actions, errors, token usage, latency, events, replay SHA256 and Arena replay verification (8,400 ticks each).
- `llm-rl/decisions-<seed>.jsonl`: every LLM request scene, model reply, applied result, rejection reason, latency and token usage. The API key is read from the environment and never logged.
- `traces/*.json.gz`: `arena-match-trace-v1` traces downsampled from the native replays by `examples/extract_match_traces.py` (every 10 frames = 0.5 s; robot position/HP/alive, drone positions, outpost/base HP, control degree). They drive the figures in `docs/images`.

The full native replays (about 415 MB each) stay local under `runs/team/` and are not committed; their hashes are in the reports. Two to three matches per configuration only show that the pipeline works in real full matches; they are not a win-rate estimate.

```bash
uv run python examples/extract_match_traces.py runs/team/llm-rl/match-9300001.json evaluations/team-matches-2026-10-19/traces/llm-rl-9300001.json.gz
ARENA_ROOT=../RMUC-Arena uv run --with matplotlib python examples/make_figures.py
```
