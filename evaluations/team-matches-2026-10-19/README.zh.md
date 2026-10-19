[English](README.md) | **中文**

# 2027 整队对局记录（2026-10-19）

本目录是[整队 LLM + 通用 RL](../../docs/zh/multi-robot.md) 结果的证据。每局都是完整 420 s 的 2027 对局，双方使用默认阵容，正常经济、复活和裁判。蓝方始终是声明的规则队伍（doctrine 指挥 + 脚本专家控制器）。

| 目录 | 红方指挥 | 红方低层 | 种子 | 红方胜 |
| --- | --- | --- | --- | --- |
| `doctrine-expert/` | doctrine | 专家控制器（镜像基线） | 9300000–9300002 | 2 / 3 |
| `doctrine-rl/` | doctrine | 通用 RL 策略 | 9300000–9300002 | 2 / 3 |
| `llm-rl/` | DeepSeek（`deepseek-flash`，函数调用） | 通用 RL 策略 | 9300000–9300001 | 1 / 2 |

- `*/report.json`：每局胜方、比分、结构血量、Harness 动作统计、错误、token 用量、延迟、事件、回放 SHA256 和 Arena 回放校验（每局 8,400 tick）。
- `llm-rl/decisions-<种子>.jsonl`：每次 LLM 请求的场景、模型回复、应用结果、拒绝原因、延迟和 token 用量。API 密钥只从环境变量读取，不写入日志。
- `traces/*.json.gz`：由 `examples/extract_match_traces.py` 从原生回放降采样得到的 `arena-match-trace-v1` 轨迹（每 10 帧即 0.5 s；机器人位置/血量/存活、空中机器人位置、前哨站/基地血量、控制度），用于生成 `docs/images` 中的图表。

完整原生回放（每局约 415 MB）保存在本地 `runs/team/`，未纳入 Git，哈希记录在报告中。每组只有 2–3 局，只能说明链路在真实整局中可用，不能作为胜率估计。

```bash
uv run python examples/extract_match_traces.py runs/team/llm-rl/match-9300001.json evaluations/team-matches-2026-10-19/traces/llm-rl-9300001.json.gz
ARENA_ROOT=../RMUC-Arena uv run --with matplotlib python examples/make_figures.py
```
