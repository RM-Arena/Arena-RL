[English](README.md) | **中文**

# 真实 API 评估证据

参见[结果解读](../../docs/zh/llm-match-results.md)和[运行器文档](../../docs/zh/llm-agent.md)。

报告和日志来自三场由真实 `deepseek-flash` 监督的 420 秒对局，种子为 9,000,000–9,000,002。所有原生回放均验证了 8,400 tick 以及记录的结果。这是一台红方 W3 对阵两台脚本控制蓝方机器人的场景。最终结果：红方胜一场，蓝方胜两场。

`report.json` 中的 `replay` 文件名需在原始本地输出目录 `runs/deepseek-v4.1-flash-full/` 中解析，而不是在本证据目录中。完整的原生 JSON 每场约 190 MiB，仍被 Git 忽略并仅保存在本地。`sha256.json` 记录了它们的 SHA256 和位置。完整的决策日志和逐秒合法遥测日志包含在此处；凭据和模型内部推理记录不包含在内。

包版本在评估之后有所提升；`execution-source-at-start.json` 保留了被评估源码的指纹。`healthy_control` 表示集成/连续性检查通过，并不代表战术实力。请分别查看错误、失败技能和可用性。记录本文时本仓库为私有；自 2026-10-19 起已公开。
