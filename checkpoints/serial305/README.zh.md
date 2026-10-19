[English](README.md) | **中文**

# 串联腿 305 参考策略

当前 0.5.1 版本的六策略族库仅控制红方 3 号位、型号 305。各策略先用成功示范初始化，再进行真实的 PPO 更新。当前启用的机动策略是组合微调版本；其原始初始化策略保留在 `lineage/mobility-initial` 中。导航策略由该原始策略初始化，然后单独训练。其 0.5.0 网络保留在 `navigation/transit` 下，用于 4 m 以外的目标；新的局部网络又接受了 16,384 步 PPO 训练。飞坡策略使用 84 个赛道相对输入以及显式的阶段与间隙观测重新训练，并增加了 8,192 步 PPO。归档的 74 输入 305 飞坡权重保留在 `lineage/flight-v050` 中。

见[训练与命令约定](../../docs/zh/serial305.md)和[当前评估结果](../../docs/zh/serial305-refinement-results.md)。`metadata.json` 记录模拟器来源指纹、schema、型号、种子、PPO 更新、示范哈希和策略摘要。`manifest.json` 汇总策略库，并对发布的产物计算哈希。

`mobility/evaluation.json`、`idle.json` 和 `random.json` 使用相同的独立种子。`navigation/evaluation-composed.json` 评估包含恢复与局部脱困的部署策略库；`evaluation-single-policy.json` 记录的是早先单独的导航策略测试。二者的执行条件不同。飞坡在 2026 地图上单独评估，在 2027 对局中不能作为专用技能使用。

`physical-regressions` 包含八份经过确定性验证的原生回放样本和一份报告。导入 Arena 或使用 `rmuc-arena replay verify` 之前，请先解压 `.json.gz`。它们保留了普通的物理准备过程，包括真实伤害、虚弱复活、跳跃/侧倾引起的翻倒以及串联髋关节自扶正；技能执行期间不会修补血量或位姿。

机械扶正属于模拟器已有的 305 固件。该策略库不学习原始关节力矩、基于图像的感知，也不包含不受限制的全图驾驶策略。在解读结果之前，请先查看明确列出的独立种子失败案例以及脚本对手对局的适用范围。
