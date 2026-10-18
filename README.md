# Arena-RL

基于 [Arena-1](https://github.com/RM-Arena/Arena-1) 的单机器人强化学习训练库。复用 Arena 的 Rust 物理、规则、回放，以及 `ArenaEnv` / `VectorArena` 并行环境，不维护另一套模拟器。

第一版选择 **2027 赛季 W3 双轮足步兵，型号 3，槽位 3**。每个进程只训练一台红方机器人；蓝方机器人按场景脚本运行。训练使用真实 PPO 更新网络权重。

## 安装

需要 Python 3.10+、Rust/Cargo 和两个相邻的源码目录：

```text
workspace/
  RMUC-Arena/    # Arena-1 源码，需包含并行环境和训练场景接口
  Arena-RL/
```

```bash
cd ../RMUC-Arena
cargo build --release --manifest-path simulation/Cargo.toml
cd ../Arena-RL
uv sync --locked --extra dev
uv run arena-rl --arena-root ../RMUC-Arena doctor
```

也可以用 `ARENA_ROOT=/absolute/path/to/RMUC-Arena` 指定模拟器。`doctor` 会检查需要的接口并显示源码与二进制指纹。Arena 的完整前端和桌面程序不是训练依赖。

**当前开发依赖**：需要 Arena 中的 `rmuc_arena/env.py`、`vector_env.py` 和 `simulation/src/training_scenario.rs`。如果下载的 Arena 版本缺少这些文件，需要使用包含这批改动的源码版本。训练元数据会记录实际 Git 提交、工作区是否有改动、关键源文件 SHA256、二进制 SHA256 和每回合的规则/引擎哈希。无法满足接口时直接报错，不会退回另一个物理实现。

## 训练低层技能

```bash
uv run arena-rl train --task move --steps 32768 --envs 4 --output runs/w3/move
uv run arena-rl train --task ramp --steps 32768 --envs 4 --output runs/w3/ramp
uv run arena-rl train --task outpost --steps 32768 --envs 4 --output runs/w3/outpost
uv run arena-rl train --task defend --steps 32768 --envs 4 --output runs/w3/defend
```

低层决策频率 **10 Hz（模拟时间）**；计算速度可以快于实时。动作由移动方向、瞄准偏移、俯仰偏移、开火、腿长和跳跃组成。学习器只能控制自己的槽位。

| 技能 | 初始课程 | 成功条件 |
| --- | --- | --- |
| `move` | 开阔区内随机出生点和目标点 | 进入 0.35 m 目标范围，速度低于 0.8 m/s |
| `ramp` | 真实 2027 地图的红方 15° 公路坡道，出生点有横向扰动 | 从低端越过坡顶，升高超过 0.12 m，到达坡后平台且保持直立 |
| `outpost` | 从敌方前哨附近开始，200 发训练弹量 | 对真实前哨装甲累计造成至少 40 点伤害 |
| `defend` | 基地入口附近，一台脚本控制的步兵持续接近 | 对入侵者累计造成至少 40 点伤害 |

**瞄准辅助是显式的**：射击技能使用 Arena SDK 提供的弹道瞄准，PPO 学习移动、瞄准修正、开火时机和轮足动作。第一版没有从原始图像或原始云台指令开始学习完整感知与瞄准；也不训练电机电流或关节力矩。过坡任务不要求腾空，不能把成功率解读成飞跃特技成功率。

`outpost` / `defend` 是短距离、低伤害阈值的入门课程，目标是验证训练链路。它们不代表已能摧毁整座前哨、消灭完整入侵队伍或赢得完整比赛。奖励和课程定义集中在 `src/arena_rl/tasks.py`，可以逐步提高距离、伤害目标和干扰强度。

## 高层决策

```bash
uv run arena-rl train-high --skills runs/w3 --steps 4096 --envs 4 --output runs/w3/manager
uv run arena-rl evaluate-high --checkpoint runs/w3/manager --episodes 32 --output runs/w3/manager/evaluation.json
```

管理策略每 **1 秒模拟时间** 选择 `outpost`（进攻）或 `defend`（守家）。它在同一回合中调用冻结的低层模型，最多执行 10 次低层动作；任务成功、机器人失效或入侵越界可提前结束。低层模型的权重和摘要会在训练前后核对。

第一版高层课程包含“前哨附近进攻”和“基地入口拦截”两种局部场景，场景种类不会直接进入模型输入。高层通过本机状态和合法视觉观测作决策；奖励来自实际伤害、存活和入侵越界，没有直接奖励某个正确选项标签。敌人行为独立于所选技能。场景较简单，完整地图长距离调度和突然出现的新威胁属于后续课程。

## 评测、回放和继续训练

```bash
uv run arena-rl evaluate --checkpoint runs/w3/move --episodes 32 --seed 1000000 --output runs/w3/move/evaluation.json --replays runs/w3/move/replays
uv run arena-rl evaluate --task move --baseline random --episodes 32 --seed 1000000
uv run arena-rl train --task move --steps 32768 --envs 4 --resume runs/w3/move --output runs/w3/move-continued
uv run arena-rl suite --config configs/w3.json --output runs/full
```

评测使用独立种子，训练默认从 7 开始，低层评测默认从 1,000,000 开始，高层默认从 2,000,000 开始。`suite` 在相同评测种子上比较随机策略与训练策略；低层技能完全没有评测成功时停止高层训练。

每个训练目录包含 `policy.zip`（权重与优化器）、`metadata.json`（版本和训练证据）、`metrics.jsonl`（学习过程）。继续训练恢复网络和优化器，从新的回合开始，不承诺逐比特恢复中断瞬间的采样器状态。默认拒绝加载与模拟器源码及内嵌引擎/规则哈希不匹配的检查点（同源码的不同平台构建可兼容）；只有显式指定 `--allow-engine-change` 才允许低层继续训练或评测时更换模拟器。PyTorch 检查点仅加载自己生成或可信来源的文件。

回放采用 Arena 原生 v4 格式，保留训练出生点、弹量和完整动作，可直接用 Arena 回放页面导入：

```bash
cd ../RMUC-Arena
uv run rmuc-arena replay verify ../Arena-RL/runs/w3/move/replays/move-1000000.json
```

Gymnasium 接口可从 `arena_rl.envs.SingleRobotEnv` 使用；SB3 适配器 `SkillVecEnv` 使用 Arena 的持久并行进程，不再额外包一层 SB3 子进程。环境区分成功/失败结束与时间截断，保留终止观测，再为 SB3 自动重置。

## 测试

```bash
uv run pytest -q
```

覆盖 Gymnasium 合约、动作归属、并行进程隔离、固定种子复现、真实坡道和伤害、回放验证、网络权重更新、摘要校验和冻结技能的高层训练。

实现依据：[Gymnasium Env 接口](https://gymnasium.farama.org/api/env/)、[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)。项目遵循随仓库提供的非商业学习许可证；Arena 和训练依赖各自保留其许可。

## 已跑通的参考模型

见 [第一轮训练记录](docs/first-training.md) 和 [参考检查点](checkpoints/w3)。四项低层技能各 32 个独立场景均完成；高层为 31/32。记录同时公布随机基线：短距离、辅助瞄准的射击课程本来就很容易，暂不能据此宣称训练出了更强的实战射击能力。
