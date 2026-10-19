[English](../llm-tactics.md) | **中文**

# LLM 战术指令与学习式执行

LLM 为单台机器人选择目标。当前的 `checkpoints/serial305` 策略库包含六个 PPO 策略族，以 10 Hz 执行十一种参数化指令，控制**红方 3 号位上的 305 型**串联腿机器人。较早的 `checkpoints/tactics` 策略库保留了 W3（3 型）及其九种指令的基准测试。该服务与 LLM 提供商无关；`serve-tactics` 本身不调用任何 LLM API。独立的 `llm-matches` 宿主程序负责接入真实的规划器。早期的二选一 PPO 管理器仍作为基线保留。当前模型、观测量与评估局限见 [305 训练与恢复](serial305.md)。

## 指令与支持的地图

| 技能 | 参数 | 完成条件 / 执行方式 |
| --- | --- | --- |
| `navigate` | `point: [x,y]` | 以低速到达 `radius_m` 范围内；公开路径规划加学习式局部控制 |
| `attack_structure` | `target_id: blue_outpost` / `blue_base` | 攻击选定的建筑，遵守基地保护规则 |
| `engage_target` | 敌方机器人 ID | 接近该身份的目标，并在 `preferred_range_m` 距离上与之交战 |
| `pursue_target` | 敌方机器人 ID | 跟随短时预测拦截点，保持目标身份不变并遵守牵引半径（leash）约束 |
| `defend_region` | `center`，可选的有序 `priority_ids` | 在指定时长内于区域内驻守/拦截，或达到 `damage_goal` |
| `resupply` | `ammo_target`、`hp_target_fraction`、`budget_limit`，可选 `point` | 到达己方补给区，购买本地弹药，并等待真实的被动回血 |
| `escort` | 友方 `target_id`，可选目的地 `point` | 跟随移动中的友方；满足近距覆盖率要求以及目的地/时长条件 |
| `patrol` | 有序的 `waypoints`、`patrol_loops`、`dwell_s` | 按顺序访问每个航点，以低速驻留，并完成每一圈 |
| `fly_slope` | `lane_team`、`flight_direction`、`landing_hold_s` | 越过实际的 650 mm 间隙，记录腾空时间，直立着陆并稳定 |
| `retreat` | 可选 `point`、`radius_m`、`timeout_s` | 即使低血量或零弹药也会撤退；默认目的地为己方补给区 |
| `recover` | `timeout_s` | 对 305：等待物理自扶正并恢复稳定站姿；可挂起并随后恢复另一个目标 |

前八种指令在 2027 上训练/评估。专用的飞坡几何结构仅存在于 **2026**，因此该技能需使用 `--season 2026`。2027 的道路坡道属于地面导航地形，并非专用的腾空跨越结构。赛季选择是回合级属性：指令不能切换地图。`navigate` 也可在 2026 上用于就位（staging）；其全图独立种子基准在 2027 上进行。`retreat` 与 `recover` 在 2027 上训练；`recover` 需要 305 型。空中机器人控制、任意地形跳跃以及全局胜率结论均不在已记录的覆盖范围内。

`ready` 与 `capabilities` 中包含 `available_skills`，按已加载的权重和当前赛季过滤。协议级的 `skills`/`command_json_schema` 描述全部十一种指令。不可用的指令会在替换当前目标之前被拒绝。

![RMUC 2027 场地：基地、前哨站、堡垒卡、中央高地卡、补给区和公路起伏路段](../images/zh/field-2027.png)

## JSON 示例

```jsonl
{"id":"supply-1","skill":"resupply","ammo_target":100,"hp_target_fraction":0.9,"budget_limit":150,"timeout_s":60}
{"id":"protect-1","skill":"escort","target_id":"red-4","point":[7.5,5.35],"follow_distance_m":1.2,"required_coverage":0.6,"timeout_s":30}
{"id":"route-1","skill":"patrol","waypoints":[[6.8,5.15],[6.8,5.7],[4.8,5.5]],"patrol_loops":2,"dwell_s":0.4,"timeout_s":60}
{"id":"fly-1","skill":"fly_slope","lane_team":"red","flight_direction":"forward","timeout_s":18}
{"id":"return-1","skill":"fly_slope","lane_team":"red","flight_direction":"reverse","timeout_s":18}
```

以上为指令对象；传输时需将每个对象包装进 `{"op":"command","command": ...}`。只有当该执行器在**同一物理回合**内完成了同一赛道的正向飞越之后，才会接受反向飞行。重置会清除此许可。反向课程会先真实地完成一次正向飞越；绝不会注入裁判系统的通过判定。飞行评估从赛道的就位点出发，覆盖两条镜像赛道和两个方向，且没有敌方火力。任意远距离接近以及受扰着陆需要额外的课程。

补给技能仅在机器人真正位于己方补给卡上时才发送普通的 `buy_ammo` 动作。批量大小、价格、队伍购买上限、资源扣除与回血均由模拟器执行。执行过程中不会直接修改血量、弹药或货币。`budget_limit` 保守地统计该目标运行期间己方队伍本地 17 mm 购买账目的增量；队友同期的购买也会消耗这一额度。资金/预算不足、达到队伍上限、超时或机器人损毁都会产生明确的失败结果。受伤的机器人可以无视通常的低血量停止条件前往补给。该技能目前覆盖本地弹药购买与补给区被动回血，不包括远程购买/付费远程回血。2027 常规开局弹药仍为零：作战前需先下达 `resupply`。

护送使用合法的友方遥测数据，且绝不接管被保护友方的控制。指定 `point` 时，友方必须到达该目的地；未指定时，使用 `escort_duration_s`。`required_coverage` 指目标已用时间中，与友方的距离保持在 0.45 m 到 `follow_distance_m + 1.1 m` 之间的时间占比。距离过远（`leash_radius_m`）或友方被摧毁会导致指令失败。所提供的护送课程使用一个缓慢移动的脚本控制友方。它并不能证明在敌方火力下的保护能力。

巡逻至少需要两个点。机器人离开航点或移动过快时，驻留计时会重置。零弹药不会妨碍巡逻/护送。两者都接受 `engage_on_contact: true` 和有序的敌方 `priority_ids`，以便在沿路线/保持队形行进时可选地开火；只有 3 号位自身可见的敌人才能成为目标。当前的独立种子保障课程中没有此类作战干扰，因此该选项需要单独的作战评估。

## 指令监管

在 1,024 条指令的缓存范围内，ID 使完全相同的重试具有幂等性。以相同 ID 提交内容已变化的指令会被拒绝。有效的替换会保持模拟器进程、tick、位姿、血量与弹药不变。无效/过期/来自未来/乱序的指令不会影响正在运行的目标。可选的 `basis_tick` 加 `max_staleness_s` 用于限制延迟到达的 LLM 回复。`cancel` 用于停止执行；它不是学习得到的技能。

针对机器人的攻击保留明确的目标身份。只有 3 号位合法第一人称视角（FPV）中的敌人才会进入目标记忆；被遮挡的目标会按最后观测到的速度在有限时间内外推，期间不开火。攻击友方、阵容中不存在的 ID 以及受保护的建筑都会被拒绝/停止。攻击完成以目标被摧毁或达到可选的正值 `damage_goal` 为准。血量损失是观测到的进展；当队友也在攻击时，并不保证可归因于本射手。

反馈包括状态（`idle/running/succeeded/failed/cancelled`）、原因、阶段、已用模拟时间、距离、目标身份/可见性以及 `execution_family`。`details` 进一步提供补给区接触/弹药/血量/花费、护送距离/覆盖率、巡逻航点进度，或飞行的腾空/着陆时间。失败原因包括受阻、超时、丢失目标、超出牵引半径、机器人不可用、低血量、无弹药、预算耗尽以及达到队伍购买上限。旧版 3 型路线在六秒内没有明显移动/伤害时报告受阻；305 型路线则允许 25 秒用于物理脱困。保障/飞行运动使用单独的监管逻辑。终止反馈会保留完成耗时。

提供给 LLM 的场景包括本机与友方遥测、己方资源/购买计数器、己方增益卡接触情况、可见敌人、带时间标记的最后目击敌人、公开的建筑以及控制点。公开的地图/阵容随 `ready` 下发。不会把对手自身的实时观测发送给 LLM 或学习式控制器。课程宿主控制器可以使用其自身队伍的观测来移动/射击其脚本控制的角色。

## 服务与宿主集成

```bash
uv run arena-rl serve-tactics --policies checkpoints/serial305 --realtime --record
uv run arena-rl serve-tactics --policies checkpoints/serial305 --season 2026 --realtime --record
```

默认的实时控制每 100 ms 墙钟时间推进 100 ms 模拟时间；`--speed` 可改变该比例。外部 LLM 推理进行期间，强化学习策略会继续执行当前目标。周期性状态消息大约每墙钟秒到达一次。不使用 `--realtime` 时，需手动发送 `{"op":"step","steps":10}` 来推进模拟。

其他操作包括 `observe`、`cancel`、`capabilities`、`reset`、`save_replay` 和 `close`。`env_id` 用于选择某一个并行 actor。`reset` 接受每个 actor 各自的种子以及可选的已声明初始训练场景；只有 reset 才会改变回合。EOF/close/终止时会清理模拟器进程。格式错误或过大（>64 KiB）的消息会返回带帧格式的错误。`examples/tactical_client.py` 演示了一个缓慢的脚本规划器，不涉及付费推理。

`--lineups path.json` 接受 Arena 阵容格式，要求红方 3 号位与已加载策略库的型号一致。默认的 2027 阵容包含用于护送的红方 4 号位。除非对局宿主提供控制器，否则队友在该服务中保持静止；本库绝不会暗中驱动护送目标。`--opponent` 通过 Arena 的沙箱选择一个现有的整队 Arena 提交。不指定时，场上敌人保持静止，`ready` 会声明此模式。

在现有的对局宿主中：

```python
from arena_rl.executor import TacticalExecutor
from arena_rl.tactical_envs import encode_observation, decode_action

executor = TacticalExecutor(slot="3")
executor.submit(command_from_llm, legal_red_observation)
# Every 100 ms simulated time:
executor.update(legal_red_observation)
family = executor.execution_family
features = encode_observation(executor, legal_red_observation, family)
action, _ = policies[family].predict(features, deterministic=True)
robot_actions = decode_action(executor, legal_red_observation, action, family)
feedback_to_llm = executor.feedback()
```

只将该槽位返回的动作与队友控制器合并。长距离地面接近可以先使用导航策略，再切换到局部的保障/作战/防守策略，并保留指令 ID；导航模式下禁止开火。宿主负责观测下发、推理调度和回合生命周期。

## 训练契约与可复现性

```bash
uv run arena-rl train-tactics --family support --steps 32768 --envs 4 --output runs/support
uv run arena-rl train-tactics --family flight --steps 16384 --envs 4 --output runs/flight
uv run arena-rl evaluate-tactics --checkpoint runs/flight --episodes 40 --seed 6000000 --replays runs/flight/replays
uv run python examples/validate_extensions.py --policies checkpoints/tactics --output runs/runtime
```

旧的导航/作战/防守策略沿用 `w3-tactics-v3`（52 维观测）。新的保障与飞行策略使用独立的 `w3-support-v1` / `w3-flight-v1` 契约（74 维观测）。全部使用 `MultiDiscrete([5,4,3,3,2,3,2])`。新权重不能在无提示的情况下替代旧的策略族。加载时会校验检查点摘要以及模拟器源码/内嵌规则引擎的来源指纹。

保障策略从随机权重出发，使用普通 PPO 训练。飞行策略默认尝试 64 个脚本示范回合，只保留成功的轨迹，通过行为克隆拟合策略，然后在新的回合上运行 PPO。其元数据记录了被接受/被拒绝的种子、数据集 SHA256、模仿拟合结果、PPO 步数/更新次数以及发生变化的权重哈希。`--expert-episodes 0` 会禁用该初始化；续训的策略不会再次初始化。示范是训练数据，而不是运行时的后备方案。部署时的飞行动作来自训练好的网络，加上文档中说明的几何航向/间隙/着陆契约。从零开始的飞行 PPO 在最初 32,768 步的试验中未能成功。

公开 SDK 的路线提示和弹道瞄准属于明确声明的辅助。W3 导航现在会保持标称的腿部离地间隙，并在转向前沿当前轮轴方向越过凸起；物理/地图几何均未改动。网络负责局部的运动/速度/步态/开火/跳跃决策，而非电机电流、关节力矩或原始图像感知。

原生 v4 回放保留了普通动作，包括制造受伤/正向飞越的准备过程。`.commands.json` 附属文件记录运行时的指令 ID 和策略摘要。验证脚本检查指令的 tick/弹药/进程连续性，并通过 Arena 的校验器确定性地重放具有代表性的保障/飞行回合。确切的种子、基线和局限见 [扩展结果](extended-results.md)，最初的五技能基准测试见 [早期战术结果](tactical-results.md)。


## 基于 API 的顶层规划器

`arena-rl llm-matches` 现已将真实的 DeepSeek V4.1 Flash 规划器接入此接口，使用异步函数调用，同时冻结的策略继续以 10 Hz 运行。参见 [规划器配置](llm-agent.md) 与 [完整对局结果](llm-match-results.md)。通用的 `serve-tactics` 接口和脚本示例仍然可用；它们本身不会调用付费推理。
