# 双臂轮式机器人：数据选择与长程策略基线

调研日期：2026-11-14。目标是**训练网络策略，并在物理仿真中闭环执行移动操作**，不是把规划器或状态回放的视频当作策略结果。

## 1. 数据候选与结论

| 数据 | 已核实内容 | 与当前服务器的匹配 | 本轮用途 |
|---|---|---|---|
| MoMaGen 自带 R1 源演示 | `dishes_away` 3 条，原始长度 2006 / 2267 / 2788 步；默认处理好的源演示为 `demo_2`。包含导航、双臂分别取盘及叠放 | 当前固定 R1、21 维动作、OmniGibson 3.7.1 栈已部署；可生成匹配场景的状态、动作及观测 | **第一轮选用**，先扩增并验证状态策略链路 |
| BEHAVIOR 2026 Challenge Demos | 数据卡：100 个任务、20,000 episodes、210,916,774 frames，约 3 TB；LeRobot v3；元数据为 R1Pro、23 维动作、30 Hz | **不能直接套用到现有 R1 21 维控制器**。需要与数据匹配的独立 R1Pro 环境、关节/观测/场景版本核对 | 更完整长程任务的后续首选；不下载全量，按任务及 episode 取子集 |
| BEHAVIOR 2025 Challenge Demos | 数据卡：50 个任务、10,000 episodes、119,094,660 frames；R1Pro、23 维动作；LeRobot v2.1 | 同样不是 R1 控制器的直接替换数据；不能只删两维动作冒充兼容 | 备选成熟训练集 |
| BEHAVIOR Robot Suite (BRS) | 官方数据卡公开 5 个真实家庭任务：派对后清理、清洁厕所、倒垃圾、上架物品、整理衣物；提供 WB-VIMA 训练入口 | 真实 R1 系统数据很相关，但不能据此认为有匹配当前仿真场景的初始状态/资产；需要单独的控制和观察适配 | 后续真实机器人策略参考，不是本轮最快的仿真闭环路线 |

**最新公开数据不一定是最快可运行的数据。**第一轮选择 MoMaGen 是因为机器人、动作空间、场景和重置状态能对齐，不是认为其数据规模优于 BEHAVIOR 2026。

### 官方来源

- MoMaGen 仓库：<https://github.com/ChengshuLi/MoMaGen>
- MoMaGen 数据生成教程：<https://chengshuli.github.io/MoMaGen/tutorials/generating-data/>
- BEHAVIOR 2026 数据卡：<https://huggingface.co/datasets/behavior-1k/2026-challenge-demos>
- BEHAVIOR 2026 schema：<https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/blob/main/meta/info.json>
- BEHAVIOR 2025 数据卡：<https://huggingface.co/datasets/behavior-1k/2025-challenge-demos>
- BRS 数据卡：<https://huggingface.co/datasets/behavior-robot-suite/data>
- BRS 训练代码：<https://github.com/behavior-robot-suite/brs-algo>

2026-11-14 官方 Hub API 返回的候选仓库 revision（**仅调研记录，不表示本轮已下载或训练这些数据**）：

- `behavior-1k/2026-challenge-demos`: `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`
- `behavior-1k/2025-challenge-demos`: `33639692425296185f245aaa648530178c7b3d2b`

数据卡的 MIT 声明不能替代仿真资产、机器人资产或 NVIDIA 软件的单独使用条款。本轮只读取服务器已有的、用户已明确确认的许可配置，不自动接受新条款。

## 2. 本轮任务到底是什么

选用 **`r1_dishes_away_D0`**。根据固定版本的 BDDL 与场景实例，而不是英文任务名判断：

- `plate.n.04_1` = `plate_603`，起始在架子上。
- `plate.n.04_2` = `plate_602`，起始在台面上。
- `plate.n.04_3` = `plate_601`，起始在台面上。
- 目标：两个台面上的盘子依次叠放到架子上的盘子上，允许两种上下顺序。
- 源任务规格四阶段：右臂取 `plate_602` → 左臂取 `plate_601` → 搬运后左臂放到 `plate_603` 上 → 右臂放到 `plate_601` 上。
- 中间包含底盘导航和持物移动；双臂分别持盘，不是双手同步抓同一物体的协调控制。

这是**厨房内四阶段移动搬运叠盘**，不是跨房间洗碗、送水或全屋清理。源演示最长 2788 控制步，按 30 Hz 约 93 秒；生成后长度会变，须按实际结果报告。

另一个 `bringing_water` 的 BDDL 仅要求瓶子不再位于冰箱内，不要求送到人或桌面；因此不把它包装成完整送水长程任务。

固定源码：见 `/home/yuanyue/double-arm/config/upstream.env`。服务器上任务证据位于：

```text
/mnt/yuanyue/double-arm/src/MoMaGen/momagen/datasets/base_configs/r1_dishes_away.json
/mnt/yuanyue/double-arm/src/MoMaGen/momagen/scene_instances/house_single_floor/house_single_floor_task_datagen_dishes_away_0_0_template.json
/mnt/yuanyue/double-arm/src/MoMaGen/BEHAVIOR-1K/bddl/bddl/activity_definitions/datagen_dishes_away/problem0.bddl
```

## 3. 数据工程与已发现的问题

### 源数据不等于现成可训练观测

原始源 HDF5 用单数 `action` / `state`，没有可直接训练的 `obs`。本轮由 MoMaGen 生成与当前仿真匹配的 `actions` / `states` / `obs`，不假定原始数据能直接交给通用 robomimic 配置。

### 三个盘子的观测别名覆盖

上游 `process_obj_robot_frame()` 把 BDDL 实例名按 `.` 截断，三个 `plate.n.04_i` 都写入 `object::plate`，导致只保留最后一个盘子。对叠盘策略而言缺失关键状态。

本轮在轻量入口中补充独立的机器人坐标系位姿：

```text
task::plate.n.04_1
task::plate.n.04_2
task::plate.n.04_3
```

其他任务物体也保留实例后缀。生成和评估使用同一函数，避免训练/测试 schema 不一致。不改动上游规划、可见性约束或 BDDL 成功判据。

### 动作和本体状态的排列不同

不能用 `prop_state` 的相同列直接作为全部动作的关节残差基准：动作顺序为 base3 / trunk4 / left6 / left-grip1 / right6 / right-grip1，而本体状态把右臂放在左臂后、两个夹爪宽度放在最后。本轮显式映射右臂 13:19 → 动作 14:20，并用数值单测验证。

生成环境的底盘控制器虽然叫 position mode，但输入是当前机器人局部坐标系的一步位移 / 转角；原始人类源演示则是 velocity mode，二者不能混用。模型输出直接对应生成环境动作，不沿用源数据速度的假设。

### 状态数据，非视觉数据

为了快速验证策略链路，本轮只保存低维状态观测到训练 HDF5。分割图仍在生成时提供给上游可见性检查，但写盘时去掉；RGB/depth 不保存为训练输入。已有示范视频只是检查用视频，不是视觉策略训练集。

## 4. 学习策略与公平评估约定

基线：**state-history action-chunk behavior cloning**，自定义小型 MLP，非官方 MoMaGen 预训练模型，非强化学习。

- 输入：当前及历史本体状态、底盘位姿、抓取状态、每个任务物体的机器人坐标系位姿。
- **使用仿真特权物体状态，不声称是 RGB 端到端策略。**
- 默认历史 4 帧、间隔 4 控制步；预测未来 16 步动作。
- 输出所有 21 维：底盘、躯干、左右机械臂和夹爪。
- 关节目标以当前本体关节状态为基准学习残差；底盘学习局部坐标系下的单步位移和转角（生成环境的 `motor_type=position`），夹爪用二分类。
- 每一步重新观测、推理，用短动作块的时间集成平滑动作。
- 不输入时间、示范阶段编号、未来状态、源示范动作或手工任务状态机。
- 默认 15,000 次更新，固定种子，训练集统计量归一化，checkpoint 由整条留出演示的验证损失选择。

训练 / 验证按**完整生成 episode**划分，不随机打散相邻帧来虚构泛化。至少 3 条成功生成的 episode 才启动；最后一条留出。生成数据可能都来自同一条人类源演示，**这不等于新场景、新布局或新人类轨迹泛化**。

闭环评估要求：

1. `init_curobo=False`，不启动示范规划器。
2. 只在每个 episode 开头恢复一次记录的初始状态，以定义可复现 reset；初始目标若已满足则拒绝评估。
3. 后续每一步都由网络动作驱动物理仿真，禁止逐帧 `load_state`。
4. 不读取教师动作控制机器人，也不在失败时切回规划器完成剩余阶段。
5. 输出实际动作 / 状态 trace、底盘路径长度、抓取进展、最终 BDDL goal status、闭环视频。
6. 同时记录最终成功与持续 30 帧成功；低验证损失、移动了几米、抓到了一个盘子，都不能替代完整任务成功。

## 5. 本次运行与复现

本轮数据生成根目录：

```text
/mnt/yuanyue/double-arm/runs/dishes_away-data-20261114-190245-Um8Y0d
```

初始设置：D0、seed 10、**8 次有限尝试**，不是保证成功 8 条；超时 3 小时。成功示范不足 3 条时流水线停止，不在失败轨迹上静默训练。

服务器运行入口（不要在本地执行）：

```bash
# 生成数据，成功率与控制步数以 important_stats.json 为准
DATA_ATTEMPTS=8 CUDA_VISIBLE_DEVICES=0 \
  bash /mnt/yuanyue/double-arm/code/scripts/run_long_task_data.sh

# 指定已经存在的本次数据生成目录；等待生成退出后，训练→闭环评估
bash /mnt/yuanyue/double-arm/code/scripts/run_state_policy_pipeline.sh \
  /mnt/yuanyue/double-arm/runs/dishes_away-data-20261114-190245-Um8Y0d
```

流水线输出：

- `generation.log`、`process-result.json`：示范生成日志与进程退出状态。
- `generated/demo_src_r1_dishes_away_task_D0/important_stats.json`：实际生成成功率。
- `policy-pipeline-status.json`：区分等待、训练、评估和失败。
- `state-bc-*/manifest.json`：数据 SHA-256、训练/验证 episode、特征、参数。
- `state-bc-*/metrics.jsonl`、`best.pt`、`last.pt`：训练曲线与权重，**仅在服务器**。
- `state-bc-*/evaluation/result.json`：闭环结果；`demo_*.mp4` 为网络闭环视频。

当前脚本不会创建通知或周期监控；后台 job 仅执行本轮有限实验。运行失败时查看对应日志，不用此前 `pick_cup` 的规划示范替代本次策略结果。

## 6. 验收和下一步

分别验收：已安装依赖 → 仿真启动 → 成功示范数量 → 完成参数训练 → 网络闭环完成四阶段任务。不能跨级报喜。

若本轮状态 BC 能完整完成，下一步扩展成功数据量、增加独立随机种子和布局留出，再尝试视觉策略。若闭环失败，应按导航 / 抓取 / 双臂持物搬运 / 叠放拆解失败点，增加对应的数据覆盖，而不是直接宣称长程任务已经解决。

真正的跨房间、多分钟家庭任务优先从 BEHAVIOR 2026 挑一个任务的子集，在**独立且版本匹配的 R1Pro 环境**验证；保留现有 MoMaGen 固定栈，不混装资产与仿真版本。

## 7. 后续核查更新（2026-11-14 晚）

上述首轮 Dishes Away 实验已结束：8 次尝试、0 条成功，约 2.64 小时，策略训练未启动。不能再把本轮状态描述为“仍在等待生成”。

官方生成流程、成功条数计数语义、原生对照和诊断顺序另见
[MoMaGen 官方生成办法核查](momagen-official-generation-method-20261114.md)。
已另行启动 Tidy Table D0 官方入口对照，目标成功 1 条，时限 45 分钟；它用于诊断生成链路，不替代最终双臂四阶段任务。
