# 抓杯「瞬间贴爪」对照审计（2026-11-18）

## 已查明的机制

Uniform B `final19999` 的 TRAIN/ASSISTED 视频确实运行在 `grasping_mode=assisted`，结果记录 `assisted_constraint_seen=true`，`object_pose_write_count=0`。这不是无辅助物理夹持，也不是模型接口输出了物体位姿：模型输入仍为真实 RGB、21D proprio 和指令，控制输出为 21D 底盘/关节/夹爪动作。

对照服务器固定版本：

- `.../omnigibson/robots/manipulation_robot.py`（SHA256 `af1e0245078a69a67368be4e70af4f5b79190c7d7dcff4fe9be210da1c39b0ba`）：`deploy_control` 只在非 physical 模式执行辅助。`_calculate_in_hand_object_rigid` 对 assisted 检查射线与至少两个手指 link 的接触（不同于 sticky）；`_establish_grasp_rigid` 在实际接触点构造两端约束坐标，创建 `ag_constraint`，冻结夹爪。它没有在此处调用物体位姿 setter；装置靠约束持杯而不是仅靠摩擦。机器人显式传送时确实有另一条带动物体的 setter 路径，但无迹象显示它在 B 的已审计闭环里触发。
- `.../robomimic/envs/env_omnigibson.py`（SHA256 `f40b783bd482e868d1c2286e2bee77a4b012f848f9ea903e0d4b742e13c87d5f`）：初始化将 `GRASP_WINDOW=0.0`；辅助候选一旦在后续检查中继续合格，时间门槛不会再引入正值等待。并非允许隔空抓取。
- 五条随机 raw HDF5 的实际 env mode 同样都是 `assisted`。源码的 `obs/grasp_left` 在该模式表示左手辅助拿持状态（不带指定物体），五条都记录了首次 `TRUE`，所以「画面看起来顺」不等于「数据没有辅助」。五条都未保存关节创建子步日志，不能填造精确时刻。

## 同一观测窗口的实际对照

仅服务器 CPU 读取已完成的 HDF5 / trace / 视频，不重跑模拟、策略，不修改生成任务。原始 `obs/object::teacup` 为相对机器人位姿，使用同一观测 `obs/base_link_pose` 转成世界坐标后求杯位姿原点的相邻观测位移。原始生成存的是动作前观测，π₀/物理回放 trace 存的是动作后观测；各自对齐首次抓取报告，不直接对齐全局 step。标记不是精确的物理子步约束创建时刻。

| 数据 | 模式 | 首次报告索引（0-based） | 到首次报告这一步的杯位移 | 首次报告 ±3 采样窗口的相邻最大步长 |
|---|---|---:|---:|---:|
| 随机 #1 | raw assisted | 600 | 4.35mm | 8.06mm |
| 随机 #2 | raw assisted | 676 | 2.59mm | 9.65mm |
| 随机 #3 | raw assisted | 675 | 2.06mm | 9.97mm |
| 随机 #4 | raw assisted | 738 | 3.33mm | 10.05mm |
| 随机 #5 | raw assisted | 661 | 4.09mm | 6.45mm |
| 同 TRAIN/demo0 源轨迹 | raw assisted | 642 | 1.86mm | 9.93mm |
| 学得 B，execute10 | assisted | 688 | 2.20mm | 8.18mm |
| 同 B，execute10 | physical | 690 | 1.19mm | 3.35mm |
| 同 B，execute5 | physical | 无 | — | — |
| 同 TRAIN/demo0 专家动作回放 | physical，不是 π₀ | 644 | 0.74mm | 2.23mm |

B assisted 在 685–688 步的左手杯接触 link 数为 `0,1,2,2`，两次较大位移在抓取报告**之前**（7.84、8.18mm）。首次报告时末端原点到杯原点约 47.43mm；这不是指尖与杯的间隙，不能说杯隔着 4.7cm 被吸过去。30Hz 采样不排除帧间瞬态、杯姿态快速转动，也不支持此段出现远距离跳变或把某一位移唯一归因于约束。旧视频 15fps 只保留隔一控制步，视角也不同；不能从观感直接量化位移。

**分开结论：**辅助视频的“立即随爪”有已证实的仿真约束机制，不能作为抓取成功展示。无辅助同权重 execute10 的确能抬杯（约37.6cm）但仍闭爪时掉杯，execute5 无稳定抓取（抬升约4mm；6021×8 边界采样均无辅助）。同一来源专家动作在无辅助条件下却能运输并释放（3449×8 边界采样均无辅助，且非模型输出）。还不能把剩余的模型任务失败唯一归因为训练数据、摩擦、接触几何、关夹时序或某一个参数。根据用户要求，后续对外抓杯验收以 `physical` 且启用 `--audit-no-assistance` 为准，辅助测试只保留为诊断；另查松手与掉杯，不用 inside 谓词替代。

下一次若需隔离**约束本身**对 B 视频的影响，在一张卡从现有生成工作流释放后再做：用同一 B assisted 已执行 21D 动作前缀和同一空手 source0 初始状态，分别在 assisted / physical 条件回放，从接触前开始记录约束状态、杯相对夹爪位姿和接触随时间的变化；输入动作恒定，不进行新训练，也不把回放叫成新策略结果。当前未抢占 GPU、未修改旧数据或控制器。

## 证据与复核

- 量化脚本：`/home/yuanyue/double-arm/scripts/audit_cup_attachment_onset.py`。远程原报告 `/mnt/yuanyue/double-arm/runs/cup-attachment-onset-20261118-111847/report.json`，SHA256 `454c0b3b33caafcd2a11d701fcb6afb7ddd9ad88730954a6169c5987153fedca`。
- 抓取前后同类型头部视角关键帧对比：第一行来自随机样本 #1 的原 HDF5 RGB，其余来自对应完整视频头部视角，不插帧；本地 `/home/yuanyue/double-arm/outputs/demo-review/cup-attachment-onset-20261118/grasp-onset-contact-sheet.jpg`，SHA256 `872ba4becc5de83424e098c19afaa7c35050bb1681532e629d88353be0686683`。本地原报告与接触图均核验 SHA，已目视检查。
- 新的 2 项数值测试通过远程 CPU 验证；本地全套 203 项、153 通过、50 依赖跳过。代码不依赖新的模拟器 / GPU 配置。
