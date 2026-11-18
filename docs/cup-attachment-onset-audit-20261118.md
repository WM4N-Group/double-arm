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

### 11:25 CPU 输入预检完成；成对仿真尚未启动

在服务器只读核对 B 的实际 assisted native 动作与 review 的原 trace SHA（`89276a1f5ebc819dbd30f4b361b796136c78f84ce4fda5be76f745564f38dbd5`）。全部 1694×21 FP32 动作等于已执行的 bounded targets，未重新裁剪。两个条件拟共享第 `[0,1200)` 步，SHA `fe618b0ec85f255cea4878e551d11ca642be12d8791c404c2a663de43ca37da5`；原闭环首个杯抓取报告为 688，前缀覆盖该窗口及之后约 17 秒。源训练 demo0 指纹、空手 frame0 序列化状态、原21维动作范围、20 次刷新与原摩擦也已锁定；源码证据中的 688 仍是抓取报告，不是精确约束创建时刻。

远程输入计划：`/mnt/yuanyue/double-arm/runs/B-assisted-tape-pair-preflight-20261118-112520/plan.json`，SHA `dcdba8950dddffb3c44d1b7c436ed5756080e4c20b4884740e2ea263f3c5805e`，状态 `INPUT_PREFLIGHT_PASSED_NOT_LAUNCHED`。两卡原采集继续，未启动 GPU、训练或推理。不要复用会暂停旧控制器的历史 `run_grasp_audit.sh`；成对实验的运行程序须按现行 GPU UUID / lock、物理无辅助与两分支初始化一致性要求单独审查。1200 步结果只回答接触、吸附和早期持杯差异，不是完整放置验收。

### 12:24 跨分片专家物理回放小样本已预先固定，未跑仿真

为了回答「示范在关闭吸附后是否还能抓稳并放下」，使用 11:24 跨池 QA 的 **255 条、445007 帧** immutable snapshot（审计 SHA `f1c72ee74ff2941f94dfa732d421f3bcfc0f4a2d3d530fd537441036c6e71908`）预先固定 8 条专家轨迹、每个分片最多 1 条。脚本 `/home/yuanyue/double-arm/scripts/plan_expert_physical_pilot.py` 以固定命名空间 + audit SHA 分别排序分片和 episode 指纹；排序不依赖源 JSON 内顺序、视频外观、接触结果，不事后替换。两项单元测试本地与服务器 CPU 均通过。

冻结文件：`/mnt/yuanyue/double-arm/runs/expert-physical-pilot-20261118-122448/plan.json`（SHA `ea90461c1c9d7d6193eda55957e6837779146dde9df7698434cada4e5c6ed96a`）。8 个 seed / episode：10007/demo_9、10005/demo_24、10001/demo_12、10006/demo_10、20002/demo_2、10002/demo_17、10003/demo_3、20001/demo_12。该小面板是**跨分片可行性诊断**，不是按单条全池抽样的总体估计、场景多样性证明、训练划分或 π₀ heldout；后续新增 shard 不更改本轮名单。

下一可用算力窗口依次做固定 B 动作成对对照和跨 shard 专家测试；物理分支从各自空手 frame0 恢复一次，20 次刷新，按原 native 动作执行并审计实际无辅助。报告接触、抬升、运输保留、松手、松手后杯在水槽；初始化错误 / 中断与执行完整但任务失败分列，不以“杯在水槽”替代释放，不将专家回放称为 π₀。在两卡现有采集工作流期间没有启动这两项仿真。

### 13:23 固定动作成对仿真的输出校验已准备（不含仿真启动器）

新增 `/home/yuanyue/double-arm/scripts/compare_paired_grasp_replay.py`：先按完整 trace SHA 与 FP32 逐字节核对 1200 步指令；未来两例都必须证明实际执行数组同字节、空手 frame0 一次加载且完成 20 次刷新、没有策略调用或显式物体位置写入。开始控制前，根、杯、左末端的实际世界位姿差各需在 0.5mm / 0.005rad 内，21 维原生 proprio 最大逐坐标差在 0.001（混合单位）以内；超差停止比较，不靠重定位修补。每个例需要初始、20 次刷新及每条动作前后两只手的确认记录；physical 中任何未知或辅助记录都拒绝，且检查 Python 与 USD 的约束一致。只报告实际观察到的首次杯约束边界及其前后位姿、接触差异，不冒称精确物理子步。没有观察到目标杯约束时不强行给出 onset。

服务器 3 项数值测试通过；真实 B 计划和 1200 步动作前缀的 CPU SHA 合约通过。**尚未执行新的成对仿真**，仍需后续按现有 GPU 锁和独立新进程接线。两卡官方采集为活跃任务（seed10008/20004），未抢占。此前 12:29 跨池 QA 已纳入 aux100 的首个正式分片，确认 280 条 / 488651 帧；新增分片不更换已冻结专家面板。

### 14:23 成对回放启动器完成 CPU 预检，GPU 尚未运行

新增 `/home/yuanyue/double-arm/scripts/run_B_paired_attachment_replay.py`。只允许在 GPU1 的整项 aux100 成功退出之后（状态验收满 100 条、控制器/子进程已退、无 GPU1 compute process、取得同一 UUID 锁）显式启动；GPU0 不碰，不能用分片间的暂时空卡。两个分支各用一个新模拟器进程、同一 source0 只加载一次、相同 20 次刷新与 1200 个原始命令，除此之外只切换 `assisted` / `physical`。实际执行命令、位姿、手指接触和 Python/USD 边界证据与模式标注视频各自落盘，完整两个分支后才跑 CPU 比较验收。无 PI₀ 推理、无专家动作接管、无新训练；专家数据只用于初始化和身份核查。已知风险：实际模拟器集成和阈值可行性尚未实跑，不能预报成功。

服务器 `--check-only` 已读实际冻结 B 命令、action limits 和 HDF5 frame0 并核验 SHA；返回 `gpu_launch_performed=false`，GPU1 aux100 仍活跃且仅验收 25/100，因此未启动。释放门禁测试本地及服务器通过，数值测试服务器 3 项通过。下一次实跑需要 GPU1 确认释放后显式给 `--campaign` 一个新的远程目录，不创建等待或抢占进程。

### 15:25 跨分片专家面板的八条实际输入已验收（只读 CPU）

`/home/yuanyue/double-arm/scripts/verify_expert_physical_pilot_inputs.py` 从冻结 audit 重建相同的 8 条名单；核对八个正式分片的 generation verdict / manifest SHA、数据集路径、D0/30Hz/assisted 元数据、21 维 FP32 原动作和帧对齐、动作与初始 proprio 指纹、每个源 frame0 的序列化状态与三路 RGB 的单独 SHA。实际共 **13,567 条动作**，首帧均记录双手未抓取，全部源状态宽度 3082；这些是原始 HDF5 证据，不证明实际重置无约束或任何物理结果。没有裁剪专家动作、套用旧 16 条训练的经验上下界、重抽名单或输出训练集。以后运行须复查 SHA 并校验实际加载布局、真实观测和两手无辅助，不能把源文件标志当仿真证据。

远程报告 `/mnt/yuanyue/double-arm/runs/expert-physical-pilot-inputs-20261118-152505/report.json`（SHA `26758174a7dea626ca74dcba4df29f3c3b3d80db1395e53454d32e4a27825a2d`）。没有启动新的仿真或抢占两卡运行中的采集；当前没有新策略或物理回放结果。
