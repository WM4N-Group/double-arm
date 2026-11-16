# 新官方原始示范池首批25条完成与只读审计（2026-11-17）

## 1. 实际完成，不是临时日志计数

04:06 CST核查确认GPU0首个seed10000 shard已finalize，并在约03:56完成controller验收：

```text
/mnt/yuanyue/double-arm/runs/tidy_table-official-20261116-231116-TQQPAq
```

- 官方生成器exit0，31次尝试、25次来源任务成功、6次失败。
- 最终`demo.hdf5`、`important_stats.json`和`generation-result.json`已经生成，`state=passed`，不是tmp文件或仅成功日志。
- controller检查真实三路RGB、原21D proprio、native21D动作、完整回合与字节指纹去重后，原始池为 **25条、43,767帧**。
- source task为Tidy Table / D0，作者**assisted**配置。此阶段是官方数据生成，**不是π₀推理、物理抓取验收或论文训练完成**。
- 固定MoMaGen / BEHAVIOR / robomimic revision与原source SHA保持不变。GPU0已自动进入seed10001第二批，未重启主controller；当前第二批中的planner retry是官方重试流程，不应单看一条规划失败日志就宣称整个worker失败。

GPU1的seed20000辅助批次仍在生成；04:06官方日志为8次来源成功 / 14次尝试，**尚未finalize或计入verified pool**。

## 2. 独立CPU数据审计，没有新划分或数据变更

新增可复用的`audit_official_pool_snapshot.py`。它先把一个或多个已完成raw provenance读成快照，支持跨worker去重 / 旧split重复排除，再对**已finalize**的source actions / 初始proprio重新验证指纹，并检查qpos、动作、proprio和三路RGB header帧数对齐。不读取正在写入的tmp，不调用模型或仿真，不重新解码所有RGB像素，不改HDF5 / LeRobot / labels / normalization。

本次报告：

```text
/mnt/yuanyue/double-arm/runs/official-raw-pool-audit-20261117-0406/report.json
runs/latest-official-raw-pool-audit.txt
```

结果：

| 检查 | 结果 |
|---|---:|
| raw provenance输入回合 |25|
| 快照内重复 |0|
| 与旧16条train的fingerprint重合 |0|
| 与旧1条heldout的fingerprint重合 |0|
| 待后续版本选择的eligible raw episodes |25|
| 原始帧数 |43,767|
| 清洗 / 新split / 新norm写入 |均没有|

指纹定义为native FP32 action bytes + 原始初始proprio bytes。零重复只是该定义下的**字节轨迹去重**，不是25个全新场景、语义独立或20个未见物理评估回合的证明。当前没有把25条分成train/validation，更没改旧16/1的provenance。

报告SHA256：`04754d7b3b2e6c61c25fcc30da6b3adcf9d62a22b11eab66beef92b62149377e`。报告内包含原provenance快照 / SHA、各final shard manifest与result SHA、每回合action/proprio指纹和qpos SHA；当次source脚本也已保存并核对SHA。

## 3. Frozen-segment判据只读结果

对记录的`obs/joint_qpos`26D使用已有论文C.1实现：比较q[i]与q[i+5]端点，逐坐标严格<1e-3，候选删除inclusive[i,i+5]并集。**作者精确joint subset尚未确认，不把这个字段选择说成已完整复现作者清洗。**

本次25条检出 **6个候选帧**，仅在demo_12的557–562（含端点）：

- 最大端点差约0.0006610；该窗口内最大相邻差约0.0007156。
- 左夹爪命令六帧均-1，右夹爪六帧均+1，没有命令转换被候选窗口覆盖。
- 细节保存在`candidate-window-detail.json`，对应qpos SHA亦核实。

**没有实际删除这6帧**。43,767仍是raw frame count，不能改口称清洗后43,761帧已进入训练，也不能因未覆盖夹爪命令转换就断言删除一定安全。它证明新数据上这个只读判据会产生非零候选，不意味着清洗已改善策略。

## 4. 验证与接下来

- 本地120项测试：91通过、29数值依赖skip；新增跨worker去重 / 旧split排除 / 无效指纹2项夹具在服务器Python3.10通过。
- 两张GPU仍各自运行官方生成，没有插入第三个GPU任务、没有打断或重启当前shard。
- 后续在更多final shards到齐时，对各worker快照统一去重、固定选择与90/10拆分，再另建cleaning/action/norm数据版本；不能把原converter“一条heldout”默认当论文90/10。
- 论文action/exec/freeze与精确joint-subset仍待核实；host微批64内存smoke通过与这25条来源数据成功，都不能覆盖此前coverage v3学得策略失败的结论。
- 首批finalized/独立QA是新里程碑；后续正常shard递增作为健康in-progress处理，保持quiet，不为每25条例行通知。目标完成、质量失败或必须处理的阻塞再通知。
