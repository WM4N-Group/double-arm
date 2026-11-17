# Source0物理回放审计修复：零动作初始化中止，不是新策略失败（2026-11-18）

## 1. 首次source回放没有执行动作，不能读取父状态当成功

01:27启动的`source0-physical-replay-20261118-012724/campaign`实际在首次初始化无辅助审计时停止：

- `replay/result.json`: `stage=source_replay_aborted_NOT_learned_policy / aborted=true`
- 实际执行source commands **0/1714**、physics refresh **0/20**、no-assistance样本1。
- Python constraint/in-hand和真实USD `ag_constraint` prim均为空；但两手`_ag_freeze_gripper`不是Boolean，被旧审计器保守标为未知-1并拒绝。
- 实际进程exit0，旧父控制器只按exit code误写`completed_source_physical_replay_NOT_PI0`。**这是新回放harness状态汇总缺陷，不是source完成证据，不是π₀新失败，不是物理不可能抓取证据。** exit0与已写入的abort不一致，底层退出处理具体原因未单独定位，不能据此解释模型行为。

原结果、父status、日志、源码快照及rejected JSON全部保留，不追改成成功；没有回放视频可交付，因为未执行任何动作。

## 2. 核实pinned源码里的None不是一般未知值

服务器实际pinned `omnigibson/robots/manipulation_robot.py` SHA256：

`af1e0245078a69a67368be4e70af4f5b79190c7d7dcff4fe9be210da1c39b0ba`

已只读核查：

- `__init__`明确将每只手`_ag_freeze_gripper`初始化为**None**。
- `deploy_control`使用`if self._ag_freeze_gripper[arm]`决定是否覆盖gripper控制；**该初始化None在这份实现中不激活freeze分支**。
- `_establish_grasp_rigid`才设为True；release设置False。

所以旧审计的“只接受bool、None一律未知”不兼容这个已知初始化sentinel。修复的是**只读解释逻辑**，不是robot/controller/仿真物理：

- `snapshot_no_assistance`增加显式可选`inactive_freeze_none_source`。
- **只有实际读取到的源码hash精确匹配上面已审计文件**，才能把初始化None解释为inactive；缺失/改动/未知源码仍拒绝。
- 输出保留`gripper_freeze_raw=None`及source SHA、解释scope；不把原值伪装成bool，不向robot写False，不清除约束，不删USD prim。
- 默认调用不提供这个已验证源码时，None仍保留unknown/-1/errors，不能把任意未知值普遍当0。
- 真实USD presence query仍必须已知False，Python constraints/in-hand/actual physical mode仍必须检查；任一missing/active/error仍拒绝。

这不修改已有B的训练、norm、标签、物理配置、评估结果或后验证据，也不把新审计观测倒填给旧B。

## 3. 父控制器从实际完成内容判定

修复为只有以下全部成立，才能标`completed_source_physical_replay_NOT_PI0`：

- sim exit0；
- actual result stage确为`source_replay_completed_NOT_learned_policy`；
- `aborted=false`、无error；
- 真实source commands executed=1714；
- 全部已检查no-assistance样本明确通过。

缺少这些实际内容，即使进程exit0仍标setup/replay失败。`inside goal`是否成立与“source1714动作确实执行完”分开；两者都不是学得π₀成功。

## 4. 显式新attempt，不是恢复旧模拟器或盲目重试

```text
/mnt/yuanyue/double-arm/runs/source0-physical-replay-sentinel-v2-20261118-022531/campaign
runs/latest-source0-physical-replay.txt
controller221423 / sim221426 / wrapper221418
```

02:25启动前检查旧controller/sim实际退出、GPU1无compute进程、UUID lock可用，GPU0 seed10006原生成不变。

- 使用`--repair-aborted-run`精确验证旧campaign：只允许上述初始freeze审计类型错误、0动作/0refresh、仅一次审计、Python/ USD均无constraint/in-hand。未知错误或已发生动作的失败不走此自动repair入口。
- 重新source0空手初始化，原1714动作、20refresh、source0 RGB/body对齐、root`.98`、friction4.0及native limits完全不变。没有模型训练、模型推理、holding/velocity额外干预、planner或逐步状态恢复。
- 当前GPU1/map指针已移到新attempt；旧B pointer保留terminal failure，新attempt保留旧owner和repair-of指针。
- 原source代码、已审计上游实现及helper快照/SHA均留在新campaign服务器目录。运行中不修改这些文件。

截至02:26仍在初始化，实际USD/None sentinel解释和1714动作执行**尚待真实验证**。启动和11个相关CPU fixture通过不能代替仿真结果。

GPU0原worker已完成150raw episodes/261,496 frames并进入seed10006，但最后global immutable QA仍180/314,090；本次未做新跨池审计，不直接相加报告新全局计数。论文规模/新split/cleaning/norm均无变化。

## 测试

无辅助观察器7项、回放完成/repair门禁4项在服务器Python3.10通过；本地189项145通过/44依赖skip。新增测试包括：exit0不能掩盖abort/0动作、已知setup错误精确匹配、默认None未知、确切源码sentinel保留原值、源码改动拒绝。没有把mock确认写成实际physical通过。
