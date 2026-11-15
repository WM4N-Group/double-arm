# 抓杯视频异常检查：辅助抓取，不是纯物理验收

检查时间：2026-11-15 13:28–13:40 CST。

## 结论与对之前报告的更正

已交付的 `nearest_state_memory_BC / demo_1` 视频启用了 **assisted grasping**。之前“无规划器 / 在线教师接管”的说法仍对应上层策略执行，但**不能推出仿真底层没有辅助抓取**。先前没有披露此机制，属于验收遗漏；该视频只能标为“辅助抓取条件下、训练初始状态的记忆式 BC 任务成功”，不能作为纯接触 / 摩擦抓取成功证明。

不能仅凭观感断言远距离瞬移。保存的 30Hz trace 显示约第 778–781 步（0.1 秒）杯子中心在底盘坐标系中累计偏移 **16.54mm**、旋转 **21.22°**，单步最大位移 **9.90mm**；底盘在该段几乎静止。原视频 15fps，快速位移与旋转可能只表现成 1–2 帧跳变。这不排除接触 / 约束求解中的快速位置校正，但没有支持“从远处直接传送到夹爪”的大位移证据。

## 服务器实际配置与锁定源码

1. 成功策略 `manifest.json` 的 `env_args.env_kwargs.robots[0].grasping_mode` 为 `assisted`。
2. 锁定的 robomimic `env_omnigibson.py:264–267` 将 `macros.robots.manipulation_robot.GRASP_WINDOW` 设为 **0.0**。
3. 锁定的 OmniGibson `manipulation_robot.py`：
   - `grasping_mode` 文档区分 physical / assisted / sticky。
   - assisted 分支检查手指接触与射线命中，再由 `_establish_grasp_rigid` 创建 `FixedJoint`（部分关节物体可用 SphericalJoint）。
   - 会设置 `_ag_obj_in_hand`、冻结夹爪位置。因此 grasp flag 为 true 在该模式下代表**辅助 attachment 已建立**，不能替代摩擦夹持证明。
   - `_establish_grasp_rigid` 此处使用接触点构造约束坐标，并非直接调用杯子的 `set_position_orientation`；另有机器人主动传送时搬运已附着物体的逻辑，但本策略控制循环没有调用机器人主动传送。
4. 记录的 grasp flag 从 observation 780 的 -1 变为 781 的 1。

源码均以服务器实际安装的固定版本为准，没有升级或修改上游抓取实现。

## 已执行的修正与后续检查

- 评估结果 / 新视频增加实际 grasping_mode 标注，导出检查报告明确 assisted caveat。
- 后续评估包装器默认请求 `physical`；每回合先恢复空手初始状态，再检查不存在辅助对象 / 约束，最后关闭辅助分支。每步检查不得出现辅助约束。模式切换是针对上游不同序列化布局的显式诊断处理，不重定位物体、不改夹爪控制器或摩擦参数。
- 新版本正式 goal 发布要求 `grasping_mode=physical` 且 `assisted_constraint_seen=false`，辅助任务成功不满足此门槛。
- 增加同一已记录策略动作的 assisted / physical 对照审计，记录杯子与 EEF 位姿、指尖接触、辅助约束建立时刻和杯子显式位姿写入调用。它是**诊断性动作回放，不是新的学得策略成功**。
- 对照审计等待 GPU0 当前纯物理策略评估结束后执行，单独设置有限等待 / 运行超时；不会在同一卡启动第二个仿真。第一次审计因未导出已有私有许可标志而未启动，已保留失败日志并修复读取方式，没有接受新条款。

所有仿真继续在服务器；本地只取回约 26 秒附近的 8 帧接触检查图与小型 JSON。原视频和原始成功事实保留，但物理抓取结论已撤回。
