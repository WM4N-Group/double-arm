# B命令重放逐数组重现失持；下一步只干预底盘接近前缀（2026-11-18）

## 1. 已完成：移除在线模型后，失败仍由相同命令重现

```text
/mnt/yuanyue/double-arm/runs/recorded-B-physical-replay-20261118-032522/campaign
runs/latest-recorded-B-physical-replay.txt
```

实际于**03:37:28 CST**完整执行3000/3000条原B失败时的native命令，exit0、unaborted、6021份已检查无辅助Python/USD状态全0，inside-goal持续成功帧0。模型调用0，没有source专家手臂/夹爪控制、planner或逐步恢复。

CPU `audit_B_recorded_replay.py`比较原B live physical与这次固定命令回放，以下**全部3000步数组逐元素一致**（max difference=0）：

- 原21D observations；
- 实际执行actions；
- post-step杯子位置；
- 双EEF世界姿态；
- 手指接触计数；
- 杯子backend grasp flags。

两次都在step1136 /37.867s起失去接触，杯子同样坠落，最大抬升同为0.375941m。**这次失败可以在不运行在线RPC/视觉推理的固定命令回放中原样复现。** 不是只有在线推理时才出现的运动偏差；不需要把CPU推理等待当作本次掉杯的必要解释。

**边界：**这只确认这对已完成执行的状态数组一致，不是所有跨进程JAX采样bit-exact的声明，不取消先前模型采样差异记录；也没有证明未来所有仿真都确定性。仍不唯一分解哪些命令轴/抓取姿态/动力学是原因。

证据：

- 新trace SHA256 `9c1cd7e22fe1196ea038b810711328b888afdad64039ef34effc93245188ee83`
- equivalence report SHA256 `d40846fc61126dcc25d32b7e438911ed54293c265ce5913092d78d8cc2465fed`
- 新NO-LIVE-POLICY录像SHA256 `0509b6de7e3ebd3d0885fe9f23e208ed010141356f77418994818f4ec0f52e62`
- review SHA256 `10be94991713994cba6272dc1dffbae171677df1c9bd7170085040cd0efbd5c0`

小型压缩review与报告已SHA核对取回：`/home/yuanyue/double-arm/outputs/demo-review/RECORDED-B-COMMANDS-FAILED-NOT-LIVE-PI0-20261118/`。服务器完整decode通过；其动作/几何与已目视检查的原B失败逐数组一致，不把该录像当新的learned-policy episode。

## 2. 抓取前已经不是同一个空间构型

按每条轨迹**自身首次有效close之前一步**作描述性对照（source635，B681），不是相同elapsed time：

| 量 | 可行source物理回放 | 失败B |
|---|---|---|
| root XY(m) | (5.49442,-0.75560) | (5.51038,-0.67718) |
| cup XYZ(m) | (6.04976,-0.00391,1.07539) | 同值 |
| post-step左EEF原点XYZ(m) | (5.99793,-0.00020,1.08598) | (6.10720,-0.00486,1.11003) |

- B相对source root XY距离约**0.08003m**，yaw差约**-2.985°**。
- 此时杯子尚未移动，左EEF原点相对杯子世界X差在source为-0.05184m、B为+0.05743m；同时手臂关节也不同。
- **EEF原点不是指尖/夹持中心**，不能直接用它判碰撞、距离容限或唯一抓取位姿。上表是抓取构型不同的证据，不是“必须到source底盘点才能抓”的证明，也不能唯一归咎底盘。
- 与前一杯质心EEF-local漂移P95 source约5mm/B约17mm的只读统计相容，进一步支持检查接近/接触几何和保持动作，而不先调摩擦掩盖问题。

## 3. 新有界因果探测：仅替换固定命令的底盘接近前缀

在精确回放全流程退出、GPU1实际无compute进程并取得UUID锁后，于**04:26**启动：

```text
/mnt/yuanyue/double-arm/runs/B-base-prefix-diagnostic-20261118-042624/campaign
runs/latest-B-base-prefix-diagnostic.txt
controller224581 / sim224584 / watchdog224577
```

**这是OFFLINE HYBRID ACTION REPLAY，不是π₀或修复后的policy。** 为检验“底盘接近带来的抓取构型偏差是否有作用”，只做一项明确干预：

1. 取原B已执行命令的前1200步（覆盖B原step1136失持事件）。
2. 在前682步（0..681，原B first-close之前）只把**base轴0:3**换为原source在同elapsed indices的base命令。
3. trunk、左右臂、左右gripper共其它18轴在全1200步**逐值保持原B命令**；step682以后的base也恢复原B命令。
4. 不修改源HDF5/训练label、模型adapter、bounds或friction，不混入expert arm/gripper，不做任何重规划或target blending。

`diagnostic_base_prefix.py`显式验证shape/dtype/finite，逐值断言非base轴以及prefix之后base不变，输入数组不修改。Source-base参与的是**回放控制干预本身**，不能像learned case那样声称没有teacher control，也绝不能把该结果作为mainline模型成功。

### 实际设置和解释限制

- 同source0空手初态、load1/20refresh/真实相机及q对齐、原native limits/.98 planarity/friction4.0；已验证无辅助边界检查与raw None来源合同保持，不插入root pose writes、导航planner或逐步恢复。
- Source base命令是同elapsed索引，不做phase对齐/时间warp；B原关节目标来自原B路径状态，干预后它们不再是新state的在线模型反应。因此这不是“只修正导航的真实闭环π₀”，也不保证是自然一致的完整策略。必须记录实际轨迹，结果只用于敏感性/动作轴作用定位。
- 1200步局部窗口不测完整放置/释放，`learned_policy_task_success=false`始终不变；没有inside也不等价为这一局部探测“task失败”，目的在保持接触/是否失持。
- 若原掉杯事件消失，支持该prefix动作块在本固定命令上下文里有影响，不证明底盘是唯一原因；若仍掉，也不证明底盘误差无关，可能与arm目标的耦合有关。Source位姿不是唯一合法解。
- 当前只做一个预定义candidate；不是反复挑成功视频。04:27仍冷启动、0动作，暂无完成/保持结果。40min simulator/50min outer是防挂起，不重置用户预算。

GPU1/map已指向新hybrid诊断；`latest-recorded-B-physical-replay.txt`与`latest-source0-physical-replay.txt`都保留已完成历史，不再代表当前卡空闲。GPU0 seed10006原官方生成继续，无新增training/checkpoint/terms acceptance/app automation。

## 测试

本地194项146通过/48依赖skip；新增base-prefix仅改预定义3轴前缀、输入不变/异常拒绝2项在服务器CPU通过。fixture不证明实际root被纠正或杯子不掉，需读本新run真实结果。
