# 物理一致观测的反馈探测：仅初始化快照，不是策略闭环（2026-11-17）

## 为什么增加这一步

之前final19999同RGB/同噪声、假设q改变的离线probe发现native target随anchor近1:1偏移。但图像和假q不一致；不能直接据此说真实相机下也不纠偏。本次改为：**实际改变机器人初始关节状态，重新运行物理刷新、重新获取真实三路RGB和实际proprio**，再离线询问同一个π₀。它仍不是自然闭环扰动恢复或任务成功测试。

## 当前运行与独占

```text
/mnt/yuanyue/double-arm/runs/pi0-consistent-feedback-20261117-202643/campaign
runs/latest-pi0-consistent-feedback-probe.txt
controller206672 / snapshot worker206676 / watchdog wrapper206668
```

20:26启动前实际核对：旧raw5及CPU跨池QA terminal且进程退出、GPU1无compute进程、UUID锁可获得。GPU0 seed10004原生成不变。GPU1/map已指向本新probe；不要从旧generation历史指针重复启动任务。

**GPU1所有权覆盖真实快照采集→模拟器完全退出→CPU离线采样整个workflow**，CPU阶段即使显卡空闲也不能插队。snapshot/score各20分钟、外层45分钟watchdog仅防挂起，不重置用户预算。不创建app自动化、不接受新条款、只使用既有license confirmations。

20:28启动观察还在Isaac冷加载、已创建空条件列表；六组真实快照/18次推理均尚未完成。mock算术/门禁通过不代表真实模拟器或task通过。

## 实验条件：所有初始化干预显式记录

- 数据仅原TRAIN/demo0，source frames **0与635**（635是原首close636之前），每帧nominal/minus/plus三组，各自只加载一次source serialized initial state。
- **不是从frame635完成整任务或导航的测试。** 使用中间source状态仅用于初始化分解诊断，不能当端到端能力或heldout成功。
- ±0.02rad请求只作用于左臂六关节，初始target约束在原native训练bounds内；记录实际截断target、刷新后的实际q及实际变化，不假造达到完整±0.02。
- nominal与perturbed都显式设置一次trunk+双臂的**初始化holding drive target**、一次初始化body velocity清零。非nominal另有一次初始化左臂pose write。它们是已标明的人工初始化操作，**绝不以“没有env.step”为由隐去这些motor/pose写入**，也不是部署的规则策略。
- 之后每组仅20次`og.sim.step()`刷新及3次render，实际获取head/双wrist RGB、原21D proprio；零teacher/learned action执行。整个snapshot采集不调用模型。
- Source RGB只在服务器事后计算nominal对齐MAE，**从不替换快照中的真实相机或送入策略作为相机修复**。source/phase/index/object pose、holding target与遥测也不是模型输入。
- 原task grasp模式assisted、friction4.0保持；不做实机标定或物理抓取验收声明。ObjectPoseAudit在每次初始化后监测物体显式pose写入，机器人初始化写入另有真实累计计数。

## 门禁

1. 源数据任务/split验证、serialized布局验证、base实际POSITION控制器验证。
2. 沿用原`.98` root planarity检查；不放宽既有安全门槛。
3. 每帧nominal相机MAE≤20（0–255）/body q最大差≤0.01rad/物理刷新root平移≤0.02m，且无assisted attachment和初始化后的物体显式pose write。只是本快照诊断的保守一致性门禁，**不是新的任务验收标准**。
4. 扰动后的真实left q变化范数至少0.01rad、与初始target最大差≤0.01rad，且同样无attachment/物体pose write；失败就保留已知快照、停止，不用fake q/source RGB兜底。
5. 只有六组均通过且模拟器进程实际exit0，才启动CPU原π₀推理。

若中间source reset因驱动/刷新无法稳定，报告的是**新初始化诊断未通过**，不是又一条learned-policy闭环失败，不得与已交付的真实3000步失败混淆。

## CPU离线采样与解释边界

- 同一个relative-full final19999、同saved norm、保留FP32/BF16实际参数dtype；四个checkpoint耦合runtime源码不改。
- 六个真实snapshot各固定Gaussian seeds123/124/125，共18个native50×21预测，**不执行这些动作**。评分进程不打开专家HDF5；只读实际snapshot和当前模型。
- 同进程、同noise比较每帧扰动组相对nominal的前10 absolute joint target变化，用实际q变化作方向投影gain；记录真实相机改变及其它state变化。不是“只改q、RGB保持相同”的隔离干预。
- 原gain helper的fake-q scope在本报告中明确覆盖为actual refreshed RGB/proprio scope；不混淆两类实验。
- 即使gain接近1，也只是这些初始化条件下的输出响应描述。q、相机和少量物理状态同时变化，合法恢复动作不唯一，不能由此直接证明唯一根因、完全忽略视觉或宣布某个训练修复成立。
- 所有快照NPZ/模型/预测数组只在服务器；本地只保留源码、测试、说明，必要时取小型review接触图，不复制数据集/权重。

## 测试

本地171项：127通过/44依赖skip；本次新增门禁拒绝未知/漂移/assistance与初始关节界限算术共2项，在服务器Python3.10也通过。它们验证代码条件，不替代真实快照、全模型采样或物理任务结果。
