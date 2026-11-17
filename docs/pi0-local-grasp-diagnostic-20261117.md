# source635初始化局部抓取诊断：专家动作门禁 → original π₀短闭环（2026-11-17）

## 问题与范围

全任务final19999失败、真实初始化反馈快照仍表现为target随q偏移，但两者还不能回答：**主要是没到达正确pre-grasp条件，还是到了该条件也不能正确抓取？** 本次用有界动作实验分解，不继续盲加训练。

这是**privileged source635 INITIALIZATION的局部辅助抓取诊断**，不是完整Tidy Table、导航、heldout或physical成功。frame635是source首close636前一帧；原source的`obs/grasp_left`首True在642。source本来应在这附近能抓取，故120步窗口足够覆盖source抓取，不把过短窗口的失败误称模型缺陷。

## 当前workflow

```text
/mnt/yuanyue/double-arm/runs/pi0-local-grasp-20261117-213259/campaign
runs/latest-pi0-local-grasp-probe.txt
controller208599 / CPU service208603 / sim208744 / watchdog208595
```

21:32确认前一真实快照workflow完全退出、GPU1无compute进程且锁可用；服务器可用RAM约85.9GiB。GPU0原seed10004不变。CPU模型于21:34实际ready并完成compile-only warmup，随后启动模拟器；21:35仍在冷启动，**source回放及learned case均无完成结果**。

GPU1锁覆盖CPU加载→source replay gate→learned local control→视频导出整个workflow。任何阶段显卡看似空闲也不得插入新GPU任务。模型加载20min、simulation40min、外层65min watchdog仅防挂起，不是新预算窗口。无新app自动化或terms acceptance。

## 1. 同初始化setup的source replay门禁

- 每个case先独立env reset，再**一次**load source635 scene state；没有逐步恢复。
- 对trunk、双臂、双finger设置**一次初始化holding drive target及一次初始化velocity清零**；这比上一快照probe多包含finger，是为防止source replay闭合目标泄漏到下一独立learned reset。没有直接left q扰动，也没有循环holding规则策略。
- 20次physics刷新+3次render；保持原`.98` root planarity、新快照nominal RGB/body/root对齐门禁，并额外要求两侧gripper width与source差≤0.001m。全过程attachment累计与物体pose setter监测从初始化后开始。
- Source replay只读原source `[635:755]` 的120条动作，逐步普通`env.step`，原native bounds不变。明确`mode=source_action_replay / teacher_action_reads=120 / inference_calls=0`，不是学得策略。
- 只有source case真实完成120步、未abort、cup-specific grasp report曾True、物体显式pose writes=0、initial load=1，才允许下一learned case。该门禁仅验证本局部初始化setup能够抓取，**不是任务验收门槛或physical证明**。

## 2. Original π₀局部短闭环

- 重新独立env reset及load source635一次；同初始化holding/zero setup，不继承source动作数组或attachment。Source动作数组销毁；learned分支不读取HDF5 actions，`teacher_action_reads=0`。
- CPU服务保持原`serve_pi0_relative_full_r1.py`，同final19999、same saved norm、原dtype，Gaussian10/execute10、50×21 native输出，逆变换仅服务端做一次。没有新service、norm、标签、模型、planner或辅助控制策略。
- 编译warmup用的是上一轮**真实重采的**snapshot，输出丢弃、零动作；不是把stored source RGB当live输入。Learned每次请求都只含**当前真实head+双wrist RGB、原21D proprio、instruction**。
- 最多120真实步/12次模型调用，约4秒仿真录像，不宣称实时控制。不是直到成功才停；每步沿用原planarity与native bounds，物体显式pose write则abort。
- 模式仍为作者configured **assisted/friction4.0**，与source一致；任何attachment/杯子抬升必须标明assistance，不称physical grasp/完整任务成功。

## 3. 记录与视频交付

每case保留初始化检查、实际motor contract、完整实际完成步trace、预测chunks（learned）、raw proposal与capped actions、post-step EEF/手指接触/杯子位置、实际gripper宽度、cup-specific grasp与assisted约束累计、物体pose writes。未知遥测保留NaN/-1/errors，不填零。

`task_success=false / physical_task_acceptance=false`始终保留，因为不是完整任务验收。

- Source视频固定标为`SOURCE ACTION REPLAY ... NOT LEARNED OR FULL TASK`，绝不当模型成功。
- Learned没抓到：`FAILED LOCAL GRASP ... INIT635 ... NOT FULL TASK OR PHYSICAL SUCCESS`。
- Learned有杯子grasp report：`ASSISTED LOCAL GRASP OBSERVED ... INIT635 ... NOT FULL TASK OR PHYSICAL SUCCESS`；这只是局部现象，不替代全任务失败、TRAIN物理门禁或heldout物理。
- Abort单独标记，保留已完成步，不填充失败步。CPU导出压缩MP4、SHA、完整服务器解码、当前checkpoint inference-tree fingerprint；只按实际结果取小型review到本地。

Source gate失败时停止，不启动learned case，报告初始化/控制有效性未证实；不能拿它作新模型闭环失败。无新训练、checkpoint或数据split修改。

## 测试

本地173项129通过/44依赖skip；新增source replay完成门禁、局部assisted结果不能进入完整physical acceptance及标签隔离2项在服务器Python3.10也通过。Mock/contract测试不代表真实source回放或局部抓取已通过。

## 21:42结果：策略触发辅助附着并抬杯（未验证无辅助物理夹持）

**两case实际完成、sim exit0，workflow于21:42:29完成。** Source有效性门禁通过后才运行learned case，顺序与实际trace一致。

| 指标 | source action replay（非模型） | learned original π₀ |
|---|---:|---:|
| 实际步数 | 120 | 120 |
| 推理调用 | 0 | 12 |
| teacher action records读取 | 120 | **0** |
| cup-specific grasp report曾True | true | **true** |
| assisted attachment曾出现 | true | **true** |
| 最大杯子抬升 | 0.12181m | **0.13309m** |
| 初始化后物体显式pose writes | 0 | 0 |
| first effective close（runtime threshold0.5） | step1 | step3 |
| ≥1左手finger-link接触帧 | 105/120 | 114/120 |
| ≥2 finger-link接触帧 | 3/120 | 26/120 |
| contact未知样本 | 0 | 0 |

- Learned新初始化的body q最大差约0.000334rad、三相机source MAE4.95/8.82/5.88，gripper width初始误差约8.94e-8m。没有继承source replay闭合或attachment；nominal gate实际通过。
- 新只读motor collector真实记录了binary/non-inverted/input[0,1]/output[-1,1]，threshold0.5现在有**实际runtime证据**，不是往旧trace回填的假设。Learned117个close步是一段`[3,119]`，不是117次抓取尝试。
- Learned第一10步左臂target相对source replay RMSE约0.00340rad，仍能在本assisted条件下抓起；因此“晚关夹两帧”“任何target误差/clip”不能单独当所有失败的充分原因。它也不代表给出了通用精度容差或优于专家。
- 杯子最大高度/接触数量较source高，不证明grasp质量更好；attachment、局部轨迹、采样和实际姿态都会影响指标，未测力，EEF-origin距离不是指尖净空。

### 能确认什么 / 不能确认什么

**只能确认：**真实相机+proprio→同原π₀模型→普通native控制，在特权pre-grasp初始化后输出了关夹/抬臂，触发assisted attachment，随后杯子随臂抬升。Learned段未用teacher action、planner或每步状态恢复，但模拟器辅助约束确实参与承载；backend grasp report不是独立的物理夹持验收。不能据此宣称模型已经学会在无辅助条件下夹住杯子。Source replay只验证相同assisted初始化setup可用。

**不可以确认：**完整起点出发的Tidy Table成功、导航成功、physical grasp、heldout/generalization或整篇论文复现。这里有明确的**privileged source635 pre-positioning、初始化holding/velocity操作和assisted constraint**。原3000步完整TRAIN/ASSISTED失败仍有效，物理/heldout门禁没有因局部结果解锁。

这把排查重心收窄到**全流程到达pre-grasp状态、轨迹覆盖、误差累积/恢复**，但不是已经证明唯一根因。完整rollout与本局部case的初态、速度/控制器历史、长度、噪声路径也有区别；不能说已隔离“只有导航”这一因素。接下来的uniform-sampling B只改变训练采样，用新完整source0闭环验证，不把本4秒片段当最终成功。

`task_success=false / physical_task_acceptance=false`在本诊断中意为**不具备/不宣称全任务验收**，不是声称本次另外测量了全部goal predicates并得出失败。

### 可审阅的新模型局部片段

```text
/home/yuanyue/double-arm/outputs/demo-review/pi0-local-init635-ASSISTED-NOT-FULL-TASK-20261117/learned_pi0_local_grasp.mp4
```

- 4.000s、960×536、H264、60帧，小型压缩MP4。画面及片尾明确`ASSISTED LOCAL GRASP OBSERVED / LEARNED PI0 INIT635 / NOT FULL TASK OR PHYSICAL SUCCESS`。
- 视频SHA256 `e453ccdb606cbf2d189342bbc999059536eef56a19906bafbc2d53445ec1ec61`；review JSON SHA256 `f8bd4252ebbc2716776e728abbefaf660fde81a2d2ac6d714f72b059cfca4cd3`。
- 服务器完整解码通过；传输SHA一致，0/0.5/2/3.9秒接触图已目视检查。接触图SHA `9e8d7198c1bd8986fafa499f35e3d9287ea5126ef8f065496ddc78096d71bf82`。未声称本地ffmpeg解码（本地没有ffmpeg）。
- CPU事后分析`campaign/local-forensics.json`与源码`analyze_pi0_local_grasp.py`，只读两份已完成trace，无模型/仿真调用；source动作只当明确标记的事后参照。

新片段的模型仍是A final19999/full parameter SHA `ffe08f...b78180b`，不是下一uniform B权重。原100秒失败视频继续保留；没有拿这次局部assisted抓取覆盖原始端到端失败。

## 22:10用户视频审阅后的措辞纠正

用户指出杯子瞬间被吸附到夹爪，这是有效且重要的审阅意见。前述“模型并不是完全不会抓杯”“真实抓起”容易扩大证据含义，应收紧为**触发辅助附着后抬杯**，不以局部视频证明物理夹持或完整抓取技能。

- 该learned episode实际`assisted_constraint_seen=true`。Pinned OmniGibson的`manipulation_robot.py`在assisted候选/接触/射线检查后，通过`_establish_grasp_rigid`创建末端与物体之间的`ag_constraint`，并设置辅助gripper freeze状态。约束承担了额外稳定/附着作用，接触力和摩擦能否独立维持杯子未被验证。
- `object_pose_write_count=0`只说明没有被审计到显式物体pose setter，不排除constraint solver把物体拉住/带走，**绝不等于无辅助或无“吸附”**。
- 只读trace中最大相邻post-step杯位移约8.93mm，发生于local step7（约0.233s）。本trace没有逐步attachment flag，不能把这一跳变精确分解为constraint建立瞬间、正常运动或碰撞解算的贡献；这不改变本片确有assistance、不是physical proof的结论。
- 13.3cm是**辅助约束条件下**的杯子抬升，不是物理抓取成功率/抓力/稳定性证据。原始视频、JSON、trace保持不变，保留原始字段`local_cup_grasp_report_seen=true`，但解释应明确它是辅助模式的backend报告。
- 均匀采样B仍作为覆盖诊断继续，22:10实查275个真实updates、无error；未修改或中断训练、norm、标签、adapter。局部assisted结果仍不解锁完整TRAIN/physical或heldout门禁。若做局部physical-only对照，必须另立诊断、真实禁用attachment，并以接触/夹持/抬升结果独立评估；不能用这段视频替代。
