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
