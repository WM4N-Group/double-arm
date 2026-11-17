# π₀训练策略审计（2026-11-17）

## 结论

**现有native coverage v3是已失败的诊断配方，不能继续作为已对齐的正式MoMaGen复现策略。** 本次没有发现能解释全部失败的图像/张量/50步标签错位；也没有证据证明“微调完全没学到任何R1动作”或“视觉灾难性遗忘已确定发生”。真正需要处理的是配方差异、预训练保留的对照缺口、控制执行约定与任务级验收，而不是继续把训练loss/离线误差当抓杯能力。

全部新计算为服务器CPU-only；两卡的官方生成未中断、未重启，未新增GPU训练、改变标签/归一化/控制器或降低物理门槛。

证据根目录：

```text
/mnt/yuanyue/double-arm/runs/pi0-training-strategy-audit-20261117-112440
runs/latest-pi0-training-strategy-audit.txt
```

## 1. 实际数据与训练/推理接口：抽查通过

使用真实固定openpi loader构建原TRAIN16条LeRobot数据，而非mock数组；核对实际训练checkpoint中记录的三个adapter源码SHA，均与当前对应文件一致。抽查每回合开始、中段、首次负gripper命令前后、倒数第二帧和末帧，共 **96个窗口**：

- 三路原始HDF5 RGB与LeRobot图片解码后逐像素相同，state与source一致。
- 每个50步action chunk与source逐值一致；末尾重复本回合最后动作，没有跨episode读下一条动作。
- train侧的resize/tokenize/normalize/pad，与使用checkpoint stats的inference侧一致；原始训练/保存的mean/std等数组一致。
- 匹配真实loader的batch/collate/JAX转换后，模型Observation图像也完全一致；不是用无效的unbatched scalar-mask mock来判断真实管线。
- 21→32 padding后11个state/action通道为0。inverse norm + unpad还原source，最大绝对误差约5.42e-20。
- adapter输出模型字段中没有dataset index、timestamp、phase、物体位姿等额外条件；future actions仅为flow-matching训练标签，不是在线VLA输入。

限制：这是覆盖所有16回合的**采样窗口审计**，不是逐帧审完27,657张图；也不证明live所有时刻的相机/场景分布都等于source。比较发生在模型内部有意的train=True augmentation之前；官方训练还有head95%crop/±5deg旋转、三相机color jitter，不把正常augmentation差异误判成数据错位。

首版audit fixture遗漏真实batching，使类型检查拒绝np.bool_单值mask；它是审计脚本缺陷，已保留旧log/source并修复后完整重跑通过，**不是原训练失败原因**。

报告：`contract-probe-batched/report.json`。

## 2. 未针对R1微调的底座对照：已补离线部分

在同一脚本、同一TRAIN/demo_0的0/571/1142帧、同一R1 adapter、同一TRAIN norm、同一显式Gaussian noise和10步去噪下，分别在独立CPU进程加载：

- 官方π₀ base原始权重（plain原π₀，无R1梯度更新；不是π₀.₅/FAST，也不是别的平台expert checkpoint）。
- coverage v3 final89999权重。

输入、噪声、norm与provenance SHA在两次运行间逐项匹配；均exit0，3+3次采样。

| 条件 | 前10动作步unitless normalized MSE |
|---|---:|
| 官方base权重 + R1 adapter + TRAIN norm |0.8094236|
| v3权重 + 同adapter / norm |0.0261761|

这支持“微调确实改善了这些训练观测上的R1动作模仿”，不支持“完全没学到东西”。但：

- **不是原生R1零样本任务验证**：base用了本项目R1通道映射和训练集统计量；不能据此评价π₀在其支持平台上的一般能力。
- 仅3个训练帧，没有形成完整机器人闭环，不证明v3抓杯更好、泛化或语义能力未受损；多个有效轨迹也可能与唯一expert参考不同。
- 旧30k报告只有历史参考意义，没有此次全套paired hash，不将其与新报告强称严格单变量比较。

报告：`paired-weights-control/status.json`、两个完整采样JSON及冻结benchmark源码。**完整未微调R1闭环baseline仍缺失**，不能用本离线对照替代。

## 3. P1：LoRA名称掩盖了大视觉分区训练（事实已核实，伤害未证实）

实际filter参数分区：

| 分区 | 参数数 |
|---|---:|
| 全模型（含LoRA） |3,315,905,296|
| 可训练总数 |495,908,624|
| 其中PaliGemma/img视觉分区 |414,803,696|
| 其中llm / action-expert LoRA等分区 |77,856,768|

原配置使用官方`model.get_freeze_filter()`；该规则冻结相应LLM非LoRA部分，**没有冻结SigLip视觉编码器**。所以不是“只有小LoRA在训”。若只额外冻结img、其余不变，可训练数应降至81,104,928，仍含投影等非LoRA参数，不应又误叫绝对pure-LoRA。

为验证不只是shape/filter理论，CPU部分restore了视觉`encoder_norm`的scale/bias两条各1152元素张量，没有读出/保存完整模型：旧30k、v3中这2304个抽查元素均较官方base改变。v3的相对L2变化约0.706%/0.720%。

**只能证明这些视觉参数确实更新；不能将两个小向量的变化外推成整个414M encoder漂移量，也不能因此证明灾难性遗忘。** 少量数据反复更新大视觉分区是需控制实验排除的风险。当前使用同一个AdamW/全局LR schedule（peak2.5e-5），没有单独的低vision LR；这仍不是“该LR已证明错误”。作者实际vision freeze设置仍未核实。

报告：`vision-encoder-norm-drift-repro.json`，复现代码为`audit_pi0_vision_drift.py`。

## 4. P1：现有配方不是论文训练策略

| 项目 | 当前已失败诊断 | 已核对的论文目标 |
|---|---|---|
| 数据 |16 train /1整回合heldout |1000 demos，90/10|
| effective batch |1 |64|
| 权重 / optimizer |旧30k权重warm start、fresh optimizer，再90k |独立论文配方，从指定pretrained底座微调|
| 正式样本曝光 |30k+90k=120k（不计独立smoke）|50k×64=3.2M|
| norm |mean/std |1%–99% quantile|
| cleaning |未应用 |endpoint frozen-segment cleaning；精确joint subset待核对|
| rollout |先单个训练回合；失败后未做heldout |固定20个未见episode等完整评测|

不能用“90k比50k多”解释为更多论文训练；当前正式曝光约论文目标3.75%，示范多样性更不能靠重复epoch替代。warm start与LR schedule同时改变也不构成纯“步数”因果对照。

### 已确认的配置设计限制：normalization与action representation绑定

`pi0_r1_config.py`用`absolute`决定quantile；`run_pi0_entrypoint.py`与`serve_pi0_r1.py`又把quantile等价于absolute_base作为硬校验。它正确保护了两个历史实验，却**不能表达native动作＋quantile的新论文对齐版本**。

必须在新版本中把“动作表示”和“norm方案”作为独立、受checkpoint contract约束的字段。**不能为启用quantile而切进失败的absolute-base V2，也不能给旧native checkpoint换统计量。** 本次未回改这些历史语义。

## 5. P1：没有验证语言/杯子条件与恢复，不是缺一个奖励就算实现错

- 数据只有一个任务、一个固定prompt；不能从这16条拟合证明模型根据不同语言目标做决策，也不能据此断言它完全不使用language。单任务复现允许固定prompt，这不自动构成配方错误，不要求为“目标意识”混入别的任务；重点是补诊断/验收。
- 使用官方flow-matching action loss，不包含额外抓杯reward/成功分类器。BC能够学抓取，这并不是漏接任务reward的代码bug。
- 当前数据来自成功来源轨迹；未专门采集/验收空夹后恢复分布。不能宣称策略已学会失败识别/重试，也不能假定成功轨迹绝无局部纠偏。若以后增加恢复数据/RL，应另列扩展，不能取代原论文对照。
- 预测50/执行10/重新看实时观测是真正receding-horizon控制，不是整条专家轨迹回放。但作者实际execution chunk/节奏、state/action约定仍需核实；当前10/50不能当已验证论文设置。
- Source-action replay成功只能证明该回合的native执行接口可复现goal（assisted），不是π₀或physical成功。真实v3来源条件仍无杯接触/抓起，这个验收结论不变。

## 6. 额外控制干预与一处分析口径修正

### 控制器不能只看HDF5初始metadata

HDF5原始robot base config写velocity，但`EnvOmniGibson`在baseline=None时reload为position Holonomic控制，position分支执行local displacement。评估已有runtime position断言且source replay通过。因此**不是已发现velocity/delta搞反**，也不能据旧meta擅自改成velocity。

同一wrapper将gripper reload为binary，input[0,1]、output[-1,1]、lower方向；源代码推导有效close阈值是raw<0.5，而旧forensics仅统计raw<0。

- v3原始负值1146步；按静态控制契约推断effective close1147步，first onset仍step1829/60.97s，run数仍2。
- 两个旧30k报告的计数/first onset没有变化。
- 这是**分析口径的已确认缺口**，不是新“抓不到杯子的根因”。旧trace没有保存完整controller实例字段，因此明确写source-inferred，不能伪称旧runtime值已实测。

已保留raw-negative字段并增加effective字段/证据范围；未来eval只读保存实际motor contract（含source SHA、未知错误），不改变动作/控制器，不作为模型输入或验收新门槛。

### train-range clipping不是原始网络输出

本项目还把raw output限制到训练数据的逐维范围，不能把clipped目标叫raw prediction，也未确认是作者同设定。实际v3：any-action clip1863/3000，left gripper1599、left arm282、right arm199，base只有59（均yaw）；同一步可跨多维，不能直接相加。

二值夹爪越过±1的clip不改变其open/close类别；大量总clip不等于大量底盘饱和。左臂最大裁剪约0.0867rad等值得做独立执行对照，但当前数据不足以说移除它就会抓杯。**本次未取消任何限幅或降低.98 root-planarity安全阈值。**

## 7. 已排除/仍未排除

- 这条失败轨迹21D proprio所有元素|z|>10计数为0，最大|z|8.161；没有看到“小方差state维度归一化数值爆炸”的证据。单变量范围正常不代表图像/姿态组合没有分布偏移。
- serving将参数恢复为BF16，而训练只把frozen参数存为BF16，trainable参数FP32；**模型前向本身也配置BF16，不能说训练全FP32、推理才降精度**。是否有实质精度影响尚未做受控对照，不能当既定bug。
- 当前相机初始同步已验证，离线pipeline一致不排除后续runtime视觉/状态分布问题。
- 视觉遗忘、chunk模式切换、关键接近/夹爪时序、作者state/action预处理差异仍是待验证方向，不编造唯一根因。

## 8. 下一正式训练前的优先事项

1. 保留两卡当前生成，不盲目再续原16-demo batch1权重。
2. 设计同官方base初始化、同数据/seed/batch/norm/LR、仅vision freeze不同的A/B；不要同时改batch64/quantile再把效果归因freeze，也不要冻结已改过的v3视觉后假装恢复了base。
3. 补图像/指令离线敏感性与真正R1未微调闭环对照。输入变化导致输出变化也不等于正确语义；仍要用真实抓取/运输门槛验证。
4. 用新dataset/config/contract版本独立对齐norm、动作/state语义、execution、cleaning/split与多episode评测。作者未知项明确标注；不用规则/planner/privileged输入伪装π₀成功。

本地130项：101通过、29数值依赖skip；真实CPU loader96窗口、两个paired模型推理、partial restore等实际验证均在服务器完成。两张GPU只保留原官方数据生成；这次没有启动新梯度训练或新模拟器评估。
