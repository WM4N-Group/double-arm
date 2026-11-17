# π₀相对关节完整demo诊断：持久化权重与自动闭环（2026-11-17）

## 当前状态与范围

这是上一轮9窗口编码对照之后的**新完整轨迹诊断**，不是再把旧v3延长90k，也不是论文1000-demo训练。16:05已真实完成1875/20000个optimizer updates，梯度/loss有限；16:11已达3225步，约3.56 updates/sec。没有新任务成功，正式新checkpoint保存/CPU重载和闭环仍待执行，不能从loss推定抓杯。

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-full-demo-20261117-154317/campaign
controller193740 / trainer193743（以权威指针为准）
runs/latest-pi0-relative-full-demo-workflow.txt
runs/latest-gpu1-job.json
runs/latest-multi-gpu-jobs.json
```

GPU1 UUID固定，GPU0原始1000目标raw generation不变。**GPU1所有权覆盖训练→保存→CPU重载→自动评估的整个workflow，卡看起来空闲/CPU加载也不能插入其它GPU任务。** 旧`latest-pi0-relative-joint-fit.txt`、`latest-pi0-single-demo-fit.txt`均只是已完成小诊断，不再代表当前卡空闲。

## 17:24更新：20k真实更新与最终保存完成，独立重载进行中

- 2026-11-17 **17:23:35 CST**，训练报告实际进入`completed_full_demo_training_restore_required`；父控制器于17:23:37确认trainer **exit0**，切换到`relative_full_demo_checkpoint_CPU_reload`。不是只按loss日志推测完成。
- 真实optimizer updates、AdamW count、LR count均为 **20000**；最终`19999/_CHECKPOINT_METADATA`及`assets/r1-relative-profile-contract.json`实际存在。4999/9999/14999中间点保留。
- 最终完整参数SHA256：`ffe08f59577357a9523ce148f13555960fcdb20e3f8150216986de830b78180b`；保存dtype计数FP32 **495,908,624**、BF16 **2,819,996,672**。norm SHA256：`77f6443649bf3468ebd164f27cb21f01898f89f42f8fa27faf50c89f4bd10fc5`。这是保存时报告/contract证据，**独立CPU重载校验仍未完成**。
- 独立重载PID199365，输出应为本campaign的`checkpoint-reload.json`；必须实际达到`cpu_reload_parameters_exact_actions_finite_NOT_task_success`才可进入自动闭环。GPU1仍归整个workflow持有，CPU重载阶段不得插队。
- 训练末步loss约0.001564、grad norm约0.29886，均有限，**不代表抓取成功**。当前无本checkpoint新闭环结果或失败视频；旧v3失败仍是最近已完成的learned-policy任务结果。`training_is_diagnostic=true / formal_training_completed=false`保持不变。
- GPU0原workflow继续，worker内4个shard共100条raw assisted episodes/174,102帧已finalized，进入seed10004；这不是新的跨worker全局去重结果、训练split、物理或策略成功。不重启生成器。

## 1. 训练配方：明确的新实验

- 原版π₀，官方pi0_base初始化、fresh AdamW，rank32，batch1。不是旧v3或小窗口拟合权重warm start。
- 唯一训练episode：原TRAIN/demo_0完整1714帧；原16 TRAIN+1 heldout数据文件、split、旧norm、旧checkpoint完全不变。
- 20000次真正optimizer updates；50%从完整1714帧打乱循环抽样，另50%从606/621/631/635/636/637/641/651/666九个抓取窗口打乱循环抽样。所有完整帧均覆盖，采样frame/index仅元数据，绝不送入模型。
- 相对关节编码与上一轮相同：native base三坐标、gripper两坐标/mean/std保持旧值；16个joint targets减去**当前chunk输入q**，inverse只加回同一个anchor。原21D proprio、state mean/std、真实三相机和instruction不变。
- 关节action mean/std仍使用已验证的全16 TRAIN/horizon50派生统计，保存在新checkpoint asset id `momagen/r1_relative_full_demo_v1`。没有heldout统计、新raw pool、quantile或作者编码宣称。
- 全视觉可训练（495,908,624 trainables）；不额外freeze，也不改旧模型。上轮同seed小样本A/B没有支持“冻视觉就修复”，但这也不是大数据/泛化结论。
- 官方flow loss、global clip/AdamW；warmup1000、peak2.5e-5、decay30000、end2.5e-6。相对9窗口诊断还改变了数据范围、混合采样和LR，因此它是完整诊断recipe，不是隔离一个超参的因果A/B。
- 全部1714帧的native动作逆变换检查实际通过，最大误差1.19e-7；11个unused padded state/action channels都为0。完整初始model/dtype hash仍与小诊断相同`a3ee6c...28e1b`。
- 图片/标签缓存仅在服务器HOST RAM，GPU每步仅一个batch，避免把全1714帧的float RGB常驻24GB显存。原source HDF5直接只读，使用同R1/官方预处理；没有逐步恢复状态或读取未来RGB。

## 2. 不再丢失训练后的模型：真正checkpoint

每5000次update调用**固定官方`openpi.training.checkpoints.save_state`**：

```text
runs/pi0-checkpoints/pi0_r1_relative_full_demo_v1/
  pi0-relative-full-demo-20261117-154317/{4999,9999,14999,19999}/
```

每个保存点等待实际Orbax commit，验证`_CHECKPOINT_METADATA`、实际AdamW/LR/step计数、保存norm数组，再写**独立**`assets/r1-relative-profile-contract.json`；本新目录的中间点都保留，不清理旧权重。

- contract明确新profile/model编码与最终native motor编码、原21D输入、asset id/实际norm SHA、参数原dtype计数、完整parameter-bit hash、原始source、runtime代码SHA、真实更新数和诊断范围。
- 4999/9999/14999 contract明确未完成全训练；新service只允许已提交19999且确有20000 updates的final contract，不评估半成品。
- 正常旧native/absolute-base contract不能冒充此profile；旧native服务遇到新contract会立即拒绝，而不是按legacy fallback加载。
- 完成训练仍`formal_training_completed=false / training_is_diagnostic=true / paper_reproduction_complete=false`；20k是真实梯度诊断，不是论文规模声明。
- CPU真实Tiny NNX实际1步optimizer更新→官方Orbax save→norm/混合FP32+BF16参数restore逐值相等的fixture已通过。**这不是3.3B参数保存通过的替代证据**；大模型真实保存以本run实际报告为准。

## 3. 独立CPU重载门禁

训练进程完全退出后，`verify_pi0_relative_full_checkpoint.py`在CUDA空可见/JAX CPU中：

1. 验证new profile、commit、norm SHA与runtime source hashes。
2. 使用固定官方`restore_params(dtype=None)`保留原保存的混合dtype，而非默默把全部trainables压成BF16。
3. 实际加载完整模型；完整参数bits/dtype/count hash必须与训练时、contract一致（3,315,905,296个元素）。
4. 在4个真实source TRAIN观测上采样native50×21动作并检查有限性，记录CPU与GPU内存采样的数值差异（如有）。

CPU/GPU sampling不要求逐bit相同，不能据此宣称完全数值等价或实时性。这里的硬门禁是参数/保存norm正确、变换profile匹配与有效动作输出；不是task acceptance。旧小诊断都没有保存checkpoint，不会伪造它们的重载结果。

## 4. 自动真实闭环，严格分条件

CPU重载通过后，GPU1用于Isaac，模型CPU推理；复用现有真正在线RGB、原21D proprio、instruction的原版π₀评估。

1. **TRAIN/demo0 + configured AUTHOR-ASSISTED**。
2. 仅当该真实task result成功，执行 **同TRAIN/demo0 + PHYSICAL**。
3. 仅当TRAIN PHYSICAL成功，执行原 **heldout/validation demo0 + PHYSICAL**（完整1624帧来源，未参加训练/统计）。

- 固定sync20、执行前10动作/每次预测50、10去噪/Gaussian；不改native motor bounds、`.98` root-planarity、杯特定抓取/抬升/运输/稳定goal门槛。
- 新service通过`--policy-profile pi0_r1_relative_full_demo_v1`明确选择；RPC输出是**已经inverse的native21**，simulator不再加q、不用world-base V2转换。ready/RPC profile必须逐项匹配，拒绝错服务/错模型/无声fallback。
- 新profile service不打开source HDF5、teacher overlays或dataset；warmup在外层离线做，正式控制始终用模拟器真实三相机，从不替换source RGB。
- 读EEF/contact/runtime motor metadata仅为遥测，不加入VLA，也不是新验收门槛。未知NaN/-1/errors仍保留。
- 新review仍按结果标FAILED/ABORTED；source train-assisted成功不叫physical，TRAIN physical也不叫heldout，视频时间是仿真时间不是实时控制速度。checkpoint、profile、单demo诊断范围与`formal_training_completed=false`写入review。
- 失败即停止自动后续条件，不用planner、规则抓取、MLP/近邻或teacher挽救。将交付这个新checkpoint对应的小型压缩review及证据分级分析，不能拿旧v3视频替代。

## 5. 边界与watchdog

- 训练最多3h、CPU重载最多40min、每个评估最多3h、总体12h是防挂死watchdog，不是新的用户计算授权窗口或交付承诺。SIGTERM会清理自己的child process groups；不杀别人的任务或跨GPU fallback。
- 模型/标签/normalization/profile代码已被冻结在plan runtime hashes中，训练中不可修改；评估壳/视频只读元数据独立保存源码快照。
- 训练失败不使用部分checkpoint直接推进任务测试；首次真实大模型checkpoint save或CPU重载若失败，保留证据，不把CPU fixture/安装成功/离线分数当成功。
- 1个episode+混合过采样不是1000 demos/90-10/batch64/50k quantile论文训练；没有保证heldout或physical成功。

测试：本地完整162项（120通过、42依赖skip）；服务器新增profile/routing6项通过，含真正Tiny checkpoint混合dtype往返以及legacy拒绝new-contract的行为测试。真实大模型目前仅验证了已完成的训练更新，后续checkpoint/reload/task仍未完成。


16:19追加：已实际完成5150/20000更新，首个真实3.3B参数checkpoint4999在5000更新后完成Orbax commit、保存norm数组与实际AdamW/LR计数验证。这不是Tiny fixture推断。该中间contract training_completed=false，服务仍拒绝加载；final19999、CPU完整重载和任务结果仍未完成，不提前报成功。
