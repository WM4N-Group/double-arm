# 单demo抓取过渡拟合：严格视觉冻结A/B结果（2026-11-17）

## 结论

**当前original π₀ + R1输入/动作接口能够学会在这组训练观测上发起关闭；额外冻结视觉并没有解决剩余的关节目标精度问题。** 这是小样本训练可拟合性结论，不是视觉语义能力、全16条泛化、闭环抓杯或物理任务成功。

两分支最终采样和父控制器均已完成，exit0。只用同一TRAIN/demo_0的9个真实窗口，两个分支各2000个实际optimizer updates、54次初始/最终离线采样。**3个采样噪声不是3个训练seed；本次仅一个训练seed42。**

## 1. 受控条件确实成立

- 官方pi0_base/fresh optimizer，两个分支都不是v3 warm start。
- 相同原21D proprio、真实head+双wrist RGB、instruction、native absolute joint labels、旧TRAIN mean/std、rank32、batch1、训练增广、官方flow loss/clip/AdamW和LR（诊断warmup100/总2000）。
- 9个source frames为606/621/631/635/636/637/641/651/666；每个保留原50步未来监督动作，不给模型frame/index/phase。
- 完整初始参数与dtype hash两支逐bit相同：`a3ee6c232ac89b0e4e5437e326dc5af9234f3a4436cb4cc0b138fbe605e28e1b`。
- batch、raw model-input、frame schedule、norm、LR、seed全部匹配。54组初始/最终采样input/noise对应核验通过。
- 存储cast使用同一个原始freeze规则；B图像虽然没有梯度，仍保持与A相同的FP32初始存储，避免把freeze与BF16压缩混为一个因素。
- A trainable495,908,624，B trainable81,104,928。完整视觉分区414,803,696参数hash：A实际变化、B最终与初始完全一致，不只是检查两条小向量。
- 原始权重相同不等于跨进程推理逐bit相同：两支初始预测全50步最大差约0.007603；报告保留该数值差异，不夸称完全确定性。

## 2. 结果

全部为唯一expert目标的离线模仿评分，native动作坐标。hold只是一种离线精度参照，不是任务控制器；偏离唯一expert并非必然物理无效。

| 指标 | A：原规则，视觉可训练 | B：额外冻结视觉梯度 |
|---|---:|---:|
|实际optimizer / AdamW / LR计数|2000 /2000 /2000|2000 /2000 /2000|
|初始首夹爪命令匹配|15/27|15/27|
|最终首夹爪命令匹配|**26/27**|**25/27**|
|首次close frame636正确发起关闭|3/3 noises|3/3 noises|
|前一帧635正确保持open|2/3|1/3|
|已关闭15帧后651正确close|3/3|3/3|
|9窗口首左臂目标RMSE中位数：初始(rad)|0.89495|0.89483|
|同指标：最终(rad)|**0.01512**|**0.01866**|
|首次close首左臂目标RMSE中位数(rad)|0.01062|0.01000|
|前10步左臂目标RMSE中位数(rad)|0.01667|0.02105|
|前10步比当前q hold误差还大的采样|19/27|21/27|
|最后一随机训练batch的flow loss|0.001059|0.001392|

不能将最后单batch loss直接视为完整数据误差或任务得分。A/B整体差异也不能从一个seed推广到所有训练规模；这里没有证据支持“冻视觉就能修复原失败”。两支都能学会关闭，且精度仍是残留问题。

这与v3的44/48首步保持open并不矛盾：v3是16条全回合随机窗口、旧30k warm start+90k，而这里是官方base、9窗口高重复、短warmup的新诊断配方。**不能把跨实验差异全部归因于过采样某一个因素**。能排除的是“当前接口/原架构根本无法学会关闭”的强说法；也不能证明π₀已有完整任务意图理解。

## 3. 一个新收尾bug，已修复，不混为原策略失败原因

先前allocator90尝试的A在13:50真正完成2000次更新，flow loss约0.000832，参数/AdamW/LR计数均完成。但新诊断收尾为了释放optimizer，试图构造`TrainState(opt_state=None)`，被固定运行时类型检查拒绝，**最终采样缺失，B未运行**。这不是旧v3训练或机器人失败原因。

旧报告/完整异常保留于`runs/pi0-single-demo-freeze-ab-allocator90-20261117-133856/campaign`。没有保存checkpoint，进程退出后不能从loss“补出”最终动作或声称A/B完成。

修复做法：先保留普通`params, model_def, verified_step`推理payload，再删除TrainState释放optimizer，不构造非法状态；CPU真实NNX初始化→删除TrainState→模型合并/JIT推理的回归夹具通过。初始/最终小型NPZ预测在各阶段结束即落盘。

由于旧权重不可恢复，新版本通过显式`--repair-postfit-report`验明该已完成训练/仅缺收尾的报告，记录重复来源，再从同一官方base**重新独立训练**，不是恢复旧optimizer、覆盖旧运行或隐式追加2000步。新两支最终均完成，不能把旧失败attempt算进这对因果A/B。

## 4. 观测是否在关闭边界完全重复？

另一个CPU-only审计比较16回合的close-1与close：

- 16对的224分辨率三路预处理图像均不是逐像素完全相同。
- 仅部分restore v3的state_proj33792个元素，镜像默认BF16参数restore、FP32输入线性投影、Gemma BF16 token cast；1024元素state token有467–978个变化，中位851.5，没有完全相同对。
- 这是CPU局部stem镜像，不是GPU激活捕获，也没有跑SigLip/完整VLA。只能排除本次检查的简单“输入字节/该state token完全一样却标签冲突”解释，不证明视觉语义足够或真实策略必然区分得准。

证据：`runs/pi0-transition-observation-audit-20261117-143150`，没有训练或模拟器调用。

## 5. 下一项已启动：关节相对目标 + 新关节action norm（新独立诊断）

精度问题尚未解决，所以优先检验**绝对关节目标/归一化尺度是否掩盖微小修正**，而不是再随意加90k或直接冻结视觉：

- 新profile`relative_joints_meanstd_v1`，同官方base/seed42/9窗口/2000 updates，保持A的视觉可训练条件。
- 只将trunk+左右臂future targets表示为相对**当前chunk输入关节状态**的差值，inverse一次性加回该同一chunk anchor。不是逐步累计，不用未来状态/teacher/物体位姿/阶段或planner。
- **原21D proprio及其norm、真实三图、instruction、native base三坐标、gripper两坐标及其norm全部保持不变。** 特别处理right action[14:20]对应state[13:19]，不把left width state[19]误当右关节。
- 原source动作/HDF5/LeRobot/旧norm不修改。新动作norm只在新runs下，16个关节坐标使用全部16 TRAIN的episode-clamped horizon50差值统计（每坐标1,382,850个值）；base/gripper stats逐值保留旧值。没有heldout、新raw pool或quantile宣称。
- 这是**关节编码+关节action norm的联合新recipe变体**，不是声称只隔离了其中一个因素，也不是已确认的论文action编码。
- CPU实际96窗口验证三相机/proprio/token/masks完全相同，base/gripper归一化标签完全相同，逆变换回native动作最大误差约1.19e-7。worker9窗口还重复检查，完整模型初始hash与绝对目标基线一致。
- 新profile不得使用旧native checkpoint/normalization/serving contract解释。旧三份training adapter源码SHA保持一致；不改变旧训练/评估语义。
- 当前运行：`runs/pi0-relative-joint-fit-v1-20261117-150043/campaign`；GPU1/combined map和`latest-pi0-relative-joint-fit.txt`指向它。`latest-pi0-single-demo-fit.txt`保留已完成A/B，不能用它误判卡空闲。
- 15:03时初始27次采样完成、正编译真实梯度；新增2000步及精度改善**尚未证实**。不把预检或更接近当前q的未训练动作当任务成功，最终仍需与hold参照和真实闭环区分。

## 溯源

完成的A/B：

```text
/mnt/yuanyue/double-arm/runs/pi0-single-demo-freeze-ab-finaleval-v2-20261117-142606/campaign
```

- A report SHA：`8461edaee27f5dbde4fdc8c13028cc2bf6179140095238aa04b8da79799e8896`
- B report SHA：`b6a394165f0ec1ac13c842192e3fb92fc1d1bbf3065c1875a51ade3d505431fb`
- strict summary SHA：`6a3f72cb860d0aea09c74b857047c9915f3eb2ffe176e9cbe70e2d592c89bb9e`
- summary脚本拒绝仅完成2000updates但缺最终采样、source/norm/初值不匹配或视觉冻结未生效的记录；不伪造或外推缺失数据。

相对编码preflight：`runs/pi0-relative-joints-preflight-20261117-145842`，新norm SHA：`a80f36755cb6917deac448cec5de2cecc7e98dbbd537e85fa55ee56bd5982451`。

**本轮没有新的闭环视频或task success。最新真实策略闭环仍是FAILED的v3 TRAIN/ASSISTED。** GPU0原官方生成持续，未重启/中断，raw assisted generation仍不是learned-policy/physical成功。

## 15:13追加：相对关节编码诊断也已完成

新profile父子均exit0，实际2000个optimizer/AdamW/LR更新、54次初始/最终采样完成。所有初始参数/dtype与A/B相同，9窗口、frame顺序、seed、LR、batch、视觉可训练条件一致；输入图像/proprio/token/norm保持一致。唯一有意改变的配方是**16关节action编码+这些关节的action mean/std**，不把两部分拆成各自已证明的原因。

为公平评分，事后从保留NPZ将绝对/相对两支的native预测都转成实际motor使用的float32，再读取source标签评分，不修改原始报告。旧绝对支是在float64反归一化输出上打分，这个小cast差异单独记录；不是重新推理或用teacher控制。

| canonical native float32评分 | 绝对目标A | 相对关节profile |
|---|---:|---:|
|初始首左臂目标RMSE中位数(rad)|0.89495|0.06674|
|最终首左臂目标RMSE中位数(rad)|**0.015117**|**0.002924**|
|最终前10步左臂目标RMSE中位数(rad)|**0.016666**|**0.003785**|
|前10步比hold更差的样本|**19/27**|**3/27**|
|首次关闭frame636首左臂目标RMSE中位数(rad)|0.010616|0.002222|
|首夹爪命令匹配|26/27|26/27|
|首次关闭位置正确发起close|3/3|3/3|

前10步误差约改善4.4倍，首步约5.2倍。**这在同一小样本/seed条件下，确实支持“关节表示与action尺度是控制精度的可改进瓶颈”，比继续盲加步数或唯一归因视觉冻结有更直接的证据。** 不等于已经证明原v3所有闭环失败的唯一根因，也没有证明作者论文使用这个编码。

相对编码初始就把目标锚在输入当前q，因此未经训练也有更小误差；不能把这种结构先验当学会任务。更有用的是训练后27次中仅3次前10步仍比hold更差，说明不只是把当前位置原样返回。仍仅一个TRAIN/demo_0的9个观测，不能代替完整轨迹、heldout、恢复行为或物理成功。

这些动作由训练态mixed-storage参数（trainable FP32/frozen BF16，compute配置BF16）直接采样。尚未保存/导出并重载为完整可服务checkpoint；上线前必须验证新profile的保存/restore精度与明确contract，**不能把相对输出或新norm套到旧native服务**。

完成结果/paired比较位于：

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-joint-fit-v1-20261117-150043/campaign/
  original_vision_trainable/report.json
  original_vision_trainable/offline-predictions.npz
  original_vision_trainable/relative-norm-stats.json
  encoding-comparison.json
  status.json
```

下一步应在**新的、可保存/验证重载的完整demo诊断**检验这个配方，而不是从9个过渡窗口直接启动整任务然后把无导航训练的失败当结论。原版π₀、真实在线三相机/原proprio/instruction、原native motor bounds、assisted/physical区分与最终任务门槛继续保持。

相对profile report SHA：`0423d04ca0ef083eb814531fde232a73c0e461dbbfe7283d753fe265c4fbce25`；paired comparison SHA：`f1f394781a2e7709cb71db4b2f936345372654930da6f6bf2b3a599c5b98a2e5`。本地156项测试115 passed/41依赖skip；服务器本轮新增17项NNX收尾、严格汇总、相对编码/统计与比较夹具均通过，另有真实CPU96-window preflight与16-pair部分stem审计。它们不等于新的物理任务测试。
