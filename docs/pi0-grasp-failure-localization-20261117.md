# π₀抓杯失败定位：训练源观测上的精度与开合转折（2026-11-17）

## 核心结论与边界

用户要求尽快找出“像死轨迹、不知道抓杯”的原因。本次不继续盲目训练，也不只重复已有loader审计：直接检验**正确的训练源观测下，模型是否预测对抓取转折与微小关节修正**，再用单模态/单坐标干预区分依赖。

**已确认的故障位置：final89999在原16条TRAIN的抓取转折点上，开合时机与左臂目标精度仍有系统性问题。** 因此不能把实际失败完全归于heldout泛化、接触摩擦或首次相机不同步。没有发现新的张量错位bug；这也不证明只有一个配方因素导致全部失败。

没有新训练、checkpoint或模拟器调用；无新闭环成功。干预输入均是**离线反事实**，有意构造图像/状态不一致，不可接入在线控制，不可用来声称抓杯成功。

## 实验对象与资源

- 对象：原native/mean-std/rank32 coverage v3 final89999，推理树fingerprint `199d3068d22b5ed6959fe76aafe7065626f4b1277b4d33fd2da748f50c66f5ce`。
- 模型、归一化及三份实际training adapter源码SHA均核对，不改旧配置/标签/adapter。
- 默认正式推理协议：10步去噪、horizon50；比较实际执行的前10步，以及首步。
- 全部用原16 TRAIN；没有heldout调参。源动作仅用于选择诊断帧与事后评分，绝不作为模型输入。
- GPU1原seed20001数据生成、父级最终QA和进程退出均确认完成，卡空闲且锁取得后才启动短时离线推理；固定GPU1 UUID，无跨卡fallback。GPU0原始1000-target生成未重启/打断。
- 共三组新诊断，各有源码快照、SHA、exit status、模型/输入/噪声证据及独立remote目录。原失败视频/权重保持不动。

## 1. 先检查“误差小”是否已经足够精确

旧3帧CPU control的首动作事后重新评分（不是新推理）：TRAIN/demo_0 frame571左臂目标RMSE约0.03184 rad，而专家当步target-current-q仅0.004438 rad；frame1142分别0.007071和0.0003426 rad。这使精度问题值得优先验证，但不是把一个唯一expert动作当作唯一可行任务动作。

随后新的GPU固定Gaussian seed123对照，采样demo_0的0/571/606/636帧。636是专家首次raw gripper<.5（本固定binary controller的effective-close阈值，见下述证据限定），不是按模型失败视频手工挑的某个“看起来不好”的时刻。

| demo_0 frame | 首步左臂目标RMSE(rad) | 专家首步target-current-q RMS(rad) | 前10步模型目标RMSE(rad) | 前10步保持当前q的RMSE(rad) |
|---|---:|---:|---:|---:|
|0|0.018942|0|0.012442|0|
|571|0.023564|0.004438|0.024529|0.010938|
|606|0.026732|0.000460|0.027311|0.039170|
|636|0.035729|0.000394|0.037033|0.006279|

不能说模型在每个位置都不如hold：606的10步预测优于hold。也不能把hold当有效任务策略——它不会完成抓取。它只是证明：**低绝对位置误差并不等于学会正确的微小运动；在关键抓取位置，误差可能超过动作本身。**

## 2. 跨16训练回合、3噪声复核（已完成，exit0）

对每条TRAIN回合取首次关闭frame与15帧之后，分别用固定Gaussian seeds123/124/125推理，合计32个真实观测、96次推理。**不是96个独立episode或任务rollout。** 全部使用真实三路RGB、原21D proprio、固定任务prompt。

同时按旧评估实际native action bounds做事后clip，分别评分raw与clipped；以下为clipped结果，避免把未实际执行的越界值当唯一证据：

| 指标 | 首次关闭位置 | 15帧后（source30Hz下约0.5s） |
|---|---:|---:|
|推理次数|48（16观测×3noise）|48|
|专家首步均为close|48/48|48/48|
|模型首步仍open|**44/48**|0/48|
|三个noise均首步不一致的回合数|13/16|0/16|
|至少两个noise首步不一致的回合数|15/16|0/16|
|前10步模型close命令数|104/480|480/480|
|前10步专家close命令数|480/480|480/480|
|首步左臂目标RMSE中位数(rad)|**0.026072**|**0.029959**|
|专家首步target-current-q RMS中位数(rad)|0.000370|0.002217|
|首步模型目标RMSE比hold差的采样数|45/48|48/48|
|前10步模型目标RMSE比hold差的采样数|48/48|48/48|

raw与clipped的这些聚合结果一致。不能据此称“从不发起关闭”：首次位置的480个chunk坐标仍有104个close。能说的是**关闭转折显著延迟/不稳定**，且不是只中了一个不好的随机噪声。

单凭专家标签不一致不能证明所有替代时机都物理失败，也不能直接把关节角误差换算成指尖厘米误差。但与真实v3闭环3000步无已检测手指/杯接触、无抓取的独立结果合看，精度/转折问题是强于“只是摩擦差”的定位证据。

## 3. 图像、proprio和指令是否真的被用到？

第一组32次反事实推理，固定同一个模型与Gaussian噪声：原输入、原输入重复、换另一source frame三图、全灰图、仅灰head、仅灰双wrist、换另一帧proprio、改成不抓杯指令。

- 四个观测的原输入重复均输出完全一致（max abs difference0）。
- 真实换三图使前10步左臂输出RMS变化约0.0207–0.0355 rad；仅灰head或双wrist也会改变输出。
- 在636位置，仅灰head还把预测首步从open改为close，显示相机条件会影响抓取命令；**这是OOD反事实，不是移除head的有效修复**。
- 把整段proprio换成另一姿态，输出变化更大，但输入pose差异本来就很大；不能把输出变化量之比直接解释为“proprio贡献百分比”。
- 不抓杯prompt的影响较小，但原训练只有一条指令，这不是完整语言语义能力测试。

所以“完全不看图、只播放固定时间轨迹”不符合这些干预结果；“已经正确视觉定位/理解目标/保留预训练语义”同样尚未被证明。

## 4. 单坐标夹爪宽度干预

另一次独立、明确的离线counterfactual：固定RGB、所有关节/底盘proprio、prompt和seed123，只在state[19]交换同回合两个诊断时刻的左夹爪实测width；同时设置仅改state[20]右width的负对照。

源观测中左width首次close时约0.1000m，15帧后约0.0111–0.0149m。源回合间明确有真实“已张开/已关闭”的观测差异，不能仅由标签推定。

该组也已完成96次推理（16回合×2位置×3变体，**一个seed，不是三个seed**），exit0。RGB、其它proprio、prompt与Gaussian noise在每个triplet内固定：

| 条件 | 首次关闭位置：首动作close | 同位置：前10步close | +15位置：首动作close |
|---|---:|---:|---:|
|原输入|1/16|68/160|16/16|
|只交换左width state[19]|**16/16**|156/160|16/16|
|只改右width state[20]负对照|4/16|43/160|16/16|

- 在发起抓取的位置，只有左width从“开”改为“已关闭”，就使15/16条回合的首命令open→close，表明**左width确实对发起判断有强的因果影响**。对应图像、其它机器人姿态完全没变。
- 但反向控制同样重要：+15位置把左width伪改回“开”，模型仍16/16预测close。**因此不能把模型描述为“只复制当前width、完全不会看其它条件”。** “简单状态复制”假设被这个结果限制，合理定位是过渡区的状态/视觉融合与触发条件没学稳。
- 右width负对照也导致5/16首命令分类翻转（其中一条close→open），并非完全无影响。不能声称只有左width起作用；不一致输入的OOD/cross-channel敏感性仍需考虑。
- 这仅表明单坐标对该固定模型的输出有因果影响，**不证明去掉width就能抓杯**，更不允许将未来实测width替换到在线输入。

额外复现性限定：两次GPU进程的32组原输入与noise hashes逐项匹配，首动作gripper分类32/32一致，但整段50步输出并非bit-exact：最大差0.360512出现在中间gripper坐标，首动作最大差0.011610，左臂全段差异RMS约0.001970 rad。跨进程数值差异原因尚未定位，不冒称完全确定性。**上述干预因果对照均在各自同一个进程内比较**；第一组原输入重复四次max diff0，不用跨进程输出相减推断width因果。

## 5. 对训练策略的直接影响

1. **不能再只用全动作平均loss/三个均匀帧离线MSE当“抓取已学好”的依据。** 新诊断单独报告抓取首次转折、已关闭后的保持、左臂微小目标误差、同状态hold对照，且任务验收仍必须真实闭环。
2. **不能把同一16-demo/batch1/mean-std/warm-start配方再延长90k当主要修复。** 已有来源条件错误必须先有受控对照解释。
3. “整块视觉编码器414.8M参数也在训练”、小数据、绝对位置监督的精度需求、转折数据覆盖/目标混淆、优化/采样，是待区分的配方因素；本次不把其中某个直接命名为已证实的唯一根因。没有证据给视觉变化扣上“已确定灾难性遗忘”的结论。
4. 下一轮诊断应在新显式版本上，先过**单demo/抓取过渡段可拟合性**及同噪声分项精度检查；冻结视觉A/B必须同官方base、同数据和其他配方，只改freeze。不把该过渡段诊断、额外采样或损失权重称为论文原配方。
5. 后续正式1000-demo/90-10/quantile/effective-batch64仍需独立版本对齐；不能改变旧checkpoint norm，或使用已失败absolute-base V2偷换native语义。

## 6. 证据位置

```text
/mnt/yuanyue/double-arm/runs/pi0-failure-mechanisms-20261117-125842/
  probe/report.json
  probe/status.json
  preexisting-control-precision.json
/mnt/yuanyue/double-arm/runs/pi0-grasp-onset-replication-20261117-130327/
  probe/report.json
  probe/summary.json
  probe/status.json
/mnt/yuanyue/double-arm/runs/pi0-gripper-width-counterfactual-20261117-130734/
  probe/report.json
  source-onset-widths.json
```

三份report SHA256：

- `37ec0bd0e923ef99e7feba13b0471dba43e0ce2434ecd360b2b11e561b8ed325`
- `00f4820779c598fd4c9a22e92870d838baca221d33b5a80c2f70e557bd179639`
- `f7c9ebbfd8d6339a4ff0937b078ea9901934201e198374db1414798a3eddfc95`

相关代码：`scripts/probe_pi0_failure_mechanisms.py`、`scripts/probe_pi0_grasp_onset.py`、`scripts/summarize_pi0_grasp_onset.py`。各次执行目录保存当时源码，后续新增width模式不会覆盖前一次使用的快照或历史报告。

close阈值来自已审过的固定R1 binary input[0,1]→output[-1,1]、noninverted约定，**不是本次新模拟器运行捕获的controller实例**。这些离线探针不创建模拟器；实际runtime metadata留待下一次真实评估只读记录。

## 测试与结束状态

- 本地138项测试：104 passed /34依赖skip；服务器8项新增数值/聚合测试全部通过。
- 三组真实推理全部exit0，分别32/96/96次，没有训练更新或模拟器动作；GPU1已释放。GPU0原官方生成进程继续。
- 没有新policy rollout，所以没有拿旧视频充当本次诊断的新demo；当前最新闭环仍是FAILED的coverage v3 TRAIN/ASSISTED。
