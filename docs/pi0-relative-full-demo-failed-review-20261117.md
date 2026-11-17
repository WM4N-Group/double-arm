# 新相对关节完整demo诊断：final19999闭环失败及证据分级（2026-11-17）

## 结论先行

**18:13完成的 TRAIN/demo_0 + AUTHOR-ASSISTED 闭环仍失败。** 这是新`pi0_r1_relative_full_demo_v1 /19999`的结果，不是旧native30k或coverage v3录像。20k真实更新、最终保存、CPU独立重载均通过，但不能据此宣称学会抓杯。后续TRAIN/physical及heldout/physical按原门禁没有运行。

当前证据将故障定位在**到达杯子/接近定位的闭环控制环节，早于有效抓取接触**；不能把它归咎于“夹持摩擦不足”。离线抓取窗口拟合和闭环稳定性仍有明显差距；尚未证明唯一训练超参或实现错误是根因，不再盲加90k训练。

## 1. 对象、完成门禁与范围

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-full-demo-20261117-154317/campaign
  training-report.json
  checkpoint-reload.json
  train-author-diagnostic/status.json
  train-author-diagnostic/physical-rollout/result.json
  train-author-diagnostic/physical-rollout/demo_0-trace.npz
  failed-rollout-forensics.json
```

- 一条完整TRAIN/demo0，1714帧，50%完整轨迹抽样+50%九个抓取转折窗口；official pi0_base、fresh AdamW、原π₀rank32，batch1、mean/std。相对关节+关节norm的新诊断配方，不是论文1000条/batch64/quantile训练。
- 17:23 trainer exit0，实际optimizer/AdamW/LR counts全部20000；final19999已commit。独立CPU重载验证完整3,315,905,296个参数的bits/dtype/hash及saved norm一致，四个真实TRAIN输入输出50×21有限。
- 参数SHA256 `ffe08f59577357a9523ce148f13555960fcdb20e3f8150216986de830b78180b`；norm SHA256 `77f6443649bf3468ebd164f27cb21f01898f89f42f8fa27faf50c89f4bd10fc5`。
- CPU/GPU部分采样有约0.0043–0.0072的最大单坐标数值差异，涉及不同物理单位的21维native输出，**不是统一rad误差，也不要求跨backend逐位相同**。保存参数逐位一致与采样数值差异应分开报告。
- 真实head+双wrist、原21D proprio、instruction；teacher_action_reads=0、planner=false、初始state load=1、object pose writes=0。新profile输出已逆变换的native动作，非旧absolute-base服务。

## 2. 已确认的闭环事实

| 指标 | 新final19999 TRAIN/ASSISTED |
|---|---:|
| 实际控制步/推理调用 | 3000 /300 |
| 录像仿真时长 | 100秒，非实时速度声明 |
| 底盘路径 | 8.3803m |
| 任务/持续成功 | false /0帧 |
| 左右grasp ever | false /false |
| 杯子最大抬升/路径 | 0m /0m |
| assisted attachment | 从未触发 |
| 左手指-杯接触已知样本 | 3000 |
| 检测到左手指-杯接触 | 0帧 |
| 左EEF原点到杯质心最近距离 | 0.3380m |
| 左臂重规划边界target最大joint jump P95 | 0.01952rad |
| 专家所有步同类jump P95 | 0.00994rad |

EEF原点距离**不是指尖净空、力或稳定抓取质量**。0次检测接触不能等价为传感器全知，但足以反对把现有证据描述为“夹住了但摩擦不足”。录像关键帧显示手臂没有完成杯子接近；后段相机贴近柜台/橱柜，杯子没有被带走。

初始三相机相对source MAE约5.23/6.56/5.63（0–255），与已修复的reset同步量级一致，不能重新归因于旧“楼梯/门”的第一帧错误。

### clipping与gripper报告的边界

- 2748/3000步至少一个action轴被训练native bounds截断；**不是2748次底盘限幅或抓取尝试**。左臂部分坐标达到约1745/1882/2070次，右臂最大单坐标截断约1.337rad，说明后段动作分布已经严重偏离允许范围。
- trace proposal→action按差异>1e-6计底盘仅8/3000，全部local dy、最大约0.000275m。result里的`clipped_native_base_steps=0`只计**absolute-base转换后的额外base cap**，native分支没有累加它，不能拿这个0说所有原始native底盘输出均未截断。未改任何cap/动作。
- raw negative左gripper命令178步、四段；首段step2355/78.5秒。**这是raw<0，不是已验证的runtime effective-close门槛**。关节width确有闭合与重新张开，不能说整段一直闭合。
- 本次runtime motor snapshot因pinned gripper没有公共`motor_type`属性，错误地丢弃了整组gripper其他可读字段，保存为null+errors。原结果保留；不能伪造该run已记录threshold0.5。修复为未来只读采集可回退到实际`_motor_type`并保留独立mode/limits；不是控制/模型修复，也不追改历史metadata。

## 3. 精度改善尚未转化为闭环恢复：事实与假设

### 已确认

1. 同一新模型在source frame636离线能给出close（raw约-1.02），左臂前10预测RMSE约0.00187rad；但这只是一个noise下的source观测，不能推断在线必然走到该观测。
2. 精确source初始条件下，frame0离线trunk前10RMSE仍约0.00775rad、base XY约0.001868m/步，左臂约0.00186rad。末帧1713左臂约0.04275rad、甚至比当前q不动的离线参照0.02639rad更差。**没有证据说全1714帧已经精确拟合**。
3. 真实初始proprio与source逐组RMSE约1e-7；按相同elapsed frame作描述性对照，到step30的trunk q误差约0.0292rad，step400左臂约0.2934rad。它们在来源关夹frame636之前已出现，非单纯“到了杯口却不肯关夹”。这些不是逐态配对的唯一正确轨迹证明。
4. 新rollout左臂边界jump P95比旧v3约0.108rad小，但仍失败。两轮配方不同，不能把差异隔离归因于一个超参，也不能以平滑度代替成功。

### 待验证假设

- 小的joint residual偏置通过relative anchor反馈被不断带到下一chunk，视觉/状态纠偏不足，最终越界；这是**反馈增益/分布漂移假设**，不是已证明的相对编码实现bug。
- 单demo完整段拟合、50%窗口偏采样、初期trunk/base精度、Gaussian采样噪声和观测分布变化可能共同促成漂移。不能把“加训练数据”“冻视觉”“取消clip”任一项直接称为已证实修复。
- 新profile实际在线inverse/输入与离线source行为的差异仍需定量核查。CPU重载exact和既有roundtrip降低了“保存时丢失权重/明显轴错位”的可能，不自动排除所有serving问题。

## 4. 此轮失败视频交付

```text
/home/yuanyue/double-arm/outputs/demo-review/relative-full-demo-19999-train-assisted-FAILED-20261117/demo.mp4
```

- 新checkpoint的完整100秒压缩FAILED review，960×508/H264/15fps/1500帧，8,455,815字节。
- 视频SHA256：`e23af3dcbd28a0694e40101ece2f0153b1d5b2d1ad0297984f3fe630171a2ced`。
- review JSON SHA256：`ca89cbfa2894aea4e67649b92c51abb9bf12aaf597a9840fe842825aeb64056f`。
- trace SHA256：`893ce0a59c899e4b5470e085e8e96e9355853f60bd9b436113beb1d34625e852`（仅留服务器）。
- 服务器`ffmpeg -v error ... -f null -`完整解码通过；本地视频/JSON/接触图SHA与服务器一致，已目视检查2/20/40/60/80/98秒及18/21.2/78.5/85秒事件图。本地没有ffmpeg，**没有声称本地重复解码通过**。
- 所有录像仅本地忽略目录，保留restricted assets来源，不提交/公开上传。仅取小型review，未下原始RGB、数据集或权重。

## 5. 下一步：已启动受控离线反馈诊断，不加训

原workflow/controller和子进程已退出、GPU1无compute进程、UUID锁可用后，于18:28启动：

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-feedback-20261117-182807/campaign
runs/latest-pi0-relative-feedback-probe.txt
wrapper202125 / probe202130（以实际指针为准）
```

同一新checkpoint/profile，21个真实TRAIN source帧×3Gaussian noises、真实在线初始观测×3同noise配对；frame0/636另做同RGB/noise的左右方向0.02rad左臂q扰动，合计70次离线采样。扰动是刻意不一致的反事实输入，**只测输出敏感性，不进模拟器、不用heldout、不当成功控制器**。有25分钟watchdog和独占UUID锁，更新GPU1/map指针并保存源码hash/依赖状态；GPU0原生成不变。

截至18:29仍在加载，未得到最终probe结论。模型/标签/norm/旧checkpoint/runtime四个契约源码未改；没有新训练、checkpoint或模拟器调用。新增反馈算术fixture服务器通过；全本地165项测试122通过/43依赖skip，不能代替真实probe结果。

## 6. 18:35补充：70次离线probe真实完成，得到局部反馈薄弱证据

Probe于**18:30:45完成、exit0**，非仍在loading。21source帧×3noises +3真实initial配对 +4q扰动共70次，预测数组/源码/报告保留服务器；report SHA256 `5d7db388049ae0e88201c52b2bea6c2e01af20cc236359778c667dd73eadde88`。

### 同进程同noise的局部反事实

在frame0、636分别把左臂六个q同时加/减0.02rad，其它state、RGB、noise不变；比较逆变换后的前10个**native absolute joint target**均值变化在扰动方向的投影增益：

| source frame | q扰动 | target响应gain |
|---|---:|---:|
| 0 | -0.02rad | 1.00157 |
| 0 | +0.02rad | 1.00336 |
| 636 | -0.02rad | 1.00001 |
| 636 | +0.02rad | 1.00200 |

这里gain≈1表示输出absolute target几乎原样跟随被改动的anchor，gain≈0才表示保持原absolute target。**直接确认的只是这些反事实下预测residual几乎没有抵消q变化**；支持“相对增量拟合好，但局部反馈校正弱”的机制假设。不是自然状态扰动后的真实机器人实验；RGB与q刻意不一致，不能据此认定所有真实视觉变化下均不纠偏或完全不看图，也不是已证明唯一失败因果。

### 其他对照

- 同noise真实reset初始三相机+proprio vs source0，左臂前10输出RMS差约0.000140–0.000186rad、trunk约0.000333–0.000547rad。相比source预测本身误差，此处的初始观测差异较小；范围仅此起点，不证明整段在线摄像头都与专家一致。
- source636首gripper在3/3 noises均输出约-1.01至-1.02；左臂前10RMSE中位约0.001834rad。source1713中位约0.06118rad，说明不能宣称全demo在所有阶段拟合精准。
- 没有新训练/模拟器调用。下一轮修复必须明确验证**真实观测下的误差纠偏与全轨迹覆盖**，不能只以转折窗口loss/close命中率作为充分条件，也不能静默改oldnorm或取消安全bounds。

### 计算衔接

确认probe进程退出、GPU1无compute进程后，于**18:34**用现有官方流程启动一次有界**5条raw assisted示范**补充（seed20002、最多1shard，非新模型训练）：

```text
/mnt/yuanyue/double-arm/runs/paper-official-aux5-gpu1-20261117-183402/campaign
controller203505 / child203510（权威GPU1/map指针已更新）
```

此时仅进入实际启动/冷加载，verified=0，不能当新5条已完成。先前GPU1两批25条和GPU0原池provenance均保留在registry；本批最终完成后需追加registry并做跨池QA，**不要简单把各worker计数相加**。选择小而有界的补充而非另一整轮训练，保留下一诊断的调度空间；GPU0 seed10004不变。2h总/5400s shard watchdog是挂起上限，不是新预算窗口。原新profile的四个runtime源码hash与权重保持不变。
