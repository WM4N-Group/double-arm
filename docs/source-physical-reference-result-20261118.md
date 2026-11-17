# 原source动作在物理模式下可保持、运输并释放：不是π₀策略成功（2026-11-18）

## 1. 关键结论

修复初始化审计后，**完整1714条原source命令真实执行完毕**，无新模型推理、无planner或逐步状态恢复。本次在physical模式、friction4.0不变、已检查时刻无Python/USD辅助约束的条件下，杯子抬升、运输到水槽，并在末段放开，inside目标连续78帧成立。

**这只是专家动作回放，不是π₀闭环成功，不是论文复现成功或实机标定。** 但它排除了对这条轨迹/这一配置的强说法：“原source只能靠吸附，关闭辅助后必然无法完成”。学得B仍然在37.87s失持并失败，heldout没有运行。

下一排查重点是**B实际命令/抓取姿态/保持过程与可行source之间的偏差**，而非单纯增加摩擦/抓力、取消安全bounds或继续盲目训练。单次source可行不证明其它初态/场景都可行，更不能唯一定位模型失败超参。

## 2. 真实完成证据

```text
/mnt/yuanyue/double-arm/runs/source0-physical-replay-sentinel-v2-20261118-022531/campaign
runs/latest-source0-physical-replay.txt
```

- 实际workflow于**02:36:32 CST**完成，sim exit0、原result stage完成、unaborted、1714实际commands，严格新parent完成门禁通过；不是上次0动作abort/exit0的假完成。
- 原source0空手初态load1，20physics refresh，无额外holding/velocity姿态干预、不逐步恢复；1714动作原样执行，native bounds/.98 planarity/friction未改。source初始body q最大差约5.08e-7rad，初始RGB MAE5.25/6.51/5.67。
- 模型调用0、所有控制来自明确标记的source actions；`learned_policy_task_success=false`。

| 指标 | source动作物理回放 |
|---|---:|
| 实际source命令 | 1714/1714 |
| 最大杯子抬升 | 0.38963m |
| inside goal最长连续帧 | 78 |
| 手指接触区间 | 639–1679 |
| backend grasp主要区间 | 644–660、662–1675，1678–1679 |
| effective close区间 | 636–1679 |
| 最终grasp /左右contact | false /0、0 |
| 最终cup位置XYZ | 5.65134、-1.80336、0.75696m |
| 初始化后显式cup/object pose writes | 0 |

Source在step1680开始张开，最后34步不再报告拿持/接触，inside目标继续成立。**仅观测到约1.13秒的释放后时间**，不是长时间稳定性/实机可靠性证明。相比之下，B辅助模式达到inside但没松手，B物理模式提前掉杯，两种旧结果不被更名。

## 3. 无辅助证据及其边界

- 原source0转换至physical，初始一次、20次refresh后、1714次控制pre/post，总计**3449份**no-assistance检查；trace数组shape `(3449,8)`，全部已知0，无未知/错误。
- 每份包括双手Python constraint/in-hand/freeze以及真实USD stage末端`ag_constraint` prim presence。
- freeze原始None保留，通过精确pinned source SHA合同解释为已知inactive sentinel；没有把robot字段写成False、删除prim或清空辅助对象来让门禁过。
- 这些是**指定边界时刻的检查**，不声称观测了每个内部physics substep；也不声称关节contact force/摩擦已校准。

证据SHA256：

- 原始result：`f4ba9c1ada11f567ad0593df715adffa39f83b395adcbf68c07630a3e13912ca`
- 完整trace：`a8f6b9acec6a6235cccc835b85dcddff5bbc8f932f07ac8cbd94b771b8d93f14`
- 原pinned manipulation源码：`af1e0245078a69a67368be4e70af4f5b79190c7d7dcff4fe9be210da1c39b0ba`

## 4. 与B的只读几何对照（假设线索，不是唯一因果）

`analyze_physical_grip_geometry.py`仅CPU读取两个已完成trace，将杯子质心变换到各自记录的left EEF-local XYZW frame，以各自first30 backend-grasp样本中位数为anchor：

| 描述量 | source物理回放 | B原live physical失败 |
|---|---:|---:|
| first backend grasp step | 644 | 690 |
| first EEF-origin/杯质心距离 | 0.03495m | 0.05074m |
| 拿持报告true期间，质心相对自身early-anchor漂移P50 | 0.00477m | 0.01320m |
| 同上P95 | 0.00497m | 0.01699m |
| 同上max | 0.00559m | 0.02073m |

报告保存在source campaign的`expert-vs-B-geometry.json`。**不是时间/姿态配对实验**；质心位置不是指尖净空，不包含cup orientation、contact normals、force/torque，source也不是唯一合法抓取姿态。数据与B保持时相对移动较大相容，支持继续查抓取接触几何/动作精度/保持动态；不能直接说“偏了X毫米就必掉”或“摩擦完全没有影响”。

## 5. 新参考视频身份明确

```text
/home/yuanyue/double-arm/outputs/demo-review/SOURCE-REPLAY-PHYSICAL-NOT-PI0-20261118/demo.mp4
```

57.134秒/H264/960×536/857帧，3,998,035字节。视频顶/底明确`SOURCE ACTION REPLAY / PHYSICAL / NOT LEARNED PI0`，不冒充模型demo。

- 视频SHA256 `284a332d55f6d02e3ef50b29164704c0c0267d03317b1bf751f5f5047e84da61`
- review SHA256 `4342497180856331d6fb9e67120a065163aee8a7f74adf8373be17e91e6087a2`
- 服务器完整decode通过，本地传输SHA匹配、22/30/38/50/56/57秒关键帧已目视检查；没有声称本地ffmpeg解码。

媒体与reference JSON只在本地忽略目录，原NPZ/数据/权重仍在服务器，未公开上传或提交。

## 6. 已启动下一项：B已执行命令的物理重放（仍不是live π₀）

成功source replay整体退出、GPU1无compute进程且取得UUID锁后，于**03:25**启动：

```text
/mnt/yuanyue/double-arm/runs/recorded-B-physical-replay-20261118-032522/campaign
runs/latest-recorded-B-physical-replay.txt
controller222667 / sim222670 / watchdog222663
```

问题是：**原B失败时记录的3000条实际native命令，脱离模型推理后在相同source0/physical协议中，是否也会失持？**

- 仅重放已完成B trace里的`actions`，验证trace SHA `170050b3e0d0b6fcc59f01c0fc1eb490c66b9117346cc49dade996d5dcd09f3c`。不加载权重，不调用VLA，不调用source专家动作控制，也不做新的标签/模型训练。
- Source HDF5用于初始化/源完整性核对，其中source actions只参与原fingerprint校验，不作为控制流；实际动作来源在结果中明确`recorded_FAILED_B_physical_native_actions`。
- 同source0/load1/20refresh/真实sensor对齐/原native bounds/.98/friction4.0/已验证无辅助边界检查，fresh independent scene。
- 每个输出视频固定`RECORDED B ... NO LIVE POLICY`，不能当新闭环π₀视频或source专家结果。采样控制是固定action tape，**不是用最近邻/规则策略替代学得主线**。
- 若重放也掉杯，有助于表明命令序列在这个物理初始化下本身可重现失持；若重放不掉，则必须检查reset差异/physics随机性/原live时序，而不是继续宣称已经唯一归因于模型精度。任一结果不是单次因果定论。
- 截至03:27仍冷启动、0/3000命令，实际完成/结果尚待验证。40min sim/50min外层watchdog防挂起，GPU0原seed10006不变；无新预算/terms/app automation。

最后global原始池QA仍180/314090；GPU0 worker150/261496是不同统计范围。没有修改B/A训练、norm、profile、安全阈值或旧结果。

## 测试

新recorded-tape completion fixture1项及XY ZW逆旋转/未知拒绝数值2项在服务器CPU通过；全本地192项146通过/46依赖skip。它们不代替实际GPU replay结果；当前主线学得π₀physical结果仍为B失败，heldout未测。

## 04:27更新：B固定命令确切复现掉杯

新recorded-B回放已实际完成3000步，observations/actions/cup/EEF/contact/grasp六数组与原live B逐元素一致，失持仍在step1136；因此不需要在线模型/RPC等待来复现这次失败。新的base-prefix OFFLINE动作干预已启动，不是live π₀修复；详见`docs/B-command-tape-localization-20261118.md`。Source replay成功仍只归属reference，未变成learned success。
