# Uniform B：TRAIN辅助模式达到上游inside目标；physical测试进行中（2026-11-18）

## 结论与限制

**这是新B final19999从原任务起点source0出发的结果，不是source635预定位片段，也不是source action replay。** 在TRAIN/demo0、作者configured assisted条件下，原版π₀真实闭环把杯子带入水槽区域，满足上游任务inside谓词连续30帧，1694步正常结束。

但必须同时明确：

1. **assisted constraint确实触发，尚非无辅助物理夹持证据。** 用户之前指出的吸附问题仍然成立；本次不因轨迹更完整就把辅助附着改称物理抓取。
2. **结束时杯子仍被backend报告拿持，未观察到放开。** 上游谓词只检查inside，没有release条件；不能说已经验证“放下并松手”。
3. TRAIN不等于heldout；一条训练轨迹诊断不等于论文规模或泛化成功。
4. 自动TRAIN/physical正在运行，**尚无结果**；heldout/physical未启动。全部物理门禁不变，不拿assisted结果替代。

## 1. 训练与独立重载实际完成

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-uniform-B-20261117-215312/campaign
runs/latest-pi0-relative-uniform-workflow.txt
```

- 11月17日**23:35左右**trainer完成20000真实optimizer updates并exit0，final19999已commit。AdamW/LR counts均20000。
- 全参数SHA256 `bd591dd187acff973edb632a3d65e3c7d90530fc4cb3ae604b60751644707bfa`，与A的`ffe08f...b78180b`不同；这是**B自己的新权重**。
- 独立CPU重载完整3,315,905,296个参数的bits/dtype/hash与saved norm一致，四个真实TRAIN观测native50×21预测有限。495,908,624 FP32 trainables /2,819,996,672 BF16 frozen元素。
- 某source输入跨CPU/GPU输出的最大单坐标差约0.0909，混合21D物理单位，不能称统一rad误差或bit-exact inference。参数重载逐位一致与跨backend采样差异分开记录，不能从重载通过推断任务成功。
- 仍是officialbase/freshAdamW、单demo/batch1/meanstd、20000 updates、原relative推理profile，新uniform training_variant；**不是论文1000demo/batch64/quantile训练**。

## 2. 已完成的TRAIN/ASSISTED闭环

于11月18日**00:09:55 CST**实际完成并导出review；上游流程随后按原门禁进入TRAIN/physical。

| 指标 | B TRAIN/ASSISTED |
|---|---:|
| 初始serialized load | 1，原source0起点 |
| 实际步数 /推理调用 | 1694 /170 |
| 视频仿真时长 | 56.467秒，非实时控制声明 |
| 底盘路径 | 2.60588m |
| 上游inside goal | true，连续30帧 |
| backend杯子grasp曾true | true |
| assisted constraint曾出现 | **true** |
| 最大杯子抬升 | 0.43541m，**辅助模式** |
| 物体显式pose writes | 0，不代表无constraint辅助 |
| teacher action reads /planner | 0 /false |
| left finger-cup已知接触样本 | 1694 |
| ≥1 /≥2 finger-link接触帧 | 1009 /987 |
| 接触未知样本 | 0 |
| 原native base proposal被bound截断 | 0 |

真实初始三相机source MAE约5.25/6.57/5.67；输入为当前head+双wrist RGB、原21D proprio、instruction，没有物体pose、phase/time、future输入或逐步恢复。接触数和EEF原点距离是只读几何遥测，不代表校准抓力或无辅助稳定性。

### 结束时未松手：不把goal true夸大为完整放置

- 实际runtime binary threshold=0.5，有motor metadata证据。
- first effective close step682（22.733s），之后到最后1693**全部维持close**；1012个控制步是一段连续区间，不是1012次尝试。
- 杯子backend grasp flags从step688开始，最后一步仍为true；最后10个命令都闭合，视频末帧也显示手仍在水槽中。
- Pinned任务`datagen_tidy_table/problem0.bddl`目标只有`inside(teacup,sink)`，无显式release条件。SHA256 `3d73488895f5593609208c6c4e9e9c7635885ace915b32d460b045c82d420555`。
- robomimic `EnvOmniGibson.is_success()`对该任务使用BDDL unsatisfied数为0，没有单独的“松手后稳定停留”分支。故原`training_split_author_config_task_success=true`与“仍拿着杯子”不矛盾；它只是**原目标谓词满足**，不是已完成用户语言意义的放下。
- 不能推断如果继续运行就一定不放手；这次在原谓词保持30帧后停止，后续行为未测。

权威补充证据为`campaign/assisted-goal-evidence/claim-scope.json`及BDDL源码快照。没有改原goal、结果JSON、判定门禁或in-flight evaluator；只是把度量边界说清楚。以后即使physical gate true，仍应人工检查release，不能默认原谓词包含它。

## 3. A/B如何解释

A同base/20k、50%窗口oversampling从source0完整轨迹失败；B同base/20k、uniform采样在这一次TRAIN/ASSISTED里达到inside目标。支持**全轨迹采样覆盖是有用因素**，而不是“模型无论何种条件都只会无意义播放轨迹”。

但只有一个training seed、一个TRAIN测试episode，跨进程噪声/仿真与backend采样也有差异，且assistance仍存在。不能声称已证明唯一根因、普遍解决误差恢复、物理抓取domain gap或heldout泛化。原A及更早native失败全部保留，没有被重新标记成功。

## 4. 匹配B权重的56秒review

```text
/home/yuanyue/double-arm/outputs/demo-review/pi0-uniform-B-TRAIN-ASSISTED-NOT-PHYSICAL-20261118/demo.mp4
```

- 960×508/H264/15fps/847帧/4,326,468字节；字幕明确TRAIN/ASSISTED/NOT PHYSICAL SUCCESS。
- 视频SHA256 `31ad1307e734959633c1bff3273fbb730d37354e15d3057e2f406249c2477533`。
- review JSON SHA256 `ed6855ade8d2e6f1e7fc54678469eb651b36ac52435b02891787a9e27f19dd6f`。
- B inference-tree fingerprint `35bab2d11ef7f971820f91ae2ba6ded24334733f016ef069bb13728a1e29322f`。
- trace SHA256 `89276a1f5ebc819dbd30f4b361b796136c78f84ce4fda5be76f745564f38dbd5`（trace仅服务器）。
- 服务器完整ffmpeg decode通过；本地视频/JSON/接触图SHA核对，6个概览关键帧及21/23/24/40/55/56.3秒事件图目视检查。未声称本地ffmpeg解码（本地无ffmpeg）。小型review忽略目录，不提交媒体或restricted assets。

## 5. 当前物理流程保持不动

- 父controller210194，于00:09:55后自动进入`train-physical-diagnostic`；子controller216890、CPU service216897、sim wrapper217043/actual217044，以实际指针为准。
- 00:26日志约step500，还未到抓取阶段；`result.json`空episodes/false是**初始placeholder**，不是已完成的失败。无新physical结论。
- 原source0/init1、物理模式、sync20/execute10/Gaussian10/native bounds/.98 planarity及原task验收不变。新CPU准备的no-assistance审计helper**没有接入此次in-flight eval**；不虚构新增USD/perstep观测已经存在。
- GPU1锁仍覆盖全部自动评估链，CPU/低利用率间隙也不插队。只有TRAIN/physical实际通过，才可能原heldout/physical；失败就保留本B的新FAILED review。
- GPU0原seed10005官方source generation继续，worker125/217942与最后global immutable QA180/314090分开；不把临时日志加到完成计数。没有新训练、labels/norm/adapter/sampler/eval改动或部署。

## 01:28最终补记：TRAIN物理模式未过

原physical流程于00:58:35实际结束：3000步任务失败，heldout未运行。检测到接触与短时抬杯，但约37.87秒开始失持下落，夹爪当时仍命令关闭；这不是任务成功或主动放杯。最新匹配FAILED视频、证据分级及独立source0物理动作回放见`docs/pi0-uniform-B-physical-failure-20261118.md`。本页00:26“进行中”段落保留历史，不再代表当前状态。
