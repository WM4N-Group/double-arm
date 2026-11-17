# 同基座采样对照B：全轨迹uniform，而不是继续延长旧训练（2026-11-17）

## 动机与当前状态

A final19999在source635人工pre-grasp初始化下输出关夹/抬臂并触发辅助附着后抬杯，**未验证无辅助物理夹持**；source0完整3000步仍失败；实际50/50采样中普通帧只有5–6次曝光，九个抓取窗口各1117–1118次。**轨迹覆盖不足是待验证假设，而非唯一根因。** 因此启动新同基座受控B，不再把旧checkpoint盲加90k。

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-uniform-B-20261117-215312/campaign
runs/latest-pi0-relative-uniform-workflow.txt
controller210194 / trainer210198 / watchdog210190
```

21:53启动前，local grasp整个service/sim/export流程已退出、GPU1无compute进程并取得UUID锁。21:55处于真实HOST缓存阶段（101/1714帧、**0 real optimizer updates**），初始化参数hash/首次梯度仍待验证，不能因controller stage叫training就说已经完成fit。

GPU0仍为原官方1000-target数据生成，未中断；当时seed10004在合并25个source文件，未将log数量算成final。GPU1所有权覆盖B训练→保存→独立CPU重载→自动评估全过程，空闲/CPU阶段不得插入其它GPU任务。

## 唯一预期学习配方变化：采样schedule

| 项目 | 已保留A | 新B |
|---|---|---|
| 起点 | official pi0_base、fresh AdamW | **完全相同**，不是A warm start |
| 数据 | 一条原TRAIN/demo0 /1714帧 | 完全相同、原文件只读 |
| 正式updates /batch /seed | 20000 /1 /42 | 相同 |
| 采样 | 50%全轨迹循环+50%九转折窗口 | **100%全轨迹打乱循环** |
| 普通帧曝光 | 5–6 | **11–12** |
| 九转折窗口曝光 | 1117–1118 | **11–12** |
| 相对joint编码、21D proprio、RGB/prompt | 原relative profile | 相同 |
| normalization | 原16TRAIN-derived relative-joint mean/std；native base/gripper和state stats保留 | 相同，绝不重算/改旧stats |
| model/rank/vision/freezing | original π₀ /32 /视觉可训练 | 相同，495,908,624 trainables |
| LR | warmup1000、peak2.5e-5、decay30000、end2.5e-6 | 相同 |
| 保存 /重载 /eval | 每5k官方Orbax、原dtype独立重载、source0 sync20 /Gaussian10 /execute10 | 相同 |

新sampler不使用onset信息，只保留兼容函数签名；frame index/phase/time/demo索引不进入模型。没有proprio noise、image替换、teacher输入、planner或额外state字段。

## 配对门禁与显式版本

- `training_variant=uniform_full_trajectory_same_base_v1`，新trainer/controller/helper文件，旧A脚本和四个checkpoint-coupled runtime源码完全不改。
- A权威指针`latest-pi0-relative-full-demo-workflow.txt`**保留不动**，B另用uniform pointer。A原失败和局部grasp诊断都保留，B新checkpoint路径使用新experiment目录，不覆盖任何旧权重。
- Controller逐项验证baseline plan与B的common fields完全一致（模型接口、updates、batch、seed、rank、norm anchor/SHA、provenance、初始参数hash、初始化方式、LR、保存/评估配置、runtime source hashes）。只允许schedule、训练scope及新实验源码/元数据不同。
- Worker核对整个1714帧原始RGB/state/prompt输入digest与A一致，逐帧native动作逆变换检查；初始化后完整参数/dtype hash必须与A的`a3ee6c...28e1b`一致，否则在fit前停止。
- Sampler/trainer源码SHA在运行中受保护，保存时再次核对；不能在in-flight修改本配方来追成功。

**为什么复用`policy_profile=pi0_r1_relative_full_demo_v1`？** 这是原21D输入、relative label语义、mean/std、输出inverse与dtype的**推理接口profile**；本B没有改变其中任何一项，所以不重新解释adapter，也不改serving。训练条件独立以`training_variant`、`sampling_recipe`、新schedule SHA、新experiment路径和`paired_baseline_campaign`记录进plan/report/checkpoint contract。Controller恢复/服务前要求新variant及A provenance一致；绝不能把B称为旧50/50 checkpoint。

同seed一次A/B和跨进程数值/仿真随机性仍限制因果推广；即使B改善，也不是大数据或多seed证明。更均匀采样也可能损失边界窗口拟合，**不保证通过**。

## 完成后的自动流程（门禁不因局部抓取而绕过）

1. 20000实际updates，final19999 commit、真实AdamW/LR counts、asset norm与profile contract保存。
2. 独立CPU完整参数bits/dtype/count hash+saved norm重载、实际TRAIN预测有限。
3. **从source0完整起点**做TRAIN/author-assisted诊断，仍是原20次reset physics refresh，不使用source635特权起点。
4. 只有完整TRAIN/assisted task true，才TRAIN/physical；再只有TRAIN/physical true，才原heldout/physical。失败则保留并交付**B本次权重**的新FAILED review，不拿A视频代替。
5. 最终须检查视频/实际物理指标，局部杯grasp或训练loss不解锁以上门禁。

本B仍是一条demo、batch1、mean/std、20k更新的**优化/覆盖诊断**，不是1000 demos /batch64 /quantile论文复现；`formal_training_completed=false / training_is_diagnostic=true`保持。训练3h、CPU重载40min、每评估3h、父12h是防挂起上限，不是新计算预算或交付ETA。

## 测试与保留项

本地176项132通过/44依赖skip；新uniform sampler覆盖/界限2项及local forensics连续close区间1项在服务器CPU也通过。既有profile/Orbax fixture仍在原测试集中；本新run最终参数验证以实际报告为准，不用fixture替代。用户survey与x目录未改未提交，原PDF、数据、权重与assets不动。

## 22:01状态补记

B仍在HOST缓存，22:00实查801/1714帧、0 optimizer updates、无error；不是已完成模型初始化或实际梯度通过。不要用controller的training标签替代worker实际stage。

GPU0 seed10004已真实exit0/finalized25（35 attempts/25 successes/10 failures），primary worker累计**125 raw episodes /217,942帧**，原控制器自动进入seed10005，新sim cold-start。这里只是worker内计数；最后跨池immutable QA仍155/270,250 snapshot，**没有直接把它与旧aux计数相加宣称新全局180**。GPU0没有被重启或占用来做策略测试。

## 22:31运行核对：真实配对检查及首个5k保存通过

- B实际达到**5000/20000 optimizer updates**，当前首个Orbax保存阶段，无error；末步loss约0.03109、grad norm约1.29568，均有限，不代表任务成功。
- 全1714 HOST cache完成，原始RGB/state/prompt digest与A完全相同：`3f216b350f63895556028f1bd2d4dd9012a24036804a1faa0be5b44d1b97c6f0`；全source动作inverse max1.19e-7。
- 实际初始完整参数/dtype hash与A相同：`a3ee6c232ac89b0e4e5437e326dc5af9234f3a4436cb4cc0b138fbe605e28e1b`。不再只是plan中的预期值。
- 新uniform schedule SHA `f6f75ec9e122a85c22df8fb929d4392748921763123dd74bf5a8c22db9b662d0`；source与基座相同，采样顺序/权重是明确差异。
- 首个checkpoint `runs/pi0-checkpoints/pi0_r1_relative_full_demo_v1/pi0-relative-uniform-B-20261117-215312/4999`实际保存，AdamW/LR counts均5000，norm SHA仍与A一样`77f6443649bf3468ebd164f27cb21f01898f89f42f8fa27faf50c89f4bd10fc5`。contract明确uniform variant、paired A provenance及`training_completed=false`，**中间点不可服务、不提前评估**。
- 无新B闭环/physical结果；训练、sampler、norm、adapter与现有eval源码保持不变。另备的无assistance只读审计工具只做CPU测试，未接入本in-flight流程。

## 同时完成的原始池只读QA（不是新策略结果）

CPU读取已登记4个pool的final provenance，再排除旧16/1并跨worker去重，已真实得到**180 eligible unique /314,090 raw frames**、重复0/旧split指纹重合0；不是简单相加计数。仍为assisted来源，不是180新场景或physical成功。

```text
/mnt/yuanyue/double-arm/runs/official-raw-pool-audit-20261117-222630/audit
runs/latest-official-raw-pool-audit.txt
```

Report SHA256：`fbe258ef18ead6e1a1c5c763e0508473b9a43571d1147fcd29aaadc11b279d42`。14个candidate frozen-segment frames、覆盖gripper transitions0；精确author joint subset未核实，**未实际cleaning/split/normalization修改**。GPU0 seed10005继续，GPU1 B训练未被中断。
