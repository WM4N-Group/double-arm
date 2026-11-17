# 同基座采样对照B：全轨迹uniform，而不是继续延长旧训练（2026-11-17）

## 动机与当前状态

已证实A final19999在source635人工pre-grasp初始化下能用真实π₀抓起杯子（assisted），但source0完整3000步仍失败；实际50/50采样中普通帧只有5–6次曝光，九个抓取窗口各1117–1118次。**轨迹覆盖不足是待验证假设，而非唯一根因。** 因此启动新同基座受控B，不再把旧checkpoint盲加90k。

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
