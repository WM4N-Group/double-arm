# GPU1有界数据衔接：等待当前批次完成，再审计，再开下一批

## 背景与不变项

2026-11-17 07:06 CST，GPU0第二shard仍在生成（日志18成功/22次尝试，原池已验收25）；GPU1 seed20000为24成功/32次尝试，尚未finalize。两卡都健康。这里不打断任务、不修改原25目标，也不因为临近日志成功数就提前认定完成。

为落实用户“两张卡都放开给用，跑满”的授权，避免GPU1完成有界辅助批次后等下一次heartbeat才衔接，启动一个**CPU-only、有界、仅一次后继数据批次**控制器。这不是新的Codex app automation，也不是旧memory/state-policy双卡控制器。

## 当前CPU后继控制器

```text
/mnt/yuanyue/double-arm/runs/paper-gpu1-data-followup-20261117-071110
watchdogPID160850 / controllerPID160852
started2026-11-17 07:11:11 CST
runs/latest-paper-gpu1-followup.txt
```

启动时为`waiting_for_existing_auxiliary_pool`，GPU1实际job指针仍属于原seed20000控制器，**没有启动第三个GPU任务**。

## 明确门禁

1. 只等待既有`paper-official-aux-gpu1-20261117-022105`；不向该进程发signal，也不停止GPU0。
2. 依赖必须是完整目标完成状态、至少25条已验证source episodes、active_pid0、assisted来源、未进行policy training，且controller / child都退出。普通日志成功数或`in_progress`里的计数不算。
3. 若依赖失败，不自动绕过、伪造final文件或静默重启；CPU后继控制器明确失败并保留日志。
4. 先进行跨pool只读审计：原GPU0 raw provenance + 已完成GPU1 auxiliary provenance，统一去重并排除旧16/1 split重复，检查final HDF5 / vectors / qpos与相机header。GPU1必须至少增加1条eligible新轨迹，否则停止扩张。
5. 确认GPU1仍由预期任务指针拥有、实际compute进程已释放，再用现有按UUID的job锁启动新worker；不kill别的任务，不fallback到GPU0。
6. 后继worker是**新的独立25条author-assisted D0 source pool**，seed20001，CPU88-103、GPU1 UUID、每shard6h / max2 shards。原seed20000结果不改。共享官方config准备锁继续生效。
7. 后继完成后再做一次合并只读QA，随后控制器结束；不是无限生成循环，不启动π₀训练 / 评估。

CPU后继控制器总8小时watchdog只防挂起，不是重置用户总预算或需要新授权。超时/用户pause时只终止自己后来创建的child，不杀原dependency或GPU0。

## 原始池registry与状态指针

后继跨pool QA首次运行时创建/更新：

```text
runs/official-raw-pool-registry.json
runs/latest-official-raw-pool-audit.txt
```

registry保留各worker的raw provenance路径，即使`latest-paper-generation-gpu1-campaign.txt`以后指向新worker，也不会忘掉先前的25条候选。registry是**未划分的author-assisted原始数据来源目录**，不是train/validation清单，也不把per-worker去重数直接相加。

实际新worker启动后更新`latest-gpu1-job.json`和合并map的GPU1角色；CPU后继另有`GPU1_finite_data_followup`记录，不能将它误认为额外GPU占用。每次只读QA把当时provenance内容与SHA冻结进report；下一shardappend不会改写该快照。

## 验证 / 通知边界

- 本地123项：94通过、29数值依赖skip；新增3项依赖门禁 / registry夹具在服务器Python3.10通过。
- 代码源快照/SHA随CPU控制器保存。启动观察为等待依赖，未修改两张卡当前任务；后继是否真正启动必须读取实际status，而非把排队当完成。
- 25条辅助数据正常完成、接着正常进入下一有界批次，属于健康data in-progress衔接，保持quiet。质量门禁失败、任务故障、整体数据目标完成或需要用户操作才通知。
- 没有新learned-policy结果；此前coverage v3的失败不因数据增长而变成成功。源演示仍assisted，不冒称physical grasp或论文完整复现。

## 08:06核实：门禁已经按预期衔接，属于正常数据进展

- seed20000辅助批次于 **07:57:19 CST**最终完成：25条、43,282帧；dependency进程退出后才开始后继步骤，没有被提前终止。
- 跨pool CPU审计实际通过：与GPU0首批合计 **50条eligible unique / 87,049 raw frames**，快照内重复0、旧16/1 fingerprint重合0。仍只有6个候选清洗帧，没有实际清洗、split或norm变更，author joint subset仍未核实。
- 审计目录：本CPU控制器的`before-next-pool-audit/`；report SHA256为`5e149a80da9f2347f097f1a61a50235348285c1846b24cfefec6bb1b09e2f132`。这不是直接相加per-worker计数：统一指纹去重与旧split排除已经执行。
- 新seed20001 worker于 **07:57:32 CST**启动：`paper-gpu1-data-followup-20261117-071110/next-auxiliary-pool`，controllerPID161674、实际模拟器PID161729，GPU1 UUID。它已进入真实官方生成，不是仅排队，08:06仍未完成新shard。
- `latest-paper-generation-gpu1-campaign.txt`、`latest-gpu1-job.json`与合并map都已指向这个新worker；完成的seed20000路径保存在registry与dependency记录，没有被遗忘。原GPU0 source仍正常执行seed10001。
- 这是预期的健康数据衔接，不另发每批通知；没有新的π₀训练 / 评估或成功demo，继续保持两个既有GPU任务。
