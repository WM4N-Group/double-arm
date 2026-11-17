# 两卡官方raw数据推进：新GPU1辅助100候选，旧实验均保留（2026-11-18）

## 当前任务分配

只有在整个execute5 controller/service/sim/export已退出、GPU1无compute进程/UUID锁可用后，于**07:25:44 CST**启动原官方分片控制器，不重部署或升级：

```text
GPU0（原任务，不变）
/mnt/yuanyue/double-arm/runs/paper-official-pool-gpu0-20261116-231115
controller143656，primary target1000，当前seed10007
UUID GPU-925e5f5c-3cbb-7e93-5a98-1f06d3a71c0d / CPU0-15

GPU1（新辅助raw pool）
/mnt/yuanyue/double-arm/runs/paper-official-aux100-gpu1-20261118-072543/campaign
controller228643 / generator wrapper228648 / launcher228639
target100，25-success shards，seed20003起，max5 shards
UUID GPU-50e5d38d-2825-e379-f537-7f0bf4609aba / CPU88-103
```

`latest-gpu1-job.json`、`latest-paper-generation-gpu1-campaign.txt`及combined map已移到新raw pool，旧execute5指针独立保留，map保存completed B provenance。不要把旧B指针终态当当前GPU1空闲，或重复启动相同seed。

07:26已见控制器`official_shard_generation`和实际新run `tidy_table-official-20261118-072544-lDaMkc`，verified=0；**启动/冷加载不等于新100条已完成，也不保证每个时刻GPU利用率100%**。生成有CPU规划阶段，不为瞬时利用率0重启它。

单shard6h、max5 shards、外层36h只防挂起/去重不足，不是新的授权时间窗口。两卡使用时间仍开放；不再申请计算窗口、不恢复旧state/memory替代策略双卡控制器，不创建新app自动化、不接受新条款。现有license confirmations及共享config准备锁沿用。

## 这不是新的策略训练或物理成功

全部为official author-assisted Tidy Table / D0 source generation，不是learned pi0、physical validation、已划90/10的数据或论文复现完成。Aux100只是新raw候选池，**没有修改GPU0正在运行的1000目标**，也没有把总paper样本数改成1100。

后续选定1000时必须跨池去重/固定manifest/cleaning与episode级90/10，新norm只用新TRAIN，不能随意用all raw或heldout统计量。原16train/1heldout、旧norm、所有权重/结果与PDF保留，不把raw文件复制到本地。

## 07:26的新跨池immutable QA真实完成

仅读取此前登记的四个已finalized pool（包括GPU0当前已finalized shard，不读tmp），再次核对旧16/1排除：

```text
/mnt/yuanyue/double-arm/runs/official-raw-pool-audit-20261118-072614/audit
runs/latest-official-raw-pool-audit.txt
```

- **230 eligible unique raw episodes /401,653 frames**；snapshot内部duplicate0、与旧16TRAIN/1heldout完全轨迹指纹重合0。
- 14个candidate frozen-segment frames、覆盖gripper transition0。使用已记录26D qpos endpoint predicate，author精确joint subset仍未核实，**没有实际删除帧**。
- 没有split/cleaning/normalization变更。Byte-trajectory dedup不是230种新scene/语义多样性或unseen评估证明。
- Report SHA256 `a50afa6ae9841aab358db102723c5f8f092a16b8afb0fc4e78f983c692c75e22`。
- 此230统计**不包括**刚启动的aux100临时候选，不能直接加目标数报告330。

Registry仍保留原GPU1 seed20000/20001/20002以及GPU0池。新aux100首次shard final后才有合法`raw-episode-provenance.json`可追加：在registry lock下保留全部旧paths并加入新path，然后CPU再做cross-worker immutable QA。后续指针移动也不能丢掉旧辅助池。没有提前登记不存在的文件或把临时success日志当正式数据。

## 下一阶段门禁（目前未启动）

不在这次raw增长或单demo诊断基础上宣布新formal recipe可用。原paper preflight事实继续有效：host64micro/1update内存/计数smoke通过，不是完整50k训练稳定性或direct real batch64等价证明。

正式新数据/模型版本仍需明确：action/base语义、1%–99% normalization与action representation解耦、cleaning实际joint subset、effective batch64及optimizer/LR计数、vision冻结设置、作者execution频率/评测grasp mode、固定unseen评估协议。尚未核实项必须标未知，不能用失败的absolute-base V2只为得到quantile，也不能给旧B换norm/label。

当前没有新formal training/checkpoint或策略成功结果。GPU持续用于既定官方数据工作，代码/文档同步Git，媒体/资产/数据/权重不提交。
