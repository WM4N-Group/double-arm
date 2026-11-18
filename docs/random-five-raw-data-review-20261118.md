# 随机五条官方原始数据视频检查（2026-11-18）

用户要求随机抽取5条看视频，非选择“最好”的5条，也不是新增策略评估。

## 抽样方法与范围

- 固定输入：`/mnt/yuanyue/double-arm/runs/official-raw-pool-audit-20261118-072614/audit/report.json`，230条已finalized、跨池QA合格且排除旧16TRAIN/1heldout精确指纹的原始示范。
- audit SHA256：`a50afa6ae9841aab358db102723c5f8f092a16b8afb0fc4e78f983c692c75e22`。
- 先按episode fingerprint排序，再用Python `random.Random(seed).sample(population,5)`均匀无放回抽样。随机seed由系统随机源一次产生：`1220516183893957583`。
- 在读取视频像素前先写入不可替换的`selection.json`；导出失败应停止，不补抽或换“好看”的数据。
- **总体本身仅含已验收成功生成的raw episodes**，不含生成失败attempt或未完成分片；不能称从所有尝试中抽样，也不能据此估计生成成功率或场景多样性。

| 样本 | 分片seed | 原episode | 帧数 | 仿真时长（30Hz） |
|---|---:|---|---:|---:|
| 1 | 10002 | demo_9 | 1554 | 51.800s |
| 2 | 10005 | demo_2 | 1684 | 56.133s |
| 3 | 10004 | demo_23 | 1745 | 58.167s |
| 4 | 10002 | demo_3 | 1735 | 57.833s |
| 5 | 10004 | demo_6 | 1601 | 53.367s |

总计8319帧/277.3秒。两个样本来自同一分片是随机结果，未人为强制跨分片覆盖。

## 导出语义

脚本`/home/yuanyue/double-arm/scripts/export_random_raw_reviews.py`只在服务器CPU运行，明确`CUDA_VISIBLE_DEVICES=""`。读取原HDF5 `obs`中的头部、左腕、右腕RGBA，去alpha后分别由原256×256等比例放大到320×320并排，保留全部源帧、采用env_args真实action_frequency；标签标明仿真时间。

**OFFICIAL SOURCE / ASSISTED / NOT LEARNED PI0 / NOT PHYSICAL GRASP VALIDATION**，不是新模拟、逐步状态恢复视频或策略输出；画面里的辅助吸附不能当无辅助抓取证明。没有物体状态叠加、专家RGB替代在线policy输入或任何控制执行。

每条在读取前验证finalized verdict、action+初始proprio指纹、完整长度及非partial；导出时计算三路原RGBA全帧SHA，导出后核对MP4帧数/时长、服务器ffmpeg完整decode并生成六时刻接触图。只取压缩MP4、小型review/selection JSON及JPEG到本地，不取HDF5、NPZ、权重或资产。

远程运行目录：`/mnt/yuanyue/double-arm/runs/random-raw5-review-20261118-105836`。低优先级CPU affinity88-91/2线程，与GPU1生成共享允许的CPU范围但不启动GPU任务，两张现有生成保持不动。

首次启动`...105807.log`因cpuset不允许CPU32-35在抽样和建立输出目录前退出；保留失败日志。检查实际允许`0-15,88-103`后显式新run启动，没有已抽样数据的重新抽取或美化筛选。

## 测试与验证进度

新增3个纯CPU测试（固定seed/唯一性/无关输入顺序、错误总体拒绝、从记录元数据读取fps）本地与服务器通过。全本地201 tests /153 passed /48 dependency skips，非GPU或策略成功证明。

## 实际完成与交付验收

5条全部实际导出成功，8319源帧全部保留；每条服务器完整ffmpeg decode、MP4帧数/时长门禁通过。共17,078,529字节（约17.08MB）压缩MP4，均已取回本地并核对视频/接触图SHA与对应review字段；五份review逐一核对了selection SHA、episode fingerprint及帧数。已目视检查每条0/20/40/60/80/100%六时刻三路接触图；未发现需要替换的导出失败或缺失视图，**没有做补抽**。部分近距离搬运画面左腕被杯体遮挡明显，固定D0场景的视觉相似性不能当多场景多样性证明。

Selection SHA256：`1f53f93102415cb0db97b29fb68058c5e526f385f13cd3c926d9465c9c6b667d`。

本地selection/status：`/home/yuanyue/double-arm/outputs/demo-review/raw-random5-20261118/`；视频、review与接触图分别位于`/home/yuanyue/double-arm/outputs/demo-review/raw-random5-20261118-sample-01/`至`...-05/`。没有声称本地ffmpeg解码，也未复制源HDF5。

| 样本 | MP4 SHA256 |
|---|---|

| 1 | `5b3304ba61a2697142a87c97e2c4512f8d94a632d90e5eae353725a0ab558119` |
| 2 | `8079bc9da09aa6d4adddaee91c18530c80a600233d63e793ed344ccfd277dca7` |
| 3 | `8b77ac96948c57906b7bf248e21a358c4e4cd9d460374c77381aef1604d1ba03` |
| 4 | `ba310d2ea48001c7a39a576e4ad2176342f48c52d56432db9c5e9593b080cbd3` |
| 5 | `66a62f9ca44f23610c6a02e58fd942f3d85c6f72adf062404ed87f82a20d9302` |
