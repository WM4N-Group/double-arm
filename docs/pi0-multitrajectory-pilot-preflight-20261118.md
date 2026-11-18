# π₀ 多轨迹试训：冻结候选与 TRAIN-only 原生分位数（2026-11-18 15:43 CST）

先准备下一次 OFFICIAL π₀ 多轨迹对照，不再在单条 B 上增加更新数。**这一轮只运行 CPU，未制作 LeRobot 样本、未写运行时 norm、未开始训练、未启动新的 GPU 任务。**

## 实际固定的 225 + 25 条完整回合

输入为跨池 QA 的 280 条已定稿官方 assisted 原始示范 / 488651 帧（`official-raw-pool-audit-20261118-122926`）；先排除 8 条物理专家探针及旧 16/1 split 的精确指纹。新版哈希排序从一整个独立生成分片选 25 验证条（seed 20000），训练取其余分片中的 225 条；验证分片及 seed 都不出现在训练中。预先固定方法，尚未见到专家物理结果，不事后挑好轨迹。单个 D0 验证分片只能用来作离线开发，不能宣称未见场景泛化或替代正式无辅助闭环评估。

- TRAIN：**225 条 / 393802 帧**；验证：**25 条 / 43282 帧**。
- 远程 `/mnt/yuanyue/double-arm/runs/pi0-multitrajectory-pilot-manifest-20261118-154313/split-candidate.json`；SHA `1a0517801299e46107a2f58ece715be3f8a0e46d1bb94be4475c6c4571beb2be`。
- 数据未清洗（端点规则候选不是实际删除帧），未把原 assisted 改标为 physical，也未修改旧 data / checkpoint / config。

## CPU 统计结果，不是新的运行时参数

用 TRAIN 393802 个原始 observation/action 向量算 1%/99% 经验分位数，逐条重验 FP32 动作 + 初始 proprio 指纹；未读取验证观测或动作。此处是**原生动作空间**（局部增量底盘、绝对关节、原夹爪），作者编码尚未锁定，不能移植到相对关节变换后，也不能给 B/v3 换 norm。

- `native-raw-train-quantiles.json` SHA `a631333467a9687f6a4efa2f2bba3b40deb1d8e1a54c61770805cf5e65f3ff7d`。
- 21 个动作维度中 `right_gripper_command` 分位数跨度为零；单独数全体原始 TRAIN 命令确认 **393802/393802 为 +1**，不是分位数忽略了稀少右爪动作。其他动作维度和 21D proprio 均无零分位数跨度。右爪常量需由新运行时规范明确处理并往返测试，不能直接除零，也不能据此称已验证双臂协同抓取。

脚本 `/home/yuanyue/double-arm/scripts/freeze_pi0_multitrajectory_pilot.py`，测试覆盖带排除项的可复现分割与不可行时拒绝跨分片补足。

仍以 GPU1 现有辅助数据采集自然退出后的 **physical-only** 8 条专家回放为首个物理诊断；该队列 239122 未启动 GPU，和两卡目前运行无冲突。下一步另行确定动作表示、清洗和常量轴处理并核验 effective batch，最终只用物理留出闭环验收新策略；225/25 不冒称论文正式的 900/100。

## 16:25 源变换往返实查；新原始数据增加但候选不变

服务器 CPU 对固定的 **225 TRAIN 回合全部 393802 帧**，重验动作+初始 proprio 指纹，调用固定 OpenPI `openpi-215abfb/src/openpi/transforms.py` 的真实 `Normalize(use_quantiles=True) → PadStatesAndActions(32) → Unnormalize(use_quantiles=True)`。21 个有效维往返最大绝对误差 state/actions 均 **3.58e-7**（混合原单位），11 个 pad 维为零，验证集读取 0 帧。OpenPI 分位数变换**不裁剪**超出区间的尾部；这次没有额外 clipping。右爪原命令 +1 正向精确归为 -1、训练样本逆变换精确回 +1；对模拟模型输出 `[-10,-1,0,1,10]`，逆变换约落在 `[0.9999955,1.0000055]`，源码使用 epsilon，**不是硬锁右爪**。均值/标准差占位只供此 quantile smoke（该分支不读取），未生成运行时资产；动作表示与实训仍未确定。

- 远程 `$P/pinned-openpi-native-quantile-cpu-smoke.json`（`P=/mnt/yuanyue/double-arm/runs/pi0-multitrajectory-pilot-manifest-20261118-154313`），SHA `d6447711d132a6ae13b8d0b9bacf0e94ffe1d4b3e658fbfa5b2f436508e26434`。
- 新完成 GPU0 seed10008、GPU1 seed20004 已进入正式跨池 QA：**330 条 / 574636 帧**，重复与旧轨迹重合均为 0，未清洗、未改冻结 225/25 及 8 条探针。远程 QA `runs/official-raw-pool-audit-20261118-162523/audit/report.json`，SHA `6878475179265259e4e6dac4aaac6660f38b1a5aadd080243f3bf14a5870525d`。
- 两卡现有生成继续，GPU1 未释放，physical-only 专家队列未启动模拟器；没有新的抓取或训练结果。

## 17:25 相对关节候选的 50 步窗口边界 CPU 审计

脚本 `scripts/audit_multitrajectory_window_labels.py` 对同一份冻结 225 TRAIN / 393802 anchor 帧重验指纹；未读验证、不修改冻结候选。在既有诊断的“future action index 不越过本 episode，越界重复末步；16 个关节目标减同一当前 anchor proprio，底盘和两爪不变”定义下，全部 50 步索引计数 **19,690,100**，其中 **275,625** 个末步重复（约 1.40%，不是额外真实观测）。在 lag 0/9/19/49 对所有 anchor 实测编解码，逐 episode 的边界样本还核对既有编码函数；最大关节往返误差 **2.38e-7 rad**。报告含这些 lag 的相对关节分位数，仅用于比较候选编码；不能拿原生分位数喂相对标签，也不能把逐 lag 分位数简单平均成运行时 norm。无作者真实动作编码、实际数据加载器配方或 checkpoint 验收结论。

`/mnt/yuanyue/double-arm/runs/pi0-multitrajectory-pilot-manifest-20261118-154313/window-labels-cpu-audit.json`，SHA `d8d03321a1aa87490ceac64ea6205d1dc996b16165c63bbe5ecd2a85e5d33a99`。服务器 2 项窗口测试通过；未占 GPU、未训练。两卡采集继续，physical-only 专家回放仍在等待 GPU1。
