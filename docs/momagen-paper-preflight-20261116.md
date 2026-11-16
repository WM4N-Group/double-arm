# MoMaGen论文对齐预检：只读清洗审计与有效batch64数学验证

## 边界

当前native coverage v3额外90k训练继续运行，数据、归一化、优化器、模型输入和训练源码均未改动。本文件记录CPU侧准备，不是新的机器人训练或闭环成功。

远端预检目录：

```text
/mnt/yuanyue/double-arm/runs/paper-preflight-20261116-195823
/mnt/yuanyue/double-arm/runs/latest-paper-preflight.txt
```

## 1. 清洗判据核对与只读审计

核对arXiv:2510.18316v2 AppendixC.1：比较q[i]与q[i+5]的端点差，而不是要求每个相邻差分连续5次很小；所有维度差均严格小于1e-3时，删除闭区间[i,i+5]。因此一次窗口覆盖6个记录样本；实现使用重叠窗口的并集。

新增 `frozen_segment_audit.py` 的数组算法与 `audit_momagen_frozen_segments.py` 只读审计工具：

- 使用HDF5里真实记录的 `obs/joint_qpos`（本数据为26维，包含移动基座位置），**不把原21维proprio中的基座速度当位置，也不把恒定的原生基座增量命令当成静止**。
- 记录源字段、qpos指纹、静止窗口、候选移除比例和受影响夹爪命令转换。只输出候选mask / indices，不修改HDF5、LeRobot或在训标签。
- 作者训练预处理中究竟选择哪些joint字段尚需官方实现确认；当前映射标记 `field_author_mapping_verified=false`，不把数学判据一致说成完整作者预处理已复现。
- 单元测试包括端点返回原位但中间移动的例子：按论文端点规则仍被标记，不自行换成“窗口内最大运动”规则；还验证了严格阈值、短序列和重叠边界。

实际审计：

| split | 演示数 | 原帧数 | 当前qpos候选移除 | 保留 |
|---|---:|---:|---:|---:|
| train |16|27,657|0|27,657|
| validation |1|1,624|0|1,624|

这表示**当前字段与判据在这些已有演示上没有检出静止段**，不是“清洗已经改善了策略”，也不证明1000条数据或作者关节子集会得到同样结果。不能硬删数据来制造清洗效果。

## 2. 单卡有效batch64：先核对计数与优化器顺序

新增 `pi0_accumulation_preflight.py`，与当前TrainConfig完全隔离，未接入在训任务。microbatch1、累计64次才更新一次优化器；论文的50k优化器更新需要320万microsteps / 320万次样本曝光，而不是保持50k训练循环就自动等价。

在服务器CPU上运行 `audit_pi0_accumulation_cpu.py`，用小型确定性线性损失对照直接batch64：

- 3次有效AdamW更新，参数、优化器一阶 / 二阶状态及学习率计数匹配，最大参数绝对差约9.3e-10。
- 每个窗口的前63次microstep参数不改变，64次后更新；不能把microstep数当优化器更新数。
- 验证先平均原始梯度、再做全局裁剪 / AdamW；不能先裁剪每个microbatch再平均。反例中正确平均为0.015625，逐样本裁剪后平均变成0，说明顺序会改变训练。
- 对当前官方freeze分区的495,908,624个可训练参数，仅一个float32累积缓冲的理论额外开销约1.847GiB；这只是下界，且当前分区包含视觉等非LoRA参数，不是假设“只训练LoRA矩阵”。

**没有用真实π₀做GPU累积训练，没有验证24GB显存可放下，也不证明随机增强 / dropout / 真实π₀梯度与多GPU逐位等价。** 不占GPU0，不为预检打断GPU1当前训练。真实GPU内存 / 梯度验证放在当前工作流完成之后，正式复现仍需另建配置、数据版本与计数日志。

论文Table5的warmup1000 / cosine30000等调度单位应按优化器更新计，而不是按320万microstep计；当前coverage v3的90k调度是诊断设置，未改名为论文设置。

## 3. 作者资源补查的范围

再次访问官方README、官方文档首页和公开issue6：README明确定位为数据生成代码；当前文档导航为安装 / 数据生成 / 自定义任务；issue6仍Open且页面无可见作者回复，提议公开模型与数据。这些页面没有直接提供可用的π₀训练 / 评估细节，**不能据此断言所有渠道都没有发布资源**。

仍待核对：动作编码与底盘转换、作者关节清洗子集、视觉冻结分区、执行chunk频率、评测初态来源和抓取模式。不代用户向上游发帖，不接受新条款，不拉取新上游版本覆盖锁定环境。

## 4. 验证状态

- 本地98项测试通过，其中25项数值依赖相关测试跳过；CPU侧真实累积算术检验已通过。
- 本轮没有改变当前GPU训练的任何参数或数据，也没有产生新的策略成功率或成功视频。

## 5. 接手后的隔离GPU预检准备（22:53 CST，尚未运行真实π₀累积）

再次核查权威指针：coverage v3仍为 `native_coverage_finetuning`，约76.3k / 新增90k更新，控制器117227、watchdog118863、实际trainer118864均存活。没有启动新的训练或仿真，没有新闭环结果。当前训练及后续自动评估的任何配置 / 标签 / 数据 / normalization均保持不动。远端已部署的只读grasp telemetry、evaluator、evaluation orchestrator与本地对应文件SHA256一致，未来自动评估会使用这套遥测；这不是实际接触数据已经产生的声明。

新增 `scripts/preflight_pi0_accumulation_gpu.py`，供**整个现有训练→自动评估工作流结束且GPU1释放后**单独执行：

- `--check-only` 只用标准库、分配文件、权威指针、进程状态和 `nvidia-smi` 检查。非终态、控制器 / child仍存活、指针改指新任务、GPU1占用或用户pause均拒绝；不能把训练结束到评估开始之间的短暂空卡当释放。
- 非check-only路径也先过门禁，再独占本预检lock并复查，才加载JAX / 模型；仅使用明确授权的GPU1 UUID。此lock只序列化本预检，不是其他控制器也遵守的全局GPU锁。
- 复用固定openpi的 `init_train_state` / `train_step` 与现有真实三图训练loader；仅在本进程中将原clip+AdamW封装进 `MultiSteps(64)`，随后立即还原优化器工厂。未改上游文件，未接入当前训练器。
- 只测现有native / mean-std / rank32配方的内存与计数可行性，不声称quantile或完整论文新版本已经对齐。使用显式提交的native权重热启动和fresh optimizer，无模型推理 / 仿真 / teacher控制。
- 计划做64次真实microstep / 1次effective optimizer update，记录实际accumulator字节 / dtype、loss与norm有限性、显存统计、micro与AdamW / LR计数。在0/1/63/64处对完整trainable树做内存哈希：检查1/63不变，64更新。**不是每一步完整树都做哈希，也没有与真实直接batch64逐位对比。**
- 不调用上游main，不保存权重或checkpoint，只写小型JSON。异常 / OOM明确保留失败，不把CPU数学通过或仅初始化成功写成GPU累积通过。实际首次初始化 / 编译仍可能发现问题。

独立预检脚本暂存于远端已有CPU预检产物下：

```text
/mnt/yuanyue/double-arm/runs/paper-preflight-20261116-195823/staged-accumulation-gpu/preflight_pi0_accumulation_gpu.py
```

真实执行命令（**本次没有执行，也没有后台排队**；先处理新评估视频，再等完整工作流释放）：

```bash
R=/mnt/yuanyue/double-arm
C=$(cat "$R/runs/latest-pi0-coverage-campaign.txt")
P="$R/runs/paper-preflight-20261116-195823/staged-accumulation-gpu/preflight_pi0_accumulation_gpu.py"
INITIAL="$R/runs/pi0-checkpoints/pi0_r1_tidy_table_lora32/pi0-momagen-20261115-233529/29999/params"
OUT="$R/runs/pi0-accumulation-gpu-preflight-$(date +%Y%m%d-%H%M%S)/report.json"
CUDA_VISIBLE_DEVICES=GPU-50e5d38d-2825-e379-f537-7f0bf4609aba \
JAX_PLATFORMS=cuda PYTHONPATH="$R/code/scripts" \
timeout --signal=TERM --kill-after=90s 3600 taskset -c 88-103 \
"$R/envs/openpi/bin/python" -u "$P" --campaign "$C" \
    --initial-params "$INITIAL" --output "$OUT"
```

这里3600秒是单次预检watchdog，不是重置预算或要求用户重新授权。进程被硬kill时JSON可能停在最近一个已完成microstep，不能据此认为全部通过。未来执行时还需记录该独立预检的job指针；此处没有改写在训工作流的任何指针。

验证：本地共运行107项测试（80通过、27项数值依赖skip，含7项新增标准库门禁测试）。服务器用隔离脚本实际运行 `--check-only` 得到退出码3，并明确报告训练 / 控制器 / GPU占用阻断；同一进程 `jax / numpy / torch / openpi` 均未导入。**真实π₀ GPU累积、显存可行性和真实直接batch64数值一致性仍未验证。**

自动化防重复：本次可用工具中未发现 `automation_update`，本地可见automations目录亦没有 `double-arm-46207-2` 对应配置，因此未能核实该app侧自动化指向；未创建、替换或宣称已修复自动化。远端现有控制器仍独立运行，其训练→诊断→条件性heldout自动链路不依赖此处新建monitor。
