# MoMaGen 官方数据生成办法与本项目适配

核查日期：2026-11-14。结论：**有适配现有 R1 / OmniGibson 栈的官方方法；先做原生生成对照，不需要先更换机器人或导入 R1Pro 数据。** 但开源代码不等于任意机器上都能直接复现论文成功率。

## 1. 查到的官方依据

| 依据 | 能证明什么 | 不能据此声称什么 |
|---|---|---|
| [官方生成教程](https://chengshuli.github.io/MoMaGen/tutorials/generating-data/) | 复制仓库自带场景实例、运行 `generate_configs.py`、调用 `generate_dataset.py --bimanual`，支持四个核心任务 | import 成功不等于生成任务成功 |
| [固定提交的配置生成器](https://raw.githubusercontent.com/ChengshuLi/MoMaGen/5da621667669b15b97d7a220d086283b3c8ad0e5/momagen/scripts/generate_configs.py) | 默认 `GUARANTEE=True`；使用仓库 `processed_source_demos`；D0/D1/D2；自带 MoMaGen 和基线配置 | 不能把基线配置与 MoMaGen 任意混用 |
| [官方自定义任务教程](https://chengshuli.github.io/MoMaGen/tutorials/custom-tasks/) | 扩展任务需有源演示、阶段划分、对象引用、阶段终止信号及接口，再处理源演示 | 不能只改任务英文名或随机给目标位姿就得到新任务数据 |
| [官方安装教程](https://chengshuli.github.io/MoMaGen/installation/) | 要用含子模块的仓库，安装配套 BEHAVIOR-1K 与 robomimic | 不应盲目升级成另一套 Isaac Sim / OmniGibson 版本 |
| [论文 v2，2026-02-22，表 2 与 §5.4](https://arxiv.org/html/2510.18316v2) | 有生成成功率和后续策略学习实验 | 论文数值不是当前服务器的实测结果 |
| [GitHub Releases](https://github.com/ChengshuLi/MoMaGen/releases) | 本次检查时没有发布包 | 不能把“没有 Release”扩大成“世界上没有任何相关数据” |

本轮未找到可直接替代当前任务的**官方整套大规模已生成数据与策略 checkpoint 下载包**。可核实、且与本机栈直接相关的路线是：使用仓库已附的少量源演示，在服务器仿真内扩增。

> 来源纠正：对话中一度将 `tidy_table` 成功率归为“作者 issue 回复”，该归属未核实，应撤回。下述数值来自论文表 2，不是 issue 回复。

## 2. 官方实验对任务选择的启示

论文表 2：D0 生成成功率为 Pick Cup 86%、Tidy Table 80%、Put Dishes Away 38%、Clean Frying Pan 51%。论文 §5.4 的策略实验在 Pick Cup 与 Tidy Table 上使用 1000 条生成示范，训练 WB-VIMA / 微调 π0。

因此本项目应把验收顺序改成：

1. **官方原生 Tidy Table D0 成功一条**：验证多阶段导航→抓取→持物搬运→放置的生成链路。
2. 原生链路通过后，扩大到少量成功 episode，做状态策略的端到端训练 / 评估冒烟；几条数据不声称达到论文策略效果。
3. 最终双臂目标仍是 Dishes Away；单独诊断抓盘后的回缩、持物导航与第二臂 IK，再扩增。
4. 数据生成稳定后再按资源增加成功数据量，最后评估独立布局或随机种子的闭环成功率。

**Tidy Table 是左臂搬杯到水槽，不能当成双臂协作成果；Dishes Away 是双臂分别持盘，不是同步双手抓同一物体。** 这些任务语义在本项目固定 BDDL 和阶段配置中逐一核对。

## 3. 官方生成链路

```text
仓库附带的人类源演示 HDF5
  → 已处理的源数据与阶段配置
  → 随机化任务初始状态
  → 接触操作按对象相对轨迹适配
  → 可达性 / 可见性检查、底盘与机械臂规划
  → 在物理仿真内执行并检查任务目标
  → 保存成功示范的 states / actions / obs
  → 训练学生策略
  → 关闭教师规划器做网络闭环评估
```

上游代码与安装文档匹配当前 R1 路线。不要把原始 R1Pro 的 23 维动作直接塞进现有 R1 的 21 维控制器。

### 必须核对的配置

- 使用 MoMaGen 的原始 `demo_src_r1_TASK_task_D0.json`，不是 `_mimicgen` 或 `_skillgen`。
- 场景实例必须从**固定版本**仓库复制到实际 `OMNIGIBSON_DATA_PATH` 下；存在冲突应拒绝运行，不能覆盖不明版本资产。
- 源演示的 `filter_key=use` 与阶段标注对应，不能随便换成没有 datagen_info 的另一条演示。
- 保留上游硬 / 软可见性设置；不能靠删除约束“修复”任务而不重新标注实验。
- `generation.guarantee=True`：按成功示范数结束；`False`：按尝试数结束。这个语义直接见上游代码注释。
- `max_num_failures` 在此版本主要控制失败数据保留数量，**不是可靠的全局重试上限**。用外层 timeout 限制算力，不让保证成功模式无限跑。
- 以 `important_stats.json` 的成功数与有效 HDF5 episode 为验收，**进程退出 0 不足以判断任务成功**。

## 4. 上一轮失败的已知事实和待检验假设

已知事实：2026-11-14 19:02 开始的 Dishes Away D0 / seed 10 任务共 8 次尝试，0 条成功，生成耗时约 2.64 小时。失败记录包括 5 次 ArmMPIKFailed、2 次 BaseSamplingFailed；还有 1 次没有规划错误分类但任务目标未满足。无训练权重、无学习策略闭环结果。

上轮为了限制成本采用 `guarantee=False`，这只意味着“尝试 8 次”，不是“收集 8 条”。**改回 True 只修正计数与停止条件，并不能修复 IK。**

待检验，而不是已证明的根因：

- 当前场景 / 模型资产与源演示的几何、关节范围是否完全匹配。日志中有 USD hash 警告，需要核对，不能仅凭警告断言一定不兼容。
- 第一臂抓住盘子后，回缩失败是否导致后续持物导航 / 第二臂操作无解。
- 本地轻量观测封装是否改变上游调用假设。先用原生入口排除该因素，不主张它就是已知根因。
- 与论文的采样种子、源演示筛选、随机化范围和运行依赖差异。

当前修复不是“多重跑几次”：先固定版本、场景、源演示与配置，建立无状态观测封装的官方对照；检查哪一阶段首次偏离预期，再做单变量修改。

## 5. 已部署的原生对照入口

新增 `/home/yuanyue/double-arm/scripts/run_official_generation.sh`：

- 检查 MoMaGen、BEHAVIOR-1K、robomimic 的提交号。
- 检查并补齐原始任务场景文件，遇到冲突立即退出。
- **复制未经修改的上游生成配置**，直接执行 `momagen/scripts/generate_dataset.py`。
- 不调用我们的 `generate_state_demos.py`，不改规划器、阶段、控制器或目标判据。
- 保留原生观测，包括后续视觉训练可能需要的图像；不在此对照中裁掉观测。
- 记录源 HDF5 的 SHA-256、随机种子与提交号。
- 外层设置时间上限；结束后使用 `check_generation_result.py` 同时检查进程退出状态、任务成功数与有效 HDF5。
- 仍保留之前只涉及检查视频拼接的 robomimic render 补丁；它不是运动规划修改，不把环境宣称为完全 pristine checkout。

服务器运行命令（**禁止在本机运行仿真**）：

```bash
TASK=tidy_table TARGET_SUCCESSES=1 GENERATION_SEED=0 \
GENERATION_TIMEOUT_SECONDS=2700 CUDA_VISIBLE_DEVICES=0 \
  bash /mnt/yuanyue/double-arm/code/scripts/run_official_generation.sh
```

实际于 2026-11-14 23:22（北京时间）启动，根目录：

```text
/mnt/yuanyue/double-arm/runs/tidy_table-official-20261114-232245-5ufWgQ
```

`manifest.json`、`generation.log`、`process-result.json`、`generation-result.json` 分别记录运行依据、仿真日志、进程结果、数据质量验收。仅在最终 `generation-result.json` 为 `passed` 后，才报告原生数据生成验收通过。**本轮原生生成对照不会自动声称或启动成功的策略训练。**

## 6. 自定义“真正双臂 + 跨房间”任务时怎么办

沿用官方 Custom Tasks 流程，在服务器采集 / 处理与新目标一致的源演示，再定义分阶段的对象引用和终止条件。不能把现有厨房叠盘任务改个名字就称为跨房间收纳，也不能把从冰箱取出瓶子的目标当作送水到人。

本地仅保存源码、配置和此文档。所有仿真、生成数据、训练、权重与长视频仍在 `/mnt/yuanyue/double-arm`。现有本地参考文件和用户未提交改动保留。
