# 双臂轮式移动操作：MoMaGen 远端仿真与训练

代码仓库：<https://github.com/WM4N-Group/double-arm>

本项目保存 MoMaGen 的部署脚本、版本配置、实验记录和调研文档。**本地仅编辑和同步轻量文件；仿真、数据生成、训练全部在服务器运行。**

## 当前路线

```text
本地轻量 Git 工作区 → 服务器代码副本
                         ├── MoMaGen（示范生成）
                         ├── 固定版本 BEHAVIOR-1K / OmniGibson
                         ├── Isaac Sim 4.5（需许可确认）
                         └── Robomimic（学习与数据工具）
```

- 上游代码：<https://github.com/ChengshuLi/MoMaGen>
- 上游安装文档：<https://chengshuli.github.io/MoMaGen/installation/>
- 数据生成教程：<https://chengshuli.github.io/MoMaGen/tutorials/generating-data/>
- 固定版本：[config/upstream.env](config/upstream.env)
- 部署记录：[docs/deployment.md](docs/deployment.md)

**MoMaGen 是示范生成系统，不等于开箱即用的端到端强化学习工程。**先验收仿真和单条示范，再接策略训练。

## 存储约定

服务器持久化根目录：`/mnt/yuanyue/double-arm`。

| 子目录 | 内容 |
| --- | --- |
| `code/` | 本仓库轻量文件的部署副本 |
| `src/` | MoMaGen 与固定版本的依赖源码 |
| `envs/` | Python 独立环境 |
| `data/` | 额外数据、资产与后续训练集 |
| `runs/` | 实验输出、模型、视频 |
| `cache/` | 包下载、编译和运行缓存 |
| `logs/` | 安装、诊断、运行日志 |

运行脚本设置 `OMNIGIBSON_DATA_PATH` 为远端 `data/`，统一放置机器人、场景和任务实例。根目录的共享盘剩余容量不代表用户配额。

本地已有 PDF 保留，但 `.gitignore` 排除 PDF、模型、数据、视频和环境，不推到 GitHub。不在本机安装 Isaac、CUDA 工具链或训练依赖。

## 本地连接与同步

```bash
cp config/server.example.env config/server.local.env
# 编辑 server.local.env；不要写入密码。使用交互 SSH 或已有 SSH key。
./scripts/remote.sh hostname
./scripts/sync_code.sh
```

`server.local.env` 不提交 Git。同步脚本只上传 Git 跟踪的文件和未忽略的轻量文件，不删除远端数据。请勿强制跟踪被忽略的大文件。

## 服务器上的阶段

以下命令只能在服务器执行，所有路径均指向持久化盘：

```bash
# 阶段 1：源码固定、Python 独立环境与依赖
bash /mnt/yuanyue/double-arm/code/scripts/bootstrap_sources.sh
bash /mnt/yuanyue/double-arm/code/scripts/install_python_deps.sh

# 阶段 2：诊断（不是完整仿真验收）
bash /mnt/yuanyue/double-arm/code/scripts/check_server.sh
```

安装网络需要可访问 GitHub、Python 包源、NVIDIA 和 BEHAVIOR 资产源。源码与包均直接下载到服务器；临时网络转发可能需要本机保持在线，这不表示仿真或训练在本机运行。

## 许可确认：仿真器和资产安装前的门槛

上游安装脚本要求接受：

1. [NVIDIA Isaac Sim 许可](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement)。
2. BEHAVIOR Data Bundle 条款，见固定版本的 `BEHAVIOR-1K/setup.sh` 完整文本：**仅限非商业学术研究使用，不得再分发资产或密钥，不得提取或逆向加密资产。**

脚本默认**不代用户接受**。只有得到明确确认后，才在服务器进程中设置两个 acceptance 环境变量并运行：

```bash
# 仅在用户明确同意许可且用途符合要求之后执行；不写入公共仓库。
ACCEPT_NVIDIA_EULA=YES ACCEPT_BEHAVIOR_DATASET_TERMS=YES \
  bash /mnt/yuanyue/double-arm/code/scripts/install_simulator_assets.sh
```

此阶段可能下载较大资产并编译运动规划依赖，需要查看远端日志；不能只凭 pip 安装成功声称仿真已经跑通。

## 有限部署流水线与状态查看

许可确认后，在服务器记录接受标志到 `code/config/licenses.local.env`（权限 600，Git 忽略），必要时将网络入口设置到 `code/config/runtime.local.env`。不要将密码、数据密钥、资产放入这些可公开的示例或 Git。

```bash
# 在服务器执行；后台运行，关闭本地编辑器不会停止任务。
nohup bash /mnt/yuanyue/double-arm/code/scripts/run_server_pipeline.sh \
  > /mnt/yuanyue/double-arm/logs/pipeline.log 2>&1 < /dev/null &

# 在本地执行：只通过 SSH 读取小型状态和日志，不运行仿真。
./scripts/show_remote_status.sh
```

流水线有单实例锁，按阶段记录 `logs/pipeline-status.json`。它只执行一次部署与有时间上限的最小任务，不是定时监控或长期训练。失败后保留日志并停止；不要仅看进程退出码之外的“安装完成”文字。

资产下载可能较久：固定场景包约 31.28 GB，另有约 0.5 GB 机器人资产和 0.1 GB 任务实例，解包还需额外空间。所有资产在服务器下载和 SHA-256 校验，Git 只保存小型版本／校验清单。

## 最小仿真入口

许可确认、安装和资产准备完成后，在服务器执行：

```bash
source /mnt/yuanyue/double-arm/code/scripts/server_env.sh
# 接受标志仅在明确确认许可后设置。
ACCEPT_NVIDIA_EULA=YES python /mnt/yuanyue/double-arm/code/scripts/isaac_smoke.py
ACCEPT_NVIDIA_EULA=YES ACCEPT_BEHAVIOR_DATASET_TERMS=YES \
  bash /mnt/yuanyue/double-arm/code/scripts/run_momagen_smoke.sh
```

第二个脚本只做一次有时间上限的 `pick_cup` 生成尝试，输出到唯一命名的 `runs/` 子目录，不删除旧实验。只有统计显示成功并生成非空 HDF5，才报告示范生成成功。首次尝试可能失败，需按日志诊断，而不是视为环境必定已跑通。

## 验收顺序

1. GPU / Vulkan 可见。
2. 固定源码与 Python 环境准备完成。
3. Isaac Sim 无界面启动成功。
4. BEHAVIOR 场景与 Galaxea R1 加载成功。
5. `pick_cup` 至少生成一条成功示范，保存视频／统计／HDF5。
6. 更完整的移动＋双臂操作任务通过。
7. 再建立训练集、策略训练和独立评测。

每一步分别记录，不将“源码已下载”“依赖已安装”“仿真已启动”“任务已成功”“模型已训练”混为一谈。

## 轻量工作流检查

```bash
python3 -m unittest discover -s tests -v
```

这里只检查脚本语法、SSH 参数转义、固定版本和忽略规则，不启动仿真，也不在本机训练。

## 导出实际成功 demo 的检查视频

`run_momagen_smoke.sh` 记录原项目的单一第三人称跟随视角。不能从 `camera_names` 推断实际视频布局：本版本的实时录像由原项目另一条录制路径写出，不使用 `Env.render()` 拼接。`patches/robomimic/` 中的布局补丁仅影响后者；它不修改控制、相机标定或感知分辨率。不会将源示范回放或空场景渲染当成任务成功。

成功运行后会自动导出检查包到该次运行的 `review/`。若需要补导出（目标目录尚不存在），在服务器执行：

```bash
source /mnt/yuanyue/double-arm/code/scripts/server_env.sh
python /mnt/yuanyue/double-arm/code/scripts/export_demo_review.py \
  --run-dir /mnt/yuanyue/double-arm/runs/pick_cup-smoke-实际运行目录
```

脚本要求成功统计、非空生成动作数据、对应的尝试视频均存在；检查视频可完整解码后才生成 `review/demo.mp4`、六帧缩略图和 `review.json`。检查包不包含资产、密钥、数据集或模型。

在本地仅回传上述三个小文件：

```bash
bash /home/yuanyue/double-arm/scripts/fetch_demo_review.sh \
  /mnt/yuanyue/double-arm/runs/pick_cup-smoke-实际运行目录/review
```

文件位于本地 `outputs/demo-review/`，不进入 Git。完整资产、HDF5 与运行日志继续保留在服务器。示范生成成功不代表训练策略已学会该任务。

### 更清楚的全景＋腕部检查版

本次已用成功生成的 456 个状态制作固定全景＋左腕回放，原始生成录像同时保留。该回放不重新执行规划或评估策略；成功统计来自原始实际生成。

```bash
# 服务器：加载成功生成的 HDF5，只重放状态、改变观察镜头。
python /mnt/yuanyue/double-arm/code/scripts/render_generated_demo.py \
  --dataset /mnt/yuanyue/double-arm/runs/实际目录/generated/实际实验名/demo.hdf5 \
  --output /mnt/yuanyue/double-arm/runs/实际目录/generated-inspection.mp4

# 验证回放清单、数据集 SHA-256 和步数对应关系后导出新检查包。
python /mnt/yuanyue/double-arm/code/scripts/export_demo_review.py \
  --run-dir /mnt/yuanyue/double-arm/runs/实际目录 \
  --source-video /mnt/yuanyue/double-arm/runs/实际目录/generated-inspection.mp4 \
  --output /mnt/yuanyue/double-arm/runs/实际目录/review-wide
```

运行前需 source 服务器环境并载入已确认的许可标志。已存在的输出不覆盖，需换新名称。检查版及真实结果详见 [本次 demo 记录](docs/demo-pick-cup-20261114.md)。

## 长程数据与学习策略实验（2026-11-14）

候选数据、R1 / R1Pro 动作兼容性、四阶段叠盘任务和训练 / 闭环评估协议：
[docs/data-and-long-horizon-policy-20261114.md](docs/data-and-long-horizon-policy-20261114.md)。

新增服务器入口为 `run_long_task_data.sh`（有限示范生成）和
`run_state_policy_pipeline.sh`（生成结束后训练状态 BC，再关闭规划器做闭环评估）。
**状态输入基线不等于视觉策略；训练完成不等于任务成功。**
