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

BEHAVIOR 默认数据目录可能位于其源码树的 `datasets/` 下；它也在服务器持久化盘，不在本机。根目录的共享盘剩余容量不代表用户配额。

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

## 最小仿真入口（尚未执行验收）

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
