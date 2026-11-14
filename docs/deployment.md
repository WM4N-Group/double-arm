# 部署记录

## 2026-11-14：首次部署准备

### 已确认

- 本地代码目录：`/home/yuanyue/double-arm`。
- Git 远端：`git@github.com:WM4N-Group/double-arm.git`。
- 服务器系统：Ubuntu 22.04.4；系统 Python 3.10.12。
- GPU：2 × NVIDIA GeForce RTX 4090，每张约 24 GB；驱动 580.82.07。
- 容器根文件系统约 100 GB，不用它承载大型资产和训练结果。
- 持久化挂载：`/mnt/yuanyue`（JuiceFS）；项目使用 `/mnt/yuanyue/double-arm`。
- 已安装缺失的 Vulkan loader、EGL 与诊断工具；`vulkaninfo --summary` 实际枚举到两张 RTX 4090。
- 已在持久化盘建立 Python 3.10 虚拟环境和 uv 工具，不修改本地 Python。
- MoMaGen、BEHAVIOR-1K 与 Robomimic 的固定版本源码已在服务器检出，三者 commit 与 `config/upstream.env` 一致。
- 已读取 `r1_pick_cup.hdf5`：包含 `data`、`mask` 分组和 15 条源示范；不是 Git LFS 占位文件。
- PyTorch 2.6.0+cu124、Torchvision 0.21.0、Torchaudio 2.6.0 与基础数值依赖已装入远端独立环境；两张 GPU 分别通过 128×128 矩阵乘法检查。

### 当前阶段

- 基础 CUDA/PyTorch 依赖与固定版本源码已完成；MoMaGen 完整 Python 依赖安装已在服务器后台启动，结果以 `momagen-python-install.log` 为准。尚未完成完整仿真验收。
- 未接受 NVIDIA Isaac Sim / BEHAVIOR 数据许可，未安装 Isaac Sim、未启动授权资产下载。
- 尚无成功生成示范、成功训练或最终策略评测结果。

### 运行日志

远端：

- `/mnt/yuanyue/double-arm/logs/base-python-install.log`
- `/mnt/yuanyue/double-arm/logs/momagen-python-install.log`（后台完整 Python 依赖安装）
- 后续仿真启动单独记录日志。

### 网络说明

远端访问部分海外服务较慢。部署时临时通过 SSH 反向 SOCKS 转发访问上游 GitHub；本机只转发字节，不落地下载大型源码／资产，也不运行仿真或训练。

该转发不是长期服务承诺。使用前应测试连通性；训练运行时不应依赖本机维持网络转发。若资产下载中断，应确认下载器支持续传再恢复，不重新盲目下载。

### 许可门槛

固定的 BEHAVIOR 安装器要求 NVIDIA EULA 和 BEHAVIOR 数据许可确认。后者限定非商业学术研究且禁止再分发资产与密钥。在用户明确确认前，不设置自动接受标志。

### 轻量化措施

- BEHAVIOR 与 Robomimic 使用固定 commit 的部分克隆／稀疏检出，保留运行所需源码，跳过无关文档图片和 JoyLo sampled_task 大量示例。
- 本地新增的代码、配置、文档体积很小；原有约 92 MB PDF 不改动、不提交。
- 本地测试仅做 shell 语法、Python AST、SSH 参数转义、Git 忽略规则检查，没有本地仿真／训练。

### 已验证的代码工作流

- GitHub `main` 已初始化并推送成功；不包含用户 PDF、数据、环境或凭据。
- 本地 6 项轻量工作流测试通过。
- 服务器上验证许可门槛：未设置接受标志时，安装脚本按预期以状态码 3 退出，不启动 Isaac／资产下载。
- `scripts/isaac_smoke.py`、`scripts/run_momagen_smoke.sh` 是后续验收入口，已做语法检查，但尚未完成实际仿真验证。

### Python 镜像限流与重试

- 完整 Python 依赖的首次安装在 `termcolor` wheel 下载时收到 USTC 镜像 HTTP 429，安装未成功，不将其报告为已完成。
- 首次日志保留为 `logs/momagen-python-install.attempt1.log`。
- 已降低 uv 下载／构建／安装并发并设置重试、超时参数，重新启动后台安装；当前进度查看 `logs/momagen-python-install.log`。
- 本地新增源码、配置和 Git 元数据合计不到 1 MB（不含用户原有 PDF）。

## 2026-11-14：用户确认许可后继续部署

以下记录覆盖上文的初始“等待许可”状态；前文保留为部署过程记录。

- 用户已在本聊天明确确认 NVIDIA Isaac Sim 和 BEHAVIOR 数据许可，以及非商业学术研究用途。
- 接受标志只保存于服务器权限为 600 的 `code/config/licenses.local.env`；没有放入 Git。
- 完整 Python 依赖安装已经成功；`uv pip check` 检查通过（226 个包）。
- 已从 NVIDIA 官方源下载固定安装器列出的全部 26 个 wheel，Isaac Sim 正在远端独立环境内安装；不得据此提前声称无界面启动成功。
- cuRobo 固定源码为 `d083916436f1f749cdd351bdf96c4c5df4e6d192`，跳过未使用的示例 LFS 资产，待编译。
- OMPL 使用上游指定的 1.6.0 wheel。
- 资产包固定为 `behavior-1k-assets-3.7.2rc1.zip`，不是最新 3.9 系列；任务实例固定使用 2025 版。
- 从 Hugging Face 官方 API 获取并保存了资产仓库 commit、文件大小与 LFS SHA-256，见 `config/assets.json`。
- 服务器直连 Hugging Face 不稳定，下载使用服务器本身可访问的镜像入口；最终文件必须匹配官方 API 的 SHA-256。下载、解包都不在本地进行。
- 资产改用 aria2 分段续传，保留并复用首次顺序下载的数据。
- 已启动服务器上的有限流水线：安装确认 → Isaac 无界面检查 → 运动规划依赖 → 等待资产 → 校验解包 → 一次限时 `pick_cup` 生成尝试。
- 代码同步改为逐文件原子替换，不截断正在运行的脚本；不删除远端数据或私有配置。

### 状态与日志

- 总状态：`/mnt/yuanyue/double-arm/logs/pipeline-status.json`。
- 总日志：`/mnt/yuanyue/double-arm/logs/pipeline.log`。
- Isaac 安装：`logs/isaac-install.log`。
- Isaac 无界面启动：`logs/isaac-smoke.log`。
- 运动规划安装：`logs/motion-install.log`。
- 资产下载：`logs/asset-download.log`；资产校验、解包：`logs/asset-install.log`。
- MoMaGen 尝试：`logs/pick-cup-smoke.log`，实验输出在 `runs/pick_cup-smoke-*`。

每个阶段失败会停止后续步骤并保留日志。此流水线不启动长期训练；成功生成示范后还需要独立的策略训练与评测。

### 2026-11-14 15:56（上海时间）阶段验收

- **Isaac Sim 安装通过**：全部 26 个固定 wheel 已安装，`import isaacsim` 成功。
- **无界面启动通过**：应用完成启动、执行 30 次更新并正常退出。首次启动编译驱动着色器缓存，耗时约 6 分钟；不将 `app ready` 提前当作最终成功。
- **GPU 渲染通过**：另一次检查产生了非空、非恒定的 320×240 RGBA 图像，保存于服务器 `runs/isaac-render-check/cube.png`，结果在同目录 `result.json`。这只是合成立方体渲染，不是 R1／MoMaGen 任务成功。
- 补齐了首轮启动发现的 `libGLU.so.1`，系统依赖固化在 `scripts/install_system_deps.sh`。
- 额外渲染检查首次遇到 Replicator API 名称不匹配，已根据安装的 4.5 SDK 改用实际存在的 `rep.randomizer.materials`，重新运行并通过。
- 修正仿真验收的异常处理：保留异常退出，并使用唯一 JSON 结果文件验证，不仅依赖可能被 Kit 关闭流程覆盖的进程退出码。
- 运动规划安装首次遇到 Google Storage 下载超时留下的零字节 OMPL 文件。保留失败日志，改用临时转发获取官方 wheel，并确认大小为 26,787,172 字节且 ZIP 有效；后续安装先检查 ZIP，再原子替换最终文件。
- 服务器上已重新启动有限流水线；当前未验收 R1 场景任务，也未开始策略训练。
- 本地 15 项轻量测试通过，包括代码同步原子性、路径穿越防护、校验参数、固定版本与忽略规则。

故障记录保留在服务器：`logs/pipeline.original-attempt.log`、`logs/motion-install.attempt1.log`、`logs/isaac-render-smoke.attempt1.log`。后续状态仍以 `logs/pipeline-status.json` 和实际生成结果为准。

## 2026-11-14：继续推进实际 demo 与检查视频

- cuRobo 和 OMPL 的编译、导入检查均已通过。
- 三个资产包均已完成 SHA-256 校验并成功解包；许可密钥由官方函数安装并校验格式，保持权限 600，内容不记录到文档或 Git。
- 共享盘串行解包在小文件 I/O 上较慢，改为最多 8 线程；显式复用了同一已校验归档的解包暂存目录，保留完整文件，重新解包未写完的文件。资产目录仍在所有任务完成后原子切换。
- MoMaGen 配置生成与 CLI 入口预检查通过。
- 第一次实际启动发现稀疏检出漏掉了 OmniGibson 运行时需要的 `docs/assets/OmniGibson_logo.png`。已从固定 commit 恢复并比较 Git blob hash，未替换为自制图标；检查固化到 `scripts/ensure_runtime_files.sh`。
- OmniGibson 专用 Kit profile 首次补装了原配置需要的 Flow 扩展；不将基础 Isaac 启动通过等同于这个 profile 的端到端成功。
- 取杯任务在服务器进入真实场景中的生成尝试；后续成功与否以该运行 `important_stats.json`、HDF5 和实际视频为准。
- 视频补丁仅将第三人称画面放在拼接首位，保留其原生清晰度；没有修改传感器标定、感知分辨率、控制、规划或任务成功条件。配置只保留一个渲染标签，避免重复拼接。
- 新增检查包导出和回传脚本：要求成功统计、非空生成动作数据、对应视频，并通过完整解码检查；只回传 MP4、六帧缩略图和 JSON。
- 避免流水线重启后无谓重编译：仅在干净、固定版本的 cuRobo 已装入当前环境且扩展齐全时复用，再执行实际导入检查。
- 目前未训练策略。检查视频不得标为学习策略自主执行。
- 本地轻量测试扩展到 22 项，通过了视频成功／失败匹配、补丁可重复应用、解包恢复与安全同步等检查。

## 2026-11-14：取杯生成和检查视频完成

`r1_pick_cup_D0` 在服务器的一次实际任务尝试成功，保存 456 步，日志无底盘／机械臂规划失败。完整统计、原始视频、新角度状态回放与验证边界见 `docs/demo-pick-cup-20261114.md`。

原始视频实际是单一跟随视角，不能从配置渲染标签推断为双视角。已更正导出清单，另用同一生成数据渲染固定全景＋左腕视角；保留 SHA-256 和状态数的对应关系。两版视频均可完整解码，已抽查各六帧。当前仍未训练策略，也未验证双臂共同搬运任务。

本地轻量测试增加至 23 项，其中专门验证 `IsGraspingState.FALSE=-1` 不会被当作通用布尔真值。
