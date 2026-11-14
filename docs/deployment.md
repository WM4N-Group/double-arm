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
