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
- 已识别 MoMaGen 固定版本及其 BEHAVIOR、Robomimic 子模块版本，见 `config/upstream.env`。
- MoMaGen 源码中已有 `pick_cup` 等任务的处理后源示范，不需要先用遥操作采集才能开始最小复现。

### 当前阶段

- 源码与基础 Python 依赖部署中；尚未完成完整仿真验收。
- 未接受 NVIDIA / BEHAVIOR 许可，未启动授权资产下载。
- 尚无成功生成示范、成功训练或最终策略评测结果。

### 运行日志

远端：

- `/mnt/yuanyue/double-arm/logs/base-python-install.log`
- 后续完整依赖安装和仿真启动分别记录独立日志。

### 网络说明

远端访问部分海外服务较慢。部署时临时通过 SSH 反向 SOCKS 转发访问上游 GitHub；本机只转发字节，不落地下载大型源码／资产，也不运行仿真或训练。

该转发不是长期服务承诺。使用前应测试连通性；训练运行时不应依赖本机维持网络转发。若资产下载中断，应确认下载器支持续传再恢复，不重新盲目下载。

### 许可门槛

固定的 BEHAVIOR 安装器要求 NVIDIA EULA 和 BEHAVIOR 数据许可确认。后者限定非商业学术研究且禁止再分发资产与密钥。在用户明确确认前，不设置自动接受标志。
