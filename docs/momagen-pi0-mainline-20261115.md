# MoMaGen 作者式数据生成 → π₀ LoRA → 联合移动操作

## 用户选定路线与边界

主线正式切换为 **MoMaGen 原始生成算法 → 官方 openpi 的原始 π₀（不是 π₀.₅ / FAST）→ LoRA 微调 → R1 闭环联合控制**。

不再将此前状态 MLP / GRU / RBF / 记忆式 BC 当作目标策略。这些历史结果和失败录像保留，但不冒充作者 π₀ 的复现。

### 固定版本与隔离环境

- MoMaGen：`5da621667669b15b97d7a220d086283b3c8ad0e5`，BEHAVIOR / robomimic / Isaac 版本保持 `config/upstream.env` 不变。
- openpi：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- openpi 所锁 LeRobot：`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。
- Python3.11、JAX0.5.3、Flax0.10.2及依赖安装在服务器独立 `envs/openpi`，不修改原 MoMaGen Python3.10 / Isaac 环境。
- 服务器直连 GitHub 不稳定，已按精确 commit 的官方源码归档流式传入服务器，写 `UPSTREAM_REVISION`。没有把上游仓库、大文件放在本地 checkout。
- 包版本约束从固定 openpi `uv.lock` 导出。使用 `--no-sources` 仅避免同一已验证源码再走 Git 网络拉取；仍用固定本地 openpi / LeRobot source。最初安装因来源 URL 冲突失败，已保留日志并修正；USTC 源403后改用实测可达的清华源。

## 真实数据接口

源 HDF5 已在服务器打开核实，原生成功示范包含：

- head RGB、left wrist RGB、right wrist RGB，均为真实 `256×256×4 uint8`，转换时只丢 alpha；不以零图伪造右腕。
- `prop_state`：21维本体观测。
- `actions`：21维底盘 / 躯干 / 双臂 / 双夹爪原生命令。
- 原任务30Hz动作频率。

π₀适配器仅使用三路RGB、本体状态和任务指令，不使用任务物体坐标、仿真状态、源演示阶段、时间或未来观测。

**本体观测与动作的21维顺序不同，不可逐维默认对应。**

- state：base velocity3、trunk4、left arm6、right arm6、left/right jaw width2。
- action：base3、trunk4、left arm6、left jaw1、right arm6、right jaw1。

本次先保持 MoMaGen 原生 motor action 接口：底盘局部单步位移 / 转角，躯干 / 手臂绝对关节目标，原生夹爪命令；32维 π₀模型空间只做 padding / 截断，不套 ALOHA 转换，也不重复做 joint delta。

**这是一项明确的适配选择，不宣称已逐项复现论文作者内部 π₀ 的绝对关节标签处理。** 先验收 roundtrip / 真正梯度微调，再决定是否需转成统一绝对全身关节标签。源 `env_args` 中旧 base velocity 配置也不能直接当运行时事实：锁定 wrapper 会 reload 为位置控制，已按实际代码核对。

## π₀设置

- 官方 `Pi0Config`，预训练 `pi0_base`，动作维度保持官方32，原生有效21；action horizon **50**。
- 在 VLM 和 action expert 两侧使用 **LoRA rank32**；冻结规则调用官方 `get_freeze_filter()`，EMA关闭。
- 固定 openpi 的默认 VLM LoRA是rank16，适配配置通过构建时覆盖配置到rank32，不修改原模型源码；会在真实配置导入阶段验证。
- 单RTX4090，batch size从1开始；模型与图像规格不为绕开显存而偷偷换小模型。先执行20次真实梯度更新的显存 / 数据联调，再启动30,000次正式微调。
- 官方文档给出LoRA大于22.5GB级别显存需求；这只是计划依据，不等于本机rank32、三图50步配置已实测通过。若OOM，应记录失败并调整批次 / 重计算，不抢GPU0。
- 预训练权重必须从官方公共 bucket 下载并对每个对象核验MD5；缺少权重不允许退化成随机初始化训练并称作 π₀微调。

## 数据分批与诚实验收

第一批目标 **16条新成功视觉示范**，另有原生seed0成功示范1条整回合留出。16条是接口 / 训练试跑批次，**不是论文1000条规模复现**；接口通过后才扩大数据。

转换器只收最终成功 `demo.hdf5` 或已关闭的成功 `tmp/`，跳过partial / failed目录，按动作与初始本体指纹去重。使用本地LeRobot格式，不向HuggingFace上传受限BEHAVIOR数据；训练 / 验证整回合隔离。相机长度、图像尺寸、三相机存在性及采样图像非空进行检查。

作者生成配置使用 assisted attachment，**保留该来源标签，不悄悄称其为纯物理示范**。后续策略评估需要分别报告作者式assisted条件和physical条件；最终对用户提交的物理抓取成功应满足零辅助约束、零物体位姿写入、任务持续成功、可检查的视频。上游夹爪摩擦4.0尚未硬件标定。

## 已启动的服务器任务

```text
/mnt/yuanyue/double-arm/runs/pi0-momagen-20261115-233529
```

- 真实官方生成：`tidy_table-official-20261115-233529-SvPgea`，原始 `generate_dataset.py`，D0、seed600、guarantee=True、目标16成功，生成外层上限4小时。
- GPU只用GPU1 UUID `GPU-50e5d38d-2825-e379-f537-7f0bf4609aba`。GPU0正在跑其他项目，没有本项目进程。
- 约12.0GB原始π₀基础检查点下载到服务器，分对象续传 / MD5核验；服务器GCS连接较慢，下载并发12，不占GPU。
- 独立环境安装和检查点下载与生成并行；JAX GPU训练等待生成结束并检查GPU空闲，绝不与22GB生成仿真争抢同一张卡。
- 新控制器依次等待环境、转换RGB数据、计算归一化统计、等待基础权重、20步微调联调、30k步微调。总上限12小时，自原批次开始计时；修复准备错误没有无限延长预算。
- 仿真闭环部署还需要实现 / 验证跨Python环境的π₀推理与R1控制连接。训练结束状态会明确标为 evaluation_required，**不会从训练完成自动宣称移动操作成功**。

记录时：生成运行中、环境依赖下载中、基础权重下载中；**尚未开始π₀梯度微调，尚无π₀成功策略视频**。

## 官方参考

- MoMaGen论文：https://arxiv.org/html/2510.18316v2
- 作者生成教程：https://chengshuli.github.io/MoMaGen/tutorials/generating-data/
- openpi：https://github.com/Physical-Intelligence/openpi
- 所锁官方训练配置：https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py

### 跨日准备修正（2026-11-16 00:00 CST）

- 独立CPython默认编译器是clang，但服务器仅有GCC；LeRobot依赖的evdev首次构建失败，已经明确设置现有GCC/G++，构建成功。未通过删依赖 / 降级模拟器绕过。
- GCS直连服务器速度慢，已启动**只允许官方公共pi0_base路径、localhost绑定的零磁盘字节中继**，经SSH反向转发到服务器下载器；权重始终写在服务器。中继不接受任意URL、凭据或写请求，本地不缓存权重；接收端仍对官方对象大小和MD5验证。
- 下载已恢复至接近0.9GB，并持续进行；整个模型约12.0GB，尚未宣称下载完成。
- 本批原始MoMaGen前2次尝试均因 `Attached object mismatch` 失败，未写入成功训练数据。失败事实保留，不将“生成器运行”说成“已经有16条成功”。旧原生seed0视觉示范仍只作为整回合留出。
- 后续是否可以进入训练由实际成功视觉数据数量和基础权重完整性决定；不足则明确报告数据 / 权重阶段失败，不能用无图像的状态数据冒充视觉π₀训练输入。
