# MoMaGen 按需迁移：无辅助专家回放优先

全目录历史迁移停止后保留已传的暂存文件，不删除旧源。源机计算已按用户要求暂停；去程是服务器间只读、可续传的迁移，接收内容先写到目的机隔离目录，不覆盖共享项目中的 X7 内容，不切换默认路由。

## 顺序与独立门禁

1. **回放最小输入**：固定原八条跨分片专家轨迹（13567 条原生 21D 动作），每回合只带 frame0 序列化状态、21D proprio、三路原始 RGB 的 frame0 與该 case 环境配置。加起来约 2.7 MB；不包含动作后的 RGB、深度、分割、全量 HDF5 或权重。保持原始序列化布局（源标记为 assisted）；未来实际回放从空手状态恢复后，在任何诊断动作前必须切换 physical 并检查 Python/USD 无辅助约束。
2. **代码与环境**：只迁 pinned MoMaGen、CuRobo、仿真 Python 3.10 环境和需要的工具。仍在网络传输的目录不能视为可用。目的机两种 Python 环境分别做绝对路径/依赖复核，不复制旧进程状态或套用旧 GPU UUID / CPU affinity。当前不迁新的训练池、OpenPI 权重或完整实验档案。
3. **场景 allowlist**：从 `house_single_floor_best.json` 与 `house_single_floor_task_datagen_tidy_table_0_0_template.json` 的原始 object registry 提取 365 个直接引用对象模型，加 R1、相关场景布局、元数据和系统，共 374 个明确相对路径。搬运所选模型目录内的材料和纹理；未复制其他场景、登录凭据、BEHAVIOR 解密密钥或本地许可确认。名单覆盖静态直接引用，不保证运行时传递依赖齐备；首次启动前复核已有许可与资产访问配置，不代用户接受新条款。
4. **可启动不是完成**：验收顺序是接收文件完整 → 目的机依赖导入 → 场景/headless 相机启动 → 无辅助专家回放。两台机器的活跃任务状态不得混合。仅在核验实际完成启用新任务，不能用旧日志或 port 改动代替验收。

## 目的机的紧凑输入读取（CPU）

```bash
CUDA_VISIBLE_DEVICES='' python scripts/verify_portable_expert_inputs.py \
  --bundle /path/to/migrated/compact-panel \
  --original-preflight /path/to/migrated/source-preflight/report.json
```

校验固定八条名单、原始 CPU preflight SHA、每条 action/frame0/RGB 数组字节和环境 JSON。目前目的机 CPU 校验 **8/8、13567 动作**通过；未导入 Isaac、未启动 GPU 工作、未产生新回放结果。便于后续 simulator worker 直接使用 `load_case()`，无需伪造旧完整 HDF5。移植路由配置信息、实际 staging 路径及传输状态仅保存在主机本地或服务器，不提交媒体、秘钥、资产与数据。

## 冲突与切换前核查

`audit_momagen_migration_cutover.py --root … --staging …` 只读列出待落地的六个命名空间叶目录、是否与目的机现有文件冲突，以及 runtime/assets/compact 三个独立状态。根下已有的 `code` 属于并行 X7 工作流，**不是合并或覆盖目标**；MoMaGen 的代码保持独立。脚本仅查 `data/omnigibson.key` 是否已由有权限的用户以私有权限配置，不读取、不复制、不出示它。工具没有任何启动、切换、拷密钥或写 `latest` 指针的路径。目的机运行一次，现有六处预定落盘路径无冲突，但 runtime / scene 数据仍在传输，且新机关联的受限资产访问尚未配置；未激活或占卡。
