# X7：从双手转向源示范迁移四个新停靠布局

2026-11-18 在新服务器 46152 的 X7 CPU 环境执行。原始 `source-cw` / `source-ccw` 的动作和报告逐项检查哈希；+34° 的 `heldout-angle` 不读取其轨迹文件、不参与迁移、没有覆盖。没有启动 GPU 或策略训练，也没有碰旧服务器的已停任务或新服务器的 R1 迁移。

## 方法与边界

保留从 A 侧取箱、双手合握与松开的源指令。运输段的双臂和夹爪共用同一个底座坐标系下的运输构型，不拼接两条独立手臂。使用源轨迹的各段姿态随时间变化，计算源和目标的路点坐标系刚体变换 `T_target * inverse(T_source)`，在 SE(2) 内插值修正后变换源底座路径。对超参考速度的段做时间拉长，使三轴仍限速并精确到达目标路点。将新动作在带完整碰撞台的 Bullet 场景以 240 Hz 物理实际执行，不写运行中的机器人或箱体位姿；输出 30 Hz 动作、姿态和接触，外部 RGB 5 Hz 用动作索引对齐。

这是 **参考 MoMaGen 的小范围源分段迁移**：A 固定，箱体及把手几何固定，改变 B 的位置、旋转与路径。拒绝不满足双侧持物、悬空、5 mm 站台间隙、释放稳定等条件的候选。不是官方 MoMaGen/OmniGibson 运行、轮胎动力学、真实机器人或学习策略成绩，也未将这四条划为正式训练集。

## 结果

四个新布局角度 **−31° / −21° / +21° / +29°** 全部通过。机器人与站台零接触、零负间距，最近约 **13.44 mm**；所有转向子步有双侧四指接触。共有 **4,937 个控制步 / 824 帧外部 RGB**，四条动作哈希均不同。独立复核按源数据重新算出四条动作并与归档逐条完全一致；再从零重放 −31° 的归档命令，底座/手臂/夹爪、箱体位姿、接触、支承与间距数组逐项一致。另有 4 个 SE(2) 和轨迹契约测试通过。

- 远端 `/mnt/yuanyue/double-arm/runs/x7-turning-se2-20261118-193054`：`dataset/{summary.json,attempts.json,accepted/}`、`qa-se2.json`、`se2-tests.log`。
- 源 `/mnt/yuanyue/double-arm/runs/x7-turning-carry-20261118-192041` 保持只读。`dataset/attempts.json` SHA-256 `b182f9011aa1b11dbe7c6aadf095aa4dbed630486bc2b73516531d69bc22b482`。
- 复现入口：远端 `envs/x7-cpu/bin/python scripts/generate_x7_turning_se2.py ASSET OUT --cad-zip CAD --sources SOURCE_RUN/collection --source-qa SOURCE_RUN/turning-qa.json --rgb`，明确设置 `CUDA_VISIBLE_DEVICES=''`。资产为既有 X7 固定底座派生 URDF 和原始 X7 ZIP。样本与远端环境不提交 Git。
