# X7：双手持箱、转向、斜向停靠 B 台

在新服务器 46152 的独立 X7 CPU 环境执行。基于严格碰撞版 `462bbd3`，保留真实碰撞台结构和原始 CAD 包络；不改直线数据、不占用 GPU、不改 R1 环境。将 B 台整个刚体旋转（台面、腿、梁和撑条的渲染及碰撞同转），底座以受力和限速的三个平面关节追踪 X、Y、yaw。不瞬移物体或底座，不给物体添加约束。这是平面底座代理，不是实物车轮、轮胎滑移或导航验收。

## 持物运动验收

路线是：双手合握并抬离 A 台 → 离开 A 台 → 到转向区转向 → 到 B 台后方 → 按 B 台局部法向进站 → 双爪松开、物体稳定留台。两臂在搬运段保持相对底座的持箱构型；这个设计检查真实的共同载荷与运动，不声称两臂已经学会动态调整抓持。每个物理子步均检查整机对两工位的有符号距离、实际接触及内部夹爪约束只连机器人夹指。

| 回合 | 用途 | 最终底座转角 | 真实 XY 路程 | 到位后箱子中心误差 | 最近站台距离 |
|---|---|---:|---:|---:|---:|
| `source-cw` | 源示范 | −25° | 4.63 m | 1.80 mm | 13.46 mm |
| `source-ccw` | 源示范 | +25° | 4.18 m | 0.51 mm | 13.46 mm |
| `heldout-angle` | 独立 B 工位位置与转角，不作源 | +34° | 4.45 m | 2.12 mm | 13.45 mm |

以上各回合物体全程未触地，转向采样子步均有四指接触，转向控制帧的箱子无桌面支承。机器人–站台零接触/负距离；每台实测最小距离均超过 5 mm。松开后稳定 32 个 30 Hz 控制帧。不是实物认证，也不是对连续时间全部未建模几何的证明。

## 证据与用法

远端 `/mnt/yuanyue/double-arm/runs/x7-turning-carry-20261118-192041/`。`probe/` 是独立的无相机预检，`collection/` 为相同布局的独立采集；回合分 `source/accepted` 和 `heldout/accepted`。`report.json` 带夹紧、悬空抬升、持物转向、落位松爪的实测事件，以及路线子区间、三个物体/工位阶段、归档哈希。三条采集合计 **3,671 控制步、613 帧 RGB**。RGB 在对应动作前以 5 Hz 采样，通过 `rgb_action_index` 对齐到 30 Hz 动作，随 NPZ 只保留在服务器。

`verify_x7_turning_sources.py` 核对源/留出布局、报告指纹、图像对齐、实测事件并比较独立预检动力学。三条记录各自的 11 个动作／轨迹／接触／间距数组逐项一致。转向任务使用 22 维命令（底座 3 个位置参考／3 个速度前馈＋14 个臂轴＋2 个夹爪目标），**不混入旧 18 维直线数据集**。目前是脚本源采集与样机几何任务预检，不是官方 MoMaGen 执行、闭环策略训练或成功率评估。下一步将两个转向源用于 SE(2) 分段迁移和真实物理筛选，不把 +34° 用作源。

```bash
# Remote only. Use the X7 environment, not the R1/Isaac environment.
R=/mnt/yuanyue/double-arm
X="$R/runs/x7-fixed-base-20261118-154526/final/derived-model/x7_duo_fixed_base.urdf"
CAD="$R/data/local-reference-imports/x7-fixed-base/source-bd786d16bf99da35af094eeb4cdc69b9f96721092a9598748d782d8433288590.zip"
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$R/envs/x7-cpu/bin/python" /absolute/path/to/checkout/scripts/run_x7_turning_carry.py \
  "$X" "$R/runs/x7-turning-reproduction" --cad-zip "$CAD" --rgb --review heldout-angle
```

转向路点、终点时刻和速度边界，以及旋转后台面、腿、梁、横撑的渲染/碰撞配对，共 4 个契约测试在新服务器通过。直行 seed 0 独立回归所得 NPZ 与既有严格版逐字节相同（SHA-256 `b55181fb00d1f922465b4924860a33f326eae39db6777bf55c046c81379747cd`）。
