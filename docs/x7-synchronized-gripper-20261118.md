# X7：夹指局部运动排查与同步夹爪 v2

## 旧视频中发生了什么

对 2026-11-18 的学得策略视频 `learned-heldout-900.mp4`，用其 317 条保存命令作逐步物理重放。关节、方块位姿、接触等七个数组与视频对应原始轨迹精确一致。结果将运动拆为**夹爪本体相对桌面**与**夹指相对腕部**：

- 爪座通过 fixed joint 固定到 Link7，无额外平面自由度。在 4.40–6.37 s 搬运中，爪座在桌面 XY 平移约 **168 mm**，来自 Joint1–7；此时共享夹爪电机目标恒定。合爪阶段，爪座 XY 净位移仅 **0.048 mm**，实测夹指内间隙从约 **76 mm** 变为 **44 mm**，与夹住 44 mm 方块一致。
- 两根旧夹指只沿腕部局部 X 轴朝相反方向滑行，但各有独立位置电机，并非机械联动。原视频约 **6.33 s** 处，当共享电机目标为 6 mm，两条滑轨实测位置约 30.01 / 6.02 mm，夹口中心相对腕部偏了 **11.995 mm**。这个额外偏移确实存在，不能只归因于观察角度。它不是夹爪新增了平面控制指令。
- 原视频、源数据和 v1 checkpoint 保留不变。实测报告位于旧 run 的 `gripper-motion-audit/report.json`，SHA-256 `05415937e591efecf8e264dfaa46b3d7e854a089d91a74e74fd20f7aaaa1701f`。

## v2 改了什么，验了什么

`x7_sync_gripper_preflight.py` 使用独立派生模型：只驱动 A，B 由仿真的机器人内部齿轮约束带动；两根夹指仍沿镜像方向开合。每个受控物理子步检查只有这一个约束，且端点、类型、齿轮比、力限和目标与指定的内部传动完全一致。它不连方块，也不吸附。对称门槛预设为开口中心最大偏差 0.5 mm。物体和机器人只在初始化时设置位姿。电机、力限和约束均是仿真设定，不是实物夹爪参数。

| 远端 CPU 实验 | 物理拾放与对称门槛 | 最大中心偏差 |
| --- | ---: | ---: |
| v2 脚本专家，初始方块沿合爪轴偏置 0 / −8 / +8 mm | 3/3 | 0.0364 mm |
| v2 使用旧 seed 900 保存的相同命令，偏置 0 / −8 / +8 mm | 3/3 | 0.0663 mm |
| 旧视频 seed 900，原始 v1 对照，无偏置 | 原回合放置成功，但非同步爪 | 11.995 mm |

无偏置的同命令比较是 **11.995 mm → 0.0414 mm**。v2 此回合抬升 97.47 mm，双指接触下完成搬运，松爪后稳定在桌面目标内 30 个控制周期。六个 v2 检查各执行 2,536 个物理子步，内部传动约束逐子步通过，未发现连物体的约束。旧模式兼容性通过 seed 700 逐步精确重放；新增 6 个契约测试在服务器全部通过。

上表后两种对照不构成新版学习策略成绩。特别是 v2 同命令实验只是**开环重放旧命令做机械对照**，未在 v2 上采集新版训练集或训练。下一步在 v2 单独采集新数据，版本化划分，再测试不同初始朝向、偏置和目标，而不是复用 v1 权重冒充新版结果。

## 远端证据与复现

运行根 `/mnt/yuanyue/double-arm/runs/x7-sync-gripper-20261118-164514`：

- `final-expert/report.json`：脚本专家三种偏置。
- `final-tape/report.json`：匹配命令的三个偏置。
- `final-review/synchronized-review.mp4`：同命令、零偏置的 v2 回放，片头明确标为 `RECORDED COMMANDS`。
- `rack-test.log`、`v1-replay-regression.log`：约束契约测试和 v1 回归。

使用已安装的 PyBullet 3.2.7 CPU 环境。仅在远端运行，例如：

```bash
R=/mnt/yuanyue/double-arm
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
nice -n 15 taskset -c 15 "$R/envs/momagen/bin/python" \
  /absolute/path/to/checkout/scripts/x7_sync_gripper_preflight.py \
  "$R/runs/x7-fixed-base-20261118-154526/final/derived-model/x7_duo_fixed_base.urdf" \
  "$R/runs/x7-sync-gripper-example" --offset-mm 0 -8 8
```

原始 X7 ZIP、STL、生成模型、权重、轨迹和视频不入 Git。这个检查未修改 R1 主线、当前 GPU 作业或厂商资产。
