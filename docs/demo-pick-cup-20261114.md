# MoMaGen 取杯 demo：2026-11-14 实际运行结果

## 结论与边界

在服务器完成了 `r1_pick_cup_D0` 的一次真实示范生成：底盘规划、机械臂规划、抓取回放均完成，任务判定成功，动作和状态已保存。

- 本次生成尝试：1 次；成功：1 条；问题轨迹：0。
- 记录：456 步，动作和状态长度一致。
- 主要抓取由左臂完成，右臂没有抓取物体；不是双臂协同搬运验证。
- 这是 MoMaGen 的示范生成，不是训练后的学习策略评测，未开始策略训练。
- 验证的是室内局部底盘移动／站位调整和取杯，不据此宣称跨房间导航或开放世界泛化。

## 可检查的视频

| 版本 | 含义 | 参数 |
| --- | --- | --- |
| 原始视角检查副本 | 来自本次实际生成录像，检查包做过 H.264 压缩；单一机器人后方跟随视角 | 720×720，20 fps，456 帧，22.8 秒，633,982 字节 |
| 检查版 | 对本次成功生成的状态重新渲染；固定全景＋左腕近景，不重新规划／运行策略 | 1334×508，20 fps，456 帧，22.8 秒，917,501 字节 |

检查版更容易观察全身底盘移动和杯子／夹爪接触。保留原始录像用于对照，不将检查版称为新一次闭环执行。

本地检查包（位于 Git 忽略的 `outputs/`，不进入仓库）：

- 检查视频：`/home/yuanyue/double-arm/outputs/demo-review/momagen-pick-cup-wide/demo.mp4`
- 六帧缩略图：`/home/yuanyue/double-arm/outputs/demo-review/momagen-pick-cup-wide/contact-sheet.jpg`
- 数据、视频与回放方式说明：同目录 `review.json`。
- 原始视角检查副本：`/home/yuanyue/double-arm/outputs/demo-review/pick_cup-smoke-20261114-164009-xU5fGd/demo.mp4`

## 独立检查的运动数据

以下来自生成 HDF5，而不是从视频估计：

| 指标 | 值 |
| --- | --- |
| 底盘起终点平移距离 | 0.639628 m |
| 底盘朝向变化 | −48.58848° |
| 杯子高度变化 | +0.200124 m |
| 最终左手抓取状态 | TRUE，枚举值 1 |
| 最终右手抓取状态 | FALSE，枚举值 −1 |

注意 OmniGibson 的 `IsGraspingState` 是 `FALSE=-1 / UNKNOWN=0 / TRUE=1`，不能用通用 `bool(value)` 解码；代码和测试已显式处理，避免把右手误报为抓取。

## 服务器保留的完整证据

运行根目录：

```text
/mnt/yuanyue/double-arm/runs/pick_cup-smoke-20261114-164009-xU5fGd
```

- `generated/demo_src_r1_pick_cup_task_D0/important_stats.json`：成功统计和阶段日志。
- `generated/demo_src_r1_pick_cup_task_D0/demo.hdf5`：生成数据，310,546,718 字节，未回传本地。
- `generated/demo_src_r1_pick_cup_task_D0/videos/0000.mp4`：原始录像。
- `generated-inspection.mp4`：新观察角度的状态回放。
- `generated-inspection.json`：回放数据来源、状态数、镜头布局与数据 SHA-256。
- `review-final/review.json`：任务、源视频、动作长度、运动数据、视频元信息。
- `inspection-render.log`、`generation.log`：生成和检查渲染日志。

检查回放绑定的数据 SHA-256：

```text
72e2b45dfa6dcf5d7f3869f5d6c0ffaa3e25ef09c95da134edf322cd9ee305bd
```

该哈希是生成 HDF5 的校验，不是资产许可密钥。

## 做过的验证

1. 原任务成功统计为 `num_success=1 / num_attempts=1 / num_problematic=0`。
2. HDF5 中存在 `demo_0`，动作和状态均为 456 步。
3. 日志记录底盘 MP 和机械臂 MP 成功，实际抓取完成。
4. 检查版绑定同一生成 HDF5 的 SHA-256，回放 456 个状态，不跳帧。
5. 原始及检查版视频均经过完整解码检查。
6. 已抽查两版各六帧：原始录像的背后视角有遮挡，因此额外制作固定全景和腕部视角；检查版可见底盘位置／朝向变化以及杯子进入夹爪近景。
7. 回放与导出只在服务器运行。本地仅保存两个小型检查包（合计约 2 MB），没有同步资产、HDF5 或模型。

单条示范成功不代表稳定成功率，更不代表训练后策略表现；用户视频检查仍待反馈。
