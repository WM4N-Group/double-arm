# π₀反馈输入与真实采样曝光量复核（2026-11-17）

## 范围

这是19:31完成的**CPU-only局部核查**，接续新relative19999的失败定位。没有新训练、完整模型加载、仿真动作、归一化修改或新任务成功；GPU0/1各自已有的官方生成未被中断。

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-state-stem-FP32-audit-20261117-193101/evidence
runs/latest-pi0-relative-state-stem-audit.txt
```

报告SHA256：`e0e31c460209fdd0f52437a10b86e1e85abcf16d5dc877ddfe634e8b1cabb375`。

## 1. 关节扰动没有在输入归一化/局部state stem处被完全吞掉

直接从新final19999 **部分恢复state_proj的33,792个FP32参数**，而非重载3.3B模型。输入走该checkpoint真实saved norm及原profile transform，再按实际JAX模型输入转换为FP32；仅用CPU镜像计算state_proj及随后BF16 token cast。

对source0/636，保持真实RGB与其余state不变，将左臂六个q同时±0.02rad：

| frame | q扰动 | 1024维CPU镜像BF16 token中变化元素数 |
|---|---:|---:|
| 0 | -0.02rad | 978 |
| 0 | +0.02rad | 978 |
| 636 | -0.02rad | 956 |
| 636 | +0.02rad | 951 |

- normalized state恰好只有7:13六个坐标变化；其他state、padded11坐标不变。
- normalized perturb绝对值约0.0182–0.0954，并不是全为零；token L2差约0.677–0.684。
- **支持排除的窄范围解释：**在该CPU镜像中，先前反馈probe的q干预不是因为输入轴丢失或进入state_proj/BF16后完全相同，才导致输出residual近乎不变。
- **不能推断：**真实GPU全模型attention有效使用了这些差异、所有状态都可辨识、或某个具体loss/数据策略是唯一失败原因。这不是完整模型激活抓取；刻意不一致的q/RGB组合仍只用于离线检查。

19:29首次CPU镜像直接沿用了NumPy norm计算的float64中间量。发现实际JAX批处理会转为FP32后，显式补上同样的转换，另开新目录重跑；旧报告与源快照保留，最终表只引用上面的FP32版本。两次变化元素数一致，也不把这个诊断实现细节称为原训练bug。

## 2. 20k optimizer updates不能等价为完整轨迹充分训练

从实际保存plan重建seed42的`full_schedule(1714,636,20000,42)`，并与训练报告和plan的schedule SHA逐一一致：

`099dea055134b12e28959d2f2941b3f5ec41c91d7dc99484711c38360546d15f`

实际采样计数：

- 全1714帧均被覆盖；**普通帧每帧仅5–6次**。
- 九个转折窗口606/621/631/635/636/637/641/651/666，每帧**1117–1118次**。
- 总数恰好20000。batch1，不是batch64，也不是每个完整demo重复20000遍。

这与原先明确的50%全轨迹+50%九窗口配方一致，没有发现运行中采样器被偷换。它说明新诊断高度偏向局部抓取窗口，**不能从窗口拟合好/总step数大，推断普通导航段、末尾段或误差恢复已学好**。与之前source1713误差较大相容，但尚未做同初始化/同预算的采样策略因果A/B，不能把“偏采样”直接宣称为唯一根因或认为改均匀采样一定成功。

下一受控实验应同时报告：原始source精度、偏离后的校正响应、整轨迹覆盖、真实闭环及训练条件；不再只看转折loss或是否预测close。若测试proprio扰动训练，只能先作为单独标明的离线信号通路实验；不得将人造q/source RGB组合注入真实在线策略，或未经物理一致观测验证就当部署修复。原预训练π₀+R1适配器的完整闭环对照仍缺，不能断言它已成功、或微调必然摧毁了能力。

## 3. 有界raw5完成后的自动CPU登记/跨池QA

19:30启动了**有限CPU-only finalizer**，而非新GPU控制器或新app automation：

```text
/mnt/yuanyue/double-arm/runs/raw5-finalize-20261117-193008/campaign
controller205482 / watchdog205481
runs/latest-raw-pool-finalize.txt
combined map: raw5_finite_CPU_QA (cpu_only=true)
```

它等待现有seed20002 raw5：最终门禁通过、真实controller/child退出、finalized provenance存在。之后才：

1. 在registry锁下保留所有旧池并追加本批final provenance，不因GPU1 campaign指针变化而丢历史。
2. 用原CPU audit脚本做跨worker指纹去重、排除旧16TRAIN/1heldout重合、检查任务/assisted来源及结构、候选frozen segment统计。
3. 冻结registry/source快照与SHA；只有审计真正通过才更新`latest-official-raw-pool-audit.txt`。

不启动/停止任何GPU任务，不改目标、split、cleaning或norm。依赖失败则停止、不绕过；2h等待+15min QA及外层150min watchdog只是防挂起，不重置用户计算预算。

截至19:33 GPU1日志已有5/7 source successes，处于HDF5合并中，**尚未最终验证**；CPU finalizer仍等待。GPU0原seed10004保持运行。不能提前把日志5条计为新全局合格数据；也不能把raw来源成功称为π₀抓取成功。

## 测试与保护

- 本地169项：126通过、43依赖skip；新增曝光计数/覆盖拒绝2项、raw5完成门禁/失败拒绝2项在服务器CPU也全部通过。
- 部分恢复/真实norm/state-stem报告是服务器实际执行，非mock通过的替代品。
- 四个checkpoint耦合runtime源码、旧16/1数据、所有旧权重、物理阈值和动作bounds未改。未重复交付旧视频、未产生新learned-policy任务结果。

## 19:37完成补记（非新策略成功）

- seed20002 raw5于19:34:08通过final gate：generator exit0、7 attempts/5 source successes/2 failures，5条/8,544 raw frames；controller、child及模拟器均已退出。
- finite CPU finalizer于19:35:00真实完成跨池QA。registry保留原3个pool并追加raw5的final provenance；总计**155 eligible unique /270,250 raw frames**，snapshot内重复0、与旧16/1完全指纹重合0。
- `obs/joint_qpos26D` endpoint predicate给出14个候选清洗帧、覆盖gripper转换0；author精确joint subset仍未核实，**没有清洗、split或norm修改**。这是原始assisted来源、字节轨迹去重，不是155个新场景、物理抓取或学得策略成功。
- 最新QA：`/mnt/yuanyue/double-arm/runs/raw5-finalize-20261117-193008/campaign/audit`；report SHA256 `328fa53b0b2f0d5d5bbdbf019ef779e1c4576cb6162d20bb64ed4ecac6aab6ae`，权威audit指针已更新。19:36:59实查GPU1无compute进程，raw5和CPU finalizer均退出；旧generation campaign指针现在是**已完成历史**，不能把它误读成仍占卡。
- GPU0 seed10004继续原官方生成；没启动替代策略或另一轮盲目训练。下一GPU诊断应先把“同RGB假q”的局限变成**受控初始化扰动后重新获取的物理一致RGB+proprio**，同时记录真实扰动、相机同步/姿态变化、初始化写入；这种初始化实验也不能冒称完整任务成功。预训练R1闭环对照仍待明确profile/normalization/视频身份后执行。
