# Uniform B物理模式失败：抓起后失去接触并掉落，未完成运输（2026-11-18）

## 结论

**B的TRAIN/physical评估实际于00:58:35结束并失败**，3000步、没有inside-goal成功，heldout未运行。与旧native“完全没碰到杯子”的失败不同，本次实际接触并抬升了杯子，但约37.87秒起失去手指接触，杯子下落，后续未再接触。**失持时夹爪仍接收关闭命令，不是策略主动张开释放。**

这定位了当前B的一个具体瓶颈：**搬运阶段保持夹持失败，且没有有效重获物体**。尚未证明唯一底层原因是摩擦、夹持力、姿态、轨迹加速度、碰撞或source动作本身对assistance的依赖；没有力/扭矩/接触法向等测量，不能选其中一个当定论。

## 1. 已完成结果与范围

```text
/mnt/yuanyue/double-arm/runs/pi0-relative-uniform-B-20261117-215312/campaign
  train-physical-diagnostic/status.json
  train-physical-diagnostic/physical-rollout/result.json
  train-physical-diagnostic/physical-rollout/demo_0-trace.npz
  train-physical-failure-forensics.json
  retention-forensics.json
```

同B final19999，全参数SHA `bd591dd187acff973edb632a3d65e3c7d90530fc4cb3ae604b60751644707bfa`。原始source0起点，load1、实时3RGB/原21D proprio/instruction，teacher reads0/plannerfalse/物体pose writes0，无逐步恢复。mode=`physical`，已记录控制步检查`assisted_constraint_seen=false`。

**记录边界：**本次原evaluator没有额外保存每一步USD joint/freeze/in-hand详细证据，新CPU no-assistance helper此前没有接入该in-flight run，不能事后伪造。使用的是原物理转换及记录门禁；friction4.0未实机标定，成功/失败不是实机可靠性结论。

| 指标 | B TRAIN/physical |
|---|---:|
| 真实控制步 /模型调用 | 3000 /300 |
| episode abort | false |
| task success /stable frames | false /0 |
| 底盘路径 | 3.69096m |
| backend cup grasp曾True | true |
| 最大杯子抬升 | 0.37594m |
| grasp report且lift>3cm帧 | 414 |
| 该条件连续区间支持的底盘路径 | 0.07761m |
| active assisted constraint已记录出现 | false |
| 物体显式pose writes | 0 |
| known /unknown left contact样本 | 3000 /0 |
| ≥1 /≥2 finger-link接触帧 | 449 /448 |
| native base raw proposal截断 | 4/3000 |

接触计数为几何link检测，backend grasp报告不是校准夹力，EEF原点距离不是指尖净空。不能把瞬时抬升/走路距离当任务成功。

## 2. 失败时间线：不是开夹导致的主动放下

来自实际post-step trace和runtime binary threshold0.5：

- step682 /22.733s：首个有效close；原始raw<0首步683，两者不混淆。
- step690–1130：backend杯子grasp连续true，1133又短暂true一次。
- 手指接触主要为687–1129、1131–1135；step1130瞬时掉线不单独当永久失持。
- **step1136 /37.867s开始，余下全部1864步均没有检测到左手指接触，backend grasp也没有再true。**
- 失持前step1135最大抬升约0.37594m；step1136杯子世界z约1.44987m，0.5秒后step1151降到约0.12629m，最终停在z约0.03027m。
- step1136的raw jaw command仍-1，之后30步仍全部close；初段关闭区间一直持续到step1712。它**不是当时主动张开夹爪放杯**。
- 后段出现若干开/关命令变化，但没有任何新杯子手指接触或grasp报告；不能把它们直接称为有目的的多次重新抓取尝试。

`retention-forensics.json` SHA256：`839726524e044a6da28efc0d077aae901588742f16737e501632c3f49bc9ed59`；源码`analyze_pi0_grasp_retention.py`只读取已完成两种mode trace，无模型/仿真/source HDF5访问。10步持续失联只是事后事件定位阈值，不改变原任务验收。

### 证据分级

**已确认：**本B能到达并接触杯子，在本物理配置与原记录检查下短时抬升，随后夹爪仍命令close但物体脱离/坠落，任务失败；跟辅助模式成功把物体带到inside区域形成差别。不是相机看楼梯、原生base98%截断、或始终完全没接触的那类旧故障。

**待验证：**指尖接触几何不稳定、摩擦/力设置、动作轨迹/姿态/加速度、接触解算或由辅助source带来的保持动作不足。命令接近专家不保证无辅助保持成立；反过来也不能用source assisted来源就断言模型没有学到任何物理行为。两种mode是各一次闭环，非强制逐步同动作/同噪声轨迹，不构成严格单变量因果证明。

## 3. 这轮B自己的100秒FAILED视频

```text
/home/yuanyue/double-arm/outputs/demo-review/pi0-uniform-B-TRAIN-PHYSICAL-FAILED-20261118/demo.mp4
```

- 完整100秒、TRAIN/PHYSICAL/FAILED/NOT HELDOUT SUCCESS字幕；不是旧A失败片、不是上一条assisted成功片。
- 视频SHA256 `985e2a37332f1f1854bd6e9f42c64595d019ac31b9089022909d2bba0acfa902`。
- review JSON SHA256 `38e678208bd40c97ed7926cf2234854bfd7fb720068c2c69205ab2b26ec2ca8f`。
- trace SHA256 `170050b3e0d0b6fcc59f01c0fc1eb490c66b9117346cc49dade996d5dcd09f3c`（只在服务器）。
- 服务器full ffmpeg decode通过；本地媒体/JSON/接触图SHA匹配，概览及23/30/37.6/38/39/60秒事件图已目视检查。掉落阶段物体不总在两个review视角内，精确失持时点依赖实际trace，不假称视频帧能看清所有指尖接触细节。没有声称本地ffmpeg解码。

## 4. 已继续推进：原source0动作的PHYSICAL可执行性对照（不是π₀）

B全流程controller/CPU服务/模拟器均实际退出，卡和UUID锁释放后，01:27启动：

```text
/mnt/yuanyue/double-arm/runs/source0-physical-replay-20261118-012724/campaign
runs/latest-source0-physical-replay.txt
controller219834 / sim219837 / watchdog219830
```

**问题：**原assisted source记录的1714条动作，如果不允许辅助附着，是否本身也会失持？如果source也掉，不能单纯把B掉杯解释成“学得不够准”；如果source可保持，则更应该量化B轨迹、抓取姿态与保持误差。任一结果仍不是唯一原因证明。

### 严格对照范围

- 明确`SOURCE_ACTION_REPLAY_PHYSICAL_NOT_LEARNED_POLICY`，模型调用0、无新训练/权重/归一化；专家动作回放不冒充π₀。
- 从**原source0空手初态**只加载一次，按现有`activate_physical_after_reset`转换；不转换持杯中间状态、不增加预定位、不加入holding/velocity额外初始化干预、不逐步恢复。
- 原20次physics刷新、3次render、RGB/body对齐门禁、native bounds、root`.98`检查保持。Source命令逐条原样执行，不重新规划/改标签/调friction、gain或gripper输出；源动作超界则拒绝，不静默修正。
- 新只读helper本次才实际接入：source0初始、每次20refresh后、每动作pre/post检查两手Python constraint/in-hand/freeze及**真实USD stage ag_constraint prim**。缺失/未知保留errors/-1并停止，不当成空；不删除约束来强过。
- 采样是在这些边界，不是每个内部physics substep；不能扩大到未观测时间。此helper只确认未见该辅助机制，不把absence确认等同物理抓取成功。
- 所有source replay媒体固定标`SOURCE ... NOT LEARNED PI0`，即使inside goal true也不是新策略成功。保持原inside谓词/是否松手的解释边界。
- 截至01:28仍Isaac冷启动，0命令，实际无辅助验证/完整1714动作回放均尚未通过。父流程40min simulator、50min outer watchdog仅防挂起，不重置用户预算。

GPU0 seed10005原1000目标生成不变。GPU1/map指针已移交新source replay，B uniform pointer保留终态失败与完整历史；不要把旧B指针当卡空闲或重启B。无新formal training/terms acceptance/app automation。

## 测试

新增B全链路释放门禁2项、持续失持时间线2项在服务器CPU通过；全本地186项142通过/44依赖skip。Mock/source gate单元测试不替代实际source0物理初始化/运行结果。用户survey、x目录及参考PDF未改未提交。

## 02:26后续审计状态纠正

首次source replay没有执行动作：原始result在initial no-assistance检查因未支持pinned freeze=None而abort，0/1714命令；旧parent仅按exit0误标completed。不能拿它推断source物理是否可行。保留原始证据，已按确切源码sentinel语义、raw值保留及严格完成内容验证修复只读harness，另开显式新attempt；见`docs/source0-physical-replay-audit-repair-20261118.md`。B原物理掉杯结果不受这项新诊断工具问题影响。
