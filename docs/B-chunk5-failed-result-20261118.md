# 同B权重execute5物理闭环失败：更频繁重规划没有形成稳定抓取（2026-11-18）

## 1. 实际结果

新episode于**06:50:34 CST**完成3000步/600次真实π₀推理，正常退出但task false、stable goal帧0；没有杯子grasp报告，未形成稳定运输，heldout未运行。

```text
/mnt/yuanyue/double-arm/runs/B-chunk5-physical-20261118-052838/campaign
runs/latest-B-chunk5-physical.txt
```

这是LIVE original π₀、同uniform B final19999 /full parameter SHA `bd591dd187acff973edb632a3d65e3c7d90530fc4cb3ae604b60751644707bfa`，不是source replay、hybrid或新训练权重。真实head+双wrist RGB、原21D proprio、instruction；source0/init1/sync20，teacher0/plannerfalse/posewrites0，physical/friction4.0不变。

| 指标 | 原execute10 TRAIN/physical | 新execute5 TRAIN/physical |
|---|---:|---:|
| 真实控制步/模型调用 | 3000 /300 | 3000 /600 |
| task success /stable frames | false /0 | false /0 |
| cup grasp report曾True | true | **false** |
| max cup lift | 0.37594m，随后掉落 | **0.003998m** |
| ≥1 /≥2 finger-link接触帧 | 449 /448 | **41 /1** |
| 底盘路径 | 3.69096m | 4.64332m |
| 首个effective close | 22.733s | **25.700s** |
| left action target重规划边界jump P95 | 0.02154rad | 0.01063rad |

更短chunk在这一次测试里动作边界更平滑，但**没有带来任务改善**，不能把“更频繁反馈/更平滑”当抓取能力充分条件。不同query数量也改变了噪声消耗、观测历史与到达轨迹，仅各一个TRAIN episode，不是对所有seed的统计优劣证明。

## 2. 接触与关夹：事实和未知分开

- 初始source RGB MAE约5.23/6.55/5.61，非旧reset相机看楼梯问题。
- 真实motor metadata支持binary threshold0.5；首次effective close在step771，第一段只有771–774，后续早期有多个短区间。**15个effective close连续区间不是15次有目的的抓取尝试。**
- 41个left finger-contact样本分布在649–774附近，只有1帧≥2finger-link接触，之后无新接触；所有3000样本已知，未知0。
- 夹爪width全程最小约1.76e-7m、最大0.1m，说明关节能够闭合和重新张开，但这不证明曾形成稳定夹持或有足够力。
- 2485步any-action clipping主要不能直接读成base clipping；原native base raw proposal只1步发生变化，不能套用旧V2的98%底盘限幅解释。
- 左臂命令与下一步实际q的RMSE约0.000132rad，仅是joint tracking描述、非EEF精度/抓力；即使此值很小也没有任务成功。

### 不编造“未来关夹被不断截断”的结论

本trace只存执行过的proposal/target，没有完整每次50步预测chunk。确有短close pulses与无稳定抓取现象，但**不能从这些历史数组证明未来close总在chunk第6步以后、或重规划反复把它推迟**。这只能列为待验证假设；后续若调查，需要单独只读记录实际query/full output chunk，不能用source RGB伪造在线输入或把假设写成已确认因果。

## 3. 无辅助证据在这次live evaluator中真实产出

`--audit-no-assistance`实际传入模拟器，不是仅plan字段：初始+20reset refresh+每步pre/post共**6021×8**样本，trace全部已知0，result all-confirmed true、errors无。

包含双手Python AG constraint/in-hand/freeze与真实USD `ag_constraint` prim；None原值及pinned source SHA来源保留，未改robot字段。采样边界不覆盖每个内部physics substep。没有手指法向/力/力矩标定，仍不作实机可靠性判断。

这是新增只读loop集成的实际运行证据；不能倒填给更早execute10 B或旧A。Observer用于证据完整性、不是VLA输入或控制动作；默认关闭不改历史路径。

## 4. 本次100秒FAILED review

```text
/home/yuanyue/double-arm/outputs/demo-review/pi0-B-execute5-TRAIN-PHYSICAL-FAILED-20261118/demo.mp4
```

- 960×508/H264/15fps/1500帧，7,830,049字节；明确TRAIN/PHYSICAL/FAILED/NOT HELDOUT SUCCESS。
- 视频SHA256 `74aeb4f26afc50b31a3395610d1280ea417fe4c61b3b9bf321de323962c07f2c`
- review SHA256 `3128cf41fa623f241528ec0d6328d32052f19febb77e2ea57e0556f7f547ccf0`
- trace SHA256 `b827f3d201083b13f87e8c2f7aeaaa957fc2e15f03f822494d9fb80e5c6df855`（仅服务器）
- forensics SHA256 `3b1a95813767c91742e7a88500fd3ebef047ca0883de0feccb6642bdefb38a42`

服务器完整decode通过；本地MP4/JSON/接触图SHA一致，2/20/40/60/80/98秒关键帧已目视检查。没有声称本地ffmpeg解码。旧execute10失败视频与本次不同，绝不互相代替。

## 5. 当前训练策略的审计结论，不再连续猜chunk

已确认的几条证据应同时保留：

1. 同一原生控制接口中，source expert物理回放可保持、运输并释放，所以这条参考轨迹并非物理上必不可能；它不是模型成功。
2. B的已生成execute10命令不依赖在线推理也能逐数组重现掉杯，故不能只归咎CPU/RPC等待。
3. source-base prefix替换虽纠正站位仍失持，故站位差本身不是已证明唯一原因；hybrid不是live policy。
4. execute5没有改善这个episode，变成未形成稳定grasp，故不继续把提高频率当万能修复。
5. 现有模型只是一条demo/batch1/meanstd/20k的诊断。A/B已经揭示采样覆盖的重要性，但这不等于达到了论文数据量、有效batch或规范化。不能继续给这一小配方盲加步数，再称论文训练。

剩余是policy动作生成、全流程覆盖、接触几何、时序与保持鲁棒性的耦合问题，尚无唯一超参/代码错误被证实。数据增多也**不是已证明一定修复**，但既定的官方1000条与论文对齐工作应继续，而不是在一条demo上不断挑推理参数。

因此本次完整流程退出后，GPU1转回独立官方**raw assisted示范收集100条候选**，GPU0原1000目标继续；没有新formal training、norm/label修改或heldout tuning。新数据必须经过跨worker QA、版本化cleaning/split/norm及batch64实际训练预检，再启动新正式recipe；action/base编码、作者execution/freeze等未核实项仍明确保留。见`docs/two-gpu-raw-pool-scaling-20261118.md`。
