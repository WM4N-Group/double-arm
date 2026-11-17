# 底盘接近前缀单独替换仍失持；转向同权重真实闭环execute5诊断（2026-11-18）

## 1. 已完成的OFFLINE hybrid结果：单独替换base接近前缀不充分

```text
/mnt/yuanyue/double-arm/runs/B-base-prefix-diagnostic-20261118-042624/campaign
runs/latest-B-base-prefix-diagnostic.txt
```

04:34:53实际完成1200/1200动作，exit0、unaborted、2421份no-assistance边界样本已知全0。**该实验没有live π₀，只是显式source-base+B其它轴的回放控制干预。** 它不是新模型成功或新的完整任务评估。

CPU检查实际trace证明：

- 前682步base0:3逐值等于source对应命令；
- 全1200步其余18轴逐值等于B记录命令；
- step682之后base逐值恢复B；没有额外修改动作、数据或bounds。
- close前root相对可行source参考的XY误差降到约**0.000616m**，yaw误差约**-0.0215°**（原B约0.0800m/-2.985°）。

但杯子仍失持：

| 指标 | 固定B原命令 | source-base prefix hybrid |
|---|---:|---:|
| 首次持续无接触步 | 1136 | **1144** |
| 仿真时刻 | 37.867s | **38.133s** |
| 最大杯子抬升 | 0.37594m | 0.36288m |
| 1200步末杯子grasp/contact | false /0 | false /0 |
| hybrid最终cup z | — | 0.02263m |

这说明**在保留B原躯干/手臂/夹爪固定命令的上下文里，只替换底盘接近前缀并不足以消除失持**。不能再把8cm参考站位差单独当已证实唯一根因。

也不能反过来说底盘误差完全无关：干预后的B arm targets仍来自旧路径状态，没有重算为新观测下的在线动作；root/arm/接触几何存在耦合。Source位姿不是唯一合法解，8步延后也不是有效恢复或成功。

证据`base-prefix-forensics.json` SHA256：`ab02799cece6f49ee3c3d115bd390669eb650803527a0f2651c87cc56ce1f837`。原trace SHA256 `657335295bb1d30406dd1e869d20dda653a2864efd2e78ef274d06afb94c345e`。未训练新模型，没有把hybrid tape当π₀修复部署。

## 2. 现在启动的是真实π₀闭环，不再混入source命令

前一hybrid全流程controller/sim均退出、GPU1实际无compute进程并取得UUID锁后，于**05:28**启动：

```text
/mnt/yuanyue/double-arm/runs/B-chunk5-physical-20261118-052838/campaign
runs/latest-B-chunk5-physical.txt
controller225808 / evaluation225811 / CPU service225818（以实际指针为准）
```

**同B final19999 checkpoint、原norm/profile/21D state/真实3RGB/instruction、不新训练。** 唯一预期控制设置变化是每次执行预测chunk前**5步而非10步**，然后重新用真实观测询问模型。horizon50、denoising10、Gaussian、source0/init1、sync20、原native limits、physical/friction4.0和root`.98`都保持。

- `optimizer_updates_this_job=0`，不是再训20k/90k，不加载source635预定位。
- **没有teacher base或arm命令、planner、规则/最近邻策略、fake RGB/proprio、逐步状态恢复。** 之前的hybrid不参与新控制。
- CPU推理，GPU1只给模拟器；仿真等待推理，不宣称实时。最多3000步/约600次模型调用，比原execute10更耗时，日志ETA不当交付承诺。
- 这检验更多反馈是否改变持杯轨迹/恢复情况；改变执行chunk也会改变请求数量、噪声消耗和观测历史，故不是对一个独立数学参数的无随机性因果证明。只有一个TRAIN episode，不用heldout调参。

## 3. 新只读无辅助证据明确opt-in，默认旧路径不变

这次在所有前序实验已终止之后才修改评估入口，添加显式`--audit-no-assistance`：

- `run_pi0_evaluation.py`仅显式flag时将该选项传到模拟器；configured模式+flag直接拒绝，不偷换assisted为physical。
- `evaluate_pi0_r1.py`在physical转换后、20次reset physics刷新后、每个控制步pre/post读取两手Python AG constraint/in-hand/freeze及真实USD stage的`ag_constraint`。
- 返回8轴0/1/-1数组、计数、初始raw None来源证据；缺失/异常仍unknown/errors并中止。源码exact SHA验证None inactive语义，不写False、删约束或修改controller。
- 这些字段**不是模型输入**，不改变动作、force/friction、normalization或BDDL判定。新门禁是这次诊断的证据完整性检查，不是篡改任务成功标准。
- 默认flag=false不调用该观察器；原evaluator/observer源码先复制到新run父目录`baseline-evaluation-sources/`，旧B result和源码hash保留不追改。四个checkpoint耦合runtime源文件和existing service未改。
- 控制步post检查放在真实transition遥测保存后；若发现问题而abort，仍保留已完成转移，不把未知sample伪造为0。pre检查失败不算已执行动作。
- 原来只准备过的helper直到source物理回放才实际验证；本次是它**首次进入live π₀ evaluator的显式opt-in**。不能把新观测倒填给先前B run。

新的parent不只看exit code：要求actual单episode、steps>0、非abort、physical/TRAIN/execute5/teacher0/init1、无error、无辅助样本全部已知通过且计数=`21+2*steps`。否则标setup/rollout abort，不冒充完整task failure/pass。

## 4. 输出与验收范围

- 只测**TRAIN/physical**；即使通过也是TRAIN结果，需人工看视频和释放，不自动启动heldout，不宣称论文复现。
- 上游inside-goal没有release条件的限制仍存在。若最终held=true/未松手，不能说已放下；要在结果中另行检查，原task条件不偷偷改。
- 无论成功/失败均导出**本execute5 episode的新review**，检查实际B fingerprint/trace/video SHA及动作/接触；不可用旧execute10失败视频替换。
- 最大3h评估、parent等待11100s、外层4h只防挂起。GPU1锁覆盖加载、仿真、导出全过程；CPU加载空隙也不能插入其它GPU任务。
- 05:29实际是`loading_pi0_for_visual_inference`，模拟器/策略动作尚未开始，**还没有新闭环结论**。新增no-assistance真实loop集成也需此次运行验证，不能以fixture代替。

GPU0原seed10006生成保持，worker150/261496与最后global QA180/314090分开。旧B/A权重、原16/1数据、所有split/norm/assets/PDF均未变。

## 测试

本地198项150通过/48依赖skip；本次新routing2项、actual completion2项，以及observer7项、原profile路由3项共14项在服务器CPU通过。没有安装/升级环境。真实live集成结果仍等待。

## 07:26最终结果更新

Execute5 live episode于06:50完成3000步/600calls，未形成cup grasp、max lift约4mm，task失败。实际6021份新增无辅助样本全已知0，说明只读审计live集成本轮确已运行；不是策略修复通过。匹配失败视频与诊断边界见`docs/B-chunk5-failed-result-20261118.md`。整个流程释放GPU1后，该卡已转独立官方raw100候选收集；不要从本页旧进行中段落推断还在跑policy或卡空闲。
