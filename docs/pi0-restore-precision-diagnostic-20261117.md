# 同权重参数restore精度对照（2026-11-17）

## 结论

**将checkpoint参数restore从BF16改为保留FP32，没有消除训练源上的抓取转折错误。不能把“服务时参数全部转BF16”作为目前失败的充分解释，也没有据此修改生产服务。** 这项实验只改变restore目标参数dtype，原模型配置的compute dtype仍为bfloat16，不是full-FP32计算测试。

## 原因与方法

固定官方`policy_config.py`的JAX推理工厂对checkpoint所有参数使用BF16 restore。旧训练的trainable分区保存为FP32，而冻结分区已转BF16，所以有必要先验证这一步是否丢失R1小幅绝对关节修正所需的精度。

在GPU1完成释放门禁、UUID锁和实际空闲核对后，**同一进程**依次加载同一个coverage v3 final89999：

1. `official_bfloat16_restore`：全部参数恢复为BF16，镜像固定官方推理工厂的其它输入/输出变换。
2. `preserve_float32_params`：全部参数恢复为FP32；本来已存成BF16的参数只能上转换，不能恢复丢失的信息。模型compute dtype、rank、归一化、三路图像、原21D proprio、prompt、horizon50、10步去噪等均不变。

16条原TRAIN各取首次close和15帧后，共32真实source观测，3个固定Gaussian noise。每条件96次推理，总192次；两条件间逐项核验96组input/noise/frame hashes相同。没有heldout、训练梯度、模拟器动作、保存checkpoint或实际闭环任务。

两条件参数dtype计数实际均3,315,905,296个元素，分别为bfloat16或float32。依次释放模型和编译缓存，不同时在卡上常驻两个完整模型。GPU0的原始官方生成不变。

## 结果：实际native action bounds clip之后

| 指标 | 官方BF16参数restore | 保留FP32参数restore |
|---|---:|---:|
|首次close处首动作仍open|44/48|44/48|
|三个noise均首步不一致的回合|13/16|13/16|
|首次close处前10步预测close数|103/480|105/480|
|首次close处左臂首目标误差RMS中位数(rad)|0.027021|0.026784|
|同位置专家target-current-q RMS中位数(rad)|0.000370|0.000370|
|+15帧首动作仍open|0/48|0/48|
|+15帧前10步close数|480/480|480/480|
|+15帧左臂首目标误差RMS中位数(rad)|0.030439|0.028258|
|前10步左臂目标误差比hold更差的采样|96/96|96/96|

部分关节误差有小幅改善，但关键的首次close错误模式仍完全存在。没有宣称FP32对所有precision问题无影响；没有测试更改全部计算精度或做闭环任务，不能越界推断。

## 对后续视觉freeze A/B的额外约束

固定官方`train.py:init_train_state`按freeze_filter把冻结参数的**存储dtype也改成BF16**。因此普通“只额外freeze img”会同时改变梯度分区和图像权重存储dtype。

新的单demo拟合诊断显式将storage cast与gradient filter解耦：两分支都使用旧原始freeze规则做storage cast，只有B额外排除image梯度。这样两分支开始时图像参数都仍FP32。不能把冻视觉同时存BF16的复合对照误叫严格freeze-only。

服务器CPU真实NNX小模型fixture验证同seed的完整初值/dtype一致，梯度分区不同；真实pi0 CPU shape确认495,908,624 vs81,104,928可训练参数。它们是预检，不是两分支GPU训练或任务成功。另见`docs/pi0-single-demo-controlled-fit-design-20261117.md`。

## 溯源与验证

```text
/mnt/yuanyue/double-arm/runs/pi0-restore-precision-20261117-132153/
  probe/report.json
  probe/summary.json
  probe/status.json
  probe-source.py
  metrics-source.py
  launch.sh
  exit-status.txt
runs/latest-pi0-restore-precision-probe.txt
```

- probe实际完成2026-11-17 13:24:24 CST，exit0，192次推理。
- report SHA256：`e7e73469157e16047dc28bee1b8fd20bdd43314f680f21ccbcd7437e3227966d`。
- 旧三份实际training adapter源码SHA仍一致；没有修改旧checkpoint、norm、dataset或服务。
- `scripts/probe_pi0_restore_precision.py`为新独立诊断；`scripts/summarize_pi0_grasp_onset.py --restore-precision`只聚合实际存在的clipped字段，不伪造raw结果。匹配和缺失字段夹具在服务器CPU通过。
- 首次close阈值<.5仍是固定binary controller源码约定，不是本次新模拟器controller实例数据。
