# Coverage v3最终权重：训练来源条件也失败（2026-11-17）

## 结果与身份

- 原native30k权重热启动、fresh optimizer、额外90k正式更新的coverage v3已经训练完成。这份评估来自新final89999，**不是旧30k视频，不是V2，也不是专家回放**。
- 新checkpoint：`/mnt/yuanyue/double-arm/runs/pi0-checkpoints/pi0_r1_tidy_table_lora32/pi0-native-coverage-v3-20261116-172526/89999`。
- 全推理树fingerprint：`199d3068d22b5ed6959fe76aafe7065626f4b1277b4d33fd2da748f50c66f5ce`。
- `coverage-training-author-diagnostic/`，TRAIN / demo_0 / author-assisted / sync20，real三图 + 原21D proprio + instruction，CPU推理、GPU1仿真，单次加载、无teacher动作输入 / planner / 逐步状态恢复。
- 2026-11-17 **00:38 CST**结束：3000控制步、300次模型调用，未abort。底盘6.3002m，杯子特定抓取 / 抬升 / 运输均0，稳定成功0；实际辅助约束0，显式任务物体pose writes0。
- 训练来源条件门禁失败，控制器诚实停在`coverage_trained_training_condition_still_failed`，**没有启动heldout physical评估**。这不是已测heldout失败率，也不是纯物理测试。

## 已交付的失败视频

[完整100秒FAILED视频](/home/yuanyue/double-arm/outputs/demo-review/coverage-v3-train-assisted-FAILED-20261117/demo.mp4)

[闭合事件关键帧](/home/yuanyue/double-arm/outputs/demo-review/coverage-v3-train-assisted-FAILED-20261117/event-sheet.jpg)

视频是仿真时间，省略CPU推理时的等待，不代表实时速度。仅取回约8.6MiB已压缩MP4、JSON与小型接触图，未下载权重 / 原始trace / RGB数据集。

原自动review只有TRAIN/ASSISTED与NOT PHYSICAL SUCCESS范围标签，scope覆盖了原FAILED结果文字。已修复**仅导出标签**的优先级，并另存`review-explicit-failed/`；原review和原始rollout均保留。现在画面明确写`FAILED - PI0 TRAIN-SPLIT ASSISTED DIAGNOSTIC - NOT PHYSICAL SUCCESS`。

- marked review SHA256：`e0ed4bc5761d3b874efdeccc2b978b4dd2fca6acb5fd01a8b84fd02f524d056a`，本地实际文件与服务器review JSON相符。
- H264、960×508、15fps、1500帧 / 100秒；服务器完整ffmpeg解码exit0。
- 人工查看均匀六帧和58 / 60.97 / 61.97 / 62.70 / 70 / 98秒事件帧，不宣称逐帧审查全部物理接触。末段杯子仍留桌面，机器人移动 / 伸臂但未持杯。

## 证据分级原因分析

### 已确认

1. **同步不是这次失败的充分解释**：真实独立部署首图MAE（头 / 左腕 / 右腕）5.274 / 6.507 / 5.704，处于正确台面视角；不是用源RGB替换在线相机。
2. **不是简单没发close或夹爪不响应**：首次负值在step1829 / 60.97秒，单步短脉冲；step1855–2999连续负值（61.83秒至结束），总计1146步，不是1146次尝试。闭合后约1秒关节位置之和从约0.100降到0.0165，末端约2.7e-6。该变量不是校准过的指尖间距。
3. **这次确实产出了新的只读EEF / 接触遥测**：左侧3000个post-step采样均已知，没有NaN/-1未知计数；检测到的左finger-link与杯接触帧0。左EEF原点到杯质心的最小距离约0.529m。这支持“本回合没有形成有效抓取接近 / 接触”，**不是精确指尖净间隙、接触力，或对每个physics substep都无瞬时接触的断言**；不把底盘偏离专家轨迹当成不可达证明。
4. **不是V2的98%底盘限幅**：原始网络base输出实际限幅59/3000步（1.97%）。`clipped_action_steps=1863`包含其他动作维度，不能叫1863步底盘限幅；旧native `clipped_native_base_steps=0`字段仍不是原始输出完全未clip的证据。
5. 重规划边界左臂最大joint-target跳变P95约0.0608rad，source全步P95约0.00994rad；一步关节跟踪RMSE约0.00258rad。相比旧30k的对应来源条件跳变有所减小，但不是逐速度 / 逐状态配对因果对照，**降低这项指标没有在本回合带来抓杯成功**。
6. 来源条件仍失败，因此不能只用heldout泛化或physical-only grasp domain gap解释；额外90k覆盖量也没有在这条训练回合上产生任务成功。

### 仍待验证

策略的接近精度、动作衔接 / 控制执行语义、监督分布与闭环分布偏移、数据多样性和作者训练设置差异仍是后续检查方向。不能据此独断“唯一问题是训练不够 / 数据不够”，也不能把摩擦或夹持力当成已经证实的原因。16 demos / batch1 / mean-std试验仍不是1000 demos / batch64 / quantile的论文规模复现。

离线分析来自服务器`coverage-v3-failure-forensics.json`，源动作仅用来事后评分，model/simulator控制调用0；对应trace SHA256为`5ee8536158899a9f855d63b2506b33cc0e7d489800b72221df45f3c6507fcf6c`。

## 后续计算，没有盲目重训

- GPU0继续官方原始示范池生成；01:06日志10次来源演示成功 / 12次尝试，但首个25条shard未finalize，verified unique count仍0，不冒充已验收新数据。
- GPU1全coverage链路已退出且卡空闲，释放门禁实际通过，才启动独立真实π₀batch1×64内存 / optimizer计数预检，不保存checkpoint、不把1次effective update称为新正式训练。
- 首次预检`pi0-accumulation-gpu-preflight-20261117-010842`在模型初始化前因假定上游有Git元数据失败；服务器是已有的固定revision归档安装，`UPSTREAM_REVISION`正是现有安装器使用的核验字段。已保留失败JSON / 日志，修正为严格检查该marker（若有.git也核对HEAD），无部署 / 版本 / 模型配置变更。
- 第二次归档pin修复后的尝试`pi0-accumulation-gpu-preflight-archive-pin-20261117-011150`在`jax.eval_shape`阶段因上游`TrainState.tx`要求`optax.GradientTransformation`，不接受duck-typed `MultiSteps`对象而失败；0实际microsteps，错误完整保留。只把相同init/update包装为该标准NamedTuple，不修改梯度 / averaging / clipping语义；服务器CPU三次effective update / moments / LR数值重检通过（最大差<1e-9），仍非真实π₀ GPU通过。
- 第三次独立尝试：`/mnt/yuanyue/double-arm/runs/pi0-accumulation-gpu-preflight-typed-optax-20261117-011522`，watchdog PID149207；开始时仍未验证GPU fit，以其report实际64 microsteps / 1 optimizer update及有限性结果为准，OOM也必须诚实保留。
- 本地114项测试（87通过、27skip），包括FAILED/ABORTED标签不能被scope覆盖、归档marker严格验证的新增夹具。没有安装 / 升级依赖。
- 当前GPU1指针为`runs/latest-gpu1-job.json`，合并map已更新；legacy `latest-single-gpu-job.json`保留旧coverage结果以供门禁 / 溯源。不要只看legacy pointer误以为GPU1仍在跑评估。
