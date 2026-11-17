# 单demo抓取过渡拟合与视觉冻结A/B：执行前设计（2026-11-17）

状态：**已通过CPU预检，独立控制器已启动；GPU更新/完成数以实际worker报告为准**。不覆盖旧native30k、coverage v3或1000-demo原始生成，不是论文正式训练。

## 为什么先插入参数restore精度对照

新抓取定位已经发现训练源观测上的精度与转折问题，但还没有锁定某个优化/冻结参数。检查固定上游时又发现两项必须先隔离的事实：

1. 固定`src/openpi-215abfb/src/openpi/policies/policy_config.py`的JAX推理工厂调用`restore_params(..., dtype=jnp.bfloat16)`，即旧训练中保存为FP32的trainable参数在服务时也转换到BF16。这个行为来自固定官方实现，不应直接称为本项目代码bug。它是否影响R1需要的小幅绝对关节修正，必须做实际对照，不能仅凭精度名称推断。
2. 固定`src/openpi-215abfb/scripts/train.py:init_train_state`会按`config.freeze_filter`把冻结参数存储类型转换为BF16。因此，**直接换freeze_filter调用官方init，并不是严格只改变梯度冻结的A/B**：图像权重存储dtype也随之变化。不能把这种复合变化的结果全部归因于视觉冻结。

故先执行新的、独立的GPU1离线参数restore对照：同final89999、同32个TRAIN观测、同3个noise、同配置compute dtype、同normalization与后处理，只改变目标parameter restore dtype。BF16原服务条件与保留FP32参数条件在同一进程顺序运行，分别释放模型/编译缓存。**不是full-FP32算力试验，也不能恢复旧checkpoint中本来已保存为BF16参数丢失的精度。** 不修改线上服务或旧checkpoint。

这一步比重新训练更直接地隔离“训练有效但服务精度丢失”的假设。无论改善还是不改善都保留报告；改善也仍需新的同权重实际闭环验证，不能把离线分数当任务成功。

## 后续拟合诊断的冻结计划

目标是先回答“固定任务、很小的真实source过渡样本，模型在默认采样下能否可靠拟合动作细节”，而不是靠新任务奖励、planner或特权输入规避。

- 只用原TRAIN/demo_0；从真实首次effective-close附近取9个**离线训练样本**：相对frame offsets `[-30,-15,-5,-1,0,1,5,15,30]`，每个样本保留原始source后50步动作标签、原图和原proprio，不拼接、不跨episode。
- frame/phase/index只属于采样/报告元数据，不加入模型字段；图像始终head+双wrist，state原21D，固定instruction。
- 新显式diagnostic版本：original pi0、LoRA rank32、native21D、原TRAIN mean/std、batch1；官方base权重初始化、fresh AdamW，不用失败v3 warm start。
- 计划每条件最多2000实际optimizer updates，固定同seed/相同9帧循环顺序。若缩短warmup必须作为新诊断配方明示，两条件相同，不能宣称论文Table5配置。
- A维持旧官方freeze规则；B只额外冻结img视觉分区。**两条件的所有参数初值与存储dtype必须一致**：使用统一的原规则storage cast，再分别设置gradient/optimizer filter；不直接让新freeze规则改变storage cast。不编辑上游或旧三个adapter源码。
- 抽查/计数实际NNX状态，验证同seed的完整初始参数fingerprint、dtype tree一致；step0预测允许记录GPU数值差异，但不能把明显不同初始模型当freeze-only对照。
- 训练增广、损失、AdamW/clip、LR、数据和噪声约定一致。没有grasp reward、近邻、状态MLP、teacher控制或物体位姿。
- 先验证两个分区实际trainable参数计数（预期495,908,624 vs81,104,928，按真实tree为准），以及optimizer/LR更新次数。小模型CPU夹具验证storage/filter解耦；真实GPU初始化/梯度/拟合另行报告，不能用夹具冒充。
- 评价首先是同9个训练观测上的flow loss、固定Gaussian action samples、抓取前open/首次close/已关闭保持分项、左臂目标误差与hold-state离线参照。模型能拟合9样本不等于泛化、闭环或物理任务成功。
- 保存明确new-variant来源与报告。不得把小样本重采样诊断结果覆盖到正式dataset、norm、旧checkpoint contract或正在跑的官方raw generator。

## 启动/结束约束

- 先核对当前GPU1权威指针、完整前置workflow终止、PID退出、实际GPU空闲，并取得同一UUID锁；已有任务健康时不抢占、不跨卡fallback。
- 若OOM/非有限梯度/门禁失败，保留日志与失败报告，不隐式减少batch/改变freeze或切换GPU。
- 两张卡开放时间不等于允许同时加载两个24GB训练任务到同一张卡。GPU0原始生成继续，GPU1诊断串行。
- 真正闭环任务仍使用真实在线RGB和原物理/assisted标签，原任务门槛不改。需要新的闭环失败时交付匹配该变体的FAILED视频。


## 13:33实际启动记录

- 精度对照192次推理完成：保留FP32参数没有消除44/48首次close仍open的模式，详见`docs/pi0-restore-precision-diagnostic-20261117.md`，故不直接改服务。
- 第一独立拟合尝试`runs/pi0-single-demo-freeze-ab-20261117-133141`在loader阶段误用了HF默认根路径，尝试查找不存在的Hub dataset ref后失败；**0 optimizer updates、0初始采样、没有正式训练**。旧训练loader原来有正确HF_LEROBOT_HOME；这是新诊断wrapper漏配环境，不是原16-demo数据或训练失败原因。其日志/报告保留。
- 新显式localdata尝试：`/mnt/yuanyue/double-arm/runs/pi0-single-demo-freeze-ab-localdata-20261117-133301/campaign`，controller176456、A worker176459。复用既有`HF_HOME=cache/huggingface`和`HF_LEROBOT_HOME=data/pi0/lerobot`，同时打开offline flags，不下载或重部署数据。原尝试0更新且父子PID均退出、实际GPU1空闲后才启动，非中断/恢复部分训练。
- `runs/latest-pi0-single-demo-fit.txt`以及GPU1/combined pointers已移动到新尝试；原GPU0控制器仍继续seed10003。
- 控制器A/B串行，每分支最多2000真实updates、1h watchdog；总体7500s watchdog，有信号cleanup只回收自己的child process group。它们是防挂死上限，不是向用户申请的新预算窗口。
- A先从官方base运行，B随后再从相同官方base/同seed运行；B在开始更新前必须匹配A完整初始参数/dtype fingerprint、9个训练窗口batch/input hashes、训练frame顺序、norm、LR、seed。非有限、OOM或计数/匹配失败就停止，不自动改batch、freeze或fallback卡。
- 初期stage为`offline_initial_sampling`，GPU上实际A trainable count495,908,624已核对；不是2000步拟合完成，也还没有B的GPU更新结果。新source9窗口均与HDF5逐值/逐像素核对。
- 不保存checkpoint，保留初始/最终离线预测、参数fingerprints、梯度/AdamW/LR计数与loss记录。任务成功仍未验证。


## 13:39资源布局修正（旧失败保留，尚不算拟合通过）

- localdata尝试实际完成27个初始离线样本，随后**第一次训练step的XLA执行OOM**（申请5,865,592,992 bytes），0完成optimizer updates；不是初始采样失败，也不是已训练2000步。A初始完整参数hash为`a3ee6c232ac89b0e4e5437e326dc5af9234f3a4436cb4cc0b138fbe605e28e1b`，参数存储495,908,624个FP32、2,819,996,672个BF16。B未运行。父子已退出，报告保留。
- 对照旧成功native训练launcher，旧流程明确设置`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`；新诊断wrapper初版漏了该变量，使用JAX默认预算，且关闭了预分配。这是新诊断的资源布局差异，不是旧模型失败原因。不能将OOM称为任务策略失败或超参A/B结论。
- 新显式allocator90尝试：`/mnt/yuanyue/double-arm/runs/pi0-single-demo-freeze-ab-allocator90-20261117-133856/campaign`。两分支统一0.90预算和预分配，保持数据、初值、loss、model/adapter、batch1、LR与2000-update计划不变。A更新前还必须匹配上次0-update尝试的完整初始参数/dtype hash，不能在资源修正时暗换模型。
- `latest-pi0-single-demo-fit.txt`与GPU1/combined map已移至这个新尝试。13:39时仅是启动/初始化，GPU实际fit和2000步完成数仍待实际报告；不宣称资源修正已经成功。B只有A完整通过后才开始。

13:41补充：allocator90尝试已实际恢复A完整模型，初始全参数/dtype hash与上次0-update尝试逐bit一致（`a3ee6...28e1b`），trainable count495,908,624；处于初始离线采样编译，还未宣称完成真实梯度。两项真实CPU NNX fixture（storage分区和小模型完整init一致性）通过，真实pi0形状分区也通过；本地143项测试107 passed/36 dependency skips，服务器新增8项精度汇总/拟合计划/NNX测试通过。
