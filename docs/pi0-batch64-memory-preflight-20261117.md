# 真实π₀ batch64累积内存预检：device OOM与host变体

## 1. 设备驻留MultiSteps确实失败，不是已验证的batch64

02:06 CST检查发现独立预检已于 **2026-11-17 01:19:18 CST** 在第一个真实gradient microstep执行时OOM：

```text
runs/pi0-accumulation-gpu-preflight-typed-optax-20261117-011522/report.json
stage=failed
actual_microsteps=0
actual_optimizer_updates=0
RESOURCE_EXHAUSTED: Out of memory while trying to allocate5669637416 bytes
actual_pi0_gpu_memory_fit_verified=false
```

这次已成功初始化真实nativeπ₀和优化器，实际trainable参数495,908,624，GPU FP32累积缓冲1,983,634,496 bytes（约1.847GiB）。但首个microstep没有完成，不能把初始化成功算64步通过。XLA日志的`GPU_0_bfc`是`CUDA_VISIBLE_DEVICES`后的逻辑设备0，进程始终固定物理GPU1 UUID，没有落到另一个物理卡。

本结论只针对这个**设备驻留MultiSteps + 完整upstream train_step编译**变体；不是证明任何单卡或双卡batch64方案都不可能，也不是任务失败视频。原报告、日志与source快照完整保留，旧模型检查点未修改。

## 2. 新的显式隔离变体：host FP32 raw-gradient累积

为避免设备上的额外累积状态及合并执行图峰值，新增独立host变体：

1. 使用相同真实三图 / 原21D proprio / instruction、native动作与mean/std、rank32和原始trainable/freeze分区。
2. 同一旧native30k权重、fresh官方优化器；64个microbatch期间参数及optimizer/LR计数不更新。
3. 原模型的`compute_loss(..., train=True)`、mean loss、NNX DiffState过滤计算GPU原始梯度；每个microstep使用独立fold-in RNG。
4. 每次真实gradient完成后转到**服务器RAM中的float32**累积器，不逐micro裁剪；验证每个原始gradient元素有限、树结构 / dtype一致。
5. 第64次后在host除以64，再回传一次平均梯度；单独JIT执行**官方全局clip + AdamW**，仅推进1次optimizer与LR计数。
6. 记录0/1/63/64微步参数hash不变，effective update后hash改变；保存JSON，不保存checkpoint、不进行仿真。

这里同时改变**累积缓冲placement与梯度/优化器执行图拆分**，不能将结果单独归因于一项。没有冻结额外视觉层、换动作标签、改变归一化、降低有效batch或切换模型来假装内存改善。

`HostGradientAccumulator`采用host FP32顺序求和 / 最后求均值，不要求与device MultiSteps在线均值逐位相同。直接真实batch64、随机增强 / flow noise的数值等价仍需独立核验；一次内存smoke不是完整训练稳定性或论文复现。

## 3. 先CPU验证，再在空闲GPU1启动

- 本地117项：88通过、29因数值依赖skip。
- 服务器CPU新增3项夹具通过：严格FP32 / 有限性、失败add不部分修改buffer、未满64不得取mean、均值后再clip反例。
- 真实NNX **小线性模型**的3次effective AdamW update，与直接batch64的参数 / moments / LR计数一致，最大参数差约9.3e-10；64次micro期间参数不变，NNX梯度树host roundtrip通过。报告：`runs/paper-preflight-20261116-195823/host-fp32-nnx-cpu-20261117.json`。
- 这仍是CPU算术 / 树结构验证，不是实际π₀梯度或24GB内存成功。

2026-11-17 **02:11:48 CST**，旧coverage全链路已完成、上个device OOM进程已退出、GPU1实际空闲、释放门禁再次通过后，启动：

```text
campaign=/mnt/yuanyue/double-arm/runs/pi0-host-accumulation-preflight-20261117-021148
watchdogPID=152433
GPU=GPU-50e5d38d-2825-e379-f537-7f0bf4609aba
CPU affinity=88-103
watchdog=3600s
```

新entrypoint参数为`--accumulator-placement host`；默认仍是原device变体，不悄悄重定义旧失败。entrypoint、host probe、accumulator源文件与SHA存于本次campaign。`latest-gpu1-job.json`及合并map已更新；legacy single-job指针仍保留旧coverage来源。

**02:13实测处于gradient-only编译，0个已完成microstep，尚未验证fit。** 后续必须以最终report的64次真实梯度、1次effective update、有限性和实际显存统计为准。继续OOM则明确失败，不标成成功。

## 4. 并行工作与主任务边界

GPU0始终继续官方author-assisted Tidy Table/D0生成；02:12日志16次来源演示成功 / 20次尝试，首25条shard尚未finalize，verified unique仍0。未为预检中止它，也没有下载原始数据。

coverage v3 final89999已在train/assisted来源条件失败，匹配的新FAILED视频已交付。这轮没有重启它、没有篡改物理门槛、没有创建新正式训练。即便host内存smoke通过，后续仍要对齐论文数据规模/划分、cleaning关节子集、action/exec语义、quantile、冻结分区和20个固定未见评估回合，不能直接将pilot native/mean-std配方叫paper reproduction。

## 5. 02:16:33 CST：host变体完成真实64 micro / 1 optimizer更新

最终report为`stage=completed`，进程已经退出、GPU1释放；日志没有失败traceback（有XLA rematerialization与部分buffer donation未可用警告，不能把警告单独当失败或当优化完成证据）。核实：

- **64次真实π₀ forward/backward + 1次effective官方clip/AdamW更新完成**，不是CPU mock或仅初始化。
- 实际495,908,624 trainable参数；host FP32 buffer1,983,634,496 bytes（1.847GiB）。
- micro1/63/64处完整trainable hash都等于初始值；平均梯度更新后hash改变，参数有限；AdamW与LR两个count均为1。micro期间两个count均为0。
- JAX allocator `peak_bytes_in_use=17,181,966,336`，约 **16.002GiB**。这是该进程allocator统计，不是整卡 / 其他进程 / 驱动总显存峰值。
- 64个micro的记录用时之和约204.43秒，包含首次gradient编译、数据获取与传输，且不含所有初始化 / optimizer编译 / hash开销；**不是稳态训练吞吐或完整50k训练交付时间估计**。
- `checkpoint_saved=false`，没有新checkpoint或正式训练；`direct_real_pi0_batch64_numerical_equivalence_verified=false`保持false。

因此可以说：**这个host-placement / 分图的单卡micro1×64真实π₀内存与计数smoke通过**。不能说：设备驻留版本通过、真实直接batch64数值一致性已证明、50k训练稳定性已验证、论文复现完成或机器人任务成功。

## 6. 计算后续：GPU1转入25条官方辅助数据批次

不为占卡重复跑无意义smoke，也不在数据 / 语义尚未对齐时盲目启动50k正式训练。02:21:05 CST在确认GPU1实际空闲、旧coverage门禁通过后启动一个**有界25条author-assisted D0数据辅助批次**：

```text
/mnt/yuanyue/double-arm/runs/paper-official-aux-gpu1-20261117-022105
controllerPID155207
GPU1 UUID / CPU88-103
seed20000, target25, shard25, max2 shards（去重不足时的小补充上限）
```

GPU0原1000条目标工作流继续，不中止 / 不篡改其目标。辅助批次只记录自身去重后的raw provenance；**不能把两worker计数直接相加当全局unique1000**，最终需要合并去重、选定1000条并另建清洗 / split / norm版本。多出的有限原始候选不等于改变论文训练样本数。

为避免两worker互改共享生成配置，在现有shell helper仅加共享config/scene准备锁；各自复制私有config后释放锁再启动仿真，上游入口 / 参数内容未改。预检入口也加同一按GPU UUID的占用锁，防止另一任务仍在CPU冷启动、尚未出现在`nvidia-smi`时抢卡。

GPU0与GPU1各有campaign / job pointer / CPU affinity；`latest-paper-generation-campaign.txt`仍只指GPU0，GPU1使用`latest-paper-generation-gpu1-campaign.txt`，合并map已切换。`latest-official-generation.txt`只是任意worker最后一次启动的legacy指针，不能据它推断GPU0或整个项目进度；应读取各campaign的`current_generation_run`。

02:22实测GPU1模拟器已进入Isaac扩展冷启动（约577MiB），尚无新成功演示，不声称此时已稳定100%负载。本地最终118项测试：89通过、29skip；服务器host数值测试与NNX AdamW检查通过。两卡任务仍只使用官方生成 / original pi0，不恢复旧替代策略。
