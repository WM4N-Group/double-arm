# 两张GPU的明确授权与并行工作流（2026-11-16）

## 1. 新授权覆盖旧预留，现有实验不改配方

用户明确说 **“两张卡都放开给用，跑满”**。23:05核查GPU0空闲，GPU1约22,124MiB / 100%，coverage v3约79.4k / 新增90k训练中。23:10:52 CST记录新授权，两张卡均可用于项目、时间开放；不再把GPU0称为预留卡，也不需要重复索要窗口。

但资源授权不是改变科学对照的理由：GPU1上的native coverage v3继续原样，仍是16 demos / batch1 / native mean-std / warm-start weights + fresh optimizer的诊断，不是双卡paper训练。其后续自动评估与视频交付要求不变。未恢复旧memory / state-MLP双卡控制器。

## 2. 资源元数据兼容性

在写入新授权前，旧单卡分配与旧checker副本归档到远端 `runs/resource-allocation-history/`，没有改写历史campaign状态。

当前根分配文件的新增主项目字段：

```text
dual_gpu_authorized = true
allowed_gpu_indices = [0, 1]
allowed_gpu_uuids = [GPU-925e5f5c-3cbb-7e93-5a98-1f06d3a71c0d,
                    GPU-50e5d38d-2825-e379-f537-7f0bf4609aba]
total_authorized_gpu_count = 2
parallel_authorization_text = 两张卡都放开给用，跑满
```

**兼容字段不是项目总数**：旧控制器已把旧`resolve_gpu`函数加载到内存，且运行期间不断重读分配文件。因此保留 `allowed_gpu_index=1`、`allowed_gpu_uuid=GPU1 UUID`、`max_gpu_count=1`，并显式写`max_gpu_count_scope=legacy_per_job_limit_NOT_project_total_keep_live_GPU1_controller_compatible`。直接把max改为2会使旧函数主动中止当前训练链路。已用归档的原checker在新JSON上实际验证，仍解析到原GPU1。

新`resource_allocation.py`默认依然GPU1；只有完整的新双卡allowlist、时间与授权说明同时存在，显式`--requested 0`才可解析到GPU0。单个任务仍固定一个UUID；不接受`0,1`隐式扩成多卡。旧`require_dual_gpu_allocation`接口仍拒绝恢复不符合original-pi0目标的历史控制器。

## 3. GPU0任务：原版官方D0示范池，非策略训练

```text
controller: /mnt/yuanyue/double-arm/code/scripts/run_paper_generation_shards.py
campaign: /mnt/yuanyue/double-arm/runs/paper-official-pool-gpu0-20261116-231115
controller PID: 143656
GPU UUID: GPU-925e5f5c-3cbb-7e93-5a98-1f06d3a71c0d
CPU affinity: 0-15
start: 2026-11-16 23:11:15 CST
```

调用现有 `run_official_generation.sh` 与锁定官方MoMaGen入口 / 作者D0配置，不新部署环境、不升级库、不修改源动作标签、不加入规则策略。使用此前已经确认的本地license文件；未写入任何新的接受标志或重新替用户接受条款。

- 目标：1000条**独立去重的有效原始官方成功演示**，每shard请求25个保证成功的回合，seed10000起逐shard增加。并非声称这1000条已经产生。
- 每shard6小时watchdog，最多80 shards作为防无限重试界限；不是限制用户的开放授权时间。失败停止、保留临时数据和日志供诊断，不无声重启。
- 控制器CPU-only；仿真子进程GPU0 UUID固定，单独的GPU锁 / 进程占用检查。不为“跑满”启动无意义GPU忙循环，也不把启动阶段低利用率说成没工作。
- 只有生成器退出0、官方final stats / HDF5通过且真实三路RGB / 原21D proprio / native21D动作可读的final shard，才做去重计数。图像抽查首 / 中 / 尾帧非空，256×256原记录；不是每帧图像人工复核。
- 去重复用已有转换器的`actions bytes + initial original proprio bytes`指纹。这是轨迹字节去重，不保证语义多样性或1000个独立环境布局；D0仍是原D0。
- 数据保留 `source_grasping_mode=assisted`。官方任务成功与辅助附着不等于学得策略成功，也不是纯物理抓取数据。
- 不修改现有16条train / 1条heldout、不创建新train/validation split、不计算新norm stats、不启动新的正式π₀训练。后续全新数据版本需要独立清洗 / 动作语义 / 固定划分与provenance审查，不能把旧converter的一条heldout默认直接叫90/10复现。
- 全部HDF5 / 图像 / 产物只在服务器。此工作流不下载任何原始数据至本地，也不提交媒体 / 数据。

## 4. 两条独立状态指针

```text
runs/latest-multi-gpu-jobs.json            两卡角色与各自pointer映射
runs/latest-single-gpu-job.json            保留原GPU1 coverage指针，未改写
runs/latest-pi0-coverage-campaign.txt       同上
runs/latest-gpu0-job.json                  新GPU0示范池工作流
runs/latest-paper-generation-campaign.txt 新GPU0 campaign路径
```

GPU0完整结束后是“原始池达到目标、清洗划分与训练仍待进行”，绝不写“论文复现成功”。GPU1若评估失败，仍按相应新checkpoint导出并展示FAILED review；GPU0生成视频不能冒充π₀推理视频。

## 5. 验证边界

- 本地112项测试：85通过、27数值依赖skip，包括新增授权、去重、run-path解析，以及历史单卡 / 禁用旧控制器的回归测试。
- 服务器确认新checker默认GPU1，显式0解析GPU0；旧checker在新分配文件上仍通过。原controller117227 / trainer118864存活。
- 服务器Python3.10环境的5项新增授权 / 去重 / 路径测试通过。新shard验收函数在旧16条官方final数据上CPU实测通过（27,657帧、assisted）；这些旧演示没有计入新的1000条池。
- 23:13确认GPU0模拟器PID143710已进入Isaac扩展启动，`nvidia-smi`归属GPU0，约588MiB；当时仍在冷启动，不是稳定100%利用率或新成功示范。GPU1仍约100%，训练约81.4k/90k。已直接读取两进程环境和CPU亲和性，分别固定GPU0/0-15与GPU1/88-103。实际场景加载完成和生成成功仍需后续日志核验。
- 没有新策略闭环结果，没有已验证的真实π₀batch64 GPU累积。待GPU1完整工作流结束后，可以在GPU1做该独立预检，GPU0继续生成；两者不互相改写状态。
