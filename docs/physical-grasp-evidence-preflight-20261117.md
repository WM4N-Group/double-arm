# 无辅助抓取证据预检：仅CPU只读工具准备，未运行physical实验（2026-11-17）

## 当前边界

用户指出局部视频中的吸附后，结论继续保持：**A source635片段只证实触发辅助附着后抬杯，没有验证物理夹持。** 当前GPU1 uniform B训练/自动评估链路不改、不重启、不插入其它GPU任务。下面只是未来独立physical诊断的CPU侧证据工具准备，**没有生成新physical结果、没有接入B的in-flight evaluator或改变验收门槛**。

## 新增只读工具

`/home/yuanyue/double-arm/scripts/physical_grasp_observation.py`

它不导入模拟器、不调用任何setter、不改controller、不产生VLA输入。返回每只手的已知0/1或未知-1：

- 当前`grasping_mode`是否实际为physical；
- `_ag_obj_constraints`中是否有active对象；
- `_ag_obj_in_hand`是否为空；
- `_ag_freeze_gripper`是否明确False；
- **实际USD stage**上末端`.../ag_constraint` prim是否存在。

USD检测由未来模拟器中的只读callback提供。未提供、query失败、返回未知、字段缺失、缺少任一手，均保留errors/-1并拒绝确认。不能只改一个mode字符串、清空Python字典或靠`object_pose_write_count=0`就称没有辅助。工具不会对engine object调用`bool(obj)`，避免把非法/特殊truthiness当作空状态。

`no_assistance_confirmed`只确认上述已知证据下**未见该assistance机制**，不包含/不返回`grasp_success`或`task_success`。真实物理夹持仍需独立的接触、持续持杯、抬升/运输和视频检验；EEF-origin距离不是指尖净空，friction4.0也不是实机标定。

目前只有mock/fixture验证，尚未连接实际USD stage或收集真实per-step样本；**不得把6个测试通过写成physical-only验证通过**。

## 后续初始化仍需预检，不能简单切mode

既有`grasping_validation.py`注释只允许empty-handed reset后转换，明确警告不要直接转换demonstration mid-state；该文件未修改。后续如果设计source635附近的局部physical对照，必须另立初始化协议，验证完整状态/assisted状态布局、手中对象/约束/freeze及USD joint，再在整个实际控制过程中记录，不是把一行`_grasping_mode='physical'`当完整修复。不能从已吸附/正在拿持的状态切模式后宣称物理夹持。

没有排队新的GPU job，也没有绕过B完整source0 TRAIN/assisted → TRAIN/physical → heldout/physical门禁。B full workflow释放前，任何新局部physical实验都不能启动。

## 测试

新增6项纯CPU测试在本地和服务器Python3.10通过：实际query缺失/异常保持未知、物理标签不能掩盖USD joint、held object/freeze/assisted mode拒绝、缺失arm拒绝、无assistance报告不冒充grasp/task成功。全本地182项：138通过/44依赖skip。未安装/升级环境，没有加载模型或运行模拟器。
