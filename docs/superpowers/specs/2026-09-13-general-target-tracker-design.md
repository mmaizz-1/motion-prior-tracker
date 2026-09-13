# 通用目标运动先验跟踪器设计

## 目标

将当前面向鸟类的模板跟踪脚本抽象为可复用的视觉状态估计器。目标类别、标签编号、模板参数和运动模型由调用方提供；跟踪结果同时支持 YOLO 标注和供具身智能使用的结构化状态流。

## 现状与问题

- `motion_prior_tracker.py` 的匹配循环已经独立于运动模型，但变量、阈值和输出仍以 bird 为中心。
- 运动模型只返回中心点，控制模块无法获得框、速度、置信度、丢失时长和恢复状态。
- 全局常量和 `process_folder` 让库调用、单帧调用以及不同目标配置变得困难。
- 现有测试覆盖运动模型数学行为，缺少非鸟类目标、丢失恢复和输出协议测试。

## 设计

### 1. 领域对象

在 `motion_prior_tracker.py` 中增加：

- `TargetSpec`：目标类别、初始 YOLO 框、模板 padding、上下文 margin、搜索半径、细搜范围、尺度范围和阈值。
- `TrackerConfig`：运行级配置，包括模板 EMA、重捕获间隔/缩放、上下文尺度、输出选项和运动模型名称。
- `TrackObservation`：每帧不可变状态，包含 `frame_index`、`bbox_xywh`、`center_px`、`velocity_px`、`score`、`mode`、`lost_frames`、`class_id` 和图像尺寸，并提供 `to_dict()`。
- `MotionPriorTracker`：持有模板与运动状态，提供 `initialize(image, bbox_xywh)`, `update(image, frame_index)` 和 `track_sequence(images)`。

旧的 `process_folder(folder_path, output_base)` 保留为兼容包装器，内部使用新组件。

### 2. 通用目标与输出

- 所有 bird 前缀改为 target；默认 `class_id=0`，CLI 增加 `--class-id`。
- YOLO 文件继续输出 `class_id cx cy w h`，保证现有数据链兼容。
- 可选 `--telemetry PATH` 输出 JSONL，每行对应一个 `TrackObservation`。JSON 字段使用稳定的 snake_case，速度为像素/帧，`mode` 取 `ref/matched/predicted/recovered`。
- telemetry 作为具身接口：机器人跟随、视线控制或主动搜索模块只依赖 `TrackObservation`，不依赖 OpenCV 模板细节。

### 3. 运动先验

- 保留 `ConstantVelocity` 和 `KalmanFilter` 的已有行为。
- 增加纯 NumPy `ConstantAcceleration`，状态为 `[px, py, vx, vy, ax, ay]`，实现 `predict/update/advance`，并注册到 `make_motion_model`。
- `MotionModel` 增加 `velocity` 属性约定；实现无法提供速度时返回零向量，telemetry 仍保持 schema 稳定。

### 4. 错误处理

- 对空序列、缺少首帧标签、无法读取图片和非法 bbox 给出 `ValueError` 或清晰的跳过日志。
- 所有归一化框继续限制在合法范围；匹配失败时显式记录 `predicted` 与 `lost_frames`。

## 测试策略

- 运动模型：恒加速度轨迹、无测量 advance、注册表错误信息。
- 跟踪器：构造带平移矩形的合成图像，验证 `class_id` 可为非鸟类值、模式和 bbox 输出。
- 丢失恢复：中间帧遮挡后验证 `predicted`、`lost_frames` 递增以及恢复后回到 `recovered`。
- telemetry：验证 JSONL 每行字段、数值类型和速度字段。
- 保持现有 `tests/test_motion_models.py` 全部通过。

## 非目标

本次不加入真实机器人控制、相机标定、3D 世界坐标转换或深度学习检测器训练；这些能力通过 telemetry 接口在下一阶段接入。

