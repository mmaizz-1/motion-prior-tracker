# Motion-Prior Tracker

**可替换运动先验的通用单目标视觉跟踪器。** 给定首帧中的目标框，利用外观模板与运动预测估计后续位置，在短时漏检或遮挡期间输出明确标记的预测状态。

项目起源于鸟类跟踪，目前把类别、初始化、匹配参数、运动模型和状态输出分离。行人、车辆、物体等可以使用同一套接口；更换 `class_id` 只改变目标标识，不会训练新的检测器，也不保证对所有类别都有效。

## 能做什么

- 由人工标注或 YOLO 等外部检测器提供首帧框，跟踪选中的一个目标。
- 通过上下文匹配、目标多尺度匹配和模板更新获得视觉测量；失配时只推进运动状态，间隔执行全图重捕获。
- 使用 `ConstantVelocity`、`KalmanFilter` 或 `ConstantAcceleration`；跟踪循环无需随模型变化。
- 输出 YOLO 标签，以及包含位置、速度、匹配状态和丢失时长的 `TrackObservation` / JSONL。

每个实例跟踪一个目标。当前没有自动目标发现、多目标关联、在线检测器融合或检测器训练；需要这些能力时，应在上层加入检测和管理模块。首帧只接受一个参考目标，多行标签会报错。初始化目标在两个方向都须至少占 3 个像素；更小目标缺乏足够的模板支持，直接报错。

## 安装与运行

建议使用 Python 3.11 的独立环境：

```bash
python -m pip install -r requirements.txt
```

输入可以是一个序列目录，也可以是包含多个序列的父目录。支持 `.jpg`、`.jpeg`、`.png`、`.bmp`，文件按数字自然排序，例如 `frame1`、`frame2`、`frame10`。每个序列按统一时间间隔采样，图像尺寸和通道数保持一致；程序逐帧解码，不一次载入所有原图。

```text
sequences/
  object01/
    frame1.png
    frame1.txt       # 一行：class_id cx cy width height，后四项归一化到 [0, 1]
    frame2.png
    frame10.png
  object02/
    ...
```

```bash
python motion_prior_tracker.py --input sequences --output results --model KalmanFilter --telemetry results/tracks.jsonl
```

处理单个序列并覆盖输出类别：

```bash
python motion_prior_tracker.py --input sequences/object01 --output results --class-id 7 --model ConstantAcceleration --search-radius 100 --fine-margin 40 --n-scales 25
```

不传 `--class-id` 时，保留各序列首帧标签中的类别。类别编号的语义由你的数据集定义。Windows 中文路径可直接使用，引号包裹含空格的路径。

输出目录为 `OUTPUT/序列名`，其中每帧一个同名 `.txt`，另有 `_tracking_stats.json`。根目录 `_tracking_stats.json` 是本次全部序列的汇总列表。指定 `--telemetry` 后，每个序列额外保留 `_track.jsonl`，指定路径汇总所有序列，并增加 `sequence`、`source`、`image_name`。重复运行替换上次完整汇总，不重复追加。

输出和 telemetry 路径不得覆盖源标注；telemetry 必须位于输入目录之外。缺失首帧标签、非法参数、无图像或解码失败会返回非零退出码。若运行中途失败，已生成的部分序列输出可能保留，应根据退出码判断整次运行是否完成。

## Python 接口

```python
from motion_prior_tracker import MotionPriorTracker, TargetSpec, TrackerConfig, imread

tracker = MotionPriorTracker(
    TargetSpec(class_id=7, search_radius=100, fine_margin=40),
    TrackerConfig(motion_model_name="KalmanFilter"),
)

# 框由你的检测器或标注提供，格式为归一化的 (cx, cy, width, height)。
initial_box = (0.40, 0.45, 0.10, 0.15)
first = tracker.initialize(imread("frame1.png"), initial_box)
next_observation = tracker.update(imread("frame2.png"))
print(next_observation.to_dict())

tracker.write_jsonl([first, next_observation], "results/observations.jsonl")
```

批量接口可接收迭代器；用 `TargetSpec(bbox_xywh=initial_box)` 提供初始化框：

```python
from motion_prior_tracker import MotionPriorTracker, TargetSpec, process_folder, imread

initial_box = (0.40, 0.45, 0.10, 0.15)
tracker = MotionPriorTracker(TargetSpec(class_id=7, bbox_xywh=initial_box))
observations = tracker.track_sequence(
    (imread(path) for path in ["frame1.png", "frame2.png"]),
    telemetry_path="results/observations.jsonl",
)

# 兼容原有目录处理入口；默认从首帧参考标签读取类别和初始框。
stats = process_folder("sequences/object01", "results")
```

已有状态的 `track_sequence()` 把传入的每一帧作为新帧继续跟踪。`initialize()` 可重置到新序列。`update(frame_index=...)` 若传帧号，必须在当前索引基础上加一；它不会推断不规则帧间隔。

## 状态协议与具身智能接口

`TrackObservation.to_dict()` 的主要字段：

| 字段 | 含义与单位 |
|---|---|
| `frame_index` | 序列内从 0 开始的帧索引 |
| `bbox_xywh` | 归一化的 `(中心 x, 中心 y, 宽, 高)`，显示框限制在图像内 |
| `center_px` | 显示框中心，单位像素 |
| `velocity_px` | 运动模型内部速度，单位像素/帧 |
| `image_size` | `(width, height)`，单位像素 |
| `class_id` | 调用方或首帧标签提供的类别编号 |
| `score` | 外观匹配相似度，不是校准后的检测概率 |
| `mode` | `ref` 初始化；`matched` 匹配成功；`predicted` 纯预测；`recovered` 丢失后恢复 |
| `lost_frames` | 连续没有有效视觉测量的帧数 |
| `measurement_valid` | `ref/matched/recovered` 为 true，`predicted` 为 false |

`ref` 的 `score=1` 表示采用外部给定的初始框；`predicted` 的 `score=0`，预测框不能当作新的检测结果。边界裁剪可能导致 `center_px` 与内部运动状态不一致，因此不要用显示框差分替代 `velocity_px`。

与具身智能的连接点是**视觉感知到行动决策之间的目标状态接口**。例如，上层视线控制器可根据图像中心误差和速度调整注意方向；主动搜索模块可根据 `lost_frames` 和预测位置安排搜索区域。最小接入逻辑可写为：

```python
state = next_observation.to_dict()
if state["measurement_valid"]:
    attention_target = state["center_px"]
    # 将有效观测交给你的目标选择/视线控制模块。
else:
    search_hint = (state["center_px"], state["lost_frames"])
    # 触发外部检测器重检，或交给你的主动搜索模块。
```

这些是上层模块可消费的接口和使用示意。目前输出是二维图像平面状态，没有相机标定、深度、三维世界坐标、机器人闭环控制或真实机器人验证。移动相机引起的全局运动也会影响目标的图像速度。面向具身任务的下一步需要接入相机运动补偿、深度/标定和任务闭环，并评估跟随成功率、重新发现目标时间等指标。

## 运动模型与参数

| 模型 | 当前行为 |
|---|---|
| `ConstantVelocity` | 以测量位置差估计速度，无测量时匀速推进 |
| `KalmanFilter` | 二维匀速 Kalman 状态估计，维护协方差；无测量时仅做时间更新 |
| `ConstantAcceleration` | 六维位置/速度/加速度状态，由实际测量间隔估计加速度，测量不足时退化为匀速 |

加速度先验可表达短时间内的加速运动，但对噪声和错误匹配更敏感，不意味着必然优于匀速或 Kalman 模型。需要使用相同数据和匹配参数进行对照实验。

自定义运动先验继承 `motion_models.MotionModel`，实现 `predict()`（不改状态）、`update(measurement)`（接受真实视觉测量）、`advance()`（无测量推进）及 `velocity`，再注册到 `make_motion_model` 使用的模型表。模型层可定义 `dt`；跟踪器当前使用每次调用一帧的默认时间单位。

CLI 暴露搜索、阈值、重捕获、EMA 和尺度数量参数，可用 `--help` 查看。`--target-thresh` 默认 0.65，`--template-update-thresh` 默认 0.8：只有较强的上下文/目标联合匹配才刷新模板，以减少背景污染。除带 padding 的模板定位外，还会单独比较未加 padding 的目标内容，防止目标已消失但周围背景仍然相似时误报恢复。`score` 是这些外观证据中的最低分；保守阈值可能增加丢失/预测帧，需要结合具体序列权衡。这些数值不是概率。

Python 的 `TargetSpec` 还可设置模板 padding、上下文范围和尺度区间；`TrackerConfig` 可设置上下文搜索尺度。默认参数是起点，应根据目标大小、采样间隔、背景和形变调节。

## 可复现实验示例

不需要外部数据即可运行两个合成非鸟类目标的三种运动模型对照：

```bash
python examples/synthetic_demo.py --output demo_out
```

脚本生成序列、可视化 PNG 和指标，便于检查定位、遮挡与不同先验的行为。合成数据结果不代表真实场景性能。

`examples/embodied_attention.py` 提供可调用的 `attention_hint(observation)` 示例：有效且分数不低于 0.5 的观测产生 `observe` 与图像中心误差；较弱观测或短时丢失产生 `hold`；连续丢失超过 3 帧产生 `search`。

```python
from examples.embodied_attention import attention_hint

hint = attention_hint(next_observation)
print(hint)
```

该示例输出注意/搜索提示，不执行机器人动作。接入实际设备仍需设计控制与安全边界，并完成闭环验证。

## 验证与局限

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试覆盖运动模型、非鸟类编号、状态协议、目标丢失与恢复、文件顺序、CLI 聚合以及源标注保护。合成序列用于验证行为和接口；类别编号测试只证明没有鸟类编号硬编码，不能证明跨类别跟踪精度。真实数据的效果需要带完整逐帧真值的基准实验，且应将预测帧和实际匹配帧分开评估。

模板方法仍可能在相似背景、长期遮挡、严重形变、镜头运动或目标离开画面时漂移；重新匹配也不等于已经验证目标身份。

原鸟类示例图像保留在 `examples/demo/`，用于回顾项目来源，不构成通用目标效果评测：

![首帧目标局部图](examples/demo/seq16_000461_crop.jpg)

![原示例跟踪局部图](examples/demo/seq16_000711_crop.jpg)

## 目录

```text
motion_prior_tracker.py  # 初始化/更新/序列 API、公共类型导出
tracking_types.py       # 目标与跟踪配置、结构化观测
template_matching.py    # 外观搜索与模板工具
motion_models.py        # 可替换运动先验
sequence_io.py          # 逐帧图像读取与标签/状态输出
tracking_cli.py         # 命令行与多序列汇总
tests/                  # 数学、跟踪和 CLI 测试
examples/               # 原有示例材料
```

本轮修改、验证结果与实际能力边界见 [实施与验证记录](docs/implementation-report-2026-09-14.md)。

## License

[MIT](LICENSE) © 2026 Kevin Yan
