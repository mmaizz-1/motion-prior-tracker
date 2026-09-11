# Motion-Prior Tracker

一个「物理先验可插拔」的视觉目标跟踪框架。

> 不是"一个模型吃遍所有物理"，而是"**框架不变，物理模型可换**"。

## 一句话

模板匹配 + EMA 平滑的跟踪主循环保持不动，运动学先验（常速度 / 常加速度 / 卡尔曼 / 鸟飞行动力学 / 抛体……）作为 `MotionModel` 接口可插拔替换。

## 为什么做这个

跟踪小目标（如鸟类）时，纯外观匹配在快速机动、遮挡、形变下容易丢失。把目标运动的**物理先验**显式抽出来，既可作为预测先验补足外观，也让「力学 × 视觉」的交叉变成一行代码的切换。

本项目源自中山大学大创《大规模鸟类数据集构建与跟踪算法研究》中自研的 `bird_tracker`，去鸟化后抽成通用框架。

## 核心特性

- **可插拔运动模型**：`motion_models.MotionModel`（`predict` / `update` 两个方法）
- 上下文放大模板匹配（鲁棒定位）
- 50 尺度稠密匹配 + 二次插值（连续 w/h）
- EMA 平滑（尺度 / 模板时序一致性）
- 速度预测 + 原始模板重捕获（丢失恢复）

## 快速开始

依赖：`numpy`、`opencv-python`

```bash
pip install numpy opencv-python
```

把待跟踪的图片序列放到一个文件夹里，每段序列一个子文件夹（首帧放一个 YOLO 参考 `.txt`），然后：

```bash
python motion_prior_tracker.py --input /path/to/sequences --output /path/to/out --model ConstantVelocity
```

输出 YOLO 格式 `.txt`（`class_id cx cy w h`）。

## Demo

真实鸟序列（4K 视频 5 帧采样，目标约 40×25 像素，占整幅不到 1%）上的跟踪结果，绿色框即目标：

**参考帧（第 0 帧）放大裁剪：**

![ref crop](examples/demo/seq16_000461_crop.jpg)

**跟踪中（第 50 帧）放大裁剪：**

![track crop](examples/demo/seq16_000711_crop.jpg)

全帧缩小图（可见目标有多小）：`examples/demo/seq16_000461.jpg`、`seq16_000711.jpg`。

## 换一个物理模型

```python
from motion_models import make_motion_model

model = make_motion_model("ConstantVelocity", init_center)  # 默认
# model = make_motion_model("KalmanFilter", init_center)     # roadmap
```

把 `motion_prior_tracker.py` 里的 `MOTION_MODEL` 改成对应名字即可，主循环一行都不用动。

## 运动模型路线图

| 模型 | 状态 | 说明 |
|------|------|------|
| `ConstantVelocity` | ✅ v0.1 | 与原始 tracker 等价，作为 baseline |
| `ConstantAcceleration` | 计划中 | x'' = const |
| `KalmanFilter` | 计划中 | 状态空间 + 协方差（力学主场） |
| `BirdFlight` | 计划中 | 鸟飞行动力学先验（创新点） |
| `Projectile` | 计划中 | 抛体先验（证明框架通用性） |

## 测试

```bash
python tests/test_motion_models.py
```

验证抽出的 `ConstantVelocity` 与原始 `bird_tracker` 内联数学**逐帧等价**。

## 目录

```
motion-prior-tracker/
├── motion_prior_tracker.py   # 主循环（框架）
├── motion_models.py          # 可插拔运动模型
├── examples/
│   ├── visualize.py          # 画框/裁剪工具
│   └── demo/                 # 示例图
├── tests/
│   └── test_motion_models.py # 等价性测试
└── README.md
```

## License

待定（v0.1 先用 MIT，正式发前确认）。
