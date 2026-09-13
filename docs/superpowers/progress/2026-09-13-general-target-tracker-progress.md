# 通用目标跟踪器工作记录

更新时间：2026-09-13

## 当前仓库

- 工作目录：`C:\Users\Lenovo\Documents\Codex\2026-09-13\new-chat\work\motion-prior-tracker`
- 分支：`feature/generalize-target-tracker`
- 远程仓库：`https://github.com/mmaizz-1/motion-prior-tracker`

## 已完成

1. 已阅读 GitHub 仓库、README、跟踪主循环、运动模型和测试。
2. 已盘点 E 盘数据：`E:\26年大创—观鸟(E Yolo)`，其中跟踪统计显示不同序列匹配率差异很大。
3. 已提交设计规格：`1245940 docs: specify general target tracker design`。
4. 已提交实施计划：`2d9c801 docs: add general tracker implementation plan`。
5. Task 1 已实现并提交：`8e8ff2e feat: add constant-acceleration motion prior`。
   - 新增 `ConstantAcceleration` 六状态模型。
   - 所有运动模型提供 `velocity` 属性。
   - 当前测试：10 passed。

## 当前阻塞点

Task 1 审查发现一个 P1 问题：`ConstantAcceleration.advance()` 只更新位置，没有按 `v += a * dt` 更新速度。连续丢帧时会违反恒加速度方程。修复需要补充“两次连续 advance”回归测试，并重新运行测试。

## 下一步顺序

1. 修复并审查 `ConstantAcceleration.advance()`。
2. 实现 `TargetSpec`、`TrackerConfig`、`TrackObservation`、`MotionPriorTracker` 和 JSONL telemetry。
3. 泛化 CLI，加入 `--class-id`、`--telemetry`，更新 README 和 CLI 测试。
4. 运行完整 `python -m pytest -q`、`python motion_prior_tracker.py --help`，做最终代码审查。

## 重要约束

- 不修改 E 盘原始数据。
- 不加入真实机器人控制、相机标定或检测器训练。
- 保持 YOLO 输出格式和旧 `process_folder()` 调用兼容。

