# General Target Tracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将鸟类专用的运动先验跟踪脚本升级为可跟踪任意目标、可输出具身状态的组件化视觉跟踪器。

**Architecture:** `TargetSpec` 和 `TrackerConfig` 描述目标与运行参数；`MotionPriorTracker` 负责模板匹配、运动先验和逐帧状态；`TrackObservation` 是面向 YOLO 与具身模块的稳定输出。旧 CLI 和 `process_folder` 保持兼容，并通过 JSONL telemetry 暴露速度、置信度和丢失状态。

**Tech Stack:** Python 3.11 recommended, NumPy, OpenCV, unittest/pytest-compatible tests, JSONL.

**Spec:** `docs/superpowers/specs/2026-09-13-general-target-tracker-design.md`

## Global Constraints

- 依赖仅使用现有的 `numpy` 和 `opencv-python`。
- 保留 `ConstantVelocity`、`KalmanFilter`、`make_motion_model` 现有调用方式。
- YOLO 输出格式保持 `class_id cx cy w h`。
- telemetry 字段使用 snake_case；速度单位为像素/帧。
- 不加入真实机器人控制、相机标定或检测器训练。

---

### Task 1: Extend motion-prior models

**Files:**
- Modify: `motion_models.py`
- Modify: `tests/test_motion_models.py`

**Interfaces:**
- Consumes: existing `MotionModel` contract and `make_motion_model(name, initial_state)`.
- Produces: registered `ConstantAcceleration`; `MotionModel.velocity` property available to telemetry; existing models remain backward compatible.

- [x] **Step 1: Write failing tests**

Add tests for `ConstantAcceleration` constant-acceleration trajectory, `advance()` matching `predict()`, model velocity exposure, and registry error listing the new model.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m pytest tests/test_motion_models.py -q`
Expected: failures because `ConstantAcceleration` is not registered and the velocity contract is absent.

- [x] **Step 3: Implement the minimal model changes**

Implement a six-state NumPy model `[px, py, vx, vy, ax, ay]` with discrete `dt`, pure `predict()`, measurement `update()`, no-measurement `advance()`, and `velocity` accessors on all models. Register it under `ConstantAcceleration`.

- [x] **Step 4: Run focused and existing tests**

Run: `python -m pytest tests/test_motion_models.py -q`
Expected: all tests pass.

- [x] **Step 5: Commit**

```powershell
git add motion_models.py tests/test_motion_models.py
git commit -m "feat: add constant-acceleration motion prior"
```

### Task 2: Componentize the generic tracker and telemetry

**Files:**
- Modify: `motion_prior_tracker.py`
- Create: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `make_motion_model`, existing matching helpers, and Task 1 velocity contract.
- Produces: `TargetSpec`, `TrackerConfig`, `TrackObservation`, `MotionPriorTracker`, `TrackObservation.to_dict()`, and compatibility `process_folder()`.

- [x] **Step 1: Write failing component tests**

Create deterministic synthetic frames containing a moving colored rectangle. Test non-bird `class_id`, normalized bbox, observation modes, lost-frame accounting, recovery mode, and JSON-serializable telemetry fields.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m pytest tests/test_tracker.py -q`
Expected: import/API failures because the component classes do not exist.

- [x] **Step 3: Implement the componentized tracker**

Move the existing matching loop into `MotionPriorTracker`, replacing bird-specific names with target names. Add dataclasses with validation, injectable `motion_model_name`, configurable thresholds, `update()` and `track_sequence()`, explicit `matched/predicted/recovered` modes, and `TrackObservation` velocity/lost-frame fields. Keep matching helpers available for compatibility.

- [x] **Step 4: Add output writers and compatibility wrapper**

Keep YOLO output behavior, add JSONL telemetry writing when requested, and make `process_folder()` construct the new tracker from the first-frame label and optional config without changing existing callers.

- [x] **Step 5: Run focused and full tests**

Run: `python -m pytest tests/test_tracker.py tests/test_motion_models.py -q`
Expected: all tests pass with no warnings.

- [x] **Step 6: Commit**

```powershell
git add motion_prior_tracker.py tests/test_tracker.py
git commit -m "feat: expose generic target tracking observations"
```

### Task 3: Generalize CLI and document embodied-intelligence integration

**Files:**
- Modify: `motion_prior_tracker.py`
- Modify: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `TargetSpec`, `TrackerConfig`, `MotionPriorTracker`, and telemetry writer from Task 2.
- Produces: CLI flags `--class-id`, `--telemetry`, `--model`, and a documented `TrackObservation` integration example.

- [x] **Step 1: Write failing CLI tests**

Test parser defaults and overrides for class ID, model, input, output, and telemetry path; test a one-folder run writes both YOLO and JSONL output.

- [x] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m pytest tests/test_cli.py -q`
Expected: failures because the new flags and telemetry path are not wired.

- [x] **Step 3: Implement CLI wiring**

Use parsed values to build `TargetSpec`/`TrackerConfig`, remove mutation of module globals, pass the selected class ID through output, and write one JSON object per frame when `--telemetry` is provided.

- [x] **Step 4: Update README**

Document generic targets, the three motion models, CLI examples for a bird and a non-bird target, JSONL schema, and a short pseudocode adapter showing how a robot policy consumes `TrackObservation`.

- [x] **Step 5: Run the complete verification suite**

Run: `python -m pytest -q`; then run `python motion_prior_tracker.py --help`.
Expected: all tests pass and help lists `--class-id`, `--telemetry`, and `--model`.

- [x] **Step 6: Commit**

```powershell
git add motion_prior_tracker.py README.md tests/test_cli.py
git commit -m "feat: add generic tracking CLI and embodied telemetry"
```


## Completion — 2026-09-14

All three tasks are complete. Final verification: 108 tests passed using real OpenCV; CLI help, compilation and whitespace checks passed. Two synthetic sequences were processed through the CLI (56 frames), and three priors were compared over both sequences (168 model-frames). A 12-frame real-sequence smoke check produced 3 accepted states including initialization and 9 predictions; this is not an accuracy measurement.

Implementation was split into tracking_types.py, template_matching.py, sequence_io.py and tracking_cli.py while retaining public exports from motion_prior_tracker.py. Review added stricter unpadded-target validation, a three-pixel initialization minimum, full scale-aware search windows, and corrected acceleration estimation across unequal measurement gaps. The attention adapter and reproducible synthetic demo are executable examples.

See ../../implementation-report-2026-09-14.md for the final implementation choices, evidence and limitations. The development branch is feature/generalize-target-tracker; consult GitHub for publication and merge status.
