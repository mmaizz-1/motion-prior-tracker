# Task 1 Report: Extend Motion-Prior Models

Implemented and committed the constant-acceleration motion prior.

## Verification

- `python -m pytest tests/test_motion_models.py -q` (before implementation): failed as expected with 4 failures because `ConstantAcceleration` was not registered and the registry error did not list it.
- `python -m pytest tests/test_motion_models.py -q`: passed, 10 tests.
- `python -m pytest -q`: passed, 10 tests.

The model uses state `[px, py, vx, vy, ax, ay]`, supports pure `predict()`, measurement `update()`, no-measurement `advance()`, and exposes velocity through the common model contract. Existing ConstantVelocity and KalmanFilter behavior remains covered.

## Commit

`8e8ff2e feat: add constant-acceleration motion prior`

## Review Fix: Consecutive No-Measurement Advances

Fixed `ConstantAcceleration.advance()` so each no-measurement step propagates
both position and velocity under the stored acceleration. Added a regression
test covering two consecutive advances and preserving acceleration.

## Verification

- Regression test failed before the fix: the second position was `[8, 8]`
  instead of `[9, 9]` because velocity remained `[2, 2]`.
- `python -m pytest tests/test_motion_models.py -q`: passed, 11 tests.
- `python -m pytest -q`: passed, 11 tests.
- `git diff --check`: passed.
