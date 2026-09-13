"""Tests for the pluggable motion models.

- ConstantVelocity must reproduce the original tracker's hardcoded inline math.
- KalmanFilter must behave like a correct constant-velocity Kalman filter.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from motion_models import make_motion_model


def test_constant_velocity_matches_original_hardcoded_math():
    rng = np.random.default_rng(0)
    for _ in range(100):
        init = rng.uniform(-1000, 1000, size=2)

        # original inline math (bird_tracker_v4.py)
        last_center = np.array(init, dtype=np.float64)
        velocity = np.array([0.0, 0.0])

        # extracted model
        model = make_motion_model("ConstantVelocity", init)

        for _ in range(50):
            pred_old = last_center + velocity
            pred_new = model.predict()
            assert np.allclose(pred_old, pred_new)

            m = rng.uniform(-1000, 1000, size=2)
            velocity = m - last_center
            last_center = m
            model.update(m)
            assert np.allclose(last_center, model.state)
            assert np.allclose(velocity, model.velocity)


def test_failure_branch_advance_carries_prediction_forward():
    # On a lost frame the original did `last_center = pred_center` (velocity
    # unchanged). advance() must do the same.
    model = make_motion_model("ConstantVelocity", [10.0, 20.0])
    model.update([10.0, 20.0])          # seed: state == measurement, v = 0
    v_before = model.velocity.copy()
    pred = model.predict()
    state = model.advance()
    assert np.allclose(model.velocity, v_before)
    assert np.allclose(state, pred)


def test_unknown_model_raises():
    try:
        make_motion_model("Nope", [0, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model")


def test_kalman_registered_and_instantiable():
    model = make_motion_model("KalmanFilter", [100.0, 200.0])
    assert model.x.shape == (4,)
    assert model.x[0] == 100.0 and model.x[1] == 200.0


def test_kalman_predict_is_pure():
    model = make_motion_model("KalmanFilter", [0.0, 0.0])
    x0 = model.x.copy()
    P0 = model.P.copy()
    model.predict()
    assert np.array_equal(model.x, x0)
    assert np.array_equal(model.P, P0)


def test_kalman_advance_lands_where_predict_said():
    model = make_motion_model("KalmanFilter", [10.0, 20.0])
    pred = model.predict()
    state = model.advance()
    assert np.allclose(state, pred)
    assert np.allclose(model.x[:2], pred)


def test_kalman_filter_reduces_error_on_constant_velocity():
    rng = np.random.default_rng(1)
    true_vel = np.array([4.0, -3.0])
    pos = np.array([100.0, 200.0])
    model = make_motion_model("KalmanFilter", pos)

    meas_errs, filt_errs = [], []
    for _ in range(200):
        pos = pos + true_vel
        z = pos + rng.normal(0.0, 8.0, size=2)
        est = model.update(z)
        meas_errs.append(np.linalg.norm(z - pos))
        filt_errs.append(np.linalg.norm(est - pos))

    # steady-state filter beats the raw measurement
    assert np.mean(filt_errs[-50:]) < np.mean(meas_errs[-50:])
    # velocity recovered
    assert np.allclose(model.x[2:], true_vel, atol=1.5)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} tests passed")
