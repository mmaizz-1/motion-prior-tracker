"""Tests for the pluggable motion models.

- ConstantVelocity must reproduce the original tracker's hardcoded inline math.
- KalmanFilter must behave like a correct constant-velocity Kalman filter.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from motion_models import ConstantAcceleration, KalmanFilter, make_motion_model


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
    except ValueError as exc:
        assert "ConstantAcceleration" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown model")


def test_constant_acceleration_predicts_constant_acceleration_trajectory():
    # x(t) = 0.5*t + 0.5*t**2: x(0..3) = 0, 1, 3, 6.
    model = make_motion_model("ConstantAcceleration", [0.0, 0.0])
    model.update([1.0, 1.0])
    model.update([3.0, 3.0])

    assert np.allclose(model.velocity, [2.5, 2.5])
    assert np.allclose(model.predict(), [6.0, 6.0])


def test_constant_acceleration_advance_matches_predict():
    model = make_motion_model("ConstantAcceleration", [0.0, 0.0])
    model.update([1.0, 1.0])
    model.update([3.0, 3.0])
    predicted = model.predict()

    advanced = model.advance()

    assert np.allclose(advanced, predicted)
    assert np.allclose(model.state[:2], predicted)


def test_constant_acceleration_consecutive_advances_propagate_velocity():
    model = make_motion_model("ConstantAcceleration", [0.0, 0.0])
    model.update([1.0, 1.0])
    model.update([3.0, 3.0])

    first = model.advance().copy()
    second = model.advance()

    assert np.allclose(first, [6.0, 6.0])
    assert np.allclose(second, [10.0, 10.0])
    assert np.allclose(model.velocity, [4.5, 4.5])
    assert np.allclose(model.state[4:6], [1.0, 1.0])


def test_constant_acceleration_recovers_after_unequal_measurement_gaps():
    # x(t) = t**2; y(t) = 10 - 2*t**2.
    model = ConstantAcceleration([0.0, 10.0])
    model.update([1.0, 8.0])  # t=1, acceleration is not yet observable
    model.advance()          # t=2, an extrapolation is not a measurement
    model.advance()          # t=3
    model.update([16.0, -22.0])  # t=4, establishes acceleration over a gap
    assert np.allclose(model.velocity, [8.0, -16.0])
    assert np.allclose(model.predict(), [25.0, -40.0])

    model.advance()          # t=5
    model.update([36.0, -62.0])  # t=6, another interval length
    assert np.allclose(model.velocity, [12.0, -24.0])
    assert np.allclose(model.predict(), [49.0, -88.0])


@pytest.mark.parametrize("dt", [0.25, 2.0])
def test_constant_acceleration_obeys_trajectory_equations_at_nonunit_dt(dt):
    # Independent world trajectory: p(t) = p0 + v0*t + 0.5*a*t**2.
    p0 = np.array([7.0, -5.0])
    v0 = np.array([-3.0, 2.0])
    acceleration = np.array([4.0, -6.0])
    model = ConstantAcceleration(p0, dt=dt)
    for step in (1, 2, 3):
        t = step * dt
        model.update(p0 + v0 * t + 0.5 * acceleration * t**2)
    assert np.allclose(model.velocity, v0 + acceleration * (3 * dt))
    expected = p0 + v0 * (4 * dt) + 0.5 * acceleration * (4 * dt)**2
    assert np.allclose(model.predict(), expected)
    assert np.allclose(model.advance(), expected)


def test_motion_model_factory_forwards_configuration():
    model = make_motion_model("ConstantAcceleration", [0.0, 0.0], dt=2.0)
    model.update([4.0, -4.0])
    model.update([16.0, -16.0])
    assert np.allclose(model.velocity, [8.0, -8.0])
    assert np.allclose(model.predict(), [36.0, -36.0])


@pytest.mark.parametrize("model_class", [ConstantAcceleration, KalmanFilter])
@pytest.mark.parametrize("dt", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_motion_models_reject_nonpositive_or_nonfinite_dt(model_class, dt):
    with pytest.raises(ValueError, match="dt"):
        model_class([0.0, 0.0], dt=dt)


@pytest.mark.parametrize("name", ["ConstantVelocity", "ConstantAcceleration", "KalmanFilter"])
@pytest.mark.parametrize("position", [[1.0], [[1.0, 2.0]], [1.0, 2.0, 3.0], [np.nan, 2.0], [1.0, np.inf]])
def test_motion_models_reject_invalid_initial_positions(name, position):
    with pytest.raises(ValueError, match="initial_state"):
        make_motion_model(name, position)


@pytest.mark.parametrize("name", ["ConstantVelocity", "ConstantAcceleration", "KalmanFilter"])
@pytest.mark.parametrize("position", [[1.0], [1.0, 2.0, 3.0], [np.nan, 2.0], [1.0, np.inf]])
def test_invalid_measurements_do_not_mutate_motion_model(name, position):
    model = make_motion_model(name, [0.0, 0.0])
    model.update([2.0, 4.0])
    prediction = model.predict().copy()
    velocity = model.velocity.copy()
    covariance = model.P.copy() if name == "KalmanFilter" else None
    with pytest.raises(ValueError, match="measurement"):
        model.update(position)
    assert np.array_equal(model.predict(), prediction)
    assert np.array_equal(model.velocity, velocity)
    if covariance is not None:
        assert np.array_equal(model.P, covariance)


@pytest.mark.parametrize("name", ["ConstantVelocity", "ConstantAcceleration", "KalmanFilter"])
def test_caller_arrays_and_velocity_snapshots_cannot_mutate_model(name):
    initial = np.array([1.0, 2.0])
    model = make_motion_model(name, initial)
    initial[:] = 99.0
    assert np.allclose(model.predict(), [1.0, 2.0])
    measurement = np.array([3.0, 4.0])
    model.update(measurement)
    prediction = model.predict().copy()
    measurement[:] = 99.0
    model.velocity[:] = 99.0
    assert np.allclose(model.predict(), prediction)


def test_motion_models_expose_velocity():
    constant_velocity = make_motion_model("ConstantVelocity", [0.0, 0.0])
    kalman = make_motion_model("KalmanFilter", [0.0, 0.0])
    acceleration = make_motion_model("ConstantAcceleration", [0.0, 0.0])

    assert np.allclose(constant_velocity.velocity, [0.0, 0.0])
    assert np.allclose(kalman.velocity, [0.0, 0.0])
    assert np.allclose(acceleration.velocity, [0.0, 0.0])


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
    raise SystemExit(pytest.main([__file__]))
