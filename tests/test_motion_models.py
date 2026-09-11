"""Equivalence tests: the extracted ConstantVelocity model must reproduce the
original tracker's hardcoded inline math exactly."""

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
            # predict
            pred_old = last_center + velocity
            pred_new = model.predict()
            assert np.allclose(pred_old, pred_new)

            # measurement
            m = rng.uniform(-1000, 1000, size=2)
            velocity = m - last_center          # old update
            last_center = m
            model.update(m)                     # new update
            assert np.allclose(last_center, model.state)
            assert np.allclose(velocity, model.velocity)


def test_failure_branch_carries_prediction_forward():
    # On a failed frame the original did `last_center = pred_center`
    # (velocity unchanged). The model's update(pred) must do the same.
    model = make_motion_model("ConstantVelocity", [10.0, 20.0])
    model.update([10.0, 20.0])          # seed: state == measurement, v = 0
    v_before = model.velocity.copy()
    pred = model.predict()
    model.update(pred)                  # failure-branch carry-forward
    assert np.allclose(model.velocity, v_before)
    assert np.allclose(model.state, pred)


def test_unknown_model_raises():
    try:
        make_motion_model("Nope", [0, 0])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model")


if __name__ == "__main__":
    test_constant_velocity_matches_original_hardcoded_math()
    test_failure_branch_carries_prediction_forward()
    test_unknown_model_raises()
    print("OK: motion_models equivalence verified")
