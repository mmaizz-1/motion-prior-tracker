"""
motion_models.py — pluggable motion priors for the Motion-Prior Tracker.

A motion prior is a stateful object with three methods:

    predict()            -> predicted position next frame (does NOT mutate state)
    update(measurement)  -> fold a measurement into the state (mutates)
    advance()            -> advance the dynamics one step with no measurement (mutates)

The tracking loop only ever calls these three methods, so the physics can be
swapped without touching the loop. This is the whole point of the framework:
"the loop stays fixed, the physics model is pluggable."

`advance()` exists because the loop has a third branch the two-method interface
could not express cleanly: a frame where the target is *lost* (no measurement).
ConstantVelocity used to fake this as update(prediction); KalmanFilter must not —
feeding its own prediction back as a measurement would spuriously shrink its
covariance.
"""

import numpy as np


class MotionModel:
    """Base class for pluggable motion priors."""

    def predict(self):
        """Predicted position for the next frame (pure — no state mutation)."""
        raise NotImplementedError

    def update(self, measurement):
        """Fold a measurement into the state (mutates). Returns position."""
        raise NotImplementedError

    def advance(self):
        """Advance one time step with no measurement (mutates). Returns position."""
        raise NotImplementedError


class ConstantVelocity(MotionModel):
    """Discrete constant-velocity model.

        x(t+1) = x(t) + v          (one frame == one time step)
        v      = x(t) - x(t-1)

    This is the baseline model the original tracker used, hardcoded in its
    main loop; it is now expressed as a MotionModel so it can be swapped out.
    """

    def __init__(self, initial_state):
        self.state = np.asarray(initial_state, dtype=np.float64)
        self.velocity = np.zeros_like(self.state)

    def predict(self):
        return self.state + self.velocity

    def update(self, measurement):
        m = np.asarray(measurement, dtype=np.float64)
        self.velocity = m - self.state
        self.state = m
        return self.state

    def advance(self):
        # carry the prediction forward with velocity unchanged (== old update(pred))
        self.state = self.state + self.velocity
        return self.state


class KalmanFilter(MotionModel):
    """Constant-velocity Kalman filter over a 2D position (pure numpy, no scipy).

    State      x = [px, py, vx, vy]   (pixels, pixels/frame)
    Measurement z = [px, py]          (observed centre)

        x_k = F x_{k-1}             (time update / predict)
        P_k = F P_{k-1} F^T + Q
        K   = P_k H^T (H P_k H^T + R)^{-1}
        x   = x_k + K (z - H x_k)   (measurement update)
        P   = (I - K H) P_k

    `predict()` peeks at H F x without committing; `advance()` commits the time
    update only (lost target); `update(z)` commits time update + correction.

    Noise/prior parameters are constructor kwargs with defaults suitable for
    pixel-space centres; tune them per dataset:
        pos_std    initial position uncertainty (px)
        vel_std    initial velocity uncertainty (px/frame)
        accel_std  acceleration noise std (px/frame^2)
        meas_std   measurement noise std (px)
    """

    def __init__(self, initial_state, dt=1.0, pos_std=20.0, vel_std=10.0,
                 accel_std=1.0, meas_std=10.0):
        px, py = float(initial_state[0]), float(initial_state[1])
        self.dt = dt
        self.x = np.array([px, py, 0.0, 0.0])

        # State transition (constant velocity, discrete, dt = 1 frame)
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        # Process noise Q from a discrete white-noise-acceleration model
        q = accel_std ** 2
        dt2 = dt * dt
        self.Q = q * np.array([
            [dt2 * dt2 / 4.0, 0.0, dt2 * dt / 2.0, 0.0],
            [0.0, dt2 * dt2 / 4.0, 0.0, dt2 * dt / 2.0],
            [dt2 * dt / 2.0, 0.0, dt2, 0.0],
            [0.0, dt2 * dt / 2.0, 0.0, dt2],
        ])
        # Measurement noise
        self.R = (meas_std ** 2) * np.eye(2)

        # Initial covariance (diagonal prior)
        self.P = np.diag([pos_std ** 2, pos_std ** 2,
                          vel_std ** 2, vel_std ** 2])

    def _time_update(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def predict(self):
        # peek: where the state says the target should be next, without committing
        return self.H @ (self.F @ self.x)

    def advance(self):
        # no measurement this frame: commit the time update only
        self._time_update()
        return self.H @ self.x

    def update(self, measurement):
        z = np.asarray(measurement, dtype=np.float64)
        self._time_update()
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.H @ self.x


_REGISTRY = {
    "ConstantVelocity": ConstantVelocity,
    "KalmanFilter": KalmanFilter,
    # Roadmap (coming next):
    #   "ConstantAcceleration" — x'' = const
    #   "BirdFlight"           — bird flight dynamics prior (the research novelty)
    #   "Projectile"           — ballistic prior (proves the framework is general)
}


def make_motion_model(name, initial_state):
    """Instantiate a motion model by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown motion model {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](initial_state)
