"""
motion_models.py — pluggable motion priors for the Motion-Prior Tracker.

A motion prior is a stateful object with two methods:

    predict()            -> next state, from the current state + dynamics
    update(measurement)  -> fold a new measurement into the state

The tracking loop only ever calls these two methods, so the physics can be
swapped without touching the loop. This is the whole point of the framework:
"the loop stays fixed, the physics model is pluggable."
"""

import numpy as np


class MotionModel:
    """Base class for pluggable motion priors."""

    def predict(self):
        raise NotImplementedError

    def update(self, measurement):
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


_REGISTRY = {
    "ConstantVelocity": ConstantVelocity,
    # Roadmap (coming next):
    #   "ConstantAcceleration" — x'' = const
    #   "KalmanFilter"         — state-space + covariance (mechanics home turf)
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
