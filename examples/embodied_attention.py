"""Image-plane attention hints for a downstream policy, not actuator commands.

`image_error` uses right-positive, down-positive coordinates in [-1, 1].
It is neither a joint angle nor a world-space target position. A real system
needs calibrated optics, ego-motion, timing and an independently tested policy.
"""
from tracking_types import integer, unit


def attention_hint(observation, max_prediction_frames=3, min_score=0.5):
    integer(max_prediction_frames, 'max_prediction_frames')
    unit(min_score, 'min_score')
    if not observation.measurement_valid:
        mode = 'hold' if observation.lost_frames <= max_prediction_frames else 'search'
        return {'mode': mode, 'image_error': None}
    if observation.score < min_score:
        return {'mode': 'hold', 'image_error': None}
    cx, cy, _, _ = observation.bbox_xywh
    return {'mode': 'observe', 'image_error': [2*cx - 1, 2*cy - 1]}
