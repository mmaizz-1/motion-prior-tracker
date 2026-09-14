import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_prior_tracker import (
    MotionPriorTracker,
    TargetSpec,
    TrackerConfig,
    TrackObservation,
)


def _frame(x, y, visible=True):
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    if visible:
        image[y : y + 12, x : x + 16] = (30, 180, 240)
    return image


def test_tracks_non_bird_target_with_normalized_bbox():
    spec = TargetSpec(class_id=7, bbox_xywh=(0.18, 0.25, 0.16, 0.15))
    tracker = MotionPriorTracker(spec, TrackerConfig(search_radius=30))

    observations = tracker.track_sequence([_frame(10, 14), _frame(14, 16), _frame(18, 18)])

    assert observations[0].mode == "ref"
    assert all(observation.class_id == 7 for observation in observations)
    assert all(0.0 <= value <= 1.0 for observation in observations for value in observation.bbox_xywh)
    assert observations[-1].bbox_xywh[0] > observations[0].bbox_xywh[0]
    assert observations[-1].bbox_xywh[1] > observations[0].bbox_xywh[1]


def test_lost_frames_and_recovery_modes_are_explicit():
    spec = TargetSpec(class_id=3, bbox_xywh=(0.18, 0.25, 0.16, 0.15))
    config = TrackerConfig(search_radius=20, reacq_interval=1)
    tracker = MotionPriorTracker(spec, config)

    observations = tracker.track_sequence(
        [_frame(10, 14), _frame(14, 16), _frame(18, 18, visible=False), _frame(22, 20)]
    )

    assert observations[2].mode == "predicted"
    assert observations[2].lost_frames == 1
    assert observations[3].mode == "recovered"
    assert observations[3].lost_frames == 0


def test_observation_to_dict_is_json_serializable():
    observation = TrackObservation(
        frame_index=4,
        bbox_xywh=(0.2, 0.3, 0.1, 0.2),
        center_px=(20.0, 30.0),
        velocity_px=(2.0, -1.0),
        score=0.8,
        mode="matched",
        lost_frames=0,
        class_id=9,
        image_size=(100, 80),
    )

    payload = observation.to_dict()
    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert payload["class_id"] == 9
    assert payload["velocity_px"] == [2.0, -1.0]

