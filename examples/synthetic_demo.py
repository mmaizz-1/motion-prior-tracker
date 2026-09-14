"""Reproducible class-independent demo with known boxes and four hidden frames.

Run: python examples/synthetic_demo.py --output demo_out
The generated data is synthetic; its metrics do not establish real-world accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_prior_tracker import MotionPriorTracker, TargetSpec, TrackerConfig
from examples.embodied_attention import attention_hint


def make_sequence(kind):
    width, height, size = 400, 240, 24
    rng = np.random.default_rng(2026)
    texture = rng.integers(70, 245, (size, size, 3), dtype=np.uint8)
    if kind == 'ball_accelerating':
        yy, xx = np.mgrid[:size, :size]
        texture[(xx-size/2)**2 + (yy-size/2)**2 > (size/2-1)**2] = 0
    frames, boxes, visible = [], [], []
    for index in range(28):
        if kind == 'ball_accelerating':
            x, y = round(24 + 2*index + .25*index**2), round(45 + .125*index**2)
        else:
            x, y = 24 + 4*index, 45 + 2*index
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        seen = not 12 <= index <= 15
        if seen:
            frame[y:y+size, x:x+size] = texture
        frames.append(frame)
        boxes.append(((x+size/2)/width, (y+size/2)/height, size/width, size/height))
        visible.append(seen)
    return frames, boxes, visible


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lo = np.maximum(a[:2]-a[2:]/2, b[:2]-b[2:]/2)
    hi = np.minimum(a[:2]+a[2:]/2, b[:2]+b[2:]/2)
    intersection = float(np.prod(np.maximum(0, hi-lo)))
    return float(np.clip(intersection / (float(np.prod(a[2:]) + np.prod(b[2:])) - intersection), 0, 1))


def run_demo(output):
    output = Path(output)
    summaries = []
    for kind, class_id in [('box_constant', 7), ('ball_accelerating', 32)]:
        frames, truth, visible = make_sequence(kind)
        source = output / 'sequences' / kind
        source.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(frames):
            cv2.imencode('.png', image)[1].tofile(str(source / f'{index:04d}.png'))
        (source / '0000.txt').write_text(f"{class_id} " + ' '.join(map(str, truth[0])) + '\n')
        (source / 'ground_truth.json').write_text(json.dumps({'boxes': truth, 'visible': visible}))
        for model in ('ConstantVelocity', 'KalmanFilter', 'ConstantAcceleration'):
            tracker = MotionPriorTracker(
                TargetSpec(class_id=class_id, bbox_xywh=truth[0], context_margin=.5,
                           scale_range=(1., 1.), match_thresh=.7, target_thresh=.7),
                TrackerConfig(motion_model_name=model, search_radius=30, fine_margin=12,
                              reacq_interval=1, n_scales=1),
            )
            observations = tracker.track_sequence(frames)
            destination = output / 'results' / kind / model
            destination.mkdir(parents=True, exist_ok=True)
            tracker.write_jsonl(observations, destination / 'track.jsonl')
            errors = [float(np.linalg.norm((np.array(o.bbox_xywh[:2])-g[:2])*[400, 240]))
                      for o, g in zip(observations, np.asarray(truth))]
            valid_indices = [i for i, seen in enumerate(visible) if seen and i > 0]
            summary = {
                'sequence': kind, 'model': model, 'frames': len(frames),
                'visible_mean_iou': float(np.mean([box_iou(observations[i].bbox_xywh, truth[i]) for i in valid_indices])),
                'visible_mean_center_error_px': float(np.mean([errors[i] for i in valid_indices])),
                'hidden_mean_center_error_px': float(np.mean([errors[i] for i in range(12, 16)])),
                'predicted_frames': [o.frame_index for o in observations if o.mode == 'predicted'],
                'recovered_frames': [o.frame_index for o in observations if o.mode == 'recovered'],
            }
            summaries.append(summary)
            for index, (frame, observation) in enumerate(zip(frames, observations)):
                preview = frame.copy()
                cx, cy, bw, bh = observation.bbox_xywh
                color = (40, 210, 90) if observation.measurement_valid else (0, 185, 255)
                cv2.rectangle(preview, (round((cx-bw/2)*400), round((cy-bh/2)*240)),
                              (round((cx+bw/2)*400), round((cy+bh/2)*240)), color, 2)
                hint = attention_hint(observation)
                cv2.putText(preview, f'{model}  frame {index}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (220, 220, 220), 1)
                cv2.putText(preview, f'{observation.mode} | policy: {hint["mode"]}', (10, 225), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                cv2.imencode('.png', preview)[1].tofile(str(destination / f'{index:04d}.png'))
    (output / 'metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding='utf-8')
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='demo_out')
    arguments = parser.parse_args()
    print(json.dumps(run_demo(arguments.output), indent=2))
