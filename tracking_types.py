"""Public configuration and camera-image observations (all time units: frames)."""
from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Sequence, Tuple

import numpy as np


def positive(value, name, allow_zero=False):
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be finite and {"non-negative" if allow_zero else "positive"}')


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def unit(value, name):
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be between 0 and 1')


def validate_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        raise ValueError('bbox_xywh must contain normalized cx, cy, width, height')
    values = tuple(float(v) for v in bbox)
    cx, cy, w, h = values
    if not np.all(np.isfinite(values)) or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError('bbox_xywh must be finite, normalized, and have positive dimensions')
    return values


def clamp_norm(cx, cy, bw, bh):
    """Keep the entire displayed box inside the image, preserving its size."""
    values = np.asarray([cx, cy, bw, bh], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('box must be finite')
    w, h = np.clip(values[2:], 1e-8, 1.)
    return [float(np.clip(cx, w/2, 1-w/2)), float(np.clip(cy, h/2, 1-h/2)), float(w), float(h)]


@dataclass(frozen=True)
class TargetSpec:
    class_id: int = 0
    bbox_xywh: Optional[Sequence[float]] = None
    template_padding: int = 4
    context_margin: float = 2.
    search_radius: int = 200
    fine_margin: int = 80
    scale_range: Tuple[float, float] = (0.5, 1.5)
    match_thresh: float = 0.35
    target_thresh: float = 0.65

    def __post_init__(self):
        integer(self.class_id, 'class_id')
        for name in ('template_padding', 'search_radius', 'fine_margin'):
            integer(getattr(self, name), name)
        positive(self.context_margin, 'context_margin', allow_zero=True)
        if len(self.scale_range) != 2:
            raise ValueError('scale_range must have two values')
        lo, hi = self.scale_range
        positive(lo, 'scale_range lower bound')
        positive(hi, 'scale_range upper bound')
        if lo > hi:
            raise ValueError('scale_range must be ordered')
        object.__setattr__(self, 'scale_range', (float(lo), float(hi)))
        unit(self.match_thresh, 'match_thresh')
        unit(self.target_thresh, 'target_thresh')
        if self.bbox_xywh is not None:
            object.__setattr__(self, 'bbox_xywh', validate_bbox(self.bbox_xywh))


@dataclass(frozen=True)
class TrackerConfig:
    motion_model_name: str = 'ConstantVelocity'
    template_ema: float = 0.3
    template_update_thresh: float = 0.8
    scale_ema: float = 0.35
    n_scales: int = 50
    reacq_interval: int = 10
    reacq_scale: float = 0.25
    context_scales: Tuple[float, ...] = (0.88, 0.94, 1., 1.06, 1.12)
    telemetry_path: Optional[str] = None
    search_radius: Optional[int] = None
    fine_margin: Optional[int] = None

    def __post_init__(self):
        unit(self.template_ema, 'template_ema')
        unit(self.template_update_thresh, 'template_update_thresh')
        unit(self.scale_ema, 'scale_ema')
        integer(self.n_scales, 'n_scales', 1)
        integer(self.reacq_interval, 'reacq_interval', 1)
        unit(self.reacq_scale, 'reacq_scale')
        positive(self.reacq_scale, 'reacq_scale')
        if not self.context_scales:
            raise ValueError('context_scales cannot be empty')
        for scale in self.context_scales:
            positive(scale, 'context scale')
        object.__setattr__(self, 'context_scales', tuple(self.context_scales))
        for name in ('search_radius', 'fine_margin'):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)


@dataclass(frozen=True)
class TrackObservation:
    """A 2-D image estimate. `score` is appearance similarity, not probability.

    Boxes use normalized (cx, cy, w, h); image_size is (width, height).
    Pixel center is the center of the clipped display box. During prediction
    the internal motion state may lie outside the image. Never treat a
    predicted box as a new detector measurement.
    """
    frame_index: int
    bbox_xywh: Tuple[float, float, float, float]
    center_px: Tuple[float, float]
    velocity_px: Tuple[float, float]
    score: float
    mode: str
    lost_frames: int
    class_id: int
    image_size: Tuple[int, int]

    @property
    def measurement_valid(self):
        return self.mode in ('ref', 'matched', 'recovered')

    def to_dict(self):
        return {
            'frame_index': int(self.frame_index),
            'bbox_xywh': [float(v) for v in self.bbox_xywh],
            'center_px': [float(v) for v in self.center_px],
            'velocity_px': [float(v) for v in self.velocity_px],
            'score': float(self.score), 'mode': self.mode,
            'lost_frames': int(self.lost_frames), 'class_id': int(self.class_id),
            'image_size': [int(v) for v in self.image_size],
            'measurement_valid': self.measurement_valid,
        }
